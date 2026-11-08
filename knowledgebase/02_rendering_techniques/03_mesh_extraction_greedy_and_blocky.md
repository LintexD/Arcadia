# 03. Mesh Extraction: Greedy & Blocky

## 1. Overview & Context

For engines that rely on traditional rasterization hardware (Direct3D 11/12, Vulkan graphics pipelines, or mobile GPUs) rather than pure compute ray marching, voxel data must be converted into triangle meshes.

A naive mesh generator emits a 2-triangle quad for every exposed voxel face. For a $256^3$ chunk, this can produce **millions of redundant triangles**, crippling GPU vertex processors and fill rates.

**Greedy Meshing** merges adjacent coplanar voxel faces sharing identical materials into large merged quads, reducing triangle counts by **$85\%\text{--}95\%$**.

---

## 2. The Classical Greedy Meshing Algorithm (Mikola Lysenko)

### 2.1 The 3-Axis Slice Sweep
The algorithm evaluates the volume along each principal axis ($X, Y, Z$) independently:
1. For axis $d \in \{0, 1, 2\}$, sweep a 2D slice through the chunk from $0$ to $N$.
2. At each slice, compute a 2D mask of visible faces:
   $$\text{FaceVisible}(u, v) = \begin{cases} \text{voxel}(u, v) & \text{if } \text{voxel}(u, v) \ne \text{Air} \text{ and } \text{neighbor}(u, v) == \text{Air} \\ 0 & \text{otherwise} \end{cases}$$

### 2.2 2D Quad Merging (Greedy Expansion)
Once the 2D face mask for a slice is generated:
1. Find the first non-zero face at $(u, v)$ with material $M$.
2. Extend along the $u$-axis as far as possible while faces match $M$. Let width be $W$.
3. Extend that row along the $v$-axis as far as possible while all $W$ faces match $M$. Let height be $H$.
4. Emit a single quad spanning $W \times H$.
5. Zero out the merged cells in the 2D mask.
6. Repeat until the entire 2D slice mask is empty.

```text
Unoptimized Voxel Faces (16 Quads)     Greedy Merged Face (1 Quad)
┌───┬───┬───┬───┐                     ┌───────────────┐
│ M │ M │ M │ M │                     │               │
├───┼───┼───┼───┤                     │               │
│ M │ M │ M │ M │          ════►      │   W=4, H=4    │
├───┼───┼───┼───┤                     │  Material M   │
│ M │ M │ M │ M │                     │   (2 tris)    │
├───┼───┼───┼───┤                     │               │
│ M │ M │ M │ M │                     └───────────────┘
└───┴───┴───┴───┘
```

---

## 3. Binary Greedy Meshing (Bitmask Slice Parallelism)

Classical greedy meshing uses 2D arrays and nested CPU loops, which can take $10\text{--}30\text{ ms}$ per chunk.

**Binary Greedy Meshing** (pioneered by Robert O'Leary and optimized by cgerrikk):
1. Represents voxel occupancy as 64-bit integers (`u64`).
2. Computes face visibility across an entire $64 \times 64$ slice using parallel bitwise operations:
   $$\text{ColFaceMask} = \text{Slice}_k \ \& \ \sim \text{Slice}_{k+1}$$
3. Uses hardware bit-scan intrinsics (`trailing_zeros` / `ctz`) to find quad boundaries in single clock cycles.
4. **Performance:** Generates meshes for $32^3$ chunks in **$< 0.2\text{ ms}$**, making real-time CPU mesh regeneration viable during runtime destruction!

---

## 4. The Ambient Occlusion & Texturing Challenge

While greedy meshing drastically reduces triangle counts, it introduces two major graphics issues:

### 4.1 Ambient Occlusion Interpolation Artifacts
In blocky games, per-vertex Ambient Occlusion (AO) is calculated based on the 8 neighbor blocks surrounding each vertex.
- When faces are merged into large $W \times H$ quads, intermediate vertex AO values are lost!
- **Solution:** Restrict greedy merging so faces only merge if all merged voxels share identical AO boundary gradients, or evaluate AO in screen-space (SSAO/GTAO) rather than vertex attributes.

### 4.2 Texture Tiling & Texture Atlases
A standard quad stretched across $10 \times 4$ voxels will stretch texture coordinates $(u, v) \in [0, 1]$ across the entire surface.
- **Solution:** Pass un-normalized quad dimensions $(W, H)$ in vertex attributes, and tile texture coordinates in the fragment shader using `fract(uv * vec2(W, H))`.

---

## 5. Performance Comparison: Greedy Meshing vs Compute Ray Marching

| Metric | Binary Greedy Meshing + Raster | Pure Compute Ray Marching (Arcadia) |
|---|---|---|
| **CPU Frame Budget** | $1.0\text{--}4.0\text{ ms}$ (Mesh building) | **$< 0.05\text{ ms}$ (Zero meshing)** |
| **GPU Memory Required** | Vertex & Index Buffers ($100\text{--}400\text{ MB}$) | **$25\text{--}60\text{ MB}$ (CS64 Trees)** |
| **Micro-Voxel Scalability ($6.25\text{ cm}$)** | ❌ Chokes on billions of polygons | **⭐⭐⭐⭐⭐ Seamless (Ray cone LOD)** |
| **Destruction Latency** | Must regenerate & re-upload mesh | **Instant (Mask patch in VRAM)** |
| **Hardware Compatibility** | All raster GPUs (Mobile, WebGL) | Requires Compute Shader Support |
