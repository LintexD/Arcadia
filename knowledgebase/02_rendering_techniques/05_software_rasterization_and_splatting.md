# 05. Software Compute Rasterization & Splatting

## 1. Overview & Modern Paradigm Shifts

For decades, real-time graphics was divided into two rigid camps:
1. Hardware polygon rasterization via fixed-function hardware units.
2. Software ray tracing.

In recent years, the computer graphics industry underwent a revolution—spearheaded by Epic Games' **Unreal Engine 5 (Nanite)** and modern compute research (Sebastian Aaltonen, Brian Karis):
- Fixed-function hardware rasterizers become inefficient when triangles approach sub-pixel sizes ($< 1\text{ pixel}$ across), suffering from quad-overdraw waste where an $80\%$ inactive $2 \times 2$ pixel quad is shaded for 1 tiny triangle.
- By using **GPU Compute Shaders to software-rasterize micro-geometry** directly into a 64-bit atomic integer depth buffer, GPUs achieve staggering throughput—rendering tens of millions of micro-primitives per frame.

This technique is directly applicable to micro-voxel rendering.

---

## 2. 64-Bit Atomic Integer Depth Buffering

### 2.1 How Compute Shaders Avoid Race Conditions
In a traditional graphics pipeline, depth testing and writing is handled by hardware ROPs (Raster Operations Pipelines).
In a compute shader rasterizer:
- Allocate a single 2D storage buffer: `atomic<u64>` of dimensions `Width x Height`.
- Pack 32-bit floating-point depth and a 32-bit payload (e.g. Voxel ID, Material ID, or Normal) into a single 64-bit unsigned integer:
  $$\text{packed\_val} = (\text{as\_u32}(\text{depth}) \ll 32) \mid (\text{payload} \ \& \ \text{0xFFFFFFFF})$$

### 2.2 Atomic Minimum (`atomicMin`)
To write a pixel:
```wgsl
let packed_depth_payload = (u64(bitcast<u32>(depth)) << 32u) | u64(voxel_material_id);
atomicMin(&depth_buffer[pixel_index], packed_depth_payload);
```
On modern GPUs (NVIDIA Turing/Ampere/Ada, AMD RDNA2/3), `atomicMin` on `u64` is executed directly in L2 cache memory controllers with hardware arbitration at hundreds of millions of operations per second!

---

## 3. Micro-Voxel Point & Surfel Splatting

Instead of treating micro-voxels as 12-triangle cubes or marching rays through them:
- Treat each surface micro-voxel as an oriented disc or square (**Surfel** - Surface Element).
- Project the surfel center $(x, y, z)$ through the camera projection matrix to screen space $(u, v)$.
- Calculate screen-space bounding radius $r = \frac{\text{VoxelRadius} \cdot f}{z}$.

```text
[ 3D Voxel (x,y,z) ] ───► Camera Projection ───► [ Screen Circle / Quad (u, v, r) ]
                                                          │
                                                    atomicMin (u64)
                                                          ▼
                                                 [ G-Buffer Surface ]
```

### 3.1 Sub-Pixel Micro-Splatting
- If $r \le 0.5$ pixels: Write a single atomic pixel directly at $(\lfloor u \rfloor, \lfloor v \rfloor)$.
- If $0.5 < r \le 2.0$ pixels: Shade a $2 \times 2$ or $3 \times 3$ pixel footprint using bitmask coverage.
- If $r > 2.0$ pixels: Switch to compute ray marching or quad rasterization.

---

## 4. Nanite-Style Workgroup Tile Binning

To prevent millions of compute threads from contending on the same atomic memory addresses:
1. **Cluster Binning:** Subdivide the screen into $64 \times 64$ pixel tiles.
2. Compute bounding boxes for voxel chunks or leaf bricks.
3. Bin active micro-voxels into screen tiles in shared LDS (Local Data Share / Workgroup Memory).
4. Each thread in the workgroup rasterizes into fast shared memory before flushing to global VRAM.

---

## 5. Performance Trade-Off: Splatting vs DDA Ray Marching

| Metric | Compute Splatting / Software Raster | Hierarchical DDA Ray Marching |
|---|---|---|
| **Complexity with Distance** | $O(N)$ (Scales with voxel count) | **$O(\log N)$ (Ray Cone LOD skips geometry)** |
| **Overdraw Handling** | Requires atomic depth buffer | **Early-ray termination (Zero overdraw)** |
| **Fine Geometric Details** | Outstanding for high-density foliage | Excellent with Ray-Cone LOD |
| **Memory Bandwidth** | High write traffic to depth buffer | Read-heavy texture/buffer lookups |
| **Secondary Rays (GI/Shadows)**| ❌ Cannot compute secondary bounces | **⭐⭐⭐⭐⭐ Unified (Same code for GI & Primary)**|

> **Conclusion for Arcadia:** Pure compute ray marching remains the most architecturally cohesive choice for unified primary rendering, soft shadows, and ReSTIR multi-bounce GI. Compute splatting is exceptionally valuable as a secondary pass for high-density dynamic particles and voxel debris.
