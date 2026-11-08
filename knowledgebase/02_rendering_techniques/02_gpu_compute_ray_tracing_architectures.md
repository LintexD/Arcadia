# 02. GPU Compute Ray Tracing Architectures

## 1. Overview: Hardware RT Cores vs Compute Shaders

When implementing real-time voxel ray tracing on modern GPUs, developers face a core architectural choice:
1. **Hardware Ray Tracing Cores (Vulkan `VK_KHR_ray_tracing` / DirectX Raytracing DXR):** Requires wrapping voxel bounding boxes into an Acceleration Structure (BLAS / TLAS) and tracing rays via `traceRayEXT`.
2. **Pure GPU Compute Shaders (WGSL / HLSL / GLSL):** Directly marhces rays through custom voxel data structures (CS64, SVO) in compute kernels.

---

## 2. Why Pure Compute Shaders Dominate Dynamic Voxels

While hardware RT cores excel at static triangle meshes:
1. **BLAS Rebuild Overhead:** Every time a voxel is carved or placed by an explosion, the corresponding AABB bounding volume in the BLAS must be refitted or rebuilt. Rebuilding hundreds of BLAS structures per frame stalls the GPU and consumes $2\text{--}5\text{ ms}$ of frame time.
2. **AABB Intersection Inefficiency:** Hardware RT cores test rays against generic AABB bounding boxes using procedural hit shaders. Procedural hit shaders disable the hardware BVH traversal pipeline's most aggressive optimizations (dual-issue ray-box tests).
3. **Hardware Independence:** Compute shaders run uniformly on Vulkan 1.2+, Metal, DirectX 12, Apple Silicon, and AMD/Intel integrated GPUs with **zero hardware RT core requirements**.

---

## 3. Workgroup Dispatch Architectures

### 3.1 Screen-Tiled Workgroups ($8 \times 8$ or $16 \times 8$)
The standard approach maps a 2D grid of compute threads directly to display pixels:
$$\text{DispatchGroups} = \left( \lceil \text{Width} / 8 \rceil, \lceil \text{Height} / 8 \rceil, 1 \right)$$

```text
Screen (1920x1080)
┌───────┬───────┬───────┬───────┐
│ (8x8) │ (8x8) │ (8x8) │  ...  │  <-- Each 8x8 tile is 1 GPU Workgroup (64 threads = 2 Warps)
├───────┼───────┼───────┼───────┤
│ (8x8) │ (8x8) │ (8x8) │  ...  │
└───────┴───────┴───────┴───────┘
```

**Pros:**
- Direct pixel mapping.
- High spatial coherence: rays in the same $8 \times 8$ tile point in nearly identical directions, maximizing L1 texture cache hits.

**Cons:**
- **Warp Divergence:** If 63 threads hit an obstacle at $t=2\text{ m}$, but 1 thread shoots past a corner into the infinite sky, the entire 32-thread warp remains active until the last thread finishes!

### 3.2 Persistent Threads & Stream Compaction
In advanced compute ray tracers:
1. A fixed pool of worker threads is launched (matching the exact number of SMs / Compute Units on the GPU).
2. Threads fetch ray tasks from a global atomic ring buffer.
3. When a ray hits a surface or terminates, the thread immediately fetches the next ray without waiting for neighboring pixels.
4. **Ray Compaction:** Rays that require secondary bounces (e.g. diffuse GI or reflections) are compacted into a contiguous buffer using atomic prefix sums (`subgroupAdd` / wave intrinsics), ensuring 100% warp utilization.

---

## 4. Tile Depth Pre-Pass (Coarse-to-Fine Acceleration)

To virtually eliminate warp divergence in screen-tiled architectures:
- Dispatch a lightweight **1/4-resolution (or 1/8-resolution) Depth Pre-Pass** (e.g. $480 \times 270$).
- Compute the conservative minimum hit distance $t_{\text{min}}$ for each $8 \times 8$ pixel tile.
- When the full-resolution 1080p primary shader executes:
  - Read the tile's $t_{\text{min}}$.
  - Rays start marching directly from $t = t_{\text{min}} - \delta$ rather than from the camera lens ($t=0$).
  - **Result:** Up to $70\%$ of empty-space traversal steps are completely bypassed across all primary rays!

```text
[ Camera (t=0) ] ═══════ Empty Air (Bypassed) ═══════► [ t_min ] ──► [ Surface Hit ]
```

---

## 5. Optimal Wavefront Shading Pipeline

Rather than calculating lighting, shadows, and reflections inside the primary ray marching loop (which causes register pressure and spills to VRAM):
```text
Pass 1: Primary G-Buffer Compute (Primary Ray March -> Depth, Normal, Material, Motion)
        │
Pass 2: Direct Shadow Ray March (Compact 1 shadow ray per pixel, early exit on any hit)
        │
Pass 3: ReSTIR GI / Indirect Diffuse (Stochastic rays, reservoir spatio-temporal reuse)
        │
Pass 4: Spatiotemporal Wavelet Denoising & Reconstruction
        │
Pass 5: DLAA / TSR Temporal Super-Resolution Composite
```

This decoupled **Wavefront Architecture** keeps register usage below 32–48 registers per thread, allowing **100% GPU warp occupancy**.
