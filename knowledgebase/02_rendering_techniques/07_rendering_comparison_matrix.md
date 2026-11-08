# 07. Rendering Comparison & Decision Matrix

## 1. Quantitative Real-Time Benchmark Matrix

The following table compares primary real-time rendering paradigms evaluated on modern desktop GPUs (NVIDIA RTX 3080 / AMD RX 6800 XT) running a 1080p viewport at 60+ FPS:

| Rendering Technique | Primary G-Buffer Pass Time | Dynamic Destruction Cost | Secondary Rays (Shadows & GI) | Memory Footprint (Geometry) | Implementation Complexity |
|---|---|---|---|---|---|
| **Hierarchical DDA (CS64 Compute)** | **$0.8\text{--}1.4\text{ ms}$** | **$O(1)$ Instant (Mask patch)** | **Native (Same CS64 tree)** | **$25\text{--}65\text{ KB}$ / chunk** | Moderate |
| **Uniform Grid DDA** | $6.0\text{--}18.0\text{ ms}$ (Slow) | $O(1)$ Instant | Native | $16\text{ MB}$ / chunk | Low |
| **Binary Greedy Meshing + Raster** | $0.6\text{--}1.2\text{ ms}$ (Raster) | $2.0\text{--}8.0\text{ ms}$ CPU Mesh rebuild| ❌ Requires separate BVH / Voxel grid | $150\text{--}400\text{ MB}$ (VBOs) | Moderate |
| **Surface Nets / Dual Contouring** | $1.2\text{--}2.5\text{ ms}$ (Raster) | $5.0\text{--}25.0\text{ ms}$ QEF/Tri rebuild | ❌ Requires separate BVH | $200\text{--}600\text{ MB}$ (VBOs) | High |
| **Software Compute Raster / Splat** | $1.1\text{--}2.8\text{ ms}$ | $O(1)$ Instant | ❌ Poor (Cannot trace rays) | $40\text{--}90\text{ MB}$ | High |
| **Hardware DXR / Vulkan RT Cores** | $0.7\text{--}1.1\text{ ms}$ | $3.0\text{--}10.0\text{ ms}$ BLAS rebuild | Native | $200\text{--}500\text{ MB}$ (BVH) | Very High |

---

## 2. In-Depth Comparative Evaluation

### 2.1 The Dynamic Destruction Imperative
If the game features high-frequency destruction (explosions, drilling, projectile impacts, crumbling ruins):
- **Polygon Meshing (Greedy Meshing / Marching Cubes) fails to scale:** Every carved sphere triggers a CPU or compute job to regenerate triangle indices, recalculate normal smoothing, rebuild vertex buffers, and synchronize with the GPU. If 50 explosions occur simultaneously, the CPU mesh generator chokes.
- **Hardware DXR fails to scale:** Rebuilding hardware acceleration structures (BLAS) for hundreds of modified AABBs causes severe GPU command queue stalls.
- **Compute CS64 HDDA thrives:** An explosion directly clears bits in the `u64` occupancy mask. The next frame's compute ray march naturally steps through the carved void with **zero intermediate conversion steps**.

### 2.2 Secondary Illumination Coherence
A complete game engine requires more than primary camera visibility: it needs soft directional shadows, ambient occlusion, reflection bounces, and indirect diffuse global illumination.
- Meshed voxel engines must maintain two separate representations: the polygon mesh for rasterization, and a coarse 3D voxel texture or BVH for lighting. This doubles memory usage and introduces visual desynchronization.
- In **Compute CS64 HDDA**, the identical tree structure is traversed by primary rays, shadow rays, reflection rays, and ReSTIR diffuse GI rays.

---

## 3. Technology Selection Guide

```text
                                  Does your engine require
                              real-time physical destruction?
                                      /              \
                                   [YES]             [NO]
                                    /                  \
                        Does it require smooth,     Is memory bandwidth
                           organic surfaces?          the top priority?
                            /             \             /            \
                         [YES]           [NO]        [YES]          [NO]
                          /                 \          /              \
                   [Surface Nets /      [CS64 HDDA   [Clustered     [Hardware DXR
                    Dual Contouring]     Compute]     SVDAG]         Mesh Pipeline]
```

> **Final Recommendation for Arcadia:** The **Chunked Sparse 64-Tree (CS64) with Compute HDDA** is overwhelmingly the superior architectural choice for a high-performance, real-time destructive micro-voxel engine.
