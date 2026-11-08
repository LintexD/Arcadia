# 03. Clipmaps, Brickmaps & Virtual Grids

## 1. Overview & Problem Definition

In massive real-time open worlds, loading the entire world geometry into GPU memory at highest detail is impossible. Even with sparse structures, a player moving through infinite terrain requires a dynamic mechanism that:
1. Provides millimeter-to-centimeter resolution near the camera.
2. Gracefully falls back to coarser resolutions at distance.
3. Streams in newly visible regions and evicts stale regions with zero frame drops or memory reallocation spikes.

Three dominant architectures have emerged in real-time graphics:
- **Geometry Clipmaps / 3D Voxel Clipmaps** (Toroidal Rolling Buffers).
- **Brickmaps & Hierarchical VDB / NanoVDB Grids**.
- **Virtual Voxel Page Tables** (GigaVoxels-style).

---

## 2. 3D Toroidal Voxel Clipmaps

### 2.1 The Concentric Nested Hierarchy
A 3D Voxel Clipmap maintains a series of $L$ concentric nested grids centered on the camera:
- **Level 0 (Finest):** Covers radius $R_0$ (e.g. $32\text{ m}$) at full voxel resolution ($6.25\text{ cm}$).
- **Level 1:** Covers radius $R_1 = 2 \cdot R_0$ ($64\text{ m}$) at half resolution ($12.5\text{ cm}$).
- **Level $k$:** Covers radius $R_k = 2^k \cdot R_0$ at resolution $2^k \cdot \Delta x$.

```text
       ┌───────────────────────────────┐
       │ Level 2 (Low Res, 256m)       │
       │    ┌─────────────────────┐    │
       │    │ Level 1 (Med, 128m) │    │
       │    │    ┌───────────┐    │    │
       │    │    │ L0 (Fine) │    │    │
       │    │    │     🎥    │    │    │
       │    │    └───────────┘    │    │
       │    └─────────────────────┘    │
       └───────────────────────────────┘
```

### 2.2 Toroidal Address Wrapping (Rolling Buffer)
Rather than shifting all voxel data in memory when the camera crosses a chunk or cell boundary, clipmaps use **toroidal modulo arithmetic**:
$$\text{buffer\_coord} = \text{world\_coord} \pmod{\text{GRID\_SIZE}}$$

When the player moves East by $\Delta x$:
1. Only the newly exposed "slab" on the eastern boundary is uploaded or generated.
2. The stale western slab is overwritten in place.
3. **CPU-to-GPU bandwidth:** Only $O(N^2)$ data is transferred per step rather than $O(N^3)$, reducing streaming bandwidth to tens of kilobytes per frame.

---

## 3. Brickmaps & NanoVDB

### 3.1 What is a Brickmap?
A **Brickmap** decouples high-level spatial indexing from fine voxel data:
- An indexing hierarchy (sparse grid, octree, or hash table) points to fixed-size dense blocks called **Bricks** (commonly $4 \times 4 \times 4 = 64$ voxels or $8 \times 8 \times 8 = 512$ voxels).
- **Why Bricks?**
  1. Dense memory inside a brick allows standard 3D texture filtering and linear addressing.
  2. Empty space is skipped at the brick level.
  3. Single-brick editing does not invalidate high-level tree topology.

### 3.2 OpenVDB and NanoVDB Architecture
Developed by Ken Museth (DreamWorks / NVIDIA), OpenVDB and its GPU counterpart **NanoVDB** use a tree with very high, tailored branching factors:
- Typical configuration: **Root $\to$ Node5 ($32^3$) $\to$ Node4 ($16^3$) $\to$ Internal3 ($8^3$) $\to$ Leaf ($8^3$)**.
- NanoVDB packs this structure into a single, linearized, read-only GPU buffer with absolute 64-bit aligned offsets.
- **Ray Traversal in NanoVDB:** HDDA (Hierarchical Digital Differential Analyzer) traverses the high-level tree until a leaf is struck, then marches through the leaf mask.

---

## 4. Virtual Voxel Page Tables (GigaVoxels)

Pioneered by Cyril Crassin et al. (GigaVoxels, 2009):
- Inspired by virtual memory and John Carmack's MegaTexture.
- The GPU allocates a fixed-size **Voxel Cache Pool** (e.g., $1\text{ GB}$ of 3D texture containing thousands of $8^3$ voxel bricks).
- A hierarchical **Page Table** maps virtual world coordinates to physical cache slots.

### 4.1 Demand-Driven GPU Ray Feedback
1. During ray marching, if a ray hits an unallocated page table entry, the shader writes the missing node ID into a **GPU Feedback Buffer**.
2. At the end of the frame, the CPU reads the feedback buffer, streams or generates the missing bricks from disk/procedural noise, and uploads them to the GPU cache pool.
3. Stale bricks (not requested for $N$ frames) are evicted using LRU (Least Recently Used) replacement.

---

## 5. Architectural Trade-Offs for Real-Time Games

| Metric | Toroidal Clipmap | NanoVDB / Brickmap | Virtual Page Table |
|---|---|---|---|
| **Best Used For** | Continuous open-world terrain, dynamic camera | Sparse smoke/clouds, static high-res assets | Massive streaming worlds (Giga-scale) |
| **GPU Traversal Speed** | ⭐⭐⭐⭐⭐ Extremely fast | ⭐⭐⭐⭐ Fast | ⭐⭐⭐ Moderate (Page indirection) |
| **Dynamic Carving** | ⭐⭐⭐⭐⭐ Trivially editable | ⭐⭐ Slow (Rebuild required) | ⭐⭐⭐ Good (Dirty brick re-upload) |
| **Memory Predictability**| 100% Fixed (No allocation spikes) | Variable based on scene complexity | 100% Fixed (Bounded cache pool) |
| **Implementation Complexity**| Moderate | High | Very High (Feedback loop required) |

> **Arcadia Takeaway:** Arcadia's toroidal chunk clipmap combined with $4^3$ Leaf Bricks inside a CS64 tree captures the best of both worlds: fixed bounded GPU allocation, toroidal streaming across infinite terrain, and nanosecond local brick carving.
