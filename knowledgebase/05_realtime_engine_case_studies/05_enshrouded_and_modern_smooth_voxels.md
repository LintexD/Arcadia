# 05. Enshrouded & Modern Smooth Voxel Games

## 1. Overview & Market Success

Developed by Keen Games (released in 2024), *Enshrouded* demonstrated that a modern AAA open-world survival action-RPG could be built entirely on a **destructible, editable 3D voxel density engine**:
- Players can dig tunnels through mountains, build medieval castles with custom stone arches, hollow out underground bunkers, and sculpt smooth organic terrain.
- The world looks completely smooth, detailed, and organic, with zero visible cubic blockiness on the landscape.

---

## 2. Density Fields & Dual Contouring

### 2.1 Continuous Signed Distance / Density Fields
Instead of binary solid/air flags, *Enshrouded's* world stores continuous scalar density fields $\rho(\mathbf{x}) \in [-128, +127]$:
- Negative density represents solid bedrock or soil.
- Positive density represents open air.
- Zero represents the exact surface interface.

### 2.2 Smooth Mesh Extraction via Transvoxel Dual Contouring
To render the world on standard rasterization hardware:
- The engine runs an optimized variant of **Dual Contouring with Transvoxel LOD Transition Cells** (Eric Lengyel).
- Solves the Quadratic Error Function (QEF) to preserve sharp architectural corners on player-built houses, while smoothly interpolating curved hillsides.
- Transvoxel transition cells seal seams between near high-detail chunks ($0.25\text{ m}$) and distant low-detail chunks ($1.0\text{ m}\text{--}4.0\text{ m}$) without T-junction cracks.

---

## 3. Dynamic Tri-Planar Material Texturing

A major challenge with polygonized voxel terrain is texture coordinate generation:
- Standard UV unwrapping is impossible because the terrain can be carved arbitrarily at runtime.
- **Tri-Planar Projection:** Textures are projected along the 3 principal world axes ($X, Y, Z$) and blended based on the surface normal:
  $$\mathbf{w} = \frac{|\mathbf{n}|^p}{\sum |\mathbf{n}_i|^p} \quad (p \approx 4\text{--}8)$$
  $$\text{Albedo} = w_x \, T(y, z) + w_y \, T(x, z) + w_z \, T(x, y)$$
- Grass automatically coats upward-facing surfaces ($n_y > 0.7$), while sheer cliff faces display layered vertical rock strata.

---

## 4. Trade-Offs: Smooth Density Meshing vs Micro-Voxel Ray Marching

| Metric | Smooth Voxel Density Meshing (*Enshrouded*) | Micro-Voxel Compute Ray Marching (*Arcadia*) |
|---|---|---|
| **Visual Aesthetic** | Smooth, continuous, organic landscape | Stylized, ultra-detailed micro-voxel bricks |
| **Destruction Latency** | High ($5\text{--}25\text{ ms}$ CPU/GPU mesh regen) | **Instant ($< 0.05\text{ ms}$ bitmask patch)** |
| **Fine Structural Precision** | Difficult for millimeter-scale foliage | **Exceptional ($6.25\text{ cm}$ micro-voxels)** |
| **Secondary Illumination** | Requires separate BVH / probe grid | **Unified CS64 tree for GI & Shadows** |
| **Hardware Compatibility** | Runs on all traditional raster GPUs | Requires modern compute shader support |

> **Conclusion:** *Enshrouded* demonstrates the zenith of smooth polygonized voxel engines. However, for ultra-high-frequency real-time destruction, instantaneous carving, and unified ReSTIR ray tracing, Arcadia's **micro-voxel compute ray marching** provides superior real-time responsiveness and zero polygon overhead.
