# 01. Teardown Architecture Deep Dive

## 1. Overview & Cultural Impact

Developed by Dennis Gustafsson and Tuxedo Labs (released in 2020), *Teardown* proved to the video game industry that an entirely destructible, physically simulated micro-voxel world could run smoothly at 60 FPS on mid-range hardware without pre-baked physics geometry.

---

## 2. World Representation & Dual-Depth Ray Marching

### 2.1 The Voxel Scale & Chunk Octree
- **Voxel Scale:** $10\text{ cm} \times 10\text{ cm} \times 10\text{ cm}$ ($1,000\text{ voxels per m}^3$).
- The static world is represented by a Sparse Voxel Octree (SVO) divided into spatial chunks.
- Each voxel stores an 8-bit material index (referencing physical parameters: density, hardness, flammability, reflectivity).

### 2.2 Dual-Depth Compute Ray Marching
*Teardown* does not use hardware DXR or triangle meshes for its environment. Rendering is performed via **custom GPU compute ray marching**:
1. **Low-Resolution Coarse Pre-Pass:** Rays march through a coarse mipmapped hierarchy at 1/2 or 1/4 resolution to quickly skip empty air and establish a depth bounding envelope.
2. **High-Resolution Fine DDA:** Full-resolution rays start marching from the coarse depth envelope, testing fine leaf voxels.
3. **Primary + Reflection Buffers:** Reflections and refractive glass are ray-traced directly against the identical voxel octree.

---

## 3. Structural Integrity & The Flood-Fill Engine

*Teardown's* defining technical triumph is its **instantaneous structural collapse engine**:

### 3.1 The Destructive Event Loop
When a sledgehammer, shotgun, bomb, or driving vehicle modifies voxels:
1. **Dirty AABB Identification:** A bounding box is constructed around the carved volume.
2. **Boundary Anchor Test:** Voxels along the bottom plane of the world ($y \le Y_{\text{ground}}$) or connected to uncarved static bedrock are flagged as **Anchors**.
3. **3D Multi-Source Breadth-First Search (BFS):**
   - Seed the search queue with all anchor boundary voxels.
   - Visited voxels are tagged `STABLE`.
   - Any occupied voxel cluster that cannot reach an anchor is identified as an **Unsupported Island**.
4. **Rigid Body Instantiation:**
   - The island voxels are erased from the static world octree.
   - A dynamic `RigidBody` is instantiated.
   - Its Center of Mass, Mass, and $3 \times 3$ Moment of Inertia Tensor are computed directly from the voxel cluster.
   - An Oriented Bounding Box (OBB) is fitted via Principal Component Analysis (PCA).

---

## 4. Physics Solver: Discrete Voxel Contact Manifolds

*Teardown* does not convert rigid bodies into convex hulls or polygon meshes:
- Dynamic rigid bodies remain **collections of micro-voxels**.
- When an OBB collides with the world or another body, surface boundary voxels are transformed into world space.
- Penetration depth $\delta$ is sampled directly against the static world voxel grid.
- Contact impulses are resolved using a custom Projected Gauss-Seidel sequential impulse solver.
- Because contact points are discrete voxel corners, falling walls realistically snag on window sills, crumble, and tumble over debris heaps.

---

## 5. Architectural Lessons for Arcadia

1. **Avoid Meshing for Destruction:** Generating polygon meshes during continuous destruction creates CPU bottlenecks and GPU buffer stalls. Pure compute ray marching makes destruction instantaneous.
2. **Two-Tiered Physics:** Reserve full rigid bodies for macro structural collapse; handle small blast fragments ($< 64$ voxels) via lightweight GPU compute particle buffers.
3. **Voxel Scale Sweet Spot:** *Teardown* used $10\text{ cm}$ voxels. Arcadia's $6.25\text{ cm}$ scale provides almost $4\times$ the visual fidelity while maintaining 60+ FPS via 64-tree bitmask acceleration.
