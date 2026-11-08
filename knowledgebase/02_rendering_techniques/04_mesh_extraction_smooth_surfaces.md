# 04. Mesh Extraction: Smooth Surfaces

## 1. Overview & Isosurface Extraction

While blocky voxels (Minecraft, Teardown) render sharp cubic step-boundaries, many voxel applications require **smooth, organic terrain** (e.g. *Enshrouded*, *No Man's Sky*, *Astroneer*).

To produce smooth surfaces, voxels store a continuous scalar value (a **Signed Distance Field (SDF)** or continuous density value $\rho \in \mathbb{R}$):
- $\rho < 0$: Inside solid terrain.
- $\rho > 0$: Outside in empty air.
- $\rho = 0$: The exact boundary surface (**isosurface**).

Extracting a smooth triangle mesh from discrete 3D density grids is known as **Isosurface Polygonization**.

---

## 2. Marching Cubes vs Surface Nets vs Dual Contouring

### 2.1 Marching Cubes (Lorensen & Cline, 1987)
- Evaluates the 8 corner vertices of each grid cell.
- An 8-bit index ($2^8 = 256$ cases, reduced to 15 canonical configurations) looks up a pre-calculated triangulation table.
- Vertices are placed along cell edges via linear interpolation:
  $$\mathbf{p} = \mathbf{p}_1 + \frac{0 - \rho_1}{\rho_2 - \rho_1} (\mathbf{p}_2 - \mathbf{p}_1)$$
- **Limitations:**
  - Produces high triangle counts (up to 5 triangles per cell).
  - Incapable of preserving sharp mechanical edges or 90-degree corners (all sharp features are smoothed out or chamfered).

---

### 2.2 Surface Nets (Sarah Gibson, 1998)
A dual method that generates significantly cleaner topology than Marching Cubes:
1. For every grid cell that contains a sign change (crosses the isosurface), compute **exactly one vertex** inside the cell.
2. The vertex position is calculated as the average of all edge-crossing intersection points on that cell's 12 edges.
3. Connect adjacent cell vertices to form regular quad/triangle meshes.
4. **Advantages:**
   - Approximately **$3\times$ fewer triangles** than Marching Cubes.
   - Uniform, well-conditioned mesh topology without sliver triangles.
   - Very easy to implement on GPU compute shaders.

---

### 2.3 Dual Contouring (Tao Ju et al., 2002)
Dual Contouring represents the pinnacle of isosurface extraction for engineering and game engines because it **reproduces both smooth organic surfaces and razor-sharp geometric creases**:
- Requires **Hermite Data**: In addition to corner densities, store the exact intersection point $\mathbf{p}_i$ and surface normal $\mathbf{n}_i$ on edges that cross the isosurface.
- Within each active cell, place a vertex $\mathbf{v}$ that minimizes the **Quadratic Error Function (QEF)**:
  $$E(\mathbf{v}) = \sum_i ((\mathbf{v} - \mathbf{p}_i) \cdot \mathbf{n}_i)^2$$
- Solved via Singular Value Decomposition (SVD) or Particle Swarm / Gradient Descent.
- If planes intersect at a 90-degree angle, the QEF minimum resolves to the exact sharp corner!

```text
Marching Cubes: Chamfered Corner      Dual Contouring: Sharp Edge (QEF)
         /                                    ▲
        /                                    / \
       /                                    /   \
  ────┘                                ────┘     └────
```

---

## 3. The Multi-Resolution LOD Problem & Transvoxel

When adjacent chunks are rendered at different Levels of Detail (e.g., a near chunk at $0.25\text{ m}$ resolution adjacent to a far chunk at $0.5\text{ m}$ resolution), T-junctions and visible cracks appear along chunk seams.

### 3.1 The Transvoxel Algorithm (Eric Lengyel, 2010)
Used in commercial voxel engines (including *Enshrouded*):
- Inserts special **Transition Cells** along chunk boundary faces.
- Transition cells evaluate higher-frequency samples from the fine neighbor and generate transitional triangulation that hermetically seals all cracks.
- Guarantees seamless watertight meshes across arbitrary LOD steps ($2:1$ scale transitions).

---

## 4. Real-Time Performance & Dynamic Editing Overhead

Extracting smooth meshes in real time during dynamic destruction poses significant challenges:
1. **QEF Solver Cost:** Solving SVD matrices for thousands of cells on the CPU can take $15\text{--}50\text{ ms}$, causing visible stutter during explosions.
2. **GPU Surface Nets:** Running Surface Nets entirely in GPU compute shaders into a `DrawIndirect` buffer reduces extraction time to **$< 1.5\text{ ms}$**, making smooth dynamic carving viable at 60 FPS.
