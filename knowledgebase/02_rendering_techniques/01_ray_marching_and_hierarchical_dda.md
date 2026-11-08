# 01. Ray Marching & Hierarchical DDA

## 1. Overview & Theoretical Foundations

Ray marching through a discrete voxel grid differs fundamentally from traditional triangle ray tracing:
- Triangle ray tracing tests rays against arbitrary polygons organized in a BVH (Bounding Volume Hierarchy).
- Voxel ray marching traverses an axis-aligned rectilinear coordinate space.

The gold standard for rectilinear grid traversal is the **Amanatides & Woo Fast Voxel Traversal Algorithm (DDA - Digital Differential Analyzer)** (1987). It steps from voxel cell boundary to voxel cell boundary using only simple additions and comparisons, guaranteeing that **no intersected voxels are ever skipped**.

---

## 2. The 3D Amanatides & Woo DDA Algorithm

### 2.1 Mathematical Formulation
Given a ray with origin $\mathbf{o} = (o_x, o_y, o_z)$ and normalized direction $\mathbf{d} = (d_x, d_y, d_z)$:

1. **Step Direction ($\mathbf{step} \in \{-1, +1\}^3$):**
   $$\text{step}_i = \begin{cases} +1 & \text{if } d_i \ge 0 \\ -1 & \text{if } d_i < 0 \end{cases}$$

2. **Delta T ($\Delta \mathbf{t} \in \mathbb{R}^3$):** The distance along the ray required to travel 1 unit along axis $i$:
   $$\Delta t_i = \left| \frac{1}{d_i} \right|$$

3. **Next T Boundary ($\mathbf{t}_{\text{max}} \in \mathbb{R}^3$):** The distance along the ray to the first voxel boundary along axis $i$:
   $$t_{\text{max}, i} = \begin{cases} ((\lfloor o_i \rfloor + 1) - o_i) \cdot \Delta t_i & \text{if } d_i > 0 \\ (o_i - \lfloor o_i \rfloor) \cdot \Delta t_i & \text{if } d_i < 0 \end{cases}$$

### 2.2 The Traversal Loop
In each iteration, the algorithm steps along the axis whose next boundary is closest:

```rust
// Core DDA Stepping
if t_max.x < t_max.y {
    if t_max.x < t_max.z {
        current_voxel.x += step.x;
        t_current = t_max.x;
        t_max.x += delta_t.x;
        hit_normal = Vec3::new(-step.x as f32, 0.0, 0.0);
    } else {
        current_voxel.z += step.z;
        t_current = t_max.z;
        t_max.z += delta_t.z;
        hit_normal = Vec3::new(0.0, 0.0, -step.z as f32);
    }
} else {
    if t_max.y < t_max.z {
        current_voxel.y += step.y;
        t_current = t_max.y;
        t_max.y += delta_t.y;
        hit_normal = Vec3::new(0.0, -step.y as f32, 0.0);
    } else {
        current_voxel.z += step.z;
        t_current = t_max.z;
        t_max.z += delta_t.z;
        hit_normal = Vec3::new(0.0, 0.0, -step.z as f32);
    }
}
```

---

## 3. Hierarchical DDA (HDDA) on N-ary Trees

A uniform DDA stepping through individual $(1/16\text{ m})^3$ voxels would require thousands of iterations per ray through empty air. **Hierarchical DDA (HDDA)** solves this by traversing the tree hierarchically:
- Traverse the coarse root grid ($16\text{ m}$ chunks).
- When an occupied chunk is entered, push down into Level 1 ($4\text{ m}$ cells).
- When an occupied Level 1 node is entered, push down into Level 2 ($1\text{ m}$ cells).
- When an occupied Level 2 node is entered, push down into Level 3 ($0.25\text{ m}$ leaf bricks).
- Step through the $4 \times 4 \times 4$ leaf brick bitmask.
- If a cell is empty, advance the ray to the exit boundary of that coarse cell in **a single step**, skipping up to $64^3$ empty voxels!

```text
Ray ───────────────► [ Empty Chunk ] (1 Step: Skip 16m)
                    │
                    ▼ [ Occupied Chunk ]
                    └───► [ Empty Level 1 ] (1 Step: Skip 4m)
                          │
                          ▼ [ Occupied Level 2 ]
                            └───► [ Leaf Brick (4³ voxels) ]
                                  └───► Hit Voxel Surface!
```

---

## 4. GigaVoxels Ray-Cone Footprint LOD

### 4.1 The Nyquist Problem in Ray Marching
When rendering distant terrain, a single screen pixel covers multiple meters of world space. Marching rays down to individual $6.25\text{ cm}$ micro-voxels causes severe aliasing (pixel flickering/shimmering) and wastes massive memory bandwidth.

### 4.2 Cone Tracing Equation
Let $r_0$ be the sub-pixel ray radius at the camera lens ($t=0$), and let $\kappa = 2 \tan(\text{FOV} / 2) / \text{ViewportHeight}$ be the cone angular spread:
$$\text{Footprint}(t) = r_0 + \kappa \cdot t$$

```text
Camera (t=0)
  \                     /
   \  Footprint(t)     /
    \   ┌─────────┐   /
     \  │ Level 2 │  /    (Terminates ray when CellSize >= Footprint(t))
      \ └─────────┘ /
       \           /
```

### 4.3 Early Level-Of-Detail Termination
During HDDA traversal:
- If $\text{Footprint}(t) \ge \text{CellSize}_{\text{Level 1}}$ ($4.0\text{ m}$), terminate immediately at Level 1! Read the node's pre-filtered volumetric color.
- If $\text{Footprint}(t) \ge \text{CellSize}_{\text{Level 2}}$ ($1.0\text{ m}$), terminate immediately at Level 2!
- Leaf bricks ($4^3$ voxels) are **never fetched** for geometry beyond $\approx 50\text{ meters}$, cutting memory bandwidth by up to **$80\%$** on distant landscapes.

---

## 5. Critical Floating-Point Traversal Pitfalls & Solutions

### 5.1 Grazing-Angle Sky Streaks (The Epsilon Trap)
When rays travel almost parallel to a voxel slab (e.g. $d_y \approx 0.0001$ along a horizontal terrace), adding a fixed constant epsilon ($\epsilon = 0.001$) to advance the ray across the boundary can fail due to floating-point truncation:
$$(t + \epsilon) \cdot d_y \approx t \cdot d_y$$
The ray remains stuck in the same cell until `max_steps` expires, producing bright visual sky streaks.

### 5.2 The Exact Axis-Proportional Exit Slab Epsilon
To prevent boundary sticking, calculate the exit distance analytically:
$$\text{exit\_plane\_t} = \min(t_{\text{max}.x}, t_{\text{max}.y}, t_{\text{max}.z})$$
$$t_{\text{next}} = \text{exit\_plane\_t} + \max(1\times 10^{-5}, \text{exit\_plane\_t} \times 1\times 10^{-6})$$
By scaling epsilon proportionally with distance $t$, precision issues at long ranges ($t > 100\text{ m}$) are completely eliminated.

---

## 6. Performance Characteristics on Modern GPUs

| Metric | Uniform DDA | 2-Level Grid DDA | 4-Level CS64 HDDA |
|---|---|---|---|
| **Average Steps / Ray (1080p View)** | 850–2,400 | 80–220 | **14–28** |
| **Primary Ray March Time (RTX 3080)** | >16.0 ms | 3.2 ms | **0.85 ms** |
| **Divergent Warp Stall Ratio** | High | Medium | **Low (Coherent early exits)** |
