# 03. Dynamic Rigid Body Voxels

## 1. Overview & Physics Representation

When a structure collapses or debris is detached in a voxel world, it transforms from static world geometry into a **Dynamic Rigid Voxel Body**.

Unlike traditional game engines that approximate objects with simple boxes, spheres, or convex hulls, a voxel rigid body retains its **exact, discrete micro-voxel geometry**:
- A hollow stone urn or jagged broken pillar collides and snags along its actual voxel steps.
- The object can continue to be broken or chipped into smaller pieces upon high-velocity impact.

---

## 2. Center of Mass & Moment of Inertia Tensor

Because a voxel body is composed of discrete cubic elements of uniform density $\rho$ and dimension $s = 0.0625\text{ m}$:

### 2.1 Total Mass & Center of Mass
Let $N$ be the count of active voxels in the body, each of mass $m_0 = \rho \cdot s^3$:
$$M = N \cdot m_0$$
The Center of Mass $\mathbf{c} \in \mathbb{R}^3$ is the arithmetic average of voxel center coordinates $\mathbf{p}_i$:
$$\mathbf{c} = \frac{1}{N} \sum_{i=1}^N \mathbf{p}_i$$

### 2.2 Discrete Moment of Inertia Tensor
For a rigid body, the moment of inertia tensor $\mathbf{I} \in \mathbb{R}^{3 \times 3}$ about its center of mass is computed by summing over all voxels:
Let $\mathbf{r}_i = \mathbf{p}_i - \mathbf{c} = (x_i, y_i, z_i)$:

$$I_{xx} = \sum_{i=1}^N m_0 \left( y_i^2 + z_i^2 + \frac{s^2}{6} \right)$$
$$I_{yy} = \sum_{i=1}^N m_0 \left( x_i^2 + z_i^2 + \frac{s^2}{6} \right)$$
$$I_{zz} = \sum_{i=1}^N m_0 \left( x_i^2 + y_i^2 + \frac{s^2}{6} \right)$$
$$I_{xy} = I_{yx} = -\sum_{i=1}^N m_0 \, x_i y_i$$
$$I_{xz} = I_{zx} = -\sum_{i=1}^N m_0 \, x_i z_i$$
$$I_{yz} = I_{zy} = -\sum_{i=1}^N m_0 \, y_i z_i$$

*(The term $\frac{s^2}{6}$ accounts for each individual voxel's own cubic inertia via the parallel axis theorem).*

---

## 3. Collision Detection: Dynamic Voxel vs Static CS64 World

### 3.1 Broadphase: Oriented Bounding Box (OBB) vs World AABB
1. Compute the world-space Oriented Bounding Box (OBB) of the dynamic body from its position $\mathbf{x}$ and orientation quaternion $\mathbf{q}$.
2. Test the OBB against the static world chunk AABBs using the Separating Axis Theorem (SAT).
3. If no intersection occurs, collision is ruled out with zero voxel lookups.

### 3.2 Narrowphase: Bitwise Voxel Point Sampling
When an OBB intersects a static chunk:
1. Transform the dynamic body's boundary voxels into world coordinates:
   $$\mathbf{w}_i = \mathbf{x} + \mathbf{R}(\mathbf{q}) \cdot \mathbf{r}_i$$
2. Test $\mathbf{w}_i$ against the static world CS64 tree using nanosecond bitmask lookups (`test_voxel(x, y, z)`).
3. For any penetrating voxel, compute penetration depth $d$ and contact normal $\mathbf{n}$ from the gradient of the static voxel occupancy.

```text
Dynamic Voxel Body (Rotated OBB)
         ┌────────┐
         │  ■  ■  │
         │  ■  ■  │
         └───┬────┘
             │ (Penetrating Contact Point: w_i)
             ▼
    ═════════════════════ Static Voxel Floor (CS64)
```

---

## 4. Contact Resolution & Impulse Solver

When contact points $\{\mathbf{p}_k, \mathbf{n}_k, d_k\}$ are found:
- Apply a sequential impulse solver (Projected Gauss-Seidel or Baumgarte stabilization):
  $$J \mathbf{v} = (\mathbf{v}_1 - \mathbf{v}_2) \cdot \mathbf{n} + (\boldsymbol{\omega}_1 \times \mathbf{r}_1 - \boldsymbol{\omega}_2 \times \mathbf{r}_2) \cdot \mathbf{n}$$
  $$j = \frac{-(1 + e) v_{\text{rel}} + \beta d}{\frac{1}{M_1} + \frac{1}{M_2} + (\mathbf{I}_1^{-1}(\mathbf{r}_1 \times \mathbf{n}) \times \mathbf{r}_1 + \dots) \cdot \mathbf{n}}$$
- Apply friction impulses along tangential contact planes to allow tumbling, sliding, and stacking.

---

## 5. GPU Compute Rendering of Dynamic Rigid Bodies

To render dynamic rotating voxel bodies without polygon meshing:
1. Pass an array of `RigidBodyInstance` uniforms to the primary G-Buffer compute shader:
   ```wgsl
   struct RigidBodyInstance {
       transform_world_to_local: mat4x4<f32>,
       transform_local_to_world: mat4x4<f32>,
       half_extents: vec3<f32>,
       node_pool_offset: u32,
   };
   ```
2. Primary camera rays test against the body's OBB using an analytical ray-box intersection.
3. If hit, transform the ray into the body's **local coordinate space**:
   $$\mathbf{o}_{\text{local}} = \mathbf{T}^{-1} \mathbf{o}_{\text{world}}, \quad \mathbf{d}_{\text{local}} = \mathbf{T}^{-1} \mathbf{d}_{\text{world}}$$
4. Traverse the body's local CS64 tree using standard HDDA!
5. Transform the resulting hit normal back to world space:
   $$\mathbf{n}_{\text{world}} = \mathbf{T} \cdot \mathbf{n}_{\text{local}}$$

This enables seamless rendering of dynamic, spinning, falling voxel structures with **zero triangle overhead**.
