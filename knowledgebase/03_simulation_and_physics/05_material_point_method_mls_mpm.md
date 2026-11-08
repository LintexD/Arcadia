# 05. Material Point Method (MLS-MPM)

## 1. Overview: The Ultimate Multiphase Physics Solver

Granular materials (snow, avalanches, wet mud, dry sand, tearing dough, and soft clay) exhibit behaviors that defy both pure rigid bodies and pure grid fluids:
- At rest, snow behaves like a solid rigid body that supports weight.
- Under high stress or shear, snow yields, fractures, and flows like a viscous liquid.

The **Material Point Method (MPM)**—and its state-of-the-art real-time variant, **Moving Least Squares Material Point Method (MLS-MPM)** (Hu et al., SIGGRAPH 2018)—is the undisputed gold standard for simulating multiphase elastoplastic materials.

---

## 2. The Hybrid Particle-Grid (Eulerian-Lagrangian) Architecture

MLS-MPM combines:
1. **Lagrangian Particles:** Carry mass $m_p$, position $\mathbf{x}_p$, velocity $\mathbf{v}_p$, deformation gradient $\mathbf{F}_p$, and elastoplastic stress $\boldsymbol{\sigma}_p$.
2. **Eulerian Voxel Background Grid:** A regular Cartesian grid where momentum equations are solved without mesh entanglement or self-intersection issues.

```text
[ Lagrangian Particles ]
         │
         ▼ (1. P2G: Particle-to-Grid Scatter)
[ Eulerian Voxel Grid ]
         │
         ▼ (2. Grid Operations: Boundary Collisions, Gravity)
[ Solved Voxel Grid ]
         │
         ▼ (3. G2P: Grid-to-Particle Gather)
[ Updated Particles ]
```

---

## 3. The Real-Time MLS-MPM Pipeline

Standard MPM requires costly B-spline gradient evaluations and grid mass matrix inversions. **MLS-MPM** simplifies the affine momentum transfer, eliminating the need to track deformation gradient derivatives:

### Step 1: Particle-to-Grid (P2G) Transfer
For each particle $p$, scatter mass and momentum to the surrounding $3 \times 3 \times 3 = 27$ grid nodes $i$:
$$w_{ip} = N(\mathbf{x}_i - \mathbf{x}_p) \quad (\text{Quadratic B-Spline Kernel})$$
$$m_i = \sum_p w_{ip} m_p$$
$$(m\mathbf{v})_i = \sum_p w_{ip} \left( m_p \mathbf{v}_p + \mathbf{C}_p (\mathbf{x}_i - \mathbf{x}_p) \right)$$
Where $\mathbf{C}_p$ is the affine velocity gradient matrix (APIC formulation), which completely preserves angular momentum and prevents artificial numerical damping.

### Step 2: Grid Momentum Update & Voxel Boundary Collisions
On the voxel grid:
1. Normalize grid velocity: $\mathbf{v}_i = (m\mathbf{v})_i / m_i$.
2. Apply gravity: $\mathbf{v}_i \leftarrow \mathbf{v}_i + \Delta t \, \mathbf{g}$.
3. **Voxel World Collision:** If grid node $\mathbf{x}_i$ collides with a static world voxel:
   $$\text{If } \text{Voxel}(\mathbf{x}_i) == \text{Solid}: \quad \mathbf{v}_i \leftarrow \mathbf{0} \quad (\text{or project onto tangential plane})$$

### Step 3: Grid-to-Particle (G2P) Gather & Constitutive Model Update
Gather velocities back to particles and update internal stress:
$$\mathbf{v}_p = \sum_i w_{ip} \mathbf{v}_i$$
$$\mathbf{C}_p = \frac{4}{\Delta x^2} \sum_i w_{ip} \mathbf{v}_i (\mathbf{x}_i - \mathbf{x}_p)^T$$
$$\mathbf{x}_p \leftarrow \mathbf{x}_p + \Delta t \, \mathbf{v}_p$$

### 3.1 Constitutive Models (Drucker-Prager for Sand, von Mises for Metal)
- **Dry Sand:** Follows the Drucker-Prager yield criterion. If the shear stress exceeds internal friction $\mu \cdot p$, the deformation gradient $\mathbf{F}_p$ is projected onto the yield surface.
- **Snow / Clay:** Hardening model with plastic compression and tensile fracture.

---

## 4. Coupling MLS-MPM with Voxel Engines

In a micro-voxel engine:
1. **Dynamic Voxel-to-Particle Conversion:** When a snowdrift or sand dune is blasted by an explosion, the affected voxels in the CS64 tree are instantly converted into MLS-MPM particles ($1\text{ to }4$ particles per voxel).
2. **Particle-to-Voxel Freezing:** When MPM particles come to rest (velocity $\|\mathbf{v}_p\| < \epsilon$ for $N$ consecutive frames), they rasterize back into static leaf bricks in the world CS64 tree and are removed from the active particle buffer.

---

## 5. Performance on Modern GPUs

Using Taichi-style or compute-shader implementations:
- **Particle Count:** $100,000\text{ to }300,000$ active particles.
- **Background Grid:** $128 \times 64 \times 128$ voxel cells.
- **Compute Time on RTX 3080:** **$1.8\text{--}3.2\text{ ms}$** per frame (60 FPS capable).
- **Visual Result:** Unprecedented cinematic avalanches, mudslides, collapsing sandbanks, and fluid-solid transitions.
