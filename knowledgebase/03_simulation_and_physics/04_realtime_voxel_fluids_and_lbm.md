# 04. Real-Time Voxel Fluids & Lattice Boltzmann Method (LBM)

## 1. Overview & Fluid Simulation Regimes in Games

Simulating fluid dynamics (water, smoke, fire, toxic gas) in real-time voxel worlds spans two primary regimes:
1. **Eulerian Grid Methods (Stable Fluids / Navier-Stokes):** Solves pressure projection and advection across a fixed 3D Cartesian velocity grid.
2. **Mesoscopic Particle Methods (Lattice Boltzmann Method - LBM):** Models fluid packets streaming and colliding across discrete velocity directions on a regular cubic lattice.

Both methods operate natively on regular 3D grids, making them extraordinarily well-suited for voxel engines.

---

## 2. Eulerian Voxel Fluids: Jos Stam's Stable Fluids

### 2.1 The Incompressible Navier-Stokes Equations
$$\frac{\partial \mathbf{u}}{\partial t} = -(\mathbf{u} \cdot \nabla)\mathbf{u} - \frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$$
$$\nabla \cdot \mathbf{u} = 0 \quad (\text{Incompressibility})$$
Where $\mathbf{u}$ is the velocity vector field, $p$ is pressure, $\rho$ is density, and $\nu$ is kinematic viscosity.

### 2.2 The Real-Time Split Pipeline
In a GPU compute pipeline, each time-step $\Delta t$ is solved in 4 decoupled passes:
1. **Add External Forces:** $\mathbf{u}^* = \mathbf{u} + \Delta t \, \mathbf{f}_{\text{ext}}$ (gravity, buoyancy, wind).
2. **Semi-Lagrangian Advection:** Trace a virtual particle backwards along the velocity field:
   $$\mathbf{x}_{\text{prev}} = \mathbf{x} - \Delta t \, \mathbf{u}(\mathbf{x})$$
   Sample $\mathbf{u}(\mathbf{x}_{\text{prev}})$ via trilinear interpolation. Unconditionally stable for arbitrarily large $\Delta t$!
3. **Diffusion (Viscosity):** Solved via Jacobi iterations: $\mathbf{u}^{**} = (\mathbf{I} - \nu \Delta t \nabla^2)^{-1} \mathbf{u}^*$.
4. **Pressure Projection (Poisson Equation):**
   $$\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^{**}$$
   Solved via 20–40 Red-Black Gauss-Seidel or Conjugate Gradient iterations on the GPU.
   Finally, subtract the pressure gradient: $\mathbf{u}^{n+1} = \mathbf{u}^{**} - \frac{\Delta t}{\rho} \nabla p$.

### 2.3 Boundary Conditions with Static Voxel Geometry
Where the fluid grid touches static solid voxels:
- Enforce the **No-Slip Condition** ($\mathbf{u} = \mathbf{0}$) or **Free-Slip Condition** ($\mathbf{u} \cdot \mathbf{n} = 0$).
- Solid voxels are encoded as a 1-bit boundary mask. In pressure projection iterations, boundary neighbors are replaced with Neumann conditions ($\frac{\partial p}{\partial \mathbf{n}} = 0$).

---

## 3. Lattice Boltzmann Method (LBM) on GPU Compute

### 3.1 The Mesoscopic Kinetic Foundation
Rather than tracking velocity vectors directly, LBM tracks discrete velocity distribution functions $f_i(\mathbf{x}, t)$ representing particle probability densities moving along discrete lattice vectors $\mathbf{c}_i$.

The standard 3D lattice is **D3Q19** (3 dimensions, 19 discrete velocities: center, 6 cardinal faces, 12 diagonal edges):
$$\mathbf{c}_i \in \{ (0,0,0), (\pm 1, 0, 0), (0, \pm 1, 0), (0, 0, \pm 1), (\pm 1, \pm 1, 0), \dots \}$$

```text
D3Q19 Discrete Velocity Lattice
        ▲ +Y
        │   ↗ +Z
        │  /
  ◄─────┼─────► +X
       /│
      / │
     ▼
```

### 3.2 The BGK (Bhatnagar-Gross-Krook) Collision & Streaming Steps
The simulation loop consists of two beautifully parallelizable steps:

1. **Collision Step (Local Relaxation):**
   $$f_i^*(\mathbf{x}, t) = f_i(\mathbf{x}, t) - \frac{1}{\tau} \left( f_i(\mathbf{x}, t) - f_i^{\text{eq}}(\mathbf{x}, t) \right)$$
   Where $\tau$ is the relaxation time determining kinematic viscosity: $\nu = c_s^2 (\tau - 0.5) \Delta t$.
   $f_i^{\text{eq}}$ is the Maxwell-Boltzmann equilibrium distribution calculated strictly from local density $\rho$ and velocity $\mathbf{u}$. **Zero linear system solvers or Poisson iterations are needed!**

2. **Streaming Step (Advection):**
   $$f_i(\mathbf{x} + \mathbf{c}_i \Delta t, t + \Delta t) = f_i^*(\mathbf{x}, t)$$
   Move distribution values to neighboring lattice cells.

### 3.3 Bounce-Back Boundary Conditions for Arbitrary Voxel Shapes
LBM handles intricate, broken, crumbling voxel geometry with unmatched elegance:
- If neighbor cell $\mathbf{x} + \mathbf{c}_i$ is a solid voxel:
  $$f_{\text{opposite}(i)}(\mathbf{x}, t + \Delta t) = f_i^*(\mathbf{x}, t)$$
- The particle distribution simply reflects backwards along its incidence angle!
- **Zero normal calculation or complex collision meshes are required.** Any arbitrary voxel destruction is handled automatically without special cases.

---

## 4. Performance & Memory Comparison: Stable Fluids vs LBM

| Metric | Stable Fluids (Eulerian) | LBM (D3Q19 Lattice) |
|---|---|---|
| **Memory per Cell** | $\approx 20\text{ bytes}$ ($\mathbf{u}: 12\text{B}, p: 4\text{B}, \rho: 4\text{B}$) | $\approx 80\text{ bytes}$ ($19 \times 4\text{B}$ floats, ping-ponged) |
| **GPU Compute Type** | Iterative (20–40 passes for pressure) | **Single-pass Streaming & Collision** |
| **Voxel Boundary Handling** | Masked Poisson boundary coefficients | **Instant Bounce-Back (Bitwise check)** |
| **Frame Budget ($128^3$ Grid)** | $2.5\text{--}4.5\text{ ms}$ (RTX 3080) | **$1.1\text{--}1.8\text{ ms}$ (RTX 3080)** |
| **Physical Fidelity** | Viscous damping, artificial dissipation | Exceptionally realistic turbulence & vortices |

> **Takeaway:** For large smoke, gas, and dynamic fire plumes in destructible voxel games, **GPU LBM with D3Q19** provides unmatched physical turbulence and trivial boundary collisions with broken voxel ruins at lower frame times than iterative Navier-Stokes solvers.
