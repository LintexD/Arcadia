# 07. IOLITE: Real-Time Path-Traced Voxel Engine

## 1. Overview & Heritage

Created by **Benjamin Wrensch** (known in the graphics community as *begla*, creator of the *Intrinsic* Vulkan engine and prominent shader developer) under *Missing Deadlines*, **IOLITE** is an indie C++ game engine and editor built from scratch around a **pure voxel, path-traced rendering architecture**:
- **Target Aesthetic:** Highly detailed, cubical voxel environments (authorable via MagicaVoxel and Avoyd) featuring physically accurate lighting, soft shadows, glossy reflections, and dynamic destruction.
- **Technology Stack:** High-performance data-oriented C++ core, low-level Vulkan graphics backend, Lua scripting for gameplay logic, and an embedded immediate-mode editor.

---

## 2. Rendering Pipeline: Real-Time GPU Path Tracing

Unlike hybrid voxel engines that mesh blocks into triangles, IOLITE executes **pure ray/path tracing directly through its voxel structures**:

### 2.1 Low-Level Vulkan Compute Ray Marching
- **Hardware-Agnostic Traversal:** Rather than relying exclusively on DXR hardware BVH acceleration (which incurs massive rebuild penalties when voxels are carved), IOLITE leverages **custom Vulkan compute pipelines** to step rays through hierarchical voxel partitions.
- **Blue Noise & Low-Discrepancy Sampling:** Primary, shadow, and indirect rays are jittered using blue noise textures to convert structured aliasing into high-frequency, easily filterable noise.
- **Spatiotemporal Denoising:** Utilizes temporal history reprojection and edge-stopping spatial filtering (inspired by SVGF) to reconstruct clean diffuse and specular global illumination from $1\text{ spp}$ path tracing.

### 2.2 Minimalist Polynomial AgX Color Transform
Traditional game engines use ACES filmic tonemapping, which can cause hue shifts (e.g. bright orange fire shifting towards harsh yellow) and blown-out saturation in high-energy emissive scenes.
- IOLITE pioneered a lightweight, drop-in implementation of **Troy Sobotka’s AgX Color Display Transform**.
- Rather than loading a massive $65 \times 65 \times 65$ 3D Look-Up Table (LUT) in VRAM, Wrensch formulated a **pure polynomial sigmoid approximation** running in single-digit ALU cycles in shader post-processing.
- **Result:** Pristine handling of super-bright emissive voxels (torches, lasers, molten lava) without chromatic clipping or color shifts.

### 2.3 Reverse-Z Floating-Point Depth Architecture
- Standard depth buffers map the near plane to $0.0$ and far plane to $1.0$. Because IEEE 754 floating-point numbers have high density near $0.0$, precision is wasted in the near field, causing severe Z-fighting on distant voxel geometry.
- IOLITE strictly uses **Reverse-Z** (Near $= 1.0$, Far $= 0.0$) with an unsigned float depth buffer (`VK_FORMAT_D32_SFLOAT`).
- **Result:** Infinite far-clip planes with zero Z-fighting across kilometers of voxel landscapes.

---

## 3. Physics & Numerical Integration

In destructible voxel worlds, frame rates can fluctuate from 30 FPS to 120+ FPS depending on explosions. A major technical contribution of IOLITE’s engineering devlogs is its formulation of **stable numerical integration under variable delta times**:

### 3.1 Numerical Integrators
- Explicit (Forward) Euler is prone to catastrophic energy inflation when objects collide.
- IOLITE implements and evaluates:
  1. **Semi-Implicit (Symplectic) Euler:**
     $$\mathbf{v}_{t+\Delta t} = \mathbf{v}_t + \mathbf{a}_t \Delta t$$
     $$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \mathbf{v}_{t+\Delta t} \Delta t$$
     *Crucial property:* Symplectic integrators preserve phase-space volume, preventing orbiting or resting bodies from gaining artificial energy.
  2. **Velocity Verlet:**
     $$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \mathbf{v}_t \Delta t + \frac{1}{2} \mathbf{a}_t \Delta t^2$$
     $$\mathbf{v}_{t+\Delta t} = \mathbf{v}_t + \frac{1}{2}(\mathbf{a}_t + \mathbf{a}_{t+\Delta t}) \Delta t$$
     Provides second-order accuracy for smooth parabolic ballistic trajectories (falling debris, projectiles).

### 3.2 Time-Step Clamping & Sub-Stepping
- To prevent tunneling through thin 1-voxel walls during sudden framerate drops (e.g. giant explosive collapses), the simulation clamps max frame $\Delta t$ and executes fixed mini-substeps (e.g. $4 \times \Delta t_{\text{sub}}$), guaranteeing that high-speed projectiles never pass through solid voxel barriers.

---

## 4. Architectural Lessons for Arcadia

| IOLITE Engineering Feature | Arcadia Architecture Comparison | Strategic Takeaway for Arcadia |
|---|---|---|
| **AgX Color Display Transform** | Arcadia currently uses ACES filmic tonemapping in [`composite_dlaa.wgsl`](../../shaders/composite_dlaa.wgsl). | Adding an AgX polynomial mode in [`blit.wgsl`](../../shaders/blit.wgsl) would prevent glowing crystals and magma from blowing out to white/yellow. |
| **Reverse-Z Projection** | Arcadia’s camera matrix uses standard depth mapping. | Adopting Reverse-Z (`near = 1.0, far = 0.0`) in [`camera.rs`](../../crates/arcadia-render/src/camera.rs) eliminates distant depth quantization during Catmull-Rom upscaling. |
| **Symplectic Sub-Stepping** | Arcadia’s [`rigid_voxel.rs`](../../crates/arcadia-physics/src/rigid_voxel.rs) uses 4 substeps per frame. | Validates Arcadia's 4-substep impulse solver; calibrating damping prevents micro-bouncing. |
| **MagicaVoxel `.vox` Pipeline** | Arcadia uses procedural generation in [`generator.rs`](../../crates/arcadia-world/src/generator.rs). | Adding a `.vox` importer directly into `ChunkTree64` allows artists to build ruins, props, and shrines in MagicaVoxel. |
