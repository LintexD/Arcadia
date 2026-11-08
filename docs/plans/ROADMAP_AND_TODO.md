# Arcadia Voxel Engine — Comprehensive Roadmap & Technical TODO List

*Updated: November 2026*  
*Target Hardware Context: Discrete RTX 3080 (10 GB VRAM) & AMD Ryzen 5 7535U (Radeon 660M RDNA2 Integrated GPU, Vulkan 1.3)*

---

## 1. Project Health & Implementation Overview

| Milestone / Subsystem | Primary Crates / Shaders | Current Status |
|---|---|---|
| **Phase 1: CS64 Sparse 64-Tree Data Structure** | `arcadia-core` | **Completed** |
| **Phase 2: GPU Compute Raymarcher & Fantasy World** | `arcadia-render`, `shaders/` | **Completed** |
| **Phase 3: Kinematic Character Controller & AABB Physics** | `arcadia-physics` | **Completed** |
| **Phase 4: Voxel Destruction & 16k GPU Debris** | `arcadia-world`, `shaders/` | **Completed** |
| **Phase 5: ReSTIR GI, Wavelet Denoiser, DLAA/TSR** | `shaders/` | **Completed** |
| **Phase 5B: Sanctuary Benchmark, RCAS Upscaler & TPDF Dither** | `arcadia-app`, `arcadia-render` | **Completed** |
| **Phase 6: Structural Island Detachment & 6-DOF Rigid Voxel Bodies** | `arcadia-world`, `arcadia-physics` | **Completed** |
| **Phase 6B: Hardware Reverse-Z Infinite Depth (`Depth32Float`)** | `arcadia-render`, `shaders/` | **Completed** |
| **Phase 7: In-Place Leaf Brick Mutation & Zero-Repack Sculpting** | `arcadia-core`, `arcadia-world` | **Completed** |
| **Phase 8: Compound Colliders, Multi-Point Manifolds & Stacking Shock** | `arcadia-physics` | **Completed** |
| **Phase 8B: Asynchronous Resting Rigid Body Baking Pipeline** | `arcadia-app`, `arcadia-world` | **Completed** |
| **Phase 8C: Rolling Frame Pacing Tracker & HUD Overlay (`[F3]`)** | `arcadia-app` | **Completed** |
| **Phase 8D: 2D Boundary Surface Shell Flood-Fill ($O(N^2)$ Surface Traversal)** | `arcadia-world` | **Completed** |
| **Async Mega-Island Graph Worker (`WorldRegionSnapshot`)** | `arcadia-world`, `arcadia-app` | **In Progress / Optimization** |
| **Dynamic Voxel Articulated Joints & Cable Constraints** | `arcadia-physics` | **Planned / Near-Term** |
| **Advanced Visuals (ReSTIR DI, Roughness GGX, AgX, Caustics)** | `shaders/`, `arcadia-render` | **Planned / Active Research** |
| **World Simulation (Fluids, Fire, Infinite 3D Clipmap)** | `arcadia-world` | **Planned / Mid-Term** |
| **Asset Pipelines (MagicaVoxel `.vox` Ingestion & Prefabs)** | `arcadia-core`, `arcadia-world` | **Planned / Mid-Term** |
| **Engine Infrastructure (Vulkan Ray Query, Voxel Audio, Sandbox Tools)** | `arcadia-render`, `arcadia-app` | **Planned / Long-Term** |

---

## 2. Completed Milestones (Authoritative Technical Record)

### Core Architecture & Hierarchical Voxel Representation (Phases 1–3)
- **Hierarchical CS64-Tree (`arcadia-core`)**: 4-level Sparse 64-Tree ($256^3$ voxels per $16\text{ m}^3$ chunk, $6.25\text{ cm}$ resolution), 16-byte `Node64` with hardware POPCNT child indexing, homogeneous solid node collapsing, and BFS serialization.
- **Compute Raymarcher (`arcadia-render`, `shaders/`)**: Pure compute DDA traversal (`traversal64.wgsl`), 3D Morton chunk sorting, and adaptive ray-cone LOD with zero-brick termination for indirect GI rays.
- **Kinematic Character Controller (`arcadia-physics`)**: Bitwise `u64` AABB collision solver against the CS64 hierarchy, 1..3 voxel auto-step-up ($6.25\text{ cm} - 18.75\text{ cm}$), Move-To-Front (MTF) collision result caching, water immersion buoyancy/drag, and dual-mode walking/creative flight controls.

### Real-Time Destruction & GPU Simulation (Phase 4)
- **Real-Time World Sculpting**: Explosive sphere carving, scorch/ember rims, and voxel brush placement.
- **16k GPU Voxel Debris**: 16,384 GPU-simulated bouncing $6.25\text{ cm}$ voxel debris cubes in `primary_gbuffer.wgsl` interacting with world geometry via compute shaders.

### ReSTIR GI, Reconstruction & Temporal Upscaling (Phases 5 & 5B)
- **Lighting & Upscaling Stack (`shaders/`)**: Spatio-temporal ReSTIR GI with plane-distance validation, 16-tap edge-stopping wavelet denoiser, 5-tap Catmull-Rom TSR + AMD FSR 1.0 RCAS sharpening, and Halton sub-pixel temporal jitter.
- **Hardware Reverse-Z Depth (`Depth32Float`)**: Infinite far plane (`Mat4::perspective_infinite_reverse_rh`) with `CompareFunction::GreaterEqual` depth testing and $0.0$ clear, eliminating floating-point depth compression across vast distances.
- **Hardware Auto-Calibration Engine**: Calibrated 30/60 FPS quality presets (`[C]` key / CLI) scaling internal resolution and secondary ray step budgets quadratically.
- **Visual Artifact Eradication**: Restored daytime sun shadows (`min_shift = 0u`), eradicated mirror border white dots with normal offsets and continuation rays, removed cave daylight leakage, and implemented decorrelated 2D PCG TPDF dithering (`[K]` toggle).
- **Deterministic Sanctuary Benchmark Suite**: Catmull-Rom spline camera flying through a stress-test scene (colonnade, mirror walls, reflecting basin, magma cascade, 6 point lights) with automated JSON and Markdown reporting.

### Structural Island Collapse & 6-DOF Rigid Bodies (Phase 6)
- **Targeted Perimeter-Seeded Island Detection (`arcadia-world/src/islands/perimeter.rs`)**: Replaced artificial bounding-box compressive support and cantilever limits with local crater boundary shell flood fill ($r \in [R, R+2.5\text{ voxels}]$). Supports downward $-Y$ load paths, instant bedrock ($y \le 4$, `MAT_BEDROCK`) short-circuiting ($<0.1\text{ ms}$ turnaround), and mass threshold cutoffs.
- **Crater Rim Crumb Cleanup**: Detached fragments $\le 8$ voxels (`SMALL_ISLAND_PARTICLE_MAX_VOXELS`) instantly convert into physical GPU debris particles or dissolve, eliminating floating single-voxel artifacts even when structural detachment is disabled.
- **6-DOF Rigid Body Dynamics (`arcadia-physics`)**: Discrete center of mass, exact $3\times3$ moment of inertia tensor with $\frac{s^2}{6}$ cubic voxel parallel-axis correction, 15-axis SAT OBB broadphase, dynamic GPU instance serialization (`GpuRigidBodyInstance`), and symplectic Euler integration.
- **Dynamic Body In-Flight Sculpting & Splitting**: Blasting through dynamic bodies partitions them into independent rigid bodies (`find_connected_components`) preserving local coordinate frames, orientation quaternions, and momentum. Right-click places voxels directly onto dynamic bodies, expanding bounds and recomputing inertia on the fly.
- **Kinetic Impact Shattering**: High-speed collisions ($>18\text{ m/s}$) shatter small bodies ($<300$ voxels) into physical GPU debris particles.

### In-Place Leaf Brick Mutation & Zero-Repack Sculpting (Phase 7)
- **Zero-Repack Engine (`crates/arcadia-core/src/chunk_tree/edit/`)**: Completely eliminated full-chunk `unpack_to_build` and BFS `pack_build_tree` allocations during real-time carving, placement, clearing, and setting.
- **In-Place Mutation**: Directly mutates `LeafBrick64` in `self.bricks` and updates parent `Node64` bitmasks and LOD payloads in `self.nodes`.
- **Lazy Branch Expansion**: Homogeneous uniform nodes uncollapse lazily on demand when edited. Contiguous sibling slices append and re-index in `POPCNT` rank order.
- **In-Place Sibling Compaction**: When leaf nodes empty (`mask == 0`), pruning shifts remaining sibling nodes within their existing contiguous allocation (`nodes[old_base + rank .. old_base + new_count]`) and zeroes the trailing slot without vector reallocations.
- **Fine-Grained GPU Staging (`ChunkEditReport`)**: Tracks `[min_dirty_node..=max_dirty_node]` and `[min_dirty_brick..=max_dirty_brick]`. Used by `WorldMap::sync_dirty_chunk_partial` to stream sub-slice GPU buffer writes (`queue.write_buffer`) in sub-milliseconds.
- **Compaction Pass (`ChunkTree64::compact()`)**: Dedicated garbage-collection pass to reclaim orphaned node/brick slices after heavy destruction and restore clean contiguous BFS level-order packing.

### Advanced Contact Mechanics, Shock Passes & Stacking Stability (Phase 8)
- **Compound 3D Greedy Collider Synthesis (`rigid_voxel/collision/compound.rs`)**: Decomposes sparse voxels into disjoint solid cuboids (`VoxelSubBox`) with a median-split flat AABB BVH.
- **Exposed-Face Bitmasks & Internal Seam Culling**: Tracks exposed face bitmasks (`FACE_NEG_X` .. `FACE_POS_Z`). Unexposed internal seams are culled during SAT tests, completely eliminating phantom snagging, edge catching, and micro-jitter.
- **Multi-Point Contact Manifolds & Concave Interlocking (`collision/manifold.rs`)**: Tandem BVH traversal with exposed-face 15-axis SAT narrowphase, Sutherland-Hodgman polygon clipping ($3\text{ mm}$ slop), optimal 4-point quad selection, and multi-manifold clustering (up to 4 directional manifolds per pair with $\cos \theta \ge 0.85$), enabling true concave interlocking (arches, bowls, crossed pillars) without air padding.
- **DAG Support Level Graph & Stacking Shock Passes (`solver.rs`)**: Orders contact manifolds bottom-up from bedrock/terrain. Executes bottom-up velocity shock passes and split-impulse position shock passes (treating lower support bodies as infinite mass during the pass) to eliminate stacking sponginess.
- **Fluid Mechanics in Rigid Bodies**: Stratified volume-weighted water submersion calculation, progressive hydrodynamic drag ($1.5v + 0.8v^2$), rotational viscous damping, and buoyancy ($18 \frac{\rho_w}{\rho_b}$).
- **2D Support Polygon Stability & Island Sleeping**: Union-Find clustering of resting contact graphs. Evaluates 2D support polygons on XZ (`is_com_stably_supported`) into `SupportSleepKind` (`StablePolygon`, `MultiCradled`, `SingleContact`, `None`) with low-speed micro-jitter damping.

### Asynchronous Resting Rigid Body Baking Pipeline (Phase 8B)
- **Frustum Visibility Gating**: Bodies resting for $t \ge \text{object\_bake\_timeout\_s}$ (configurable 5..180s) evaluate camera frustum visibility (`is_visible_to_camera`) with a 15% peripheral guard margin and 150m cutoff distance. Off-screen bodies (or bodies exceeding $3\times$ forced timeout) are selected for baking.
- **Multi-Threaded Background Rasterization**: Dispatches resting bodies to `rayon::spawn`, rasterizing rotated voxels into integer world coordinates (`body.to_world_voxels()`) and transmitting via `mpsc::channel` (`BakeResult`).
- **Time-Sliced Main-Thread Commits & Zero Pop-In**: Consumes at most 1 bake completion per frame into `WorldMap::set_voxels_global`. Baking bodies continue rendering dynamically on GPU until the exact frame their static voxels commit, eliminating visual popping and frame latency spikes.
- **Eviction Immunity**: Baking bodies are protected from dynamic capacity eviction.

### Rolling Frame Pacing & Stability Tracking (Phase 8C)
- **Rolling Frame Time History (`frame_graph.rs`)**: Tracks 180 frames (3 seconds at 60 FPS) in a rolling `VecDeque<f32>` buffer.
- **Real-Time Statistical Metrics**: Evaluates current ms, average ms, min/max ms, standard deviation, 1% low FPS, and frametime consistency percentage.
- **Bottom-Right Egui HUD Overlay (`[F3]`)**: Toggled with `[F3]`, displaying color-coded FPS, jitter rating (`STABLE`, `MODERATE`, `JITTER`), 16.7ms and 33.3ms reference lines, dynamic histogram bars, and min/max/1% low stats.

### 2D Boundary Surface Shell Flood-Fill Optimization (Phase 8D)
- **Separable 26-Neighborhood Air Dilation (`bitwise.rs`)**: Evaluates Chebyshev distance 1 air coverage within each $4\times 4\times 4$ brick via 3 separable 1D bit shifts in $\approx 5\text{ ns}$ of pure ALU register ops. Solves the digital concave crease problem so orthogonal floor/pillar junctions preserve 6-connected surface continuity.
- **Surface-Only Traversal (`perimeter.rs`)**: Replaces $O(N^3)$ solid interior volume expansion with $O(N^2)$ 2D outer shell flood-fill during crater perimeter island detection. Skips 100% of subterranean solid rock in attached mountains and cliffs.
- **Deferred 3D Solid Interior Fill**: Only when a candidate surface boundary shell closes without touching bedrock ($y \le 4$) or exceeding bounds (`!is_grounded`) does a 3D flood fill run inward from the shell to harvest internal rock for dynamic `RigidVoxelBody` instantiation.
- **Performance Impact**: Slashes non-detaching terrain carve turnaround by $>10\times$ (visiting $<600$ surface voxels instead of thousands of subterranean solid rock voxels), eliminating destruction frame hitches.

---

## 3. Active Technical Challenges & Near-Term Action Items

### P1. Asynchronous Mega-Island Graph Worker (`WorldRegionSnapshot`)
- **Current State**:
  - Perimeter-seeded island detection runs synchronously on the main thread in `carve_sphere_meters_with_limits` and resolves typical destruction ($<768$ voxels) in $<0.1\text{ ms}$.
  - `WorldRegionSnapshot` exists in `arcadia-world/src/islands/query.rs` with thread-safe voxel querying.
- **Challenge**:
  - Extremely large explosions or deep structural cuts that sever massive monuments ($>32,768$ voxels) can cause a single-frame stutter during BFS queue expansion if synchronous limits are set high.
- **Action**:
  - When perimeter search detects an ungrounded component exceeding a synchronous exploration budget (e.g., 2,048 voxels), extract a `WorldRegionSnapshot` and dispatch the remainder of the flood fill to a background `rayon` task.
  - Return the severed component to the main thread via an `mpsc` channel to instantiate the new `RigidVoxelBody` on the subsequent frame.

### P3. High-Capacity Dynamic Body GPU Caching
- **Current State**:
  - `MAX_RIGID_BODIES = 32`: Active dynamic bodies are raymarched in `primary_gbuffer.wgsl` at `@group(0) @binding(6)`.
  - When more than 32 bodies are spawned, sleeping bodies are baked or non-baking bodies are evicted.
- **Challenge**:
  - In massive demolition scenes, having only 32 simultaneous dynamic bodies can cause premature static baking or eviction of moving rubble.
- **Action**:
  - Expand GPU storage buffer capacity to 64 or 128 instances.
  - Implement a two-tier dynamic body hierarchy: the top 32 closest/largest bodies are full raymarched CS64 trees; distant or small debris bodies are rendered via instanced oriented bounding boxes (OBBs) or simplified single-node raymarchers.

### P4. Explosive Impulse Proximity & Velocity Clamping
- **Current State**:
  - Carving near dynamic bodies applies impulse and torque in `carve_sphere_meters`.
- **Challenge**:
  - Deep penetration blasts centered directly inside or underneath a small rigid body can transfer excessive vertical impulse ($>50\text{ m/s}$).
- **Action**:
  - Clamp explosive blast velocity to a maximum linear speed (e.g. $22\text{ m/s}$) and add a mass-proportional inertia cap to prevent lightweight fragments from catapulting into the upper atmosphere.

---

## 4. Future Architectural Plans (Detailed Technical Roadmap)

### Category C: Advanced Visuals, Lighting & Material Shaders

#### C1. Multi-Candidate ReSTIR DI for Dynamic Emissive Voxels
- **Architecture**:
  - Replace the fixed 8-point-light limit in `lighting_restir_gi.wgsl` with a spatial acceleration grid supporting thousands of dynamic emissive voxels (crystals, lava cascades, magical runes, torch braziers).
  - Construct a two-level $16^3$ voxel macro-cell acceleration grid in GPU VRAM. Each macro-cell accumulates radiant power $\Phi_{\text{cell}} = \sum A_i \cdot L_e$ using 32-bit atomic additions.
  - **Algorithm**:
    1. Primary hit samples 16 candidate emissives using two-stage importance sampling: select macro-cell with probability $\propto \Phi_{\text{cell}}$, then choose an active voxel within that cell.
    2. Maintain temporal and spatial reservoirs evaluating target PDF $\hat{p} = \frac{|\mathbf{n} \cdot \mathbf{l}| \cdot L_e}{\|\mathbf{x} - \mathbf{y}\|^2}$.
    3. Resample across a 1-tap spatial neighbor and trace exactly 1 shadow ray to the winning light source.
- **Target Performance**: $< 1.2\text{ ms}$ on Radeon 660M for 8,192 active emissive sources.

#### C2. Roughness-Aware Glossy Reflection Tracing & GGX Importance Sampling
- **Architecture**:
  - Expand `primary_gbuffer.wgsl` and `denoiser_reconstruct.wgsl` beyond ideal mirror reflections ($R = 0.02$) to support arbitrary material roughness (e.g. brushed brass, polished marble, damp slate, wet mud).
  - Use GigaVoxels ray-cone LOD spread: initialize ray cone angle $\theta_{\text{cone}} = 2 \arctan(\alpha_{\text{roughness}})$, filtering the CS64 tree payload to higher LOD nodes when the cone footprint exceeds micro-voxel bounds.
  - Employ GGX importance sampling for the reflection direction:
    $$\mathbf{h} = \text{sample\_ggx}(\mathbf{n}, \alpha_{\text{roughness}}, \xi), \quad \mathbf{r} = 2(\mathbf{v} \cdot \mathbf{h})\mathbf{h} - \mathbf{v}$$
  - Integrate with Pass 3 wavelet filter, modulating spatial kernel footprint by $\sqrt{\alpha_{\text{roughness}}}$ to blur glossy reflections without noise.

#### C3. Volumetric Caustics & Atmospheric Light Shafts
- **Architecture**:
  - Raymarch volumetric single-scattering media for underwater lagoon depths and sunlight piercing broken temple ceilings.
  - Evaluate transmittance $T(s) = \exp(-\sigma_t s)$ across 16 jittered shadow-tested steps along the primary camera ray in water or fog.
  - For water caustics, generate a dual-layer procedural wave interference normal field on the water surface and project caustic intensity using forward ray divergence or screen-space directional chromatic dispersion.

#### C4. Polynomial AgX Color Display Transform
- **Architecture**:
  - Implement Troy Sobotka's AgX display transform as a selectable alternative to ACES in `blit.wgsl`.
  - AgX uses a wide color gamut (Rec.2020/Linear), applies an input inset matrix to prevent edge-of-gamut clipping, passes luminance through an optimized polynomial sigmoid tone curve, and rotates chrominance smoothly toward white at peak intensities.
  - Solves color blowout artifacts on high-energy emissive voxels (molten lava staying rich orange-red instead of shifting to canary yellow; blue energy crystals keeping cyan/cobalt hues).

#### C5. Voxel Radiance Cache (VRC) for Enclosed Space GI Convergence
- **Architecture**:
  - Implement a 3D clipmap radiance cache (e.g. $1\text{m}$ grid spacing covering $64\text{m} \times 32\text{m} \times 64\text{m}$ around the camera).
  - Each cell stores incoming radiance projected into L1 Spherical Harmonics (4 coefficients per RGB channel).
  - Secondary ReSTIR GI rays that travel beyond $16\text{m}$ sample the VRC instead of tracing full depth or defaulting to sky radiance.
  - Completely eliminates the 2–3 frame convergence lag when moving from bright daylight into deep, enclosed caves and temples.

#### C6. Hardware-Accelerated Voxel Ray Queries (`VK_KHR_ray_query`)
- **Architecture**:
  - On hardware supporting Vulkan ray tracing extensions (`VK_KHR_ray_query` / DXR 1.1), build coarse bottom-level acceleration structures (BLAS) for chunk Level-1/Level-2 macro-nodes.
  - Dispatch rays using inline ray queries (`rayQueryInitializeEXT`, `rayQueryProceedEXT`) to let hardware RT cores execute coarse spatial ray-box culling, falling back to compute shader bitwise leaf brick intersection.
  - Accelerates primary and shadow raymarching by $2\times$ to $3\times$ on modern RTX and RDNA3 GPUs while retaining the pure compute fallback path for integrated GPUs.

---

### Category D: World Simulation, Generation & Assets

#### D1. Infinite Toroidal Chunk Streaming (3D Clipmap)
- **Architecture**:
  - Replace the fixed `grid_dims` grid in `WorldMap` with a toroidal (wrap-around) 3D clipmap ring buffer centered on the player's position.
  - Chunks outside the active viewing radius are persisted to disk or evicted, while new chunks ahead of movement are generated asynchronously via `rayon` worker pools.
  - Chunk coordinates wrap via modulo indexing:
    $$\text{slot}(x, y, z) = (x \bmod N_x, y \bmod N_y, z \bmod N_z)$$
  - Updates stream incrementally to the GPU chunk pool via `GpuPatch` descriptors without requiring full VRAM buffer reallocations.

#### D2. Cellular Automata "Living Water" & Viscous Lava Simulation
- **Architecture**:
  - Implement a micro-voxel cellular automata simulation running over CS64 leaf bricks.
  - **State Representation**: Pack fluid mass ($0..15$), velocity vector (3 bits per axis), and fluid type (`WATER`, `LAVA`, `MUD`) into a compact 16-bit payload per voxel cell.
  - **Rules**:
    1. Gravity: Downward vertical propagation into empty cells (`MAT_AIR`).
    2. Lateral Pressure Equalization: Fluid levels equalize across horizontal neighbors when downward flow is blocked.
    3. Phase Interaction: Water contact with Lava quenches the lava into `MAT_BASALT` or `MAT_OBSIDIAN`, producing steam particles (`MAT_DUST_SMOKE`).
  - Simulation runs on a worker thread or GPU compute pass, synchronizing modified bricks into the world tree via in-place mutation (Phase 7).

#### D3. Thermal Conduction & Firestorm Simulation
- **Architecture**:
  - Voxel temperature state machine running across flammable materials (`MAT_WOOD`, `MAT_LEAVES`, `MAT_MOSS`).
  - Heat transfers via conduction through 6-connected neighbor voxels and upwards via convection.
  - When material temperature exceeds flashpoint ($300^\circ\text{C}$ for wood), material converts to `MAT_EMBER_GLOW`, emitting heat and light.
  - Upon combustion depletion, embers convert to `MAT_SCORCHED_ROCK` and then disintegrate to `MAT_AIR`, naturally triggering Phase 6 structural collapse on burned support columns.

#### D4. MagicaVoxel `.vox` Asset Ingestion Pipeline
- **Architecture**:
  - Develop a zero-dependency RIFF parser for MagicaVoxel `.vox` files in `arcadia-core` or dedicated `arcadia-assets` crate.
  - Read `MAIN`, `SIZE`, `XYZI`, `RGBA`, and scene graph `nTRN` / `nGRP` / `nSHP` chunks.
  - Map MagicaVoxel 256-color palettes to Arcadia's PBR material indices using CIELAB $\Delta E$ color distance matching with emissive and metallic heuristics.
  - Support exporting `.vox` directly into:
    1. Isolated `ChunkTree64` objects for static world placement.
    2. Dynamic `RigidVoxelBody` templates for destructible props, furniture, and vehicles.

#### D5. Procedural Weather & Surface Wetness / Snow Accumulation
- **Architecture**:
  - Dynamic weather state machine: Clear Sky, Overcast Fog, Heavy Rain, Snowfall.
  - Rain accumulates a global `surface_wetness` scalar: darkens porous albedo (limestone, dirt, slate) by $20\%$, drops roughness to $\sim 0.05$ (mirror-like puddle reflections on horizontal faces), and drives procedural raindrop ripple rings on water basins.
  - Snow accumulates procedural white voxel caps on exposed upward-facing voxels ($n_y > 0.7$) using a depth-buffer snow deposition pass.

---

### Category E: Advanced Physics, Dynamics & Destruction Mechanics

#### E1. Dynamic Voxel Articulated Joints & Constraints
- **Architecture**:
  - Implement pairwise constraint equations between `RigidVoxelBody` instances solved within `rigid_voxel/solver.rs`:
    1. **Revolute Joint (Hinge)**: Locks 3 linear DOFs and 2 angular DOFs, allowing free rotation around a single hinge axis (destructible wooden gates, swinging castle doors, trapdoors, signboards).
    2. **Prismatic Joint (Slider)**: Constrains motion to a single linear axis (hydraulic elevators, sliding portcullis gates).
    3. **Spherical Joint (Ball-and-Socket)**: Constrains translation while allowing full 3D rotation (hanging lantern chains, wrecking balls).
    4. **Fixed Weld Joint**: Rigidly locks two bodies together with a breakable stress threshold $\sigma_{\text{break}}$ (welded metal beams, glued scaffolding).

#### E2. Dynamic Cable, Rope & Chain Physics (Position-Based Dynamics)
- **Architecture**:
  - 1D particle chains simulated via Position-Based Dynamics (PBD) / Verlet integration.
  - Cables attach directly to voxel coordinates on dynamic bodies and static terrain.
  - Enforces distance constraints ($C(\mathbf{x}_1, \mathbf{x}_2) = \|\mathbf{x}_1 - \mathbf{x}_2\| - d_0 = 0$) and bending constraints.
  - Transfers tension forces directly into rigid body linear and angular velocities: enables swinging cranes, suspension bridges that snap when supports are carved, and grappling hook mechanics.

#### E3. Aerodynamic Drag, Planar Lift & Wind Interaction
- **Architecture**:
  - Calculate cross-sectional area of falling dynamic bodies based on their oriented bounding boxes (OBBs) along the velocity vector $\mathbf{v}$.
  - Flat, thin voxel structures (planks, severed roof slabs, large stone sheets) experience aerodynamic drag and lateral lift forces:
    $$\mathbf{F}_{\text{lift}} = \frac{1}{2} \rho_{\text{air}} C_L A (\mathbf{v} \cdot \mathbf{t}) (\mathbf{v} \times \mathbf{t})$$
  - Causes detached wooden boards and thin stone slabs to glide, flutter, and tumble realistically rather than dropping straight down.

---

### Category F: Engine Infrastructure, High-Performance Compute & Tooling

#### F1. Pluggable Temporal Super-Resolution Stack (FSR 3 / DLSS)
- **Architecture**:
  - Create an abstract `UpscalerBackend` trait in `arcadia-render`:
    ```rust
    pub trait UpscalerBackend {
        fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder, inputs: UpscalerInputs);
        fn resize(&mut self, device: &wgpu::Device, input_size: UVec2, output_size: UVec2);
    }
    ```
  - **Backends**:
    1. `BuiltinTsr`: Current Catmull-Rom TSR + FSR 1.0 RCAS (zero dependencies, fully portable).
    2. `FSR3Backend`: Integrate pure-wgpu FidelityFX FSR 3 compute shaders (`wgpu-ffx`). Plugs directly into existing G-buffer HDR color, Reverse-Z depth, and 3D motion vectors.
    3. `DlssBackend`: Optional feature-gated backend via Vulkan NGX bindings for NVIDIA RTX hardware.

#### F2. Discrete Voxel Kinetic Audio & Geometric Acoustics
- **Architecture**:
  - Synthesize dynamic procedural audio driven directly by the physics contact manifold (`RigidVoxelBody` and `CharacterController`).
  - Map material IDs to physical acoustic properties:
    - Density, restitution, elastic modulus, and resonance decay time.
  - **Contact Sound Generation**:
    - When contact impulse $J_n > J_{\text{threshold}}$, trigger an impact event scaled by kinetic energy $E_k$.
    - When tangential sliding occurs with normal force ($F_t = \mu F_n$), generate continuous granular grinding noise with pitch modulated by slip velocity $|\mathbf{v}_t|$.
    - Disintegrated voxels trigger high-frequency splinter/crumble bursts.
  - **Voxel Acoustic Occlusion**: Trace 8–16 low-resolution acoustic rays in the CS64 tree between audio sources and the player ear to compute geometric occlusion and low-pass filtering through solid walls.

#### F3. Wavefront Stream Compaction & GPU Indirect Dispatch
- **Architecture**:
  - Mitigate SIMD warp divergence during Pass 2 (secondary ReSTIR GI and reflection rays) where some pixels miss geometry early while others march full depth.
  - **Pass 1 Compaction**:
    - After the primary G-buffer pass, run a prefix-sum stream compaction compute shader over active pixels.
    - Pack coordinates of pixels requiring secondary rays into a contiguous 1D linear dispatch buffer `pixel_indices[]`.
  - **Pass 2 Indirect Execution**:
    - Write the compacted count into an `IndirectArgs` buffer (`workgroups_x = (count + 63) / 64`).
    - Dispatch Pass 2 via `wgpu::ComputePass::dispatch_workgroups_indirect`, ensuring 100% compute unit occupancy on both discrete GPUs and 15W laptop APUs.

#### F4. Hybrid Raster/Voxel Depth Compositing
- **Architecture**:
  - Combine compute-raymarched voxel terrain with standard forward-rasterized polygonal meshes (player hands, weapons, skeletal characters, particle effects).
  - Bind the G-buffer depth attachment `gbuf_depth_tex` (`TextureFormat::Depth32Float` Reverse-Z) as the depth-stencil attachment for standard rasterization pipelines.
  - Rasterized polygons write directly to G-buffer depth and normal targets using `CompareFunction::GreaterEqual`.
  - Raymarcher naturally occludes voxels behind polygonal meshes and vice-versa with sub-pixel depth precision.

#### F5. In-Engine Creative Voxel Sandbox & Sculpting Suite
- **Architecture**:
  - Expand the in-game debug interface (`crates/arcadia-app/src/gui.rs`) with dedicated creative building and editing modes:
    1. **Brush Geometry**: Sphere, Cuboid, Cylinder, Cone, and Planar Slice brushes with live wireframe preview.
    2. **Material Replacement ("Paint" Mode)**: Mutates voxel material IDs without altering occupancy bits, utilizing in-place `LeafBrick64` mutation.
    3. **Box Selection & Clipboard**: Select, copy, cut, and paste 3D voxel regions as local `ChunkTree64` prefabs.
    4. **Undo / Redo Transaction Buffer**: Records inverted voxel tuples `Vec<(IVec3, MaterialId, MaterialId)>` to allow multi-step undo/redo for sculpting operations.

#### F6. Headless Automated Benchmark & CI Performance Regression Suite
- **Architecture**:
  - Support headless execution mode (`--headless --bench --auto-exit`) that runs without opening an OS window or creating an interactive surface (using Vulkan offscreen rendering).
  - Automatically runs the Sanctuary benchmark, measures average FPS, 1% lows, and frametime variance, and outputs `benchmark_ci.json`.
  - Integrates into GitHub Actions CI: flags pull requests that cause $>5\%$ regression in raymarching speed or frame pacing consistency.
