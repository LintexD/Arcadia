# Arcadia Phased Implementation History & Architectural Milestones

> Authoritative architectural milestone record detailing the engineering phases of the Arcadia Voxel Engine.  
> Cross-reference: **[ARCHITECTURE.md](../../ARCHITECTURE.md)** | **[docs/plans/ROADMAP_AND_TODO.md](ROADMAP_AND_TODO.md)** | **[docs/keybindings.md](../keybindings.md)**

---

## 1. Executive Milestone Summary

Arcadia has evolved from a pure bitwise 64-tree CPU prototype into a high-performance, real-time Vulkan compute voxel game engine featuring ReSTIR global illumination, in-place zero-repack leaf mutation, *Teardown*-style structural island destruction, 6-DOF dynamic rigid body physics with concave multi-manifolds, and hardware Reverse-Z infinite perspective projection.

| Phase | Subsystem / Focus | Primary Crates & Shaders | Status |
|---|---|---|---|
| **Phase 1** | Core CS64 Tree & CPU Reference DDA | `crates/arcadia-core` | **Completed** |
| **Phase 2** | GPU Compute Ray Marcher & Procedural World | `arcadia-world`, `arcadia-render`, `shaders/` | **Completed** |
| **Phase 3** | Walkable World & Bitwise Kinematic Collisions | `crates/arcadia-physics` | **Completed** |
| **Phase 4** | Real-Time Destruction & 16k GPU Debris | `arcadia-world`, `shaders/primary_gbuffer.wgsl` | **Completed** |
| **Phase 5** | ReSTIR GI, Wavelet Reconstruction & DLAA/TSR | `shaders/`, `arcadia-render` | **Completed** |
| **Phase 5B** | Sanctuary Benchmark, Calibration & TPDF Dithering | `arcadia-app`, `arcadia-render` | **Completed** |
| **Phase 6** | Structural Island Collapse & 6-DOF Rigid Bodies | `arcadia-world`, `arcadia-physics` | **Completed** |
| **Phase 6B** | Hardware Reverse-Z Infinite Depth (`Depth32Float`) | `arcadia-render`, `shaders/` | **Completed** |
| **Phase 7** | In-Place Leaf Mutation & Zero-Repack Sculpting | `arcadia-core`, `arcadia-world` | **Completed** |
| **Phase 8** | Multi-Manifold Physics, Compound Colliders & Stacking Shock | `crates/arcadia-physics` | **Completed** |
| **Phase 8B** | Asynchronous Resting Rigid Body Baking Pipeline | `crates/arcadia-app`, `crates/arcadia-world` | **Completed** |
| **Phase 8C** | Rolling Frame Pacing Tracker & HUD Overlay (`[F3]`) | `crates/arcadia-app` | **Completed** |
| **Phase 8D** | 2D Boundary Shell Flood-Fill Optimization ($O(N^2)$) | `crates/arcadia-world` | **Completed** |
| **Phase 9** | World Streaming, Toroidal Clipmaps & Mesh Hybridization | `arcadia-world`, `arcadia-render` | **Future Roadmap** |

---

## 2. Detailed Phased Implementation Record

### Phase 1: Core 64-Tree & CPU Reference (`crates/arcadia-core`) — COMPLETED
*Foundation of Arcadia's Chunked Sparse 64-Tree (CS64) data structures, bitwise coordinate math, and baseline CPU traversal.*

- **Coordinate System & Bit Packing:**
  - Implemented `VoxelCoord` and `ChunkCoord` with right-handed Y-up coordinate conventions.
  - Engineered 6-bit child indexing `(z << 4) | (y << 2) | x` providing $O(1)$ bitwise octave octant addressing across $4 \times 4 \times 4$ branching factors.
- **256-Entry PBR Material Palette:**
  - Designed `MaterialPalette` holding 256 physical and emissive material definitions: grass, moss, limestone, dark slate, ancient wood, water, cyan/magenta/amber glowing crystals, magma, scorched rock, `MAT_POINT_LIGHT`, and `MAT_MIRROR_CHROME`.
  - Enforced 32-byte `#[repr(C)]` POD layout with 16-byte alignment matching GPU storage buffers.
- **Binary Node Layouts & Memory Pooling:**
  - Implemented 16-byte `Node64` (`child_mask: u64`, `child_ptr: u32`, `lod_material: u16`, `flags: u16`) with hardware `POPCNT` child address calculation (`count_ones` / `countOneBits`).
  - Implemented 64-byte `LeafBrick64` (`mask: [u64; 4]` representing $4 \times 4 \times 4 = 64$ voxels with homogeneous material/LOD payload).
  - Implemented Breadth-First Search (BFS) level-order intra-chunk node packing and homogeneous subtree collapsing.
  - Implemented volumetric coverage-weighted mipmapping for distance LOD.
- **Reference CPU Raycaster & Testing:**
  - Implemented reference CPU 64-tree DDA raycaster ([`crates/arcadia-core/src/raycast_cpu.rs`](../../crates/arcadia-core/src/raycast_cpu.rs)) used for physics raycasts, collision validation, and test assertions.
  - Verified unit test suite with 100% pass rate (`cargo test -p arcadia-core`).
- **Architectural Specification:** [docs/architecture/01_storage_and_memory.md](../architecture/01_storage_and_memory.md)

---

### Phase 2: GPU Compute Ray Marcher & Fantasy World (`arcadia-world`, `arcadia-render`, `arcadia-app`) — COMPLETED
*Establishment of the Vulkan 1.3 compute backend, 3D Morton space-filling layout, and procedural fantasy environment.*

- **Vulkan & wgpu Infrastructure:**
  - Initialized `wgpu` (v23.0.0) targeting Vulkan 1.3 backend with discrete GPU selection priority and graceful integrated GPU fallback.
  - Built unified device context and swapchain management in [`crates/arcadia-render/src/context.rs`](../../crates/arcadia-render/src/context.rs).
- **Morton Z-Order World Map:**
  - Implemented `WorldMap` with 3D Morton (Z-order curve) chunk addressing for maximum cache locality in spatial lookups.
  - Procedural "Fantasy Nature & Ruins" world generator: multi-octave terraced hills, Ancient Temple Ruins with walkable staircases, archways, columns, Cliffside Crystal Grotto with crystal spires, and reflecting lagoon.
- **GPU Traversal Shaders & Ray-Cone LOD:**
  - Implemented [`shaders/common/traversal64.wgsl`](../../shaders/common/traversal64.wgsl) performing hierarchical Chunk $\to$ 4-Level 64-Tree DDA raymarching.
  - Implemented [`shaders/primary_gbuffer.wgsl`](../../shaders/primary_gbuffer.wgsl) generating primary geometric G-buffer attributes (albedo, normal, motion vectors, roughness/metallic).
  - Integrated Cyril Crassin *GigaVoxels* Ray-Cone LOD termination, adjusting ray footprint based on distance and projection angle to bound inner-loop march iterations.
- **Architectural Specification:** [docs/architecture/02_rendering_and_lighting.md](../architecture/02_rendering_and_lighting.md)

---

### Phase 3: Walkable World & Bitwise Collisions (`arcadia-physics`) — COMPLETED
*Pure bitwise AABB collision detection and first-person kinematic character controller without polygon mesh generation.*

- **Bitwise CS64 AABB Collision:**
  - Implemented bitwise `u64` AABB intersection against `ChunkTree64` nodes (`ChunkTree64::intersects_aabb_local`) and world coordinates (`WorldMap::intersects_aabb_meters`).
  - Completely bypassed traditional polygonization (marching cubes / dual contouring), computing exact collision directly against voxel bitmasks.
- **First-Person Kinematic Controller:**
  - Implemented [`crates/arcadia-physics/src/character.rs`](../../crates/arcadia-physics/src/character.rs) with Move-To-Front (MTF) collision result caching for sub-microsecond collision queries.
  - Implemented gravity, continuous velocity accumulation, ground detection, and wall-sliding along orthogonal contact normals.
  - Implemented automatic 1..3 voxel auto-step-up ($6.25\text{ cm} - 18.75\text{ cm}$), allowing seamless stair and uneven terrain ascension without jumping.
  - Implemented fluid immersion physics: buoyant upward lift and progressive hydrodynamic viscous drag when submerged.
  - Implemented creative Fly Mode toggle via double-tap `Space` (`< 0.35s`) or `F`, featuring $38\text{ m/s}$ sprint boost.
- **Architectural Specification:** [docs/architecture/03_physics_and_destruction.md](../architecture/03_physics_and_destruction.md)

---

### Phase 4: Real-Time Destruction & GPU Voxel Debris (`arcadia-world`, `particles_debris.wgsl`) — COMPLETED
*Interactive world destruction, dynamic crater creation, and high-density GPU particle physics.*

- **Real-Time World Sculpting:**
  - Implemented interactive spherical carving (`Left Click`) and material placement (`Right Click`) directly onto `ChunkTree64`.
  - Added dynamic brush radius scaling via mouse wheel ($0.25\text{ m}$ to $4.50\text{ m}$).
  - Added procedural scorch/ember rim hashing around blast craters (`[G]` toggle), baking thermal discoloration into crater walls.
  - Implemented incremental GPU staging buffer patch uploads to avoid streaming unmodified chunk sectors.
- **16k GPU Voxel Debris Simulation:**
  - Implemented 16,384 GPU-simulated $6.25\text{ cm}$ voxel debris cubes simulated entirely in compute shaders.
  - Debris particles collide with CS64 world geometry, bounce with restitution and friction, and splat directly into the primary G-buffer with accurate motion vectors.
- **Architectural Specification:** [docs/architecture/03_physics_and_destruction.md](../architecture/03_physics_and_destruction.md)

---

### Phase 5: ReSTIR GI, Soft Shadows, Ray Reconstruction & DLAA (`shaders/lighting_*`, `denoiser_*`, `composite_*`) — COMPLETED
*State-of-the-art real-time path reconstruction, spatio-temporal reservoir resampling, and temporal super-resolution.*

- **Dynamic Sun & Analytical Beacons:**
  - Implemented blue-noise jittered soft sun shadows with penumbra estimation.
  - Implemented dynamic continuous diurnal sun orbit (`[Q] / [E]`) and discrete Day / Sunset / Midnight presets (`[N]`).
  - Added analytical Point Light Beacons spawned via `MAT_POINT_LIGHT` (`Key 7`).
- **ReSTIR GI Spatio-Temporal Resampling:**
  - Implemented [`shaders/lighting_restir_gi.wgsl`](../../shaders/lighting_restir_gi.wgsl) with 2D $R_2$ / PCG decorrelated hemisphere sampling.
  - Temporal candidate reuse followed by spatial neighbor candidate pooling ($M$-capped reservoirs) delivering multi-bounce indirect global illumination.
- **Two-Regime Voxel Edge-Stopping Wavelet Denoiser:**
  - Implemented [`shaders/denoiser_reconstruct.wgsl`](../../shaders/denoiser_reconstruct.wgsl) featuring exact voxel plane-distance edge-stopping weights.
  - Eradicated cross-corner light bleeding over $6.25\text{ cm}$ voxel boundaries while smoothing high-frequency indirect GI noise.
- **Halton Sub-Pixel DLAA / Temporal Super-Resolution:**
  - Implemented [`shaders/composite_dlaa.wgsl`](../../shaders/composite_dlaa.wgsl) with 3D analytic voxel motion vectors.
  - YCoCg bounding-box variance clipping, Beer-Lambert volumetric underwater fog, and ACES filmic tonemapping.
- **Strategic Implementation Plan:** [docs/plans/category_c_lighting_architecture_plan.md](category_c_lighting_architecture_plan.md)
- **Architectural Specification:** [docs/architecture/02_rendering_and_lighting.md](../architecture/02_rendering_and_lighting.md)

---

### Phase 5B: Benchmark Suite, Upscaler, Calibration & Visual Polish — COMPLETED
*Standardized performance benchmarking, spatial upscaling, dynamic hardware auto-tuning, and dithering.*

- **Sanctuary Benchmark Suite:**
  - Constructed deterministic benchmark scene ($160\text{ m} \times 64\text{ m} \times 160\text{ m}$) featuring colonnades, water basins, chrome mirrors, magma falls, and 6 point lights.
  - 9-waypoint Catmull-Rom spline camera flying through stressful geometric and illumination scenarios, generating automated performance reports and JSON logs (`[B]`).
- **Spatial Presentation Pipeline (Pass 5):**
  - Implemented [`shaders/blit.wgsl`](../../shaders/blit.wgsl) with 5-tap Catmull-Rom edge-adaptive upscaling and AMD FSR 1.0 RCAS (Robust Contrast Adaptive Sharpening).
  - Added pixel-perfect native-resolution HUD crosshair rendering.
- **Dynamic Hardware Auto-Calibration Engine:**
  - Live 40-frame GPU throughput measurement (`[C]`, `[L]`, or CLI `--calibrate`).
  - Quadratic fillrate scaling automatically selecting internal render scale and secondary ray budgets for stable 60 FPS or 30 FPS.
- **PCG 2D Decorrelated TPDF Dithering:**
  - Implemented temporal Triangular Probability Density Function (TPDF) dither system (`[K]`).
  - Permanently eliminated color banding in dark gradients, nocturnal skies, and specular reflections without pattern crawling.
- **Artifact Remediation:**
  - Eradicated cave daylight/sun leakage, resolved mirror seam artifacts via continuation rays, and added tunable ambient floor lighting.
- **Architectural Specification:** [docs/architecture/04_benchmarking_and_hud.md](../architecture/04_benchmarking_and_hud.md)

---

### Phase 6: Structural Island Collapse & 6-DOF Rigid Voxel Dynamics (`islands/`, `rigid_voxel/`, `primary_gbuffer.wgsl`) — COMPLETED
*Teardown-style physical world collapse, topological support detection, and rigid body dynamics.*

- **Targeted Perimeter-Seeded Island Detection:**
  - Implemented [`crates/arcadia-world/src/islands/perimeter.rs`](../../crates/arcadia-world/src/islands/perimeter.rs) running local 6-connected topological flood-fill seeded only along carve crater boundary shells.
  - Instant bedrock anchor short-circuiting ($y \le 4$, `MAT_BEDROCK`) resolving attached structures in $< 0.1\text{ ms}$.
- **6-Way Compressive Support Overhaul:**
  - Overhauled [`crates/arcadia-world/src/islands/bitwise.rs`](../../crates/arcadia-world/src/islands/bitwise.rs) and [`connectivity.rs`](../../crates/arcadia-world/src/islands/connectivity.rs): enabled 6-directional support propagation including downward $-Y$, removed artificial cantilever limits, and eradicated mid-air boundary false anchors.
  - Integrated into carving: severed structures detach into `DestructionEvent::detached_islands` while intact surrounding terrain remains completely stable.
- **6-DOF Rigid Voxel Dynamics:**
  - Implemented [`crates/arcadia-physics/src/rigid_voxel/`](../../crates/arcadia-physics/src/rigid_voxel/) with discrete center of mass and exact $3 \times 3$ moment of inertia tensor with $\frac{s^2}{6}$ cubic voxel parallel-axis correction.
  - 15-axis SAT OBB broadphase and Symplectic Euler integration.
  - Dynamic GPU instance packing (`GpuRigidBodyInstance`, 224 bytes) rendered directly in `primary_gbuffer.wgsl`.
  - In-flight dynamic body carving and splitting into multiple independent sub-bodies.
- **Kinetic Impact Shattering:**
  - High-velocity impacts ($> 18\text{ m/s}$) shatter small bodies ($< 300$ voxels) into physical GPU debris particles.
- **Architectural Specification:** [docs/architecture/03_physics_and_destruction.md](../architecture/03_physics_and_destruction.md)

---

### Phase 6B: Reverse-Z Floating-Point Depth Architecture (`camera.rs`, `lib.rs`, `gbuffer.rs`, `pipeline.rs`) — COMPLETED
*Next-generation floating-point depth buffer eliminating far-clip planes and depth precision loss.*

- **Infinite Reverse-Z Perspective Projection:**
  - Implemented `reverse_z_perspective_matrix` (`near = 1.0, far = 0.0`) in [`crates/arcadia-render/src/camera.rs`](../../crates/arcadia-render/src/camera.rs), mapping near plane ($0.1\text{ m}$) to $1.0$ and infinite distance to $0.0$.
  - Completely eliminated far-clip planes and floating-point depth compression at horizons.
- **Hardware Texture & Attachment Upgrades:**
  - Added full-resolution hardware `Depth32Float` (`VK_FORMAT_D32_SFLOAT`) texture `gbuf_depth_tex` and view to `ScreenResources`.
  - Configured pipeline depth testing with `CompareFunction::GreaterEqual` and clear value $0.0$.
- **Motion Vector & Debris Integrity:**
  - Verified analytic motion vector reprojection ($w_{\text{clip}} = -z_{\text{view}} > 0$) and debris splatting in `primary_gbuffer.wgsl` remain mathematically distortion-free.
- **Architectural Specification:** [docs/architecture/02_rendering_and_lighting.md](../architecture/02_rendering_and_lighting.md)

---

### Phase 7: In-Place Leaf Brick Mutation & Zero-Repack Sculpting (`chunk_tree/edit/`, `world_map.rs`) — COMPLETED
*Elimination of heap allocations and tree rebuilds during dynamic voxel modification.*

- **Zero-Repack Mutation Engine:**
  - Implemented [`crates/arcadia-core/src/chunk_tree/edit/`](../../crates/arcadia-core/src/chunk_tree/edit/) performing direct in-place mutation of `LeafBrick64` in `self.bricks` and leaf `Node64` bitmasks/LOD payloads in `self.nodes`.
  - Completely eliminated full `unpack_to_build` and BFS `pack_build_tree` allocations during gameplay.
- **Lazy Branch Expansion:**
  - Homogeneous uniform nodes uncollapse lazily on demand when edited; sibling slices allocate contiguously and re-index in single-cycle `POPCNT` rank order.
- **In-Place Sibling Compaction:**
  - When leaf nodes empty (`mask == 0`), pruning shifts remaining sibling nodes within their contiguous allocation (`nodes[old_base + rank .. old_base + new_count]`) and zeroes the trailing slot without vector allocations.
- **Fine-Grained GPU Staging (`ChunkEditReport`):**
  - Tracks dirty bounds `[min_dirty_node..=max_dirty_node]` and `[min_dirty_brick..=max_dirty_brick]`.
  - Enables `WorldMap::sync_dirty_chunk_partial` to stream targeted sub-slice GPU buffer writes (`queue.write_buffer`) in sub-milliseconds.
- **Compaction Pass:**
  - Implemented `ChunkTree64::compact()` to garbage-collect orphaned node/brick slices after heavy destruction and restore clean contiguous BFS level-order packing.
- **Architectural Specification:** [docs/architecture/01_storage_and_memory.md](../architecture/01_storage_and_memory.md)

---

### Phase 8: Concave Multi-Manifold Physics, Compound Colliders & Background World Baking — COMPLETED
*Industrial-grade rigid body stacking, multi-contact manifold clipping, and seamless world re-integration.*

- **Compound 3D Greedy Colliders (`rigid_voxel/collision/compound.rs`):**
  - Decomposes sparse voxels into disjoint solid cuboids (`VoxelSubBox`) with exposed-face bitmasking (`FACE_NEG_X` .. `FACE_POS_Z`).
  - Unexposed internal seams are culled during SAT tests, completely eliminating phantom edge snagging and micro-jitter.
- **Multi-Point Box3D Manifolds (`collision/manifold.rs`):**
  - Tandem BVH traversal with exposed-face 15-axis SAT narrowphase, Sutherland-Hodgman polygon clipping ($3\text{ mm}$ slop), optimal 4-point quad selection, and multi-manifold clustering (up to 4 directional manifolds per pair with $\cos \theta \ge 0.85$), enabling true concave interlocking (arches, bowls, crossed pillars) without air padding.
- **DAG Stacking Shock Passes & Polygon Sleep (`solver.rs`):**
  - Orders contact manifolds bottom-up from bedrock/terrain.
  - Executes bottom-up velocity shock passes and split-impulse position shock passes (treating lower support bodies as infinite mass during the pass) to eliminate stacking sponginess.
  - Union-Find clustering of resting contact graphs with 2D convex hull support polygon sleep states (`SupportSleepKind`).
- **Fluid Submersion in Rigid Bodies:**
  - Stratified volume-weighted water submersion calculation, progressive hydrodynamic drag ($1.5v + 0.8v^2$), rotational viscous damping, and buoyancy ($18 \frac{\rho_w}{\rho_b}$).
- **Dynamic GPU Pool Growth:**
  - Dynamic in-engine GPU pool growth with GPU-to-GPU copies (`copy_buffer_to_buffer`) without CPU roundtrips.
- **Strategic Implementation Plan:** [docs/plans/rigid_body_physics_optimization_plan.md](rigid_body_physics_optimization_plan.md)
- **Architectural Specification:** [docs/architecture/03_physics_and_destruction.md](../architecture/03_physics_and_destruction.md)

---

### Phase 8B: Asynchronous Resting Rigid Body Baking Pipeline (`app/state.rs`, `app/simulation.rs`) — COMPLETED
*Automated background voxelization and static world integration for settled physics bodies.*

- **Frustum Visibility Gating:**
  - Resting bodies ($t \ge \text{object\_bake\_timeout\_s}$, configurable 5..180s) evaluate camera frustum visibility (`is_visible_to_camera`) with a 15% peripheral guard margin and 150m cutoff distance.
  - Off-screen bodies (or bodies exceeding $3\times$ forced timeout) are selected for baking.
- **Multi-Threaded Background Rasterization:**
  - Dispatches resting bodies to `rayon::spawn`, rasterizing rotated voxels into integer world coordinates (`body.to_world_voxels()`) and transmitting via `mpsc::channel` (`BakeResult`).
- **Time-Sliced Commits & Zero Pop-In:**
  - Consumes at most 1 bake completion per frame into `WorldMap::set_voxels_global`.
  - Baking bodies continue rendering dynamically on GPU until the exact frame their static voxels commit, eliminating visual popping and frame latency spikes.
  - Baking bodies are protected from dynamic capacity eviction.
- **Architectural Specification:** [docs/architecture/03_physics_and_destruction.md](../architecture/03_physics_and_destruction.md)

---

### Phase 8C: Rolling Frame Pacing Tracker & HUD Overlay (`frame_graph.rs`, `[F3]`) — COMPLETED
*Real-time frame pacing, standard deviation jitter analysis, and HUD telemetry.*

- **Rolling Frame Time History:**
  - Tracks 180 frames (3 seconds at 60 FPS) in a rolling `VecDeque<f32>` buffer in [`crates/arcadia-app/src/frame_graph.rs`](../../crates/arcadia-app/src/frame_graph.rs).
- **Statistical Pacing Analysis:**
  - Evaluates current ms, average ms, min/max ms, standard deviation, 1% low FPS, and frametime consistency percentage.
- **Bottom-Right Egui HUD Overlay (`[F3]`):**
  - Displays color-coded FPS, jitter rating (`STABLE`, `MODERATE`, `JITTER`), 16.7ms and 33.3ms reference lines, dynamic histogram bars, and min/max/1% low stats.
- **Architectural Specification:** [docs/architecture/04_benchmarking_and_hud.md](../architecture/04_benchmarking_and_hud.md)

---

### Phase 8D: 2D Boundary Surface Shell Flood-Fill Optimization — COMPLETED
*$O(N^2)$ surface boundary traversal slashing crater turnaround time by $> 10\times$.*

- **Separable 26-Neighborhood Air Dilation (`bitwise.rs`):**
  - Evaluates Chebyshev distance 1 air coverage within each $4\times 4\times 4$ brick via 3 separable 1D bit shifts in $\approx 5\text{ ns}$ of pure ALU register ops.
  - Solves the digital concave crease problem so orthogonal floor/pillar junctions preserve 6-connected surface continuity.
- **Surface-Only Traversal (`perimeter.rs`):**
  - Replaces $O(N^3)$ solid interior volume expansion with $O(N^2)$ 2D outer shell flood-fill during crater perimeter island detection.
  - Skips 100% of subterranean solid rock in attached mountains and cliffs.
- **Deferred 3D Solid Interior Fill:**
  - Only when a candidate surface boundary shell closes without touching bedrock ($y \le 4$) or exceeding bounds (`!is_grounded`) does a 3D flood fill run inward from the shell to harvest internal rock for dynamic `RigidVoxelBody` instantiation.
- **Performance Impact:**
  - Slashes non-detaching terrain carve turnaround by $>10\times$ (visiting $<600$ surface voxels instead of thousands of subterranean solid rock voxels), eliminating destruction frame hitches.
- **Architectural Specification:** [docs/architecture/03_physics_and_destruction.md](../architecture/03_physics_and_destruction.md)

---

### Phase 9: Endless World Streaming & Toroidal Clipmaps — FUTURE ROADMAP
*Next architectural horizon for massive, unbounded voxel exploration.*

- **Asynchronous Disk & Procedural Streaming:**
  - Background worker thread pool streaming chunks to and from compressed binary disk storage.
  - Procedural terrain generation asynchronously pre-generating outer rings ahead of the player camera.
- **Toroidal 3D Clipmap Scrolling:**
  - Modulo chunk grid wrapping allowing unbounded coordinates without translating or reallocating the GPU root descriptor buffer.
- **Hybrid Raster/Voxel Compositing:**
  - Interleaved hardware rasterization passes writing to the shared Reverse-Z `Depth32Float` buffer.
  - Seamless compositing of traditional polygonal meshes (characters, weapons, animated skeletal rigs) inside the compute raymarched voxel world.
- **Roadmap & Action Items:** [docs/plans/ROADMAP_AND_TODO.md](ROADMAP_AND_TODO.md)

---

## 3. Cross-Document Navigation

- **Master Executive Architecture:** [ARCHITECTURE.md](../../ARCHITECTURE.md)
- **Keybindings & Controls Quick Reference:** [docs/keybindings.md](../keybindings.md)
- **Subsystem Architecture Specifications:**
  - [Storage, Memory Layout & Tree Mutation Architecture](../architecture/01_storage_and_memory.md)
  - [Rendering, Shaders & Lighting Pipeline Architecture](../architecture/02_rendering_and_lighting.md)
  - [Physics, Structural Islands & Rigid Body Dynamics Architecture](../architecture/03_physics_and_destruction.md)
  - [Benchmarking, Quality Presets & HUD Profiler Architecture](../architecture/04_benchmarking_and_hud.md)
- **Strategic Implementation Plans:**
  - [Comprehensive Project Roadmap & Backlog](ROADMAP_AND_TODO.md)
  - [Category C Lighting Architecture Plan](category_c_lighting_architecture_plan.md)
  - [Rigid Body Physics Optimization Plan](rigid_body_physics_optimization_plan.md)
- **Historical Session Execution Reports:**
  - [docs/reports/archive/](../reports/archive/) (Sessions 1 through 7)
