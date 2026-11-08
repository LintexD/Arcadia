# Arcadia Voxel Engine — Master Architecture & Agent Guide

## 1. Project Overview & Target Specifications
- **Voxel Scale:** $(1/16\text{ m})^3 = 0.0625\text{ m} = 6.25\text{ cm}$ ($4,096\text{ voxels per m}^3$)
- **Target Hardware:** NVIDIA RTX 3080 (10 GB VRAM) / Discrete & Integrated GPU support (Vulkan 1.3)
- **Target Display & Framerate:** $1920 \times 1080$ (1080p) @ **60+ FPS** (internal GPU frame budget $\le 8\text{ ms}$, >120 FPS capable)
- **Language & Graphics API:** **Rust (2024 Edition)** + **`wgpu` (23.0.0, Vulkan 1.3 Backend, WGSL Compute Shaders)**
- **Visual Direction:** **John Lin Fantasy Nature & Ruins** — lush terraced hills, mossy stone ruins and archways, glowing crystals & lava, soft directional sunlight, volumetric fog, and multi-bounce global illumination.
- **Core Capabilities:**
  1. **Chunked Sparse 64-Tree (CS64)** world representation with single-cycle `POPCNT` (`count_ones` / `countOneBits`) child indexing, homogeneous solid-node collapsing, and BFS level-order GPU node packing.
  2. **In-Place Leaf Brick Mutation & Zero-Repack Sculpting:** Direct in-place mutation of `LeafBrick64` in `self.bricks` and leaf `Node64` bitmasks in `self.nodes`, with lazy branch expansion, in-place sibling compaction, and fine-grained `ChunkEditReport` staging updates.
  3. **Hierarchical GPU Compute DDA Ray Marching** with Ray-Cone LOD termination and coarse tile depth pre-pass.
  4. **Bitwise `u64` AABB Collision & Kinematic Character Controller** (walking, wall sliding, 3-voxel auto-step-up, gravity, jump, water buoyancy/swimming, fly-mode toggle) with zero triangle mesh generation.
  5. **Real-Time World Destruction** (sphere/beam/explosion carving, scorch/ember rims, crater rim crumb cleanup, incremental sub-slice GPU staging uploads) + **16k GPU-Simulated Voxel Debris Particles**.
  6. **NVIDIA RTX/DLSS-Inspired Lighting & Reconstruction Pipeline:**
     - **ReSTIR GI** (Temporal + Spatial reservoir resampling for multi-bounce indirect lighting & emissive crystals/lava).
     - **Exact Voxel Plane-Distance Ray Reconstruction** (edge-stopping spatial-temporal wavelet filter with zero edge bleeding across $6.25\text{ cm}$ voxel corners).
     - **Halton Sub-Pixel Temporal Super-Resolution / DLAA** (analytic 3D voxel motion vectors + YCoCg variance clipping for 4K-grade anti-aliasing on a 1080p monitor).
     - **Hardware Reverse-Z Depth Buffer** (`Depth32Float`, infinite perspective projection mapping $0.1\text{m} \to 1.0$, $\infty \to 0.0$).
     - **FSR 1.0 RCAS Sharpening & 5-Tap Catmull-Rom Upscaler** with native-resolution HUD crosshair.
     - **High-Precision Decorrelated TPDF Dithering System** (PCG 2D temporal frame rotation, channel decorrelation).
  7. **Structural Island Detection (*Teardown*-Style Collapse):**
     - Targeted perimeter-seeded 6-connected flood-fill around carve craters with instant bedrock anchor short-circuiting ($y \le 4$, `MAT_BEDROCK`).
     - Bitwise macro-brick graph traversal with 6-way compressive support propagation (+Y, -Y, $\pm X$, $\pm Z$).
  8. **6-DOF Dynamic Rigid Voxel Physics:**
     - Compound 3D greedy box decomposition with exposed-face bitmasking and flat local BVH.
     - 15-axis SAT broadphase & narrowphase with internal seam culling.
     - Multi-point Box3D-style contact manifolds with Sutherland-Hodgman polygon clipping and up to 4 directional manifolds per pair ($\cos \theta \ge 0.85$) for concave interlocking (arches, bowls, fallen pillars).
     - Discrete $3\times3$ moment of inertia tensor with $\frac{s^2}{6}$ cubic voxel parallel-axis correction.
     - Symplectic Euler solver with DAG support leveling, bottom-up velocity/position shock passes, split-impulse position correction, and 2D convex hull support polygon sleep states.
     - High-energy kinetic impact shattering (> 18 m/s).
  9. **Asynchronous Resting Rigid Body Baking Pipeline:**
     - Visibility-gated resting body detection (camera frustum culling with 15% peripheral guard margin).
     - Multi-threaded background voxelization via `rayon::spawn` and `mpsc` channels.
     - Time-sliced commit (max 1 body/frame) with zero pop-in GPU rendering until static world integration.
  10. **Standardized Profiler & Real-Time Tooling:**
      - Deterministic pure render benchmark suite with Catmull-Rom spline camera in Sanctuary scene.
      - Immediate-mode `egui` debug UI with 6 collapsible tuning sections and dynamic hardware auto-calibration.
      - Rolling frame pacing HUD graph (`[F3]`) tracking min/max/avg, 1% lows, std dev, and consistency purity.

---

## 2. Workspace Layout & Crate Responsibilities

```text
arcadia/
├── Cargo.toml                        # Workspace root & shared dependencies
├── ARCHITECTURE.md                   # Master executive entry point & system index
├── docs/
│   ├── keybindings.md                # Dedicated controls and keybindings quick reference
│   ├── architecture/                 # Modular subsystem technical specifications
│   │   ├── 01_storage_and_memory.md  # CS64 tree, layouts, POPCNT, in-place mutation, Morton
│   │   ├── 02_rendering_and_lighting.md # 7 pipelines, Cone LOD, ReSTIR GI, DLAA, RCAS, TPDF
│   │   ├── 03_physics_and_destruction.md # Islands, compound colliders, Box3D manifolds, solver, baking
│   │   └── 04_benchmarking_and_hud.md # Sanctuary scene, quality presets, calibration, frame HUD
│   ├── plans/                        # Strategic architecture plans & checklists
│   │   ├── category_c_lighting_architecture_plan.md
│   │   ├── implementation_history.md # Comprehensive phased history & architectural milestones (Phases 1-9)
│   │   ├── rigid_body_physics_optimization_plan.md
│   │   └── ROADMAP_AND_TODO.md
│   └── reports/archive/              # Historical session execution reports (1..7)
├── knowledgebase/                    # Theoretical research library (00..07)
├── crates/
│   ├── arcadia-core/                 # Pure data structures, bitwise math, material palette, CPU raycast & in-place edits
│   │   ├── src/
│   │   │   ├── lib.rs                # Crate root exports (ChunkTree64, Node64, LeafBrick64, coords, palette)
│   │   │   ├── coord.rs              # VoxelCoord, ChunkCoord, 6-bit child index packing ((z<<4) | (y<<2) | x)
│   │   │   ├── material.rs           # 256-entry PBR + Emissive Material Palette (32B Pod, 16B aligned)
│   │   │   ├── node64.rs             # Node64 (16B), LeafBrick64 (64B), POPCNT child offset math, uniform flags
│   │   │   ├── slab.rs               # Free-list SlabAllocator<T> for memory pooling
│   │   │   ├── mtf_cache.rs          # Fixed-capacity Move-To-Front (MTF) LRU cache with zero heap allocations
│   │   │   ├── raycast_cpu.rs        # Reference CPU 64-tree DDA raycaster (for physics, tools & verification)
│   │   │   ├── chunk_tree.rs         # ChunkTree64 coordinator module & scale constants
│   │   │   └── chunk_tree/
│   │   │       ├── tree.rs           # ChunkTree64 struct, POPCNT voxel queries, bitwise AABB intersection
│   │   │       ├── builder.rs        # BuildNode intermediate tree, procedural generators, BFS level-order packing
│   │   │       ├── edit.rs           # Dynamic modification entry point, ChunkEditReport, BrickVoxelListMap
│   │   │       └── edit/             # Zero-repack in-place mutation engine (carve, clear, place, set)
│   │   │           ├── carve.rs      # Spherical carving, scorch/ember rim hashing, destroyed voxel tracking
│   │   │           ├── clear.rs      # Local voxel and AABB batch clearing to air
│   │   │           ├── place.rs      # Spherical and cuboid voxel brush placement
│   │   │           └── set.rs        # In-place leaf mutation engine, lazy branch uncollapse, sibling compaction
│   │   └── tests/
│   │       └── edit_set.rs           # Integration tests for in-place leaf mutation and lazy uncollapse
│   ├── arcadia-world/                # World grid, procedural terrain, destruction & island detection
│   │   ├── src/
│   │   │   ├── lib.rs                # Crate root exports (WorldMap, generators, destruction, islands, debris)
│   │   │   ├── world_map.rs          # 3D chunk grid, Morton (Z-order) GPU packing, partial dirty sync, registration
│   │   │   ├── generator.rs          # Rayon parallel procedural "Fantasy Nature & Ruins" generator (6.25cm scale)
│   │   │   ├── benchmark_scene.rs    # Dedicated benchmark sanctuary (Colonnade, Water, Mirrors, Spline)
│   │   │   ├── destruction.rs        # Real-time sphere carving/placement, debris emission, rim crumb cleanup
│   │   │   ├── islands.rs            # Structural island detection coordinator & public exports
│   │   │   └── islands/
│   │   │       ├── bitwise.rs        # Intra-brick 64-bit dilation, boundary face masks, ALU flood fill
│   │   │       ├── query.rs          # VoxelQuery trait, thread-safe WorldRegionSnapshot, bedrock anchor probing
│   │   │       ├── perimeter.rs      # Targeted crater perimeter-seeded island detection & short-circuiting
│   │   │       └── connectivity.rs   # Macro-brick graph traversal, 6-way compressive support propagation
│   │   └── tests/
│   │       └── island_collapse.rs    # Integration test suite for bitwise island detection and collapse edge cases
│   ├── arcadia-physics/              # Bitwise AABB collision, character controller, 6-DOF dynamic rigid bodies
│   │   ├── src/
│   │   │   ├── lib.rs                # Crate root exports (CharacterController, RigidVoxelBody, compound colliders)
│   │   │   ├── character.rs          # First-person kinematic controller (1..3 step-up, MTF cache, swimming)
│   │   │   ├── rigid_voxel.rs        # Rigid voxel body coordinator & re-exports
│   │   │   └── rigid_voxel/          # Detached OBB voxel bodies (6-DOF solver, SAT broadphase, inertia tensor)
│   │   │       ├── types.rs          # GpuRigidBodyInstance (224B), RigidStepOutcome, RigidVoxelBody, ContactPoint3D
│   │   │       ├── transform.rs      # Local/world matrices, inverse inertia tensor, camera frustum culling
│   │   │       ├── construction.rs   # CoM, discrete inertia tensor with s^2/6 correction, CompoundCollider
│   │   │       ├── collision.rs      # Collision coordinator module & submodule declarations
│   │   │       ├── collision/        # Collision detection & contact generation submodules
│   │   │       │   ├── broadphase.rs # Obb struct, 15-axis SAT tests, 3-tier broadphase (sphere -> AABB -> SAT)
│   │   │       │   ├── compound.rs   # 3D greedy cuboid merge, exposed-face bitmasks (seam culling), local BVH
│   │   │       │   ├── manifold.rs   # Box3D SAT manifold, Sutherland-Hodgman clipping, concave multi-manifolds
│   │   │       │   ├── terrain.rs    # Static terrain & bedrock contacts, neighbor brick caching, water submersion
│   │   │       │   └── voxel.rs      # Pairwise collision entry points
│   │   │       ├── solver/           # Symplectic Euler, DAG support levels, shock pass, split-impulse, polygon sleep
│   │   │       │   ├── mod.rs        # Solver coordinator, substep loop & restitution gating
│   │   │       │   ├── position.rs   # Split-impulse position projection & shock propagation
│   │   │       │   ├── sleep.rs      # 2D convex hull support polygon sleep states & waking
│   │   │       │   └── velocity.rs   # Velocity Gauss-Seidel solver, normal/tangent impulses & friction
│   │   │       └── voxel_ops.rs      # CS64 raycasting, carving, splitting, placement, small island debris conversion
│   │   └── tests/                    # Integration test suite (9 targets + shared fixtures)
│   │       ├── common/
│   │       │   └── mod.rs            # Test harness, world scaffolding, and rigid body assertions
│   │       ├── collision.rs          # Pairwise collision detection & feature contacts
│   │       ├── compound.rs           # Greedy compound collider decomposition & exposed-face bitmasks
│   │       ├── dynamics.rs           # Freefall, linear/angular momentum, and velocity integration
│   │       ├── inertia.rs            # CoM & 3x3 inertia tensor with s^2/6 parallel-axis correction
│   │       ├── solver.rs             # Symplectic Euler, contact impulse resolution & friction
│   │       ├── stacking.rs           # Multi-body stacking stability & resting state verification
│   │       ├── terrain.rs            # Static terrain & bedrock contact generation
│   │       ├── visibility.rs         # Frustum culling & peripheral guard margin for resting body baking
│   │       └── voxel_ops.rs          # Voxel carving, splitting, placement & debris conversion
│   ├── arcadia-render/               # Modular wgpu Vulkan compute pipeline, GPU pools, ReSTIR GI & DLAA
│   │   ├── src/
│   │   │   ├── lib.rs                # Crate root exports and pass orchestration overview
│   │   │   ├── context.rs            # GpuContext (Vulkan backend, discrete/iGPU ranking, device, queue, surface)
│   │   │   ├── camera.rs             # CameraUniforms (400B), Halton sub-pixel jitter, infinite Reverse-Z projection
│   │   │   ├── quality.rs            # RaymarchQualitySettings, quality presets, dynamic framerate calibration
│   │   │   ├── gpu_pools.rs          # Group0Resources: Chunk Roots, NodePool, BrickPool, Debris, RigidBody instances
│   │   │   ├── gbuffer.rs            # ScreenResources: Ping-pong G-Buffer, Reservoirs, Lit Color, History, Depth32Float
│   │   │   ├── pipeline.rs           # 7 Pipelines (6 compute + 1 render), PassLayouts, Reverse-Z depth states
│   │   │   ├── renderer.rs           # Renderer lifecycle, uniform updates, dynamic pool growth with GPU copies
│   │   │   └── sharc.rs              # SHARC voxel-surfel radiance cache: 4.5 MB pools, layouts, update/resolve pipelines
│   │   └── tests/
│   │       ├── shader_validation.rs  # Naga WGSL validation & SPIR-V bytecode OpStore integrity tests
│   │       └── sharc_gpu.rs          # Headless GPU roundtrip, multi-thread CAS, EMA blend & eviction tests
│   └── arcadia-app/                  # Application crate (executable binary & integration test target)
│       ├── src/
│       │   ├── main.rs               # Application entry point & winit 0.30 event loop bootstrap
│       │   ├── lib.rs                # Library exports for integration tests & benchmarking
│       │   ├── app.rs                # Application coordinator module & public re-exports (App, CalibrationTask, BakeResult)
│       │   ├── app/                  # Application architecture submodules
│       │   │   ├── state.rs          # App state struct, scene bootstrap, point lights, rigid body channels
│       │   │   ├── simulation.rs     # Per-frame simulation, sun movement, camera input, rigid body stepping & baking
│       │   │   ├── render.rs         # GPU frame dispatch, egui overlay rendering, surface error handling
│       │   │   └── event.rs          # Winit 0.30 ApplicationHandler event dispatch & lifecycle callbacks
│       │   ├── cli.rs                # Command-line argument parsing, flags, presets, and launch configuration
│       │   ├── input.rs              # Keyboard/mouse input state, movement dispatch, hotkeys, palette mapping
│       │   ├── sculpting.rs          # Unified scene raycast, explosive carving, voxel placement, dynamic body carving
│       │   ├── frame_graph.rs        # Rolling 180-frame tracker, jitter/stability statistics, and egui bottom-right HUD
│       │   ├── gui.rs                # Immediate-mode egui debug interface with 6 collapsible tuning sections
│       │   └── benchmark.rs          # Deterministic trajectory runner, statistical analysis, score calculation
│       └── tests/
│           └── sculpting.rs          # Sculpting, voxel placement, and body raycast integration tests
│
└── shaders/
    ├── common/
    │   ├── types.wgsl                # CameraUniforms, Node64, Material, Reservoir, RigidBodyInstance, Debris
    │   ├── traversal64.wgsl          # Hierarchical Chunk + 4-Level 64-Tree DDA & Cone LOD
    │   ├── sharc.wgsl                # SHARC constants, face-separated surfel key hashing, f16 slot packing
    │   └── sharc_query.wgsl          # Read-only 4-way bucket lookup (query_sharc) for Pass 2 / update
    ├── primary_gbuffer.wgsl          # Pass 1: GPU debris compute + primary ray march, analytic 3D motion vectors
    ├── sharc_update.wgsl             # SHARC update: view-seeded tile paths deposit L_o(x_1) into the surfel hash
    ├── sharc_resolve.wgsl            # SHARC resolve: fast-start EMA + stale-age eviction (0 rays)
    ├── lighting_restir_gi.wgsl       # Pass 2: Blue-noise soft sun shadows, multi-bounce diffuse GI + ReSTIR reuse
    ├── denoiser_reconstruct.wgsl     # Pass 3: Voxel plane-distance aware À-trous wavelet filter & PBR composition
    ├── composite_dlaa.wgsl           # Pass 4: Halton DLAA/TSR temporal resolve, volumetric fog, ACES tonemap, TPDF
    └── blit.wgsl                     # Pass 5: Edge-adaptive Catmull-Rom 5-tap upscaler, FSR 1.0 RCAS, HUD crosshair
```

---

## 3. Architectural Subsystems & Technical Specifications

Detailed architectural formulas, mathematical derivations, memory layouts, and algorithm specifications are modularized into four comprehensive documents:

### 1. [Storage, Memory Layout & Tree Mutation Architecture](docs/architecture/01_storage_and_memory.md)
- **Scale & Coordinate Math:** 4-level 64-tree hierarchy ($256^3 \to 64^3 \to 16^3 \to 4^3$ Leaf Bricks), $6.25\text{ cm}$ voxel scale, 6-bit child bit index packing `(z << 4) | (y << 2) | x`.
- **Node Binary Layouts:** 16-byte host/device-shared `Node64` struct, 64-byte `LeafBrick64` slot, single-cycle `POPCNT` child address calculations in Rust and WGSL.
- **In-Place Zero-Repack Mutation Engine:** In-place leaf mutation in `self.bricks` and `self.nodes`, lazy branch uncollapsing, contiguous sibling compaction, and `ChunkEditReport` localized GPU uploads.
- **Node & Chunk Ordering:** Breadth-first (BFS) level-order intra-chunk packing, 3D Morton space-filling curves, and dynamic body tree registration.

### 2. [Rendering, Shaders & Lighting Pipeline Architecture](docs/architecture/02_rendering_and_lighting.md)
- **Pass Orchestration:** 7 compiled pipelines (6 compute + 1 render) sequencing debris simulation, primary G-buffer raymarching, debris splatting, ReSTIR GI, edge-stopping wavelet filtering, DLAA/TSR composite, and Catmull-Rom/RCAS presentation.
- **Ray-Cone LOD & Zero-Brick GI:** Cyril Crassin *GigaVoxels* cone footprint termination and 100% bypass of 64-byte `LeafBrick64` VRAM loads during diffuse indirect bounces.
- **Hemisphere Sampling & Seams:** 2D $R_2$ / PCG decorrelated hemisphere sampler, mirror seam continuation tracing, DDA normal tie-breaking, and enclosed cave occlusion dynamics.
- **Upscaling & Presentation:** FSR 1.0 RCAS contrast-adaptive sharpening, Catmull-Rom 5-tap upsampling, native-resolution HUD crosshairs, and high-precision decorrelated TPDF dithering.

### 3. [Physics, Structural Islands & Rigid Body Dynamics Architecture](docs/architecture/03_physics_and_destruction.md)
- **Structural Island Detection:** Targeted crater perimeter-seeded topological flood-fill, 2D boundary surface traversal ($O(N^2)$ vs. $O(N^3)$), instant bedrock anchor short-circuiting, and 6-way compressive support propagation.
- **6-DOF Rigid Voxel Dynamics:** Discrete center of mass, $3 \times 3$ moment of inertia tensor with $\frac{s^2}{6}$ cubic voxel correction, compound 3D greedy colliders with internal seam culling, and 15-axis SAT.
- **Box3D Manifolds & Solver:** Sutherland-Hodgman polygon clipping, up to 4 directional manifolds for concave interlocking, Symplectic Euler integration, DAG support leveling, shock passes, split-impulse solver, and 2D support polygon sleeping.
- **Asynchronous Resting Body Baking:** Frustum visibility gating (15% peripheral guard margin), background multi-threaded rasterization (`rayon::spawn`), and time-sliced commits with zero GPU pop-in.

### 4. [Benchmarking, Quality Presets & HUD Profiler Architecture](docs/architecture/04_benchmarking_and_hud.md)
- **Pure Render Benchmark Suite:** $160\text{m} \times 64\text{m} \times 160\text{m}$ Sanctuary demo scene, 9 curated spline waypoints, dynamic diurnal lighting cycle, and composite score calculation.
- **Interactive Quality Tuning:** `RaymarchQualitySettings` uniform contract, 6 collapsible tuning sections in immediate-mode `egui` debug UI (`[F1]`), and curated presets.
- **Auto-Calibration Engine:** 40-frame GPU throughput measurement, quadratic fillrate scaling, and calibrated 30/60 FPS switching (`[C]`, `[L]`).
- **Rolling Frame Pacing HUD:** 180-frame ring buffer, jitter classification, and bottom-right overlay graph (`[F3]`).

---

## 4. Keybindings & Input Reference

For the comprehensive, categorized controls guide and operational tips, see **[docs/keybindings.md](docs/keybindings.md)**.

### Essential Shortcuts
- **`[F1]`, `[Tab]`, ``[`]``** — Toggle immediate-mode `DebugUi` overlay & release/lock mouse cursor.
- **`[F3]`** — Toggle rolling Frame Time Graph HUD overlay (180-frame pacing, jitter, 1% lows).
- **`[Double-Space]`** (< 0.35s) — Toggle between Kinematic Walk Mode and Free Fly Mode (`[Ctrl]` to sprint).
- **`[Left Click]` / `[Right Click]`** — Explosive spherical voxel carve / Spherical voxel brush placement (`[Mouse Wheel]` scales radius).
- **`[1] - [8]`** — Select active material from palette (Crystals, Lava, Slate, Moss, Point Light, Mirror).

---

## 5. Phased Implementation History & Milestones

For the comprehensive architectural implementation record and technical notes across all development phases, see **[docs/plans/implementation_history.md](docs/plans/implementation_history.md)**.

| Phase | Subsystem / Focus | Key Deliverables | Status |
|---|---|---|---|
| **Phase 1** | Core 64-Tree & CPU Reference | `arcadia-core`: CS64 layout, POPCNT indexing, homogeneous collapsing, CPU DDA | **Completed** |
| **Phase 2** | GPU Compute Ray Marcher & Fantasy World | `arcadia-render`, `arcadia-world`: 3D Morton grid, Ray-Cone LOD, Sanctuary world | **Completed** |
| **Phase 3** | Walkable World & Bitwise Collisions | `arcadia-physics`: AABB collision, kinematic controller, 3-voxel step-up, swimming | **Completed** |
| **Phase 4** | Real-Time Destruction & GPU Voxel Debris | `arcadia-world`, `shaders/`: Spherical carving/placing, 16k GPU debris particles | **Completed** |
| **Phase 5** | ReSTIR GI, Reconstruction & DLAA | `shaders/`: Spatio-temporal ReSTIR, wavelet denoiser, DLAA/TSR, Reverse-Z depth | **Completed** |
| **Phase 5B** | Sanctuary Benchmark & Visual Polish | `arcadia-app`: Catmull-Rom spline benchmark, RCAS upscaler, 2D PCG TPDF dither | **Completed** |
| **Phase 6** | Island Collapse & 6-DOF Rigid Bodies | `arcadia-world`, `arcadia-physics`: Perimeter flood-fill, SAT broadphase, OBB dynamics | **Completed** |
| **Phase 6B** | Hardware Reverse-Z Infinite Depth | `arcadia-render`: `Depth32Float`, infinite perspective, sub-pixel motion vectors | **Completed** |
| **Phase 7** | In-Place Leaf Mutation & Zero-Repack | `arcadia-core`: In-place leaf mutation, lazy uncollapse, sibling compaction, partial staging | **Completed** |
| **Phase 8** | Compound Colliders & Resting Body Baking | `arcadia-physics`, `arcadia-app`: Box3D manifolds, shock solver, async body baking, HUD | **Completed** |
| **Phase 9** | World Streaming & Toroidal Clipmaps | Multi-chunk disk streaming, toroidal 3D clipmaps, hybrid voxel/raster composition | **Future** |

---

## 6. Implementation Plans, Historical Reports & Knowledge Base

- **Strategic Implementation Plans (`docs/plans/`):**
  - [Voxel Data, Dynamic Attributes & Sidecar Architecture Plan](docs/plans/voxel_data_and_attribute_architecture_plan.md) — Blueprint for >256 materials, per-voxel color, sparse sidecar simulation (water/fire), and conversion traits.
  - [Implementation History & Architectural Milestones](docs/plans/implementation_history.md) — Comprehensive technical record and deliverables for Phases 1 through 9.
  - [Lighting Architecture Plan (Category C)](docs/plans/category_c_lighting_architecture_plan.md) — Comprehensive plan for ReSTIR GI, soft shadows, wavelet denoising, and temporal upscaling.
  - [Rigid Body Physics Optimization Plan](docs/plans/rigid_body_physics_optimization_plan.md) — Multi-manifold collision, compound colliders, and sleep stability roadmap.
  - [Roadmap & Master Checklist](docs/plans/ROADMAP_AND_TODO.md) — Long-term technical milestones and architectural backlog.
- **Session Reports & Historical Archive (`docs/reports/archive/`):**
  - Historical development execution reports: [Session 1](docs/reports/archive/session_report_1.md), [Session 2](docs/reports/archive/session_report_2.md), [Session 3](docs/reports/archive/session_report_3.md), [Session 4](docs/reports/archive/session_report_4.md), [Session 5](docs/reports/archive/session_report_5.md), [Session 6](docs/reports/archive/session_report_6.md), [Session 7](docs/reports/archive/session_report_7.md).
- **Master Research Knowledge Base (`knowledgebase/`):**
  - [Knowledge Base Index](knowledgebase/README.md) & [Master Research Bibliography](knowledgebase/00_master_research_bibliography.md)
  - Subsystems: [01 Storage](knowledgebase/01_storage_and_data_structures/), [02 Rendering](knowledgebase/02_rendering_techniques/), [03 Simulation & Physics](knowledgebase/03_simulation_and_physics/), [04 Lighting](knowledgebase/04_lighting_and_illumination/), [05 Case Studies](knowledgebase/05_realtime_engine_case_studies/), [06 Mobs & Destruction](knowledgebase/06_mobs_pathfinding_animation_and_destruction/), [07 Arcadia Synthesis](knowledgebase/07_ideas_and_architectural_synthesis_for_arcadia/).
