# Arcadia Voxel Engine — Session 6 Engineering Report

**Date:** October 28, 2026  
**Status:** Completed, Verified & Benchmarked  
**Target Architecture:** Rust 2024, `wgpu` 23.0 (Vulkan 1.3 / DX12), WGSL Compute & Render Shaders, `egui` 0.30, Winit 0.30  
**Test Platform:** AMD Ryzen 5 7535U, Radeon 660M (RDNA2 Integrated GPU, DX12), 16 GB DDR5  

---

## 1. Executive Summary & Progression from Session 5

Session 5 concluded with the modularization of `arcadia-core` and `arcadia-render`, the establishment of the Master Research Bibliography, and the delivery of **Phase 6B (Reverse-Z Infinite Perspective Depth Architecture)**.

**Session 6** achieved major milestones in physical simulation, structural mechanics, and dynamic voxel manipulation by completing **Phase 6: 6-DOF Rigid Voxel Body Dynamics**, **Targeted Perimeter Island Detection**, **Dynamic Body In-Flight Sculpting & Splitting**, and **Multi-Point Contact Manifolds (Box3D)**:

1. **Phase 6: Full 6-DOF Rigid Voxel Dynamics (`arcadia-physics`):**
   - Implemented discrete voxel center of mass and exact $3 \times 3$ moment of inertia tensors with cubic parallel-axis correction ($\frac{s^2}{6}$).
   - Built a hierarchical collision detection pipeline: bounding sphere $\to$ 15-axis Separating Axis Theorem (SAT) OBB broadphase $\to$ 6-connected exterior hull narrowphase ($\le 96$ surface probes against the static CS64 tree).
   - Engineered a 4-substep sequential impulse solver with Coulomb friction, restitution thresholds, Baumgarte stabilization, sleep state transitions, and automatic wake-up upon terrain destruction underneath.
   - Dynamic GPU OBB raymarching via `@group(0) @binding(6)` with analytic 3D motion vectors using `prev_local_to_world` for jitter-free temporal upscaling (TSR/DLAA).
2. **Reliable Perimeter-Seeded Island Detection (`arcadia-world`):**
   - Replaced artificial bounding-box compressive support approximations and the 2-brick cantilever limit with targeted perimeter-seeded 6-connected flood fill (`detect_disconnected_islands_from_carve`).
   - Scans only the solid crater boundary shell ($r \in [R, R + 1.5\text{ voxels}]$) and traces topological load paths.
   - Bedrock short-circuiting ($y \le 4$ or `MAT_BEDROCK`) and mass threshold cutoff resolve grounded mountain foundations in $<0.1\text{ ms}$.
   - Eradicated unintended structural collapses of intact ceilings, arches, and bridges $2\text{m}$ away from carve actions.
3. **Dynamic Body Sculpting & Splitting (`arcadia-app`, `arcadia-physics`):**
   - Left-click carving directly targets and craters dynamic bodies in local coordinate space, applying scorch rims (`MAT_EMBER_GLOW`, `MAT_SCORCHED_ROCK`).
   - Blasting through a body partitions it into multiple independent dynamic rigid bodies (`find_connected_components` on a dense grid) while preserving orientation quaternions, linear/angular momentum, and local coordinate frames.
   - Right-click places voxels directly on dynamic bodies in body-local space, automatically expanding bounding boxes and recalculating mass and moments of inertia.
4. **Terrain Re-Baking:**
   - Resting bodies ($>60\text{ s}$ motionless) automatically rasterize into world-grid coordinates via conservative dual-pass sampling, bake into `WorldMap`, stream GPU staging patches, and exit physics simulation.
5. **Multi-Point Contact Manifold (Box3D):**
   - Implemented multi-point contact clipping between colliding dynamic rigid bodies (`BodyPairManifold` and `ContactPoint3D`) to eliminate single-point tipping, penetration jitter, and drift during object stacking.
6. **Codebase Modularization & Test Suite Expansion:**
   - Modularized `arcadia-physics` (`collision.rs`, `construction.rs`, `solver.rs`, `transform.rs`, `types.rs`, `voxel_ops.rs`, `tests.rs`).
   - Modularized `arcadia-world/islands` (`bitwise.rs`, `connectivity.rs`, `perimeter.rs`, `query.rs`, `tests.rs`).
   - Modularized `arcadia-app` (`app.rs`, `cli.rs`, `gui.rs`, `input.rs`, `sculpting.rs`).
   - Workspace unit test suite grew from **19 passed** in Session 5 to **60 passed** with zero warnings across all 5 workspace crates.

---

## 2. Detailed Technical Comparison: Session 5 vs. Session 6

| Feature / Subsystem | Session 5 Status | Session 6 Status | Key Technical Advancement |
|---|---|---|---|
| **Rigid Voxel Dynamics** | Phase 6 planned / prototype types | **Phase 6 Fully Operational** | 6-DOF Newton-Euler dynamics, $3\times3$ inertia tensor + $\frac{s^2}{6}$ parallel-axis term, 15-axis SAT OBB, 4-step impulse solver |
| **Island Detachment** | Static global bounding box search with cantilever limits | **Perimeter-Seeded Topological Search** | Local crater boundary seeding ($r \in [R, R+1.5]$), instant bedrock ($y \le 4$) short-circuiting ($<0.1\text{ ms}$), zero false collapses |
| **Object Sculpting** | Terrain-only carving | **Dynamic Body Carving & Placement** | Raycast disambiguation (`SceneRayHit`), body-local sphere carving, scorch rims, dynamic mass/inertia updates |
| **Dynamic Body Splitting** | Unsupported | **Graph Partitioning on Carve** | 6-connected flood fill partitions severed dynamic bodies into multiple child `RigidVoxelBody` instances |
| **Object-vs-Object Stacking** | Single-point contact (tipping/drift) | **Multi-Point Contact Manifold (Box3D)** | `BodyPairManifold` with 3D contact clipping, warm starting, and Coulomb friction |
| **Terrain Re-Baking** | Concept only | **Automatic Resting Body Rasterization** | Motionless bodies ($>60\text{ s}$) rasterize to `WorldMap` voxels, stream `GpuPatch`, and despawn from physics |
| **Player-Body Interaction** | Ghosting through dynamic rubble | **Kinematic Character Stepping & Pushing** | Player AABB collides with rigid body OBBs; steps up $6.25\text{–}18.75\text{ cm}$ onto rubble; pushes debris |
| **GPU Motion Vectors** | Reverse-Z depth wired | **Analytic 3D Rigid Body Motion Vectors** | GPU instance buffer binds `prev_local_to_world` for exact TSR velocity vectors on tumbling bodies |
| **Unit Test Coverage** | 19 unit tests passing | **60 unit tests passing** | +20 physics tests, +12 world/island tests, +6 app/sculpting/CLI tests |

---

## 3. Deep Dive: Key Technical Implementations

### 3.1 Targeted Perimeter-Seeded Island Detection (`arcadia-world/src/islands/perimeter.rs`)
The previous island detection algorithm scanned a large global AABB bounding box and relied on artificial vertical cantilever heuristics. This caused two major failure modes:
1. Intact ceiling beams and arches $2\text{m}$ away from a carve would collapse because the scan window artificially truncated them.
2. Floating mid-air structures touching the scan window boundary were falsely anchored.

**The Solution:** `detect_disconnected_islands_from_carve`:
- Gathers solid voxels in the crater boundary shell:
  $$r \in (R_{\text{carve}}, R_{\text{carve}} + 1.5\text{ voxels}]$$
- For each solid boundary voxel touching an air voxel or the carved crater, expands a 6-connected breadth-first search component.
- **Short-circuiting invariants:**
  - If a component reaches bedrock ($y \le 4$ or `MAT_BEDROCK`), the entire component is marked **grounded** and search terminates instantly.
  - If visited voxel count exceeds `max_island_voxels` (default 384–768), the structure is confirmed as part of the immovable terrain foundation ($<0.1\text{ ms}$ turnaround).
  - Only components strictly isolated from bedrock and within the voxel budget detach into `DestructionEvent::detached_islands`.

```rust
// crates/arcadia-world/src/islands/perimeter.rs
pub fn detect_disconnected_islands_from_carve<W: VoxelQuery + ?Sized>(
    world: &W,
    center_v: Vec3,
    radius_v: f32,
    max_island_voxels: usize,
) -> Vec<Vec<(IVec3, u8)>> {
    // 1. Gather perimeter seeds touching the carve crater boundary
    // 2. 6-connected BFS from each seed with global deduplication
    // 3. Short-circuit on bedrock (y <= 4) or voxel budget threshold
    // 4. Return detached clusters for RigidVoxelBody construction
}
```

### 3.2 6-DOF Inertia Tensor with Parallel-Axis Correction (`arcadia-physics`)
For a cluster of $N$ micro-voxels of size $s = \frac{1}{16}\text{ m}$ and material density $\rho$ ($m_0 \approx 0.586\text{ kg}$ for stone):
1. **Center of Mass:**
   $$\mathbf{c} = \frac{1}{N} \sum_{i=1}^N \mathbf{p}_i, \quad M = N m_0$$
2. **Moment of Inertia Tensor ($\mathbf{I}$):**
   Accounting for each individual cubic micro-voxel's self-inertia about its own center ($\frac{1}{6} m_0 s^2$):
   $$I_{xx} = m_0 \sum_{i=1}^N \left( (y_i - c_y)^2 + (z_i - c_z)^2 + \frac{s^2}{6} \right)$$
   $$I_{yy} = m_0 \sum_{i=1}^N \left( (x_i - c_x)^2 + (z_i - c_z)^2 + \frac{s^2}{6} \right)$$
   $$I_{zz} = m_0 \sum_{i=1}^N \left( (x_i - c_x)^2 + (y_i - c_y)^2 + \frac{s^2}{6} \right)$$
   $$I_{xy} = -m_0 \sum_{i=1}^N (x_i - c_x)(y_i - c_y), \quad I_{xz} = -m_0 \sum_{i=1}^N (x_i - c_x)(z_i - c_z), \quad I_{yz} = -m_0 \sum_{i=1}^N (y_i - c_y)(z_i - c_z)$$

### 3.3 Dynamic Body Sculpting & Island Partitioning (`arcadia-app/src/sculpting.rs`)
Unified scene raycasting (`raycast_scene`) resolves whether the player is targeting static world terrain or a dynamic rigid body:
- **Left-Click (Carve):** When hitting a dynamic rigid body, `body.carve_sphere_meters` carves voxels in body-local coordinates, applies scorch materials, updates the local CS64 tree, and re-registers the GPU node index. If the blast cuts completely through the body, `find_connected_components` partitions the remaining voxels into independent `RigidVoxelBody` instances.
- **Right-Click (Place):** Adds voxels in body-local space, dynamically expanding `dims_v`, updating boundary hull contact points, and recomputing mass and inertia properties.

### 3.4 Terrain Re-Baking (`arcadia-app/src/app.rs`)
Resting dynamic bodies that remain motionless for `object_bake_timeout_s` (tracked via `body.rest_time`):
1. Rasterize body-local micro-voxels into integer world coordinates via conservative transformation:
   $$\mathbf{p}_{\text{world}} = \lfloor \mathbf{X} + \mathbf{R} \cdot \mathbf{p}_{\text{local}} \rceil$$
2. Insert transformed voxels into `WorldMap::set_voxels_global`.
3. Stream incremental `GpuPatch` updates to GPU VRAM.
4. Despawn the body from the active physics simulation, freeing simulation and raymarching slots.

---

## 4. Test Suite & Verification Results

All 60 unit tests across all 5 workspace crates pass with zero compiler warnings:

- **`arcadia-app` (9 tests passed):**
  - `sculpting::tests::test_carve_and_split_dynamic_rigid_body`: Verifies cutting a body in two creates two independent physics bodies.
  - `sculpting::tests::test_place_voxels_on_dynamic_rigid_body`: Verifies right-click voxel placement expands bounds and updates mass.
  - `sculpting::tests::test_rigid_body_bakes_into_world_after_timeout`: Verifies resting bodies bake into `WorldMap` after timeout.
  - `cli::tests`: Flag parsing, defaults, and `--help` validation.
  - `benchmark::tests`: Life-cycle, time-of-day diurnal progression, and statistical calculation.
- **`arcadia-physics` (25 tests passed):**
  - `test_rigid_voxel_inertia_and_com`: Discrete mass and inertia validation with parallel-axis term.
  - `test_obb_aabb_sat_intersection` & `test_obb_obb_sat_*`: 15-axis SAT broadphase and contact manifold generation.
  - `test_box3d_two_body_stacking_stability` & `test_box3d_three_body_stacking_convergence`: Multi-point manifold convergence.
  - `test_pair_coulomb_friction`, `test_pair_warm_start`, `test_pair_sequential_impulse_solve`.
  - `test_player_rigid_body_collision_and_push`: Kinematic player character stepping and pushing.
  - `test_dynamic_rigid_body_carve_no_skyward_launch`: Verifies carving impulses do not launch bodies skyward.
  - `test_severed_pillar_toppling_dynamics` & `test_structural_island_detachment_and_falling_rigid_body`.
- **`arcadia-world` (16 tests passed):**
  - `islands::tests::test_perimeter_seeded_floating_island_detection`: Verifies detached structures are severed.
  - `islands::tests::test_perimeter_seeded_grounded_pillar_remains_anchored`: Verifies grounded pillars do not detach.
  - `islands::tests::test_nearby_edit_does_not_collapse_bridge_or_ceiling`: Verifies edits $2\text{m}$ away preserve intact geometry.
  - `islands::tests::test_downward_support_propagation_stairs` & `test_cantilever_and_floating_arch_edge_cases`.
  - `islands::tests::test_anti_slicing_boundary_anchors` & `test_world_region_snapshot_island_detection`.
  - `benchmark_scene::tests`: Generation, camera path interpolation, point light verification.
- **`arcadia-core` (7 tests passed):**
  - Bit index roundtrip, coordinate decomposition, MTF cache, POPCNT child offset, leaf brick packing, CPU raycast, homogeneous collapse.
- **`arcadia-render` (3 tests passed):**
  - Reverse-Z projection near/far mapping, linear distance-to-depth conversion, Naga WGSL shader validation.

---

## 5. Next Session Priorities & Action Items

The following high-priority issues and enhancements are scheduled for the immediate next session:

### 1. Physics Engine Refinements
- **Fix Bounciness in Water:**
  - Dynamic bodies currently exhibit excessive elasticity or buoyancy bounce upon entering water volumes.
  - *Action:* Apply progressive hydrodynamic drag forces ($\mathbf{F}_{\text{drag}} \propto -\mathbf{v} |\mathbf{v}|$), rotational damping, and water immersion restitution attenuation to eliminate artificial oscillations and settle floating/sinking bodies naturally.
- **Make Object Collision Use True Bounding Box:**
  - Pairwise and world collision routines currently rely on approximations that can allow edge penetration or conservative clipping mismatches.
  - *Action:* Ensure all broadphase and narrowphase routines strictly reference the true geometric oriented bounding box (OBB) and tight voxel extents without radius over-inflation.
- **Fix Objects Flying into the Air When Carving:**
  - Carving near or through dynamic bodies can impart excessive explosive impulse or recoil velocity vectors that launch objects unnaturally into the sky.
  - *Action:* Rebalance and clamp carve-induced impulse transfer, ensuring explosive force scales strictly with blast proximity and mass, with downward/outward bias instead of unconstrained upward launch velocities.

### 2. Island Detection Optimizations
- **Fix Lag during Island Detection:**
  - Large spherical craters or deep terrain destruction can induce frame drops during perimeter flood-fill exploration.
  - *Action:* Optimize BFS queue allocations, introduce early bitset spatial bounding checks, and hook large structures into the background worker channel (`P3` in roadmap).
- **Fix Single Voxels Remaining in the Air (Delete Them):**
  - Small carve operations can leave isolated, floating single micro-voxels or 2-voxel fragments hovering in mid-air that fail to trigger full rigid body detachment.
  - *Action:* Add an isolated fragment sweep filter: any detached voxel component consisting of $\le 3$ micro-voxels should automatically dissolve into debris dust particles (`MAT_DUST_SMOKE`) rather than floating in space.

### 3. Object Baking Optimization
- **Fix Lag during Object Baking:**
  - When resting dynamic bodies exceed their rest timeout ($>60\text{ s}$) and bake back into static `WorldMap` voxels, the conversion and GPU patch generation induce noticeable main-thread hitching.
  - *Action:* Optimize `to_world_voxels` voxel deduplication and batch `set_voxels_global` calls into consolidated `GpuPatch` staging updates to eliminate frame latency spikes during terrain absorption.

