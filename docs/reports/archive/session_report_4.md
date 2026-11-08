# Arcadia Voxel Engine — Session 4 Engineering Report

**Date:** October 26, 2026  
**Status:** Completed, Polished & Verified  
**Target Architecture:** Rust 2024, `wgpu` 23.0 (Vulkan 1.3 / DX12), WGSL Compute & Render Shaders, `egui` 0.30, Winit 0.30  
**Test Platform:** AMD Ryzen 5 7535U, Radeon 660M (RDNA2 Integrated GPU, DX12), 16 GB DDR5  

---

## 1. Executive Summary

Session 4 successfully delivered and polished **Phase 6: Structural Island Collapse & 6-DOF Rigid Body Dynamics** (*Teardown*-style voxel physics):
1. **Bitwise Structural Island Extraction ([islands.rs](../../../crates/arcadia-world/src/islands.rs), [destruction.rs](../../../crates/arcadia-world/src/destruction.rs)):**
   - Implemented intra-brick 64-bit register bitwise dilation combined with a two-tiered leaf brick graph flood-fill.
   - Eliminated floating structures by ensuring search bounding faces facing empty air are never marked as anchors; boundary faces only anchor if contiguous solid terrain continues outside the box (`world.get_voxel(...) != MAT_AIR`).
   - Expanded the vertical search window to span the full world height ($y = 0..\text{world\_max}.y$) with up to 14m horizontal padding, guaranteeing tall Roman columns, colonnades, and vaulted arches are never truncated mid-structure.
2. **6-DOF Rigid Voxel Body Dynamics ([rigid_voxel.rs](../../../crates/arcadia-physics/src/rigid_voxel.rs)):**
   - Implemented `RigidVoxelBody` computing discrete center of mass and the exact $3 \times 3$ moment of inertia tensor $\mathbf{I}_{\text{body}}$ with cubic micro-voxel parallel-axis correction $\frac{s^2}{6}$.
   - Built dynamic local `ChunkTree64` hierarchies per cluster, registered into `WorldMap::gpu_nodes` for GPU compute raymarching.
3. **GPU Dynamic Raymarching & 3D Motion Vectors ([primary_gbuffer.wgsl](../../../shaders/primary_gbuffer.wgsl), [traversal64.wgsl](../../../shaders/common/traversal64.wgsl)):**
   - Bound `RigidBodyInstance` storage buffer array at `@group(0) @binding(6)`.
   - Transforms primary camera rays into body-local tree coordinates, marches the local CS64 hierarchy, and writes G-buffer hits with exact analytic 3D motion vectors using `prev_local_to_world` for temporal anti-aliasing (DLAA/TSR).
4. **Kinematic Character & Rigid Body Collisions ([character.rs](../../../crates/arcadia-physics/src/character.rs)):**
   - Player AABB tests against dynamic rigid bodies via 15-axis SAT OBB intersection.
   - The player can jump onto, walk across, and auto-step up 1..3 voxels (6.25cm to 18.75cm) onto fallen rubble, applying physical push impulses when walking into them.
5. **Pairwise Object-vs-Object Collisions ([rigid_voxel.rs](../../../crates/arcadia-physics/src/rigid_voxel.rs)):**
   - Implemented `RigidVoxelBody::step_system` resolving mutual collisions between dynamic bodies.
   - Uses hierarchical broadphase (bounding spheres $\to$ AABB overlap $\to$ SAT OBB) and bidirectional boundary point sampling against opposing local CS64 trees, resolving normal impulse, friction, and separation.
6. **Ground Contact Stabilization & Immediate Digging Wake-Up:**
   - Eradicated terrain bouncing and micro-jitter via 5mm penetration slop, soft Baumgarte factor ($0.20$), restitution threshold ($> 2.2\text{ m/s}$), and resting vertical velocity zeroing ($|v_y| < 0.35\text{ m/s}$).
   - Replaced full-AABB sleep checks with exact bottom-boundary voxel support probes (`p.y <= aabb_min.y + 1.5 * VOXEL_SIZE`). Digging beneath a resting object immediately triggers wake-up and gravity collapse.

---

## 2. Key Accomplishments & Technical Deep Dive

### 2.1 Bitwise 64-Tree Island Detection & Anchor Invariants
When an explosion carves geometry:
- **Intra-Brick Bitwise Dilation:** Within each $4 \times 4 \times 4$ Leaf Brick (64 voxels), components are dilated in single CPU clock cycles using 64-bit bit shifts:
  $$\text{dilate}(S) = S \mid ((S \& \text{NOT\_X\_MAX}) \ll 1) \mid ((S \& \text{NOT\_X\_MIN}) \gg 1) \mid \dots$$
- **Eliminating Mid-Air False Anchors:** Previously, touching the lateral search box boundary (`vx == min_v.x`, etc.) unconditionally marked a component as anchored to the mountain, leaving severed arches and balcony overhangs floating in the sky. In `islands.rs`, a boundary face voxel is now only anchored if the adjacent voxel in the world is non-air:
  ```rust
  if vy <= 4
      || (vx == min_v.x && world.get_voxel(vx - 1, vy, vz) != MAT_AIR)
      || (vx == max_v.x && world.get_voxel(vx + 1, vy, vz) != MAT_AIR)
      || (vz == min_v.z && world.get_voxel(vx, vy, vz - 1) != MAT_AIR)
      || (vz == max_v.z && world.get_voxel(vx, vy, vz + 1) != MAT_AIR)
  {
      anchor_mask |= bit_mask;
  }
  ```
- **Full-Altitude Vertical Coverage:** In `destruction.rs`, `island_min_v.y` is set to $0$ and `island_max_v.y` is set to `world_max.y` (the sky ceiling). Tall columns and vaulted roofs are never truncated at arbitrary heights.

### 2.2 6-DOF Rigid Body Mass, Inertia & Solver
For detached clusters of $N$ voxels ($m_0 \approx 0.586\text{ kg}$ per $6.25\text{ cm}$ stone voxel):
- **Discrete Inertia Tensor with Parallel-Axis Correction:**
  $$I_{xx} = m_0 \sum_{i=1}^N \left( (y_i - c_y)^2 + (z_i - c_z)^2 + \frac{s^2}{6} \right), \quad I_{xy} = -m_0 \sum_{i=1}^N (x_i - c_x)(y_i - c_y)$$
  Diagonal dominance is verified, and the body-space inverse inertia tensor $\mathbf{I}_{\text{body}}^{-1}$ is precomputed.
- **World-Space Inverse Inertia Tensor:**
  $$\mathbf{I}_{\text{world}}^{-1} = \mathbf{R}(\mathbf{q}) \mathbf{I}_{\text{body}}^{-1} \mathbf{R}(\mathbf{q})^T$$
- **15-Axis SAT OBB Broadphase:** Evaluates 3 AABB cardinal axes, 3 OBB local axes, and 9 cross-product axes before testing voxel contacts.
- **Surface Hull Narrowphase:** Subsamples 6-connected exterior boundary voxels ($\le 96$ points) and checks terrain occupancy via `world.get_voxel`.
- **Contact Smoothing:**
  - Penetration slop: $\text{slop} = 0.005\text{ m}$ (5mm).
  - Soft Baumgarte factor: $0.20$ (was $0.65$), preventing elastic overshoots.
  - Restitution: Strictly $0.0$ for impact speeds $\le 2.2\text{ m/s}$.
  - Settling clamp: When in multi-point contact with ground, vertical velocity $|v_y| < 0.35\text{ m/s}$ is zeroed to prevent micro-vibrations.

### 2.3 Object-vs-Object & Player-vs-Object Interaction
- **Pairwise Dynamic Collisions (`step_system`):**
  - Iterates all active body pairs with $O(N)$ broadphase culling (bounding spheres + AABBs).
  - Transforms boundary voxels of body A into body B's local coordinate space and samples `body_b.local_tree.get_voxel(lx, ly, lz)`.
  - Calculates relative contact velocity, normal impulse, tangential Coulomb friction, Baumgarte separation, and dynamic wake-up.
- **Kinematic Player Collisions (`character.rs`):**
  - `CharacterController::collides_at` tests both the static 64-tree and dynamic rigid bodies via OBB-vs-AABB SAT.
  - Player can smoothly walk over and land on fallen debris; `try_auto_step` allows effortless 1..3 voxel stair climbing onto resting bodies.
  - Walking into dynamic bodies applies a physical push impulse via `push_bodies_at`.

### 2.4 Bottom-Voxel Support Tracking & Blast Wake-Up
- **Direct Contact Inspection:** Instead of checking a coarse bounding box, sleeping bodies inspect boundary voxels at the bottom slice (`p_world.y <= aabb_min.y + 1.5 * VOXEL_SIZE`).
- If neither `world.get_voxel(vx, vy - 1, vz)` nor `world.get_voxel(vx, vy, vz)` is solid, the body awakens immediately.
- In `carve_action` ([main.rs](../../../crates/arcadia-app/src/main.rs#L285)), any sleeping body within `brush_radius + 3.0m` of an explosion is immediately awakened.

---

## 3. Verification & Test Suite

All 20 unit tests across the 5 workspace crates pass with zero compiler warnings:
- `arcadia-physics`: 5 passed
  - `test_rigid_voxel_inertia_and_com`: Exact CoM and off-diagonal zeroing on symmetric pillars.
  - `test_obb_aabb_sat_intersection`: 15-axis SAT intersection and separation.
  - `test_rigid_body_pairwise_collision`: Momentum transfer and velocity reflection on body-body collision.
  - `test_player_rigid_body_collision_and_push`: Player blocking and push impulse delivery.
  - `test_structural_island_detachment_and_falling_rigid_body`: Pillar severance, island extraction, falling, settling on bedrock, and wake-up upon carving beneath.
- `arcadia-world`: 4 passed (island dilation, benchmark generation, camera path, point lights).
- `arcadia-core`: 7 passed (homogeneous collapse, POPCNT child offsets, leaf brick packing, CPU raycast, coordinate roundtrip, MTF cache).
- `arcadia-render`: 1 passed (WGSL compute/render shader syntax & layout validation).
- `arcadia-app`: 3 passed (benchmark lifecycle, time-of-day progression, statistical results compute).

---

## 4. Known Items & Next Session Priorities

The following items have been documented in `docs/ROADMAP_AND_TODO.md` for the next session:
1. **Restitution & Settle Damping Tuning:** Further soften low-speed elasticity and increase ground friction so resting objects come to an immediate dead stop without subtle micro-bouncing.
2. **Rotational Inertia & Topple Speed:** Calibrate angular air damping and off-center blast torque so severed pillars and walls topple and crash to the ground at natural gravitational speed instead of slow-motion tilting.
3. **Island Detection Edge-Case Robustness:** Continue refining boundary conditions in `islands.rs` for complex multi-connected structures (spiral stairs, wide cantilever roofs, multi-column peristyles).
4. **Secondary Ray Visibility for Dynamic Objects:** Extend secondary reflection and refraction rays (water and mirrors in `primary_gbuffer.wgsl` and `lighting_restir_gi.wgsl`) to test `trace_rigid_bodies` so dynamic objects appear in reflections and underwater views.
5. **Emissive Dynamic Rigid Bodies:** When detached clusters contain emissive materials (`MAT_CRYSTAL_*`, `MAT_LAVA_MAGMA`, `MAT_POINT_LIGHT`), inject them into point light buffers or ReSTIR DI candidates so falling and resting crystal chunks cast real-time colored light.
