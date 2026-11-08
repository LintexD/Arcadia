# 06. Architectural Synthesis & Engine Integration Roadmap

## 1. System Integration Overview

To introduce animated, destructible, pathfinding mobs into Arcadia while strictly respecting the architectural principles established in [`ARCHITECTURE.md`](../../ARCHITECTURE.md), we design a modular integration strategy across Arcadia’s workspace:

```text
arcadia/
├── crates/
│   ├── arcadia-core/                 # LeafBrick64, Node64, in-place carving, raycast_cpu
│   │                                 # (Zero breaking changes; reused directly)
│   ├── arcadia-physics/              # RigidVoxelBody, character auto-step, SAT manifolds
│   │                                 # (Dismembered limbs spawn as RigidVoxelBody)
│   ├── arcadia-world/                # islands.rs, structural collapse, debris particles
│   │                                 # (Connectivity flood fill reused for bone severance)
│   ├── arcadia-mob/                  # [NEW CRATE] Mob manager, skeletons, AI, navigation
│   │   └── src/
│   │       ├── lib.rs                # Crate root exports
│   │       ├── skeleton.rs           # Bone hierarchy, forward kinematics, 2-bone IK
│   │       ├── anatomy.rs            # Multi-layer anatomy, material stratas, weak points
│   │       ├── damage.rs             # Carve delegation, bone severance, island extraction
│   │       ├── navigation.rs         # Tier 1/2 HVG, 3D JPS, Dijkstra surface flow fields
│   │       ├── ai.rs                 # Behavior trees, perception, combat slot rings
│   │       └── manager.rs            # Horde coordinator, COW template pool, LOD updates
│   ├── arcadia-render/               # GPU Pools, CameraUniforms, GpuContext
│   │   └── src/
│   │       └── gpu_pools.rs          # Extended with MobLimbInstance storage buffer
│   └── arcadia-app/                  # Simulation loop, HUD, debug GUI
│       └── src/app/simulation.rs     # Mob manager update dispatch
│
└── shaders/
    ├── common/types.wgsl             # MobLimbInstance struct definition
    ├── primary_gbuffer.wgsl          # Ray-OBB slab test & local HDDA limb traversal
    └── mob_cull_compact.wgsl         # [NEW] 1D compute shader for frustum/depth culling
```

---

## 2. Core Rust Struct Specifications (`arcadia-mob`)

### 2.1 The `MobLimbInstance` GPU Mirror Contract
Directly uploaded to `@group(0) @binding(7)` in `arcadia-render`:

```rust
// GPU-SYNC: Mirrors `MobLimbInstance` in `shaders/common/types.wgsl` (128 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMobLimbInstance {
    pub world_to_local: [[f32; 4]; 4], // 64 bytes: Transforms ray origin & dir to bone space
    pub local_to_world: [[f32; 4]; 4], // 64 bytes: Transforms hit normals to world space
    pub half_extents: [f32; 3],        // 12 bytes: Local OBB half-extents
    pub node_pool_offset: u32,         // 4 bytes: Base index into global gpu_nodes
    pub prev_local_to_world: [[f32; 4]; 4], // 64 bytes: For analytic DLAA motion vectors
    pub mob_id: u32,                   // 4 bytes: Entity identifier
    pub limb_flags: u32,               // 4 bytes: Weakpoint flag, emissive flag, LOD tier
    pub _padding: [u32; 2],            // 8 bytes: Enforce strict 16-byte alignment
}
```

### 2.2 Mob Entity & Skeleton
```rust
//! Represents an individual animated, destructible voxel mob entity.

use arcadia_core::ChunkTree64;
use glam::{Mat4, Quat, Vec3};

pub struct VoxelMob {
    pub id: u32,
    pub archetype: MobArchetype,
    pub root_position: Vec3,
    pub root_rotation: Quat,
    pub velocity: Vec3,
    pub skeleton: MobSkeleton,
    pub anatomy: MobAnatomy,
    pub ai_state: MobAiState,
    pub lod_tier: u8,
}

pub struct MobSkeleton {
    pub bones: Vec<MobBoneData>,
}

pub struct MobBoneData {
    pub name: String,
    pub parent_idx: Option<usize>,
    pub local_bind_transform: Transform,
    pub current_local_transform: Transform,
    pub world_matrix: Mat4,
    pub prev_world_matrix: Mat4,
    pub half_extents: Vec3,
    pub voxel_tree: ChunkTree64, // Local CS64 voxel volume
    pub is_severed: bool,
}
```

---

## 3. Shader Integration Contracts (`shaders/`)

### 3.1 WGSL Type Definition (`shaders/common/types.wgsl`)
```wgsl
struct MobLimbInstance {
    world_to_local: mat4x4<f32>,
    local_to_world: mat4x4<f32>,
    half_extents: vec3<f32>,
    node_pool_offset: u32,
    prev_local_to_world: mat4x4<f32>,
    mob_id: u32,
    limb_flags: u32,
    pad0: u32,
    pad1: u32,
};
```

### 3.2 Ray Marching Pipeline (`shaders/primary_gbuffer.wgsl`)
During Pass 1 primary ray marching:
1. Primary rays iterate through the compacted `visible_mob_limbs` list.
2. For each limb, test the ray against `half_extents` using an analytical slab test:
   ```wgsl
   let ray_orig_local = (limb.world_to_local * vec4<f32>(ray_orig_world, 1.0)).xyz;
   let ray_dir_local  = (limb.world_to_local * vec4<f32>(ray_dir_world, 0.0)).xyz;
   let slab = intersect_aabb(ray_orig_local, ray_dir_local, -limb.half_extents, limb.half_extents);
   ```
3. If $t_{\text{in}} < t_{\text{hit\_world}}$, traverse the limb's local CS64 tree starting at `node_pool_offset`.
4. If a voxel is hit, update $t_{\text{hit\_world}}$, transform normal via `local_to_world`, and compute screen motion vectors using `prev_local_to_world`.

---

## 4. Step-by-Step Implementation Roadmap

```text
Phase 1: Articulated Rigging & Forward Kinematics (Zero-Damage Foundation)
   │
   ▼
Phase 2: Multi-Layer Anatomy & Volumetric Carving (Shotgun/Railgun Holes)
   │
   ▼
Phase 3: Structural Dismemberment & Rigid Body Spawning (Islands.rs Integration)
   │
   ▼
Phase 4: Hierarchical Voxel Navigation (HVG) & Auto-Step Kinematics
   │
   ▼
Phase 5: Tactical AI & Destructible Breach Mechanics
   │
   ▼
Phase 6: Horde Scaling, Copy-On-Write Instancing & GPU Culling
```

### Milestone Breakdown

#### Milestone 1: Articulated Rigging & G-Buffer Ray Marching
- Implement `MobSkeleton` with forward kinematics.
- Create a test bipedal demon model in local CS64 trees.
- Add `MobLimbInstance` buffer to `arcadia-render`.
- Render the animated mob in the benchmark sanctuary scene with real-time ReSTIR GI reflections and soft sun shadows.

#### Milestone 2: Volumetric Anatomy & Weapon Carving
- Implement the 4-layer anatomy palette (`MAT_DEMON_CHITIN`, `MAT_DEMON_FLESH`, `MAT_DEMON_BONE`, `MAT_HELL_CORE`).
- Hook player weapon raycasts to call `carve_sphere` directly on hit limbs.
- Verify that buckshot removes chitin, exposing muscle, and railguns punch visible holes through demon torsos.
- Verify ReSTIR GI picks up the exposed emissive core.

#### Milestone 3: Structural Island Dismemberment & Gibs
- Adapt `islands.rs` connectivity flood-fill to bone voxels.
- When bone voxels are severed, detach distal limb chunk, instantiate a 6-DOF [`RigidVoxelBody`](../../crates/arcadia-physics/src/rigid_voxel.rs), and trigger a burst of GPU debris particles.
- Verify severed limbs tumble, collide with terrain, and bake into rubble.

#### Milestone 4: Hierarchical Voxel Pathfinding (HVG)
- Generate Tier 2 meso-surface graphs on chunk generation.
- Implement $O(1)$ dynamic invalidation hooked to `ChunkEditReport`.
- Implement Dijkstra flow fields for horde swarm chasing.
- Implement mob auto-step (1..3 voxels) and jump trajectory solvers.

#### Milestone 5: Tactical AI & Breach Mechanics
- Implement Behavior Trees for grunts and elites.
- Implement zero-brick LOS perception and acoustic sound propagation.
- Implement concentric polar combat slot reservation around the player.
- Implement wall-breaching attack state when mobs are blocked by destructible barriers.

#### Milestone 6: Horde Optimization & GPU Culling
- Implement Copy-On-Write (COW) anatomy templating.
- Implement 1D GPU compute limb culling and workgroup stream compaction.
- Stress-test 100 active demons in Sanctuary scene, verifying 60+ FPS performance budget.

