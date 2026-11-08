//! Core data types, GPU descriptors, and simulation outcomes for rigid voxel bodies.
//! Owns CPU rigid body simulation state and defines host-shareable structs matching `shaders/common/types.wgsl`.

use super::collision::compound::CompoundCollider;
use arcadia_core::{
    bytemuck::{Pod, Zeroable},
    ChunkTree64, MaterialId,
};
use arcadia_world::DebrisParticle;
use glam::{IVec3, Mat3, Quat, Vec3};

/// Maximum simultaneously active ray-marched rigid voxel bodies on GPU.
pub const MAX_RIGID_BODIES: usize = 32;

/// GPU-aligned rigid body instance descriptor (`224 bytes`, 16-byte aligned).
/// Transforms rays between world space (meters) and body-local CS64 tree space (`[0 .. dims_v * VOXEL_SIZE]`).
// GPU-SYNC: Mirrors `RigidBodyInstance` in `shaders/common/types.wgsl` (@group(0) @binding(6), 224 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuRigidBodyInstance {
    /// Column-major 4x4 matrix mapping world-space meters to body-local tree meters.
    pub world_to_local: [[f32; 4]; 4],
    /// Column-major 4x4 matrix mapping body-local tree meters to current world-space meters.
    pub local_to_world: [[f32; 4]; 4],
    /// Column-major 4x4 matrix mapping body-local tree meters to previous frame's world-space meters.
    pub prev_local_to_world: [[f32; 4]; 4],
    /// `xyz` = local bounding box dimensions in meters (`dims_v * VOXEL_SIZE`), `w` = `f32::from_bits(root_node_idx)`.
    pub local_bounds_and_root: [f32; 4],
    /// `xyz` = world-space bounding sphere center in meters, `w` = world-space bounding sphere radius in meters.
    pub world_sphere: [f32; 4],
}

/// Per-frame simulation outcome for a `RigidVoxelBody` after stepping or carving.
#[derive(Default)]
pub struct RigidStepOutcome {
    /// Debris particles spawned by high-energy impact shattering or carving.
    pub spawned_debris: Vec<DebrisParticle>,
    /// True if the body shattered completely into debris and must be despawned.
    pub destroyed: bool,
    /// Independent child rigid bodies spawned when carving partitions this body into disconnected islands.
    pub split_bodies: Vec<RigidVoxelBody>,
    /// True if voxels or materials on this body were mutated.
    pub modified: bool,
}

/// Dynamic 6-DOF rigid body constructed from a detached cluster of $(1/16\text{ m})^3$ micro-voxels.
#[derive(Clone, Debug)]
pub struct RigidVoxelBody {
    /// Monotonically increasing unique identifier for tracking bodies across asynchronous worker tasks.
    pub id: u32,
    /// Whether this body has qualified and is currently undergoing background voxel rasterization.
    pub is_baking: bool,
    /// Local sparse voxel coordinates `(lx, ly, lz, mat_id)` relative to `[0 .. dims_v)`.
    pub local_voxels: Vec<(u32, u32, u32, MaterialId)>,
    /// Subset of `local_voxels` on the 6-connected exterior hull (`r_rel_com`, `mat_id`) for impact splintering.
    pub boundary_voxels: Vec<(Vec3, MaterialId)>,
    /// Exact 3D greedy-meshed compound cuboid partition and local flat BVH for true-shape collisions.
    pub collider: CompoundCollider,
    /// Local bounding box dimensions in voxels (`max_v - min_v + 1`, clamped to $\le 256$).
    pub dims_v: IVec3,
    /// Local CS64 hierarchical 64-tree built from `local_voxels` for GPU OBB raymarching.
    pub local_tree: ChunkTree64,
    /// Relocated root `Node64` index inside `WorldMap::gpu_nodes` (`EMPTY_CHUNK_SENTINEL` until registered).
    pub gpu_root_node_idx: u32,
    /// Center of mass in body-local tree meters (`[0 .. dims_v * VOXEL_SIZE]`).
    pub com_local_m: Vec3,
    /// Half-extents of the macro local bounding box in meters (`0.5 * dims_v * VOXEL_SIZE`).
    pub half_extents_m: Vec3,
    /// Total mass $M = N \cdot m_0$ in kilograms.
    pub mass: f32,
    /// Inverse mass $M^{-1}$ ($0.0$ if static/infinite mass).
    pub inv_mass: f32,
    /// Body-space moment of inertia tensor $\mathbf{I}_{\text{body}}$ about `com_local_m`.
    pub inertia_body: Mat3,
    /// Body-space inverse moment of inertia tensor $\mathbf{I}_{\text{body}}^{-1}$.
    pub inv_inertia_body: Mat3,
    /// World-space position of the Center of Mass (meters).
    pub position: Vec3,
    /// World-space orientation quaternion (unit length).
    pub orientation: Quat,
    /// Previous frame's world-space CoM position for analytic temporal motion vectors.
    pub prev_position: Vec3,
    /// Previous frame's world-space orientation for analytic temporal motion vectors.
    pub prev_orientation: Quat,
    /// Linear velocity of CoM in world space ($\text{m/s}$).
    pub linear_velocity: Vec3,
    /// Angular velocity in world space ($\text{rad/s}$).
    pub angular_velocity: Vec3,
    /// Restitution (bounciness) coefficient $e \in [0, 1]$.
    pub restitution: f32,
    /// Dynamic Coulomb friction coefficient $\mu \in [0, 1]$.
    pub friction: f32,
    /// Seconds spent below the sleep velocity threshold.
    pub sleep_timer: f32,
    /// Whether this body has settled and entered sleep mode.
    pub sleeping: bool,
    /// Volume-weighted fraction of the body submerged in `MAT_WATER_CRYSTAL` voxels ($[0, 1]$).
    pub submersion_ratio: f32,
    /// Total lifetime in seconds since detachment.
    pub age: f32,
    /// Duration in seconds this body has remained stationary without external interaction.
    pub rest_time: f32,
    /// Optional emissive light source `(local_offset_rel_com_m, radius_m)`.
    pub emissive_light: Option<(Vec3, f32)>,
}

/// Single 3D contact constraint point between two dynamic rigid bodies with 2D friction state.
#[derive(Clone, Copy, Debug, Default)]
pub struct ContactPoint3D {
    /// World-space position of the contact point.
    pub position: Vec3,
    /// Offset vector from Body A Center of Mass to `position` in world space.
    pub r_a: Vec3,
    /// Offset vector from Body B Center of Mass to `position` in world space.
    pub r_b: Vec3,
    /// Penetration depth in meters ($> 0$ when overlapping).
    pub penetration: f32,
    /// Accumulated normal impulse magnitude ($\text{N}\cdot\text{s}$).
    pub accum_n: f32,
    /// Accumulated tangential impulse along `tangent1` ($\text{N}\cdot\text{s}$).
    pub accum_t1: f32,
    /// Accumulated tangential impulse along `tangent2` ($\text{N}\cdot\text{s}$).
    pub accum_t2: f32,
    /// Effective mass along contact normal.
    pub effective_mass_n: f32,
    /// Effective mass along `tangent1`.
    pub effective_mass_t1: f32,
    /// Effective mass along `tangent2`.
    pub effective_mass_t2: f32,
    /// Feature identifier for contact matching across substeps (warm starting).
    pub feature_id: u32,
}

/// Multi-point 3D contact manifold between a pair of dynamic rigid bodies sharing a common normal.
#[derive(Clone, Copy, Debug)]
pub struct BodyPairManifold {
    /// Index of Body A in the active body slice.
    pub body_a_idx: usize,
    /// Index of Body B in the active body slice.
    pub body_b_idx: usize,
    /// Unit contact normal pointing from Body A toward Body B in world space.
    pub normal: Vec3,
    /// First orthonormal tangent vector in the contact plane.
    pub tangent1: Vec3,
    /// Second orthonormal tangent vector in the contact plane (`normal x tangent1`).
    pub tangent2: Vec3,
    /// Up to 4 coplanar/extremal contact points forming the support polygon.
    pub points: [ContactPoint3D; 4],
    /// Number of active contact points in `points` (`1..=4`).
    pub point_count: usize,
    /// Combined Coulomb friction coefficient.
    pub friction: f32,
    /// Combined restitution coefficient (attenuated to `0.0` when submerged).
    pub restitution: f32,
}

impl Default for BodyPairManifold {
    fn default() -> Self {
        Self {
            body_a_idx: 0,
            body_b_idx: 0,
            normal: Vec3::ZERO,
            tangent1: Vec3::ZERO,
            tangent2: Vec3::ZERO,
            points: [ContactPoint3D::default(); 4],
            point_count: 0,
            friction: 0.0,
            restitution: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ContactManifold {
    pub r_world: Vec3,
    pub normal: Vec3,
    pub penetration: f32,
    pub accum_n: f32,
    pub accum_t1: f32,
    pub accum_t2: f32,
    pub tangent1: Vec3,
    pub tangent2: Vec3,
    pub effective_mass_n: f32,
    pub effective_mass_t1: f32,
    pub effective_mass_t2: f32,
}
