//! Phase 6: Structural Island Collapse & Rigid Body Voxel Dynamics (`rigid_voxel`).
//! Owns CPU 6-DOF rigid body physics simulation, compound greedy-box BVH colliders, and GPU instance descriptors.

pub mod collision;
mod construction;
mod solver;
mod transform;
mod types;
mod voxel_ops;

pub use collision::compound::{
    CompoundCollider, SubBoxBvhNode, VoxelSubBox, ALL_FACES_EXPOSED, FACE_NEG_X, FACE_NEG_Y,
    FACE_NEG_Z, FACE_POS_X, FACE_POS_Y, FACE_POS_Z,
};
pub use collision::terrain::compute_feature_contact;
pub use solver::{MAX_PENETRATION_BIAS_VEL, PENETRATION_SLOP};
pub use types::{
    BodyPairManifold, ContactManifold, ContactPoint3D, GpuRigidBodyInstance, RigidStepOutcome,
    RigidVoxelBody, MAX_RIGID_BODIES,
};
