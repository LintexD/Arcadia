//! `arcadia-physics`: Bitwise 64-tree and compound greedy-box collision solver,
//! first-person kinematic character controller, and 6-DOF rigid voxel dynamics.
//! Owns CPU collision hierarchies, character kinematics, and rigid body simulation state.

pub mod character;
pub mod rigid_voxel;

pub use character::{CharacterController, MovementInput};
pub use rigid_voxel::{
    compute_feature_contact, BodyPairManifold, CompoundCollider, ContactManifold, ContactPoint3D,
    GpuRigidBodyInstance, RigidStepOutcome, RigidVoxelBody, SubBoxBvhNode, VoxelSubBox,
    ALL_FACES_EXPOSED, FACE_NEG_X, FACE_NEG_Y, FACE_NEG_Z, FACE_POS_X, FACE_POS_Y, FACE_POS_Z,
    MAX_PENETRATION_BIAS_VEL, MAX_RIGID_BODIES, PENETRATION_SLOP,
};
