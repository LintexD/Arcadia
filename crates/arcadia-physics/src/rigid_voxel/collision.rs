//! Collision broadphase, compound greedy-box BVH, and 15-axis SAT contact generation.
//! Owns CPU collision hierarchies and narrowphase contact manifolds against terrain and rigid bodies.

pub mod broadphase;
pub mod compound;
pub mod manifold;
pub mod terrain;
pub mod voxel;
