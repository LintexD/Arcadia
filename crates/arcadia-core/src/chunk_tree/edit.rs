//! Dynamic procedural modification of `ChunkTree64`.
//! Owns CPU voxel carving, sphere placement, batch clearing, and material insertion.
//!
//! Submodules:
//! - [`carve`]: Destructive spherical carving, scorch rims, and debris reporting callbacks.
//! - [`place`]: Additive spherical voxel placement, emission LOD weighting, and material painting.
//! - [`clear`]: Batch clearing of chunk-local voxel coordinates.
//! - [`set`]: Batch insertion or updating of chunk-local voxels and materials.

pub mod carve;
pub mod clear;
pub mod place;
pub mod set;

pub use set::{BrickVoxelListMap, ChunkEditReport};
