//! Bitwise 64-tree connected-component island detection.
//! Owns CPU connected component graph analysis and bitwise flood fill.
//!
//! Submodules:
//! - [`bitwise`]: 64-bit ALU bit shifts, boundary face masks, and intra-brick dilation.
//! - [`connectivity`]: Two-tiered macro-brick graph traversal, compressive support propagation, and component clustering.
//! - [`perimeter`]: Targeted perimeter-seeded island detection around spherical carve craters.
//! - [`query`]: Trait and snapshot definitions for thread-safe voxel queries and ground anchoring.

pub mod bitwise;
pub mod connectivity;
pub mod perimeter;
pub mod query;

pub use bitwise::{
    dilate_brick_step, dilate_support_brick_step, extract_brick_surface_mask, flood_fill_brick_u64,
    flood_fill_support_brick_u64,
};
pub use connectivity::{detect_disconnected_island_components, detect_disconnected_islands};
pub use perimeter::{
    detect_disconnected_islands_from_carve, detect_disconnected_islands_from_carve_bounded,
};
pub use query::{is_externally_anchored, VoxelQuery, WorldRegionSnapshot};
