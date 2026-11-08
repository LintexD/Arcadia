//! `ChunkTree64`: 4-Level Sparse 64-Tree covering 256×256×256 voxels (16m × 16m × 16m).
//! Owns CPU sparse octree memory, node pooling, and structural carving/editing logic.
//!
//! Level hierarchy within a single chunk:
//! - **Level 0 (Root, `nodes[0]`)**: Covers 256³ voxels (`shift = 6`, child size = 64³ voxels = 4m³).
//! - **Level 1 (Macro)**: Covers 64³ voxels (`shift = 4`, child size = 16³ voxels = 1m³).
//! - **Level 2 (Meso)**: Covers 16³ voxels (`shift = 2`, child size = 4³ voxels = 0.25m³).
//! - **Level 3 (Leaf)**: Covers 4³ voxels (`shift = 0`, child size = 1³ voxel = (1/16m)³).

pub mod builder;
pub mod edit;
pub mod tree;

pub use builder::BuildNode;
pub use edit::{BrickVoxelListMap, ChunkEditReport};
pub use tree::{ChunkTree64, LEVEL0_SCALE, LEVEL1_SCALE, LEVEL2_SCALE, LEVEL3_SCALE};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{
        MaterialPalette, MAT_AIR, MAT_CRYSTAL_CYAN, MAT_GRASS_LUSH, MAT_STONE_DEEP,
    };
    use glam::Vec3;

    #[test]
    fn test_homogeneous_collapse_and_lookup() {
        let palette = MaterialPalette::default();
        // Create a chunk where y < 64 is solid stone, y == 64 is grass, y > 64 is air:
        let mut tree = ChunkTree64::from_generator(
            &palette,
            |_ox, oy, _oz, size| {
                if oy + size <= 64 {
                    Some(MAT_STONE_DEEP)
                } else if oy > 64 {
                    Some(MAT_AIR)
                } else {
                    None
                }
            },
            |_x, y, _z| {
                if y < 64 {
                    MAT_STONE_DEEP
                } else if y == 64 {
                    MAT_GRASS_LUSH
                } else {
                    MAT_AIR
                }
            },
        );

        assert_eq!(tree.get_voxel(10, 20, 30), MAT_STONE_DEEP);
        assert_eq!(tree.get_voxel(128, 64, 128), MAT_GRASS_LUSH);
        assert_eq!(tree.get_voxel(128, 65, 128), MAT_AIR);

        // Verify AABB collision query:
        assert!(tree.intersects_aabb_local(10, 60, 10, 20, 64, 20));
        assert!(!tree.intersects_aabb_local(10, 65, 10, 20, 80, 20));

        // Now carve a sphere of radius 8 voxels centered at (128, 64, 128):
        let destroyed = tree.carve_sphere(
            Vec3::new(128.0, 64.0, 128.0),
            8.0,
            true,
            &palette,
            |_x, _y, _z, _m| {},
        );
        assert!(destroyed > 500);
        assert_eq!(tree.get_voxel(128, 64, 128), MAT_AIR);
        assert_eq!(tree.get_voxel(128, 60, 128), MAT_AIR);
        assert_eq!(tree.get_voxel(10, 20, 30), MAT_STONE_DEEP);

        // Now place a glowing crystal sphere at (128, 80, 128):
        tree.place_sphere(
            Vec3::new(128.0, 80.0, 128.0),
            4.0,
            MAT_CRYSTAL_CYAN,
            &palette,
        );
        assert_eq!(tree.get_voxel(128, 80, 128), MAT_CRYSTAL_CYAN);
    }
}
