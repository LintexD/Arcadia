//! Common test fixtures, mock helpers, and body/world builders for arcadia-physics integration tests.
//! Owns shared CPU test state setup to eliminate duplicated test boilerplate.

#![allow(dead_code)]

use arcadia_core::{ChunkTree64, MaterialPalette, MAT_STONE_DEEP};
use arcadia_physics::RigidVoxelBody;
use arcadia_world::WorldMap;
use glam::{IVec3, UVec3};

/// Generates a dense cubic cluster of stone voxels of dimension `size x size x size`.
pub fn make_test_voxels_cube(size: i32) -> Vec<(IVec3, u8)> {
    let mut voxels = Vec::with_capacity((size * size * size) as usize);
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    voxels
}

/// Constructs a valid dense stone rigid body of `size x size x size` voxels at origin.
pub fn make_test_body(size: i32) -> RigidVoxelBody {
    let palette = MaterialPalette::default();
    let voxels = make_test_voxels_cube(size);
    RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels must construct test RigidVoxelBody")
}

/// Constructs a 4x4x4 stone rigid body at origin (standard test body).
pub fn make_standard_test_body() -> RigidVoxelBody {
    make_test_body(4)
}

/// Builds a 1x1x1 chunk `WorldMap` from a list of solid `(x, y, z)` stone coordinates.
pub fn make_test_world(solid_coords: &[(i32, i32, i32)]) -> WorldMap {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::with_capacity(solid_coords.len());
    for &(x, y, z) in solid_coords {
        voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
    }
    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette)
}

/// Builds a flat stone bedrock ground in `WorldMap` of size `size_xz x height_y x size_xz`.
pub fn make_ground_world(size_xz: u32, height_y: u32) -> WorldMap {
    let palette = MaterialPalette::default();
    let mut ground_voxels = Vec::new();
    for z in 0..size_xz {
        for x in 0..size_xz {
            for y in 0..=height_y {
                ground_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
        }
    }
    let ground_chunk = ChunkTree64::from_sparse_voxels(&ground_voxels, &palette);
    WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![ground_chunk], palette)
}
