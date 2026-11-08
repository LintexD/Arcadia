//! Tests for discrete voxel mass, center of mass, and moment of inertia tensor.
//! Owns CPU unit verification of discrete CoM calculation and parallel-axis correction.

use arcadia_physics::RigidVoxelBody;
mod common;
use arcadia_core::{MaterialPalette, MAT_STONE_DEEP, VOXEL_SIZE};
use glam::{IVec3, Vec3};

#[test]
fn test_rigid_voxel_inertia_and_com() {
    let palette = MaterialPalette::default();
    // Build a symmetric 2x4x2 pillar of 16 stone voxels at global (100..102, 50..54, 100..102):
    let mut voxels = Vec::new();
    for z in 100..102 {
        for y in 50..54 {
            for x in 100..102 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create a rigid voxel body");

    assert_eq!(body.local_voxels.len(), 16);
    assert_eq!(body.dims_v, IVec3::new(2, 4, 2));

    // Expected local CoM is at the exact geometric center of the 2x4x2 voxel box:
    // (1.0, 2.0, 1.0) * VOXEL_SIZE = (0.0625, 0.125, 0.0625) m
    let expected_com = Vec3::new(1.0, 2.0, 1.0) * VOXEL_SIZE;
    assert!((body.com_local_m - expected_com).length() < 1e-5);

    // Because the pillar is taller along Y (4 voxels) than X and Z (2 voxels),
    // I_xx and I_zz must be strictly greater than I_yy, and off-diagonal products of inertia must be 0:
    let i_xx = body.inertia_body.x_axis.x;
    let i_yy = body.inertia_body.y_axis.y;
    let i_zz = body.inertia_body.z_axis.z;
    assert!(i_xx > i_yy);
    assert!((i_xx - i_zz).abs() < 1e-6);
    assert!(body.inertia_body.x_axis.y.abs() < 1e-6);
}
