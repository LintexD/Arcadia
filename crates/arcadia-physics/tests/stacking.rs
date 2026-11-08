//! Tests for Box3D multi-body stacking stability and convergence.
//! Owns CPU unit verification of multi-body stacking stability under gravity.

use arcadia_physics::RigidVoxelBody;
mod common;
use arcadia_core::{ChunkTree64, MaterialPalette, MAT_STONE_DEEP};
use arcadia_world::WorldMap;
use glam::{IVec3, Quat, UVec3, Vec3};

#[test]
fn test_box3d_two_body_stacking_stability() {
    let palette = MaterialPalette::default();
    let mut ground_voxels = Vec::new();
    for z in 0..32 {
        for x in 0..32 {
            for y in 0..=4 {
                ground_voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    let chunk = ChunkTree64::from_sparse_voxels(&ground_voxels, &palette);
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body_a = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body A");
    let mut body_b = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body B");

    body_a.position = Vec3::new(1.0, 0.35, 1.0);
    body_b.position = Vec3::new(1.0, 0.60, 1.0);

    let mut bodies = vec![body_a, body_b];

    for _ in 0..120 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
    }

    assert!(
        bodies[0].position.y > 0.35 && bodies[0].position.y < 0.45,
        "Body A must settle on bedrock, got y = {}",
        bodies[0].position.y
    );
    assert!(
        bodies[0].linear_velocity.y.abs() < 0.1,
        "Body A vertical speed must be < 0.1, got {}",
        bodies[0].linear_velocity.y
    );

    assert!(
        bodies[1].position.y > bodies[0].position.y,
        "Body B must remain above Body A"
    );

    let dy = bodies[1].position.y - bodies[0].position.y;
    assert!(
        (dy - 0.25).abs() < 0.015,
        "Distance between stacked bodies must be ~0.25m, got {}",
        dy
    );

    let dx = (bodies[1].position.x - bodies[0].position.x).abs();
    let dz = (bodies[1].position.z - bodies[0].position.z).abs();
    assert!(
        dx < 0.05,
        "Body B horizontal X drift relative to Body A must be < 0.05m, got {}",
        dx
    );
    assert!(
        dz < 0.05,
        "Body B horizontal Z drift relative to Body A must be < 0.05m, got {}",
        dz
    );

    assert!(
        bodies[0].angular_velocity.length() < 0.2,
        "Body A angular velocity must be < 0.2 rad/s, got {}",
        bodies[0].angular_velocity.length()
    );
    assert!(
        bodies[1].angular_velocity.length() < 0.2,
        "Body B angular velocity must be < 0.2 rad/s, got {}",
        bodies[1].angular_velocity.length()
    );
}

#[test]
fn test_box3d_three_body_stacking_convergence() {
    let palette = MaterialPalette::default();
    let mut ground_voxels = Vec::new();
    for z in 0..32 {
        for x in 0..32 {
            for y in 0..=4 {
                ground_voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    let chunk = ChunkTree64::from_sparse_voxels(&ground_voxels, &palette);
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body_a = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body A");
    let mut body_b = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body B");
    let mut body_c = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body C");

    body_a.position = Vec3::new(1.0, 0.35, 1.0);
    body_b.position = Vec3::new(1.0, 0.60, 1.0);
    body_c.position = Vec3::new(1.0, 0.85, 1.0);

    let mut bodies = vec![body_a, body_b, body_c];

    for _ in 0..120 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
    }

    // Assertions:
    // 1. All 3 bodies maintain their vertical order: bodies[2].position.y > bodies[1].position.y > bodies[0].position.y
    assert!(
        bodies[2].position.y > bodies[1].position.y,
        "Body C must remain above Body B"
    );
    assert!(
        bodies[1].position.y > bodies[0].position.y,
        "Body B must remain above Body A"
    );

    // 2. Distance between adjacent layers is ~0.25m (no penetration collapse)
    let dy_ab = bodies[1].position.y - bodies[0].position.y;
    let dy_bc = bodies[2].position.y - bodies[1].position.y;
    assert!(
        (dy_ab - 0.25).abs() < 0.015,
        "Distance between layer A and B must be ~0.25m, got {}",
        dy_ab
    );
    assert!(
        (dy_bc - 0.25).abs() < 0.015,
        "Distance between layer B and C must be ~0.25m, got {}",
        dy_bc
    );

    // 3. The top box remains upright and does not fall or glitch through the middle box
    assert!(
        bodies[2].linear_velocity.y.abs() < 0.1,
        "Body C vertical speed must be < 0.1, got {}",
        bodies[2].linear_velocity.y
    );
    assert!(
        bodies[2].angular_velocity.length() < 0.2,
        "Body C angular velocity must be < 0.2 rad/s, got {}",
        bodies[2].angular_velocity.length()
    );
}

#[test]
fn test_box3d_rotated_box_stacking_stability() {
    let palette = MaterialPalette::default();
    let mut ground_voxels = Vec::new();
    for z in 0..32 {
        for x in 0..32 {
            for y in 0..=4 {
                ground_voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    let chunk = ChunkTree64::from_sparse_voxels(&ground_voxels, &palette);
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body_a = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body A");
    let mut body_b = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body B");

    body_a.position = Vec3::new(1.0, 0.35, 1.0);
    body_b.position = Vec3::new(1.0, 0.60, 1.0);
    body_b.orientation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);

    let mut bodies = vec![body_a, body_b];

    for _ in 0..90 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
    }

    // Assertions:
    // 1. Box B rests stably on top of Box A without penetrating or launching
    assert!(
        bodies[1].position.y > bodies[0].position.y,
        "Body B must remain above Body A"
    );

    let dy = bodies[1].position.y - bodies[0].position.y;
    assert!(
        (dy - 0.25).abs() < 0.015,
        "Distance between stacked bodies must be ~0.25m, got {}",
        dy
    );

    assert!(
        bodies[1].linear_velocity.y.abs() < 0.1,
        "Body B vertical speed must be < 0.1, got {}",
        bodies[1].linear_velocity.y
    );
    assert!(
        bodies[1].angular_velocity.length() < 0.2,
        "Body B angular velocity must be < 0.2 rad/s, got {}",
        bodies[1].angular_velocity.length()
    );

    let dx = (bodies[1].position.x - bodies[0].position.x).abs();
    let dz = (bodies[1].position.z - bodies[0].position.z).abs();
    assert!(
        dx < 0.05,
        "Body B horizontal X drift relative to Body A must be < 0.05m, got {}",
        dx
    );
    assert!(
        dz < 0.05,
        "Body B horizontal Z drift relative to Body A must be < 0.05m, got {}",
        dz
    );
}
