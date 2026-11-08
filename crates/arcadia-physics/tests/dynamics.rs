//! Tests for world dynamics, gravity falling, toppling, and player interactions.
//! Owns CPU unit verification of multi-body world simulation and bedrock collisions.

use arcadia_physics::{CharacterController, MovementInput, RigidVoxelBody};
mod common;
use arcadia_core::{ChunkTree64, MaterialPalette, MAT_AIR, MAT_STONE_DEEP};
use arcadia_world::WorldMap;
use glam::{IVec3, UVec3, Vec3};

#[test]
fn test_structural_island_detachment_and_falling_rigid_body() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // 1. Bedrock floor: y in 0..=4, over a 16x16 area centered at (32, 32)
    for z in 24..40 {
        for x in 24..40 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    // 2. Vertical stone pillar: 2x2 column at x in 31..33, z in 31..33, from y = 5 up to y = 24
    for y in 5..=24 {
        for z in 31..33 {
            for x in 31..33 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    // Verify initial state: pillar is solid and anchored
    assert_eq!(world.get_voxel(31, 20, 31), MAT_STONE_DEEP);
    assert_eq!(world.get_voxel(31, 8, 31), MAT_STONE_DEEP);
    assert_eq!(world.get_voxel(31, 2, 31), MAT_STONE_DEEP);

    // 3. Carve sphere through the base of the pillar at y = 8 voxels (0.5m)
    let center_m = Vec3::new(2.0, 0.5, 2.0);
    let radius_m = 0.22;
    let event = world.carve_sphere_meters(center_m, radius_m, false, 0);

    // The base of the pillar (around y in 5..10) was carved away
    assert_eq!(world.get_voxel(31, 8, 31), MAT_AIR);
    // Bedrock stump below y <= 4 remains in the static world
    assert_eq!(world.get_voxel(31, 2, 31), MAT_STONE_DEEP);

    // 4. Verify detached islands:
    // The upper pillar section (y in 11..=24, 2x2x14 = 56 voxels) must be detected as a detached island
    assert_eq!(
        event.detached_islands.len(),
        1,
        "Expected exactly 1 detached island from severed pillar"
    );
    let island_voxels = &event.detached_islands[0];
    assert_eq!(island_voxels.len(), 56);

    // And verify that the upper pillar section was removed from the static WorldMap!
    assert_eq!(
        world.get_voxel(31, 20, 31),
        MAT_AIR,
        "Detached island voxels must be cleared from the static WorldMap"
    );
    assert_eq!(world.get_voxel(32, 20, 32), MAT_AIR);

    // 5. Construct a RigidVoxelBody from the detached island
    let mut body = RigidVoxelBody::from_global_voxels(island_voxels, &palette, Some(center_m))
        .expect("INVARIANT: Detached island with non-empty voxels must construct RigidVoxelBody");

    assert_eq!(body.local_voxels.len(), 56);
    assert_eq!(body.dims_v, IVec3::new(2, 14, 2));
    assert!(!body.sleeping);

    let initial_pos = body.position;
    assert!(initial_pos.y > 0.8, "Initial CoM should be elevated");

    // 6. Step forward in time and verify it falls under gravity
    let dt = 0.016;
    for _ in 0..10 {
        body.step(dt, &world);
    }
    assert!(
        body.position.y < initial_pos.y,
        "Rigid body must fall downward under gravity"
    );
    assert!(
        body.linear_velocity.y < 0.0,
        "Linear velocity must be directed downward"
    );

    // 7. Step forward until it collides with the bedrock ground and settles
    let mut settled = false;
    for _ in 0..400 {
        body.step(0.02, &world);
        if body.sleeping {
            settled = true;
            break;
        }
    }

    assert!(
        settled,
        "Rigid body must settle and enter sleep state on ground"
    );
    assert!(
        body.position.y < initial_pos.y,
        "Settled position must be lower than detachment position"
    );
    assert!(
        body.position.y > 0.0,
        "Rigid body should not fall through the bedrock"
    );
    assert_eq!(body.linear_velocity, Vec3::ZERO);
    assert_eq!(body.angular_velocity, Vec3::ZERO);

    // 8. Test wake-up on digging underneath:
    // Clear bedrock beneath the resting body across its entire toppled footprint:
    let (aabb_min, aabb_max) = body.world_aabb_meters();
    let min_vx = (aabb_min.x * 16.0).floor() as i32 - 3;
    let max_vx = (aabb_max.x * 16.0).ceil() as i32 + 3;
    let min_vz = (aabb_min.z * 16.0).floor() as i32 - 3;
    let max_vz = (aabb_max.z * 16.0).ceil() as i32 + 3;
    let mut clear_coords = Vec::new();
    for vx in min_vx..=max_vx {
        for vz in min_vz..=max_vz {
            for y in 0..=8 {
                clear_coords.push(IVec3::new(vx, y, vz));
            }
        }
    }
    world.clear_voxels_global(&clear_coords);
    // Step once:
    body.step(0.016, &world);
    assert!(
        !body.sleeping,
        "Body must wake up when ground beneath it is removed"
    );
}

#[test]
fn test_player_rigid_body_collision_and_push() {
    let palette = MaterialPalette::default();
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![ChunkTree64::new_empty()], palette);

    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let mut body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create a rigid voxel body");
    body.position = Vec3::new(10.5, 1.0, 10.0);

    let mut player =
        CharacterController::new(Vec3::new(10.0, 1.0, 10.0), std::f32::consts::FRAC_PI_2, 0.0); // facing +X
    let mut bodies = vec![body];

    let input = MovementInput {
        move_dir: Vec3::new(0.0, 0.0, 1.0), // forward (+X)
        sprint: false,
        jump: false,
    };

    // Advance player over several substeps to approach and push into the body:
    for _ in 0..10 {
        player.update(0.05, input, &world, &mut bodies);
    }

    // Player must NOT walk through the body (half_width is 0.28, body is at 10.5):
    assert!(
        player.foot_pos.x < 10.45,
        "Player must be blocked by the rigid body"
    );
    // The rigid body must have received a push impulse along +X:
    assert!(
        bodies[0].linear_velocity.x > 0.0,
        "Rigid body must be pushed along +X by player"
    );
}

#[test]
fn test_severed_pillar_toppling_dynamics() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // 1. Bedrock floor: y in 0..=4
    for z in 24..40 {
        for x in 24..40 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    // 2. Tall pillar: 2x2 column from y = 5 up to y = 28 (24 voxels tall = 1.5m)
    for y in 5..=28 {
        for z in 31..33 {
            for x in 31..33 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    // 3. Sever the base of the pillar with a carve sphere centered at (1.95, 0.45, 2.0)
    let blast_center = Vec3::new(1.95, 0.45, 2.0);
    let event = world.carve_sphere_meters(blast_center, 0.22, false, 0);

    assert_eq!(event.detached_islands.len(), 1);
    let island_voxels = &event.detached_islands[0];

    let mut body = RigidVoxelBody::from_global_voxels(island_voxels, &palette, Some(blast_center))
        .expect("INVARIANT: Detached pillar must form RigidVoxelBody");

    // The blast at the base must have imparted a significant initial overturning angular velocity:
    let init_omega_len = body.angular_velocity.length();
    assert!(
        init_omega_len >= 0.5,
        "Severed pillar must receive an initial blast topple torque, got {init_omega_len}"
    );

    // Step simulation forward 1.0 second (50 steps of 0.02s) and check topple progression:
    let mut max_omega = init_omega_len;
    for _ in 0..50 {
        body.step(0.02, &world);
        max_omega = max_omega.max(body.angular_velocity.length());
    }

    // Due to gravitational torque about the base pivot, rotation must accelerate or maintain momentum:
    assert!(
        max_omega >= init_omega_len * 0.8,
        "Gravitational torque must drive toppling rather than suffocating rotation"
    );

    // Check that orientation has tilted significantly away from upright:
    let tilt_angle = body.orientation.to_axis_angle().1;
    assert!(
        tilt_angle > 0.35, // > ~20 degrees
        "Pillar must topple substantially away from upright, got tilt angle {tilt_angle}"
    );

    // Step forward until the pillar crashes down and settles into sleep:
    let mut settled = false;
    for _ in 0..300 {
        body.step(0.02, &world);
        if body.sleeping {
            settled = true;
            break;
        }
    }

    assert!(
        settled,
        "Toppled pillar must settle firmly into sleep on the ground"
    );
    assert_eq!(body.linear_velocity, Vec3::ZERO);
    assert_eq!(body.angular_velocity, Vec3::ZERO);
}
