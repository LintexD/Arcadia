//! Tests for dynamic rigid body raycasting, carving, splitting, and voxel placement.
//! Owns CPU unit verification of in-flight voxel operations and component partitioning.

use arcadia_core::{
    ChunkTree64, MaterialPalette, MAT_CRYSTAL_CYAN, MAT_EMBER_GLOW, MAT_SCORCHED_ROCK,
    MAT_STONE_DEEP, VOXEL_SIZE,
};
use arcadia_physics::RigidVoxelBody;
mod common;
use arcadia_world::WorldMap;
use glam::{IVec3, Quat, UVec3, Vec3};

#[test]
fn test_emissive_rigid_body_light_injection() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // 2x2x2 cluster of Glowing Cyan Crystals (MAT_CRYSTAL_CYAN = 19):
    for z in 10..12 {
        for y in 10..12 {
            for x in 10..12 {
                voxels.push((IVec3::new(x, y, z), MAT_CRYSTAL_CYAN));
            }
        }
    }

    let body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels must create RigidVoxelBody");

    assert!(
        body.emissive_light.is_some(),
        "Rigid body constructed with crystals must detect emissive light"
    );

    let light = body
        .world_emissive_light()
        .expect("INVARIANT: Emissive body must generate world point light");

    // Light xyz must be centered near world position (10.5 * 0.0625 = ~0.6875m)
    assert!((light[0] - 0.6875).abs() < 0.05);
    assert!((light[1] - 0.6875).abs() < 0.05);
    assert!((light[2] - 0.6875).abs() < 0.05);
    // Light radius/intensity must be positive:
    assert!(
        light[3] > 1.0,
        "Light radius must be > 1.0m, got {}",
        light[3]
    );
}

#[test]
fn test_dynamic_rigid_body_raycast_and_carve() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // 4x4x4 stone block at global (10..14, 10..14, 10..14):
    for z in 10..14 {
        for y in 10..14 {
            for x in 10..14 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels must create RigidVoxelBody");

    assert_eq!(body.local_voxels.len(), 64);

    // Raycast from outside towards the center of the block:
    let ray_origin = body.position + Vec3::new(2.0, 0.0, 0.0);
    let ray_dir = Vec3::new(-1.0, 0.0, 0.0);
    let hit = body.raycast_meters(ray_origin, ray_dir, 10.0);
    assert!(hit.is_some(), "Raycast must hit the dynamic rigid body");
    let h = hit.expect("INVARIANT: Hit verified");
    assert!(
        (h.normal.x - 1.0).abs() < 1e-3,
        "Normal must point towards +X"
    );

    // Carve sphere into one corner of the block with scorch_rim enabled:
    let corner_pos = body.position + Vec3::new(0.08, 0.08, 0.08);
    let outcome = body.carve_sphere_meters(corner_pos, 0.12, true, &palette);
    assert!(!outcome.destroyed, "Partial carve should not destroy body");
    assert!(
        body.local_voxels.len() < 64,
        "Voxel count must decrease after carving, got {}",
        body.local_voxels.len()
    );
    assert!(
        !outcome.spawned_debris.is_empty(),
        "Debris must be spawned by carving dynamic body"
    );
    // Verify scorch rim applied amber glow or scorched rock to adjacent voxels:
    let has_scorch = body
        .local_voxels
        .iter()
        .any(|&(_, _, _, m)| m == MAT_EMBER_GLOW || m == MAT_SCORCHED_ROCK);
    assert!(
        has_scorch,
        "Scorch rim must apply amber embers or dark rock to carved body"
    );
}

#[test]
fn test_dynamic_rigid_body_carve_splits_into_separate_bodies() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // 16x2x2 stone beam along X (x in 0..16, y in 0..2, z in 0..2):
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..16 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels must create RigidVoxelBody");
    assert_eq!(body.local_voxels.len(), 64);

    // Orient the beam at a 40-degree angle to verify zero angular distortion on splitting:
    let test_orientation = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(0.35);
    body.orientation = test_orientation;
    body.prev_orientation = test_orientation;

    // Carve a sphere right through the middle of the beam (around x = 8 voxels = 0.5m):
    let middle_world = body.position; // Center of Mass is at the midpoint of the beam
    let outcome = body.carve_sphere_meters(middle_world, 0.15, false, &palette);

    assert!(!outcome.destroyed);
    assert_eq!(
        outcome.split_bodies.len(),
        1,
        "Carving beam in half must produce 1 newly detached split rigid body"
    );
    let split_body = &outcome.split_bodies[0];
    assert!(!split_body.local_voxels.is_empty());
    assert!(!body.local_voxels.is_empty());
    assert_eq!(
        body.local_voxels.len() + split_body.local_voxels.len(),
        64 - outcome.spawned_debris.len()
    );

    // Verify zero angular distortion: split body inherits exact orientation without double rotation:
    assert!(
        (split_body.orientation.dot(test_orientation).abs() - 1.0).abs() < 1e-4,
        "Split body must preserve parent orientation without double rotation"
    );
    assert!(
        (body.orientation.dot(test_orientation).abs() - 1.0).abs() < 1e-4,
        "Parent body must preserve orientation without distortion"
    );

    // Verify spatial non-overlapping separation: both pieces must maintain distinct world positions:
    let separation_dist = (split_body.position - body.position).length();
    assert!(
        separation_dist > 0.15,
        "Split pieces must be spatially separated, got distance {}",
        separation_dist
    );
}

#[test]
fn test_dynamic_rigid_body_place_voxels() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // 4x4x4 stone block:
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels must create RigidVoxelBody");
    let initial_voxels = body.local_voxels.len();

    // Place a sphere of glowing crystal on top of the block:
    let top_world = body.position + Vec3::new(0.0, 0.15, 0.0);
    let success = body.place_sphere_meters(top_world, 0.10, MAT_CRYSTAL_CYAN, &palette);
    assert!(success, "Placing voxels on rigid body must succeed");
    assert!(
        body.local_voxels.len() > initial_voxels,
        "Rigid body must have more voxels after placement"
    );
    assert!(
        body.local_voxels
            .iter()
            .any(|&(_, _, _, m)| m == MAT_CRYSTAL_CYAN),
        "Placed crystal voxels must exist in the body"
    );
    assert!(
        body.emissive_light.is_some(),
        "Body must become emissive after adding crystals"
    );
}

#[test]
fn test_dynamic_rigid_body_carve_no_skyward_launch() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // 4x16x4 stone column:
    for z in 0..4 {
        for y in 0..16 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid column voxels");
    body.linear_velocity = Vec3::ZERO;
    let pos_before = body.position;

    // Carve sphere near base of the column from below/front:
    let carve_pos = body.position - Vec3::new(0.0, 0.4, 0.1);
    let outcome = body.carve_sphere_meters(carve_pos, 0.18, false, &palette);

    assert!(!outcome.destroyed);
    // Upward impulse must be strictly clamped so resting objects never launch into the air:
    assert!(
        body.linear_velocity.y <= 0.4,
        "Upward velocity must be <= 0.4 m/s, got {}",
        body.linear_velocity.y
    );
    // Position change must match exact CoM delta without any double shift teleportation:
    let pos_delta = (body.position - pos_before).length();
    assert!(
        pos_delta < 0.25,
        "Body must not teleport after carving, delta was {}",
        pos_delta
    );
}

#[test]
fn test_rigid_body_rest_time_and_to_world_voxels() {
    let palette = MaterialPalette::default();
    let mut original_voxels = Vec::new();
    for z in 10..12 {
        for y in 20..22 {
            for x in 30..32 {
                original_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body = RigidVoxelBody::from_global_voxels(&original_voxels, &palette, None)
        .expect("INVARIANT: Valid voxels");
    assert_eq!(body.rest_time, 0.0);

    // When stationary with zero velocity, rest_time accumulates:
    body.linear_velocity = Vec3::ZERO;
    body.angular_velocity = Vec3::ZERO;
    body.sleeping = true;

    let dummy_chunk = ChunkTree64::from_sparse_voxels(&[], &palette);
    let dummy_world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![dummy_chunk], palette);
    body.step(0.016, &dummy_world);
    assert!(
        body.rest_time > 0.015,
        "Rest time must accumulate when resting"
    );

    // Waking up preserves rest_time; reset_rest_timer resets it to 0.0:
    body.wake_up();
    assert!(
        body.rest_time > 0.015,
        "wake_up must preserve accumulated rest time"
    );
    body.reset_rest_timer();
    assert_eq!(
        body.rest_time, 0.0,
        "reset_rest_timer must reset rest_time to zero"
    );

    // Reconstruct world voxels: without rotation, coordinates must match original set:
    let reconstructed = body.to_world_voxels();
    assert_eq!(reconstructed.len(), original_voxels.len());
    for (gv, mat) in reconstructed {
        assert!(
            original_voxels.contains(&(gv, mat)),
            "Reconstructed world voxel {:?} must exist in original set",
            gv
        );
    }
}

#[test]
fn test_rotated_rigid_body_to_world_voxels_flat_grid() {
    let palette = MaterialPalette::default();
    let mut original_voxels = Vec::new();
    for z in 10..13 {
        for y in 20..23 {
            for x in 30..33 {
                original_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body = RigidVoxelBody::from_global_voxels(&original_voxels, &palette, None)
        .expect("INVARIANT: Valid voxels");

    // Rotate 45 degrees around Y axis:
    body.orientation = Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4);

    let reconstructed = body.to_world_voxels();
    assert!(
        !reconstructed.is_empty(),
        "Rotated body must rasterize into non-empty world voxels"
    );

    // Conservative rasterization must produce at least the original volume of voxels:
    assert!(
        reconstructed.len() >= original_voxels.len(),
        "Conservative rasterization must not lose volume (got {} voxels, expected >= {})",
        reconstructed.len(),
        original_voxels.len()
    );

    // Verify all emitted voxels are unique and match the body material:
    let mut unique_coords = std::collections::HashSet::new();
    for (gv, mat) in &reconstructed {
        assert_eq!(*mat, MAT_STONE_DEEP, "Voxel material must be preserved");
        assert!(
            unique_coords.insert(*gv),
            "Emitted world voxels must be unique (duplicate: {:?})",
            gv
        );
    }

    // Verify that every voxel is within the world AABB (+/- 2 voxel margin):
    let (aabb_min_m, aabb_max_m) = body.world_aabb_meters();
    let vpm = arcadia_core::VOXELS_PER_METER as f32;
    let min_bound = (aabb_min_m * vpm).floor().as_ivec3() - IVec3::splat(2);
    let max_bound = (aabb_max_m * vpm).ceil().as_ivec3() + IVec3::splat(2);

    for gv in unique_coords {
        assert!(
            gv.x >= min_bound.x
                && gv.x <= max_bound.x
                && gv.y >= min_bound.y
                && gv.y <= max_bound.y
                && gv.z >= min_bound.z
                && gv.z <= max_bound.z,
            "Rasterized voxel {:?} out of world AABB bounds [min: {:?}, max: {:?}]",
            gv,
            min_bound,
            max_bound
        );
    }
}

#[test]
fn test_unmodified_body_no_blast_impulse_near_carve() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels must create RigidVoxelBody");
    body.linear_velocity = Vec3::ZERO;
    body.angular_velocity = Vec3::ZERO;
    body.sleeping = true;
    body.rest_time = 5.0;

    // Carve center is outside body voxels (body AABB ~ [0, 0.25]), e.g., at (0.5, 0.125, 0.125):
    // Radius = 0.15m. Distance from body boundary (~0.25) to carve sphere (0.5 - 0.15 = 0.35) > 0.
    // So 0 voxels are modified.
    let carve_pos = body.position + Vec3::new(0.5, 0.0, 0.0);
    let outcome = body.carve_sphere_meters(carve_pos, 0.15, false, &palette);

    assert!(!outcome.modified, "Carve sphere must not touch body voxels");
    assert_eq!(
        body.linear_velocity,
        Vec3::ZERO,
        "Unmodified body must not gain linear velocity"
    );
    // Since carve_pos is within radius (0.15) + 0.5 = 0.65 of body.position, it should wake up:
    assert!(!body.sleeping, "Body within proximity should wake up");
    // But rest time should NOT be reset:
    assert_eq!(
        body.rest_time, 5.0,
        "Unmodified body rest timer must not be reset"
    );

    // Now test a carve completely outside proximity (> radius + 0.5):
    body.sleeping = true;
    let far_carve_pos = body.position + Vec3::new(2.0, 0.0, 0.0);
    let outcome_far = body.carve_sphere_meters(far_carve_pos, 0.15, false, &palette);
    assert!(!outcome_far.modified);
    assert!(body.sleeping, "Body outside proximity must remain sleeping");
    assert_eq!(body.linear_velocity, Vec3::ZERO);
}

#[test]
fn test_carved_resting_body_clamped_blast_impulse_and_sleep_settling() {
    let palette = MaterialPalette::default();

    // Create flat stone bedrock ground in WorldMap:
    let mut ground_voxels = Vec::new();
    for z in 0..32 {
        for x in 0..32 {
            for y in 0..=4 {
                ground_voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    let ground_chunk = ChunkTree64::from_sparse_voxels(&ground_voxels, &palette);
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![ground_chunk], palette);

    // Create a 4x12x4 stone pillar resting on the ground (bedrock top is at y = 5 * 0.0625 = 0.3125m):
    let mut pillar_voxels = Vec::new();
    for z in 10..14 {
        for y in 5..17 {
            for x in 10..14 {
                pillar_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body = RigidVoxelBody::from_global_voxels(&pillar_voxels, &palette, None)
        .expect("INVARIANT: Valid voxels must create pillar");

    // Let the body settle and sleep on the ground:
    for _ in 0..60 {
        body.step(0.016, &world);
    }
    assert!(body.sleeping, "Pillar must settle and sleep on the ground");

    // Carve a small chunk from the top corner of the pillar:
    let top_corner = body.position + Vec3::new(0.08, 0.30, 0.08);
    let outcome = body.carve_sphere_meters(top_corner, 0.08, false, &palette);

    assert!(outcome.modified, "Carve must modify pillar voxels");
    assert!(
        body.linear_velocity.length() <= 0.10 + 1e-4,
        "Carved body impulse must be clamped to <= 0.1 m/s, got {}",
        body.linear_velocity.length()
    );

    // Step physics: body must dissipate residual motion and settle back into sleep cleanly:
    for _ in 0..40 {
        body.step(0.016, &world);
    }

    assert!(
        body.sleeping,
        "Carved body resting on ground must settle into sleep mode without jiggling"
    );
    assert_eq!(
        body.linear_velocity,
        Vec3::ZERO,
        "Sleeping body must have zero linear velocity"
    );
    assert_eq!(
        body.angular_velocity,
        Vec3::ZERO,
        "Sleeping body must have zero angular velocity"
    );
}

#[test]
fn test_carve_sever_small_island_under_threshold_becomes_debris() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // INVARIANT: Construct 4x4x4 main block (64 voxels), single bridge voxel at (4, 0, 0),
    // and 2x2x2 small island (8 voxels) along +X.
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    voxels.push((IVec3::new(4, 0, 0), MAT_STONE_DEEP));
    for z in 0..2 {
        for y in 0..2 {
            for x in 5..7 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels must construct RigidVoxelBody");
    assert_eq!(body.local_voxels.len(), 73);

    // INVARIANT: Target center of bridge voxel (4, 0, 0) with radius 0.035m (< VOXEL_SIZE 0.0625m)
    // to sever the connection without carving adjacent voxels.
    let bridge_world = body.position + Vec3::new(4.5, 0.5, 0.5) * VOXEL_SIZE - body.com_local_m;
    let outcome = body.carve_sphere_meters(bridge_world, 0.035, false, &palette);

    assert!(outcome.modified, "Carve must sever bridge voxel");
    assert!(
        !outcome.destroyed,
        "Main body (64 voxels) must not be destroyed"
    );
    assert_eq!(
        body.local_voxels.len(),
        64,
        "Main body must retain exactly 64 voxels"
    );
    assert_eq!(
        outcome.split_bodies.len(),
        0,
        "Severed 8-voxel island must NOT spawn a RigidVoxelBody (threshold <= 8 voxels)"
    );
    assert_eq!(
        outcome.spawned_debris.len(),
        9,
        "Bridge voxel (1) + severed island (8) must all convert into 9 debris particles"
    );
}

#[test]
fn test_carve_sever_island_above_threshold_spawns_split_body() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // INVARIANT: Construct 4x4x4 main block (64 voxels), bridge voxel (4, 0, 0),
    // and 9-voxel island (> SMALL_ISLAND_PARTICLE_MAX_VOXELS).
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    voxels.push((IVec3::new(4, 0, 0), MAT_STONE_DEEP));
    for z in 0..2 {
        for y in 0..2 {
            for x in 5..7 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    voxels.push((IVec3::new(5, 2, 0), MAT_STONE_DEEP));

    let mut body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels must construct RigidVoxelBody");
    assert_eq!(body.local_voxels.len(), 74);

    let bridge_world = body.position + Vec3::new(4.5, 0.5, 0.5) * VOXEL_SIZE - body.com_local_m;
    let outcome = body.carve_sphere_meters(bridge_world, 0.035, false, &palette);

    assert!(outcome.modified, "Carve must sever bridge voxel");
    assert!(!outcome.destroyed, "Main body must not be destroyed");
    assert_eq!(body.local_voxels.len(), 64);
    assert_eq!(
        outcome.split_bodies.len(),
        1,
        "Severed 9-voxel island (> 8 voxels) MUST spawn a new independent RigidVoxelBody"
    );
    assert_eq!(
        outcome.split_bodies[0].local_voxels.len(),
        9,
        "Spawned split body must contain all 9 island voxels"
    );
    assert_eq!(
        outcome.spawned_debris.len(),
        1,
        "Only the severed crater bridge voxel should become debris"
    );
}

#[test]
fn test_carve_surviving_remnant_under_threshold_obliterates_into_debris() {
    let palette = MaterialPalette::default();

    // INVARIANT: 1. Single surviving voxel case (2 voxels total: (0, 0, 0) and (1, 0, 0)).
    let voxels_2 = vec![
        (IVec3::new(0, 0, 0), MAT_STONE_DEEP),
        (IVec3::new(1, 0, 0), MAT_STONE_DEEP),
    ];
    let mut body_2 = RigidVoxelBody::from_global_voxels(&voxels_2, &palette, None)
        .expect("INVARIANT: Valid voxels must construct RigidVoxelBody");
    assert_eq!(body_2.local_voxels.len(), 2);

    // INVARIANT: Target voxel (0, 0, 0) directly with small radius.
    let carve_pt_0 = body_2.position + Vec3::new(0.5, 0.5, 0.5) * VOXEL_SIZE - body_2.com_local_m;
    let outcome_2 = body_2.carve_sphere_meters(carve_pt_0, 0.035, false, &palette);

    assert!(outcome_2.modified, "Carve must destroy targeted voxel");
    assert!(
        outcome_2.destroyed,
        "Rigid body must be destroyed when 1 single voxel survives (threshold <= 8)"
    );
    assert!(
        outcome_2.split_bodies.is_empty(),
        "No split bodies should be spawned when surviving remnant is obliterated"
    );
    assert_eq!(
        outcome_2.spawned_debris.len(),
        2,
        "Both carved voxel (1) and surviving single voxel (1) must be emitted in spawned_debris"
    );

    // INVARIANT: 2. Multi-voxel surviving remnant <= 8 case (10 voxels: 2 in blast, 8 surviving).
    let mut voxels_10 = Vec::new();
    for x in 0..10 {
        voxels_10.push((IVec3::new(x, 0, 0), MAT_STONE_DEEP));
    }
    let mut body_10 = RigidVoxelBody::from_global_voxels(&voxels_10, &palette, None)
        .expect("INVARIANT: Valid voxels must construct RigidVoxelBody");
    assert_eq!(body_10.local_voxels.len(), 10);

    // INVARIANT: Target x=0 and x=1 with blast radius covering both.
    let carve_pt_10 =
        body_10.position + Vec3::new(1.0, 0.5, 0.5) * VOXEL_SIZE - body_10.com_local_m;
    let outcome_10 = body_10.carve_sphere_meters(carve_pt_10, 0.07, false, &palette);

    assert!(outcome_10.modified, "Carve must destroy targeted voxels");
    assert!(
        outcome_10.destroyed,
        "Rigid body must be destroyed when surviving remnant <= 8 voxels"
    );
    assert!(
        outcome_10.split_bodies.is_empty(),
        "No split bodies should be spawned"
    );
    assert_eq!(
        outcome_10.spawned_debris.len(),
        10,
        "All 10 voxels (carved + surviving <= 8 voxels) must be emitted in spawned_debris"
    );
}

#[test]
fn test_find_connected_components_brick_bitboards_multi_brick_and_diagonals() {
    // INVARIANT: 1. Multi-brick connected 3D shape spanning 3 bricks along X.
    let mut multi_brick_voxels = Vec::new();
    for x in 0..10 {
        multi_brick_voxels.push((x, 0, 0, MAT_STONE_DEEP));
    }
    let comps = RigidVoxelBody::find_connected_components(&multi_brick_voxels);
    assert_eq!(
        comps.len(),
        1,
        "Continuous line across bricks must form exactly 1 component"
    );
    assert_eq!(comps[0].len(), 10, "Component must contain all 10 voxels");

    // INVARIANT: 2. Parallel lines separated by air gap.
    let mut parallel_voxels = Vec::new();
    for x in 0..10 {
        parallel_voxels.push((x, 0, 0, MAT_STONE_DEEP));
        parallel_voxels.push((x, 2, 0, MAT_STONE_DEEP));
    }
    let comps_parallel = RigidVoxelBody::find_connected_components(&parallel_voxels);
    assert_eq!(
        comps_parallel.len(),
        2,
        "Parallel lines separated by gap must form 2 distinct components"
    );
    assert_eq!(comps_parallel[0].len(), 10);
    assert_eq!(comps_parallel[1].len(), 10);

    // INVARIANT: 3. Diagonal corner/edge touching voxels (must NOT be 6-connected).
    let diagonal_intra = vec![
        (0, 0, 0, MAT_STONE_DEEP),
        (1, 1, 0, MAT_STONE_DEEP),
        (2, 2, 2, MAT_STONE_DEEP),
    ];
    let comps_diag = RigidVoxelBody::find_connected_components(&diagonal_intra);
    assert_eq!(
        comps_diag.len(),
        3,
        "Intra-brick diagonal edge/corner touching voxels must NOT be 6-connected"
    );

    let diagonal_inter = vec![(3, 3, 3, MAT_STONE_DEEP), (4, 4, 4, MAT_STONE_DEEP)];
    let comps_inter_diag = RigidVoxelBody::find_connected_components(&diagonal_inter);
    assert_eq!(
        comps_inter_diag.len(),
        2,
        "Inter-brick corner-touching voxels must NOT be 6-connected"
    );

    // INVARIANT: 4. Face-touching across brick boundaries in all 3 axes.
    let face_x = vec![(3, 0, 0, MAT_STONE_DEEP), (4, 0, 0, MAT_STONE_DEEP)];
    assert_eq!(RigidVoxelBody::find_connected_components(&face_x).len(), 1);

    let face_y = vec![(0, 3, 0, MAT_STONE_DEEP), (0, 4, 0, MAT_STONE_DEEP)];
    assert_eq!(RigidVoxelBody::find_connected_components(&face_y).len(), 1);

    let face_z = vec![(0, 0, 3, MAT_STONE_DEEP), (0, 0, 4, MAT_STONE_DEEP)];
    assert_eq!(RigidVoxelBody::find_connected_components(&face_z).len(), 1);

    // INVARIANT: 5. 3D loop crossing multiple bricks and returning to start.
    let mut loop_voxels = Vec::new();
    for x in 2..=5 {
        loop_voxels.push((x, 0, 0, MAT_STONE_DEEP));
        loop_voxels.push((x, 5, 0, MAT_STONE_DEEP));
    }
    for y in 1..=4 {
        loop_voxels.push((2, y, 0, MAT_STONE_DEEP));
        loop_voxels.push((5, y, 0, MAT_STONE_DEEP));
    }
    let loop_comps = RigidVoxelBody::find_connected_components(&loop_voxels);
    assert_eq!(
        loop_comps.len(),
        1,
        "Continuous loop traversing 4 bricks must form 1 component"
    );
    assert_eq!(loop_comps[0].len(), loop_voxels.len());
}

#[test]
fn test_carve_scorch_rim_only_fast_path() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let mut body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels must construct RigidVoxelBody");
    let initial_count = body.local_voxels.len();

    // INVARIANT: Blast center outside the body so r_sq (0.12m) is further than distance to nearest voxel (0.135m).
    let blast_pos = body.position + Vec3::new(-0.26, 0.0, 0.0);
    let outcome = body.carve_sphere_meters(blast_pos, 0.12, true, &palette);

    assert!(outcome.modified, "Scorch rim must recolor surface voxels");
    assert!(
        !outcome.destroyed,
        "Zero voxels destroyed, body must survive"
    );
    assert_eq!(
        body.local_voxels.len(),
        initial_count,
        "No voxels should be deleted"
    );
    assert!(outcome.split_bodies.is_empty(), "No split bodies created");
    assert!(
        outcome.spawned_debris.is_empty(),
        "No debris spawned when zero voxels destroyed"
    );
}
