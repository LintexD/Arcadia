//! Full-shape compound voxel collision, concave pass-through, carving stability, and shallow-pool tests.
//! Owns CPU regression tests for `CompoundCollider`, non-convex pillar contact, character AABB queries, and wet-impact damping.

use arcadia_physics::{CharacterController, RigidVoxelBody, FACE_NEG_X};
mod common;
use arcadia_core::{ChunkTree64, MaterialPalette, MAT_STONE_DEEP, MAT_WATER_CRYSTAL, VOXEL_SIZE};
use arcadia_world::WorldMap;
use glam::{IVec3, Mat3, Quat, UVec3, Vec3};

#[test]
fn test_compound_greedy_partition_exact_volume_and_internal_seams() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // Vertical arm: 4x12x4 = 192 voxels
    for z in 0..4 {
        for y in 0..12 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    // Horizontal base extension: 8x4x4 = 128 voxels (total = 320 voxels)
    for z in 0..4 {
        for y in 0..4 {
            for x in 4..12 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: L-shaped voxels must build RigidVoxelBody");

    let total_sub_volume: u32 = body
        .collider
        .sub_boxes
        .iter()
        .map(|s| s.volume_voxels())
        .sum();
    assert_eq!(
        total_sub_volume, 320,
        "Greedy 3D box partition must cover exact occupied voxel volume with zero air or overlap"
    );
    assert_eq!(
        body.collider.sub_boxes.len(),
        2,
        "L-shape must partition into exactly 2 greedy sub-boxes"
    );
    assert!(!body.collider.bvh_nodes.is_empty());

    // Internal seam check: whichever sub-box starts at min_v[0] == 4 sits flush against solid voxels at x == 3,
    // so its -X face must NOT be marked exposed!
    for sub in &body.collider.sub_boxes {
        if sub.min_v[0] == 4 && sub.max_v[1] <= 4 {
            assert_eq!(
                sub.exposed_faces & FACE_NEG_X,
                0,
                "Internal greedy-box seam at x = 4 must not expose -X face"
            );
        }
    }
}

#[test]
fn test_orb_topped_pillar_settles_on_fallen_shaft_without_floating() {
    let palette = MaterialPalette::default();

    // Build flat courtyard floor + raised plinth (vx in 20..=44, vz in 26..=38, vy in 0..=12, top at 13 * VOXEL_SIZE = 0.8125m):
    let mut world_voxels = Vec::new();
    for z in 0..64u32 {
        for x in 0..64u32 {
            for y in 0..=4u32 {
                world_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
        }
    }
    for z in 26..=38u32 {
        for x in 20..=44u32 {
            for y in 5..=12u32 {
                world_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
        }
    }
    let chunk = ChunkTree64::from_sparse_voxels(&world_voxels, &palette);
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    // Body A: horizontal fallen pillar shaft (20x6x6 voxels = 1.25m x 0.375m x 0.375m) resting atop the plinth:
    let mut fallen_voxels = Vec::new();
    for z in 0..6 {
        for y in 0..6 {
            for x in 0..20 {
                fallen_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let mut fallen_pillar = RigidVoxelBody::from_global_voxels(&fallen_voxels, &palette, None)
        .expect("INVARIANT: Fallen shaft must construct");
    // Plinth top is at 0.8125m; half-height of 6-voxel shaft is 3 * 0.0625 = 0.1875m -> center Y = 1.000m, top Y = 1.1875m.
    fallen_pillar.position = Vec3::new(2.0, 1.0, 2.0);

    // Body B: narrow 4x28x4 fluted shaft (width 0.25m, half-width 0.125m) flanked by wide 12x8x12 capitals
    // (width 0.75m, half-width 0.375m — 3x wider than the shaft!) plus tilted single-capital check,
    // reproducing the screenshot where wide capitals overhang in empty air while the narrow central shaft crosses Body A.
    let mut orb_pillar_voxels = Vec::new();
    for y in 0..8 {
        for z in 0..12 {
            for x in 0..12 {
                orb_pillar_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    for y in 8..36 {
        for z in 4..8 {
            for x in 4..8 {
                orb_pillar_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    for y in 36..44 {
        for z in 0..12 {
            for x in 0..12 {
                let dx = (x as f32 - 5.5).abs();
                let dy = (y as f32 - 39.5).abs();
                let dz = (z as f32 - 5.5).abs();
                if dx * dx + dy * dy + dz * dz <= 38.0 {
                    orb_pillar_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
                }
            }
        }
    }
    let mut orb_pillar = RigidVoxelBody::from_global_voxels(&orb_pillar_voxels, &palette, None)
        .expect("INVARIANT: Orb-topped pillar must construct");

    // Rotate Body B horizontally (+90 deg about X so local +Y points along +Z, placing the narrow 4x28x4 shaft across Body A at z = 2.0
    // while BOTH wide 12-voxel capitals overhang in empty air at z < 1.3 and z > 2.7, completely clear of the plinth and Body A!):
    orb_pillar.orientation = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let r_mat = Mat3::from_quat(orb_pillar.orientation);
    let shaft_mid_local = Vec3::new(6.0, 22.0, 6.0) * VOXEL_SIZE;
    let target_shaft_mid_start = Vec3::new(2.0, 1.40, 2.0);
    orb_pillar.position =
        target_shaft_mid_start - r_mat * (shaft_mid_local - orb_pillar.com_local_m);

    let mut bodies = vec![fallen_pillar, orb_pillar];
    for _ in 0..140 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
    }

    // Measure exact 3D minimum surface-to-surface distance between Body B's narrow shaft [4..8, 8..36, 4..8]
    // and Body A's horizontal shaft [0..20, 0..6, 0..6]:
    let r_a = Mat3::from_quat(bodies[0].orientation);
    let inv_r_a = r_a.transpose();
    let a_half = Vec3::new(10.0, 3.0, 3.0) * VOXEL_SIZE;

    let r_b = Mat3::from_quat(bodies[1].orientation);
    let mut min_surface_gap = f32::INFINITY;
    for ly in 8..36 {
        for lz in [4, 7] {
            for lx in [4, 7] {
                // Sample outer surface corners of the narrow 4x28x4 shaft:
                let sx = if lx == 4 { 4.0 } else { 8.0 };
                let sz = if lz == 4 { 4.0 } else { 8.0 };
                let p_local = Vec3::new(sx, ly as f32 + 0.5, sz) * VOXEL_SIZE;
                let p_world = bodies[1].position + r_b * (p_local - bodies[1].com_local_m);
                let p_in_a = inv_r_a * (p_world - bodies[0].position);
                let q = p_in_a.abs() - a_half;
                let outside_dist = q.max(Vec3::ZERO).length();
                let inside_dist = q.x.max(q.y.max(q.z)).min(0.0);
                min_surface_gap = min_surface_gap.min(outside_dist + inside_dist);
            }
        }
    }

    // INVARIANT: Legacy single macro-OBB floated at 4 * VOXEL_SIZE = 0.25m (25 cm!) air gap because of the 12-wide capitals.
    // Exact compound collision brings the visible narrow shaft into direct contact with the fallen shaft (|gap| < 0.015m)!
    assert!(
        min_surface_gap.abs() < 0.015,
        "Narrow shaft of capital-topped pillar must rest directly on fallen shaft without floating! Surface gap = {min_surface_gap:.4}m"
    );
}

#[test]
fn test_concave_arch_empty_opening_passthrough_and_inner_collision() {
    let palette = MaterialPalette::default();
    let mut arch_voxels = Vec::new();
    // 12x12x4 stone arch with a 4x8x4 empty doorway at x in 4..8, y in 0..8:
    for z in 0..4 {
        for y in 0..12 {
            for x in 0..12 {
                if (4..8).contains(&x) && y < 8 {
                    continue;
                }
                arch_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let mut arch = RigidVoxelBody::from_global_voxels(&arch_voxels, &palette, None)
        .expect("INVARIANT: Concave arch must construct");
    // Place arch origin (min corner) at world (0, 0, 0):
    arch.position = arch.com_local_m;

    let mut cube_voxels = Vec::new();
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                cube_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let mut cube = RigidVoxelBody::from_global_voxels(&cube_voxels, &palette, None)
        .expect("INVARIANT: 2x2x2 probe cube must construct");

    // 1. Place 2x2x2 cube centered inside the empty doorway opening: x = 6*s, y = 4*s, z = 2*s
    // (cube spans x in 5..7, y in 3..5, z in 1..3 — 100% empty air inside the arch's macro OBB!):
    cube.position = Vec3::new(6.0, 4.0, 2.0) * VOXEL_SIZE;

    assert!(
        RigidVoxelBody::check_broadphase_overlap(&arch, &cube),
        "Cube inside arch doorway lies within macro-OBB broadphase"
    );
    let empty_manifolds =
        RigidVoxelBody::compute_compound_pair_manifolds_indexed(&arch, &cube, 0, 1);
    assert!(
        empty_manifolds.is_empty(),
        "Cube in empty doorway opening of concave arch must generate ZERO contact manifolds, got {}",
        empty_manifolds.len()
    );
    assert!(RigidVoxelBody::compute_obb_obb_manifold(&arch, &cube).is_none());

    // 2. Raise cube so its top face (center.y + 1*s) penetrates the inner arch lintel ceiling (y = 8*s) by 0.2*s:
    cube.position = Vec3::new(6.0, 7.2, 2.0) * VOXEL_SIZE;
    let lintel_manifolds =
        RigidVoxelBody::compute_compound_pair_manifolds_indexed(&arch, &cube, 0, 1);
    assert!(
        !lintel_manifolds.is_empty(),
        "Cube touching inner lintel ceiling of arch must generate contact manifold"
    );
    // Normal points from Body A (arch) to Body B (cube), so pushing cube downward out of ceiling means normal == -Vec3::Y:
    assert!(
        lintel_manifolds[0].normal.y < -0.99,
        "Arch inner ceiling must push cube downward (-Y), got {:?}",
        lintel_manifolds[0].normal
    );
}

#[test]
fn test_character_controller_walks_through_empty_air_under_t_pillar() {
    let palette = MaterialPalette::default();
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![ChunkTree64::new_empty()], palette);

    // T-shaped overhang: 4x38x8 vertical column at x in 0..4, y in 0..38, z in 0..8
    // plus an overhanging horizontal beam at x in 4..36, y in 34..38, z in 0..8 (canopy bottom at 34 * 0.0625 = 2.125m above base!).
    let mut t_voxels = Vec::new();
    for z in 0..8 {
        for y in 0..38 {
            for x in 0..4 {
                t_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
        for y in 34..38 {
            for x in 4..36 {
                t_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut t_body = RigidVoxelBody::from_global_voxels(&t_voxels, &palette, None)
        .expect("INVARIANT: T-pillar must construct");
    // Place base at world (10.0, 0.0, 10.0), so column is x in [10.0..10.25] and overhang canopy is x in [10.25..12.25], y in [2.125..2.375]:
    t_body.position = Vec3::new(10.0, 0.0, 10.0) + t_body.com_local_m;
    let bodies = vec![t_body];

    let mut player = CharacterController::new(Vec3::new(11.2, 0.0, 10.25), 0.0, 0.0);

    // At x = 11.2, y = 0.0 (head at 1.75m < 2.125m canopy bottom, and 0.95m right of the x <= 10.25 column),
    // the player stands inside the macro-OBB [10.0..12.25] x [0.0..2.375], in 100% empty air!
    let player_bmin = Vec3::new(11.2 - player.half_width, 1e-3, 10.25 - player.half_width);
    let player_bmax = Vec3::new(
        11.2 + player.half_width,
        player.height,
        10.25 + player.half_width,
    );
    assert!(
        bodies[0].obb_intersects_aabb(player_bmin, player_bmax),
        "Sanity check: legacy macro-OBB falsely overlaps player in empty air"
    );
    assert!(
        !player.collides_at(Vec3::new(11.2, 0.0, 10.25), &world, &bodies),
        "Player must NOT collide with empty air underneath T-pillar overhang"
    );

    // Moving into the actual solid vertical shaft at x = 10.125 MUST collide:
    assert!(
        player.collides_at(Vec3::new(10.15, 0.0, 10.25), &world, &bodies),
        "Player MUST collide with actual solid vertical column voxels"
    );
}

#[test]
fn test_heavy_standing_pillar_carve_split_no_skyward_launch() {
    let palette = MaterialPalette::default();
    let mut ground_voxels = Vec::new();
    for z in 0..32u32 {
        for x in 0..32u32 {
            for y in 0..=4u32 {
                ground_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
        }
    }
    let world = WorldMap::from_chunks(
        UVec3::new(1, 1, 1),
        vec![ChunkTree64::from_sparse_voxels(&ground_voxels, &palette)],
        palette,
    );

    // Heavy 8x48x8 standing temple pillar (3,072 voxels ~ 1,800 kg):
    let mut pillar_voxels = Vec::new();
    for y in 5..53 {
        for z in 12..20 {
            for x in 12..20 {
                pillar_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let mut pillar = RigidVoxelBody::from_global_voxels(&pillar_voxels, &palette, None)
        .expect("INVARIANT: Heavy pillar must construct");

    // Sever completely across the mid-section (y = 28 * VOXEL_SIZE = 1.75m) with radius 0.45m:
    let cut_center = Vec3::new(16.0 * VOXEL_SIZE, 28.0 * VOXEL_SIZE, 16.0 * VOXEL_SIZE);
    let outcome = pillar.carve_sphere_meters(cut_center, 0.45, false, &palette);

    assert_eq!(
        outcome.split_bodies.len(),
        1,
        "Mid-section carve must split heavy pillar into base and upper section"
    );
    let split_upper = outcome
        .split_bodies
        .into_iter()
        .next()
        .expect("INVARIANT: Split body verified");

    assert!(
        pillar.linear_velocity.y <= 0.1,
        "Base pillar must not jump upward on carve, got vy = {}",
        pillar.linear_velocity.y
    );
    assert!(
        split_upper.linear_velocity.y <= 0.1,
        "Severed upper heavy pillar must not launch skyward, got vy = {}",
        split_upper.linear_velocity.y
    );
    assert!(
        split_upper.angular_velocity.length() < 0.6,
        "Heavy severed upper pillar must not receive violent whip angular speed, got {}",
        split_upper.angular_velocity.length()
    );

    let initial_upper_y = split_upper.position.y;
    let mut bodies = vec![pillar, split_upper];
    for _ in 0..10 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
    }
    assert!(
        bodies[1].position.y < initial_upper_y,
        "Severed upper pillar must fall downward under gravity rather than flying upward"
    );
}

#[test]
fn test_shallow_water_pool_eliminates_bounce() {
    let palette = MaterialPalette::default();

    let mut dry_voxels = Vec::new();
    let mut pool_voxels = Vec::new();
    for z in 0..32u32 {
        for x in 0..32u32 {
            for y in 0..=4u32 {
                dry_voxels.push((x, y, z, MAT_STONE_DEEP));
                pool_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
            // Shallow 2-voxel-deep courtyard pool (12.5 cm of water above stone floor):
            for y in 5..=6u32 {
                pool_voxels.push((x, y, z, MAT_WATER_CRYSTAL));
            }
        }
    }

    let dry_world = WorldMap::from_chunks(
        UVec3::new(1, 1, 1),
        vec![ChunkTree64::from_sparse_voxels(&dry_voxels, &palette)],
        palette,
    );
    let pool_world = WorldMap::from_chunks(
        UVec3::new(1, 1, 1),
        vec![ChunkTree64::from_sparse_voxels(&pool_voxels, &palette)],
        palette,
    );

    // Tall 4x20x4 stone column (1.25m tall -> < 10% submerged in 12.5cm shallow water!):
    let mut col_voxels = Vec::new();
    for y in 0..20 {
        for z in 0..4 {
            for x in 0..4 {
                col_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut dry_body = RigidVoxelBody::from_global_voxels(&col_voxels, &palette, None)
        .expect("INVARIANT: Dry column must construct");
    dry_body.restitution = 0.8;
    dry_body.position = Vec3::new(1.0, 1.8, 1.0);

    let mut wet_body = dry_body.clone();

    let mut max_dry_rebound_vy = 0.0f32;
    let mut max_wet_rebound_vy = 0.0f32;

    for _ in 0..90 {
        dry_body.step(0.016, &dry_world);
        wet_body.step(0.016, &pool_world);
        max_dry_rebound_vy = max_dry_rebound_vy.max(dry_body.linear_velocity.y);
        max_wet_rebound_vy = max_wet_rebound_vy.max(wet_body.linear_velocity.y);
    }

    assert!(
        max_dry_rebound_vy > 0.55,
        "High-restitution dry body must bounce upward on dry stone (got max vy = {max_dry_rebound_vy})"
    );
    assert!(
        max_wet_rebound_vy < 0.20 && max_dry_rebound_vy > max_wet_rebound_vy * 4.0,
        "Shallow 2-voxel water pool must eliminate bounce even when <10% of pillar is submerged (dry = {max_dry_rebound_vy}, wet = {max_wet_rebound_vy})"
    );
}

#[test]
fn test_pillar_balancing_on_single_pixel_topples_under_gravity() {
    let palette = MaterialPalette::default();

    // Courtyard bedrock floor (y in 0..=4) plus a single 1-voxel-thin vertical stem at (x = 24, z = 32, y = 5..=14):
    let mut world_voxels = Vec::new();
    for z in 0..64u32 {
        for x in 0..64u32 {
            for y in 0..=4u32 {
                world_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
        }
    }
    for y in 5..=14u32 {
        world_voxels.push((24, y, 32, MAT_STONE_DEEP));
    }
    let world = WorldMap::from_chunks(
        UVec3::new(1, 1, 1),
        vec![ChunkTree64::from_sparse_voxels(&world_voxels, &palette)],
        palette,
    );

    // Heavy 10x32x10 severed upper pillar (~1,800 kg) with 85% of its lower right base carved out
    // (reproducing the screenshot: zero initial angular velocity, dropping straight down onto the 1-voxel stem at x = 24!):
    let mut pillar_voxels = Vec::new();
    for y in 16..48 {
        for z in 28..38 {
            for x in 24..34 {
                // Carve away right 85% of lower 6 rows (y in 16..22) leaving only the far-left rim above x = 24:
                if y < 22 && x > 25 {
                    continue;
                }
                pillar_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let pillar = RigidVoxelBody::from_global_voxels(&pillar_voxels, &palette, None)
        .expect("INVARIANT: Carved heavy pillar must construct");
    assert_eq!(pillar.angular_velocity, Vec3::ZERO);

    let mut bodies = vec![pillar];
    let mut max_omega = 0.0f32;
    for _ in 0..120 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
        max_omega = max_omega.max(bodies[0].angular_velocity.length());
    }

    let up_y = (bodies[0].orientation * Vec3::Y).y;
    assert!(
        max_omega > 0.6,
        "Heavy pillar dropped onto a 1-voxel stem must accumulate gravitational topple angular velocity (got max_omega = {max_omega})"
    );
    assert!(
        up_y < 0.70,
        "Pillar must topple off the single-pixel stem rather than freezing upright (got up_y = {up_y})"
    );
}

#[test]
fn test_editing_sleeping_dynamic_object_recomputes_com_and_tips_over() {
    let palette = MaterialPalette::default();

    // Courtyard floor + narrow 4x6x4 pedestal at x in 28..32, z in 28..32, y in 5..=10 (top at 11 * VOXEL_SIZE = 0.6875m):
    let mut world_voxels = Vec::new();
    for z in 0..64u32 {
        for x in 0..64u32 {
            for y in 0..=4u32 {
                world_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
        }
    }
    for z in 28..32u32 {
        for x in 28..32u32 {
            for y in 5..=10u32 {
                world_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
        }
    }
    let world = WorldMap::from_chunks(
        UVec3::new(1, 1, 1),
        vec![ChunkTree64::from_sparse_voxels(&world_voxels, &palette)],
        palette,
    );

    // 4x8x4 stone block resting symmetrically atop the 4x6x4 pedestal:
    let mut block_voxels = Vec::new();
    for y in 11..19 {
        for z in 28..32 {
            for x in 28..32 {
                block_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let block = RigidVoxelBody::from_global_voxels(&block_voxels, &palette, None)
        .expect("INVARIANT: Block must construct");
    let mut bodies = vec![block];

    // Settle block into sleep atop the pedestal:
    for _ in 0..60 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
    }
    assert!(
        bodies[0].sleeping,
        "Symmetric block must settle and sleep on pedestal"
    );
    let com_x_before = bodies[0].position.x;

    // Edit the sleeping object by placing heavy cantilevered stone spheres out along +X:
    for step_x in 1..=3 {
        let place_pos = Vec3::new(
            com_x_before + (step_x as f32) * 0.16,
            bodies[0].position.y + 0.15,
            bodies[0].position.z,
        );
        let placed = bodies[0].place_sphere_meters(place_pos, 0.14, MAT_STONE_DEEP, &palette);
        assert!(
            placed,
            "Voxel placement step {step_x} on dynamic body must succeed"
        );
    }

    assert!(
        bodies[0].position.x > com_x_before + 0.08,
        "Placing voxels on +X side must shift world CoM toward +X (before = {com_x_before}, after = {})",
        bodies[0].position.x
    );
    assert!(!bodies[0].sleeping, "Edited dynamic body must wake up");

    // Step simulation: shifted Center of Mass now overhangs the pedestal's +X edge and must tip the object over!
    for _ in 0..90 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
    }

    let up_y = (bodies[0].orientation * Vec3::Y).y;
    assert!(
        up_y < 0.85,
        "Dynamic object edited with cantilevered mass must tip over toward new CoM (got up_y = {up_y})"
    );
}

#[test]
fn test_heavy_temple_pillar_detachment_realistic_topple_momentum() {
    let palette = MaterialPalette::default();
    let mut world_voxels = Vec::new();
    for z in 0..64u32 {
        for x in 0..64u32 {
            for y in 0..=4u32 {
                world_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
        }
    }
    let world = WorldMap::from_chunks(
        UVec3::new(1, 1, 1),
        vec![ChunkTree64::from_sparse_voxels(&world_voxels, &palette)],
        palette,
    );

    // Multi-ton temple pillar (12x48x12 = 6,912 voxels ~ 4,050 kg) severed near center-left of base:
    let mut pillar_voxels = Vec::new();
    for y in 6..54 {
        for z in 26..38 {
            for x in 26..38 {
                pillar_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let blast_origin = Vec3::new(31.2 * VOXEL_SIZE, 6.0 * VOXEL_SIZE, 32.0 * VOXEL_SIZE);
    let pillar = RigidVoxelBody::from_global_voxels(&pillar_voxels, &palette, Some(blast_origin))
        .expect("INVARIANT: Multi-ton temple pillar must construct");

    let init_omega = pillar.angular_velocity.length();
    assert!(
        (0.90..=2.00).contains(&init_omega),
        "Multi-ton severed temple pillar must receive realistic initial topple angular speed even on near-center cuts, got {init_omega}"
    );
    assert!(
        pillar.linear_velocity.y <= 0.08 + 1e-4,
        "Multi-ton severed temple pillar must never launch skyward, got vy = {}",
        pillar.linear_velocity.y
    );

    let mut bodies = vec![pillar];
    let mut peak_omega = init_omega;
    for _ in 0..75 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
        peak_omega = peak_omega.max(bodies[0].angular_velocity.length());
    }

    let up_y = (bodies[0].orientation * Vec3::Y).y;
    assert!(
        peak_omega > init_omega * 1.25,
        "Gravitational pivot torque must accelerate heavy pillar topple rather than damping it out (init = {init_omega}, peak = {peak_omega})"
    );
    assert!(
        up_y < 0.35,
        "Multi-ton temple pillar must topple all the way down toward horizontal (got up_y = {up_y})"
    );
}

#[test]
fn test_multi_body_stack_enters_island_sleep_cleanly() {
    let palette = MaterialPalette::default();
    let mut ground_voxels = Vec::new();
    for z in 0..32u32 {
        for x in 0..32u32 {
            for y in 0..=4u32 {
                ground_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
        }
    }
    let world = WorldMap::from_chunks(
        UVec3::new(1, 1, 1),
        vec![ChunkTree64::from_sparse_voxels(&ground_voxels, &palette)],
        palette,
    );

    let mut box_voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                box_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut a = RigidVoxelBody::from_global_voxels(&box_voxels, &palette, None)
        .expect("INVARIANT: Box A must construct");
    let mut b = a.clone();
    let mut c = a.clone();
    a.position = Vec3::new(1.0, 0.44, 1.0);
    b.position = Vec3::new(1.0, 0.70, 1.0);
    c.position = Vec3::new(1.0, 0.96, 1.0);

    let mut bodies = vec![a, b, c];
    for _ in 0..90 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
    }

    for (idx, body) in bodies.iter().enumerate() {
        assert!(
            body.sleeping,
            "Stacked body {idx} must enter island sleep without wake-up thrashing"
        );
        assert_eq!(body.linear_velocity, Vec3::ZERO);
        assert_eq!(body.angular_velocity, Vec3::ZERO);
    }

    let dy_ab = bodies[1].position.y - bodies[0].position.y;
    let dy_bc = bodies[2].position.y - bodies[1].position.y;
    assert!(
        (dy_ab - 0.25).abs() < 0.008 && (dy_bc - 0.25).abs() < 0.008,
        "Sleeping 3-box stack must have minimal overlap (< 8 mm): dy_ab = {dy_ab}, dy_bc = {dy_bc}"
    );
}

#[test]
fn test_bowl_of_objects_crushed_by_giant_stone_sphere_no_jitter_or_self_drift() {
    let palette = MaterialPalette::default();

    // Carve a hemispherical stone bowl in a 64x32x64 block:
    // Bowl cavity center at (32.0, 22.0, 32.0) with inner radius 14.0 voxels (floor at y = 8 -> 0.50m, rim at y = 22 -> 1.375m):
    let mut world_voxels = Vec::new();
    let bowl_center = Vec3::new(32.0, 22.0, 32.0);
    let inner_r_sq = 14.0f32 * 14.0f32;
    let outer_r_sq = 18.5f32 * 18.5f32;
    for z in 0..64u32 {
        for x in 0..64u32 {
            for y in 0..=4u32 {
                world_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
            for y in 5..=22u32 {
                let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                let d_sq = p.distance_squared(bowl_center);
                if d_sq >= inner_r_sq && d_sq <= outer_r_sq {
                    world_voxels.push((x, y, z, MAT_STONE_DEEP));
                }
            }
        }
    }
    let world = WorldMap::from_chunks(
        UVec3::new(1, 1, 1),
        vec![ChunkTree64::from_sparse_voxels(&world_voxels, &palette)],
        palette,
    );

    // 8 small stone blocks (3x3x3 = 27 voxels ~ 15.8 kg each) dropped into the bowl:
    let mut small_voxels = Vec::new();
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                small_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let template_small = RigidVoxelBody::from_global_voxels(&small_voxels, &palette, None)
        .expect("INVARIANT: Small block must construct");

    let offsets = [
        Vec3::new(-0.14, 0.82, -0.12),
        Vec3::new(0.14, 0.84, -0.10),
        Vec3::new(-0.12, 0.86, 0.14),
        Vec3::new(0.13, 0.88, 0.13),
        Vec3::new(0.0, 1.05, -0.08),
        Vec3::new(-0.09, 1.08, 0.04),
        Vec3::new(0.09, 1.11, 0.05),
        Vec3::new(0.0, 1.15, 0.0),
    ];

    let mut bodies = Vec::with_capacity(9);
    for off in offsets {
        let mut b = template_small.clone();
        b.position = Vec3::new(2.0 + off.x, off.y, 2.0 + off.z);
        bodies.push(b);
    }

    // Settle small objects into the bottom of the bowl for 35 frames:
    for _ in 0..35 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
    }

    // Construct a giant stone sphere (radius 7.5 voxels -> ~1,790 voxels ~ 1,050 kg, > 65:1 mass ratio!):
    let mut sphere_voxels = Vec::new();
    for z in 0..15 {
        for y in 0..15 {
            for x in 0..15 {
                let d = Vec3::new(x as f32 - 7.0, y as f32 - 7.0, z as f32 - 7.0);
                if d.length_squared() <= 7.3 * 7.3 {
                    sphere_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
                }
            }
        }
    }
    let mut boulder = RigidVoxelBody::from_global_voxels(&sphere_voxels, &palette, None)
        .expect("INVARIANT: Giant stone sphere must construct");
    assert!(
        boulder.mass > template_small.mass * 55.0,
        "Boulder must exceed 55:1 mass ratio over small bowl objects"
    );
    boulder.position = Vec3::new(2.02, 1.75, 1.99);
    let boulder_idx = bodies.len();
    bodies.push(boulder);

    let mut snapshot_160 = Vec::new();
    for step in 0..210 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
        if step == 160 {
            snapshot_160 = bodies.iter().map(|b| b.position).collect();
        }
    }

    // 1. No small object was crushed through the bowl floor (bowl floor inner surface is at y = 8 * 0.0625 = 0.50m):
    for (idx, b) in bodies.iter().enumerate() {
        assert!(
            b.position.y > 0.52 && b.position.y < 2.5,
            "Body {idx} must remain stably inside the bowl without tunneling or exploding (got pos = {:?})",
            b.position
        );
    }

    // 2. Giant boulder sits supported atop the rubble/bowl without glitching deep inside the small blocks:
    let mut max_pair_pen = 0.0f32;
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            for m in RigidVoxelBody::compute_compound_pair_manifolds_indexed(
                &bodies[i], &bodies[j], i, j,
            ) {
                for pt in &m.points[0..m.point_count] {
                    max_pair_pen = max_pair_pen.max(pt.penetration);
                }
            }
        }
    }
    assert!(
        max_pair_pen < 0.015,
        "High-mass-ratio boulder on bowl objects must not glitch deeply inside objects (max penetration = {max_pair_pen} m)"
    );

    // 3. Zero self-propelling drift or perpetual jitter over the final 50 frames (0.80 seconds):
    let boulder_drift = bodies[boulder_idx]
        .position
        .distance(snapshot_160[boulder_idx]);
    let mut max_small_drift = 0.0f32;
    for i in 0..boulder_idx {
        max_small_drift = max_small_drift.max(bodies[i].position.distance(snapshot_160[i]));
    }
    assert!(
        boulder_drift < 0.008 && max_small_drift < 0.012,
        "Objects under heavy boulder in bowl must not move on their own (boulder_drift = {boulder_drift} m, max_small_drift = {max_small_drift} m)"
    );
}

#[test]
fn test_balanced_seesaw_rod_disrupted_by_dropped_weight() {
    let palette = MaterialPalette::default();

    // Bedrock floor + narrow 2-voxel-wide central fulcrum (x in 31..33, z in 29..35, y in 5..=10, top at 11 * 0.0625 = 0.6875m):
    let mut world_voxels = Vec::new();
    for z in 0..64u32 {
        for x in 0..64u32 {
            for y in 0..=4u32 {
                world_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
        }
    }
    for z in 29..35u32 {
        for x in 31..33u32 {
            for y in 5..=10u32 {
                world_voxels.push((x, y, z, MAT_STONE_DEEP));
            }
        }
    }
    let world = WorldMap::from_chunks(
        UVec3::new(1, 1, 1),
        vec![ChunkTree64::from_sparse_voxels(&world_voxels, &palette)],
        palette,
    );

    // Symmetric 20x2x4 horizontal stone rod balanced across the fulcrum:
    let mut rod_voxels = Vec::new();
    for y in 11..13 {
        for z in 30..34 {
            for x in 22..42 {
                rod_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let rod = RigidVoxelBody::from_global_voxels(&rod_voxels, &palette, None)
        .expect("INVARIANT: Seesaw rod must construct");
    let mut bodies = vec![rod];

    for _ in 0..30 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
    }
    let balanced_up_y = (bodies[0].orientation * Vec3::Y).y;
    assert!(
        balanced_up_y > 0.98,
        "Symmetric rod must remain horizontally balanced on fulcrum before impact (got up_y = {balanced_up_y})"
    );

    // Drop a heavy 6x6x6 stone weight onto the +X overhanging end of the balanced rod:
    let mut weight_voxels = Vec::new();
    for z in 0..6 {
        for y in 0..6 {
            for x in 0..6 {
                weight_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let mut weight = RigidVoxelBody::from_global_voxels(&weight_voxels, &palette, None)
        .expect("INVARIANT: Dropped weight must construct");
    weight.position = Vec3::new(2.44, 1.25, 2.00);
    bodies.push(weight);

    for _ in 0..45 {
        RigidVoxelBody::step_system(&mut bodies, 0.016, &world);
    }

    let tipped_up_y = (bodies[0].orientation * Vec3::Y).y;
    assert!(
        tipped_up_y < 0.91,
        "Dropping weight onto overhanging end must disrupt balance and tip the seesaw rod (got up_y = {tipped_up_y})"
    );
}
