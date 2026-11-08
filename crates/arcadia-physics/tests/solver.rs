//! Tests for sequential impulse solver, friction, warm-starting, and sleeping wake-up.
//! Owns CPU unit verification of impulse resolution, Coulomb friction, and deflection.

use arcadia_physics::{
    BodyPairManifold, ContactManifold, ContactPoint3D, RigidVoxelBody, MAX_PENETRATION_BIAS_VEL,
    PENETRATION_SLOP,
};
mod common;
use arcadia_core::{ChunkTree64, MaterialPalette, MAT_STONE_DEEP, MAT_WATER_CRYSTAL};
use arcadia_world::WorldMap;
use glam::{IVec3, UVec3, Vec3};

#[test]
fn test_rigid_body_pairwise_collision() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body_a = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create a rigid voxel body");
    let mut body_b = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create a rigid voxel body");

    body_a.position = Vec3::new(10.0, 10.0, 10.0);
    body_b.position = Vec3::new(10.15, 10.0, 10.0); // overlapping/touching along X

    body_a.linear_velocity = Vec3::new(2.0, 0.0, 0.0);
    body_b.linear_velocity = Vec3::new(-2.0, 0.0, 0.0);

    RigidVoxelBody::resolve_collision_pair(&mut body_a, &mut body_b);

    // After collision, Body A should be pushed back (-X) and Body B should be pushed forward (+X):
    assert!(
        body_a.linear_velocity.x < 2.0,
        "Body A must receive a repulsive impulse along -X"
    );
    assert!(
        body_b.linear_velocity.x > -2.0,
        "Body B must receive a repulsive impulse along +X"
    );
    assert!(!body_a.sleeping && !body_b.sleeping);
}

#[test]
fn test_pair_manifold_preparation() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let body_a = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body A");
    let mut body_b = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body B");

    body_b.position = Vec3::new(body_a.position.x, 0.355, body_a.position.z);

    let mut manifold = RigidVoxelBody::compute_obb_obb_manifold(&body_a, &body_b)
        .expect("INVARIANT: Overlapping stacked boxes must produce contact manifold");

    assert_eq!(manifold.point_count, 4);
    assert_eq!(manifold.points[0].effective_mass_n, 0.0);

    RigidVoxelBody::prepare_pair_manifold(&mut manifold, &body_a, &body_b);

    for pt in &manifold.points[0..manifold.point_count] {
        assert!(
            pt.effective_mass_n > 0.0,
            "Effective normal mass must be positive"
        );
        assert!(
            pt.effective_mass_t1 > 0.0,
            "Effective tangent 1 mass must be positive"
        );
        assert!(
            pt.effective_mass_t2 > 0.0,
            "Effective tangent 2 mass must be positive"
        );
    }
}

#[test]
fn test_pair_warm_start() {
    let palette = MaterialPalette::default();
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

    body_a.linear_velocity = Vec3::ZERO;
    body_b.linear_velocity = Vec3::ZERO;

    let manifold = BodyPairManifold {
        normal: Vec3::Y,
        tangent1: Vec3::X,
        tangent2: Vec3::Z,
        point_count: 1,
        points: [
            ContactPoint3D {
                position: body_a.position + Vec3::new(0.0, 0.125, 0.0),
                r_a: Vec3::new(0.0, 0.125, 0.0),
                r_b: Vec3::new(0.0, -0.125, 0.0),
                accum_n: 10.0,
                accum_t1: 2.0,
                accum_t2: 0.0,
                ..Default::default()
            },
            ContactPoint3D::default(),
            ContactPoint3D::default(),
            ContactPoint3D::default(),
        ],
        ..Default::default()
    };

    RigidVoxelBody::apply_pair_warm_start(&manifold, &mut body_a, &mut body_b);

    let p = Vec3::new(2.0, 10.0, 0.0);
    assert!((body_a.linear_velocity - (-p * body_a.inv_mass)).length() < 1e-5);
    assert!((body_b.linear_velocity - (p * body_b.inv_mass)).length() < 1e-5);
}

#[test]
fn test_pair_sequential_impulse_solve() {
    let palette = MaterialPalette::default();
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

    // Overlap boxes along Y:
    body_b.position = Vec3::new(body_a.position.x, 0.355, body_a.position.z);
    body_a.linear_velocity = Vec3::new(0.0, 0.5, 0.0);
    body_b.linear_velocity = Vec3::new(0.0, -0.5, 0.0);

    let mut manifold = RigidVoxelBody::compute_obb_obb_manifold(&body_a, &body_b)
        .expect("INVARIANT: Overlapping boxes must produce manifold");

    RigidVoxelBody::prepare_pair_manifold(&mut manifold, &body_a, &body_b);

    let sub_dt = 0.016 / 4.0;
    for iter in 0..6 {
        RigidVoxelBody::solve_pair_manifold_iteration(
            &mut manifold,
            &mut body_a,
            &mut body_b,
            sub_dt,
            iter,
        );
    }

    // After 6 iterations, the relative normal velocity must be driven positive (separating):
    let v_rel = body_b.linear_velocity - body_a.linear_velocity;
    let vn = v_rel.dot(manifold.normal);
    assert!(
        vn >= 0.0,
        "Relative normal velocity must be non-negative after solver iterations, got {}",
        vn
    );
    for pt in &manifold.points[0..manifold.point_count] {
        assert!(
            pt.accum_n >= 0.0,
            "Accumulated normal impulse must be non-negative"
        );
    }
}

#[test]
fn test_pair_coulomb_friction() {
    let palette = MaterialPalette::default();
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

    body_a.friction = 0.5;
    body_b.friction = 0.5;

    // Overlap boxes along Y, sliding with relative tangential velocity along X:
    body_b.position = Vec3::new(body_a.position.x, 0.355, body_a.position.z);
    body_a.linear_velocity = Vec3::new(-1.0, 0.0, 0.0);
    body_b.linear_velocity = Vec3::new(1.0, 0.0, 0.0);

    let mut manifold = RigidVoxelBody::compute_obb_obb_manifold(&body_a, &body_b)
        .expect("INVARIANT: Overlapping boxes must produce manifold");

    RigidVoxelBody::prepare_pair_manifold(&mut manifold, &body_a, &body_b);

    let sub_dt = 0.016 / 4.0;
    for iter in 0..6 {
        RigidVoxelBody::solve_pair_manifold_iteration(
            &mut manifold,
            &mut body_a,
            &mut body_b,
            sub_dt,
            iter,
        );
    }

    // Tangential sliding velocity along X should be reduced significantly by friction:
    let v_rel_x = body_b.linear_velocity.x - body_a.linear_velocity.x;
    assert!(
        v_rel_x < 2.0,
        "Relative sliding velocity must be reduced by Coulomb friction, got {}",
        v_rel_x
    );
    for pt in &manifold.points[0..manifold.point_count] {
        let max_f = manifold.friction * pt.accum_n;
        assert!(
            pt.accum_t1.abs() <= max_f + 1e-4,
            "Tangential impulse 1 must satisfy Coulomb friction cone"
        );
        assert!(
            pt.accum_t2.abs() <= max_f + 1e-4,
            "Tangential impulse 2 must satisfy Coulomb friction cone"
        );
    }
}

#[test]
fn test_step_system_wakes_sleeping_body_on_contact() {
    let palette = MaterialPalette::default();
    let dummy_chunk = ChunkTree64::from_sparse_voxels(&[], &palette);
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![dummy_chunk], palette);

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

    body_a.position = Vec3::new(10.0, 10.0, 10.0);
    body_a.linear_velocity = Vec3::ZERO;
    body_a.angular_velocity = Vec3::ZERO;
    body_a.sleeping = true;

    // Body B is directly above Body A, touching or overlapping:
    body_b.position = Vec3::new(10.0, 10.23, 10.0);
    body_b.linear_velocity = Vec3::new(0.0, -1.0, 0.0);
    body_b.sleeping = false;

    let mut bodies = vec![body_a, body_b];
    RigidVoxelBody::step_system(&mut bodies, 0.016, &world);

    assert!(
        !bodies[0].sleeping,
        "Sleeping Body A must be awakened when struck by dynamic Body B"
    );
}

#[test]
fn test_step_system_pairwise_deflection() {
    let palette = MaterialPalette::default();
    let dummy_chunk = ChunkTree64::from_sparse_voxels(&[], &palette);
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![dummy_chunk], palette);

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

    body_a.position = Vec3::new(10.0, 10.0, 10.0);
    body_b.position = Vec3::new(10.20, 10.0, 10.0);

    body_a.linear_velocity = Vec3::new(2.0, 0.0, 0.0);
    body_b.linear_velocity = Vec3::new(-2.0, 0.0, 0.0);

    let mut bodies = vec![body_a, body_b];
    RigidVoxelBody::step_system(&mut bodies, 0.016, &world);

    // After step_system, Body A must be repelled towards -X and Body B towards +X:
    assert!(
        bodies[0].linear_velocity.x < 2.0,
        "Body A must be repelled along -X by pairwise collision"
    );
    assert!(
        bodies[1].linear_velocity.x > -2.0,
        "Body B must be repelled along +X by pairwise collision"
    );
}

#[test]
fn test_baumgarte_penetration_bias_clamped_on_deep_overlap() {
    let baumgarte_beta = 0.20f32;
    let sub_dt = 0.016 / 4.0;
    let deep_penetration = 0.10f32; // 10 cm overlap

    // 1. Unclamped bias calculation would inject explosive catapult velocity:
    let unclamped_bias = (baumgarte_beta / sub_dt) * (deep_penetration - PENETRATION_SLOP);
    assert!(
        unclamped_bias > 4.0,
        "Unclamped bias would inject explosive velocity: {unclamped_bias} m/s"
    );

    // 2. Verified clamped bias calculation:
    let effective_pen = (deep_penetration - PENETRATION_SLOP).max(0.0);
    let bias = ((baumgarte_beta / sub_dt) * effective_pen).min(MAX_PENETRATION_BIAS_VEL);

    assert!(
        bias <= MAX_PENETRATION_BIAS_VEL,
        "Deep penetration bias must be clamped to MAX_PENETRATION_BIAS_VEL"
    );
    assert!(
        (bias - 0.8).abs() < 1e-5,
        "Deep penetration (10 cm) bias should equal exactly 0.8 m/s clamp"
    );

    // 3. Verify under pair manifold solver iteration:
    let palette = MaterialPalette::default();
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

    body_a.position = Vec3::new(10.0, 10.0, 10.0);
    body_b.position = Vec3::new(10.0, 10.15, 10.0);
    body_a.linear_velocity = Vec3::ZERO;
    body_b.linear_velocity = Vec3::ZERO;

    let mut manifold = RigidVoxelBody::compute_obb_obb_manifold(&body_a, &body_b)
        .expect("INVARIANT: Overlapping boxes must generate manifold");

    assert!(manifold.point_count > 0);
    assert!(
        manifold.points[0].penetration > 0.05,
        "Penetration should be deep (> 5 cm)"
    );

    RigidVoxelBody::prepare_pair_manifold(&mut manifold, &body_a, &body_b);
    RigidVoxelBody::solve_pair_manifold_iteration(
        &mut manifold,
        &mut body_a,
        &mut body_b,
        sub_dt,
        0,
    );

    assert!(
        body_b.linear_velocity.y <= 0.8,
        "Body B upward velocity must not exceed MAX_PENETRATION_BIAS_VEL (got {})",
        body_b.linear_velocity.y
    );
    assert!(
        body_a.linear_velocity.y >= -0.8,
        "Body A downward velocity must not exceed MAX_PENETRATION_BIAS_VEL (got {})",
        body_a.linear_velocity.y
    );
}

#[test]
fn test_blast_impulse_upward_velocity_clamped() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    // Blast origin directly beneath the body's center of mass:
    let blast_origin = Vec3::new(0.125, -1.0, 0.125);
    let body = RigidVoxelBody::from_global_voxels(&voxels, &palette, Some(blast_origin))
        .expect("INVARIANT: Valid body creation with blast origin");

    assert!(
        body.linear_velocity.y <= 0.1001,
        "Blast upward velocity must be clamped to <= 0.1 m/s (got {})",
        body.linear_velocity.y
    );
}

#[test]
fn test_rigid_body_water_buoyancy() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body_air = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body in air");
    let mut body_water = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body in water");

    body_air.position = Vec3::new(2.0, 2.0, 2.0);
    body_water.position = Vec3::new(2.0, 2.0, 2.0);

    let empty_chunk = ChunkTree64::from_sparse_voxels(&[], &palette);
    let world_air = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![empty_chunk], palette);

    let mut water_voxels = Vec::new();
    for z in 24..40 {
        for y in 0..50 {
            for x in 24..40 {
                water_voxels.push((x as u32, y as u32, z as u32, MAT_WATER_CRYSTAL));
            }
        }
    }
    let water_chunk = ChunkTree64::from_sparse_voxels(&water_voxels, &palette);
    let world_water = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![water_chunk], palette);

    let ratio_air = body_air.compute_submersion_ratio(&world_air);
    let ratio_water = body_water.compute_submersion_ratio(&world_water);

    assert_eq!(ratio_air, 0.0, "Body in air must have 0.0 submersion ratio");
    assert_eq!(
        ratio_water, 1.0,
        "Body submerged in water crystal volume must have 1.0 submersion ratio"
    );

    // Advance 1 frame (4 substeps of 0.004s):
    body_air.step(0.016, &world_air);
    body_water.step(0.016, &world_water);

    assert_eq!(body_water.submersion_ratio, 1.0);
    assert!(
        body_water.linear_velocity.y > body_air.linear_velocity.y,
        "Water buoyancy and drag must reduce downward velocity (got water: {}, air: {})",
        body_water.linear_velocity.y,
        body_air.linear_velocity.y
    );
}

#[test]
fn test_rigid_body_water_terminal_velocity() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body_water = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body in water");
    let mut body_air = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create body in air");

    let sub_dt = 0.016 / 4.0;
    let gravity = Vec3::new(0.0, -18.0, 0.0);

    // Simulate 2 seconds of falling (120 frames * 4 substeps = 480 substeps):
    for _ in 0..480 {
        // Air body integration:
        body_air.linear_velocity += gravity * sub_dt;
        body_air.linear_velocity *= (1.0 - 0.05 * sub_dt).max(0.0);

        // Water body integration:
        body_water.linear_velocity += gravity * sub_dt;
        body_water.linear_velocity *= (1.0 - 0.05 * sub_dt).max(0.0);
        body_water.apply_fluid_dynamics(1.0, sub_dt);
    }

    // Downward terminal velocity in water settles at theoretical ~2.8 m/s (0.8*v^2 + 1.5*v = 10.5):
    assert!(
        (body_water.linear_velocity.y.abs() - 2.8).abs() < 0.2,
        "Water terminal velocity must converge to ~2.8 m/s (got {})",
        body_water.linear_velocity.y
    );
    assert!(
        body_air.linear_velocity.y.abs() > 30.0,
        "Air downward velocity after 2s must exceed 30 m/s (got {})",
        body_air.linear_velocity.y
    );
}

#[test]
fn test_rigid_body_water_restitution_attenuated() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut body_submerged = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create submerged body");
    let mut body_dry = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create dry body");

    body_submerged.restitution = 0.8;
    body_dry.restitution = 0.8;

    body_submerged.submersion_ratio = 1.0;
    body_dry.submersion_ratio = 0.0;

    // High downward velocity (-5.0 m/s) impacting upward ground normal:
    body_submerged.linear_velocity = Vec3::new(0.0, -5.0, 0.0);
    body_dry.linear_velocity = Vec3::new(0.0, -5.0, 0.0);

    let mut contact_submerged = ContactManifold {
        r_world: Vec3::ZERO,
        normal: Vec3::Y,
        penetration: 0.01,
        accum_n: 0.0,
        accum_t1: 0.0,
        accum_t2: 0.0,
        tangent1: Vec3::X,
        tangent2: Vec3::Z,
        effective_mass_n: body_dry.mass,
        effective_mass_t1: body_dry.mass,
        effective_mass_t2: body_dry.mass,
    };
    let mut contact_dry = contact_submerged;

    let sub_dt = 0.016 / 4.0;
    let inv_i_world = body_dry.world_inv_inertia();

    let rest_dry = body_dry.restitution * (1.0 - body_dry.submersion_ratio).powi(2);
    let rest_submerged =
        body_submerged.restitution * (1.0 - body_submerged.submersion_ratio).powi(2);

    assert_eq!(rest_dry, 0.8);
    assert_eq!(
        rest_submerged, 0.0,
        "Submerged restitution must attenuate to exactly 0.0"
    );

    body_dry.solve_terrain_contacts_iteration(
        std::slice::from_mut(&mut contact_dry),
        inv_i_world,
        sub_dt,
        rest_dry,
        0,
    );

    body_submerged.solve_terrain_contacts_iteration(
        std::slice::from_mut(&mut contact_submerged),
        inv_i_world,
        sub_dt,
        rest_submerged,
        0,
    );

    // Dry body rebounds upward due to e = 0.8:
    assert!(
        body_dry.linear_velocity.y > 2.0,
        "Dry body with restitution 0.8 must rebound upward (got {})",
        body_dry.linear_velocity.y
    );

    // Submerged body must not rebound upward (effective restitution = 0.0):
    assert!(
        body_submerged.linear_velocity.y <= MAX_PENETRATION_BIAS_VEL,
        "Submerged body must eliminate rebound trampoline bounce (got {})",
        body_submerged.linear_velocity.y
    );
}

#[test]
fn test_submersion_ratio_partial_immersion() {
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
        .expect("INVARIANT: Valid voxels and palette must create body");

    // Position body at water waterline:
    // Water occupies y from 0 to 32 (up to 2.0m).
    // Body is centered at y = 2.0m, so bottom half is in water, top half is in air.
    body.position = Vec3::new(2.0, 2.0, 2.0);

    let mut water_voxels = Vec::new();
    for z in 24..40 {
        for y in 0..32 {
            for x in 24..40 {
                water_voxels.push((x as u32, y as u32, z as u32, MAT_WATER_CRYSTAL));
            }
        }
    }
    let water_chunk = ChunkTree64::from_sparse_voxels(&water_voxels, &palette);
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![water_chunk], palette);

    let ratio = body.compute_submersion_ratio(&world);
    assert!(
        ratio > 0.2 && ratio < 0.8,
        "Partially submerged body must have intermediate submersion ratio (got {})",
        ratio
    );
}

#[test]
fn test_baking_body_solver_isolation_and_unique_ids() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    let mut baking_body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create baking_body");
    let mut dynamic_body = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create dynamic_body");

    // Verify IDs are unique and monotonically increasing:
    assert_ne!(baking_body.id, dynamic_body.id);
    assert!(dynamic_body.id > baking_body.id);

    // Put baking body in baking mode:
    baking_body.position = Vec3::new(5.0, 5.0, 5.0);
    baking_body.sleeping = true;
    baking_body.is_baking = true;
    baking_body.rest_time = 65.0;

    // Verify wake_up() and reset_rest_timer() are no-ops while is_baking:
    baking_body.wake_up();
    assert!(
        baking_body.sleeping,
        "Baking body must not be awakened by wake_up()"
    );
    baking_body.reset_rest_timer();
    assert_eq!(
        baking_body.rest_time, 65.0,
        "Baking body must not reset rest_time"
    );

    // Position dynamic projectile to slam into the baking body along +X:
    dynamic_body.position = Vec3::new(4.70, 5.0, 5.0);
    dynamic_body.linear_velocity = Vec3::new(5.0, 0.0, 0.0);

    let empty_world = WorldMap::from_chunks(UVec3::ONE, vec![ChunkTree64::new_empty()], palette);
    let mut bodies = vec![baking_body, dynamic_body];

    RigidVoxelBody::step_system(&mut bodies, 0.016, &empty_world);

    // Baking body (index 0) must remain completely stationary with zero velocity:
    assert_eq!(
        bodies[0].position,
        Vec3::new(5.0, 5.0, 5.0),
        "Baking body position must not drift"
    );
    assert_eq!(
        bodies[0].linear_velocity,
        Vec3::ZERO,
        "Baking body must receive zero linear impulse"
    );
    assert!(bodies[0].sleeping, "Baking body must remain asleep");

    // Dynamic projectile (index 1) must be repelled by the baking body's static boundary:
    assert!(
        bodies[1].linear_velocity.x < 5.0,
        "Dynamic body must be deflected by the baking body"
    );
}
