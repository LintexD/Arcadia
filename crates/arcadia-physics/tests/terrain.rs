//! Tests for static terrain and bedrock contact generation.
//! Owns CPU unit verification of floor support normals, bedrock contact, and lateral collisions.

use arcadia_physics::{compute_feature_contact, RigidVoxelBody};
mod common;
use arcadia_core::{ChunkTree64, MaterialPalette, MAT_STONE_DEEP, VOXEL_SIZE};
use arcadia_world::WorldMap;
use glam::{IVec3, Mat3, UVec3, Vec3};

fn create_test_world(solid_coords: &[(i32, i32, i32)]) -> WorldMap {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for &(x, y, z) in solid_coords {
        voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
    }
    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette)
}

#[test]
fn test_planar_contact_exact_normals_and_penetration() {
    // WHY: Test voxel at (10, 10, 10) with only +Y exposed to air (floor contact).
    let world_py = create_test_world(&[
        (10, 10, 10),
        (10, 9, 10),  // -Y solid
        (11, 10, 10), // +X solid
        (9, 10, 10),  // -X solid
        (10, 10, 11), // +Z solid
        (10, 10, 9),  // -Z solid
    ]);
    let p_world_py = Vec3::new(10.5, 10.8, 10.5) * VOXEL_SIZE;
    let (norm_py, pen_py) = compute_feature_contact(p_world_py, 10, 10, 10, &world_py);
    assert_eq!(norm_py, Vec3::Y);
    let expected_pen_py = (11.0 - 10.8) * VOXEL_SIZE;
    assert!(
        (pen_py - expected_pen_py).abs() < 1e-6,
        "Penetration must equal exact distance to +Y plane: expected {}, got {}",
        expected_pen_py,
        pen_py
    );

    // WHY: Test voxel with +Y solid and only +X exposed to air (lateral wall).
    let world_px = create_test_world(&[
        (10, 10, 10),
        (10, 11, 10), // +Y solid
        (10, 9, 10),  // -Y solid
        (9, 10, 10),  // -X solid
        (10, 10, 11), // +Z solid
        (10, 10, 9),  // -Z solid
    ]);
    let p_world_px = Vec3::new(10.75, 10.5, 10.5) * VOXEL_SIZE;
    let (norm_px, pen_px) = compute_feature_contact(p_world_px, 10, 10, 10, &world_px);
    assert_eq!(norm_px, Vec3::X);
    let expected_pen_px = (11.0 - 10.75) * VOXEL_SIZE;
    assert!(
        (pen_px - expected_pen_px).abs() < 1e-6,
        "Penetration must equal exact distance to +X plane: expected {}, got {}",
        expected_pen_px,
        pen_px
    );

    // WHY: Test voxel with +Y solid and only -X exposed to air (lateral wall).
    let world_nx = create_test_world(&[
        (10, 10, 10),
        (10, 11, 10), // +Y solid
        (10, 9, 10),  // -Y solid
        (11, 10, 10), // +X solid
        (10, 10, 11), // +Z solid
        (10, 10, 9),  // -Z solid
    ]);
    let p_world_nx = Vec3::new(10.3, 10.5, 10.5) * VOXEL_SIZE;
    let (norm_nx, pen_nx) = compute_feature_contact(p_world_nx, 10, 10, 10, &world_nx);
    assert_eq!(norm_nx, -Vec3::X);
    let expected_pen_nx = (10.3 - 10.0) * VOXEL_SIZE;
    assert!(
        (pen_nx - expected_pen_nx).abs() < 1e-6,
        "Penetration must equal exact distance to -X plane: expected {}, got {}",
        expected_pen_nx,
        pen_nx
    );

    // WHY: Test opposing exposed faces (+X and -X air) selecting the closer boundary plane.
    let world_opposing_x = create_test_world(&[
        (10, 10, 10),
        (10, 11, 10), // +Y solid
        (10, 9, 10),  // -Y solid
        (10, 10, 11), // +Z solid
        (10, 10, 9),  // -Z solid
    ]);
    // Point closer to +X:
    let p_closer_px = Vec3::new(10.7, 10.5, 10.5) * VOXEL_SIZE;
    let (norm_cpx, pen_cpx) = compute_feature_contact(p_closer_px, 10, 10, 10, &world_opposing_x);
    assert_eq!(norm_cpx, Vec3::X);
    assert!((pen_cpx - 0.3 * VOXEL_SIZE).abs() < 1e-6);

    // Point closer to -X:
    let p_closer_nx = Vec3::new(10.2, 10.5, 10.5) * VOXEL_SIZE;
    let (norm_cnx, pen_cnx) = compute_feature_contact(p_closer_nx, 10, 10, 10, &world_opposing_x);
    assert_eq!(norm_cnx, -Vec3::X);
    assert!((pen_cnx - 0.2 * VOXEL_SIZE).abs() < 1e-6);
}

#[test]
fn test_flat_floor_contacts_produce_pure_upward_normals() {
    // WHY: Flat terrain floor at y = 10 must produce pure Vec3::Y normals with zero horizontal components.
    let world = create_test_world(&[
        (10, 10, 10),
        (10, 9, 10),
        (11, 10, 10),
        (9, 10, 10),
        (10, 10, 11),
        (10, 10, 9),
    ]);

    // Test multiple points across the flat floor:
    let test_points = [
        Vec3::new(10.5, 10.95, 10.5),
        Vec3::new(10.1, 10.8, 10.9),
        Vec3::new(10.9, 10.7, 10.1),
    ];

    for &p_world in &test_points {
        let (normal, penetration) =
            compute_feature_contact(p_world * VOXEL_SIZE, 10, 10, 10, &world);
        assert_eq!(normal, Vec3::Y, "Flat floor normal must be pure Vec3::Y");
        assert_eq!(
            normal.x, 0.0,
            "Flat floor normal must have zero horizontal X"
        );
        assert_eq!(
            normal.z, 0.0,
            "Flat floor normal must have zero horizontal Z"
        );
        let expected_pen = (11.0 - p_world.y) * VOXEL_SIZE;
        assert!(
            (penetration - expected_pen).abs() < 1e-6,
            "Penetration must match vertical distance to top surface: expected {}, got {}",
            expected_pen,
            penetration
        );
    }

    // WHY: Voxel at platform edge (+Y and +X air) must still produce pure Vec3::Y normal when resting on top.
    let world_edge = create_test_world(&[
        (10, 10, 10),
        (10, 9, 10),
        (9, 10, 10),
        (10, 10, 11),
        (10, 10, 9),
    ]);
    let p_edge = Vec3::new(10.8, 10.9, 10.5) * VOXEL_SIZE;
    let (norm_edge, pen_edge) = compute_feature_contact(p_edge, 10, 10, 10, &world_edge);
    assert_eq!(
        norm_edge,
        Vec3::Y,
        "Platform edge top surface contact must be pure Vec3::Y to prevent sliding torques"
    );
    assert!((pen_edge - 0.1 * VOXEL_SIZE).abs() < 1e-6);
}

#[test]
fn test_deep_penetration_global_column_surface_search() {
    // WHY: Create solid terrain column from vy = 5 up to vy = 10, with vy = 11 air (surface at 11.0 * VOXEL_SIZE).
    let mut coords = Vec::new();
    for y in 5..=10 {
        coords.push((10, y, 10));
        coords.push((11, y, 10));
        coords.push((9, y, 10));
        coords.push((10, y, 11));
        coords.push((10, y, 9));
    }
    let world = create_test_world(&coords);
    let surface_y = 11.0 * VOXEL_SIZE;

    // 2 voxels deep (vy = 8):
    let p_2_deep = Vec3::new(10.5, 8.8, 10.5) * VOXEL_SIZE;
    let (norm_2, pen_2) = compute_feature_contact(p_2_deep, 10, 8, 10, &world);
    assert_eq!(
        norm_2,
        Vec3::Y,
        "Deep underground contact must push pure upward"
    );
    let expected_pen_2 = surface_y - p_2_deep.y;
    assert!(
        (pen_2 - expected_pen_2).abs() < 1e-6,
        "Penetration must scan upward to surface: expected {}, got {}",
        expected_pen_2,
        pen_2
    );

    // 3 voxels deep (vy = 7):
    let p_3_deep = Vec3::new(10.5, 7.5, 10.5) * VOXEL_SIZE;
    let (norm_3, pen_3) = compute_feature_contact(p_3_deep, 10, 7, 10, &world);
    assert_eq!(norm_3, Vec3::Y);
    let expected_pen_3 = surface_y - p_3_deep.y;
    assert!((pen_3 - expected_pen_3).abs() < 1e-6);

    // 4 voxels deep (vy = 6):
    let p_4_deep = Vec3::new(10.5, 6.2, 10.5) * VOXEL_SIZE;
    let (norm_4, pen_4) = compute_feature_contact(p_4_deep, 10, 6, 10, &world);
    assert_eq!(norm_4, Vec3::Y);
    let expected_pen_4 = surface_y - p_4_deep.y;
    assert!((pen_4 - expected_pen_4).abs() < 1e-6);
}

#[test]
fn test_bedrock_flat_plane_contact() {
    let world = create_test_world(&[]);
    let p_world = Vec3::new(10.0, 4.6 * VOXEL_SIZE, 10.0);

    // INVARIANT: Bedrock is at vy <= 4, surface at y = 5.0 * VOXEL_SIZE.
    let (normal, penetration) = compute_feature_contact(p_world, 10, 4, 10, &world);

    assert_eq!(normal, Vec3::Y);
    let expected_pen = (5.0 - 4.6) * VOXEL_SIZE;
    assert!(
        (penetration - expected_pen).abs() < 1e-6,
        "Bedrock penetration must be 5.0 * VOXEL_SIZE - p_world.y: expected {}, got {}",
        expected_pen,
        penetration
    );

    // Test deeper bedrock level at vy = 2:
    let p_bedrock_deep = Vec3::new(10.0, 2.0 * VOXEL_SIZE, 10.0);
    let (norm_bd, pen_bd) = compute_feature_contact(p_bedrock_deep, 10, 2, 10, &world);
    assert_eq!(norm_bd, Vec3::Y);
    assert!((pen_bd - 3.0 * VOXEL_SIZE).abs() < 1e-6);
}

#[test]
fn test_lateral_wall_contacts_horizontal_normals() {
    // WHY: Create solid cliff wall at vx = 10 (vy from 5 to 15), exposed to air at vx = 9.
    let mut coords = Vec::new();
    for y in 5..=15 {
        coords.push((10, y, 10));
        coords.push((11, y, 10));
        coords.push((10, y, 11));
        coords.push((10, y, 9));
    }
    let world = create_test_world(&coords);

    // Body hits cliff horizontally from -X side at vy = 10:
    let p_wall_nx = Vec3::new(10.2, 10.5, 10.5) * VOXEL_SIZE;
    let (norm_nx, pen_nx) = compute_feature_contact(p_wall_nx, 10, 10, 10, &world);
    assert_eq!(
        norm_nx,
        -Vec3::X,
        "Cliff wall must produce horizontal -Vec3::X normal"
    );
    assert_eq!(
        norm_nx.y, 0.0,
        "Cliff wall normal must have zero vertical component"
    );
    let expected_pen = (10.2 - 10.0) * VOXEL_SIZE;
    assert!((pen_nx - expected_pen).abs() < 1e-6);

    // WHY: Create solid cliff wall at vz = 10, exposed to air at vz = 9.
    let mut coords_z = Vec::new();
    for y in 5..=15 {
        coords_z.push((10, y, 10));
        coords_z.push((11, y, 10));
        coords_z.push((9, y, 10));
        coords_z.push((10, y, 11));
    }
    let world_z = create_test_world(&coords_z);
    let p_wall_nz = Vec3::new(10.5, 10.5, 10.15) * VOXEL_SIZE;
    let (norm_nz, pen_nz) = compute_feature_contact(p_wall_nz, 10, 10, 10, &world_z);
    assert_eq!(
        norm_nz,
        -Vec3::Z,
        "Cliff wall must produce horizontal -Vec3::Z normal"
    );
    assert_eq!(norm_nz.y, 0.0);
    assert!((pen_nz - 0.15 * VOXEL_SIZE).abs() < 1e-6);
}

#[test]
fn test_generate_world_contacts_no_hardcoded_penetration() {
    let palette = MaterialPalette::default();
    let body = RigidVoxelBody::from_global_voxels(
        &[(IVec3::new(0, 0, 0), MAT_STONE_DEEP)],
        &palette,
        None,
    )
    .expect("INVARIANT: Single voxel body must construct");

    // WHY: Create a solid static terrain wall at vx = 10, with only -X exposed to air at vx = 9.
    let world = create_test_world(&[
        (10, 10, 10),
        (11, 10, 10),
        (10, 11, 10),
        (10, 9, 10),
        (10, 10, 11),
        (10, 10, 9),
    ]);

    // WHY: 1x1x1 box has half-extent 0.5 * VOXEL_SIZE; center at 9.7 * s places +X outer face at 10.2 * s (0.2 * s penetration).
    let mut test_body = body.clone();
    test_body.position = Vec3::new(9.7 * VOXEL_SIZE, 10.5 * VOXEL_SIZE, 10.5 * VOXEL_SIZE);

    let inv_i = test_body.world_inv_inertia();
    let r_mat = Mat3::IDENTITY;
    let (contacts_a, _) = test_body.generate_world_contacts(&world, inv_i, r_mat);

    assert!(!contacts_a.is_empty());
    for c in &contacts_a {
        assert!((c.normal - (-Vec3::X)).length() < 1e-4);
    }
    let pen_a = contacts_a[0].penetration;

    // INVARIANT: Penetration must reflect exact outer-box geometry rather than legacy VOXEL_SIZE * 0.35.
    let legacy_hardcoded = VOXEL_SIZE * 0.35;
    assert!(
        (pen_a - legacy_hardcoded).abs() > 0.003,
        "Contact penetration must not equal legacy hardcoded constant {legacy_hardcoded}, got {pen_a}"
    );

    test_body.position.x += 0.05 * VOXEL_SIZE;
    let (contacts_b, _) = test_body.generate_world_contacts(&world, inv_i, r_mat);
    assert!(!contacts_b.is_empty());
    let pen_b = contacts_b[0].penetration;

    let delta_pen = pen_b - pen_a;
    assert!(
        (delta_pen - 0.05 * VOXEL_SIZE).abs() < 1e-4,
        "Contact penetration delta must reflect exact translation: expected {}, got {delta_pen}",
        0.05 * VOXEL_SIZE
    );
}

#[test]
fn test_generate_world_contacts_flat_floor_stability() {
    let palette = MaterialPalette::default();
    let body = RigidVoxelBody::from_global_voxels(
        &[(IVec3::new(0, 0, 0), MAT_STONE_DEEP)],
        &palette,
        None,
    )
    .expect("INVARIANT: Single voxel body must construct");

    let world = create_test_world(&[
        (10, 10, 10),
        (10, 9, 10),
        (11, 10, 10),
        (9, 10, 10),
        (10, 10, 11),
        (10, 10, 9),
    ]);

    // WHY: Center at 11.4 * VOXEL_SIZE places the 1x1x1 cube's outer -Y face at 10.9 * VOXEL_SIZE (0.1 * VOXEL_SIZE penetration into top face 11.0 * VOXEL_SIZE).
    let mut test_body = body;
    test_body.position = Vec3::new(10.5 * VOXEL_SIZE, 11.4 * VOXEL_SIZE, 10.5 * VOXEL_SIZE);

    let inv_i = test_body.world_inv_inertia();
    let r_mat = Mat3::IDENTITY;
    let (contacts, _) = test_body.generate_world_contacts(&world, inv_i, r_mat);

    assert_eq!(
        contacts.len(),
        4,
        "Clipped square floor contact must provide 4 corner support points"
    );
    for c in &contacts {
        assert!((c.normal - Vec3::Y).length() < 1e-4);
        let expected_pen = (0.1 * VOXEL_SIZE - 0.002).max(0.0);
        assert!(
            (c.penetration - expected_pen).abs() < 1e-5,
            "Contact penetration must equal exact outer-face overlap minus 0.002m skin: expected {expected_pen}, got {}",
            c.penetration
        );
    }
}

#[test]
fn test_generate_world_contacts_ceiling_overhang_pushes_downward() {
    let palette = MaterialPalette::default();
    let mut cube_voxels = Vec::new();
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                cube_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let mut body = RigidVoxelBody::from_global_voxels(&cube_voxels, &palette, None)
        .expect("INVARIANT: 2x2x2 body must construct");

    // WHY: Build a solid ceiling slab at vy = 20..=21 with air underneath at vy <= 19.
    let mut ceiling_coords = Vec::new();
    for z in 8..=13 {
        for x in 8..=13 {
            ceiling_coords.push((x, 20, z));
            ceiling_coords.push((x, 21, z));
        }
    }
    let world = create_test_world(&ceiling_coords);

    // WHY: 2x2x2 cube has half-extent 1.0 * VOXEL_SIZE; center at y = 19.2 * s places top face at 20.2 * s (penetrating ceiling bottom 20.0 * s by 0.2 * s).
    body.position = Vec3::new(10.5 * VOXEL_SIZE, 19.2 * VOXEL_SIZE, 10.5 * VOXEL_SIZE);
    let inv_i = body.world_inv_inertia();
    let (contacts, _) = body.generate_world_contacts(&world, inv_i, Mat3::IDENTITY);

    assert!(!contacts.is_empty(), "Ceiling hit must produce contacts");
    for c in &contacts {
        assert!(
            c.normal.y < -0.99,
            "Ceiling contact normal must push downward (-Y), got {:?}",
            c.normal
        );
    }
}

#[test]
fn test_long_toppled_pillar_midpoint_ridge_terrain_contact() {
    let palette = MaterialPalette::default();
    let mut pillar_voxels = Vec::new();
    // WHY: 4x64x4 pillar has 1,024 voxels (>10x legacy 96-point hull limit) to verify zero midpoint blind spots.
    for y in 0..64 {
        for z in 0..4 {
            for x in 0..4 {
                pillar_voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let mut pillar = RigidVoxelBody::from_global_voxels(&pillar_voxels, &palette, None)
        .expect("INVARIANT: 64-voxel-long pillar must construct");

    // Narrow 2-voxel-wide stone ridge at vx in 31..=32, vy in 5..=12 (top at 13 * VOXEL_SIZE):
    let mut ridge = Vec::new();
    for z in 28..=36 {
        for y in 5..=12 {
            ridge.push((31, y, z));
            ridge.push((32, y, z));
        }
    }
    let world = create_test_world(&ridge);

    pillar.orientation = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    // Horizontal shaft has half-thickness 2 * VOXEL_SIZE in Y; center at 14.8 * s places bottom face at 12.8 * s (0.2 * s into ridge top 13.0 * s).
    pillar.position = Vec3::new(32.0 * VOXEL_SIZE, 14.8 * VOXEL_SIZE, 32.0 * VOXEL_SIZE);

    let r_mat = Mat3::from_quat(pillar.orientation);
    let inv_i = pillar.world_inv_inertia();
    let (contacts, _) = pillar.generate_world_contacts(&world, inv_i, r_mat);

    assert!(
        !contacts.is_empty(),
        "4m toppled pillar must collide with narrow midpoint terrain ridge without sampling blind spots"
    );
    assert!(contacts.iter().all(|c| c.normal.y > 0.99));
}
