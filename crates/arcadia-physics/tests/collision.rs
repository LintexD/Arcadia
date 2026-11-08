//! Tests for OBB SAT intersection and contact manifold generation.
//! Owns CPU verification of 15-axis SAT OBB vs AABB and OBB vs OBB clipping.

use arcadia_core::{MaterialPalette, MAT_STONE_DEEP};
use arcadia_physics::{BodyPairManifold, ContactPoint3D, RigidVoxelBody};
use glam::{IVec3, Quat, Vec3};

mod common;

#[test]
fn test_obb_aabb_sat_intersection() {
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
        .expect("INVARIANT: Valid voxels and palette must create a rigid voxel body");
    body.position = Vec3::new(10.0, 10.0, 10.0);
    body.orientation = Quat::from_rotation_y(0.785); // 45 deg rotation

    // Overlapping AABB:
    assert!(body.obb_intersects_aabb(Vec3::new(9.9, 9.9, 9.9), Vec3::new(10.1, 10.1, 10.1)));
    // Separated AABB:
    assert!(!body.obb_intersects_aabb(Vec3::new(11.0, 10.0, 10.0), Vec3::new(12.0, 11.0, 11.0)));
}

#[test]
fn test_contact_manifold_3d_defaults() {
    let point = ContactPoint3D::default();
    assert_eq!(point.position, Vec3::ZERO);
    assert_eq!(point.penetration, 0.0);
    assert_eq!(point.accum_n, 0.0);
    assert_eq!(point.feature_id, 0);

    let manifold = BodyPairManifold::default();
    assert_eq!(manifold.body_a_idx, 0);
    assert_eq!(manifold.body_b_idx, 0);
    assert_eq!(manifold.point_count, 0);
    assert_eq!(manifold.points.len(), 4);
    assert_eq!(manifold.friction, 0.0);
    assert_eq!(manifold.restitution, 0.0);

    let copied = manifold;
    assert_eq!(copied.point_count, manifold.point_count);
}

#[test]
fn test_obb_obb_sat_face_contact_manifold() {
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

    // Body A has half-extent 0.125m in Y, centered at Y = 0.125m (top at Y = 0.25m).
    // Stacking Body B directly on top with 0.02m overlap:
    // Bottom of Body B should be at Y = 0.25 - 0.02 = 0.23m.
    // Center of Body B should be at Y = 0.23 + 0.125 = 0.355m.
    body_b.position = Vec3::new(body_a.position.x, 0.355, body_a.position.z);

    let manifold = RigidVoxelBody::compute_obb_obb_manifold(&body_a, &body_b)
        .expect("INVARIANT: Overlapping stacked boxes must produce contact manifold");

    assert_eq!(manifold.point_count, 4);
    assert!(
        (manifold.normal - Vec3::Y).length() < 1e-4,
        "Normal must point approximately along +Y, got {:?}",
        manifold.normal
    );

    for (idx, pt) in manifold.points[..manifold.point_count].iter().enumerate() {
        assert!(
            (pt.penetration - 0.02).abs() < 1e-4,
            "Contact point {} penetration must be ~0.02m, got {}",
            idx,
            pt.penetration
        );
    }
}

#[test]
fn test_obb_obb_sat_separated() {
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

    // Separate boxes by 1m along Y:
    body_b.position = body_a.position + Vec3::new(0.0, 1.0, 0.0);

    let manifold = RigidVoxelBody::compute_obb_obb_manifold(&body_a, &body_b);
    assert!(manifold.is_none(), "Separated boxes must return None");
}

#[test]
fn test_obb_obb_sat_edge_edge_contact_manifold() {
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
    body_a.orientation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);

    body_b.position = Vec3::new(10.0, 10.0, 10.27);
    body_b.orientation = Quat::from_rotation_x(std::f32::consts::FRAC_PI_4);

    let manifold = RigidVoxelBody::compute_obb_obb_manifold(&body_a, &body_b);
    assert!(
        manifold.is_some(),
        "Overlapping crossed edges must produce contact manifold"
    );
    let m = manifold.expect("INVARIANT: Manifold verified");
    assert!(m.point_count >= 1 && m.point_count <= 4);
    assert!(m.points[0].penetration > 0.0);
}

#[test]
fn test_elongated_crossing_pillars_collision() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for x in 0..4 {
        voxels.push((IVec3::new(x, 0, 0), MAT_STONE_DEEP));
    }

    let mut body_a = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create pillar A");
    let mut body_b = RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels and palette must create pillar B");

    // Position Body A horizontally along X at Y = 10.0:
    body_a.position = Vec3::new(10.0, 10.0, 10.0);
    body_a.orientation = Quat::IDENTITY;

    // Position Body B crossing diagonally (45 deg Y-axis rotation) overlapping Body A:
    // With 4x1x1 voxels, height is 1 voxel = 0.0625m.
    // Half-extent Y is 0.03125m.
    // If Body B is at Y = 10.0 + 0.04m, its bottom is at 10.00875m, which penetrates 10.03125m by ~0.0225m.
    body_b.position = Vec3::new(10.0, 10.04, 10.0);
    body_b.orientation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);

    // 1. Broadphase overlap check:
    assert!(
        RigidVoxelBody::check_broadphase_overlap(&body_a, &body_b),
        "Crossing pillars must pass broadphase overlap check"
    );

    // 2. Narrowphase manifold generation (OBB SAT or voxel sampling fallback):
    let maybe_manifold = RigidVoxelBody::compute_obb_obb_manifold(&body_a, &body_b)
        .filter(|m| m.point_count > 0)
        .or_else(|| RigidVoxelBody::detect_contact_manifold(&body_a, &body_b));

    assert!(
        maybe_manifold.is_some(),
        "Crossing pillars must produce a valid contact manifold"
    );

    let manifold = maybe_manifold.expect("INVARIANT: Valid manifold checked");
    assert!(
        manifold.point_count > 0,
        "Manifold must contain at least 1 contact point"
    );
    for i in 0..manifold.point_count {
        assert!(
            manifold.points[i].penetration > 0.0,
            "Contact point {} must have positive penetration, got {}",
            i,
            manifold.points[i].penetration
        );
    }
}
