//! Frustum culling, peripheral margins, and camera visibility tests for rigid voxel bodies.
//! Owns CPU unit verification of bounding sphere culling against perspective frustums.

use arcadia_physics::RigidVoxelBody;
mod common;
use arcadia_core::{MaterialPalette, MAT_STONE_DEEP};
use glam::{IVec3, Vec3};
use std::f32::consts::FRAC_PI_2;

/// Helper creating a valid 2x2x2 rigid voxel body for visibility testing.
fn create_test_body() -> RigidVoxelBody {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }

    RigidVoxelBody::from_global_voxels(&voxels, &palette, None)
        .expect("INVARIANT: Valid voxels must construct test RigidVoxelBody")
}

#[test]
fn test_visible_directly_in_front() {
    let mut body = create_test_body();
    body.position = Vec3::new(0.0, 0.0, 20.0);

    let eye_pos = Vec3::ZERO;
    let forward = Vec3::Z;
    let right = Vec3::X;
    let up = Vec3::Y;
    let fov_y_rad = FRAC_PI_2;
    let aspect_ratio = 1.0;
    let margin_factor = 1.0;

    assert!(body.is_visible_to_camera(
        eye_pos,
        forward,
        right,
        up,
        fov_y_rad,
        aspect_ratio,
        margin_factor
    ));
}

#[test]
fn test_invisible_behind_camera() {
    let mut body = create_test_body();
    let radius = body.bounding_radius();

    let eye_pos = Vec3::ZERO;
    let forward = Vec3::Z;
    let right = Vec3::X;
    let up = Vec3::Y;
    let fov_y_rad = FRAC_PI_2;
    let aspect_ratio = 1.0;
    let margin_factor = 1.0;

    // Completely behind the camera plane:
    body.position = Vec3::new(0.0, 0.0, -(radius + 1.0));
    assert!(!body.is_visible_to_camera(
        eye_pos,
        forward,
        right,
        up,
        fov_y_rad,
        aspect_ratio,
        margin_factor
    ));

    // Far behind the camera:
    body.position = Vec3::new(0.0, 0.0, -50.0);
    assert!(!body.is_visible_to_camera(
        eye_pos,
        forward,
        right,
        up,
        fov_y_rad,
        aspect_ratio,
        margin_factor
    ));
}

#[test]
fn test_invisible_outside_lateral_fov() {
    let mut body = create_test_body();
    let radius = body.bounding_radius();

    let eye_pos = Vec3::ZERO;
    let forward = Vec3::Z;
    let right = Vec3::X;
    let up = Vec3::Y;
    let fov_y_rad = FRAC_PI_2; // 90 deg -> half_h = depth * tan(45 deg) = depth
    let aspect_ratio = 1.0;
    let margin_factor = 1.0;

    let depth_z = 20.0;
    let half_w = depth_z;

    // Right of frustum:
    body.position = Vec3::new(half_w + radius + 1.0, 0.0, depth_z);
    assert!(!body.is_visible_to_camera(
        eye_pos,
        forward,
        right,
        up,
        fov_y_rad,
        aspect_ratio,
        margin_factor
    ));

    // Left of frustum:
    body.position = Vec3::new(-(half_w + radius + 1.0), 0.0, depth_z);
    assert!(!body.is_visible_to_camera(
        eye_pos,
        forward,
        right,
        up,
        fov_y_rad,
        aspect_ratio,
        margin_factor
    ));
}

#[test]
fn test_visible_inside_peripheral_margin() {
    let mut body = create_test_body();
    let radius = body.bounding_radius();

    let eye_pos = Vec3::ZERO;
    let forward = Vec3::Z;
    let right = Vec3::X;
    let up = Vec3::Y;
    let fov_y_rad = FRAC_PI_2;
    let aspect_ratio = 1.0;

    let depth_z = 20.0;
    // Without peripheral margin, half_w is 20.0:
    // Place body slightly outside the strict frustum bound:
    body.position = Vec3::new(20.0 + radius + 0.5, 0.0, depth_z);

    // Margin factor 1.0 (strict frustum): outside, not visible
    assert!(!body.is_visible_to_camera(eye_pos, forward, right, up, fov_y_rad, aspect_ratio, 1.0));

    // Margin factor 1.15 (15% peripheral tolerance): half_w = 23.0, now inside!
    assert!(body.is_visible_to_camera(eye_pos, forward, right, up, fov_y_rad, aspect_ratio, 1.15));
}

#[test]
fn test_invisible_outside_vertical_fov() {
    let mut body = create_test_body();
    let radius = body.bounding_radius();

    let eye_pos = Vec3::ZERO;
    let forward = Vec3::Z;
    let right = Vec3::X;
    let up = Vec3::Y;
    let fov_y_rad = FRAC_PI_2;
    let aspect_ratio = 1.0;
    let margin_factor = 1.0;

    let depth_z = 20.0;
    let half_h = depth_z;

    // Above frustum:
    body.position = Vec3::new(0.0, half_h + radius + 1.0, depth_z);
    assert!(!body.is_visible_to_camera(
        eye_pos,
        forward,
        right,
        up,
        fov_y_rad,
        aspect_ratio,
        margin_factor
    ));

    // Below frustum:
    body.position = Vec3::new(0.0, -(half_h + radius + 1.0), depth_z);
    assert!(!body.is_visible_to_camera(
        eye_pos,
        forward,
        right,
        up,
        fov_y_rad,
        aspect_ratio,
        margin_factor
    ));
}

#[test]
fn test_max_draw_distance() {
    let mut body = create_test_body();
    let radius = body.bounding_radius();

    let eye_pos = Vec3::ZERO;
    let forward = Vec3::Z;
    let right = Vec3::X;
    let up = Vec3::Y;
    let fov_y_rad = FRAC_PI_2;
    let aspect_ratio = 1.0;
    let margin_factor = 1.0;

    // Within draw distance (140m < 150m):
    body.position = Vec3::new(0.0, 0.0, 140.0);
    assert!(body.is_visible_to_camera(
        eye_pos,
        forward,
        right,
        up,
        fov_y_rad,
        aspect_ratio,
        margin_factor
    ));

    // Exceeding draw distance (150m + radius + 1m):
    body.position = Vec3::new(0.0, 0.0, 150.0 + radius + 1.0);
    assert!(!body.is_visible_to_camera(
        eye_pos,
        forward,
        right,
        up,
        fov_y_rad,
        aspect_ratio,
        margin_factor
    ));
}

#[test]
fn test_arbitrary_camera_pose() {
    let mut body = create_test_body();

    // Camera at (50, 10, -30) looking toward -X:
    let eye_pos = Vec3::new(50.0, 10.0, -30.0);
    let forward = -Vec3::X;
    let right = Vec3::Z;
    let up = Vec3::Y;
    let fov_y_rad = FRAC_PI_2;
    let aspect_ratio = 16.0 / 9.0;
    let margin_factor = 1.0;

    // Body at (30, 10, -30), which is 20m directly in forward direction (-X):
    body.position = Vec3::new(30.0, 10.0, -30.0);
    assert!(body.is_visible_to_camera(
        eye_pos,
        forward,
        right,
        up,
        fov_y_rad,
        aspect_ratio,
        margin_factor
    ));

    // Body at (70, 10, -30), which is 20m in the opposite direction (+X, behind camera):
    body.position = Vec3::new(70.0, 10.0, -30.0);
    assert!(!body.is_visible_to_camera(
        eye_pos,
        forward,
        right,
        up,
        fov_y_rad,
        aspect_ratio,
        margin_factor
    ));
}
