//! `CameraUniforms`, Reverse-Z infinite perspective projection, and Halton sub-pixel temporal jitter calculations.
//! Owns CPU camera matrices, view-projection reprojection, and uniform GPU layout.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

/// GPU-shared camera uniforms pushed to `@group(0) @binding(0)` every frame.
// GPU-SYNC: Mirrors `CameraUniforms` in `shaders/common/types.wgsl` (@group(0) @binding(0), 480 bytes, 16-byte aligned).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniforms {
    pub cam_pos_time: [f32; 4],
    pub cam_forward_fov: [f32; 4],
    pub cam_right_aspect: [f32; 4],
    pub cam_up_frame: [f32; 4],
    /// Previous frame unjittered view-projection matrix (infinite Reverse-Z: near=1.0, far=0.0).
    pub prev_view_proj: [[f32; 4]; 4],
    /// Current frame unjittered view-projection matrix (infinite Reverse-Z: near=1.0, far=0.0).
    pub curr_view_proj: [[f32; 4]; 4],
    pub world_grid_dims: [u32; 4],
    pub sun_dir_intensity: [f32; 4],
    pub jitter_brush_dt: [f32; 4],
    pub target_voxel_info: [f32; 4],
    pub point_light_count: [u32; 4],
    pub point_light_0: [f32; 4],
    pub point_light_1: [f32; 4],
    pub point_light_2: [f32; 4],
    pub point_light_3: [f32; 4],
    pub point_light_4: [f32; 4],
    pub point_light_5: [f32; 4],
    pub point_light_6: [f32; 4],
    pub point_light_7: [f32; 4],
    pub ray_steps: [u32; 4],
    pub ray_distances: [f32; 4],
    pub lod_settings: [f32; 4],
    pub quality_flags: [u32; 4],
    pub prev_cam_pos: [f32; 4],
}

/// Computes the n-th value in the low-discrepancy Halton sequence with prime base `base`.
#[inline]
pub fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    let b = base as f32;
    while index > 0 {
        f /= b;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// Evaluates sub-pixel Halton jitter `(jx, jy)` in pixel coordinates based on jitter mode and camera movement.
#[inline]
pub fn compute_subpixel_jitter(
    jitter_mode: u32,
    frame_index: u64,
    render_scale: f32,
    is_moving: bool,
    stationary_frames: u32,
) -> (f32, f32) {
    match jitter_mode {
        0 => (0.0, 0.0), // Always Off
        1 => {
            // Freeze when still: jitter only during movement, freeze to exact pixel center when stationary
            if is_moving || stationary_frames < 1 {
                let jitter_amp = if render_scale < 0.95 { 0.72 } else { 0.36 };
                let seq = (frame_index % 16) as u32 + 1;
                (
                    (halton(seq, 2) - 0.5) * jitter_amp,
                    (halton(seq, 3) - 0.5) * jitter_amp,
                )
            } else {
                (0.0, 0.0)
            }
        }
        _ => {
            // Always On: continuous 16-phase Halton
            let jitter_amp = if render_scale < 0.95 { 0.72 } else { 0.36 };
            let seq = (frame_index % 16) as u32 + 1;
            (
                (halton(seq, 2) - 0.5) * jitter_amp,
                (halton(seq, 3) - 0.5) * jitter_amp,
            )
        }
    }
}

/// Builds the infinite Reverse-Z perspective projection matrix (`near = 1.0, far = 0.0`).
///
/// Exploits IEEE 754 floating-point density near zero ($1/z$), delivering uniform millimeter
/// precision across infinite view distances and eliminating distant depth quantization.
#[inline]
pub fn reverse_z_perspective_matrix(fov_y_radians: f32, aspect_ratio: f32, z_near: f32) -> Mat4 {
    Mat4::perspective_infinite_reverse_rh(fov_y_radians, aspect_ratio, z_near)
}

/// Evaluates reverse-Z NDC depth from view-space linear distance along the camera forward axis.
///
/// Returns 1.0 at `z_near` and approaches 0.0 as distance approaches infinity.
#[inline]
pub fn linear_distance_to_reverse_z(view_dist_z: f32, z_near: f32) -> f32 {
    if view_dist_z <= 0.0 {
        0.0
    } else {
        z_near / view_dist_z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    #[test]
    fn test_reverse_z_projection_near_and_far() {
        let fov_y = 60.0f32.to_radians();
        let aspect = 16.0 / 9.0;
        let z_near = 0.1;
        let proj = reverse_z_perspective_matrix(fov_y, aspect, z_near);

        // Near plane point: in right-handed camera space, forward view is -Z:
        let near_view = Vec4::new(0.0, 0.0, -z_near, 1.0);
        let near_clip = proj * near_view;
        let near_ndc_z = near_clip.z / near_clip.w;
        assert!(
            (near_ndc_z - 1.0).abs() < 1e-5,
            "Reverse-Z near plane must map to 1.0, got {near_ndc_z}"
        );

        // Distant point at 100 meters:
        let dist_view = Vec4::new(0.0, 0.0, -100.0, 1.0);
        let dist_clip = proj * dist_view;
        let dist_ndc_z = dist_clip.z / dist_clip.w;
        assert!(
            (dist_ndc_z - (z_near / 100.0)).abs() < 1e-5,
            "Reverse-Z 100m must match z_near / z, got {dist_ndc_z}"
        );

        // Extremely far point at 100,000 meters:
        let far_view = Vec4::new(0.0, 0.0, -100000.0, 1.0);
        let far_clip = proj * far_view;
        let far_ndc_z = far_clip.z / far_clip.w;
        assert!(
            far_ndc_z > 0.0 && far_ndc_z < 1e-4,
            "Reverse-Z far point must approach 0.0, got {far_ndc_z}"
        );
    }

    #[test]
    fn test_linear_distance_to_reverse_z() {
        let z_near = 0.1;
        assert_eq!(linear_distance_to_reverse_z(0.1, z_near), 1.0);
        assert!((linear_distance_to_reverse_z(100.0, z_near) - 0.001).abs() < 1e-6);
        assert_eq!(linear_distance_to_reverse_z(0.0, z_near), 0.0);
    }

    #[test]
    fn test_camera_uniforms_layout() {
        assert_eq!(std::mem::size_of::<CameraUniforms>(), 480);
        assert_eq!(std::mem::align_of::<CameraUniforms>(), 16);
    }
}
