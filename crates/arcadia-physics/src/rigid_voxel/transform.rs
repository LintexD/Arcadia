//! Transform matrices, inertia rotation, AABB bounds, and GPU instance generation.
//! Owns CPU coordinate frame transformation, inertia rotation, and camera frustum culling.

use super::types::{GpuRigidBodyInstance, RigidVoxelBody};
use arcadia_core::VOXEL_SIZE;
use glam::{Mat3, Mat4, Quat, Vec3};

impl RigidVoxelBody {
    /// World-space inverse inertia tensor $\mathbf{I}_{\text{world}}^{-1} = \mathbf{R} \mathbf{I}_{\text{body}}^{-1} \mathbf{R}^T$.
    #[inline]
    pub fn world_inv_inertia(&self) -> Mat3 {
        let r = Mat3::from_quat(self.orientation);
        r * self.inv_inertia_body * r.transpose()
    }

    /// Evaluates whether this rigid body's bounding volume is visible to the given camera frustum,
    /// including a peripheral margin factor (e.g. 1.15 for 15% extra peripheral tolerance).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn is_visible_to_camera(
        &self,
        eye_pos: Vec3,
        forward: Vec3,
        right: Vec3,
        up: Vec3,
        fov_y_rad: f32,
        aspect_ratio: f32,
        margin_factor: f32,
    ) -> bool {
        const MAX_DRAW_DISTANCE_M: f32 = 150.0;

        let r = Mat3::from_quat(self.orientation);
        let sphere_center = self.position + r * (self.half_extents_m - self.com_local_m);
        let radius = self.bounding_radius();

        let to_center = sphere_center - eye_pos;
        let depth_z = to_center.dot(forward);

        if depth_z < -radius {
            return false;
        }
        if depth_z > MAX_DRAW_DISTANCE_M + radius {
            return false;
        }

        let half_h = depth_z.max(0.0) * (fov_y_rad * 0.5).tan() * margin_factor;
        let half_w = half_h * aspect_ratio;
        let lateral_x = to_center.dot(right).abs();
        let lateral_y = to_center.dot(up).abs();

        if lateral_x > half_w + radius || lateral_y > half_h + radius {
            return false;
        }

        true
    }

    /// Computes the world-space Axis-Aligned Bounding Box `(min_m, max_m)` enclosing the rotated OBB.
    #[inline]
    pub fn world_aabb_meters(&self) -> (Vec3, Vec3) {
        let r = Mat3::from_quat(self.orientation);
        let abs_r = Mat3::from_cols(r.x_axis.abs(), r.y_axis.abs(), r.z_axis.abs());
        let local_center_offset = self.half_extents_m - self.com_local_m;
        let world_box_center = self.position + r * local_center_offset;
        let world_half_extents = abs_r * self.half_extents_m;
        (
            world_box_center - world_half_extents,
            world_box_center + world_half_extents,
        )
    }

    /// Builds the 4x4 transform mapping body local tree coordinates (meters in `[0 .. dims_v * VOXEL_SIZE]`)
    /// to world-space meters: $\mathbf{p}_{\text{world}} = \mathbf{x}_{\text{CoM}} + \mathbf{R}(\mathbf{q}) (\mathbf{p}_{\text{local}} - \mathbf{c}_{\text{local}})$.
    #[inline]
    pub fn local_to_world_matrix(&self) -> Mat4 {
        Self::compose_local_to_world(self.position, self.orientation, self.com_local_m)
    }

    /// Produces the GPU-ready `GpuRigidBodyInstance` descriptor for `primary_gbuffer.wgsl`.
    pub fn to_gpu_instance(&self) -> GpuRigidBodyInstance {
        let local_to_world = self.local_to_world_matrix();
        let world_to_local = local_to_world.inverse();
        let prev_local_to_world = Self::compose_local_to_world(
            self.prev_position,
            self.prev_orientation,
            self.com_local_m,
        );

        let local_max_m = self.dims_v.as_vec3() * VOXEL_SIZE;
        let r = Mat3::from_quat(self.orientation);
        let local_center_offset = self.half_extents_m - self.com_local_m;
        let sphere_center = self.position + r * local_center_offset;
        let sphere_radius = self.half_extents_m.length() + VOXEL_SIZE;

        GpuRigidBodyInstance {
            world_to_local: world_to_local.to_cols_array_2d(),
            local_to_world: local_to_world.to_cols_array_2d(),
            prev_local_to_world: prev_local_to_world.to_cols_array_2d(),
            local_bounds_and_root: [
                local_max_m.x,
                local_max_m.y,
                local_max_m.z,
                f32::from_bits(self.gpu_root_node_idx),
            ],
            world_sphere: [
                sphere_center.x,
                sphere_center.y,
                sphere_center.z,
                sphere_radius,
            ],
        }
    }

    /// Exact CoM-centered bounding sphere radius in meters enclosing all occupied voxel bounds `[0 .. 2 * half_extents_m]`.
    #[inline]
    pub fn bounding_radius(&self) -> f32 {
        let full_extents = self.half_extents_m * 2.0;
        let max_arm = self.com_local_m.max(full_extents - self.com_local_m);
        max_arm.length() + VOXEL_SIZE
    }

    #[inline]
    pub(crate) fn compose_local_to_world(pos: Vec3, rot: Quat, com_local_m: Vec3) -> Mat4 {
        let r = Mat3::from_quat(rot);
        let translation = pos - r * com_local_m;
        Mat4::from_cols(
            r.x_axis.extend(0.0),
            r.y_axis.extend(0.0),
            r.z_axis.extend(0.0),
            translation.extend(1.0),
        )
    }
}
