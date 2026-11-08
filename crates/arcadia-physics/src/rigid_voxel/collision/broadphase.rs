//! Oriented Bounding Box (OBB) representation and hierarchical broadphase culling.
//! Owns CPU bounding volume queries, 15-axis SAT OBB vs AABB tests, and pair culling.

use super::super::types::RigidVoxelBody;
use super::compound::VoxelSubBox;
use arcadia_core::VOXEL_SIZE;
use glam::{Mat3, Vec3};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Obb {
    pub(crate) center: Vec3,
    pub(crate) axes: [Vec3; 3],
    pub(crate) extents: Vec3,
}

impl Obb {
    /// Builds the macro world-space OBB enclosing the entire rigid body's `dims_v`.
    #[inline]
    pub(crate) fn from_body(body: &RigidVoxelBody) -> Self {
        let r = Mat3::from_quat(body.orientation);
        let axes = [r.x_axis, r.y_axis, r.z_axis];
        let local_center_offset = body.half_extents_m - body.com_local_m;
        let center = body.position + r * local_center_offset;
        Self {
            center,
            axes,
            extents: body.half_extents_m,
        }
    }

    /// Builds the exact world-space OBB for a single `VoxelSubBox` given body world CoM and rotation matrix.
    #[inline]
    pub(crate) fn from_sub_box(sub: &VoxelSubBox, body_pos: Vec3, r_mat: Mat3) -> Self {
        Self {
            center: body_pos + r_mat * sub.center_rel_com,
            axes: [r_mat.x_axis, r_mat.y_axis, r_mat.z_axis],
            extents: sub.half_extents,
        }
    }

    /// Builds an axis-aligned world-space OBB from `[min_m .. max_m]`.
    #[inline]
    pub(crate) fn from_aabb(min_m: Vec3, max_m: Vec3) -> Self {
        Self {
            center: (min_m + max_m) * 0.5,
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            extents: (max_m - min_m) * 0.5,
        }
    }
}

impl RigidVoxelBody {
    /// Tests whether this body's macro OBB intersects `[aabb_min .. aabb_max]` using 15-axis SAT.
    pub fn obb_intersects_aabb(&self, aabb_min: Vec3, aabb_max: Vec3) -> bool {
        let aabb_center = (aabb_min + aabb_max) * 0.5;
        let aabb_extents = (aabb_max - aabb_min) * 0.5;
        let obb = Obb::from_body(self);
        obb_intersects_aabb_sat(&obb, aabb_center, aabb_extents)
    }

    /// Tests whether this body's exact compound voxel shape intersects `[aabb_min .. aabb_max]`.
    #[inline]
    pub fn compound_intersects_aabb(&self, aabb_min: Vec3, aabb_max: Vec3) -> bool {
        self.collider
            .intersects_world_aabb(self.position, self.orientation, aabb_min, aabb_max)
    }

    /// Hierarchical broadphase overlap culling: bounding sphere, world AABB, and macro 15-axis SAT.
    pub fn check_broadphase_overlap(body_a: &Self, body_b: &Self) -> bool {
        let obb_a = Obb::from_body(body_a);
        let obb_b = Obb::from_body(body_b);

        let rad_sum = obb_a.extents.length() + obb_b.extents.length() + 2.0 * VOXEL_SIZE;
        if obb_a.center.distance_squared(obb_b.center) > rad_sum * rad_sum {
            return false;
        }

        let (min_a, max_a) = body_a.world_aabb_meters();
        let (min_b, max_b) = body_b.world_aabb_meters();
        if max_a.x < min_b.x
            || min_a.x > max_b.x
            || max_a.y < min_b.y
            || min_a.y > max_b.y
            || max_a.z < min_b.z
            || min_a.z > max_b.z
        {
            return false;
        }

        obb_intersects_obb_sat(&obb_a, &obb_b)
    }
}

pub(crate) fn obb_intersects_aabb_sat(obb: &Obb, aabb_center: Vec3, aabb_extents: Vec3) -> bool {
    let u = obb.axes;
    let t_world = obb.center - aabb_center;
    let eps = 1e-5f32;

    for i in 0..3 {
        let ra = aabb_extents[i];
        let rb = obb.extents.x * u[0][i].abs()
            + obb.extents.y * u[1][i].abs()
            + obb.extents.z * u[2][i].abs();
        if t_world[i].abs() > ra + rb + eps {
            return false;
        }
    }

    for (i, &axis_u) in u.iter().enumerate() {
        let ra = aabb_extents.x * axis_u.x.abs()
            + aabb_extents.y * axis_u.y.abs()
            + aabb_extents.z * axis_u.z.abs();
        let rb = obb.extents[i];
        if t_world.dot(axis_u).abs() > ra + rb + eps {
            return false;
        }
    }

    let e = [Vec3::X, Vec3::Y, Vec3::Z];
    for &axis_e in &e {
        for &uj in &u {
            let axis = axis_e.cross(uj);
            if axis.length_squared() < 1e-6 {
                continue;
            }
            let ra = aabb_extents.x * axis.x.abs()
                + aabb_extents.y * axis.y.abs()
                + aabb_extents.z * axis.z.abs();
            let rb = obb.extents.x * u[0].dot(axis).abs()
                + obb.extents.y * u[1].dot(axis).abs()
                + obb.extents.z * u[2].dot(axis).abs();
            if t_world.dot(axis).abs() > ra + rb + eps {
                return false;
            }
        }
    }

    true
}

fn obb_intersects_obb_sat(obb_a: &Obb, obb_b: &Obb) -> bool {
    let d_vec = obb_b.center - obb_a.center;
    let eps = 1e-5f32;

    for axis in obb_a.axes {
        if !axis_overlaps(axis, obb_a, obb_b, d_vec, eps) {
            return false;
        }
    }
    for axis in obb_b.axes {
        if !axis_overlaps(axis, obb_a, obb_b, d_vec, eps) {
            return false;
        }
    }
    for ua in obb_a.axes {
        for ub in obb_b.axes {
            let cross = ua.cross(ub);
            let len_sq = cross.length_squared();
            if len_sq < 1e-6 {
                continue;
            }
            if !axis_overlaps(cross / len_sq.sqrt(), obb_a, obb_b, d_vec, eps) {
                return false;
            }
        }
    }
    true
}

#[inline]
fn axis_overlaps(axis: Vec3, obb_a: &Obb, obb_b: &Obb, d_vec: Vec3, eps: f32) -> bool {
    let ra = obb_a.extents.x * obb_a.axes[0].dot(axis).abs()
        + obb_a.extents.y * obb_a.axes[1].dot(axis).abs()
        + obb_a.extents.z * obb_a.axes[2].dot(axis).abs();
    let rb = obb_b.extents.x * obb_b.axes[0].dot(axis).abs()
        + obb_b.extents.y * obb_b.axes[1].dot(axis).abs()
        + obb_b.extents.z * obb_b.axes[2].dot(axis).abs();
    d_vec.dot(axis).abs() <= ra + rb + eps
}
