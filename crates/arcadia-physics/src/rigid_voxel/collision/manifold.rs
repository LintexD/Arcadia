//! Exposed-face-aware 15-axis SAT box-vs-box narrowphase and compound BVH manifold generation.
//! Owns CPU Sutherland-Hodgman polygon clipping, tandem BVH traversal, and directional contact clustering.

use super::super::types::{BodyPairManifold, ContactPoint3D, RigidVoxelBody};
use super::broadphase::Obb;
use super::compound::{is_axis_face_exposed, ALL_FACES_EXPOSED};
use glam::{Mat3, Vec3};

const MAX_MANIFOLDS_PER_PAIR: usize = 4;
const NORMAL_CLUSTER_COS_THRESHOLD: f32 = 0.85;
const DEDUP_DIST_SQ: f32 = 0.004 * 0.004;

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawBoxContact {
    /// Unit normal pointing from Box A toward Box B in world space.
    pub(crate) normal: Vec3,
    /// Clipped contact points `(world_position, penetration_depth)`.
    pub(crate) points: [(Vec3, f32); 4],
    pub(crate) point_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ChosenAxis {
    FaceA { axis_idx: usize, sign: f32 },
    FaceB { axis_idx: usize, sign: f32 },
    Edge { i: usize, j: usize, normal: Vec3 },
}

#[derive(Clone, Debug)]
struct NormalCluster {
    normal_sum: Vec3,
    unit_normal: Vec3,
    max_pen: f32,
    samples: Vec<(Vec3, f32)>,
}

impl RigidVoxelBody {
    /// Computes the primary multi-point contact manifold between two rigid bodies using their exact compound voxel shapes.
    pub fn compute_obb_obb_manifold(body_a: &Self, body_b: &Self) -> Option<BodyPairManifold> {
        Self::compute_obb_obb_manifold_indexed(body_a, body_b, 0, 1)
    }

    /// Computes the primary indexed multi-point contact manifold between two rigid bodies' compound voxel shapes.
    pub fn compute_obb_obb_manifold_indexed(
        body_a: &Self,
        body_b: &Self,
        body_a_idx: usize,
        body_b_idx: usize,
    ) -> Option<BodyPairManifold> {
        Self::compute_compound_pair_manifolds_indexed(body_a, body_b, body_a_idx, body_b_idx)
            .into_iter()
            .next()
    }

    /// Computes up to 4 directional multi-point contact manifolds between two compound rigid voxel bodies,
    /// enabling concave interlocking (crossed fluted pillars, arches, L-brackets) with zero macro-OBB air padding.
    pub fn compute_compound_pair_manifolds_indexed(
        body_a: &Self,
        body_b: &Self,
        body_a_idx: usize,
        body_b_idx: usize,
    ) -> Vec<BodyPairManifold> {
        if body_a.collider.bvh_nodes.is_empty() || body_b.collider.bvh_nodes.is_empty() {
            return Vec::new();
        }

        let r_a = Mat3::from_quat(body_a.orientation);
        let r_b = Mat3::from_quat(body_b.orientation);
        let inv_r_b = r_b.transpose();

        // Transform Body A CoM-relative space -> Body B CoM-relative space once for the entire pair:
        let r_ab = inv_r_b * r_a;
        let abs_r_ab = Mat3::from_cols(r_ab.x_axis.abs(), r_ab.y_axis.abs(), r_ab.z_axis.abs());
        let t_ab = inv_r_b * (body_a.position - body_b.position);

        let fallback_dir = {
            let delta = body_b.position - body_a.position;
            if delta.length_squared() > 1e-6 {
                delta.normalize()
            } else {
                Vec3::Y
            }
        };

        let overlapping_pairs =
            gather_overlapping_subbox_pairs(body_a, body_b, r_ab, abs_r_ab, t_ab);
        if overlapping_pairs.is_empty() {
            return Vec::new();
        }

        let mut clusters: Vec<NormalCluster> = Vec::new();
        let mut deeply_overlapping_pairs: Vec<(usize, usize)> = Vec::new();

        for &(ia, ib) in &overlapping_pairs {
            let sub_a = &body_a.collider.sub_boxes[ia];
            let sub_b = &body_b.collider.sub_boxes[ib];
            let obb_a = Obb::from_sub_box(sub_a, body_a.position, r_a);
            let obb_b = Obb::from_sub_box(sub_b, body_b.position, r_b);

            if let Some(raw) = compute_subbox_obb_manifold(
                &obb_a,
                sub_a.exposed_faces,
                &obb_b,
                sub_b.exposed_faces,
                fallback_dir,
            ) {
                insert_raw_contact(&mut clusters, &raw);
            } else if compute_subbox_obb_manifold(
                &obb_a,
                ALL_FACES_EXPOSED,
                &obb_b,
                ALL_FACES_EXPOSED,
                fallback_dir,
            )
            .is_some()
            {
                deeply_overlapping_pairs.push((ia, ib));
            }
        }

        // WHY: Fallback only when an object is 100% buried inside interior sub-boxes without touching any outer shell box.
        if clusters.is_empty() && !deeply_overlapping_pairs.is_empty() {
            for (ia, ib) in deeply_overlapping_pairs {
                let sub_a = &body_a.collider.sub_boxes[ia];
                let sub_b = &body_b.collider.sub_boxes[ib];
                let obb_a = Obb::from_sub_box(sub_a, body_a.position, r_a);
                let obb_b = Obb::from_sub_box(sub_b, body_b.position, r_b);
                if let Some(raw) = compute_subbox_obb_manifold(
                    &obb_a,
                    ALL_FACES_EXPOSED,
                    &obb_b,
                    ALL_FACES_EXPOSED,
                    fallback_dir,
                ) {
                    insert_raw_contact(&mut clusters, &raw);
                }
            }
        }

        if clusters.is_empty() {
            return Vec::new();
        }

        // Sort clusters descending by deepest penetration so primary manifold is always index 0:
        clusters.sort_by(|a, b| b.max_pen.total_cmp(&a.max_pen));
        clusters.truncate(MAX_MANIFOLDS_PER_PAIR);

        let combined_friction = (body_a.friction + body_b.friction) * 0.5;
        let combined_restitution = (body_a.restitution + body_b.restitution) * 0.5;

        let mut out = Vec::with_capacity(clusters.len());
        for (cluster_idx, cluster) in clusters.into_iter().enumerate() {
            let normal = cluster.unit_normal;
            let (tangent1, tangent2) = orthonormal_basis(normal);
            let final_pts = reduce_contact_samples(&cluster.samples, tangent1, tangent2, 4);
            if final_pts.is_empty() {
                continue;
            }

            let mut points = [ContactPoint3D::default(); 4];
            for (k, &(pos, depth)) in final_pts.iter().enumerate() {
                points[k] = ContactPoint3D {
                    position: pos,
                    r_a: pos - body_a.position,
                    r_b: pos - body_b.position,
                    penetration: depth,
                    accum_n: 0.0,
                    accum_t1: 0.0,
                    accum_t2: 0.0,
                    effective_mass_n: 0.0,
                    effective_mass_t1: 0.0,
                    effective_mass_t2: 0.0,
                    feature_id: ((cluster_idx as u32) << 8) | (k as u32),
                };
            }

            out.push(BodyPairManifold {
                body_a_idx,
                body_b_idx,
                normal,
                tangent1,
                tangent2,
                points,
                point_count: final_pts.len(),
                friction: combined_friction,
                restitution: combined_restitution,
            });
        }

        out
    }
}

type EdgeCandidate = (Option<(Vec3, f32)>, usize, usize);

/// Computes an exact 15-axis SAT contact manifold between two convex cuboids while disallowing internal seams.
pub(crate) fn compute_subbox_obb_manifold(
    obb_a: &Obb,
    exposed_a: u8,
    obb_b: &Obb,
    exposed_b: u8,
    fallback_dir_a_to_b: Vec3,
) -> Option<RawBoxContact> {
    let d_vec = obb_b.center - obb_a.center;

    let mut sum_rad_a = [0.0f32; 3];
    let mut d_signed_a = [0.0f32; 3];
    for i in 0..3 {
        let axis = obb_a.axes[i];
        let (ra, rb) = project_radii(axis, obb_a, obb_b);
        let d = d_vec.dot(axis);
        if d.abs() >= ra + rb {
            return None;
        }
        sum_rad_a[i] = ra + rb;
        d_signed_a[i] = d;
    }

    let mut sum_rad_b = [0.0f32; 3];
    let mut d_signed_b = [0.0f32; 3];
    for j in 0..3 {
        let axis = obb_b.axes[j];
        let (ra, rb) = project_radii(axis, obb_a, obb_b);
        let d = d_vec.dot(axis);
        if d.abs() >= ra + rb {
            return None;
        }
        sum_rad_b[j] = ra + rb;
        d_signed_b[j] = d;
    }

    let mut edge_candidates: [EdgeCandidate; 9] = [(None, 0, 0); 9];
    for i in 0..3 {
        for j in 0..3 {
            let slot = i * 3 + j;
            let raw = obb_a.axes[i].cross(obb_b.axes[j]);
            let len_sq = raw.length_squared();
            if len_sq < 1e-6 {
                edge_candidates[slot] = (None, i, j);
                continue;
            }
            let axis = raw / len_sq.sqrt();
            let (ra, rb) = project_radii(axis, obb_a, obb_b);
            let d = d_vec.dot(axis);
            let overlap = (ra + rb) - d.abs();
            if overlap <= 0.0 {
                return None;
            }
            let normal = if d > 1e-5 {
                axis
            } else if d < -1e-5 {
                -axis
            } else if fallback_dir_a_to_b.dot(axis) >= 0.0 {
                axis
            } else {
                -axis
            };
            edge_candidates[slot] = (Some((normal, overlap)), i, j);
        }
    }

    let mut min_overlap = f32::INFINITY;
    let mut best_axis: Option<ChosenAxis> = None;

    // 1. Evaluate exposed faces of Box A:
    for i in 0..3 {
        if is_axis_face_exposed(exposed_a, i, true) {
            let pen_pos = sum_rad_a[i] - d_signed_a[i];
            if pen_pos > 0.0 && pen_pos < min_overlap {
                min_overlap = pen_pos;
                best_axis = Some(ChosenAxis::FaceA {
                    axis_idx: i,
                    sign: 1.0,
                });
            }
        }
        if is_axis_face_exposed(exposed_a, i, false) {
            let pen_neg = sum_rad_a[i] + d_signed_a[i];
            if pen_neg > 0.0 && pen_neg < min_overlap {
                min_overlap = pen_neg;
                best_axis = Some(ChosenAxis::FaceA {
                    axis_idx: i,
                    sign: -1.0,
                });
            }
        }
    }

    // 2. Evaluate exposed faces of Box B:
    for j in 0..3 {
        // Outward +j face of B opposes A -> contact normal A->B is -axes[j]:
        if is_axis_face_exposed(exposed_b, j, true) {
            let pen_pos = sum_rad_b[j] + d_signed_b[j];
            if pen_pos > 0.0 && pen_pos < min_overlap {
                min_overlap = pen_pos;
                best_axis = Some(ChosenAxis::FaceB {
                    axis_idx: j,
                    sign: 1.0,
                });
            }
        }
        // Outward -j face of B opposes A -> contact normal A->B is +axes[j]:
        if is_axis_face_exposed(exposed_b, j, false) {
            let pen_neg = sum_rad_b[j] - d_signed_b[j];
            if pen_neg > 0.0 && pen_neg < min_overlap {
                min_overlap = pen_neg;
                best_axis = Some(ChosenAxis::FaceB {
                    axis_idx: j,
                    sign: -1.0,
                });
            }
        }
    }

    // 3. Evaluate true convex exterior Edge-Edge axes:
    for &(maybe_edge, i, j) in &edge_candidates {
        let Some((normal, overlap)) = maybe_edge else {
            continue;
        };

        let a1 = (i + 1) % 3;
        let a2 = (i + 2) % 3;
        let pos_a1 = obb_a.axes[a1].dot(normal) >= 0.0;
        let pos_a2 = obb_a.axes[a2].dot(normal) >= 0.0;
        if !is_axis_face_exposed(exposed_a, a1, pos_a1)
            || !is_axis_face_exposed(exposed_a, a2, pos_a2)
        {
            continue;
        }

        let b1 = (j + 1) % 3;
        let b2 = (j + 2) % 3;
        let pos_b1 = obb_b.axes[b1].dot(-normal) >= 0.0;
        let pos_b2 = obb_b.axes[b2].dot(-normal) >= 0.0;
        if !is_axis_face_exposed(exposed_b, b1, pos_b1)
            || !is_axis_face_exposed(exposed_b, b2, pos_b2)
        {
            continue;
        }

        // WHY: Require 5% shallower overlap than face axes so flat resting surfaces prefer 4-point face clipping.
        let is_better = match best_axis {
            Some(ChosenAxis::FaceA { .. } | ChosenAxis::FaceB { .. }) => {
                overlap < min_overlap * 0.95
            }
            Some(ChosenAxis::Edge { .. }) | None => overlap < min_overlap,
        };
        if is_better {
            min_overlap = overlap;
            best_axis = Some(ChosenAxis::Edge { i, j, normal });
        }
    }

    match best_axis? {
        ChosenAxis::FaceA { axis_idx, sign } => {
            let ref_normal = obb_a.axes[axis_idx] * sign;
            clip_face_contact(obb_a, obb_b, axis_idx, ref_normal, ref_normal, min_overlap)
        }
        ChosenAxis::FaceB { axis_idx, sign } => {
            let ref_normal = obb_b.axes[axis_idx] * sign;
            let contact_normal = -ref_normal;
            clip_face_contact(
                obb_b,
                obb_a,
                axis_idx,
                ref_normal,
                contact_normal,
                min_overlap,
            )
        }
        ChosenAxis::Edge { i, j, normal } => {
            let a1 = (i + 1) % 3;
            let a2 = (i + 2) % 3;
            let sign_a1 = if obb_a.axes[a1].dot(normal) >= 0.0 {
                1.0
            } else {
                -1.0
            };
            let sign_a2 = if obb_a.axes[a2].dot(normal) >= 0.0 {
                1.0
            } else {
                -1.0
            };
            let edge_center_a = obb_a.center
                + obb_a.axes[a1] * (sign_a1 * obb_a.extents[a1])
                + obb_a.axes[a2] * (sign_a2 * obb_a.extents[a2]);
            let p_a1 = edge_center_a - obb_a.axes[i] * obb_a.extents[i];
            let p_a2 = edge_center_a + obb_a.axes[i] * obb_a.extents[i];

            let b1 = (j + 1) % 3;
            let b2 = (j + 2) % 3;
            let sign_b1 = if obb_b.axes[b1].dot(normal) <= 0.0 {
                1.0
            } else {
                -1.0
            };
            let sign_b2 = if obb_b.axes[b2].dot(normal) <= 0.0 {
                1.0
            } else {
                -1.0
            };
            let edge_center_b = obb_b.center
                + obb_b.axes[b1] * (sign_b1 * obb_b.extents[b1])
                + obb_b.axes[b2] * (sign_b2 * obb_b.extents[b2]);
            let p_b1 = edge_center_b - obb_b.axes[j] * obb_b.extents[j];
            let p_b2 = edge_center_b + obb_b.axes[j] * obb_b.extents[j];

            let (closest_a, closest_b) = closest_points_segments(p_a1, p_a2, p_b1, p_b2);
            let contact_pt = (closest_a + closest_b) * 0.5;

            let mut points = [(Vec3::ZERO, 0.0); 4];
            points[0] = (contact_pt, min_overlap);
            Some(RawBoxContact {
                normal,
                points,
                point_count: 1,
            })
        }
    }
}

#[inline]
pub(crate) fn orthonormal_basis(normal: Vec3) -> (Vec3, Vec3) {
    // WHY: Y-oriented Duff et al. (JCGT 2017) basis is branchless and C-infinity smooth across the entire upper hemisphere (n.y > 0).
    let sign = 1.0f32.copysign(normal.y);
    let a = -1.0 / (sign + normal.y);
    let b = normal.x * normal.z * a;
    let tangent1 = Vec3::new(
        1.0 + sign * normal.x * normal.x * a,
        -sign * normal.x,
        sign * b,
    );
    let tangent2 = Vec3::new(b, -normal.z, sign + normal.z * normal.z * a);
    (tangent1, tangent2)
}

pub(crate) fn reduce_contact_samples(
    samples: &[(Vec3, f32)],
    u_axis: Vec3,
    v_axis: Vec3,
    max_pts: usize,
) -> Vec<(Vec3, f32)> {
    if samples.is_empty() || max_pts == 0 {
        return Vec::new();
    }

    // Deduplicate spatially coincident points (< 4 mm apart), keeping the deeper penetration:
    let mut unique: Vec<(Vec3, f32)> = Vec::with_capacity(samples.len().min(32));
    'outer: for &(pos, pen) in samples {
        for existing in &mut unique {
            if existing.0.distance_squared(pos) < DEDUP_DIST_SQ {
                if pen > existing.1 {
                    *existing = (pos, pen);
                }
                continue 'outer;
            }
        }
        unique.push((pos, pen));
    }

    if unique.len() <= max_pts {
        return unique;
    }

    if max_pts == 4 && unique.len() <= 12 {
        return select_best_quad_points(&unique, u_axis, v_axis);
    }

    // Extremal 4-direction hull + deepest penetration selection for large clusters:
    let mut selected_indices: Vec<usize> = Vec::with_capacity(max_pts);

    // 1. Always preserve the deepest contact point:
    let mut deepest_idx = 0usize;
    for (idx, &(_, pen)) in unique.iter().enumerate().skip(1) {
        if pen > unique[deepest_idx].1 {
            deepest_idx = idx;
        }
    }
    selected_indices.push(deepest_idx);

    // 2. Extremal points along u, v, u+v, u-v tangent directions:
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let dirs = [
        (1.0f32, 0.0f32),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (s, s),
        (-s, -s),
        (s, -s),
        (-s, s),
    ];
    for &(du, dv) in &dirs {
        if selected_indices.len() >= max_pts {
            break;
        }
        let mut best_proj = f32::NEG_INFINITY;
        let mut best_idx = 0usize;
        for (idx, &(pos, _)) in unique.iter().enumerate() {
            let proj = pos.dot(u_axis) * du + pos.dot(v_axis) * dv;
            if proj > best_proj {
                best_proj = proj;
                best_idx = idx;
            }
        }
        if !selected_indices.contains(&best_idx) {
            selected_indices.push(best_idx);
        }
    }

    // 3. Fill any remaining budget by descending penetration depth:
    if selected_indices.len() < max_pts {
        let mut order: Vec<usize> = (0..unique.len()).collect();
        order.sort_by(|&a, &b| unique[b].1.total_cmp(&unique[a].1));
        for idx in order {
            if selected_indices.len() >= max_pts {
                break;
            }
            if !selected_indices.contains(&idx) {
                selected_indices.push(idx);
            }
        }
    }

    selected_indices.into_iter().map(|i| unique[i]).collect()
}

fn gather_overlapping_subbox_pairs(
    body_a: &RigidVoxelBody,
    body_b: &RigidVoxelBody,
    r_ab: Mat3,
    abs_r_ab: Mat3,
    t_ab: Vec3,
) -> Vec<(usize, usize)> {
    let nodes_a = &body_a.collider.bvh_nodes;
    let nodes_b = &body_b.collider.bvh_nodes;
    let boxes_a = &body_a.collider.sub_boxes;
    let boxes_b = &body_b.collider.sub_boxes;

    let mut pairs = Vec::new();
    let mut stack = Vec::with_capacity(32);
    stack.push((0u32, 0u32));

    while let Some((idx_a, idx_b)) = stack.pop() {
        let na = &nodes_a[idx_a as usize];
        let nb = &nodes_b[idx_b as usize];

        let ca = (na.min_rel_com + na.max_rel_com) * 0.5;
        let ha = (na.max_rel_com - na.min_rel_com) * 0.5;
        let ca_in_b = r_ab * ca + t_ab;
        let ha_in_b = abs_r_ab * ha;
        let a_min_b = ca_in_b - ha_in_b;
        let a_max_b = ca_in_b + ha_in_b;

        if a_max_b.x < nb.min_rel_com.x
            || a_min_b.x > nb.max_rel_com.x
            || a_max_b.y < nb.min_rel_com.y
            || a_min_b.y > nb.max_rel_com.y
            || a_max_b.z < nb.min_rel_com.z
            || a_min_b.z > nb.max_rel_com.z
        {
            continue;
        }

        let leaf_a = na.count > 0;
        let leaf_b = nb.count > 0;

        match (leaf_a, leaf_b) {
            (true, true) => {
                let start_a = na.left_or_start as usize;
                let end_a = start_a + (na.count as usize);
                let start_b = nb.left_or_start as usize;
                let end_b = start_b + (nb.count as usize);

                for (off_a, sa) in boxes_a[start_a..end_a].iter().enumerate() {
                    let sca_in_b = r_ab * sa.center_rel_com + t_ab;
                    let sha_in_b = abs_r_ab * sa.half_extents;
                    let smin_b = sca_in_b - sha_in_b;
                    let smax_b = sca_in_b + sha_in_b;

                    for (off_b, sb) in boxes_b[start_b..end_b].iter().enumerate() {
                        let b_min = sb.center_rel_com - sb.half_extents;
                        let b_max = sb.center_rel_com + sb.half_extents;
                        if smax_b.x >= b_min.x
                            && smin_b.x <= b_max.x
                            && smax_b.y >= b_min.y
                            && smin_b.y <= b_max.y
                            && smax_b.z >= b_min.z
                            && smin_b.z <= b_max.z
                        {
                            pairs.push((start_a + off_a, start_b + off_b));
                        }
                    }
                }
            }
            (false, true) => {
                stack.push((na.right_child, idx_b));
                stack.push((na.left_or_start, idx_b));
            }
            (true, false) => {
                stack.push((idx_a, nb.right_child));
                stack.push((idx_a, nb.left_or_start));
            }
            (false, false) => {
                let ext_a = na.max_rel_com - na.min_rel_com;
                let ext_b = nb.max_rel_com - nb.min_rel_com;
                if ext_a.length_squared() >= ext_b.length_squared() {
                    stack.push((na.right_child, idx_b));
                    stack.push((na.left_or_start, idx_b));
                } else {
                    stack.push((idx_a, nb.right_child));
                    stack.push((idx_a, nb.left_or_start));
                }
            }
        }
    }

    pairs
}

fn insert_raw_contact(clusters: &mut Vec<NormalCluster>, raw: &RawBoxContact) {
    let mut best_cluster = None;
    let mut best_dot = NORMAL_CLUSTER_COS_THRESHOLD;

    for (idx, c) in clusters.iter().enumerate() {
        let d = c.unit_normal.dot(raw.normal);
        if d > best_dot {
            best_dot = d;
            best_cluster = Some(idx);
        }
    }

    if let Some(idx) = best_cluster {
        let c = &mut clusters[idx];
        let mut weight = 0.0f32;
        for &(pos, pen) in &raw.points[0..raw.point_count] {
            c.max_pen = c.max_pen.max(pen);
            weight += pen.max(1e-3);
            c.samples.push((pos, pen));
        }
        c.normal_sum += raw.normal * weight;
        if c.normal_sum.length_squared() > 1e-8 {
            c.unit_normal = c.normal_sum.normalize();
        }
    } else {
        let mut samples = Vec::with_capacity(8);
        let mut max_pen = 0.0f32;
        let mut weight = 0.0f32;
        for &(pos, pen) in &raw.points[0..raw.point_count] {
            max_pen = max_pen.max(pen);
            weight += pen.max(1e-3);
            samples.push((pos, pen));
        }
        clusters.push(NormalCluster {
            normal_sum: raw.normal * weight.max(1e-3),
            unit_normal: raw.normal,
            max_pen,
            samples,
        });
    }
}

#[inline]
fn project_radii(axis: Vec3, obb_a: &Obb, obb_b: &Obb) -> (f32, f32) {
    let ra = obb_a.extents.x * obb_a.axes[0].dot(axis).abs()
        + obb_a.extents.y * obb_a.axes[1].dot(axis).abs()
        + obb_a.extents.z * obb_a.axes[2].dot(axis).abs();
    let rb = obb_b.extents.x * obb_b.axes[0].dot(axis).abs()
        + obb_b.extents.y * obb_b.axes[1].dot(axis).abs()
        + obb_b.extents.z * obb_b.axes[2].dot(axis).abs();
    (ra, rb)
}

fn clip_face_contact(
    ref_box: &Obb,
    inc_box: &Obb,
    ref_axis: usize,
    ref_normal: Vec3,
    contact_normal: Vec3,
    sat_overlap: f32,
) -> Option<RawBoxContact> {
    let mut candidate_faces: Vec<(usize, Vec3, f32)> = Vec::with_capacity(6);
    for k in 0..3 {
        let dot_pos = inc_box.axes[k].dot(ref_normal);
        if dot_pos < -1e-4 {
            candidate_faces.push((k, inc_box.axes[k], dot_pos));
        }
        let dot_neg = -dot_pos;
        if dot_neg < -1e-4 {
            candidate_faces.push((k, -inc_box.axes[k], dot_neg));
        }
    }
    candidate_faces.sort_by(|a, b| a.2.total_cmp(&b.2));

    if candidate_faces.is_empty() {
        let mut min_dot = f32::INFINITY;
        let mut best_axis = 0;
        let mut best_normal = inc_box.axes[0];
        for k in 0..3 {
            let dot_pos = inc_box.axes[k].dot(ref_normal);
            if dot_pos < min_dot {
                min_dot = dot_pos;
                best_axis = k;
                best_normal = inc_box.axes[k];
            }
            let dot_neg = -dot_pos;
            if dot_neg < min_dot {
                min_dot = dot_neg;
                best_axis = k;
                best_normal = -inc_box.axes[k];
            }
        }
        candidate_faces.push((best_axis, best_normal, min_dot));
    }

    let side_axis_1 = (ref_axis + 1) % 3;
    let side_axis_2 = (ref_axis + 2) % 3;

    let side_planes = [
        (
            ref_box.center + ref_box.axes[side_axis_1] * ref_box.extents[side_axis_1],
            -ref_box.axes[side_axis_1],
        ),
        (
            ref_box.center - ref_box.axes[side_axis_1] * ref_box.extents[side_axis_1],
            ref_box.axes[side_axis_1],
        ),
        (
            ref_box.center + ref_box.axes[side_axis_2] * ref_box.extents[side_axis_2],
            -ref_box.axes[side_axis_2],
        ),
        (
            ref_box.center - ref_box.axes[side_axis_2] * ref_box.extents[side_axis_2],
            ref_box.axes[side_axis_2],
        ),
    ];

    let ref_plane_point = ref_box.center + ref_normal * ref_box.extents[ref_axis];

    for (inc_axis, inc_normal, _) in candidate_faces {
        let inc_face_center = inc_box.center + inc_normal * inc_box.extents[inc_axis];
        let inc_t1_idx = (inc_axis + 1) % 3;
        let inc_t2_idx = (inc_axis + 2) % 3;
        let inc_t1 = inc_box.axes[inc_t1_idx] * inc_box.extents[inc_t1_idx];
        let inc_t2 = inc_box.axes[inc_t2_idx] * inc_box.extents[inc_t2_idx];

        let mut poly = vec![
            inc_face_center + inc_t1 + inc_t2,
            inc_face_center - inc_t1 + inc_t2,
            inc_face_center - inc_t1 - inc_t2,
            inc_face_center + inc_t1 - inc_t2,
        ];

        let mut clipped_empty = false;
        for &(plane_pt, inward_norm) in &side_planes {
            poly = clip_polygon_against_plane(&poly, plane_pt, inward_norm);
            if poly.is_empty() {
                clipped_empty = true;
                break;
            }
        }
        if clipped_empty {
            continue;
        }

        let mut valid_points = Vec::with_capacity(poly.len());
        for pt in poly {
            let depth = (ref_plane_point - pt).dot(ref_normal);
            // WHY: Retain points within 3 mm contact slop to preserve a stable 4-point quad support polygon.
            if depth >= -0.003 {
                let clamped_depth = depth.max(0.0).min(sat_overlap + 0.002);
                let pos = pt + ref_normal * (clamped_depth * 0.5);
                valid_points.push((pos, clamped_depth));
            }
        }

        if valid_points.is_empty() {
            continue;
        }

        let final_points = if valid_points.len() > 4 {
            select_best_quad_points(
                &valid_points,
                ref_box.axes[side_axis_1],
                ref_box.axes[side_axis_2],
            )
        } else {
            valid_points
        };

        let mut points = [(Vec3::ZERO, 0.0); 4];
        for (k, &sample) in final_points.iter().enumerate() {
            points[k] = sample;
        }

        return Some(RawBoxContact {
            normal: contact_normal,
            points,
            point_count: final_points.len(),
        });
    }

    None
}

fn clip_polygon_against_plane(poly: &[Vec3], plane_point: Vec3, inward_normal: Vec3) -> Vec<Vec3> {
    if poly.is_empty() {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(poly.len() + 2);
    let mut prev = poly[poly.len() - 1];
    let mut prev_dist = (prev - plane_point).dot(inward_normal);

    for &curr in poly {
        let curr_dist = (curr - plane_point).dot(inward_normal);

        if curr_dist >= 0.0 {
            if prev_dist < 0.0 {
                let diff = prev_dist - curr_dist;
                let t = if diff.abs() > 1e-7 {
                    (prev_dist / diff).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                output.push(prev + (curr - prev) * t);
            }
            output.push(curr);
        } else if prev_dist >= 0.0 {
            let diff = prev_dist - curr_dist;
            let t = if diff.abs() > 1e-7 {
                (prev_dist / diff).clamp(0.0, 1.0)
            } else {
                0.0
            };
            output.push(prev + (curr - prev) * t);
        }

        prev = curr;
        prev_dist = curr_dist;
    }

    output
}

fn select_best_quad_points(points: &[(Vec3, f32)], u_axis: Vec3, v_axis: Vec3) -> Vec<(Vec3, f32)> {
    let n = points.len();
    if n <= 4 {
        return points.to_vec();
    }

    let mut pts_2d = Vec::with_capacity(n);
    for &(pos, _) in points {
        pts_2d.push((pos.dot(u_axis), pos.dot(v_axis)));
    }

    let mut best_area = -1.0f32;
    let mut best_indices = [0, 1, 2, 3];

    for i0 in 0..n {
        for i1 in (i0 + 1)..n {
            for i2 in (i1 + 1)..n {
                for i3 in (i2 + 1)..n {
                    let area = quad_area_2d(pts_2d[i0], pts_2d[i1], pts_2d[i2], pts_2d[i3]);
                    if area > best_area {
                        best_area = area;
                        best_indices = [i0, i1, i2, i3];
                    }
                }
            }
        }
    }

    let [i0, i1, i2, i3] = best_indices;
    vec![points[i0], points[i1], points[i2], points[i3]]
}

#[inline]
fn quad_area_2d(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) -> f32 {
    let cross = (p0.0 * p1.1 - p1.0 * p0.1)
        + (p1.0 * p2.1 - p2.0 * p1.1)
        + (p2.0 * p3.1 - p3.0 * p2.1)
        + (p3.0 * p0.1 - p0.0 * p3.1);
    0.5 * cross.abs()
}

fn closest_points_segments(p_a1: Vec3, p_a2: Vec3, p_b1: Vec3, p_b2: Vec3) -> (Vec3, Vec3) {
    let d1 = p_a2 - p_a1;
    let d2 = p_b2 - p_b1;
    let r = p_a1 - p_b1;

    let a = d1.length_squared();
    let b = d1.dot(d2);
    let c = d2.length_squared();
    let d = d1.dot(r);
    let e = d2.dot(r);

    let denom = a * c - b * b;
    let mut s = if denom.abs() > 1e-6 {
        ((b * e - c * d) / denom).clamp(0.0, 1.0)
    } else {
        0.5
    };

    let t = if c > 1e-6 {
        ((b * s + e) / c).clamp(0.0, 1.0)
    } else {
        0.5
    };

    if a > 1e-6 {
        s = ((b * t - d) / a).clamp(0.0, 1.0);
    }

    (p_a1 + d1 * s, p_b1 + d2 * t)
}
