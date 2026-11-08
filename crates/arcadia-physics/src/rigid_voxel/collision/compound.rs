//! Exact 3D greedy-meshed compound cuboid decomposition and flat local AABB BVH.
//! Owns CPU true-shape collision geometry so dynamic bodies collide on exact voxel surfaces without air padding.

use super::broadphase::{obb_intersects_aabb_sat, Obb};
use arcadia_core::{MaterialId, MAT_AIR, VOXEL_SIZE};
use glam::{IVec3, Mat3, Quat, Vec3};

/// Bit flag indicating the `-X` face of a sub-box borders at least one air voxel.
pub const FACE_NEG_X: u8 = 1 << 0;
/// Bit flag indicating the `+X` face of a sub-box borders at least one air voxel.
pub const FACE_POS_X: u8 = 1 << 1;
/// Bit flag indicating the `-Y` face of a sub-box borders at least one air voxel.
pub const FACE_NEG_Y: u8 = 1 << 2;
/// Bit flag indicating the `+Y` face of a sub-box borders at least one air voxel.
pub const FACE_POS_Y: u8 = 1 << 3;
/// Bit flag indicating the `-Z` face of a sub-box borders at least one air voxel.
pub const FACE_NEG_Z: u8 = 1 << 4;
/// Bit flag indicating the `+Z` face of a sub-box borders at least one air voxel.
pub const FACE_POS_Z: u8 = 1 << 5;
/// Bitmask with all 6 cardinal faces marked exposed to air.
pub const ALL_FACES_EXPOSED: u8 = 0x3F;

const BVH_LEAF_MAX_BOXES: usize = 4;

/// Exact disjoint solid cuboid in body-local space relative to the Center of Mass.
#[derive(Clone, Copy, Debug)]
pub struct VoxelSubBox {
    /// Center offset from `com_local_m` in body-local meters.
    pub center_rel_com: Vec3,
    /// Half-extents of the cuboid in meters (`0.5 * (max_v - min_v) * VOXEL_SIZE`).
    pub half_extents: Vec3,
    /// Inclusive lower local voxel grid bound in `[0 .. dims_v)`.
    pub min_v: [u16; 3],
    /// Exclusive upper local voxel grid bound in `[1 ..= dims_v]`.
    pub max_v: [u16; 3],
    /// Bitmask (`FACE_NEG_X .. FACE_POS_Z`) of outer faces bordering air rather than internal solid voxels.
    pub exposed_faces: u8,
}

impl VoxelSubBox {
    /// Volume of this cuboid measured in integer micro-voxels.
    #[inline]
    pub fn volume_voxels(&self) -> u32 {
        let dx = (self.max_v[0] - self.min_v[0]) as u32;
        let dy = (self.max_v[1] - self.min_v[1]) as u32;
        let dz = (self.max_v[2] - self.min_v[2]) as u32;
        dx * dy * dz
    }
}

/// Cache-aligned flat node in the CoM-relative local AABB Bounding Volume Hierarchy.
#[derive(Clone, Copy, Debug)]
pub struct SubBoxBvhNode {
    /// Lower local AABB corner relative to `com_local_m` (meters).
    pub min_rel_com: Vec3,
    /// Upper local AABB corner relative to `com_local_m` (meters).
    pub max_rel_com: Vec3,
    /// Left child index when `count == 0`, or starting index into `sub_boxes` when `count > 0`.
    pub left_or_start: u32,
    /// Right child index when `count == 0` (`0` for leaves).
    pub right_child: u32,
    /// Number of contiguous `VoxelSubBox` leaves (`0` indicates an internal node).
    pub count: u32,
}

/// Exact disjoint cuboid partition and local BVH representing a `RigidVoxelBody`'s full solid shape.
#[derive(Clone, Debug, Default)]
pub struct CompoundCollider {
    /// Disjoint greedy-merged cuboids covering 100% of solid voxels and 0% of air voxels.
    pub sub_boxes: Vec<VoxelSubBox>,
    /// Flat top-down median-split BVH over `sub_boxes` (`bvh_nodes[0]` is root when non-empty).
    pub bvh_nodes: Vec<SubBoxBvhNode>,
    /// Total solid voxel count summed across all `sub_boxes`.
    pub total_voxels: u32,
}

impl CompoundCollider {
    /// Partitions `local_voxels` into exact disjoint 3D greedy cuboids with exposed-face masks and builds the local BVH.
    pub fn build(
        local_voxels: &[(u32, u32, u32, MaterialId)],
        dims_v: IVec3,
        com_local_m: Vec3,
    ) -> Self {
        if local_voxels.is_empty() || dims_v.x <= 0 || dims_v.y <= 0 || dims_v.z <= 0 {
            return Self::default();
        }

        let nx = (dims_v.x as usize).min(256);
        let ny = (dims_v.y as usize).min(256);
        let nz = (dims_v.z as usize).min(256);
        let row_words = nx.div_ceil(64);
        let total_words = row_words * ny * nz;

        let mut solid_bits = vec![0u64; total_words];
        for &(lx, ly, lz, mat) in local_voxels {
            if mat == MAT_AIR {
                continue;
            }
            let x = lx as usize;
            let y = ly as usize;
            let z = lz as usize;
            if x < nx && y < ny && z < nz {
                let row_base = (y + z * ny) * row_words;
                solid_bits[row_base + (x >> 6)] |= 1u64 << (x & 63);
            }
        }

        let mut rem_bits = solid_bits.clone();
        let mut sub_boxes = Vec::with_capacity(64);
        let mut total_voxels = 0u32;
        let s = VOXEL_SIZE;

        for z0 in 0..nz {
            for y0 in 0..ny {
                let row0_base = (y0 + z0 * ny) * row_words;
                while let Some(x0) =
                    find_first_set_x(&rem_bits[row0_base..row0_base + row_words], nx)
                {
                    let x1 = find_run_end_x(&rem_bits[row0_base..row0_base + row_words], nx, x0);
                    let span_mask = build_x_span_mask(x0, x1);

                    // Expand along +Y while full [x0..x1) span remains solid:
                    let mut y1 = y0 + 1;
                    while y1 < ny {
                        let rb = (y1 + z0 * ny) * row_words;
                        if !row_contains_span(&rem_bits[rb..rb + row_words], &span_mask, row_words)
                        {
                            break;
                        }
                        y1 += 1;
                    }

                    // Expand along +Z while full [x0..x1) x [y0..y1) slab remains solid:
                    let mut z1 = z0 + 1;
                    'z_expand: while z1 < nz {
                        for y in y0..y1 {
                            let rb = (y + z1 * ny) * row_words;
                            if !row_contains_span(
                                &rem_bits[rb..rb + row_words],
                                &span_mask,
                                row_words,
                            ) {
                                break 'z_expand;
                            }
                        }
                        z1 += 1;
                    }

                    // Clear consumed cuboid bits in working copy:
                    for z in z0..z1 {
                        for y in y0..y1 {
                            let rb = (y + z * ny) * row_words;
                            for (word, &mask) in
                                rem_bits[rb..rb + row_words].iter_mut().zip(&span_mask)
                            {
                                *word &= !mask;
                            }
                        }
                    }

                    let exposed_faces = compute_exposed_faces(
                        &solid_bits,
                        nx,
                        ny,
                        nz,
                        row_words,
                        (x0, x1),
                        (y0, y1),
                        (z0, z1),
                        &span_mask,
                    );

                    let min_m = Vec3::new(x0 as f32, y0 as f32, z0 as f32) * s;
                    let max_m = Vec3::new(x1 as f32, y1 as f32, z1 as f32) * s;
                    let center_local_m = (min_m + max_m) * 0.5;
                    let half_extents = (max_m - min_m) * 0.5;

                    let sub = VoxelSubBox {
                        center_rel_com: center_local_m - com_local_m,
                        half_extents,
                        min_v: [x0 as u16, y0 as u16, z0 as u16],
                        max_v: [x1 as u16, y1 as u16, z1 as u16],
                        exposed_faces,
                    };
                    total_voxels += sub.volume_voxels();
                    sub_boxes.push(sub);
                }
            }
        }

        let mut bvh_nodes = Vec::with_capacity(sub_boxes.len() * 2);
        if !sub_boxes.is_empty() {
            let len = sub_boxes.len();
            build_bvh_recursive(&mut sub_boxes, 0, len, &mut bvh_nodes);
        }

        Self {
            sub_boxes,
            bvh_nodes,
            total_voxels,
        }
    }

    /// Tests whether any `VoxelSubBox` in this compound collider intersects the world-space AABB `[aabb_min .. aabb_max]`.
    pub fn intersects_world_aabb(
        &self,
        body_pos: Vec3,
        body_rot: Quat,
        aabb_min: Vec3,
        aabb_max: Vec3,
    ) -> bool {
        if self.bvh_nodes.is_empty() {
            return false;
        }

        let aabb_center = (aabb_min + aabb_max) * 0.5;
        let aabb_extents = (aabb_max - aabb_min) * 0.5;

        let r_mat = Mat3::from_quat(body_rot);
        let inv_r = r_mat.transpose();
        let abs_inv_r = Mat3::from_cols(inv_r.x_axis.abs(), inv_r.y_axis.abs(), inv_r.z_axis.abs());

        let local_center = inv_r * (aabb_center - body_pos);
        let local_extents = abs_inv_r * aabb_extents;
        let q_min = local_center - local_extents;
        let q_max = local_center + local_extents;

        let mut stack = [0u32; 32];
        let mut sp = 1usize;

        while sp > 0 {
            sp -= 1;
            let node = &self.bvh_nodes[stack[sp] as usize];
            if node.max_rel_com.x < q_min.x
                || node.min_rel_com.x > q_max.x
                || node.max_rel_com.y < q_min.y
                || node.min_rel_com.y > q_max.y
                || node.max_rel_com.z < q_min.z
                || node.min_rel_com.z > q_max.z
            {
                continue;
            }

            if node.count > 0 {
                let start = node.left_or_start as usize;
                let end = start + (node.count as usize);
                for sub in &self.sub_boxes[start..end] {
                    let sub_min = sub.center_rel_com - sub.half_extents;
                    let sub_max = sub.center_rel_com + sub.half_extents;
                    if sub_max.x < q_min.x
                        || sub_min.x > q_max.x
                        || sub_max.y < q_min.y
                        || sub_min.y > q_max.y
                        || sub_max.z < q_min.z
                        || sub_min.z > q_max.z
                    {
                        continue;
                    }
                    let obb = Obb::from_sub_box(sub, body_pos, r_mat);
                    if obb_intersects_aabb_sat(&obb, aabb_center, aabb_extents) {
                        return true;
                    }
                }
            } else if sp + 2 <= stack.len() {
                stack[sp] = node.right_child;
                stack[sp + 1] = node.left_or_start;
                sp += 2;
            }
        }

        false
    }
}

#[inline]
pub(crate) const fn is_axis_face_exposed(exposed_faces: u8, axis: usize, positive: bool) -> bool {
    let bit = (axis << 1) | (positive as usize);
    (exposed_faces & (1u8 << bit)) != 0
}

#[inline]
fn find_first_set_x(row: &[u64], nx: usize) -> Option<usize> {
    for (w, &word) in row.iter().enumerate() {
        if word != 0 {
            let x = (w << 6) + (word.trailing_zeros() as usize);
            return (x < nx).then_some(x);
        }
    }
    None
}

#[inline]
fn find_run_end_x(row: &[u64], nx: usize, x0: usize) -> usize {
    let mut x = x0;
    while x < nx {
        let w = x >> 6;
        let bit = x & 63;
        let ones = (row[w] >> bit).trailing_ones() as usize;
        x += ones;
        if bit + ones < 64 {
            break;
        }
    }
    x.min(nx)
}

#[inline]
fn build_x_span_mask(x0: usize, x1: usize) -> [u64; 4] {
    let mut mask = [0u64; 4];
    for (w, slot) in mask
        .iter_mut()
        .enumerate()
        .take(((x1 - 1) >> 6) + 1)
        .skip(x0 >> 6)
    {
        let start = x0.saturating_sub(w << 6);
        let end = (x1 - (w << 6)).min(64);
        let width = end - start;
        *slot = if width == 64 {
            u64::MAX
        } else {
            ((1u64 << width) - 1) << start
        };
    }
    mask
}

#[inline]
fn row_contains_span(row: &[u64], span_mask: &[u64; 4], row_words: usize) -> bool {
    for w in 0..row_words {
        let m = span_mask[w];
        if (row[w] & m) != m {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn compute_exposed_faces(
    solid_bits: &[u64],
    nx: usize,
    ny: usize,
    nz: usize,
    row_words: usize,
    (x0, x1): (usize, usize),
    (y0, y1): (usize, usize),
    (z0, z1): (usize, usize),
    span_mask: &[u64; 4],
) -> u8 {
    let mut mask = 0u8;

    // -X face:
    if x0 == 0 {
        mask |= FACE_NEG_X;
    } else {
        let px = x0 - 1;
        let pw = px >> 6;
        let pbit = 1u64 << (px & 63);
        'neg_x: for z in z0..z1 {
            for y in y0..y1 {
                if (solid_bits[(y + z * ny) * row_words + pw] & pbit) == 0 {
                    mask |= FACE_NEG_X;
                    break 'neg_x;
                }
            }
        }
    }

    // +X face:
    if x1 >= nx {
        mask |= FACE_POS_X;
    } else {
        let pw = x1 >> 6;
        let pbit = 1u64 << (x1 & 63);
        'pos_x: for z in z0..z1 {
            for y in y0..y1 {
                if (solid_bits[(y + z * ny) * row_words + pw] & pbit) == 0 {
                    mask |= FACE_POS_X;
                    break 'pos_x;
                }
            }
        }
    }

    // -Y face:
    if y0 == 0 {
        mask |= FACE_NEG_Y;
    } else {
        let py = y0 - 1;
        for z in z0..z1 {
            let rb = (py + z * ny) * row_words;
            if !row_contains_span(&solid_bits[rb..rb + row_words], span_mask, row_words) {
                mask |= FACE_NEG_Y;
                break;
            }
        }
    }

    // +Y face:
    if y1 >= ny {
        mask |= FACE_POS_Y;
    } else {
        for z in z0..z1 {
            let rb = (y1 + z * ny) * row_words;
            if !row_contains_span(&solid_bits[rb..rb + row_words], span_mask, row_words) {
                mask |= FACE_POS_Y;
                break;
            }
        }
    }

    // -Z face:
    if z0 == 0 {
        mask |= FACE_NEG_Z;
    } else {
        let pz = z0 - 1;
        for y in y0..y1 {
            let rb = (y + pz * ny) * row_words;
            if !row_contains_span(&solid_bits[rb..rb + row_words], span_mask, row_words) {
                mask |= FACE_NEG_Z;
                break;
            }
        }
    }

    // +Z face:
    if z1 >= nz {
        mask |= FACE_POS_Z;
    } else {
        for y in y0..y1 {
            let rb = (y + z1 * ny) * row_words;
            if !row_contains_span(&solid_bits[rb..rb + row_words], span_mask, row_words) {
                mask |= FACE_POS_Z;
                break;
            }
        }
    }

    mask
}

fn build_bvh_recursive(
    sub_boxes: &mut [VoxelSubBox],
    start: usize,
    end: usize,
    nodes: &mut Vec<SubBoxBvhNode>,
) -> u32 {
    let mut min_rel = Vec3::splat(f32::INFINITY);
    let mut max_rel = Vec3::splat(f32::NEG_INFINITY);
    let mut c_min = Vec3::splat(f32::INFINITY);
    let mut c_max = Vec3::splat(f32::NEG_INFINITY);

    for sub in &sub_boxes[start..end] {
        min_rel = min_rel.min(sub.center_rel_com - sub.half_extents);
        max_rel = max_rel.max(sub.center_rel_com + sub.half_extents);
        c_min = c_min.min(sub.center_rel_com);
        c_max = c_max.max(sub.center_rel_com);
    }

    let node_idx = nodes.len() as u32;
    let count = end - start;
    if count <= BVH_LEAF_MAX_BOXES {
        nodes.push(SubBoxBvhNode {
            min_rel_com: min_rel,
            max_rel_com: max_rel,
            left_or_start: start as u32,
            right_child: 0,
            count: count as u32,
        });
        return node_idx;
    }

    nodes.push(SubBoxBvhNode {
        min_rel_com: min_rel,
        max_rel_com: max_rel,
        left_or_start: 0,
        right_child: 0,
        count: 0,
    });

    let span = c_max - c_min;
    let split_axis = if span.x >= span.y && span.x >= span.z {
        0
    } else if span.y >= span.z {
        1
    } else {
        2
    };

    let mid_rel = count / 2;
    sub_boxes[start..end].select_nth_unstable_by(mid_rel, |a, b| {
        a.center_rel_com[split_axis].total_cmp(&b.center_rel_com[split_axis])
    });
    let mid = start + mid_rel;

    let left_idx = build_bvh_recursive(sub_boxes, start, mid, nodes);
    let right_idx = build_bvh_recursive(sub_boxes, mid, end, nodes);

    nodes[node_idx as usize].left_or_start = left_idx;
    nodes[node_idx as usize].right_child = right_idx;
    node_idx
}
