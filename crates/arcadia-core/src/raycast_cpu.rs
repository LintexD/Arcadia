//! Reference CPU 64-Tree DDA Raycaster (`raycast_chunk_tree`).
//! Owns CPU reference 64-tree DDA raycasting for exact voxel selection and collision queries.

use crate::chunk_tree::ChunkTree64;
use crate::coord::{child_bit_index, CHUNK_DIM};
use crate::material::{MaterialId, MAT_AIR};
use crate::node64::Node64;
use glam::{IVec3, Vec3};

/// Information returned when a ray hits a solid voxel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// Distance along the ray in world meters (or chunk-local units depending on caller).
    pub distance: f32,
    /// Hit position in the coordinate space of the ray.
    pub hit_pos: Vec3,
    /// Outward-pointing surface normal (`±X`, `±Y`, or `±Z`).
    pub normal: Vec3,
    /// Voxel coordinate hit.
    pub voxel_coord: IVec3,
    /// Material of the hit voxel.
    pub material: MaterialId,
}

/// Raycasts through a single `ChunkTree64` in chunk-local voxel space `[0.0 .. 256.0]³`.
///
/// `ray_origin` and `max_dist` are expressed in **voxel units** (`1.0 = 1/16 m`).
pub fn raycast_chunk_tree(
    tree: &ChunkTree64,
    ray_origin: Vec3,
    ray_dir: Vec3,
    max_dist: f32,
) -> Option<RayHit> {
    if tree.is_empty() {
        return None;
    }

    let safe_dir = Vec3::new(
        if ray_dir.x.abs() > 1e-7 {
            ray_dir.x
        } else {
            1e-7f32.copysign(ray_dir.x)
        },
        if ray_dir.y.abs() > 1e-7 {
            ray_dir.y
        } else {
            1e-7f32.copysign(ray_dir.y)
        },
        if ray_dir.z.abs() > 1e-7 {
            ray_dir.z
        } else {
            1e-7f32.copysign(ray_dir.z)
        },
    );
    let inv_dir = 1.0 / safe_dir;

    let chunk_max = Vec3::splat(CHUNK_DIM as f32);
    let (t_enter_raw, t_exit_raw) = intersect_aabb(ray_origin, inv_dir, Vec3::ZERO, chunk_max)?;
    let t_enter = t_enter_raw.max(0.0);
    let t_limit = t_exit_raw.min(max_dist);
    if t_exit_raw < t_enter || t_enter >= max_dist {
        return None;
    }

    let pos_dir = [inv_dir.x >= 0.0, inv_dir.y >= 0.0, inv_dir.z >= 0.0];
    let neg_orig_inv = -ray_origin * inv_dir;

    let mut t = t_enter;
    let start_p = (ray_origin + safe_dir * (t + 1e-4)).clamp(Vec3::ZERO, chunk_max - Vec3::splat(1e-3));
    let mut vi = [start_p.x as u32, start_p.y as u32, start_p.z as u32];

    let mut prev_vi = [0xFFFFFFFFu32; 3];
    let mut cached_shift = 6u32;
    let mut cached_l1_node = Node64::empty();
    let mut cached_l2_node = Node64::empty();
    let mut cached_l3_node = Node64::empty();

    for _step in 0..512 {
        if t >= t_limit {
            break;
        }
        if vi[0] >= CHUNK_DIM || vi[1] >= CHUNK_DIM || vi[2] >= CHUNK_DIM {
            break;
        }

        let diff = (prev_vi[0] ^ vi[0]) | (prev_vi[1] ^ vi[1]) | (prev_vi[2] ^ vi[2]);
        prev_vi = vi;

        let mut node: Node64;
        let mut shift: u32;

        if diff < 4 && cached_shift == 0 {
            node = cached_l3_node;
            shift = 0;
        } else if diff < 16 && cached_shift <= 2 {
            node = cached_l2_node;
            shift = 2;
        } else if diff < 64 && cached_shift <= 4 {
            node = cached_l1_node;
            shift = 4;
        } else {
            node = tree.nodes[0];
            shift = 6;
        }
        cached_shift = shift;

        let l = [vi[0] & 255, vi[1] & 255, vi[2] & 255];
        loop {
            let cx = (l[0] >> shift) & 3;
            let cy = (l[1] >> shift) & 3;
            let cz = (l[2] >> shift) & 3;
            let b = child_bit_index(cx, cy, cz);

            if !node.has_child(b) {
                let cell_span = 1u32 << shift;
                let cell_min_u = [
                    vi[0] & !(cell_span - 1),
                    vi[1] & !(cell_span - 1),
                    vi[2] & !(cell_span - 1),
                ];
                let cell_max_u = [
                    cell_min_u[0] + cell_span,
                    cell_min_u[1] + cell_span,
                    cell_min_u[2] + cell_span,
                ];

                let st = step_dda_box(
                    ray_origin,
                    safe_dir,
                    inv_dir,
                    neg_orig_inv,
                    pos_dir,
                    cell_min_u,
                    cell_max_u,
                    t,
                );
                t = st.t;
                vi = st.vi;
                break;
            }

            let is_leaf = node.is_leaf();
            if is_leaf {
                let cell_span = 1u32 << shift;
                let cell_min_u = [
                    vi[0] & !(cell_span - 1),
                    vi[1] & !(cell_span - 1),
                    vi[2] & !(cell_span - 1),
                ];
                let cell_max_u = [
                    cell_min_u[0] + cell_span,
                    cell_min_u[1] + cell_span,
                    cell_min_u[2] + cell_span,
                ];

                let mat = if node.is_uniform() {
                    node.primary_material()
                } else {
                    tree.bricks[node.base_ptr() as usize].get_material(b)
                };

                if mat != MAT_AIR {
                    let enter_coord = [
                        if pos_dir[0] { cell_min_u[0] } else { cell_max_u[0] },
                        if pos_dir[1] { cell_min_u[1] } else { cell_max_u[1] },
                        if pos_dir[2] { cell_min_u[2] } else { cell_max_u[2] },
                    ];
                    let tmin_v = Vec3::new(
                        enter_coord[0] as f32 * inv_dir.x + neg_orig_inv.x,
                        enter_coord[1] as f32 * inv_dir.y + neg_orig_inv.y,
                        enter_coord[2] as f32 * inv_dir.z + neg_orig_inv.z,
                    );
                    let cell_enter = tmin_v.x.max(tmin_v.y).max(tmin_v.z);
                    let exact_t = cell_enter.max(0.0);
                    let hit_pos = ray_origin + safe_dir * exact_t;
                    let normal = resolve_entry_normal(tmin_v, ray_dir);

                    return Some(RayHit {
                        distance: exact_t,
                        hit_pos,
                        normal,
                        voxel_coord: IVec3::new(vi[0] as i32, vi[1] as i32, vi[2] as i32),
                        material: mat,
                    });
                }

                let st = step_dda_box(
                    ray_origin,
                    safe_dir,
                    inv_dir,
                    neg_orig_inv,
                    pos_dir,
                    cell_min_u,
                    cell_max_u,
                    t,
                );
                t = st.t;
                vi = st.vi;
                break;
            }

            let next_idx = node.child_slot_offset(b) as usize;
            node = tree.nodes[next_idx];
            shift -= 2;

            cached_shift = shift;
            if shift == 4 {
                cached_l1_node = node;
            } else if shift == 2 {
                cached_l2_node = node;
            } else if shift == 0 {
                cached_l3_node = node;
            }
        }
    }

    None
}

/// Ray-AABB intersection returning entry and exit distances `(t_enter, t_exit)`.
#[inline]
pub fn intersect_aabb(origin: Vec3, inv_dir: Vec3, bmin: Vec3, bmax: Vec3) -> Option<(f32, f32)> {
    let t0 = (bmin - origin) * inv_dir;
    let t1 = (bmax - origin) * inv_dir;
    let tmin_v = t0.min(t1);
    let tmax_v = t0.max(t1);
    let tmin = tmin_v.x.max(tmin_v.y).max(tmin_v.z);
    let tmax = tmax_v.x.min(tmax_v.y).min(tmax_v.z);
    if tmax >= tmin.max(0.0) {
        Some((tmin, tmax))
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug)]
struct DdaStep {
    t: f32,
    vi: [u32; 3],
}

// INVARIANT: Advances `(t, vi)` to the exact exit of `[box_min, box_max)` without epsilon bias.
#[inline]
#[allow(clippy::too_many_arguments)]
fn step_dda_box(
    v_origin: Vec3,
    dir: Vec3,
    inv_dir: Vec3,
    neg_orig_inv: Vec3,
    pos_dir: [bool; 3],
    box_min: [u32; 3],
    box_max: [u32; 3],
    cur_t: f32,
) -> DdaStep {
    let exit_coord = [
        if pos_dir[0] { box_max[0] } else { box_min[0] },
        if pos_dir[1] { box_max[1] } else { box_min[1] },
        if pos_dir[2] { box_max[2] } else { box_min[2] },
    ];
    let t_max = Vec3::new(
        exit_coord[0] as f32 * inv_dir.x + neg_orig_inv.x,
        exit_coord[1] as f32 * inv_dir.y + neg_orig_inv.y,
        exit_coord[2] as f32 * inv_dir.z + neg_orig_inv.z,
    );
    let t_exit = t_max.x.min(t_max.y).min(t_max.z);
    let next_t = cur_t.max(t_exit);

    let is_x = t_max.x <= t_max.y && t_max.x <= t_max.z;
    let is_y = !is_x && (t_max.y <= t_max.z);
    let is_z = !is_x && !is_y;
    let is_exit = [is_x, is_y, is_z];
    let outside_vi = [
        exit_coord[0].wrapping_sub(if pos_dir[0] { 0 } else { 1 }),
        exit_coord[1].wrapping_sub(if pos_dir[1] { 0 } else { 1 }),
        exit_coord[2].wrapping_sub(if pos_dir[2] { 0 } else { 1 }),
    ];
    let inside_p = (v_origin + dir * next_t).max(Vec3::ZERO);
    let inside_vi = [
        (inside_p.x as u32).clamp(box_min[0], box_max[0].saturating_sub(1)),
        (inside_p.y as u32).clamp(box_min[1], box_max[1].saturating_sub(1)),
        (inside_p.z as u32).clamp(box_min[2], box_max[2].saturating_sub(1)),
    ];

    DdaStep {
        t: next_t,
        vi: [
            if is_exit[0] { outside_vi[0] } else { inside_vi[0] },
            if is_exit[1] { outside_vi[1] } else { inside_vi[1] },
            if is_exit[2] { outside_vi[2] } else { inside_vi[2] },
        ],
    }
}

#[inline]
fn resolve_entry_normal(tmin_v: Vec3, dir: Vec3) -> Vec3 {
    let max_t = tmin_v.x.max(tmin_v.y).max(tmin_v.z);
    let eps = 1e-4 * (max_t * 0.01).max(1.0);
    let cand_x = if (tmin_v.x - max_t).abs() <= eps {
        dir.x.abs()
    } else {
        0.0
    };
    let cand_y = if (tmin_v.y - max_t).abs() <= eps {
        dir.y.abs()
    } else {
        0.0
    };
    let cand_z = if (tmin_v.z - max_t).abs() <= eps {
        dir.z.abs()
    } else {
        0.0
    };

    if cand_x >= cand_y && cand_x >= cand_z && cand_x > 0.0 {
        Vec3::new(-dir.x.signum(), 0.0, 0.0)
    } else if cand_y >= cand_z && cand_y > 0.0 {
        Vec3::new(0.0, -dir.y.signum(), 0.0)
    } else if cand_z > 0.0 {
        Vec3::new(0.0, 0.0, -dir.z.signum())
    } else if tmin_v.x >= tmin_v.y && tmin_v.x >= tmin_v.z {
        Vec3::new(-dir.x.signum(), 0.0, 0.0)
    } else if tmin_v.y >= tmin_v.z {
        Vec3::new(0.0, -dir.y.signum(), 0.0)
    } else {
        Vec3::new(0.0, 0.0, -dir.z.signum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{MaterialPalette, MAT_AIR, MAT_STONE_DEEP};

    #[test]
    fn test_cpu_raycast_hit() {
        let palette = MaterialPalette::default();
        let tree = ChunkTree64::from_generator(
            &palette,
            |_ox, oy, _oz, size| {
                if oy + size <= 32 {
                    Some(MAT_STONE_DEEP)
                } else if oy >= 32 {
                    Some(MAT_AIR)
                } else {
                    None
                }
            },
            |_x, y, _z| if y < 32 { MAT_STONE_DEEP } else { MAT_AIR },
        );

        let origin = Vec3::new(128.5, 50.0, 128.5);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let hit = raycast_chunk_tree(&tree, origin, dir, 100.0)
            .expect("INVARIANT: Downward ray toward solid floor must hit floor");
        assert_eq!(hit.material, MAT_STONE_DEEP);
        assert_eq!(hit.voxel_coord.y, 31);
        assert_eq!(hit.normal, Vec3::Y);
        assert!((hit.distance - 18.0).abs() < 0.05);
    }

    #[test]
    fn test_grazing_angle_face_and_edge() {
        let palette = MaterialPalette::default();
        let tree = ChunkTree64::from_generator(
            &palette,
            |_ox, oy, _oz, size| {
                if oy + size <= 32 {
                    Some(MAT_STONE_DEEP)
                } else if oy >= 32 {
                    Some(MAT_AIR)
                } else {
                    None
                }
            },
            |_x, y, _z| if y < 32 { MAT_STONE_DEEP } else { MAT_AIR },
        );

        // Ray traveling horizontally directly above floor face at y = 32.0:
        let skim_origin = Vec3::new(10.0, 32.0001, 10.0);
        let skim_dir = Vec3::new(1.0, 0.0, 0.0);
        let skim_hit = raycast_chunk_tree(&tree, skim_origin, skim_dir, 100.0);
        assert!(skim_hit.is_none(), "INVARIANT: Skimming above horizontal plane must not hit below plane");

        // Grazing downward ray with very shallow slope (-0.005) toward floor:
        let shallow_origin = Vec3::new(10.0, 32.5, 10.0);
        let shallow_dir = Vec3::new(1.0, -0.005, 0.0).normalize();
        let shallow_hit = raycast_chunk_tree(&tree, shallow_origin, shallow_dir, 200.0)
            .expect("INVARIANT: Grazing downward ray must reliably hit floor");
        assert_eq!(shallow_hit.material, MAT_STONE_DEEP);
        assert_eq!(shallow_hit.voxel_coord.y, 31);
        assert_eq!(shallow_hit.normal, Vec3::Y);
    }

    #[test]
    fn test_nibble_row_skipping_sparse_and_semidense() {
        let palette = MaterialPalette::default();
        let mut tree = ChunkTree64::new_empty();

        // Populate a single sparse voxel in leaf brick at (3, 0, 0):
        tree.set_voxel(3, 0, 0, MAT_STONE_DEEP, &palette);

        // Forward ray +X skipping empty voxels 0, 1, 2 in row:
        let fwd_origin = Vec3::new(0.1, 0.5, 0.5);
        let fwd_dir = Vec3::new(1.0, 0.0, 0.0);
        let fwd_hit = raycast_chunk_tree(&tree, fwd_origin, fwd_dir, 10.0)
            .expect("INVARIANT: Ray along +X row must hit voxel 3");
        assert_eq!(fwd_hit.voxel_coord, IVec3::new(3, 0, 0));
        assert_eq!(fwd_hit.normal, Vec3::new(-1.0, 0.0, 0.0));
        assert!((fwd_hit.distance - 2.9).abs() < 1e-3);

        // Backward ray -X directly hitting voxel 3:
        let bwd_origin = Vec3::new(3.9, 0.5, 0.5);
        let bwd_dir = Vec3::new(-1.0, 0.0, 0.0);
        let bwd_hit = raycast_chunk_tree(&tree, bwd_origin, bwd_dir, 10.0)
            .expect("INVARIANT: Ray along -X row must hit voxel 3");
        assert_eq!(bwd_hit.voxel_coord, IVec3::new(3, 0, 0));
        assert_eq!(bwd_hit.normal, Vec3::new(1.0, 0.0, 0.0));

        // Semi-dense row: voxels 0 and 3 are solid, 1 and 2 are empty air:
        tree.set_voxel(0, 0, 0, MAT_STONE_DEEP, &palette);
        // Cast ray from empty space between voxels 0 and 3:
        let leap_origin = Vec3::new(1.1, 0.5, 0.5);
        let leap_hit = raycast_chunk_tree(&tree, leap_origin, fwd_dir, 10.0)
            .expect("INVARIANT: Ray must leap over empty voxels 1 and 2 to hit voxel 3");
        assert_eq!(leap_hit.voxel_coord, IVec3::new(3, 0, 0));
        assert_eq!(leap_hit.normal, Vec3::new(-1.0, 0.0, 0.0));
        assert!((leap_hit.distance - 1.9).abs() < 1e-3);
    }

    #[test]
    fn test_multilevel_ancestor_caching_across_bricks() {
        let palette = MaterialPalette::default();
        let mut tree = ChunkTree64::new_empty();

        // Place a solid target in adjacent brick (x = 5, in brick covering x in 4..8):
        tree.set_voxel(5, 0, 0, MAT_STONE_DEEP, &palette);

        // Ray starts in empty brick (x in 0..4) and steps across brick boundary into adjacent brick:
        let ray_origin = Vec3::new(0.5, 0.5, 0.5);
        let ray_dir = Vec3::new(1.0, 0.0, 0.0);
        let hit = raycast_chunk_tree(&tree, ray_origin, ray_dir, 20.0)
            .expect("INVARIANT: Ray traversing adjacent brick must hit voxel 5");
        assert_eq!(hit.voxel_coord, IVec3::new(5, 0, 0));
        assert_eq!(hit.normal, Vec3::new(-1.0, 0.0, 0.0));
        assert!((hit.distance - 4.5).abs() < 1e-3);
    }
}
