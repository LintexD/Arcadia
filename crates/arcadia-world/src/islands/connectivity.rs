//! Two-tiered macro-brick graph traversal, compressive support propagation, and component clustering.
//! Owns CPU connected component graph analysis.

use super::bitwise::{
    flood_fill_brick_u64, flood_fill_support_brick_u64, FACE_X_MAX, FACE_X_MIN, FACE_Y_MAX,
    FACE_Y_MIN, FACE_Z_MAX, FACE_Z_MIN,
};
use super::query::{is_externally_anchored, VoxelQuery};
use arcadia_core::{unpack_child_index, MtfCache, MAT_AIR, MAT_BEDROCK};
use glam::IVec3;

/// Returns separate connected components of disconnected floating clusters within a bounding box.
pub fn detect_disconnected_island_components<W: VoxelQuery + ?Sized>(
    world: &W,
    min_v: IVec3,
    max_v: IVec3,
    max_island_voxels: usize,
) -> Vec<Vec<(IVec3, u8)>> {
    let dims = (max_v - min_v + IVec3::ONE).max(IVec3::ZERO);
    let total = (dims.x as usize) * (dims.y as usize) * (dims.z as usize);
    if total == 0 || total > 256 * 512 * 256 || max_island_voxels == 0 {
        return Vec::new();
    }

    let b_min = min_v.div_euclid(IVec3::splat(4));
    let b_max = max_v.div_euclid(IVec3::splat(4));
    let b_dims = b_max - b_min + IVec3::ONE;
    let bdx = b_dims.x as usize;
    let bdy = b_dims.y as usize;
    let bdz = b_dims.z as usize;
    let num_bricks = bdx * bdy * bdz;
    if num_bricks == 0 {
        return Vec::new();
    }

    let (brick_solid, brick_anchor) =
        build_brick_solid_and_anchor_masks(world, min_v, max_v, b_min, b_dims);
    let supported_masks = propagate_compressive_support(&brick_solid, &brick_anchor, b_dims);

    cluster_unsupported_components(
        world,
        &brick_solid,
        &supported_masks,
        b_min,
        b_dims,
        max_island_voxels,
    )
}

/// Returns voxel coordinates belonging to small disconnected floating clusters.
pub fn detect_disconnected_islands<W: VoxelQuery + ?Sized>(
    world: &W,
    min_v: IVec3,
    max_v: IVec3,
    max_island_voxels: usize,
) -> Vec<(IVec3, u8)> {
    detect_disconnected_island_components(world, min_v, max_v, max_island_voxels)
        .into_iter()
        .flatten()
        .collect()
}

fn build_brick_solid_and_anchor_masks<W: VoxelQuery + ?Sized>(
    world: &W,
    min_v: IVec3,
    max_v: IVec3,
    b_min: IVec3,
    b_dims: IVec3,
) -> (Vec<u64>, Vec<u64>) {
    let bdx = b_dims.x as usize;
    let bdy = b_dims.y as usize;
    let bdz = b_dims.z as usize;
    let num_bricks = bdx * bdy * bdz;
    let brick_idx = |bx: usize, by: usize, bz: usize| -> usize { bx + by * bdx + bz * bdx * bdy };

    let mut mtf_cache: MtfCache<(i32, i32, i32), u64, 16> = MtfCache::new((i32::MIN, 0, 0), 0);
    let mut brick_solid = vec![0u64; num_bricks];
    let mut brick_anchor = vec![0u64; num_bricks];

    for bz in 0..bdz {
        for by in 0..bdy {
            for bx in 0..bdx {
                let gbx = b_min.x + bx as i32;
                let gby = b_min.y + by as i32;
                let gbz = b_min.z + bz as i32;

                let raw_mask = mtf_cache.get_or_insert_with((gbx, gby, gbz), |(x, y, z)| {
                    world.get_brick_solid_mask(x, y, z)
                });
                if raw_mask == 0 {
                    continue;
                }

                let ox = gbx * 4;
                let oy = gby * 4;
                let oz = gbz * 4;

                let mut box_clip_mask = u64::MAX;
                let mut anchor_mask = 0u64;

                if ox <= min_v.x
                    || oy <= min_v.y
                    || oz <= min_v.z
                    || ox + 3 >= max_v.x
                    || oy + 3 >= max_v.y
                    || oz + 3 >= max_v.z
                    || oy <= 4
                {
                    box_clip_mask = 0;
                    for bit in 0..64u32 {
                        let (lx, ly, lz) = unpack_child_index(bit);
                        let vx = ox + lx as i32;
                        let vy = oy + ly as i32;
                        let vz = oz + lz as i32;
                        if vx >= min_v.x
                            && vx <= max_v.x
                            && vy >= min_v.y
                            && vy <= max_v.y
                            && vz >= min_v.z
                            && vz <= max_v.z
                        {
                            let bit_mask = 1u64 << bit;
                            box_clip_mask |= bit_mask;
                            if (raw_mask & bit_mask) != 0
                                && (vy <= 4
                                    || world.get_voxel(vx, vy, vz) == MAT_BEDROCK
                                    || (vy == min_v.y && is_externally_anchored(world, vx, vy, vz)))
                            {
                                anchor_mask |= bit_mask;
                            }
                        }
                    }
                }

                let idx = brick_idx(bx, by, bz);
                brick_solid[idx] = raw_mask & box_clip_mask;
                brick_anchor[idx] = anchor_mask;
            }
        }
    }

    (brick_solid, brick_anchor)
}

fn propagate_compressive_support(
    brick_solid: &[u64],
    brick_anchor: &[u64],
    b_dims: IVec3,
) -> Vec<u64> {
    let bdx = b_dims.x as usize;
    let bdy = b_dims.y as usize;
    let bdz = b_dims.z as usize;
    let num_bricks = bdx * bdy * bdz;
    let brick_idx = |bx: usize, by: usize, bz: usize| -> usize { bx + by * bdx + bz * bdx * bdy };

    let mut supported_masks = vec![0u64; num_bricks];
    let mut support_queue = Vec::with_capacity(num_bricks.min(4096));

    for bz in 0..bdz {
        for by in 0..bdy {
            for bx in 0..bdx {
                let bi = brick_idx(bx, by, bz);
                let init_supp = brick_solid[bi] & brick_anchor[bi];
                if init_supp != 0 {
                    let filled = flood_fill_support_brick_u64(init_supp, brick_solid[bi]);
                    supported_masks[bi] = filled;
                    support_queue.push((bx, by, bz, filled));
                }
            }
        }
    }

    while let Some((cbx, cby, cbz, active_mask)) = support_queue.pop() {
        // 1. +Y neighbor (upward compressive support):
        if cby + 1 < bdy && (active_mask & FACE_Y_MAX) != 0 {
            let nby = cby + 1;
            let ni = brick_idx(cbx, nby, cbz);
            let n_seed =
                ((active_mask & FACE_Y_MAX) >> 12) & brick_solid[ni] & !supported_masks[ni];
            if n_seed != 0 {
                let prev = supported_masks[ni];
                let next = flood_fill_support_brick_u64(prev | n_seed, brick_solid[ni]);
                let added = next & !prev;
                if added != 0 {
                    supported_masks[ni] = next;
                    support_queue.push((cbx, nby, cbz, added));
                }
            }
        }

        // 2. -Y neighbor (downward support):
        if cby > 0 && (active_mask & FACE_Y_MIN) != 0 {
            let nby = cby - 1;
            let ni = brick_idx(cbx, nby, cbz);
            let n_seed =
                ((active_mask & FACE_Y_MIN) << 12) & brick_solid[ni] & !supported_masks[ni];
            if n_seed != 0 {
                let prev = supported_masks[ni];
                let next = flood_fill_support_brick_u64(prev | n_seed, brick_solid[ni]);
                let added = next & !prev;
                if added != 0 {
                    supported_masks[ni] = next;
                    support_queue.push((cbx, nby, cbz, added));
                }
            }
        }

        // 3. -X neighbor:
        if cbx > 0 && (active_mask & FACE_X_MIN) != 0 {
            let nbx = cbx - 1;
            let ni = brick_idx(nbx, cby, cbz);
            let n_seed = ((active_mask & FACE_X_MIN) << 3) & brick_solid[ni] & !supported_masks[ni];
            if n_seed != 0 {
                let prev = supported_masks[ni];
                let next = flood_fill_support_brick_u64(prev | n_seed, brick_solid[ni]);
                let added = next & !prev;
                if added != 0 {
                    supported_masks[ni] = next;
                    support_queue.push((nbx, cby, cbz, added));
                }
            }
        }

        // 4. +X neighbor:
        if cbx + 1 < bdx && (active_mask & FACE_X_MAX) != 0 {
            let nbx = cbx + 1;
            let ni = brick_idx(nbx, cby, cbz);
            let n_seed = ((active_mask & FACE_X_MAX) >> 3) & brick_solid[ni] & !supported_masks[ni];
            if n_seed != 0 {
                let prev = supported_masks[ni];
                let next = flood_fill_support_brick_u64(prev | n_seed, brick_solid[ni]);
                let added = next & !prev;
                if added != 0 {
                    supported_masks[ni] = next;
                    support_queue.push((nbx, cby, cbz, added));
                }
            }
        }

        // 5. -Z neighbor:
        if cbz > 0 && (active_mask & FACE_Z_MIN) != 0 {
            let nbz = cbz - 1;
            let ni = brick_idx(cbx, cby, nbz);
            let n_seed =
                ((active_mask & FACE_Z_MIN) << 48) & brick_solid[ni] & !supported_masks[ni];
            if n_seed != 0 {
                let prev = supported_masks[ni];
                let next = flood_fill_support_brick_u64(prev | n_seed, brick_solid[ni]);
                let added = next & !prev;
                if added != 0 {
                    supported_masks[ni] = next;
                    support_queue.push((cbx, cby, nbz, added));
                }
            }
        }

        // 6. +Z neighbor:
        if cbz + 1 < bdz && (active_mask & FACE_Z_MAX) != 0 {
            let nbz = cbz + 1;
            let ni = brick_idx(cbx, cby, nbz);
            let n_seed =
                ((active_mask & FACE_Z_MAX) >> 48) & brick_solid[ni] & !supported_masks[ni];
            if n_seed != 0 {
                let prev = supported_masks[ni];
                let next = flood_fill_support_brick_u64(prev | n_seed, brick_solid[ni]);
                let added = next & !prev;
                if added != 0 {
                    supported_masks[ni] = next;
                    support_queue.push((cbx, cby, nbz, added));
                }
            }
        }
    }

    supported_masks
}

fn cluster_unsupported_components<W: VoxelQuery + ?Sized>(
    world: &W,
    brick_solid: &[u64],
    supported_masks: &[u64],
    b_min: IVec3,
    b_dims: IVec3,
    max_island_voxels: usize,
) -> Vec<Vec<(IVec3, u8)>> {
    let bdx = b_dims.x as usize;
    let bdy = b_dims.y as usize;
    let bdz = b_dims.z as usize;
    let num_bricks = bdx * bdy * bdz;
    let brick_idx = |bx: usize, by: usize, bz: usize| -> usize { bx + by * bdx + bz * bdx * bdy };

    let mut unsupported_solid = vec![0u64; num_bricks];
    let mut total_unsupported = 0;
    for i in 0..num_bricks {
        let u = brick_solid[i] & !supported_masks[i];
        unsupported_solid[i] = u;
        total_unsupported += u.count_ones() as usize;
    }

    if total_unsupported == 0 {
        return Vec::new();
    }

    let mut visited_masks = vec![0u64; num_bricks];
    let mut comp_masks = vec![0u64; num_bricks];
    let mut touched_bricks = Vec::with_capacity(32);
    let mut work_queue = Vec::with_capacity(32);
    let mut components = Vec::new();

    for bz in 0..bdz {
        for by in 0..bdy {
            for bx in 0..bdx {
                let start_b = brick_idx(bx, by, bz);
                let mut unvisited = unsupported_solid[start_b] & !visited_masks[start_b];
                while unvisited != 0 {
                    let seed_bit = unvisited.trailing_zeros();
                    let seed_mask = 1u64 << seed_bit;

                    touched_bricks.clear();
                    work_queue.clear();

                    let initial_fill = flood_fill_brick_u64(seed_mask, unsupported_solid[start_b]);
                    visited_masks[start_b] |= initial_fill;
                    comp_masks[start_b] = initial_fill;
                    touched_bricks.push(start_b);
                    work_queue.push((bx, by, bz, initial_fill));

                    let mut voxel_count = initial_fill.count_ones() as usize;

                    while let Some((cbx, cby, cbz, active_mask)) = work_queue.pop() {
                        // 1. -X
                        if cbx > 0 && (active_mask & FACE_X_MIN) != 0 {
                            let nbx = cbx - 1;
                            let ni = brick_idx(nbx, cby, cbz);
                            let n_seed = ((active_mask & FACE_X_MIN) << 3)
                                & unsupported_solid[ni]
                                & !comp_masks[ni];
                            if n_seed != 0 {
                                let prev = comp_masks[ni];
                                if prev == 0 {
                                    touched_bricks.push(ni);
                                }
                                let next =
                                    flood_fill_brick_u64(prev | n_seed, unsupported_solid[ni]);
                                let added = next & !prev;
                                if added != 0 {
                                    comp_masks[ni] = next;
                                    visited_masks[ni] |= added;
                                    voxel_count += added.count_ones() as usize;
                                    work_queue.push((nbx, cby, cbz, added));
                                }
                            }
                        }
                        // 2. +X
                        if cbx + 1 < bdx && (active_mask & FACE_X_MAX) != 0 {
                            let nbx = cbx + 1;
                            let ni = brick_idx(nbx, cby, cbz);
                            let n_seed = ((active_mask & FACE_X_MAX) >> 3)
                                & unsupported_solid[ni]
                                & !comp_masks[ni];
                            if n_seed != 0 {
                                let prev = comp_masks[ni];
                                if prev == 0 {
                                    touched_bricks.push(ni);
                                }
                                let next =
                                    flood_fill_brick_u64(prev | n_seed, unsupported_solid[ni]);
                                let added = next & !prev;
                                if added != 0 {
                                    comp_masks[ni] = next;
                                    visited_masks[ni] |= added;
                                    voxel_count += added.count_ones() as usize;
                                    work_queue.push((nbx, cby, cbz, added));
                                }
                            }
                        }
                        // 3. -Y (downward gravity connection)
                        if cby > 0 && (active_mask & FACE_Y_MIN) != 0 {
                            let nby = cby - 1;
                            let ni = brick_idx(cbx, nby, cbz);
                            let n_seed = ((active_mask & FACE_Y_MIN) << 12)
                                & unsupported_solid[ni]
                                & !comp_masks[ni];
                            if n_seed != 0 {
                                let prev = comp_masks[ni];
                                if prev == 0 {
                                    touched_bricks.push(ni);
                                }
                                let next =
                                    flood_fill_brick_u64(prev | n_seed, unsupported_solid[ni]);
                                let added = next & !prev;
                                if added != 0 {
                                    comp_masks[ni] = next;
                                    visited_masks[ni] |= added;
                                    voxel_count += added.count_ones() as usize;
                                    work_queue.push((cbx, nby, cbz, added));
                                }
                            }
                        }
                        // 4. +Y (upward connection)
                        if cby + 1 < bdy && (active_mask & FACE_Y_MAX) != 0 {
                            let nby = cby + 1;
                            let ni = brick_idx(cbx, nby, cbz);
                            let n_seed = ((active_mask & FACE_Y_MAX) >> 12)
                                & unsupported_solid[ni]
                                & !comp_masks[ni];
                            if n_seed != 0 {
                                let prev = comp_masks[ni];
                                if prev == 0 {
                                    touched_bricks.push(ni);
                                }
                                let next =
                                    flood_fill_brick_u64(prev | n_seed, unsupported_solid[ni]);
                                let added = next & !prev;
                                if added != 0 {
                                    comp_masks[ni] = next;
                                    visited_masks[ni] |= added;
                                    voxel_count += added.count_ones() as usize;
                                    work_queue.push((cbx, nby, cbz, added));
                                }
                            }
                        }
                        // 5. -Z
                        if cbz > 0 && (active_mask & FACE_Z_MIN) != 0 {
                            let nbz = cbz - 1;
                            let ni = brick_idx(cbx, cby, nbz);
                            let n_seed = ((active_mask & FACE_Z_MIN) << 48)
                                & unsupported_solid[ni]
                                & !comp_masks[ni];
                            if n_seed != 0 {
                                let prev = comp_masks[ni];
                                if prev == 0 {
                                    touched_bricks.push(ni);
                                }
                                let next =
                                    flood_fill_brick_u64(prev | n_seed, unsupported_solid[ni]);
                                let added = next & !prev;
                                if added != 0 {
                                    comp_masks[ni] = next;
                                    visited_masks[ni] |= added;
                                    voxel_count += added.count_ones() as usize;
                                    work_queue.push((cbx, cby, nbz, added));
                                }
                            }
                        }
                        // 6. +Z
                        if cbz + 1 < bdz && (active_mask & FACE_Z_MAX) != 0 {
                            let nbz = cbz + 1;
                            let ni = brick_idx(cbx, cby, nbz);
                            let n_seed = ((active_mask & FACE_Z_MAX) >> 48)
                                & unsupported_solid[ni]
                                & !comp_masks[ni];
                            if n_seed != 0 {
                                let prev = comp_masks[ni];
                                if prev == 0 {
                                    touched_bricks.push(ni);
                                }
                                let next =
                                    flood_fill_brick_u64(prev | n_seed, unsupported_solid[ni]);
                                let added = next & !prev;
                                if added != 0 {
                                    comp_masks[ni] = next;
                                    visited_masks[ni] |= added;
                                    voxel_count += added.count_ones() as usize;
                                    work_queue.push((cbx, cby, nbz, added));
                                }
                            }
                        }
                    }

                    if voxel_count <= max_island_voxels {
                        let mut comp_voxels = Vec::with_capacity(voxel_count);
                        for &ti in &touched_bricks {
                            let mask = comp_masks[ti];
                            comp_masks[ti] = 0;
                            let t_bx = ti % bdx;
                            let t_by = (ti / bdx) % bdy;
                            let t_bz = ti / (bdx * bdy);
                            let ox = (b_min.x + t_bx as i32) * 4;
                            let oy = (b_min.y + t_by as i32) * 4;
                            let oz = (b_min.z + t_bz as i32) * 4;

                            let mut rem = mask;
                            while rem != 0 {
                                let bit = rem.trailing_zeros();
                                let (lx, ly, lz) = unpack_child_index(bit);
                                let gv = IVec3::new(ox + lx as i32, oy + ly as i32, oz + lz as i32);
                                let mat = world.get_voxel(gv.x, gv.y, gv.z);
                                if mat != MAT_AIR {
                                    comp_voxels.push((gv, mat));
                                }
                                rem &= rem - 1;
                            }
                        }
                        if !comp_voxels.is_empty() {
                            components.push(comp_voxels);
                        }
                    } else {
                        for &ti in &touched_bricks {
                            comp_masks[ti] = 0;
                        }
                    }

                    unvisited = unsupported_solid[start_b] & !visited_masks[start_b];
                }
            }
        }
    }

    components
}
