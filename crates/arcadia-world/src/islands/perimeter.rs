//! Targeted perimeter-seeded island detection around spherical carve craters.
//! Owns CPU perimeter seed harvesting, 2D boundary surface traversal, and deferred 3D interior fill.

use super::bitwise::{
    extract_brick_surface_mask, flood_fill_brick_u64, FACE_X_MAX, FACE_X_MIN, FACE_Y_MAX,
    FACE_Y_MIN, FACE_Z_MAX, FACE_Z_MIN,
};
use super::query::VoxelQuery;
use arcadia_core::{MAT_AIR, MAT_BEDROCK};
use glam::{IVec3, Vec3};
use std::collections::{HashMap, VecDeque};

/// Detects disconnected floating islands by expanding 6-connected graph components
/// outward from the perimeter seeds of a spherical carve crater, with optional
/// voxel count, surface count, and distance bounds.
pub fn detect_disconnected_islands_from_carve_bounded<W: VoxelQuery + ?Sized>(
    world: &W,
    center_v: Vec3,
    radius_v: f32,
    max_island_voxels: usize,
    max_depth_v: f32,
    max_surface_voxels: usize,
) -> Vec<Vec<(IVec3, u8)>> {
    if radius_v <= 0.0 || max_island_voxels == 0 || max_depth_v <= 0.0 {
        return Vec::new();
    }

    let effective_max_surface = if max_surface_voxels > 0 {
        max_surface_voxels
    } else {
        max_island_voxels
    };

    let min_p = (center_v - Vec3::splat(radius_v + 2.5)).floor().as_ivec3();
    let max_p = (center_v + Vec3::splat(radius_v + 2.5)).ceil().as_ivec3();

    let min_b = min_p.div_euclid(IVec3::splat(4));
    let max_b = max_p.div_euclid(IVec3::splat(4));

    let mut brick_table: HashMap<(i32, i32, i32), BrickEntry> = HashMap::new();
    let mut seed_bricks: Vec<((i32, i32, i32), u64)> = Vec::new();

    for gbz in min_b.z..=max_b.z {
        for gby in min_b.y..=max_b.y {
            for gbx in min_b.x..=max_b.x {
                let solid_mask = world.get_brick_solid_mask(gbx, gby, gbz);
                if solid_mask == 0 {
                    continue;
                }
                brick_table.insert(
                    (gbx, gby, gbz),
                    BrickEntry {
                        solid_mask,
                        surface_mask: None,
                        visited_global: 0,
                        grounded_mask: 0,
                    },
                );

                let mut bits = solid_mask;
                let mut brick_seed_mask = 0u64;

                while bits != 0 {
                    let b = bits.trailing_zeros();
                    let bit_val = 1u64 << b;
                    bits &= !bit_val;

                    let lx = (b & 3) as i32;
                    let ly = ((b >> 2) & 3) as i32;
                    let lz = ((b >> 4) & 3) as i32;

                    let vx = gbx * 4 + lx;
                    let vy = gby * 4 + ly;
                    let vz = gbz * 4 + lz;

                    let d = Vec3::new(vx as f32 + 0.5, vy as f32 + 0.5, vz as f32 + 0.5)
                        .distance(center_v);
                    if d > radius_v && d <= radius_v + 2.5 {
                        let mut touches_crater = false;
                        for offset in CARDINAL_OFFSETS {
                            let n = IVec3::new(vx, vy, vz) + offset;
                            let n_dist =
                                Vec3::new(n.x as f32 + 0.5, n.y as f32 + 0.5, n.z as f32 + 0.5)
                                    .distance(center_v);
                            if n_dist <= radius_v || world.get_voxel(n.x, n.y, n.z) == MAT_AIR {
                                touches_crater = true;
                                break;
                            }
                        }
                        if touches_crater {
                            brick_seed_mask |= bit_val;
                        }
                    }
                }

                if brick_seed_mask != 0 {
                    seed_bricks.push(((gbx, gby, gbz), brick_seed_mask));
                }
            }
        }
    }

    if seed_bricks.is_empty() {
        return Vec::new();
    }

    let mut detached_islands = Vec::new();

    for &((seed_gbx, seed_gby, seed_gbz), seed_mask) in &seed_bricks {
        let mut unvisited_seeds = seed_mask;
        if let Some(entry) = brick_table.get(&(seed_gbx, seed_gby, seed_gbz)) {
            unvisited_seeds &= !entry.visited_global;
        }

        while unvisited_seeds != 0 {
            let bit = unvisited_seeds.trailing_zeros();
            let bit_mask = 1u64 << bit;
            unvisited_seeds &= !bit_mask;

            if let Some(entry) = brick_table.get(&(seed_gbx, seed_gby, seed_gbz)) {
                if (entry.visited_global & bit_mask) != 0 {
                    continue;
                }
                if (entry.grounded_mask & bit_mask) != 0 {
                    continue;
                }
            }

            let mut comp_map: HashMap<(i32, i32, i32), u64> = HashMap::new();
            let mut comp_voxel_count = 0usize;
            let mut queue = VecDeque::new();
            let mut is_grounded = false;

            queue.push_back(((seed_gbx, seed_gby, seed_gbz), bit_mask));

            while let Some(((cbx, cby, cbz), cur_seed_mask)) = queue.pop_front() {
                let (surface_mask, grounded_mask) = {
                    let entry = brick_table.entry((cbx, cby, cbz)).or_insert_with(|| {
                        let solid_mask = world.get_brick_solid_mask(cbx, cby, cbz);
                        BrickEntry {
                            solid_mask,
                            surface_mask: None,
                            visited_global: 0,
                            grounded_mask: 0,
                        }
                    });
                    let s_mask = entry.get_or_compute_surface_mask(world, cbx, cby, cbz);
                    (s_mask, entry.grounded_mask)
                };

                let filled = flood_fill_brick_u64(cur_seed_mask, surface_mask);
                if filled == 0 {
                    continue;
                }

                if (filled & grounded_mask) != 0 {
                    is_grounded = true;
                    break;
                }

                let comp_mask = comp_map.entry((cbx, cby, cbz)).or_insert(0);
                let new_bits = filled & !*comp_mask;
                if new_bits == 0 {
                    continue;
                }

                *comp_mask |= new_bits;
                comp_voxel_count += new_bits.count_ones() as usize;

                // INVARIANT: Terrain bedrock floor is located at y <= 4.
                let is_bedrock = if cby < 1 {
                    true
                } else if cby == 1 {
                    (new_bits & FACE_Y_MIN) != 0
                } else {
                    false
                };
                if is_bedrock {
                    is_grounded = true;
                    break;
                }

                if cby <= 2 {
                    let mut temp = new_bits;
                    let mut found_bedrock = false;
                    while temp != 0 {
                        let b = temp.trailing_zeros();
                        temp &= !(1u64 << b);
                        let vx = cbx * 4 + (b & 3) as i32;
                        let vy = cby * 4 + ((b >> 2) & 3) as i32;
                        let vz = cbz * 4 + ((b >> 4) & 3) as i32;
                        if world.get_voxel(vx, vy, vz) == MAT_BEDROCK {
                            found_bedrock = true;
                            break;
                        }
                    }
                    if found_bedrock {
                        is_grounded = true;
                        break;
                    }
                }

                // INVARIANT: Short-circuit when surface voxel count exceeds maximum allowable surface budget.
                if comp_voxel_count > effective_max_surface {
                    is_grounded = true;
                    break;
                }

                // INVARIANT: Short-circuit when island extends past the maximum probe distance from crater.
                if max_depth_v.is_finite() {
                    let max_r = radius_v + max_depth_v;
                    let max_r_sq = max_r * max_r;
                    let mut temp = new_bits;
                    let mut exceeded_depth = false;
                    while temp != 0 {
                        let b = temp.trailing_zeros();
                        temp &= !(1u64 << b);
                        let vx = cbx * 4 + (b & 3) as i32;
                        let vy = cby * 4 + ((b >> 2) & 3) as i32;
                        let vz = cbz * 4 + ((b >> 4) & 3) as i32;
                        let pos = Vec3::new(vx as f32 + 0.5, vy as f32 + 0.5, vz as f32 + 0.5);
                        if pos.distance_squared(center_v) > max_r_sq {
                            exceeded_depth = true;
                            break;
                        }
                    }
                    if exceeded_depth {
                        is_grounded = true;
                        break;
                    }
                }

                if (new_bits & FACE_X_MAX) != 0 {
                    let n_seed = (new_bits & FACE_X_MAX) >> 3;
                    let n_coord = (cbx + 1, cby, cbz);
                    if propagate_to_neighbor(
                        &mut brick_table,
                        &comp_map,
                        &mut queue,
                        world,
                        n_coord,
                        n_seed,
                    ) {
                        is_grounded = true;
                        break;
                    }
                }
                if (new_bits & FACE_X_MIN) != 0 {
                    let n_seed = (new_bits & FACE_X_MIN) << 3;
                    let n_coord = (cbx - 1, cby, cbz);
                    if propagate_to_neighbor(
                        &mut brick_table,
                        &comp_map,
                        &mut queue,
                        world,
                        n_coord,
                        n_seed,
                    ) {
                        is_grounded = true;
                        break;
                    }
                }
                if (new_bits & FACE_Y_MAX) != 0 {
                    let n_seed = (new_bits & FACE_Y_MAX) >> 12;
                    let n_coord = (cbx, cby + 1, cbz);
                    if propagate_to_neighbor(
                        &mut brick_table,
                        &comp_map,
                        &mut queue,
                        world,
                        n_coord,
                        n_seed,
                    ) {
                        is_grounded = true;
                        break;
                    }
                }
                if (new_bits & FACE_Y_MIN) != 0 {
                    let n_seed = (new_bits & FACE_Y_MIN) << 12;
                    let n_coord = (cbx, cby - 1, cbz);
                    if propagate_to_neighbor(
                        &mut brick_table,
                        &comp_map,
                        &mut queue,
                        world,
                        n_coord,
                        n_seed,
                    ) {
                        is_grounded = true;
                        break;
                    }
                }
                if (new_bits & FACE_Z_MAX) != 0 {
                    let n_seed = (new_bits & FACE_Z_MAX) >> 48;
                    let n_coord = (cbx, cby, cbz + 1);
                    if propagate_to_neighbor(
                        &mut brick_table,
                        &comp_map,
                        &mut queue,
                        world,
                        n_coord,
                        n_seed,
                    ) {
                        is_grounded = true;
                        break;
                    }
                }
                if (new_bits & FACE_Z_MIN) != 0 {
                    let n_seed = (new_bits & FACE_Z_MIN) << 48;
                    let n_coord = (cbx, cby, cbz - 1);
                    if propagate_to_neighbor(
                        &mut brick_table,
                        &comp_map,
                        &mut queue,
                        world,
                        n_coord,
                        n_seed,
                    ) {
                        is_grounded = true;
                        break;
                    }
                }
            }

            if is_grounded {
                for (coord, mask) in comp_map {
                    if let Some(entry) = brick_table.get_mut(&coord) {
                        entry.visited_global |= mask;
                        entry.grounded_mask |= mask;
                    }
                }
            } else {
                // INVARIANT: Once a closed 2D surface shell is confirmed disconnected,
                // collect all enclosed interior solid voxels via 3D flood fill.
                let mut island_solid_map: HashMap<(i32, i32, i32), u64> = HashMap::new();
                let mut fill_queue: VecDeque<((i32, i32, i32), u64)> = VecDeque::new();
                let mut total_island_voxels = 0usize;

                for (&coord, &s_mask) in &comp_map {
                    fill_queue.push_back((coord, s_mask));
                    island_solid_map.insert(coord, s_mask);
                    total_island_voxels += s_mask.count_ones() as usize;
                }

                let mut deferred_grounded = total_island_voxels > max_island_voxels;
                while !deferred_grounded {
                    let Some(((cbx, cby, cbz), cur_seeds)) = fill_queue.pop_front() else {
                        break;
                    };
                    let solid_mask = brick_table
                        .get(&(cbx, cby, cbz))
                        .map_or(0, |e| e.solid_mask);
                    let filled = flood_fill_brick_u64(cur_seeds, solid_mask);
                    if filled == 0 {
                        continue;
                    }

                    let cur_entry = island_solid_map.entry((cbx, cby, cbz)).or_insert(0);
                    let new_bits = filled & !*cur_entry;
                    if new_bits != 0 {
                        *cur_entry |= new_bits;
                        total_island_voxels += new_bits.count_ones() as usize;
                        if total_island_voxels > max_island_voxels {
                            deferred_grounded = true;
                            break;
                        }
                    }

                    if (new_bits & FACE_X_MAX) != 0 {
                        let n_seed = (new_bits & FACE_X_MAX) >> 3;
                        let n_coord = (cbx + 1, cby, cbz);
                        propagate_solid_to_neighbor(
                            &mut brick_table,
                            &island_solid_map,
                            &mut fill_queue,
                            world,
                            n_coord,
                            n_seed,
                        );
                    }
                    if (new_bits & FACE_X_MIN) != 0 {
                        let n_seed = (new_bits & FACE_X_MIN) << 3;
                        let n_coord = (cbx - 1, cby, cbz);
                        propagate_solid_to_neighbor(
                            &mut brick_table,
                            &island_solid_map,
                            &mut fill_queue,
                            world,
                            n_coord,
                            n_seed,
                        );
                    }
                    if (new_bits & FACE_Y_MAX) != 0 {
                        let n_seed = (new_bits & FACE_Y_MAX) >> 12;
                        let n_coord = (cbx, cby + 1, cbz);
                        propagate_solid_to_neighbor(
                            &mut brick_table,
                            &island_solid_map,
                            &mut fill_queue,
                            world,
                            n_coord,
                            n_seed,
                        );
                    }
                    if (new_bits & FACE_Y_MIN) != 0 {
                        let n_seed = (new_bits & FACE_Y_MIN) << 12;
                        let n_coord = (cbx, cby - 1, cbz);
                        propagate_solid_to_neighbor(
                            &mut brick_table,
                            &island_solid_map,
                            &mut fill_queue,
                            world,
                            n_coord,
                            n_seed,
                        );
                    }
                    if (new_bits & FACE_Z_MAX) != 0 {
                        let n_seed = (new_bits & FACE_Z_MAX) >> 48;
                        let n_coord = (cbx, cby, cbz + 1);
                        propagate_solid_to_neighbor(
                            &mut brick_table,
                            &island_solid_map,
                            &mut fill_queue,
                            world,
                            n_coord,
                            n_seed,
                        );
                    }
                    if (new_bits & FACE_Z_MIN) != 0 {
                        let n_seed = (new_bits & FACE_Z_MIN) << 48;
                        let n_coord = (cbx, cby, cbz - 1);
                        propagate_solid_to_neighbor(
                            &mut brick_table,
                            &island_solid_map,
                            &mut fill_queue,
                            world,
                            n_coord,
                            n_seed,
                        );
                    }
                }

                if deferred_grounded {
                    for (coord, mask) in island_solid_map {
                        if let Some(entry) = brick_table.get_mut(&coord) {
                            entry.visited_global |= mask;
                            entry.grounded_mask |= mask;
                        }
                    }
                } else {
                    for (&coord, &mask) in &island_solid_map {
                        if let Some(entry) = brick_table.get_mut(&coord) {
                            entry.visited_global |= mask;
                        }
                    }
                    let mut comp_voxels = Vec::with_capacity(total_island_voxels);
                    for (coord, mask) in island_solid_map {
                        let mut bits = mask;
                        while bits != 0 {
                            let b = bits.trailing_zeros();
                            bits &= !(1u64 << b);
                            let vx = coord.0 * 4 + (b & 3) as i32;
                            let vy = coord.1 * 4 + ((b >> 2) & 3) as i32;
                            let vz = coord.2 * 4 + ((b >> 4) & 3) as i32;
                            let mat = world.get_voxel(vx, vy, vz);
                            comp_voxels.push((IVec3::new(vx, vy, vz), mat));
                        }
                    }
                    detached_islands.push(comp_voxels);
                }
            }

            if let Some(entry) = brick_table.get(&(seed_gbx, seed_gby, seed_gbz)) {
                unvisited_seeds &= !entry.visited_global;
            }
        }
    }

    detached_islands
}

/// Detects disconnected floating islands by expanding 6-connected graph components
/// outward from the perimeter seeds of a spherical carve crater.
pub fn detect_disconnected_islands_from_carve<W: VoxelQuery + ?Sized>(
    world: &W,
    center_v: Vec3,
    radius_v: f32,
    max_island_voxels: usize,
) -> Vec<Vec<(IVec3, u8)>> {
    detect_disconnected_islands_from_carve_bounded(
        world,
        center_v,
        radius_v,
        max_island_voxels,
        f32::INFINITY,
        max_island_voxels,
    )
}

struct BrickEntry {
    solid_mask: u64,
    surface_mask: Option<u64>,
    visited_global: u64,
    grounded_mask: u64,
}

impl BrickEntry {
    #[inline]
    fn get_or_compute_surface_mask<W: VoxelQuery + ?Sized>(
        &mut self,
        world: &W,
        gbx: i32,
        gby: i32,
        gbz: i32,
    ) -> u64 {
        if let Some(mask) = self.surface_mask {
            return mask;
        }
        if self.solid_mask == 0 {
            self.surface_mask = Some(0);
            return 0;
        }

        let neg_x = world.get_brick_solid_mask(gbx - 1, gby, gbz);
        let pos_x = world.get_brick_solid_mask(gbx + 1, gby, gbz);
        let neg_y = world.get_brick_solid_mask(gbx, gby - 1, gbz);
        let pos_y = world.get_brick_solid_mask(gbx, gby + 1, gbz);
        let neg_z = world.get_brick_solid_mask(gbx, gby, gbz - 1);
        let pos_z = world.get_brick_solid_mask(gbx, gby, gbz + 1);

        let mask =
            extract_brick_surface_mask(self.solid_mask, neg_x, pos_x, neg_y, pos_y, neg_z, pos_z);
        self.surface_mask = Some(mask);
        mask
    }
}

#[inline]
fn propagate_to_neighbor<W: VoxelQuery + ?Sized>(
    brick_table: &mut HashMap<(i32, i32, i32), BrickEntry>,
    comp_map: &HashMap<(i32, i32, i32), u64>,
    queue: &mut VecDeque<((i32, i32, i32), u64)>,
    world: &W,
    n_coord: (i32, i32, i32),
    n_seed: u64,
) -> bool {
    let entry = brick_table.entry(n_coord).or_insert_with(|| {
        let solid_mask = world.get_brick_solid_mask(n_coord.0, n_coord.1, n_coord.2);
        BrickEntry {
            solid_mask,
            surface_mask: None,
            visited_global: 0,
            grounded_mask: 0,
        }
    });

    let surface_mask = entry.get_or_compute_surface_mask(world, n_coord.0, n_coord.1, n_coord.2);
    let valid_n_seed = n_seed & surface_mask;
    if valid_n_seed == 0 {
        return false;
    }

    if (valid_n_seed & entry.grounded_mask) != 0 {
        return true;
    }

    let n_comp_mask = comp_map.get(&n_coord).copied().unwrap_or(0);
    let needed_seed = valid_n_seed & !n_comp_mask;
    if needed_seed != 0 {
        queue.push_back((n_coord, needed_seed));
    }
    false
}

#[inline]
fn propagate_solid_to_neighbor<W: VoxelQuery + ?Sized>(
    brick_table: &mut HashMap<(i32, i32, i32), BrickEntry>,
    island_solid_map: &HashMap<(i32, i32, i32), u64>,
    queue: &mut VecDeque<((i32, i32, i32), u64)>,
    world: &W,
    n_coord: (i32, i32, i32),
    n_seed: u64,
) {
    let entry = brick_table.entry(n_coord).or_insert_with(|| {
        let solid_mask = world.get_brick_solid_mask(n_coord.0, n_coord.1, n_coord.2);
        BrickEntry {
            solid_mask,
            surface_mask: None,
            visited_global: 0,
            grounded_mask: 0,
        }
    });

    let valid_n_seed = n_seed & entry.solid_mask;
    if valid_n_seed == 0 {
        return;
    }

    let n_comp_mask = island_solid_map.get(&n_coord).copied().unwrap_or(0);
    let needed_seed = valid_n_seed & !n_comp_mask;
    if needed_seed != 0 {
        queue.push_back((n_coord, needed_seed));
    }
}

const CARDINAL_OFFSETS: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];
