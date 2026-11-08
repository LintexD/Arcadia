//! Destructive and additive voxel operations, island flood-fill, and world-grid baking.
//! Owns CPU voxel modification, topology analysis, and island splitting.

use super::construction::compute_emissive_light;
use super::types::{RigidStepOutcome, RigidVoxelBody};
use arcadia_core::{MaterialId, MaterialPalette, RayHit, MAT_AIR, VOXELS_PER_METER, VOXEL_SIZE};
use arcadia_world::islands::bitwise::{
    flood_fill_brick_u64, FACE_X_MAX, FACE_X_MIN, FACE_Y_MAX, FACE_Y_MIN, FACE_Z_MAX, FACE_Z_MIN,
};
use arcadia_world::DebrisParticle;
use glam::{IVec3, Mat3, Vec3};

/// Maximum number of voxels for a severed island fragment to be converted directly into flying debris particles
/// instead of spawning an independent dynamic rigid body.
pub const SMALL_ISLAND_PARTICLE_MAX_VOXELS: usize = 8;

impl RigidVoxelBody {
    /// Maximum number of voxels for a severed island fragment to be converted directly into flying debris particles
    /// instead of spawning an independent dynamic rigid body.
    pub const SMALL_ISLAND_PARTICLE_MAX_VOXELS: usize = SMALL_ISLAND_PARTICLE_MAX_VOXELS;
    /// Raycasts through this body's local CS64 tree in world space.
    pub fn raycast_meters(
        &self,
        ray_origin_world: Vec3,
        ray_dir_world: Vec3,
        max_dist_world: f32,
    ) -> Option<RayHit> {
        let inv_rot = Mat3::from_quat(self.orientation).transpose();
        let local_origin_m = inv_rot * (ray_origin_world - self.position) + self.com_local_m;
        let local_dir = inv_rot * ray_dir_world;

        let hit = arcadia_core::raycast_chunk_tree(
            &self.local_tree,
            local_origin_m * (VOXELS_PER_METER as f32),
            local_dir,
            max_dist_world * (VOXELS_PER_METER as f32),
        )?;

        let rot = Mat3::from_quat(self.orientation);
        let hit_pos_world = self.position + rot * (hit.hit_pos * VOXEL_SIZE - self.com_local_m);
        let normal_world = rot * hit.normal;
        let dist_world = (hit_pos_world - ray_origin_world).length();

        Some(RayHit {
            distance: dist_world,
            hit_pos: hit_pos_world,
            normal: normal_world,
            voxel_coord: hit.voxel_coord,
            material: hit.material,
        })
    }

    /// Partitions a list of local voxels `(lx, ly, lz, mat)` into distinct 6-connected components.
    /// Uses 64-bit 4x4x4 brick bitboards and register wavefront BFS for zero-SipHash performance.
    pub fn find_connected_components(
        voxels: &[(u32, u32, u32, MaterialId)],
    ) -> Vec<Vec<(u32, u32, u32, MaterialId)>> {
        if voxels.is_empty() {
            return Vec::new();
        }

        let mut min_c = IVec3::splat(i32::MAX);
        let mut max_c = IVec3::splat(i32::MIN);
        for &(lx, ly, lz, _) in voxels {
            let c = IVec3::new(lx as i32, ly as i32, lz as i32);
            min_c = min_c.min(c);
            max_c = max_c.max(c);
        }
        let dims = max_c - min_c + IVec3::ONE;
        let b_dims = (dims + IVec3::splat(3)) / 4;
        let bdx = b_dims.x as usize;
        let bdy = b_dims.y as usize;
        let bdz = b_dims.z as usize;
        let num_bricks = bdx * bdy * bdz;

        // INVARIANT: 64-bit 4x4x4 brick bitboard wavefront BFS when volume is under ~2 million voxels (<= 32,768 bricks).
        if num_bricks > 0 && num_bricks <= 32_768 {
            let mut brick_solid = vec![0u64; num_bricks];
            let mut brick_mats = vec![0u8; num_bricks * 64];

            for &(lx, ly, lz, mat) in voxels {
                let dx = (lx as i32 - min_c.x) as usize;
                let dy = (ly as i32 - min_c.y) as usize;
                let dz = (lz as i32 - min_c.z) as usize;
                let bx = dx >> 2;
                let by = dy >> 2;
                let bz = dz >> 2;
                let rx = (dx & 3) as u32;
                let ry = (dy & 3) as u32;
                let rz = (dz & 3) as u32;
                let bit_idx = (rx | (ry << 2) | (rz << 4)) as usize;
                let b_idx = bx + by * bdx + bz * (bdx * bdy);
                brick_solid[b_idx] |= 1u64 << bit_idx;
                brick_mats[b_idx * 64 + bit_idx] = mat;
            }

            let mut unvisited_solid = brick_solid;
            let mut comp_masks = vec![0u64; num_bricks];
            let mut touched_bricks = Vec::with_capacity(32);
            let mut queue = Vec::with_capacity(32);
            let mut components = Vec::new();

            for start_b in 0..num_bricks {
                while unvisited_solid[start_b] != 0 {
                    let seed_bit = unvisited_solid[start_b].trailing_zeros();
                    let seed_mask = 1u64 << seed_bit;

                    let initial_fill = flood_fill_brick_u64(seed_mask, unvisited_solid[start_b]);
                    unvisited_solid[start_b] &= !initial_fill;
                    comp_masks[start_b] = initial_fill;
                    touched_bricks.push(start_b);

                    let cbx = start_b % bdx;
                    let cby = (start_b / bdx) % bdy;
                    let cbz = start_b / (bdx * bdy);
                    queue.push((cbx, cby, cbz, initial_fill));

                    while let Some((cbx, cby, cbz, active_mask)) = queue.pop() {
                        let c_idx = cbx + cby * bdx + cbz * (bdx * bdy);

                        // INVARIANT: Propagate -X boundary face into neighbor brick.
                        if cbx > 0 && (active_mask & FACE_X_MIN) != 0 {
                            let nbx = cbx - 1;
                            let ni = c_idx - 1;
                            let n_seed = ((active_mask & FACE_X_MIN) << 3) & unvisited_solid[ni];
                            if n_seed != 0 {
                                let next = flood_fill_brick_u64(n_seed, unvisited_solid[ni]);
                                unvisited_solid[ni] &= !next;
                                if comp_masks[ni] == 0 {
                                    touched_bricks.push(ni);
                                }
                                comp_masks[ni] |= next;
                                queue.push((nbx, cby, cbz, next));
                            }
                        }

                        // INVARIANT: Propagate +X boundary face into neighbor brick.
                        if cbx + 1 < bdx && (active_mask & FACE_X_MAX) != 0 {
                            let nbx = cbx + 1;
                            let ni = c_idx + 1;
                            let n_seed = ((active_mask & FACE_X_MAX) >> 3) & unvisited_solid[ni];
                            if n_seed != 0 {
                                let next = flood_fill_brick_u64(n_seed, unvisited_solid[ni]);
                                unvisited_solid[ni] &= !next;
                                if comp_masks[ni] == 0 {
                                    touched_bricks.push(ni);
                                }
                                comp_masks[ni] |= next;
                                queue.push((nbx, cby, cbz, next));
                            }
                        }

                        // INVARIANT: Propagate -Y boundary face into neighbor brick.
                        if cby > 0 && (active_mask & FACE_Y_MIN) != 0 {
                            let nby = cby - 1;
                            let ni = c_idx - bdx;
                            let n_seed = ((active_mask & FACE_Y_MIN) << 12) & unvisited_solid[ni];
                            if n_seed != 0 {
                                let next = flood_fill_brick_u64(n_seed, unvisited_solid[ni]);
                                unvisited_solid[ni] &= !next;
                                if comp_masks[ni] == 0 {
                                    touched_bricks.push(ni);
                                }
                                comp_masks[ni] |= next;
                                queue.push((cbx, nby, cbz, next));
                            }
                        }

                        // INVARIANT: Propagate +Y boundary face into neighbor brick.
                        if cby + 1 < bdy && (active_mask & FACE_Y_MAX) != 0 {
                            let nby = cby + 1;
                            let ni = c_idx + bdx;
                            let n_seed = ((active_mask & FACE_Y_MAX) >> 12) & unvisited_solid[ni];
                            if n_seed != 0 {
                                let next = flood_fill_brick_u64(n_seed, unvisited_solid[ni]);
                                unvisited_solid[ni] &= !next;
                                if comp_masks[ni] == 0 {
                                    touched_bricks.push(ni);
                                }
                                comp_masks[ni] |= next;
                                queue.push((cbx, nby, cbz, next));
                            }
                        }

                        // INVARIANT: Propagate -Z boundary face into neighbor brick.
                        if cbz > 0 && (active_mask & FACE_Z_MIN) != 0 {
                            let nbz = cbz - 1;
                            let ni = c_idx - bdx * bdy;
                            let n_seed = ((active_mask & FACE_Z_MIN) << 48) & unvisited_solid[ni];
                            if n_seed != 0 {
                                let next = flood_fill_brick_u64(n_seed, unvisited_solid[ni]);
                                unvisited_solid[ni] &= !next;
                                if comp_masks[ni] == 0 {
                                    touched_bricks.push(ni);
                                }
                                comp_masks[ni] |= next;
                                queue.push((cbx, cby, nbz, next));
                            }
                        }

                        // INVARIANT: Propagate +Z boundary face into neighbor brick.
                        if cbz + 1 < bdz && (active_mask & FACE_Z_MAX) != 0 {
                            let nbz = cbz + 1;
                            let ni = c_idx + bdx * bdy;
                            let n_seed = ((active_mask & FACE_Z_MAX) >> 48) & unvisited_solid[ni];
                            if n_seed != 0 {
                                let next = flood_fill_brick_u64(n_seed, unvisited_solid[ni]);
                                unvisited_solid[ni] &= !next;
                                if comp_masks[ni] == 0 {
                                    touched_bricks.push(ni);
                                }
                                comp_masks[ni] |= next;
                                queue.push((cbx, cby, nbz, next));
                            }
                        }
                    }

                    // INVARIANT: Extract (lx, ly, lz, mat) tuples by iterating set bits of the component's visited bricks.
                    let mut comp = Vec::new();
                    for &b_idx in &touched_bricks {
                        let mask = comp_masks[b_idx];
                        comp_masks[b_idx] = 0;
                        let b_bx = b_idx % bdx;
                        let b_by = (b_idx / bdx) % bdy;
                        let b_bz = b_idx / (bdx * bdy);
                        let mut bits = mask;
                        while bits != 0 {
                            let bit = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            let rx = (bit & 3) as i32;
                            let ry = ((bit >> 2) & 3) as i32;
                            let rz = ((bit >> 4) & 3) as i32;
                            let orig_lx = (min_c.x + (b_bx as i32) * 4 + rx) as u32;
                            let orig_ly = (min_c.y + (b_by as i32) * 4 + ry) as u32;
                            let orig_lz = (min_c.z + (b_bz as i32) * 4 + rz) as u32;
                            let mat = brick_mats[b_idx * 64 + bit];
                            comp.push((orig_lx, orig_ly, orig_lz, mat));
                        }
                    }
                    touched_bricks.clear();
                    components.push(comp);
                }
            }

            return components;
        }

        // WHY: Fallback for sparse or huge bounding volumes exceeding dense brick budget.
        let mut map = std::collections::HashMap::with_capacity(voxels.len());
        for &(lx, ly, lz, mat) in voxels {
            map.insert(IVec3::new(lx as i32, ly as i32, lz as i32), mat);
        }

        let mut components = Vec::new();
        let mut queue = std::collections::VecDeque::new();

        let neighbors = [
            IVec3::new(-1, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(0, 0, -1),
            IVec3::new(0, 0, 1),
        ];

        while let Some((&start_coord, &start_mat)) = map.iter().next() {
            map.remove(&start_coord);
            let mut comp = Vec::new();
            comp.push((
                start_coord.x as u32,
                start_coord.y as u32,
                start_coord.z as u32,
                start_mat,
            ));
            queue.push_back(start_coord);

            while let Some(curr) = queue.pop_front() {
                for &offset in &neighbors {
                    let n = curr + offset;
                    if let Some(mat) = map.remove(&n) {
                        comp.push((n.x as u32, n.y as u32, n.z as u32, mat));
                        queue.push_back(n);
                    }
                }
            }
            components.push(comp);
        }

        components
    }

    /// Places a sphere of voxels of `mat` centered at `center_world` with `radius_meters` onto this body.
    /// Recomputes Center of Mass, moment of inertia tensor, and boundary hull.
    pub fn place_sphere_meters(
        &mut self,
        center_world: Vec3,
        radius_meters: f32,
        mat: MaterialId,
        palette: &MaterialPalette,
    ) -> bool {
        if mat == MAT_AIR {
            return false;
        }
        let inv_rot = Mat3::from_quat(self.orientation).transpose();
        let local_center_m = inv_rot * (center_world - self.position) + self.com_local_m;
        let s = VOXEL_SIZE;
        let center_v = local_center_m * (VOXELS_PER_METER as f32);
        let r_v = radius_meters * (VOXELS_PER_METER as f32);
        let r_sq_v = r_v * r_v;
        let r_voxels = r_v.ceil() as i32;

        let cx = center_v.x.floor() as i32;
        let cy = center_v.y.floor() as i32;
        let cz = center_v.z.floor() as i32;

        let mut voxel_map = std::collections::HashMap::with_capacity(self.local_voxels.len() + 64);
        for &(lx, ly, lz, m) in &self.local_voxels {
            voxel_map.insert(IVec3::new(lx as i32, ly as i32, lz as i32), m);
        }

        let mut added = false;
        for dz in -r_voxels..=(r_voxels + 1) {
            for dy in -r_voxels..=(r_voxels + 1) {
                for dx in -r_voxels..=(r_voxels + 1) {
                    let coord = IVec3::new(cx + dx, cy + dy, cz + dz);
                    let v_center = coord.as_vec3() + Vec3::splat(0.5);
                    if (v_center - center_v).length_squared() <= r_sq_v
                        && voxel_map.get(&coord) != Some(&mat)
                    {
                        voxel_map.insert(coord, mat);
                        added = true;
                    }
                }
            }
        }

        if !added {
            return false;
        }

        let mut min_coord = IVec3::splat(i32::MAX);
        let mut max_coord = IVec3::splat(i32::MIN);
        for &coord in voxel_map.keys() {
            min_coord = min_coord.min(coord);
            max_coord = max_coord.max(coord);
        }

        let dims = max_coord - min_coord + IVec3::ONE;
        if dims.x > 256 || dims.y > 256 || dims.z > 256 {
            return false;
        }

        let shift_m = min_coord.as_vec3() * s;

        let mut updated_local = Vec::with_capacity(voxel_map.len());
        for (coord, m) in voxel_map {
            let local_c = coord - min_coord;
            updated_local.push((local_c.x as u32, local_c.y as u32, local_c.z as u32, m));
        }

        self.dims_v = dims;
        self.half_extents_m = dims.as_vec3() * (0.5 * s);
        self.local_voxels = updated_local;
        self.com_local_m -= shift_m;

        self.rebuild_after_carve(palette);
        self.prev_position = self.position;
        self.prev_orientation = self.orientation;
        self.wake_up();
        self.reset_rest_timer();
        true
    }

    /// Carves an explosion sphere at `blast_center_world` with `radius_meters` from this body.
    /// If `scorch_rim` is enabled, applies amber embers (`MAT_EMBER_GLOW`) and dark rock (`MAT_SCORCHED_ROCK`)
    /// along the crater periphery. If the blast splits the body into disconnected pieces, detached islands
    /// are partitioned and returned in `outcome.split_bodies`!
    pub fn carve_sphere_meters(
        &mut self,
        blast_center_world: Vec3,
        radius_meters: f32,
        scorch_rim: bool,
        palette: &MaterialPalette,
    ) -> RigidStepOutcome {
        let mut outcome = RigidStepOutcome::default();
        let inv_rot = Mat3::from_quat(self.orientation).transpose();
        let local_center_m = inv_rot * (blast_center_world - self.position) + self.com_local_m;
        let r_sq = radius_meters * radius_meters;
        let s = VOXEL_SIZE;
        let outer_r_m = if scorch_rim {
            radius_meters + 2.5 * s
        } else {
            radius_meters
        };
        let outer_r_sq = outer_r_m * outer_r_m;
        let ember_r_sq = (radius_meters + 0.95 * s).powi(2);

        let mut remaining_voxels = Vec::with_capacity(self.local_voxels.len());
        let mut destroyed_edits = Vec::new();
        let mut scorched_updates = Vec::new();
        let mut any_destroyed = false;
        let rot = Mat3::from_quat(self.orientation);

        for &(lx, ly, lz, mat) in &self.local_voxels {
            let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
            let offset = p_local - local_center_m;
            let d_sq = offset.length_squared();

            if d_sq <= r_sq {
                // INVARIANT: Voxel destroyed within blast radius; spawn debris particle.
                outcome.modified = true;
                any_destroyed = true;
                destroyed_edits.push((lx, ly, lz, MAT_AIR));
                let p_world = self.position + rot * (p_local - self.com_local_m);
                let blast_dir = (p_world - blast_center_world).normalize_or_zero();
                let push_vel = (blast_dir * 3.5) + self.linear_velocity * 0.5;
                let vis_mat = if scorch_rim && (lx.wrapping_add(ly).wrapping_add(lz) & 7) == 0 {
                    arcadia_core::MAT_EMBER_GLOW
                } else if scorch_rim && (lx.wrapping_add(ly).wrapping_add(lz) & 3) == 0 {
                    arcadia_core::MAT_SCORCHED_ROCK
                } else {
                    mat
                };
                let packed_col = palette.get(vis_mat).pack_lod_payload(vis_mat);
                outcome.spawned_debris.push(DebrisParticle {
                    pos_life: [p_world.x, p_world.y, p_world.z, 2.5],
                    vel_color: [
                        push_vel.x,
                        push_vel.y,
                        push_vel.z,
                        f32::from_bits(packed_col),
                    ],
                });
            } else if scorch_rim && d_sq <= outer_r_sq {
                let hash = (lx.wrapping_mul(73856093)
                    ^ ly.wrapping_mul(19349663)
                    ^ lz.wrapping_mul(83492791))
                    & 0xFF;
                let scorched_mat = if d_sq <= ember_r_sq && hash < 48 {
                    arcadia_core::MAT_EMBER_GLOW
                } else if hash < 180 {
                    arcadia_core::MAT_SCORCHED_ROCK
                } else {
                    mat
                };
                if scorched_mat != mat {
                    outcome.modified = true;
                    scorched_updates.push((lx, ly, lz, scorched_mat));
                }
                remaining_voxels.push((lx, ly, lz, scorched_mat));
            } else {
                remaining_voxels.push((lx, ly, lz, mat));
            }
        }

        if !outcome.modified {
            // WHY: An unmodified body must NOT receive an artificial blast velocity impulse when carving nearby.
            let blast_dist = (self.position - blast_center_world).length();
            if blast_dist < radius_meters + 0.5 {
                self.wake_up();
            }
            return outcome;
        }

        // WHY: Fast path when only scorch rim changed and zero voxels were destroyed.
        if !any_destroyed {
            self.local_voxels = remaining_voxels;
            self.local_tree.set_voxels_local(&scorched_updates, palette);
            self.emissive_light =
                compute_emissive_light(&self.local_voxels, palette, self.com_local_m, s);
            self.wake_up();
            self.reset_rest_timer();
            self.prev_position = self.position;
            self.prev_orientation = self.orientation;
            return outcome;
        }

        if remaining_voxels.is_empty() {
            self.local_voxels.clear();
            outcome.destroyed = true;
            return outcome;
        }

        // INVARIANT: When carving severs the body into disconnected parts, largest component remains in self.
        let mut components = Self::find_connected_components(&remaining_voxels);
        if components.is_empty() {
            self.local_voxels.clear();
            outcome.destroyed = true;
            return outcome;
        }

        components.sort_by_key(|c| std::cmp::Reverse(c.len()));
        let main_voxels = components.remove(0);

        // INVARIANT: Capture parent physical state before modifying `self` so detached pieces maintain exact world alignment.
        let parent_pos = self.position;
        let parent_rot = self.orientation;
        let parent_com_local = self.com_local_m;
        let parent_lin_vel = self.linear_velocity;
        let parent_ang_vel = self.angular_velocity;

        // WHY: If surviving component has <= SMALL_ISLAND_PARTICLE_MAX_VOXELS, obliterate all voxels into debris particles.
        if main_voxels.len() <= Self::SMALL_ISLAND_PARTICLE_MAX_VOXELS {
            for &(lx, ly, lz, mat) in &main_voxels {
                let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
                let p_world = parent_pos + parent_rot * (p_local - parent_com_local);
                let blast_dir = (p_world - blast_center_world).normalize_or_zero();
                let push_vel = (blast_dir * 4.5) + parent_lin_vel * 0.7;
                let packed_col = palette.get(mat).pack_lod_payload(mat);
                outcome.spawned_debris.push(DebrisParticle {
                    pos_life: [p_world.x, p_world.y, p_world.z, 2.5],
                    vel_color: [
                        push_vel.x,
                        push_vel.y,
                        push_vel.z,
                        f32::from_bits(packed_col),
                    ],
                });
            }
            for split_comp in components {
                for &(lx, ly, lz, mat) in &split_comp {
                    let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
                    let p_world = parent_pos + parent_rot * (p_local - parent_com_local);
                    let blast_dir = (p_world - blast_center_world).normalize_or_zero();
                    let push_vel = (blast_dir * 4.5) + parent_lin_vel * 0.7;
                    let packed_col = palette.get(mat).pack_lod_payload(mat);
                    outcome.spawned_debris.push(DebrisParticle {
                        pos_life: [p_world.x, p_world.y, p_world.z, 2.5],
                        vel_color: [
                            push_vel.x,
                            push_vel.y,
                            push_vel.z,
                            f32::from_bits(packed_col),
                        ],
                    });
                }
            }
            self.local_voxels.clear();
            outcome.destroyed = true;
            return outcome;
        }

        // WHY: Detached components with <= SMALL_ISLAND_PARTICLE_MAX_VOXELS become debris particles to prevent tiny physics bodies.
        let only_single_component = components.is_empty();
        for split_comp in components {
            if split_comp.len() <= Self::SMALL_ISLAND_PARTICLE_MAX_VOXELS {
                for &(lx, ly, lz, mat) in &split_comp {
                    let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
                    let p_world = parent_pos + parent_rot * (p_local - parent_com_local);
                    let blast_dir = (p_world - blast_center_world).normalize_or_zero();
                    let push_vel = (blast_dir * 4.5) + parent_lin_vel * 0.7;
                    let packed_col = palette.get(mat).pack_lod_payload(mat);
                    outcome.spawned_debris.push(DebrisParticle {
                        pos_life: [p_world.x, p_world.y, p_world.z, 2.5],
                        vel_color: [
                            push_vel.x,
                            push_vel.y,
                            push_vel.z,
                            f32::from_bits(packed_col),
                        ],
                    });
                }
            } else if let Some(mut split_body) = Self::from_local_component(
                &split_comp,
                parent_pos,
                parent_rot,
                parent_com_local,
                parent_lin_vel,
                parent_ang_vel,
                Some(blast_center_world),
                palette,
            ) {
                split_body.wake_up();
                split_body.reset_rest_timer();
                outcome.split_bodies.push(split_body);
            }
        }

        // INVARIANT: Re-base main_voxels so minimum coordinate starts at (0,0,0) with exact world alignment.
        let mut min_c = IVec3::splat(i32::MAX);
        let mut max_c = IVec3::splat(i32::MIN);
        for &(lx, ly, lz, _) in &main_voxels {
            let c = IVec3::new(lx as i32, ly as i32, lz as i32);
            min_c = min_c.min(c);
            max_c = max_c.max(c);
        }
        let dims = max_c - min_c + IVec3::ONE;
        let mut rebased_main = Vec::with_capacity(main_voxels.len());
        for (lx, ly, lz, mat) in main_voxels {
            rebased_main.push((
                (lx as i32 - min_c.x) as u32,
                (ly as i32 - min_c.y) as u32,
                (lz as i32 - min_c.z) as u32,
                mat,
            ));
        }

        let shift_m = min_c.as_vec3() * s;
        self.com_local_m -= shift_m;
        self.dims_v = dims;
        self.half_extents_m = dims.as_vec3() * (0.5 * s);
        self.local_voxels = rebased_main;

        // WHY: In-place tree update when bounding box origin did not shift and only a single connected component remains.
        if only_single_component && min_c == IVec3::ZERO {
            let mut tree_edits = destroyed_edits;
            tree_edits.extend(scorched_updates);
            self.local_tree.set_voxels_local(&tree_edits, palette);
            self.local_tree.compact();
            self.rebuild_after_carve_preserve_tree(palette);
        } else {
            self.rebuild_after_carve(palette);
        }

        // WHY: Clamp blast impulse on carved surviving body to a micro-nudge (<= 0.1 m/s) to avoid sliding.
        let blast_offset = self.position - blast_center_world;
        let blast_dist = blast_offset.length().max(0.2);
        let mut blast_dir = blast_offset / blast_dist;
        // WHY: Clamp upward impulse component to prevent launching resting bodies skyward.
        blast_dir.y = blast_dir.y.clamp(-1.0, 0.1);
        let blast_speed = (2.0 * self.inv_mass / blast_dist.sqrt()).clamp(0.0, 0.1);
        self.linear_velocity += blast_dir * blast_speed;
        self.wake_up();
        self.reset_rest_timer();

        // WHY: Sync previous transform on newly detached split bodies without artificial kicks.
        for split_body in &mut outcome.split_bodies {
            split_body.prev_position = split_body.position;
            split_body.prev_orientation = split_body.orientation;
        }
        self.prev_position = self.position;
        self.prev_orientation = self.orientation;

        outcome
    }

    /// Converts this rigid body's active local voxels into global grid coordinates `(IVec3, MaterialId)`.
    /// Reconstructs the voxel list in global world-grid coordinates for baking into `WorldMap`
    /// at its current world-space resting pose.
    ///
    /// WHY: Forward point-sampling alone creates diagonal 1-voxel pinhole gaps when a 3D grid
    /// is rotated by an arbitrary quaternion. We apply conservative dual-pass flat-grid voxelization:
    /// 1. Forward pass: map all local voxel centroids into world coordinates.
    /// 2. Inverse-mapping pass: query world AABB voxel centers transformed back into the local
    ///    dense grid to guarantee watertight solid volumes with zero interior spongey holes.
    /// 3. Morphological 3D pinhole fill: seal any 1-voxel aliasing gaps between opposite solid neighbors.
    pub fn to_world_voxels(&self) -> Vec<(IVec3, MaterialId)> {
        if self.local_voxels.is_empty()
            || self.dims_v.x <= 0
            || self.dims_v.y <= 0
            || self.dims_v.z <= 0
        {
            return Vec::new();
        }

        let rot = Mat3::from_quat(self.orientation);
        let inv_rot = rot.transpose();
        let is_rotated = (self.orientation.x.abs() > 0.02
            || self.orientation.y.abs() > 0.02
            || self.orientation.z.abs() > 0.02)
            && (self.orientation.w.abs() < 0.999);
        let s = VOXEL_SIZE;
        let vpm = VOXELS_PER_METER as f32;

        let local_span_x = self.dims_v.x as usize;
        let local_span_y = self.dims_v.y as usize;
        let local_span_z = self.dims_v.z as usize;
        let Some(local_volume) = local_span_x
            .checked_mul(local_span_y)
            .and_then(|v| v.checked_mul(local_span_z))
        else {
            return self.to_world_voxels_hashmap();
        };

        let (aabb_min_m, aabb_max_m) = self.world_aabb_meters();
        let min_vx = (aabb_min_m.x * vpm).floor() as i32;
        let max_vx = (aabb_max_m.x * vpm).ceil() as i32;
        let min_vy = (aabb_min_m.y * vpm).floor() as i32;
        let max_vy = (aabb_max_m.y * vpm).ceil() as i32;
        let min_vz = (aabb_min_m.z * vpm).floor() as i32;
        let max_vz = (aabb_max_m.z * vpm).ceil() as i32;

        if min_vx > max_vx || min_vy > max_vy || min_vz > max_vz {
            return Vec::new();
        }

        let grid_min_x = min_vx - 2;
        let grid_max_x = max_vx + 2;
        let grid_min_y = min_vy - 2;
        let grid_max_y = max_vy + 2;
        let grid_min_z = min_vz - 2;
        let grid_max_z = max_vz + 2;

        let span_x = (grid_max_x - grid_min_x + 1) as usize;
        let span_y = (grid_max_y - grid_min_y + 1) as usize;
        let span_z = (grid_max_z - grid_min_z + 1) as usize;

        let Some(total_voxels) = span_x
            .checked_mul(span_y)
            .and_then(|v| v.checked_mul(span_z))
        else {
            return self.to_world_voxels_hashmap();
        };

        // WHY: Dense flat grid provides zero SipHash overhead; fall back to sparse map for > 2.5M voxels.
        if total_voxels > 2_500_000 || local_volume > 2_500_000 {
            return self.to_world_voxels_hashmap();
        }

        let mut local_grid = vec![MAT_AIR; local_volume];
        let stride_ly = local_span_x;
        let stride_lz = local_span_x * local_span_y;

        for &(lx, ly, lz, mat) in &self.local_voxels {
            if mat != MAT_AIR
                && (lx as i32) < self.dims_v.x
                && (ly as i32) < self.dims_v.y
                && (lz as i32) < self.dims_v.z
            {
                let idx = (lx as usize) + (ly as usize) * stride_ly + (lz as usize) * stride_lz;
                local_grid[idx] = mat;
            }
        }

        let get_local = |x: i32, y: i32, z: i32| -> MaterialId {
            if x < 0
                || x >= self.dims_v.x
                || y < 0
                || y >= self.dims_v.y
                || z < 0
                || z >= self.dims_v.z
            {
                return MAT_AIR;
            }
            let idx = (x as usize) + (y as usize) * stride_ly + (z as usize) * stride_lz;
            local_grid[idx]
        };

        let mut world_grid = vec![MAT_AIR; total_voxels];
        let stride_wy = span_x;
        let stride_wz = span_x * span_y;

        let world_idx = |wx: i32, wy: i32, wz: i32| -> usize {
            (wx - grid_min_x) as usize
                + (wy - grid_min_y) as usize * stride_wy
                + (wz - grid_min_z) as usize * stride_wz
        };

        // Pass 1: Forward mapping of local voxel centroids into world coordinates.
        for &(lx, ly, lz, mat) in &self.local_voxels {
            if mat == MAT_AIR {
                continue;
            }
            let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
            let p_world = self.position + rot * (p_local - self.com_local_m);
            let vx = (p_world.x * vpm).floor() as i32;
            let vy = (p_world.y * vpm).floor() as i32;
            let vz = (p_world.z * vpm).floor() as i32;

            if vx >= grid_min_x
                && vx <= grid_max_x
                && vy >= grid_min_y
                && vy <= grid_max_y
                && vz >= grid_min_z
                && vz <= grid_max_z
            {
                let idx = world_idx(vx, vy, vz);
                world_grid[idx] = mat;
            }
        }

        // Pass 2: Inverse conservative mapping over resting world AABB.
        for wz in min_vz..=max_vz {
            for wy in min_vy..=max_vy {
                for wx in min_vx..=max_vx {
                    let w_idx = world_idx(wx, wy, wz);
                    if world_grid[w_idx] != MAT_AIR {
                        continue;
                    }

                    let p_world = Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5) * s;
                    let p_local = inv_rot * (p_world - self.position) + self.com_local_m;
                    let u_coord = p_local * vpm;
                    let lx = u_coord.x.floor() as i32;
                    let ly = u_coord.y.floor() as i32;
                    let lz = u_coord.z.floor() as i32;

                    let local_mat = get_local(lx, ly, lz);
                    if local_mat != MAT_AIR {
                        world_grid[w_idx] = local_mat;
                    } else if is_rotated {
                        'neighbor_search: for dz in -1..=1 {
                            for dy in -1..=1 {
                                for dx in -1..=1 {
                                    if dx == 0 && dy == 0 && dz == 0 {
                                        continue;
                                    }
                                    let n_mat = get_local(lx + dx, ly + dy, lz + dz);
                                    if n_mat != MAT_AIR {
                                        let center = Vec3::new(
                                            (lx + dx) as f32 + 0.5,
                                            (ly + dy) as f32 + 0.5,
                                            (lz + dz) as f32 + 0.5,
                                        );
                                        let diff = (u_coord - center).abs();
                                        let l_inf = diff.x.max(diff.y).max(diff.z);
                                        if l_inf <= 0.58 {
                                            world_grid[w_idx] = n_mat;
                                            break 'neighbor_search;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pass 3: Morphological 3D pinhole cavity closure.
        let mut pinholes = Vec::new();
        for _ in 0..2 {
            pinholes.clear();
            for wz in min_vz..=max_vz {
                for wy in min_vy..=max_vy {
                    for wx in min_vx..=max_vx {
                        let idx = world_idx(wx, wy, wz);
                        if world_grid[idx] != MAT_AIR {
                            continue;
                        }

                        // INVARIANT: world_grid includes 2-voxel padding, guaranteeing 6-neighbor lookups stay in-bounds.
                        let x_neg = world_grid[idx - 1];
                        let x_pos = world_grid[idx + 1];
                        let y_neg = world_grid[idx - stride_wy];
                        let y_pos = world_grid[idx + stride_wy];
                        let z_neg = world_grid[idx - stride_wz];
                        let z_pos = world_grid[idx + stride_wz];

                        let neighbors = [x_neg, x_pos, y_neg, y_pos, z_neg, z_pos];
                        let face_count = neighbors.iter().filter(|&&m| m != MAT_AIR).count();

                        let opp_pair = (x_neg != MAT_AIR && x_pos != MAT_AIR)
                            || (y_neg != MAT_AIR && y_pos != MAT_AIR)
                            || (z_neg != MAT_AIR && z_pos != MAT_AIR);

                        if opp_pair || face_count >= 3 {
                            let dominant = if x_neg != MAT_AIR {
                                x_neg
                            } else if y_neg != MAT_AIR {
                                y_neg
                            } else if z_neg != MAT_AIR {
                                z_neg
                            } else if x_pos != MAT_AIR {
                                x_pos
                            } else if y_pos != MAT_AIR {
                                y_pos
                            } else {
                                z_pos
                            };

                            if dominant != MAT_AIR {
                                pinholes.push((idx, dominant));
                            }
                        }
                    }
                }
            }
            if pinholes.is_empty() {
                break;
            }
            for &(idx, mat) in &pinholes {
                world_grid[idx] = mat;
            }
        }

        // Pass 4: Emit non-air voxels sequentially.
        let mut output = Vec::with_capacity(self.local_voxels.len() * 2);
        let mut idx = 0;
        for wz in grid_min_z..=grid_max_z {
            for wy in grid_min_y..=grid_max_y {
                for wx in grid_min_x..=grid_max_x {
                    let mat = world_grid[idx];
                    if mat != MAT_AIR {
                        output.push((IVec3::new(wx, wy, wz), mat));
                    }
                    idx += 1;
                }
            }
        }

        output
    }

    // WHY: Fallback sparse hash map implementation for oversized bodies (> 2.5M voxels).
    fn to_world_voxels_hashmap(&self) -> Vec<(IVec3, MaterialId)> {
        let rot = Mat3::from_quat(self.orientation);
        let inv_rot = rot.transpose();
        let is_rotated = (self.orientation.x.abs() > 0.02
            || self.orientation.y.abs() > 0.02
            || self.orientation.z.abs() > 0.02)
            && (self.orientation.w.abs() < 0.999);
        let s = VOXEL_SIZE;
        let vpm = VOXELS_PER_METER as f32;

        let mut voxel_map = std::collections::HashMap::with_capacity(self.local_voxels.len() * 2);

        let mut local_map = std::collections::HashMap::with_capacity(self.local_voxels.len());
        for &(lx, ly, lz, mat) in &self.local_voxels {
            if mat != MAT_AIR {
                local_map.insert((lx as i32, ly as i32, lz as i32), mat);
            }
        }

        for &(lx, ly, lz, mat) in &self.local_voxels {
            if mat == MAT_AIR {
                continue;
            }
            let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
            let p_world = self.position + rot * (p_local - self.com_local_m);
            let vx = (p_world.x * vpm).floor() as i32;
            let vy = (p_world.y * vpm).floor() as i32;
            let vz = (p_world.z * vpm).floor() as i32;
            voxel_map.insert(IVec3::new(vx, vy, vz), mat);
        }

        let (aabb_min_m, aabb_max_m) = self.world_aabb_meters();
        let min_vx = (aabb_min_m.x * vpm).floor() as i32;
        let max_vx = (aabb_max_m.x * vpm).ceil() as i32;
        let min_vy = (aabb_min_m.y * vpm).floor() as i32;
        let max_vy = (aabb_max_m.y * vpm).ceil() as i32;
        let min_vz = (aabb_min_m.z * vpm).floor() as i32;
        let max_vz = (aabb_max_m.z * vpm).ceil() as i32;

        for wz in min_vz..=max_vz {
            for wy in min_vy..=max_vy {
                for wx in min_vx..=max_vx {
                    let gv = IVec3::new(wx, wy, wz);
                    if voxel_map.contains_key(&gv) {
                        continue;
                    }

                    let p_world = Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5) * s;
                    let p_local = inv_rot * (p_world - self.position) + self.com_local_m;
                    let u_coord = p_local * vpm;
                    let lx = u_coord.x.floor() as i32;
                    let ly = u_coord.y.floor() as i32;
                    let lz = u_coord.z.floor() as i32;

                    if let Some(&mat) = local_map.get(&(lx, ly, lz)) {
                        voxel_map.insert(gv, mat);
                    } else if is_rotated {
                        'neighbor_search: for dz in -1..=1 {
                            for dy in -1..=1 {
                                for dx in -1..=1 {
                                    if dx == 0 && dy == 0 && dz == 0 {
                                        continue;
                                    }
                                    let n_key = (lx + dx, ly + dy, lz + dz);
                                    if let Some(&mat) = local_map.get(&n_key) {
                                        let center = Vec3::new(
                                            n_key.0 as f32 + 0.5,
                                            n_key.1 as f32 + 0.5,
                                            n_key.2 as f32 + 0.5,
                                        );
                                        let diff = (u_coord - center).abs();
                                        let l_inf = diff.x.max(diff.y).max(diff.z);
                                        if l_inf <= 0.58 {
                                            voxel_map.insert(gv, mat);
                                            break 'neighbor_search;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for _ in 0..2 {
            let mut pinholes = Vec::new();
            for wz in min_vz..=max_vz {
                for wy in min_vy..=max_vy {
                    for wx in min_vx..=max_vx {
                        let gv = IVec3::new(wx, wy, wz);
                        if voxel_map.contains_key(&gv) {
                            continue;
                        }

                        let x_neg = voxel_map.get(&IVec3::new(wx - 1, wy, wz)).copied();
                        let x_pos = voxel_map.get(&IVec3::new(wx + 1, wy, wz)).copied();
                        let y_neg = voxel_map.get(&IVec3::new(wx, wy - 1, wz)).copied();
                        let y_pos = voxel_map.get(&IVec3::new(wx, wy + 1, wz)).copied();
                        let z_neg = voxel_map.get(&IVec3::new(wx, wy, wz - 1)).copied();
                        let z_pos = voxel_map.get(&IVec3::new(wx, wy, wz + 1)).copied();

                        let neighbors = [x_neg, x_pos, y_neg, y_pos, z_neg, z_pos];
                        let face_count = neighbors.iter().filter(|n| n.is_some()).count();

                        let opp_pair = (x_neg.is_some() && x_pos.is_some())
                            || (y_neg.is_some() && y_pos.is_some())
                            || (z_neg.is_some() && z_pos.is_some());

                        if opp_pair || face_count >= 3 {
                            if let Some(mat) =
                                x_neg.or(y_neg).or(z_neg).or(x_pos).or(y_pos).or(z_pos)
                            {
                                pinholes.push((gv, mat));
                            }
                        }
                    }
                }
            }
            if pinholes.is_empty() {
                break;
            }
            for (gv, mat) in pinholes {
                voxel_map.insert(gv, mat);
            }
        }

        voxel_map.into_iter().collect()
    }
}
