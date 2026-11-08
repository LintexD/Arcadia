//! Real-time world destruction (explosive sphere carving, precision beam carving,
//! voxel placement) and GPU voxel debris particle emission.
//! Owns CPU voxel destruction carving and debris particle emission state.

use crate::world_map::WorldMap;
use arcadia_core::{
    MaterialId, CHUNK_DIM, MAT_AIR, MAT_EMBER_GLOW, MAT_SCORCHED_ROCK, VOXELS_PER_METER, VOXEL_SIZE,
};
use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3};

/// Maximum active GPU-simulated flying 6.25cm voxel debris cubes.
pub const MAX_DEBRIS_PARTICLES: usize = 16_384;

/// GPU-aligned flying voxel debris particle (32 bytes, matches WGSL `DebrisParticle` struct).
// GPU-SYNC: Mirrors `DebrisParticle` in `shaders/common/types.wgsl` (@group(0) @binding(5), 32 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DebrisParticle {
    /// `xyz` = world position in meters, `w` = remaining lifetime in seconds (`<= 0.0` means inactive).
    pub pos_life: [f32; 4],
    /// `xyz` = velocity in m/s, `w` = bitcast `u32` packed RGB8 color + material ID.
    pub vel_color: [f32; 4],
}

/// Summary of a world destruction or sculpting operation.
pub struct DestructionEvent {
    pub destroyed_voxels: u32,
    pub spawned_debris: Vec<DebrisParticle>,
    /// Detached structural islands (`> 8` voxels each) extracted from around the blast crater
    /// and cleared from the static `WorldMap` so they can be instantiated as dynamic `RigidVoxelBody`s.
    pub detached_islands: Vec<Vec<(IVec3, MaterialId)>>,
}

impl WorldMap {
    /// Carves an explosion or sculpting sphere at `center_meters` with `radius_meters`.
    ///
    /// - Updates all overlapped `ChunkTree64`s in place.
    /// - Queues incremental `GpuPatch` entries for sub-millisecond GPU upload.
    /// - Emits physicalized 6.25cm `DebrisParticle` cubes with outward blast velocities.
    ///
    /// Carves the blast crater sphere in chunks and emits debris particles, without running
    /// the full connected-component island detection pass.
    ///
    /// Returns `(destroyed_voxel_count, spawned_debris, Option<(island_min_v, island_max_v)>)`.
    pub fn carve_crater(
        &mut self,
        center_meters: Vec3,
        radius_meters: f32,
        scorch_rim: bool,
        max_debris_spawn: usize,
        max_probe_depth_voxels: i32,
    ) -> (u32, Vec<DebrisParticle>, Option<(IVec3, IVec3)>) {
        let center_v = center_meters * (VOXELS_PER_METER as f32);
        let radius_v = radius_meters * (VOXELS_PER_METER as f32);
        let outer_r_v = if scorch_rim { radius_v + 2.5 } else { radius_v };

        let min_v = (center_v - Vec3::splat(outer_r_v)).floor().as_ivec3();
        let max_v = (center_v + Vec3::splat(outer_r_v)).ceil().as_ivec3();

        let c_min = min_v.div_euclid(IVec3::splat(CHUNK_DIM as i32));
        let c_max = max_v.div_euclid(IVec3::splat(CHUNK_DIM as i32));

        let mut total_destroyed = 0u32;
        let mut debris = Vec::with_capacity(max_debris_spawn.min(1024));
        let palette = self.palette;

        let mut modified_chunks = Vec::new();

        for cz in c_min.z..=c_max.z {
            for cy in c_min.y..=c_max.y {
                for cx in c_min.x..=c_max.x {
                    let Some(c_idx) = self.chunk_index(cx, cy, cz) else {
                        continue;
                    };
                    if self.chunks[c_idx].is_empty() {
                        continue;
                    }

                    let chunk_origin_v = IVec3::new(cx, cy, cz) * (CHUNK_DIM as i32);
                    let local_center = center_v - chunk_origin_v.as_vec3();

                    let (destroyed, report) = self.chunks[c_idx].carve_sphere_tracked(
                        local_center,
                        radius_v,
                        scorch_rim,
                        &palette,
                        |lx, ly, lz, mat_id| {
                            if mat_id == MAT_AIR || debris.len() >= max_debris_spawn {
                                return;
                            }
                            let gx = (chunk_origin_v.x + lx as i32) as u32;
                            let gy = (chunk_origin_v.y + ly as i32) as u32;
                            let gz = (chunk_origin_v.z + lz as i32) as u32;
                            let hash = gx.wrapping_mul(73856093)
                                ^ gy.wrapping_mul(19349663)
                                ^ gz.wrapping_mul(83492791);

                            // INVARIANT: Subsample destroyed voxels so large explosions spawn a representative cloud.
                            let stride = ((radius_v * radius_v * 0.12) as u32).max(1);
                            if !hash.is_multiple_of(stride) {
                                return;
                            }

                            let world_pos = Vec3::new(
                                (chunk_origin_v.x + lx as i32) as f32 + 0.5,
                                (chunk_origin_v.y + ly as i32) as f32 + 0.5,
                                (chunk_origin_v.z + lz as i32) as f32 + 0.5,
                            ) * VOXEL_SIZE;

                            let offset = world_pos - center_meters;
                            let dist = offset.length().max(0.05);
                            let dir = offset / dist;

                            let rand_x = (((hash >> 8) & 0xFF) as f32 / 127.5) - 1.0;
                            let rand_y = (((hash >> 16) & 0xFF) as f32 / 255.0) * 1.4;
                            let rand_z = (((hash >> 24) & 0xFF) as f32 / 127.5) - 1.0;

                            let speed = if scorch_rim { 7.5 } else { 3.5 };
                            let vel =
                                (dir + Vec3::new(rand_x * 0.4, 0.7 + rand_y, rand_z * 0.4)) * speed;

                            let vis_mat = if scorch_rim && (hash & 7) == 0 {
                                MAT_EMBER_GLOW
                            } else if scorch_rim && (hash & 3) == 0 {
                                MAT_SCORCHED_ROCK
                            } else {
                                mat_id
                            };
                            let packed_col = palette.get(vis_mat).pack_lod_payload(vis_mat);

                            debris.push(DebrisParticle {
                                pos_life: [
                                    world_pos.x,
                                    world_pos.y,
                                    world_pos.z,
                                    2.5 + (hash & 0xFF) as f32 / 128.0,
                                ],
                                vel_color: [vel.x, vel.y, vel.z, f32::from_bits(packed_col)],
                            });
                        },
                    );

                    if destroyed > 0 || scorch_rim {
                        total_destroyed += destroyed;
                        modified_chunks.push((c_idx, report));
                    }
                }
            }
        }

        for (c_idx, report) in modified_chunks {
            self.sync_dirty_chunk_partial(c_idx, &report);
        }

        let bbox = if total_destroyed > 0 && max_probe_depth_voxels > 0 {
            let world_max = self.world_size_voxels() - IVec3::ONE;
            // INVARIANT: Scale search pad smoothly with user depth slider (4.0m .. 8.0m radius = 8m .. 16m span)
            // to fully enclose player-built arches and skybridges.
            let search_pad_xz =
                (outer_r_v.ceil() as i32 + (max_probe_depth_voxels / 2)).clamp(64, 128);

            // INVARIANT: Extend upward search height by max_probe_depth_voxels directly above crater
            // so tall pillars and arches are completely captured without vertical slicing.
            let max_y_extent = max_probe_depth_voxels.clamp(64, 240);

            let island_min_v = IVec3::new(
                (center_v.x as i32 - search_pad_xz).max(0),
                0,
                (center_v.z as i32 - search_pad_xz).max(0),
            );
            let island_max_v = IVec3::new(
                (center_v.x as i32 + search_pad_xz).min(world_max.x),
                (center_v.y as i32 + outer_r_v.ceil() as i32 + max_y_extent).min(world_max.y),
                (center_v.z as i32 + search_pad_xz).min(world_max.z),
            );

            Some((island_min_v, island_max_v))
        } else {
            None
        };

        (total_destroyed, debris, bbox)
    }

    /// Carves an explosion or sculpting sphere at `center_meters` with `radius_meters`.
    ///
    /// - Updates all overlapped `ChunkTree64`s in place.
    /// - Queues incremental `GpuPatch` entries for sub-millisecond GPU upload.
    /// - Emits physicalized 6.25cm `DebrisParticle` cubes with outward blast velocities.
    /// - Detects detached structural islands (up to 16,384 voxels within 15m) and detaches them from the static world.
    pub fn carve_sphere_meters(
        &mut self,
        center_meters: Vec3,
        radius_meters: f32,
        scorch_rim: bool,
        max_debris_spawn: usize,
    ) -> DestructionEvent {
        self.carve_sphere_meters_with_limits(
            center_meters,
            radius_meters,
            scorch_rim,
            max_debris_spawn,
            262_144,
            15.0,
            65_536,
        )
    }

    /// Carves an explosion or sculpting sphere with custom island voxel count, probe depth, and surface check limits.
    // WHY: Preserves non-breaking public API compatibility.
    #[allow(clippy::too_many_arguments)]
    pub fn carve_sphere_meters_with_limits(
        &mut self,
        center_meters: Vec3,
        radius_meters: f32,
        scorch_rim: bool,
        max_debris_spawn: usize,
        max_island_voxels: usize,
        max_island_depth_m: f32,
        max_surface_voxels: usize,
    ) -> DestructionEvent {
        let (mut total_destroyed, mut debris, _bbox) = self.carve_crater(
            center_meters,
            radius_meters,
            scorch_rim,
            max_debris_spawn,
            240,
        );

        let center_v = center_meters * (VOXELS_PER_METER as f32);
        let radius_v = radius_meters * (VOXELS_PER_METER as f32);

        let max_depth_v = max_island_depth_m * (VOXELS_PER_METER as f32);
        let raw_components = if max_island_voxels > 0 && max_island_depth_m > 0.0 {
            crate::islands::detect_disconnected_islands_from_carve_bounded(
                self,
                center_v,
                radius_v,
                max_island_voxels,
                max_depth_v,
                max_surface_voxels,
            )
        } else {
            // INVARIANT: Clean up isolated 1-8 voxel crumbs right on the carved crater rim even if structural island detachment is disabled.
            crate::islands::detect_disconnected_islands_from_carve_bounded(
                self, center_v, radius_v, 8, 4.0, 8,
            )
        };

        let mut detached_islands = Vec::new();
        if !raw_components.is_empty() {
            let mut all_coords_to_clear = Vec::new();
            for comp in raw_components {
                for &(gv, _) in &comp {
                    all_coords_to_clear.push(gv);
                }
                if comp.len() <= 8 {
                    for &(gv, mat) in &comp {
                        if debris.len() >= MAX_DEBRIS_PARTICLES {
                            break;
                        }
                        let pos_m = (gv.as_vec3() + Vec3::splat(0.5)) * VOXEL_SIZE;
                        let offset = pos_m - center_meters;
                        let dist = offset.length().max(0.05);
                        let dir = offset / dist;

                        let gx = gv.x as u32;
                        let gy = gv.y as u32;
                        let gz = gv.z as u32;
                        let hash = gx.wrapping_mul(73856093)
                            ^ gy.wrapping_mul(19349663)
                            ^ gz.wrapping_mul(83492791);

                        let rand_x = (((hash >> 8) & 0xFF) as f32 / 127.5) - 1.0;
                        let rand_y = (((hash >> 16) & 0xFF) as f32 / 255.0) * 1.4;
                        let rand_z = (((hash >> 24) & 0xFF) as f32 / 127.5) - 1.0;

                        let speed = if scorch_rim { 6.0 } else { 3.0 };
                        let vel =
                            (dir + Vec3::new(rand_x * 0.4, 0.5 + rand_y, rand_z * 0.4)) * speed;

                        let lifetime = 1.2 + ((hash & 0xFF) as f32 / 255.0) * 1.2;
                        let packed_col = self.palette.get(mat).pack_lod_payload(mat);

                        debris.push(DebrisParticle {
                            pos_life: [pos_m.x, pos_m.y, pos_m.z, lifetime],
                            vel_color: [vel.x, vel.y, vel.z, f32::from_bits(packed_col)],
                        });
                    }
                } else {
                    detached_islands.push(comp);
                }
            }

            // INVARIANT: Clear all detached voxels from static world and queue incremental GPU dirty patches.
            let cleared = self.clear_voxels_global(&all_coords_to_clear);
            total_destroyed += cleared;
        }

        DestructionEvent {
            destroyed_voxels: total_destroyed,
            spawned_debris: debris,
            detached_islands,
        }
    }

    /// Places a sphere of `fill_mat` at `center_meters` with `radius_meters`.
    pub fn place_sphere_meters(
        &mut self,
        center_meters: Vec3,
        radius_meters: f32,
        fill_mat: MaterialId,
    ) {
        let center_v = center_meters * (VOXELS_PER_METER as f32);
        let radius_v = radius_meters * (VOXELS_PER_METER as f32);

        let min_v = (center_v - Vec3::splat(radius_v)).floor().as_ivec3();
        let max_v = (center_v + Vec3::splat(radius_v)).ceil().as_ivec3();

        let c_min = min_v.div_euclid(IVec3::splat(CHUNK_DIM as i32));
        let c_max = max_v.div_euclid(IVec3::splat(CHUNK_DIM as i32));

        let palette = self.palette;
        let mut modified_chunks = Vec::new();

        for cz in c_min.z..=c_max.z {
            for cy in c_min.y..=c_max.y {
                for cx in c_min.x..=c_max.x {
                    let Some(c_idx) = self.chunk_index(cx, cy, cz) else {
                        continue;
                    };
                    let chunk_origin_v = IVec3::new(cx, cy, cz) * (CHUNK_DIM as i32);
                    let local_center = center_v - chunk_origin_v.as_vec3();
                    let report = self.chunks[c_idx].place_sphere_tracked(
                        local_center,
                        radius_v,
                        fill_mat,
                        &palette,
                    );
                    if report.has_dirty_nodes()
                        || report.has_dirty_bricks()
                        || report.had_structural_growth
                    {
                        modified_chunks.push((c_idx, report));
                    }
                }
            }
        }

        for (c_idx, report) in modified_chunks {
            self.sync_dirty_chunk_partial(c_idx, &report);
        }
    }
}
