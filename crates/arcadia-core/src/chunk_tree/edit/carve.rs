//! Spherical voxel carving, scorch rims, and debris reporting for `ChunkTree64`.
//! Owns CPU procedural destruction and explosive subtraction operations.

use glam::Vec3;

use crate::chunk_tree::edit::ChunkEditReport;
use crate::chunk_tree::ChunkTree64;
use crate::coord::CHUNK_DIM;
use crate::material::{
    MaterialId, MaterialPalette, MAT_AIR, MAT_EMBER_GLOW, MAT_SCORCHED_ROCK, MAT_WATER_CRYSTAL,
};

impl ChunkTree64 {
    /// Carves a destructive sphere in chunk-local voxel coordinates (`center` in `0.0..256.0`, `radius` in voxels).
    ///
    /// - Voxels inside `radius` are cleared to `MAT_AIR` (and reported to `on_destroyed_voxel`).
    /// - Voxels in the outer shell `[radius .. radius + scorch_thickness]` may turn into `MAT_SCORCHED_ROCK` or `MAT_EMBER_GLOW`.
    /// - Returns the number of destroyed solid voxels.
    pub fn carve_sphere(
        &mut self,
        local_center: Vec3,
        radius_voxels: f32,
        scorch_rim: bool,
        palette: &MaterialPalette,
        on_destroyed_voxel: impl FnMut(u32, u32, u32, MaterialId),
    ) -> u32 {
        self.carve_sphere_tracked(
            local_center,
            radius_voxels,
            scorch_rim,
            palette,
            on_destroyed_voxel,
        )
        .0
    }

    /// Carves a destructive sphere in chunk-local voxel coordinates, returning both the destroyed
    /// voxel count and a fine-grained [`ChunkEditReport`] recording modified node and brick index ranges.
    pub fn carve_sphere_tracked(
        &mut self,
        local_center: Vec3,
        radius_voxels: f32,
        scorch_rim: bool,
        palette: &MaterialPalette,
        mut on_destroyed_voxel: impl FnMut(u32, u32, u32, MaterialId),
    ) -> (u32, ChunkEditReport) {
        if self.is_empty() {
            return (0, ChunkEditReport::empty());
        }

        let outer_radius = if scorch_rim {
            radius_voxels + 2.2
        } else {
            radius_voxels
        };

        // WHY: Rejects spheres outside chunk boundaries before allocating or traversing.
        let bmin = Vec3::ZERO;
        let bmax = Vec3::splat(CHUNK_DIM as f32);
        let closest = local_center.clamp(bmin, bmax);
        if local_center.distance_squared(closest) > outer_radius * outer_radius {
            return (0, ChunkEditReport::empty());
        }

        // WHY: Early-exit if bounding box contains no solid voxels in local AABB.
        let s_min = (local_center - Vec3::splat(outer_radius))
            .floor()
            .max(Vec3::ZERO);
        let s_max = (local_center + Vec3::splat(outer_radius))
            .ceil()
            .min(Vec3::splat((CHUNK_DIM - 1) as f32));

        if s_min.x > s_max.x || s_min.y > s_max.y || s_min.z > s_max.z {
            return (0, ChunkEditReport::empty());
        }

        let min_x = s_min.x as u32;
        let min_y = s_min.y as u32;
        let min_z = s_min.z as u32;
        let max_x = s_max.x as u32;
        let max_y = s_max.y as u32;
        let max_z = s_max.z as u32;

        if !self.intersects_aabb_local(min_x, min_y, min_z, max_x, max_y, max_z) {
            return (0, ChunkEditReport::empty());
        }

        let inner_r_sq = radius_voxels * radius_voxels;
        let outer_r_sq = outer_radius * outer_radius;
        let ember_r_sq = (radius_voxels + 0.95).powi(2);

        let min_bx = min_x >> 2;
        let max_bx = max_x >> 2;
        let min_by = min_y >> 2;
        let max_by = max_y >> 2;
        let min_bz = min_z >> 2;
        let max_bz = max_z >> 2;

        let mut report = ChunkEditReport::empty();
        let mut brick_entries: Vec<(u32, MaterialId)> = Vec::with_capacity(64);
        let mut destroyed = 0u32;

        for bz in min_bz..=max_bz {
            for by in min_by..=max_by {
                for bx in min_bx..=max_bx {
                    // WHY: Rejects bricks whose 4x4x4 bounding box does not intersect carve sphere outer radius.
                    let brick_min = Vec3::new((bx * 4) as f32, (by * 4) as f32, (bz * 4) as f32);
                    let brick_max = brick_min + Vec3::splat(4.0);
                    let closest = local_center.clamp(brick_min, brick_max);
                    if local_center.distance_squared(closest) > outer_r_sq {
                        continue;
                    }

                    // WHY: Skips empty 4x4x4 bricks in 1 lookup without examining individual voxels.
                    let solid_mask = self.get_brick_solid_mask(bx, by, bz);
                    if solid_mask == 0 {
                        continue;
                    }

                    let mut rem = solid_mask;
                    while rem != 0 {
                        let b = rem.trailing_zeros();
                        rem &= rem - 1;

                        let vx = (bx << 2) | (b & 3);
                        let vy = (by << 2) | ((b >> 2) & 3);
                        let vz = (bz << 2) | ((b >> 4) & 3);

                        let v_center = Vec3::new(vx as f32 + 0.5, vy as f32 + 0.5, vz as f32 + 0.5);
                        let d2 = local_center.distance_squared(v_center);

                        if d2 <= inner_r_sq {
                            let old_mat = self.get_voxel(vx, vy, vz);
                            if old_mat != MAT_AIR {
                                destroyed += 1;
                                on_destroyed_voxel(vx, vy, vz, old_mat);
                                brick_entries.push((b, MAT_AIR));
                            }
                        } else if scorch_rim && d2 <= outer_r_sq {
                            let old_mat = self.get_voxel(vx, vy, vz);
                            if old_mat != MAT_AIR && old_mat != MAT_WATER_CRYSTAL {
                                let hash = ((vx.wrapping_mul(73856093))
                                    ^ (vy.wrapping_mul(19349663))
                                    ^ (vz.wrapping_mul(83492791)))
                                    & 0xFF;
                                if d2 <= ember_r_sq && hash < 48 {
                                    brick_entries.push((b, MAT_EMBER_GLOW));
                                } else if hash < 180 {
                                    brick_entries.push((b, MAT_SCORCHED_ROCK));
                                }
                            }
                        }
                    }

                    if !brick_entries.is_empty() {
                        self.mutate_brick_in_place(
                            bx,
                            by,
                            bz,
                            &brick_entries,
                            palette,
                            &mut report,
                        );
                        brick_entries.clear();
                    }
                }
            }
        }

        if self.solid_voxel_count == 0 {
            self.nodes[0] = crate::node64::Node64::empty();
            report.mark_node_dirty(0);
        }

        (destroyed, report)
    }
}
