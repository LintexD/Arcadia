//! Spherical and cuboid voxel placement for `ChunkTree64`.
//! Owns CPU additive material placement and brush geometry.

use glam::Vec3;

use crate::chunk_tree::edit::ChunkEditReport;
use crate::chunk_tree::ChunkTree64;
use crate::coord::CHUNK_DIM;
use crate::material::{MaterialId, MaterialPalette, MAT_AIR};

impl ChunkTree64 {
    /// Places a sphere of solid material `fill_mat` in chunk-local coordinates `(0.0..256.0)`.
    pub fn place_sphere(
        &mut self,
        local_center: Vec3,
        radius_voxels: f32,
        fill_mat: MaterialId,
        palette: &MaterialPalette,
    ) {
        self.place_sphere_tracked(local_center, radius_voxels, fill_mat, palette);
    }

    /// Places a sphere of solid material `fill_mat` in chunk-local coordinates `(0.0..256.0)`
    /// in-place without re-packing, returning a fine-grained [`ChunkEditReport`].
    pub fn place_sphere_tracked(
        &mut self,
        local_center: Vec3,
        radius_voxels: f32,
        fill_mat: MaterialId,
        palette: &MaterialPalette,
    ) -> ChunkEditReport {
        if fill_mat == MAT_AIR || radius_voxels <= 0.0 {
            return ChunkEditReport::empty();
        }

        let bmin = Vec3::ZERO;
        let bmax = Vec3::splat(CHUNK_DIM as f32);
        let closest = local_center.clamp(bmin, bmax);
        if local_center.distance_squared(closest) > radius_voxels * radius_voxels {
            return ChunkEditReport::empty();
        }

        let s_min = (local_center - Vec3::splat(radius_voxels))
            .floor()
            .max(Vec3::ZERO);
        let s_max = (local_center + Vec3::splat(radius_voxels))
            .ceil()
            .min(Vec3::splat((CHUNK_DIM - 1) as f32));

        if s_min.x > s_max.x || s_min.y > s_max.y || s_min.z > s_max.z {
            return ChunkEditReport::empty();
        }

        let min_x = s_min.x as u32;
        let min_y = s_min.y as u32;
        let min_z = s_min.z as u32;
        let max_x = s_max.x as u32;
        let max_y = s_max.y as u32;
        let max_z = s_max.z as u32;

        let r_sq = radius_voxels * radius_voxels;
        let mut voxels_to_set = Vec::new();

        for vz in min_z..=max_z {
            for vy in min_y..=max_y {
                for vx in min_x..=max_x {
                    let vc = Vec3::new(vx as f32 + 0.5, vy as f32 + 0.5, vz as f32 + 0.5);
                    if local_center.distance_squared(vc) <= r_sq {
                        voxels_to_set.push((vx, vy, vz, fill_mat));
                    }
                }
            }
        }

        if voxels_to_set.is_empty() {
            return ChunkEditReport::empty();
        }

        self.set_voxels_local_tracked(&voxels_to_set, palette)
    }

    /// Places an axis-aligned cuboid of solid material `fill_mat` in chunk-local coordinates in-place.
    pub fn place_cuboid(
        &mut self,
        min: (u32, u32, u32),
        max: (u32, u32, u32),
        fill_mat: MaterialId,
        palette: &MaterialPalette,
    ) {
        self.place_cuboid_tracked(min, max, fill_mat, palette);
    }

    /// Places an axis-aligned cuboid of solid material `fill_mat` in chunk-local coordinates in-place,
    /// returning a fine-grained [`ChunkEditReport`].
    pub fn place_cuboid_tracked(
        &mut self,
        min: (u32, u32, u32),
        max: (u32, u32, u32),
        fill_mat: MaterialId,
        palette: &MaterialPalette,
    ) -> ChunkEditReport {
        if fill_mat == MAT_AIR {
            return ChunkEditReport::empty();
        }

        let min_x = min.0.min(CHUNK_DIM - 1);
        let min_y = min.1.min(CHUNK_DIM - 1);
        let min_z = min.2.min(CHUNK_DIM - 1);
        let max_x = max.0.min(CHUNK_DIM - 1);
        let max_y = max.1.min(CHUNK_DIM - 1);
        let max_z = max.2.min(CHUNK_DIM - 1);

        if min_x > max_x || min_y > max_y || min_z > max_z {
            return ChunkEditReport::empty();
        }

        let count = ((max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1)) as usize;
        let mut voxels_to_set = Vec::with_capacity(count);

        for vz in min_z..=max_z {
            for vy in min_y..=max_y {
                for vx in min_x..=max_x {
                    voxels_to_set.push((vx, vy, vz, fill_mat));
                }
            }
        }

        if voxels_to_set.is_empty() {
            return ChunkEditReport::empty();
        }

        self.set_voxels_local_tracked(&voxels_to_set, palette)
    }
}
