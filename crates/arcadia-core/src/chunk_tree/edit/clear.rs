//! Batch clearing of chunk-local voxel coordinates for `ChunkTree64`.
//! Owns CPU localized deletion, brick grouping, and in-place tree mutation.

use crate::chunk_tree::edit::ChunkEditReport;
use crate::chunk_tree::ChunkTree64;
use crate::coord::CHUNK_DIM;
use crate::material::{MaterialPalette, MAT_AIR};

impl ChunkTree64 {
    /// Removes a specific set of chunk-local voxel coordinates (`(lx, ly, lz)` in `0..256`)
    /// from the `ChunkTree64`, setting them to `MAT_AIR` in-place without re-packing.
    ///
    /// Expects `local_voxels` to be non-empty. Returns the number of solid voxels removed.
    pub fn clear_voxels_local(
        &mut self,
        local_voxels: &[(u32, u32, u32)],
        palette: &MaterialPalette,
    ) -> u32 {
        self.clear_voxels_local_tracked(local_voxels, palette).0
    }

    /// Clears a specific set of chunk-local voxel coordinates in-place without re-packing,
    /// returning the number of removed solid voxels and a fine-grained [`ChunkEditReport`].
    pub fn clear_voxels_local_tracked(
        &mut self,
        local_voxels: &[(u32, u32, u32)],
        palette: &MaterialPalette,
    ) -> (u32, ChunkEditReport) {
        if self.is_empty() || local_voxels.is_empty() {
            return (0, ChunkEditReport::empty());
        }

        let mut voxels_to_set = Vec::with_capacity(local_voxels.len());
        let mut removed = 0u32;

        for &(vx, vy, vz) in local_voxels {
            if vx >= CHUNK_DIM || vy >= CHUNK_DIM || vz >= CHUNK_DIM {
                continue;
            }
            let old_mat = self.get_voxel(vx, vy, vz);
            if old_mat != MAT_AIR {
                removed += 1;
                voxels_to_set.push((vx, vy, vz, MAT_AIR));
            }
        }

        if voxels_to_set.is_empty() {
            return (0, ChunkEditReport::empty());
        }

        let report = self.set_voxels_local_tracked(&voxels_to_set, palette);
        (removed, report)
    }

    /// Clears an axis-aligned bounding box of chunk-local voxel coordinates in-place,
    /// returning the number of removed solid voxels.
    pub fn clear_aabb(
        &mut self,
        min: (u32, u32, u32),
        max: (u32, u32, u32),
        palette: &MaterialPalette,
    ) -> u32 {
        self.clear_aabb_tracked(min, max, palette).0
    }

    /// Clears an axis-aligned bounding box of chunk-local voxel coordinates in-place,
    /// returning the number of removed solid voxels and a fine-grained [`ChunkEditReport`].
    pub fn clear_aabb_tracked(
        &mut self,
        min: (u32, u32, u32),
        max: (u32, u32, u32),
        palette: &MaterialPalette,
    ) -> (u32, ChunkEditReport) {
        if self.is_empty() {
            return (0, ChunkEditReport::empty());
        }

        let min_x = min.0.min(CHUNK_DIM - 1);
        let min_y = min.1.min(CHUNK_DIM - 1);
        let min_z = min.2.min(CHUNK_DIM - 1);
        let max_x = max.0.min(CHUNK_DIM - 1);
        let max_y = max.1.min(CHUNK_DIM - 1);
        let max_z = max.2.min(CHUNK_DIM - 1);

        if min_x > max_x || min_y > max_y || min_z > max_z {
            return (0, ChunkEditReport::empty());
        }

        if !self.intersects_aabb_local(min_x, min_y, min_z, max_x, max_y, max_z) {
            return (0, ChunkEditReport::empty());
        }

        let count = ((max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1)) as usize;
        let mut voxels_to_set = Vec::with_capacity(count);
        let mut removed = 0u32;

        for vz in min_z..=max_z {
            for vy in min_y..=max_y {
                for vx in min_x..=max_x {
                    let old_mat = self.get_voxel(vx, vy, vz);
                    if old_mat != MAT_AIR {
                        removed += 1;
                        voxels_to_set.push((vx, vy, vz, MAT_AIR));
                    }
                }
            }
        }

        if voxels_to_set.is_empty() {
            return (0, ChunkEditReport::empty());
        }

        let report = self.set_voxels_local_tracked(&voxels_to_set, palette);
        (removed, report)
    }
}
