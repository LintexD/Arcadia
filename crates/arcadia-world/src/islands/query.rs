//! Trait and snapshot structures for voxel queries and ground anchor validation.
//! Owns CPU chunk snapshots for asynchronous island detection.

use crate::world_map::WorldMap;
use arcadia_core::{ChunkCoord, ChunkTree64, MaterialId, VoxelCoord, CHUNK_DIM, MAT_AIR};
use glam::IVec3;

/// Queries voxel materials and 4x4x4 brick occupancy masks during island analysis.
pub trait VoxelQuery: Send + Sync {
    fn get_voxel(&self, vx: i32, vy: i32, vz: i32) -> MaterialId;
    fn get_brick_solid_mask(&self, gbx: i32, gby: i32, gbz: i32) -> u64;
}

/// Thread-safe snapshot of chunk trees intersecting a bounding box for background island analysis.
#[derive(Clone, Debug)]
pub struct WorldRegionSnapshot {
    pub min_chunk: IVec3,
    pub chunk_dims: IVec3,
    pub chunks: Vec<Option<ChunkTree64>>,
}

/// Determines whether an external boundary voxel is anchored down to bedrock.
#[inline]
pub fn is_externally_anchored<W: VoxelQuery + ?Sized>(
    world: &W,
    vx: i32,
    vy: i32,
    vz: i32,
) -> bool {
    if vy <= 4 {
        return true;
    }
    if world.get_voxel(vx, vy, vz) == MAT_AIR {
        return false;
    }
    let mut y = vy - 1;
    let mut gap_count = 0;
    while y > 4 {
        if world.get_voxel(vx, y, vz) != MAT_AIR {
            gap_count = 0;
        } else {
            gap_count += 1;
            if gap_count > 4 {
                return false;
            }
        }
        y -= 1;
    }
    true
}

impl VoxelQuery for WorldMap {
    #[inline(always)]
    fn get_voxel(&self, vx: i32, vy: i32, vz: i32) -> MaterialId {
        self.get_voxel(vx, vy, vz)
    }

    #[inline(always)]
    fn get_brick_solid_mask(&self, gbx: i32, gby: i32, gbz: i32) -> u64 {
        self.get_brick_solid_mask(gbx, gby, gbz)
    }
}

impl WorldRegionSnapshot {
    /// Captures a thread-safe snapshot of all chunk trees intersecting `[min_v ..= max_v]`.
    pub fn capture(world: &WorldMap, min_v: IVec3, max_v: IVec3) -> Self {
        // WHY: Expand captured chunk bounds by 1 voxel on each side so boundary neighbor queries succeed.
        let c_min = (min_v - IVec3::ONE).div_euclid(IVec3::splat(CHUNK_DIM as i32));
        let c_max = (max_v + IVec3::ONE).div_euclid(IVec3::splat(CHUNK_DIM as i32));
        let chunk_dims = c_max - c_min + IVec3::ONE;
        let total_chunks = (chunk_dims.x * chunk_dims.y * chunk_dims.z) as usize;
        let mut chunks = Vec::with_capacity(total_chunks);

        for cz in 0..chunk_dims.z {
            for cy in 0..chunk_dims.y {
                for cx in 0..chunk_dims.x {
                    let gcx = c_min.x + cx;
                    let gcy = c_min.y + cy;
                    let gcz = c_min.z + cz;
                    if let Some(idx) = world.chunk_index(gcx, gcy, gcz) {
                        chunks.push(Some(world.chunks[idx].clone()));
                    } else {
                        chunks.push(None);
                    }
                }
            }
        }

        Self {
            min_chunk: c_min,
            chunk_dims,
            chunks,
        }
    }

    #[inline]
    fn chunk_index(&self, cx: i32, cy: i32, cz: i32) -> Option<usize> {
        let lx = cx - self.min_chunk.x;
        let ly = cy - self.min_chunk.y;
        let lz = cz - self.min_chunk.z;
        if lx >= 0
            && lx < self.chunk_dims.x
            && ly >= 0
            && ly < self.chunk_dims.y
            && lz >= 0
            && lz < self.chunk_dims.z
        {
            Some(
                (lx + ly * self.chunk_dims.x + lz * self.chunk_dims.x * self.chunk_dims.y) as usize,
            )
        } else {
            None
        }
    }
}

impl VoxelQuery for WorldRegionSnapshot {
    #[inline]
    fn get_voxel(&self, vx: i32, vy: i32, vz: i32) -> MaterialId {
        let (ChunkCoord(c), (lx, ly, lz)) = VoxelCoord::new(vx, vy, vz).to_chunk_and_local();
        let Some(idx) = self.chunk_index(c.x, c.y, c.z) else {
            return MAT_AIR;
        };
        if let Some(chunk) = &self.chunks[idx] {
            chunk.get_voxel(lx, ly, lz)
        } else {
            MAT_AIR
        }
    }

    #[inline]
    fn get_brick_solid_mask(&self, gbx: i32, gby: i32, gbz: i32) -> u64 {
        let cx = gbx.div_euclid(64);
        let cy = gby.div_euclid(64);
        let cz = gbz.div_euclid(64);
        let Some(idx) = self.chunk_index(cx, cy, cz) else {
            return 0;
        };
        if let Some(chunk) = &self.chunks[idx] {
            let lbx = gbx.rem_euclid(64) as u32;
            let lby = gby.rem_euclid(64) as u32;
            let lbz = gbz.rem_euclid(64) as u32;
            chunk.get_brick_solid_mask(lbx, lby, lbz)
        } else {
            0
        }
    }
}
