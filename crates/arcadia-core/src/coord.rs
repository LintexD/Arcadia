//! Coordinate types, scale constants, and 4x4x4 (64-tree) bitwise index packing.
//! Owns CPU coordinate types and bitwise decomposition math.

use glam::{IVec3, Vec3};

/// Physical size of a single voxel in meters: 1/16 m = 0.0625 m (6.25 cm).
pub const VOXEL_SIZE: f32 = 1.0 / 16.0;

/// Number of voxels per linear meter (16).
pub const VOXELS_PER_METER: i32 = 16;

/// Number of voxels along each axis of a single Chunk: 4^4 = 256 voxels (16.0 meters).
pub const CHUNK_DIM: u32 = 256;

/// Physical side length of one Chunk in meters (256 * 1/16m = 16.0m).
pub const CHUNK_SIZE_METERS: f32 = 16.0;

/// Global integer voxel coordinate (each unit is 1/16 meter).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VoxelCoord(pub IVec3);

impl VoxelCoord {
    #[inline(always)]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }

    /// Converts a world-space position in meters to global voxel coordinates.
    #[inline(always)]
    pub fn from_world_pos(pos: Vec3) -> Self {
        Self((pos * (VOXELS_PER_METER as f32)).floor().as_ivec3())
    }

    /// Returns the world-space minimum corner (in meters) of this voxel.
    #[inline(always)]
    pub fn to_world_min(self) -> Vec3 {
        self.0.as_vec3() * VOXEL_SIZE
    }

    /// Returns the world-space center (in meters) of this voxel.
    #[inline(always)]
    pub fn to_world_center(self) -> Vec3 {
        (self.0.as_vec3() + Vec3::splat(0.5)) * VOXEL_SIZE
    }

    /// Decomposes a global voxel coordinate into `(ChunkCoord, local_xyz in 0..256)`.
    #[inline(always)]
    pub fn to_chunk_and_local(self) -> (ChunkCoord, (u32, u32, u32)) {
        let cx = self.0.x.div_euclid(CHUNK_DIM as i32);
        let cy = self.0.y.div_euclid(CHUNK_DIM as i32);
        let cz = self.0.z.div_euclid(CHUNK_DIM as i32);

        let lx = self.0.x.rem_euclid(CHUNK_DIM as i32) as u32;
        let ly = self.0.y.rem_euclid(CHUNK_DIM as i32) as u32;
        let lz = self.0.z.rem_euclid(CHUNK_DIM as i32) as u32;

        (ChunkCoord::new(cx, cy, cz), (lx, ly, lz))
    }
}

/// Coordinates of a 16m × 16m × 16m (256³ voxel) Chunk in the world grid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ChunkCoord(pub IVec3);

impl ChunkCoord {
    #[inline(always)]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }

    /// Minimum corner of this chunk in world meters.
    #[inline(always)]
    pub fn world_min_meters(self) -> Vec3 {
        self.0.as_vec3() * CHUNK_SIZE_METERS
    }

    /// Minimum corner of this chunk in global voxel units.
    #[inline(always)]
    pub fn min_voxel_coord(self) -> VoxelCoord {
        VoxelCoord(self.0 * (CHUNK_DIM as i32))
    }
}

/// Maps local 4×4×4 child coordinates `x, y, z in 0..4` to a 6-bit index `0..64`.
///
/// Layout: `b = (z << 4) | (y << 2) | x`
#[inline(always)]
pub const fn child_bit_index(x: u32, y: u32, z: u32) -> u32 {
    debug_assert!(x < 4 && y < 4 && z < 4);
    (z << 4) | (y << 2) | x
}

/// Unpacks a 6-bit child index `b in 0..64` into `(x, y, z)` each in `0..4`.
#[inline(always)]
pub const fn unpack_child_index(b: u32) -> (u32, u32, u32) {
    debug_assert!(b < 64);
    (b & 3, (b >> 2) & 3, (b >> 4) & 3)
}

/// Returns the single-bit `u64` occupancy mask for child `(x, y, z)` in `0..4`.
#[inline(always)]
pub const fn child_bit_mask(x: u32, y: u32, z: u32) -> u64 {
    1u64 << child_bit_index(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_index_roundtrip() {
        let mut seen = 0u64;
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let idx = child_bit_index(x, y, z);
                    assert!(idx < 64);
                    assert_eq!(unpack_child_index(idx), (x, y, z));
                    seen |= child_bit_mask(x, y, z);
                }
            }
        }
        assert_eq!(seen, u64::MAX);
    }

    #[test]
    fn test_coord_decomposition() {
        let v = VoxelCoord::new(-1, 256, -257);
        let (chunk, local) = v.to_chunk_and_local();
        assert_eq!(chunk, ChunkCoord::new(-1, 1, -2));
        assert_eq!(local, (255, 0, 255));
    }
}
