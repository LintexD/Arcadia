//! `ChunkTree64` data structure, level scaling constants, and direct bitwise lookup queries.
//! Owns CPU memory buffers for 64-tree nodes, leaf bricks, and spatial occupancy queries.

use crate::coord::{child_bit_index, unpack_child_index, CHUNK_DIM};
use crate::material::{MaterialId, MAT_AIR, MAT_WATER_CRYSTAL};
use crate::node64::{LeafBrick64, Node64};

pub const LEVEL0_SCALE: u32 = 64; // 256 / 4
pub const LEVEL1_SCALE: u32 = 16; // 64 / 4
pub const LEVEL2_SCALE: u32 = 4; // 16 / 4
pub const LEVEL3_SCALE: u32 = 1; // 4 / 4

/// Authoritative 4-Level Sparse 64-Tree for a single 256³ voxel chunk.
///
/// Stores its nodes and non-uniform material bricks in flat contiguous `Vec`s
/// so they can be uploaded directly into the GPU `NodePool` and `BrickPool`.
#[derive(Clone, Debug)]
pub struct ChunkTree64 {
    /// Contiguous `Node64` array. `nodes[0]` is always the Level-0 chunk root.
    pub nodes: Vec<Node64>,
    /// Contiguous `LeafBrick64` array referenced by non-uniform Level-3 leaf nodes.
    pub bricks: Vec<LeafBrick64>,
    /// Total count of non-empty (solid) voxels in this chunk (0..=16,777,216).
    pub solid_voxel_count: u32,
}

impl Default for ChunkTree64 {
    fn default() -> Self {
        Self::new_empty()
    }
}

impl ChunkTree64 {
    /// Creates an empty air chunk containing a single empty root `Node64`.
    pub fn new_empty() -> Self {
        Self {
            nodes: vec![Node64::empty()],
            bricks: Vec::new(),
            solid_voxel_count: 0,
        }
    }

    /// Returns true if the entire 256³ chunk contains only air.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.nodes[0].mask_u64() == 0
    }

    /// Returns the root `Node64` of this chunk.
    #[inline(always)]
    pub fn root(&self) -> Node64 {
        self.nodes[0]
    }

    /// Queries the `MaterialId` at chunk-local coordinates `(x, y, z)` each in `0..256`.
    /// Executes in at most 4 bitwise `POPCNT` steps (or fewer if homogeneous-collapsed).
    #[inline]
    pub fn get_voxel(&self, x: u32, y: u32, z: u32) -> MaterialId {
        debug_assert!(x < CHUNK_DIM && y < CHUNK_DIM && z < CHUNK_DIM);
        let mut node_idx = 0usize;
        let mut shift = 6u32; // Level 0: shift=6 (64), L1: shift=4 (16), L2: shift=2 (4), L3: shift=0 (1)

        loop {
            let node = self.nodes[node_idx];
            let cx = (x >> shift) & 3;
            let cy = (y >> shift) & 3;
            let cz = (z >> shift) & 3;
            let b = child_bit_index(cx, cy, cz);

            if !node.has_child(b) {
                return MAT_AIR;
            }

            if node.is_leaf() {
                if node.is_uniform() {
                    return node.primary_material();
                } else {
                    return self.bricks[node.base_ptr() as usize].get_material(b);
                }
            }

            node_idx = node.child_slot_offset(b) as usize;
            debug_assert!(shift >= 2);
            shift -= 2;
        }
    }

    /// Sets a single chunk-local voxel coordinate `(x, y, z)` in `0..256` to `mat`.
    /// Returns true if an air voxel transitioned to solid.
    #[inline]
    pub fn set_voxel(
        &mut self,
        x: u32,
        y: u32,
        z: u32,
        mat: MaterialId,
        palette: &crate::material::MaterialPalette,
    ) -> bool {
        self.set_voxels_local(&[(x, y, z, mat)], palette) > 0
    }

    /// Queries the 64-bit solid voxel occupancy mask for the $4 \times 4 \times 4$ Leaf Brick
    /// at chunk-local brick coordinates `(bx, by, bz)` each in `0..64` (where `x = bx * 4`, etc.).
    ///
    /// - If an ancestor internal node (or leaf node) is collapsed as `IS_UNIFORM_FLAG`, returns
    ///   `u64::MAX` (all 64 voxels solid) if the material is non-air/non-water, or `0` if empty/water.
    /// - Otherwise returns the leaf `Node64`'s 64-bit occupancy mask (with water bits cleared if mixed).
    #[inline]
    pub fn get_brick_solid_mask(&self, bx: u32, by: u32, bz: u32) -> u64 {
        debug_assert!(bx < 64 && by < 64 && bz < 64);
        let x = bx << 2;
        let y = by << 2;
        let z = bz << 2;
        let mut node_idx = 0usize;
        let mut shift = 6u32;

        loop {
            let node = self.nodes[node_idx];

            if node.is_leaf() {
                if shift > 0 {
                    // Homogeneous-collapsed parent node (covers 16³ or 64³ voxels):
                    let mat = node.primary_material();
                    return if mat != MAT_AIR && mat != MAT_WATER_CRYSTAL {
                        u64::MAX
                    } else {
                        0
                    };
                }
                // Level-3 Leaf Brick (`shift == 0`):
                let mask = node.mask_u64();
                if node.is_uniform() {
                    let mat = node.primary_material();
                    return if mat != MAT_AIR && mat != MAT_WATER_CRYSTAL {
                        mask
                    } else {
                        0
                    };
                } else {
                    let brick = &self.bricks[node.base_ptr() as usize];
                    let mut solid_mask = mask;
                    let mut rem = mask;
                    while rem != 0 {
                        let bit = rem.trailing_zeros();
                        let mat = brick.get_material(bit);
                        if mat == MAT_AIR || mat == MAT_WATER_CRYSTAL {
                            solid_mask &= !(1u64 << bit);
                        }
                        rem &= rem - 1;
                    }
                    return solid_mask;
                }
            }

            let cx = (x >> shift) & 3;
            let cy = (y >> shift) & 3;
            let cz = (z >> shift) & 3;
            let b = child_bit_index(cx, cy, cz);

            if !node.has_child(b) {
                return 0;
            }

            node_idx = node.child_slot_offset(b) as usize;
            shift -= 2;
        }
    }

    /// Checks whether any solid voxel exists inside the chunk-local AABB `[min_x..max_x, min_y..max_y, min_z..max_z]`
    /// (inclusive bounds in `0..256`) using fast hierarchical `u64` bitmask intersection.
    pub fn intersects_aabb_local(
        &self,
        min_x: u32,
        min_y: u32,
        min_z: u32,
        max_x: u32,
        max_y: u32,
        max_z: u32,
    ) -> bool {
        if self.is_empty() {
            return false;
        }
        self.node_intersects_aabb(0, 0, 0, 0, 6, min_x, min_y, min_z, max_x, max_y, max_z)
    }

    #[allow(clippy::too_many_arguments)]
    fn node_intersects_aabb(
        &self,
        node_idx: usize,
        ox: u32,
        oy: u32,
        oz: u32,
        shift: u32,
        min_x: u32,
        min_y: u32,
        min_z: u32,
        max_x: u32,
        max_y: u32,
        max_z: u32,
    ) -> bool {
        let node = self.nodes[node_idx];
        let mask = node.mask_u64();
        if mask == 0 {
            return false;
        }

        // Compute the 4x4x4 child range overlapped by [min..=max] at this level:
        let child_size = 1u32 << shift;
        let c_min_x = min_x.saturating_sub(ox) >> shift;
        let c_min_y = min_y.saturating_sub(oy) >> shift;
        let c_min_z = min_z.saturating_sub(oz) >> shift;
        let c_max_x = ((max_x - ox) >> shift).min(3);
        let c_max_y = ((max_y - oy) >> shift).min(3);
        let c_max_z = ((max_z - oz) >> shift).min(3);

        // Build a 64-bit query mask for all overlapped child cells in [c_min..=c_max]:
        let mut query_mask = 0u64;
        for cz in c_min_z..=c_max_z {
            for cy in c_min_y..=c_max_y {
                for cx in c_min_x..=c_max_x {
                    query_mask |= 1u64 << child_bit_index(cx, cy, cz);
                }
            }
        }

        let overlapped = mask & query_mask;
        if overlapped == 0 {
            return false;
        }

        // If this node is a leaf (either Level 3 or a homogeneous-collapsed higher node),
        // ignore passable water (`MAT_WATER_CRYSTAL`) so the player swims instead of walking on water!
        if node.is_leaf() {
            if node.is_uniform() {
                let mat = node.primary_material();
                return mat != MAT_AIR && mat != MAT_WATER_CRYSTAL;
            } else {
                let brick = &self.bricks[node.base_ptr() as usize];
                let mut rem = overlapped;
                while rem != 0 {
                    let b = rem.trailing_zeros();
                    let mat = brick.get_material(b);
                    if mat != MAT_AIR && mat != MAT_WATER_CRYSTAL {
                        return true;
                    }
                    rem &= rem - 1;
                }
                return false;
            }
        }

        // Otherwise descend only into the overlapped active children:
        let mut rem = overlapped;
        while rem != 0 {
            let b = rem.trailing_zeros();
            let (cx, cy, cz) = unpack_child_index(b);
            let child_idx = node.child_slot_offset(b) as usize;
            if self.node_intersects_aabb(
                child_idx,
                ox + cx * child_size,
                oy + cy * child_size,
                oz + cz * child_size,
                shift - 2,
                min_x,
                min_y,
                min_z,
                max_x,
                max_y,
                max_z,
            ) {
                return true;
            }
            rem &= rem - 1;
        }
        false
    }
}
