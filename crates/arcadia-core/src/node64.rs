//! `Node64` and `LeafBrick64`: 16-byte cache-aligned 64-tree building blocks.
//! Owns CPU representation and mirrors GPU memory layout.

use crate::material::{MaterialId, MAT_AIR};
use bytemuck::{Pod, Zeroable};

/// Flag in `Node64::child_ptr` (bit 31) indicating this node is a Leaf (Level 3, 4×4×4 voxels)
/// rather than an internal node pointing to further `Node64` children.
pub const IS_LEAF_FLAG: u32 = 1 << 31;

/// Flag in `Node64::child_ptr` (bit 30) indicating that all occupied children/voxels in this node
/// share a single uniform `MaterialId` stored in `lod_payload >> 24` (requiring 0 `BrickPool` storage!).
pub const IS_UNIFORM_FLAG: u32 = 1 << 30;

/// Bitmask for extracting the 30-bit pool base index from `Node64::child_ptr`.
pub const PTR_INDEX_MASK: u32 = 0x3FFF_FFFF;

/// A single node in the 64-tree (16 bytes, matches WGSL `Node64` struct layout 1:1).
///
/// - For internal nodes (Levels 0, 1, 2): `mask` indicates which of the 4×4×4 children exist.
///   Active children are stored contiguously at `NodePool[base_ptr + popcnt(mask & ((1 << b) - 1))]`.
/// - For leaf nodes (Level 3, covering 4×4×4 = 64 individual voxels): `IS_LEAF_FLAG` is set,
///   and `mask` *is* the exact 64-voxel occupancy mask!
// GPU-SYNC: Mirrors `Node64` in `shaders/common/types.wgsl` (@group(0) @binding(2), 16 bytes, 4-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Node64 {
    /// Lower 32 bits of the 64-bit child occupancy mask.
    pub mask_low: u32,
    /// Upper 32 bits of the 64-bit child occupancy mask.
    pub mask_high: u32,
    /// Bit 31: `IS_LEAF_FLAG`. Bit 30: `IS_UNIFORM_FLAG`. Bits 0..30: base index into `NodePool` or `BrickPool`.
    pub child_ptr: u32,
    /// Bits 0..23: RGB8 average color for Cone-LOD & GI. Bits 24..31: primary `MaterialId`.
    pub lod_payload: u32,
}

impl Node64 {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            mask_low: 0,
            mask_high: 0,
            child_ptr: 0,
            lod_payload: 0,
        }
    }

    #[inline(always)]
    pub const fn from_u64(mask: u64, child_ptr: u32, lod_payload: u32) -> Self {
        Self {
            mask_low: mask as u32,
            mask_high: (mask >> 32) as u32,
            child_ptr,
            lod_payload,
        }
    }

    /// Reconstructs the 64-bit occupancy bitmask.
    #[inline(always)]
    pub const fn mask_u64(&self) -> u64 {
        (self.mask_low as u64) | ((self.mask_high as u64) << 32)
    }

    /// Updates the 64-bit occupancy bitmask.
    #[inline(always)]
    pub fn set_mask_u64(&mut self, mask: u64) {
        self.mask_low = mask as u32;
        self.mask_high = (mask >> 32) as u32;
    }

    /// Returns true if child `b in 0..64` is occupied.
    #[inline(always)]
    pub const fn has_child(&self, b: u32) -> bool {
        (self.mask_u64() & (1u64 << b)) != 0
    }

    /// Number of occupied children (1..=64) via hardware `POPCNT`.
    #[inline(always)]
    pub const fn active_child_count(&self) -> u32 {
        self.mask_u64().count_ones()
    }

    /// Computes the contiguous array index of child `b in 0..64` using single-cycle `POPCNT`.
    #[inline(always)]
    pub const fn child_slot_offset(&self, b: u32) -> u32 {
        let prefix_mask = (1u64 << b) - 1;
        let rank = (self.mask_u64() & prefix_mask).count_ones();
        self.base_ptr() + rank
    }

    #[inline(always)]
    pub const fn is_leaf(&self) -> bool {
        (self.child_ptr & IS_LEAF_FLAG) != 0
    }

    #[inline(always)]
    pub const fn is_uniform(&self) -> bool {
        (self.child_ptr & IS_UNIFORM_FLAG) != 0
    }

    #[inline(always)]
    pub const fn base_ptr(&self) -> u32 {
        self.child_ptr & PTR_INDEX_MASK
    }

    #[inline(always)]
    pub const fn primary_material(&self) -> MaterialId {
        (self.lod_payload >> 24) as u8
    }
}

/// 64-byte leaf material brick storing one `u8` `MaterialId` for each of the 64 voxels in a 4×4×4 leaf.
/// Packed as `16 × u32` so WGSL shaders can read any voxel's material with a single 32-bit word load + shift.
// GPU-SYNC: Mirrors `LeafBrick64` in `shaders/common/types.wgsl` (@group(0) @binding(3), 64 bytes, 4-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct LeafBrick64 {
    pub packed_materials: [u32; 16],
}

impl LeafBrick64 {
    #[inline(always)]
    pub fn get_material(&self, bit_index: u32) -> MaterialId {
        debug_assert!(bit_index < 64);
        let word = self.packed_materials[(bit_index >> 2) as usize];
        let shift = (bit_index & 3) * 8;
        ((word >> shift) & 0xFF) as MaterialId
    }

    #[inline(always)]
    pub fn set_material(&mut self, bit_index: u32, mat: MaterialId) {
        debug_assert!(bit_index < 64);
        let word_idx = (bit_index >> 2) as usize;
        let shift = (bit_index & 3) * 8;
        self.packed_materials[word_idx] =
            (self.packed_materials[word_idx] & !(0xFF << shift)) | ((mat as u32) << shift);
    }

    /// Checks whether all non-air voxels in `mask` have the exact same `MaterialId`.
    /// Returns `Some(uniform_mat)` if homogeneous, or `None` if mixed.
    pub fn check_uniform(&self, mask: u64) -> Option<MaterialId> {
        if mask == 0 {
            return Some(MAT_AIR);
        }
        let first_bit = mask.trailing_zeros();
        let first_mat = self.get_material(first_bit);
        let mut rem = mask & (mask - 1);
        while rem != 0 {
            let b = rem.trailing_zeros();
            if self.get_material(b) != first_mat {
                return None;
            }
            rem &= rem - 1;
        }
        Some(first_mat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_popcnt_child_slot_offset() {
        // Suppose children 0, 5, 19, and 63 are present:
        let mask = (1u64 << 0) | (1u64 << 5) | (1u64 << 19) | (1u64 << 63);
        let node = Node64::from_u64(mask, 100, 0);
        assert_eq!(node.active_child_count(), 4);
        assert_eq!(node.child_slot_offset(0), 100);
        assert_eq!(node.child_slot_offset(5), 101);
        assert_eq!(node.child_slot_offset(19), 102);
        assert_eq!(node.child_slot_offset(63), 103);
    }

    #[test]
    fn test_leaf_brick_pack_unpack() {
        let mut brick = LeafBrick64::default();
        for i in 0..64 {
            brick.set_material(i, (i as u8) + 1);
        }
        for i in 0..64 {
            assert_eq!(brick.get_material(i), (i as u8) + 1);
        }
    }
}
