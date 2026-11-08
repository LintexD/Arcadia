//! In-place leaf brick mutation, missing branch growth, and zero-repack sculpting for `ChunkTree64`.
//! Owns CPU direct 64-tree traversal, brick pool mutation, and ancestor LOD payload refreshing.

use std::collections::{HashMap, VecDeque};

use crate::chunk_tree::builder::pack_rgb_and_mat;
use crate::chunk_tree::ChunkTree64;
use crate::coord::{child_bit_index, CHUNK_DIM};
use crate::material::{MaterialId, MaterialPalette, MAT_AIR};
use crate::node64::{LeafBrick64, Node64, IS_LEAF_FLAG, IS_UNIFORM_FLAG, PTR_INDEX_MASK};

/// Type alias mapping 4x4x4 leaf brick coordinates to lists of local bit indices and material IDs.
pub type BrickVoxelListMap = HashMap<(u32, u32, u32), Vec<(u32, MaterialId)>>;

/// Fine-grained modification report tracking dirty node and brick ranges from in-place voxel edits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChunkEditReport {
    /// Number of voxels that transitioned from air to solid.
    pub added_voxels: u32,
    /// Minimum modified node index in `chunk.nodes`.
    pub min_dirty_node: usize,
    /// Maximum modified node index in `chunk.nodes`.
    pub max_dirty_node: usize,
    /// Minimum modified brick index in `chunk.bricks`.
    pub min_dirty_brick: usize,
    /// Maximum modified brick index in `chunk.bricks`.
    pub max_dirty_brick: usize,
    /// Whether node allocations or uncollapses expanded the chunk's tree hierarchy.
    pub had_structural_growth: bool,
}

impl ChunkEditReport {
    /// Constructs an empty report with inverted min/max bounds (no dirty nodes or bricks).
    #[inline]
    pub const fn empty() -> Self {
        Self {
            added_voxels: 0,
            min_dirty_node: usize::MAX,
            max_dirty_node: 0,
            min_dirty_brick: usize::MAX,
            max_dirty_brick: 0,
            had_structural_growth: false,
        }
    }

    /// Marks a node index as dirty, updating min and max bounds.
    #[inline]
    pub fn mark_node_dirty(&mut self, node_idx: usize) {
        self.min_dirty_node = self.min_dirty_node.min(node_idx);
        self.max_dirty_node = self.max_dirty_node.max(node_idx);
    }

    /// Marks a contiguous range of node indices as dirty, updating min and max bounds.
    #[inline]
    pub fn mark_node_range_dirty(&mut self, start: usize, end_inclusive: usize) {
        if start <= end_inclusive {
            self.min_dirty_node = self.min_dirty_node.min(start);
            self.max_dirty_node = self.max_dirty_node.max(end_inclusive);
        }
    }

    /// Marks a brick index as dirty, updating min and max bounds.
    #[inline]
    pub fn mark_brick_dirty(&mut self, brick_idx: usize) {
        self.min_dirty_brick = self.min_dirty_brick.min(brick_idx);
        self.max_dirty_brick = self.max_dirty_brick.max(brick_idx);
    }

    /// Returns true if at least one node is marked dirty (`min_dirty_node <= max_dirty_node`).
    #[inline]
    pub fn has_dirty_nodes(&self) -> bool {
        self.min_dirty_node <= self.max_dirty_node
    }

    /// Returns true if at least one brick is marked dirty (`min_dirty_brick <= max_dirty_brick`).
    #[inline]
    pub fn has_dirty_bricks(&self) -> bool {
        self.min_dirty_brick <= self.max_dirty_brick
    }
}

impl ChunkTree64 {
    /// Inserts, updates, or clears chunk-local voxels and materials in-place, returning a fine-grained
    /// [`ChunkEditReport`] recording modified node and brick index ranges for partial GPU staging.
    pub fn set_voxels_local_tracked(
        &mut self,
        local_voxels: &[(u32, u32, u32, MaterialId)],
        palette: &MaterialPalette,
    ) -> ChunkEditReport {
        if local_voxels.is_empty() {
            return ChunkEditReport::empty();
        }

        // Group valid chunk-local voxel updates by 4x4x4 leaf brick coordinate (bx, by, bz):
        let mut brick_set_map: BrickVoxelListMap =
            HashMap::with_capacity(local_voxels.len() / 8 + 1);

        for &(vx, vy, vz, mat) in local_voxels {
            if vx >= CHUNK_DIM || vy >= CHUNK_DIM || vz >= CHUNK_DIM {
                continue;
            }
            let bx = vx >> 2;
            let by = vy >> 2;
            let bz = vz >> 2;
            let b = child_bit_index(vx & 3, vy & 3, vz & 3);
            brick_set_map
                .entry((bx, by, bz))
                .or_default()
                .push((b, mat));
        }

        if brick_set_map.is_empty() {
            return ChunkEditReport::empty();
        }

        let mut report = ChunkEditReport::empty();

        let mut sorted_bricks: Vec<_> = brick_set_map.into_iter().collect();
        sorted_bricks.sort_unstable_by_key(|&((bx, by, bz), _)| (bz, by, bx));

        for ((bx, by, bz), voxel_entries) in sorted_bricks {
            // Guard against growing branches when setting air voxels in an empty branch:
            let all_air = voxel_entries.iter().all(|&(_, mat)| mat == MAT_AIR);
            if all_air && !self.can_reach_leaf(bx, by, bz) {
                continue;
            }

            self.mutate_brick_in_place(bx, by, bz, &voxel_entries, palette, &mut report);
        }

        if self.solid_voxel_count == 0 {
            self.nodes[0] = Node64::empty();
            report.mark_node_dirty(0);
        }

        report
    }

    /// Inserts, updates, or clears chunk-local voxels and materials in-place without re-packing.
    ///
    /// Accepts a slice of `(vx, vy, vz, mat)` where coordinates are in `0..256`.
    /// Returns the number of voxels that transitioned from air to solid.
    #[inline]
    pub fn set_voxels_local(
        &mut self,
        local_voxels: &[(u32, u32, u32, MaterialId)],
        palette: &MaterialPalette,
    ) -> u32 {
        self.set_voxels_local_tracked(local_voxels, palette)
            .added_voxels
    }

    /// Rebuilds `self.nodes` and `self.bricks` keeping only reachable nodes and referenced non-uniform bricks.
    ///
    /// Strips orphaned node slices left behind by tree expansion or child deletion and packs reachable
    /// nodes and bricks contiguously in breadth-first level order.
    pub fn compact(&mut self) {
        if self.is_empty() {
            self.nodes = vec![Node64::empty()];
            self.bricks.clear();
            self.solid_voxel_count = 0;
            return;
        }

        let mut new_nodes = Vec::with_capacity(self.nodes.len());
        let mut new_bricks = Vec::with_capacity(self.bricks.len());
        let mut queue = VecDeque::with_capacity(self.nodes.len() / 4 + 1);

        new_nodes.push(Node64::empty());
        queue.push_back((0usize, 0usize));

        while let Some((new_idx, old_idx)) = queue.pop_front() {
            let node = self.nodes[old_idx];
            if node.mask_u64() == 0 {
                new_nodes[new_idx] = Node64::empty();
            } else if node.is_leaf() {
                if node.is_uniform() {
                    new_nodes[new_idx] = node;
                } else {
                    let old_brick_idx = node.base_ptr() as usize;
                    let new_brick_idx = new_bricks.len() as u32;
                    new_bricks.push(self.bricks[old_brick_idx]);
                    let mut leaf_node = node;
                    leaf_node.child_ptr =
                        (node.child_ptr & !PTR_INDEX_MASK) | (new_brick_idx & PTR_INDEX_MASK);
                    new_nodes[new_idx] = leaf_node;
                }
            } else {
                let count = node.active_child_count() as usize;
                if count == 0 {
                    new_nodes[new_idx] = Node64::empty();
                } else {
                    let old_base = node.base_ptr() as usize;
                    let new_base = new_nodes.len() as u32;
                    new_nodes.resize(new_nodes.len() + count, Node64::empty());
                    let mut internal_node = node;
                    internal_node.child_ptr =
                        (node.child_ptr & !PTR_INDEX_MASK) | (new_base & PTR_INDEX_MASK);
                    new_nodes[new_idx] = internal_node;
                    for i in 0..count {
                        queue.push_back(((new_base as usize) + i, old_base + i));
                    }
                }
            }
        }

        self.nodes = new_nodes;
        self.bricks = new_bricks;
    }

    /// Checks whether an existing leaf or collapsed solid node exists along the path to `(bx, by, bz)`.
    fn can_reach_leaf(&self, bx: u32, by: u32, bz: u32) -> bool {
        if self.is_empty() {
            return false;
        }
        let b_0 = child_bit_index((bx >> 4) & 3, (by >> 4) & 3, (bz >> 4) & 3);
        if self.nodes[0].is_leaf() {
            return self.nodes[0].mask_u64() != 0;
        }
        if !self.nodes[0].has_child(b_0) {
            return false;
        }
        let l1 = self.nodes[0].child_slot_offset(b_0) as usize;
        if self.nodes[l1].is_leaf() {
            return self.nodes[l1].mask_u64() != 0;
        }
        let b_1 = child_bit_index((bx >> 2) & 3, (by >> 2) & 3, (bz >> 2) & 3);
        if !self.nodes[l1].has_child(b_1) {
            return false;
        }
        let l2 = self.nodes[l1].child_slot_offset(b_1) as usize;
        if self.nodes[l2].is_leaf() {
            return self.nodes[l2].mask_u64() != 0;
        }
        let b_2 = child_bit_index(bx & 3, by & 3, bz & 3);
        if !self.nodes[l2].has_child(b_2) {
            return false;
        }
        let l3 = self.nodes[l2].child_slot_offset(b_2) as usize;
        self.nodes[l3].mask_u64() != 0
    }

    /// Descends from root to leaf for a single 4x4x4 brick, mutates voxels, and ascends refreshing LOD.
    pub(crate) fn mutate_brick_in_place(
        &mut self,
        bx: u32,
        by: u32,
        bz: u32,
        voxel_entries: &[(u32, MaterialId)],
        palette: &MaterialPalette,
        report: &mut ChunkEditReport,
    ) {
        let b_0 = child_bit_index((bx >> 4) & 3, (by >> 4) & 3, (bz >> 4) & 3);
        let b_1 = child_bit_index((bx >> 2) & 3, (by >> 2) & 3, (bz >> 2) & 3);
        let b_2 = child_bit_index(bx & 3, by & 3, bz & 3);

        let nodes_len_before = self.nodes.len();
        let bricks_len_before = self.bricks.len();

        let mut path = [0usize; 4];
        path[0] = 0;
        path[1] = self.ensure_child(path[0], b_0, 6, palette);
        path[2] = self.ensure_child(path[1], b_1, 4, palette);
        path[3] = self.ensure_child(path[2], b_2, 2, palette);

        let nodes_len_after = self.nodes.len();
        if nodes_len_after > nodes_len_before {
            report.had_structural_growth = true;
            report.mark_node_range_dirty(nodes_len_before, nodes_len_after - 1);
        }

        for &node_idx in &path {
            report.mark_node_dirty(node_idx);
        }

        let leaf_idx = path[3];
        self.apply_voxel_entries_to_leaf(leaf_idx, voxel_entries, palette, report);
        if self.bricks.len() > bricks_len_before {
            report.had_structural_growth = true;
        }

        if self.nodes[leaf_idx].mask_u64() == 0 {
            self.remove_child(path[2], b_2, report);
            report.mark_node_dirty(path[2]);

            if self.nodes[path[2]].mask_u64() == 0 {
                self.remove_child(path[1], b_1, report);
                report.mark_node_dirty(path[1]);

                if self.nodes[path[1]].mask_u64() == 0 {
                    self.remove_child(path[0], b_0, report);
                    report.mark_node_dirty(path[0]);
                    if self.nodes[0].mask_u64() == 0 {
                        self.nodes[0] = Node64::empty();
                        self.solid_voxel_count = 0;
                    } else {
                        self.refresh_internal_lod(path[0], palette);
                    }
                } else {
                    self.refresh_internal_lod(path[1], palette);
                    self.refresh_internal_lod(path[0], palette);
                }
            } else {
                self.refresh_internal_lod(path[2], palette);
                self.refresh_internal_lod(path[1], palette);
                self.refresh_internal_lod(path[0], palette);
            }
        } else {
            self.refresh_internal_lod(path[2], palette);
            self.refresh_internal_lod(path[1], palette);
            self.refresh_internal_lod(path[0], palette);
        }

        let nodes_len_final = self.nodes.len();
        if nodes_len_final > nodes_len_after {
            report.had_structural_growth = true;
            report.mark_node_range_dirty(nodes_len_after, nodes_len_final - 1);
        }
    }

    /// Removes child slot `b` from `parent_idx`, compacting remaining active children in-place or zeroing.
    fn remove_child(&mut self, parent_idx: usize, b: u32, report: &mut ChunkEditReport) {
        if !self.nodes[parent_idx].has_child(b) {
            return;
        }
        let old_count = self.nodes[parent_idx].active_child_count() as usize;
        if old_count <= 1 {
            let old_base = self.nodes[parent_idx].base_ptr() as usize;
            if old_count == 1 && old_base > 0 && old_base < self.nodes.len() {
                self.nodes[old_base] = Node64::empty();
                report.mark_node_dirty(old_base);
            }
            self.nodes[parent_idx].set_mask_u64(0);
            self.nodes[parent_idx].child_ptr = 0;
            self.nodes[parent_idx].lod_payload = 0;
            report.mark_node_dirty(parent_idx);
            return;
        }

        let old_base = self.nodes[parent_idx].base_ptr() as usize;
        let prefix_mask = (1u64 << b) - 1;
        let rank = (self.nodes[parent_idx].mask_u64() & prefix_mask).count_ones() as usize;
        let new_count = old_count - 1;

        for i in rank..new_count {
            self.nodes[old_base + i] = self.nodes[old_base + i + 1];
        }
        self.nodes[old_base + new_count] = Node64::empty();
        let new_mask = self.nodes[parent_idx].mask_u64() & !(1u64 << b);
        self.nodes[parent_idx].set_mask_u64(new_mask);
        report.mark_node_range_dirty(old_base + rank, old_base + new_count);
        report.mark_node_dirty(parent_idx);
    }

    /// Traverses to child `b` of `parent_idx` at `shift`, growing the branch if absent.
    fn ensure_child(
        &mut self,
        parent_idx: usize,
        b: u32,
        shift: u32,
        palette: &MaterialPalette,
    ) -> usize {
        self.uncollapse_if_needed(parent_idx, shift, palette);

        if self.nodes[parent_idx].has_child(b) {
            return self.nodes[parent_idx].child_slot_offset(b) as usize;
        }

        let old_count = self.nodes[parent_idx].active_child_count() as usize;
        let old_base = self.nodes[parent_idx].base_ptr() as usize;
        let new_base = self.nodes.len() as u32;

        self.nodes
            .resize(self.nodes.len() + old_count + 1, Node64::empty());

        let prefix_mask = (1u64 << b) - 1;
        let rank = (self.nodes[parent_idx].mask_u64() & prefix_mask).count_ones() as usize;

        if old_count > 0 {
            for i in 0..rank {
                self.nodes[(new_base as usize) + i] = self.nodes[old_base + i];
            }
            for i in rank..old_count {
                self.nodes[(new_base as usize) + i + 1] = self.nodes[old_base + i];
            }
        }

        let child_idx = (new_base as usize) + rank;
        if shift == 2 {
            self.nodes[child_idx] = Node64::from_u64(0, IS_LEAF_FLAG | IS_UNIFORM_FLAG, 0);
        } else {
            self.nodes[child_idx] = Node64::empty();
        }

        let new_ptr = (self.nodes[parent_idx].child_ptr & !PTR_INDEX_MASK) | new_base;
        self.nodes[parent_idx].child_ptr = new_ptr;
        let new_mask = self.nodes[parent_idx].mask_u64() | (1u64 << b);
        self.nodes[parent_idx].set_mask_u64(new_mask);

        child_idx
    }

    /// Uncollapses a homogeneous subtree node into active children so it can be mutated.
    fn uncollapse_if_needed(&mut self, node_idx: usize, shift: u32, palette: &MaterialPalette) {
        if shift == 0 || !self.nodes[node_idx].is_leaf() {
            return;
        }

        let mat = self.nodes[node_idx].primary_material();
        let old_mask = self.nodes[node_idx].mask_u64();
        let count = old_mask.count_ones() as usize;
        if count == 0 {
            self.nodes[node_idx].child_ptr = 0;
            return;
        }

        let new_base = self.nodes.len() as u32;
        self.nodes.resize(self.nodes.len() + count, Node64::empty());
        let lod_payload = palette.get(mat).pack_lod_payload(mat);

        for i in 0..count {
            let child_idx = (new_base as usize) + i;
            self.nodes[child_idx] =
                Node64::from_u64(u64::MAX, IS_LEAF_FLAG | IS_UNIFORM_FLAG, lod_payload);
        }

        self.nodes[node_idx].child_ptr = new_base;
    }

    /// Mutates the voxels of a Level 3 leaf node in-place, allocating a brick only if non-uniform.
    fn apply_voxel_entries_to_leaf(
        &mut self,
        leaf_idx: usize,
        voxel_entries: &[(u32, MaterialId)],
        palette: &MaterialPalette,
        report: &mut ChunkEditReport,
    ) {
        for &(b, mat) in voxel_entries {
            let node = self.nodes[leaf_idx];
            let mask = node.mask_u64();
            let was_solid = (mask & (1u64 << b)) != 0;

            if node.is_uniform() {
                if mat == MAT_AIR {
                    if was_solid {
                        self.nodes[leaf_idx].set_mask_u64(mask & !(1u64 << b));
                        self.solid_voxel_count = self.solid_voxel_count.saturating_sub(1);
                    }
                } else if mask == 0 {
                    self.nodes[leaf_idx].set_mask_u64(1u64 << b);
                    self.nodes[leaf_idx].lod_payload = (mat as u32) << 24;
                    self.solid_voxel_count += 1;
                    report.added_voxels += 1;
                } else if mat == node.primary_material() {
                    if !was_solid {
                        self.nodes[leaf_idx].set_mask_u64(mask | (1u64 << b));
                        self.solid_voxel_count += 1;
                        report.added_voxels += 1;
                    }
                } else if was_solid && mask == (1u64 << b) {
                    self.nodes[leaf_idx].lod_payload = (mat as u32) << 24;
                } else {
                    let brick_idx = self.bricks.len() as u32;
                    let mut brick = LeafBrick64::default();
                    let primary = node.primary_material();
                    let mut rem = mask;
                    while rem != 0 {
                        let bit = rem.trailing_zeros();
                        brick.set_material(bit, primary);
                        rem &= rem - 1;
                    }
                    brick.set_material(b, mat);
                    self.bricks.push(brick);
                    report.had_structural_growth = true;

                    self.nodes[leaf_idx].child_ptr = IS_LEAF_FLAG | (brick_idx & PTR_INDEX_MASK);
                    self.nodes[leaf_idx].set_mask_u64(mask | (1u64 << b));
                    if !was_solid {
                        self.solid_voxel_count += 1;
                        report.added_voxels += 1;
                    }
                    report.mark_brick_dirty(brick_idx as usize);
                }
            } else {
                let brick_idx = node.base_ptr() as usize;
                self.bricks[brick_idx].set_material(b, mat);
                report.mark_brick_dirty(brick_idx);
                if mat == MAT_AIR {
                    if was_solid {
                        self.nodes[leaf_idx].set_mask_u64(mask & !(1u64 << b));
                        self.solid_voxel_count = self.solid_voxel_count.saturating_sub(1);
                    }
                } else if !was_solid {
                    self.nodes[leaf_idx].set_mask_u64(mask | (1u64 << b));
                    self.solid_voxel_count += 1;
                    report.added_voxels += 1;
                }
            }
        }

        // Refresh LOD payload for this leaf:
        let final_leaf = self.nodes[leaf_idx];
        let final_mask = final_leaf.mask_u64();
        if final_mask == 0 {
            self.nodes[leaf_idx].lod_payload = 0;
        } else if final_leaf.is_uniform() {
            let mat = final_leaf.primary_material();
            self.nodes[leaf_idx].lod_payload = palette.get(mat).pack_lod_payload(mat);
        } else {
            let brick_idx = final_leaf.base_ptr() as usize;
            let brick = &self.bricks[brick_idx];
            self.nodes[leaf_idx].lod_payload = compute_leaf_lod(final_mask, brick, palette);
        }
    }

    /// Recalculates pre-filtered LOD color and dominant material for an internal node.
    fn refresh_internal_lod(&mut self, node_idx: usize, palette: &MaterialPalette) {
        let node = self.nodes[node_idx];
        let count = node.active_child_count() as usize;
        if count == 0 {
            self.nodes[node_idx].lod_payload = 0;
            return;
        }

        let base = node.base_ptr() as usize;
        let mut r_sum = 0u32;
        let mut g_sum = 0u32;
        let mut b_sum = 0u32;
        let mut solid_mat = MAT_AIR;
        let mut emissive_mat = MAT_AIR;
        let mut emissive_children = 0u32;
        let mut non_empty_children = 0u32;

        for i in 0..count {
            let child = self.nodes[base + i];
            if child.mask_u64() == 0 {
                continue;
            }
            let child_lod = child.lod_payload;
            r_sum += child_lod & 0xFF;
            g_sum += (child_lod >> 8) & 0xFF;
            b_sum += (child_lod >> 16) & 0xFF;
            let c_mat = (child_lod >> 24) as u8;
            if palette.get(c_mat).is_emissive() {
                emissive_mat = c_mat;
                emissive_children += 1;
            } else if solid_mat == MAT_AIR {
                solid_mat = c_mat;
            }
            non_empty_children += 1;
        }

        if non_empty_children == 0 {
            self.nodes[node_idx].lod_payload = 0;
        } else {
            let dominant_mat = if emissive_mat != MAT_AIR
                && (emissive_children * 2 >= non_empty_children || solid_mat == MAT_AIR)
            {
                emissive_mat
            } else {
                solid_mat
            };
            let lod_payload = ((dominant_mat as u32) << 24)
                | (((b_sum / non_empty_children) & 0xFF) << 16)
                | (((g_sum / non_empty_children) & 0xFF) << 8)
                | ((r_sum / non_empty_children) & 0xFF);
            self.nodes[node_idx].lod_payload = lod_payload;
        }
    }
}

/// Computes pre-filtered Cone-LOD color and dominant material for a non-uniform leaf brick.
fn compute_leaf_lod(mask: u64, brick: &LeafBrick64, palette: &MaterialPalette) -> u32 {
    if mask == 0 {
        return 0;
    }
    if let Some(uni_mat) = brick.check_uniform(mask) {
        if uni_mat == MAT_AIR {
            return 0;
        }
        return palette.get(uni_mat).pack_lod_payload(uni_mat);
    }

    let mut r_sum = 0.0f32;
    let mut g_sum = 0.0f32;
    let mut b_sum = 0.0f32;
    let mut solid_mat = MAT_AIR;
    let mut emissive_mat = MAT_AIR;
    let mut emissive_voxels = 0u32;
    let count = mask.count_ones();
    let mut rem = mask;

    while rem != 0 {
        let b = rem.trailing_zeros();
        let m_id = brick.get_material(b);
        let m = palette.get(m_id);
        r_sum += (m.albedo_roughness[0] + m.emission_meta[0] * 0.25).min(1.0);
        g_sum += (m.albedo_roughness[1] + m.emission_meta[1] * 0.25).min(1.0);
        b_sum += (m.albedo_roughness[2] + m.emission_meta[2] * 0.25).min(1.0);
        if m.is_emissive() {
            emissive_mat = m_id;
            emissive_voxels += 1;
        } else if solid_mat == MAT_AIR {
            solid_mat = m_id;
        }
        rem &= rem - 1;
    }

    let rep_mat =
        if emissive_mat != MAT_AIR && (emissive_voxels * 2 >= count || solid_mat == MAT_AIR) {
            emissive_mat
        } else {
            solid_mat
        };

    let inv = 1.0 / (count as f32);
    pack_rgb_and_mat(r_sum * inv, g_sum * inv, b_sum * inv, rep_mat)
}
