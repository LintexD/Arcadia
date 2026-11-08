//! Intermediate `BuildNode` representation, procedural generators, and BFS level-order packing.
//! Owns bottom-up chunk construction, homogeneous subtree collapsing, and GPU-cache-friendly BFS packing.

use std::collections::{HashMap, VecDeque};

use crate::coord::{child_bit_index, CHUNK_DIM};
use crate::material::{MaterialId, MaterialPalette, MAT_AIR};
use crate::node64::{LeafBrick64, Node64, IS_LEAF_FLAG, IS_UNIFORM_FLAG};

use super::tree::ChunkTree64;

/// Intermediate tree node used during recursive bottom-up construction and editing.
#[derive(Clone)]
pub enum BuildNode {
    Empty,
    /// Collapsed node where every occupied child bit `b` in `mask` is 100% solid with `mat`.
    /// At Level 3 (`shift == 0`), any uniform leaf is represented by `Uniform { mask, mat }`.
    Uniform {
        mask: u64,
        mat: MaterialId,
        lod_payload: u32,
        /// True only if `mask == u64::MAX` AND every child is also 100% solid `u64::MAX`.
        is_fully_solid_cube: bool,
    },
    /// Level-3 leaf brick with mixed materials across its 4×4×4 voxels.
    MixedLeaf {
        mask: u64,
        brick: LeafBrick64,
        lod_payload: u32,
    },
    /// Internal node (Levels 0, 1, 2) with up to 64 active children.
    Internal {
        mask: u64,
        children: Vec<BuildNode>,
        lod_payload: u32,
    },
}

impl BuildNode {
    #[inline]
    pub(crate) fn lod_payload(&self) -> Option<u32> {
        match self {
            BuildNode::Empty => None,
            BuildNode::Uniform { lod_payload, .. }
            | BuildNode::MixedLeaf { lod_payload, .. }
            | BuildNode::Internal { lod_payload, .. } => Some(*lod_payload),
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn dominant_material(&self) -> Option<MaterialId> {
        self.lod_payload().map(|p| (p >> 24) as u8)
    }

    #[allow(dead_code)]
    pub(crate) fn count_solid_voxels(&self, shift: u32) -> u32 {
        let child_vol = (1u32 << shift).pow(3);
        match self {
            BuildNode::Empty => 0,
            BuildNode::Uniform { mask, .. } | BuildNode::MixedLeaf { mask, .. } => {
                mask.count_ones() * child_vol
            }
            BuildNode::Internal { children, .. } => children
                .iter()
                .map(|c| c.count_solid_voxels(shift - 2))
                .sum(),
        }
    }
}

#[inline]
pub(crate) fn pack_rgb_and_mat(r: f32, g: f32, b: f32, mat: MaterialId) -> u32 {
    let ru = (r.clamp(0.0, 1.0) * 255.0) as u32;
    let gu = (g.clamp(0.0, 1.0) * 255.0) as u32;
    let bu = (b.clamp(0.0, 1.0) * 255.0) as u32;
    ((mat as u32) << 24) | (bu << 16) | (gu << 8) | ru
}

impl ChunkTree64 {
    /// Constructs a compressed `ChunkTree64` from a procedural sampling closure `f(x, y, z) -> MaterialId`.
    ///
    /// To avoid evaluating 16.7 million points when large regions are empty sky or solid underground rock,
    /// `region_hint(ox, oy, oz, size)` can optionally return `Some(MAT_AIR)` or `Some(uniform_mat)`
    /// to skip entire 64³, 16³, or 4³ blocks in $O(1)$!
    pub fn from_generator<F, H>(palette: &MaterialPalette, mut region_hint: H, mut eval: F) -> Self
    where
        H: FnMut(u32, u32, u32, u32) -> Option<MaterialId>,
        F: FnMut(u32, u32, u32) -> MaterialId,
    {
        let mut solid_count = 0u32;
        let root_build = Self::build_recursive(
            0,
            0,
            0,
            6,
            palette,
            &mut region_hint,
            &mut eval,
            &mut solid_count,
        );
        Self::pack_build_tree(&root_build, solid_count)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_recursive<F, H>(
        ox: u32,
        oy: u32,
        oz: u32,
        shift: u32,
        palette: &MaterialPalette,
        region_hint: &mut H,
        eval: &mut F,
        solid_count: &mut u32,
    ) -> BuildNode
    where
        H: FnMut(u32, u32, u32, u32) -> Option<MaterialId>,
        F: FnMut(u32, u32, u32) -> MaterialId,
    {
        let node_size = 4u32 << shift; // shift=6 -> 256, shift=4 -> 64, shift=2 -> 16, shift=0 -> 4

        // Fast-path coarse region hint (e.g., high sky is all air, deep underground is solid stone):
        if let Some(uniform_mat) = region_hint(ox, oy, oz, node_size) {
            if uniform_mat == MAT_AIR {
                return BuildNode::Empty;
            } else {
                let voxels_in_node = node_size * node_size * node_size;
                *solid_count += voxels_in_node;
                let lod_payload = palette.get(uniform_mat).pack_lod_payload(uniform_mat);
                return BuildNode::Uniform {
                    mask: u64::MAX,
                    mat: uniform_mat,
                    lod_payload,
                    is_fully_solid_cube: true,
                };
            }
        }

        if shift == 0 {
            // Level 3 Leaf (4x4x4 individual voxels):
            let mut mask = 0u64;
            let mut brick = LeafBrick64::default();
            let mut first_mat = MAT_AIR;
            let mut all_same_mat = true;
            let mut r_sum = 0.0f32;
            let mut g_sum = 0.0f32;
            let mut b_sum = 0.0f32;
            let mut emissive_mat = MAT_AIR;
            let mut emissive_voxels = 0u32;

            for lz in 0..4 {
                for ly in 0..4 {
                    for lx in 0..4 {
                        let mat = eval(ox + lx, oy + ly, oz + lz);
                        if mat != MAT_AIR {
                            let b = child_bit_index(lx, ly, lz);
                            mask |= 1u64 << b;
                            brick.set_material(b, mat);
                            if first_mat == MAT_AIR {
                                first_mat = mat;
                            } else if mat != first_mat {
                                all_same_mat = false;
                            }
                            let m = palette.get(mat);
                            // GigaVoxels volumetric pre-filtering: blend albedo + emission per occupied voxel
                            r_sum += (m.albedo_roughness[0] + m.emission_meta[0] * 0.25).min(1.0);
                            g_sum += (m.albedo_roughness[1] + m.emission_meta[1] * 0.25).min(1.0);
                            b_sum += (m.albedo_roughness[2] + m.emission_meta[2] * 0.25).min(1.0);
                            if m.is_emissive() {
                                emissive_mat = mat;
                                emissive_voxels += 1;
                            }
                        }
                    }
                }
            }

            if mask == 0 {
                return BuildNode::Empty;
            }

            let count = mask.count_ones();
            *solid_count += count;

            if all_same_mat {
                let lod_payload = palette.get(first_mat).pack_lod_payload(first_mat);
                return BuildNode::Uniform {
                    mask,
                    mat: first_mat,
                    lod_payload,
                    is_fully_solid_cube: mask == u64::MAX,
                };
            }

            let inv = 1.0 / (count as f32);
            // Only classify the mixed leaf brick as an emissive material if at least half of its
            // occupied voxels are emissive (prevents 1 glowing voxel from turning a solid wall emissive):
            let rep_mat = if emissive_mat != MAT_AIR && emissive_voxels * 2 >= count {
                emissive_mat
            } else {
                first_mat
            };
            let lod_payload = pack_rgb_and_mat(r_sum * inv, g_sum * inv, b_sum * inv, rep_mat);

            BuildNode::MixedLeaf {
                mask,
                brick,
                lod_payload,
            }
        } else {
            // Internal Level (shift = 6, 4, or 2):
            let child_size = 1u32 << shift; // 64, 16, or 4
            let mut mask = 0u64;
            let mut children = Vec::new();
            let mut all_children_full_uniform = true;
            let mut common_mat = MAT_AIR;
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            let mut solid_mat = MAT_AIR;
            let mut emissive_mat = MAT_AIR;
            let mut emissive_children = 0u32;

            for cz in 0..4 {
                for cy in 0..4 {
                    for cx in 0..4 {
                        let child = Self::build_recursive(
                            ox + cx * child_size,
                            oy + cy * child_size,
                            oz + cz * child_size,
                            shift - 2,
                            palette,
                            region_hint,
                            eval,
                            solid_count,
                        );
                        if let Some(child_lod) = child.lod_payload() {
                            let b = child_bit_index(cx, cy, cz);
                            mask |= 1u64 << b;

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

                            match &child {
                                BuildNode::Uniform {
                                    mat,
                                    is_fully_solid_cube: true,
                                    ..
                                } => {
                                    if common_mat == MAT_AIR {
                                        common_mat = *mat;
                                    } else if common_mat != *mat {
                                        all_children_full_uniform = false;
                                    }
                                }
                                _ => {
                                    all_children_full_uniform = false;
                                }
                            }

                            children.push(child);
                        }
                    }
                }
            }

            if mask == 0 {
                return BuildNode::Empty;
            }

            // Homogeneous subtree collapse: if every non-empty child of this node is a 100% solid
            // uniform cube of the same material `common_mat`, collapse into a single `Uniform` leaf!
            if all_children_full_uniform && common_mat != MAT_AIR {
                let lod_payload = palette.get(common_mat).pack_lod_payload(common_mat);
                return BuildNode::Uniform {
                    mask,
                    mat: common_mat,
                    lod_payload,
                    is_fully_solid_cube: mask == u64::MAX,
                };
            }

            let count = children.len() as u32;
            // GigaVoxels coverage-weighted material selection: only mark an internal LOD node as
            // emissive if >= 50% of its occupied children are emissive (otherwise use solid_mat while
            // preserving the coverage-weighted emissive tint in the pre-filtered RGB channels!):
            let dominant_mat = if emissive_mat != MAT_AIR
                && (emissive_children * 2 >= count || solid_mat == MAT_AIR)
            {
                emissive_mat
            } else {
                solid_mat
            };
            let lod_payload = ((dominant_mat as u32) << 24)
                | (((b_sum / count) & 0xFF) << 16)
                | (((g_sum / count) & 0xFF) << 8)
                | ((r_sum / count) & 0xFF);

            BuildNode::Internal {
                mask,
                children,
                lod_payload,
            }
        }
    }

    /// Packs a recursive `BuildNode` tree into flat, contiguous `nodes` in **Breadth-First (BFS / Level-Order)**
    /// layout and `bricks` arrays so coarse LOD levels (Level 0, Level 1, Level 2) sit contiguously at the
    /// start of the buffer for maximum GPU L1/L2 cache line locality during ray marching.
    pub(crate) fn pack_build_tree(root: &BuildNode, solid_voxel_count: u32) -> Self {
        match root {
            BuildNode::Empty => Self::new_empty(),
            _ => {
                let mut nodes = Vec::with_capacity(256);
                let mut bricks = Vec::with_capacity(128);
                let mut queue = VecDeque::with_capacity(128);

                // Reserve slot 0 for the chunk root and enqueue in Breadth-First (Level-Order) order:
                nodes.push(Node64::empty());
                queue.push_back((0usize, root));

                while let Some((target_slot, build)) = queue.pop_front() {
                    match build {
                        BuildNode::Empty => {
                            nodes[target_slot] = Node64::empty();
                        }
                        BuildNode::Uniform {
                            mask, lod_payload, ..
                        } => {
                            nodes[target_slot] = Node64::from_u64(
                                *mask,
                                IS_LEAF_FLAG | IS_UNIFORM_FLAG,
                                *lod_payload,
                            );
                        }
                        BuildNode::MixedLeaf {
                            mask,
                            brick,
                            lod_payload,
                        } => {
                            let brick_idx = bricks.len() as u32;
                            bricks.push(*brick);
                            nodes[target_slot] =
                                Node64::from_u64(*mask, IS_LEAF_FLAG | brick_idx, *lod_payload);
                        }
                        BuildNode::Internal {
                            mask,
                            children,
                            lod_payload,
                        } => {
                            let base_ptr = nodes.len() as u32;
                            // Allocate contiguous sibling slots for POPCNT child indexing, then enqueue
                            // children onto the BFS queue so all nodes of the same tree level stay contiguous:
                            nodes.resize(nodes.len() + children.len(), Node64::empty());
                            nodes[target_slot] = Node64::from_u64(*mask, base_ptr, *lod_payload);
                            for (i, child) in children.iter().enumerate() {
                                queue.push_back(((base_ptr as usize) + i, child));
                            }
                        }
                    }
                }

                Self {
                    nodes,
                    bricks,
                    solid_voxel_count,
                }
            }
        }
    }

    /// Unpacks the current `ChunkTree64` back into a `BuildNode` hierarchy for structural edits.
    #[allow(dead_code)]
    pub(crate) fn unpack_to_build(&self, node_idx: usize) -> BuildNode {
        let node = self.nodes[node_idx];
        let mask = node.mask_u64();
        if mask == 0 {
            return BuildNode::Empty;
        }
        if node.is_leaf() {
            if node.is_uniform() {
                let mat = node.primary_material();
                BuildNode::Uniform {
                    mask,
                    mat,
                    lod_payload: node.lod_payload,
                    is_fully_solid_cube: mask == u64::MAX,
                }
            } else {
                let brick = self.bricks[node.base_ptr() as usize];
                BuildNode::MixedLeaf {
                    mask,
                    brick,
                    lod_payload: node.lod_payload,
                }
            }
        } else {
            let count = mask.count_ones() as usize;
            let base = node.base_ptr() as usize;
            let mut children = Vec::with_capacity(count);
            for i in 0..count {
                children.push(self.unpack_to_build(base + i));
            }
            BuildNode::Internal {
                mask,
                children,
                lod_payload: node.lod_payload,
            }
        }
    }

    /// Builds a tight local `ChunkTree64` from a sparse list of local voxel coordinates
    /// `(lx, ly, lz, mat_id)` where `lx, ly, lz < 256`.
    pub fn from_sparse_voxels(
        voxels: &[(u32, u32, u32, MaterialId)],
        palette: &MaterialPalette,
    ) -> Self {
        if voxels.is_empty() {
            return Self::new_empty();
        }
        let mut brick_map: HashMap<(u32, u32, u32), (u64, LeafBrick64)> =
            HashMap::with_capacity(voxels.len() / 8 + 1);
        for &(x, y, z, mat) in voxels {
            if mat == MAT_AIR || x >= CHUNK_DIM || y >= CHUNK_DIM || z >= CHUNK_DIM {
                continue;
            }
            let bx = x >> 2;
            let by = y >> 2;
            let bz = z >> 2;
            let b = child_bit_index(x & 3, y & 3, z & 3);
            let entry = brick_map
                .entry((bx, by, bz))
                .or_insert_with(|| (0u64, LeafBrick64::default()));
            entry.0 |= 1u64 << b;
            entry.1.set_material(b, mat);
        }

        if brick_map.is_empty() {
            return Self::new_empty();
        }

        Self::from_generator(
            palette,
            |ox, oy, oz, size| {
                // Quick check if any brick in [ox..ox+size, oy..oy+size, oz..oz+size] exists:
                let b_min_x = ox >> 2;
                let b_min_y = oy >> 2;
                let b_min_z = oz >> 2;
                let b_max_x = (ox + size) >> 2;
                let b_max_y = (oy + size) >> 2;
                let b_max_z = (oz + size) >> 2;
                for bz in b_min_z..b_max_z {
                    for by in b_min_y..b_max_y {
                        for bx in b_min_x..b_max_x {
                            if brick_map.contains_key(&(bx, by, bz)) {
                                return None;
                            }
                        }
                    }
                }
                Some(MAT_AIR)
            },
            |x, y, z| {
                let bx = x >> 2;
                let by = y >> 2;
                let bz = z >> 2;
                if let Some((mask, brick)) = brick_map.get(&(bx, by, bz)) {
                    let b = child_bit_index(x & 3, y & 3, z & 3);
                    if (*mask & (1u64 << b)) != 0 {
                        return brick.get_material(b);
                    }
                }
                MAT_AIR
            },
        )
    }
}
