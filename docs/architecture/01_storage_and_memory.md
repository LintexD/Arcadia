# Storage, Memory Layout & Tree Mutation Architecture

## Overview
Arcadia's world representation is built on a **Chunked Sparse 64-Tree (CS64)** data structure. This architecture achieves cache-friendly traversal, memory efficiency, single-cycle child indexing via bitwise operations, and real-time in-place voxel sculpting without full-tree rebuilds or allocations during gameplay.

---

## 1. Coordinate System & Scale

- **Handedness:** Right-handed Y-up coordinate system (`+X` East, `+Y` Up, `-Z` Forward).
- **Voxel Scale:** $\text{VOXEL\_SIZE} = 0.0625\text{ m} = 1/16\text{ m} = 6.25\text{ cm}$.
- **Voxel Density:** $\text{VOXELS\_PER\_METER} = 16$ ($4,096\text{ voxels per m}^3$).
- **Chunk Dimensions:** $\text{CHUNK\_DIM} = 256$ voxels per axis ($4^4 = 256$ voxels = $16.0\text{ m} \times 16.0\text{ m} \times 16.0\text{ m}$).
- **World Extents:** Chunk coordinates map to integer multiples of $16\text{ m}$.

### 4-Level 64-Tree Hierarchy
Each $16\text{ m} \times 16\text{ m} \times 16\text{ m}$ chunk is organized as an exact 4-level shallow tree with branching factor $64$ ($4 \times 4 \times 4$ per level):

| Level | Name | Voxel Extent | Metric Extent | Max Nodes / Chunk | Child Unit Size |
|---|---|---|---|---|---|
| **Level 0** | Root Node | $256 \times 256 \times 256$ | $16\text{ m} \times 16\text{ m} \times 16\text{ m}$ ($4096\text{ m}^3$) | 1 | $64^3$ voxels ($4\text{ m}^3$) |
| **Level 1** | Macro Node | $64 \times 64 \times 64$ | $4\text{ m} \times 4\text{ m} \times 4\text{ m}$ ($64\text{ m}^3$) | 64 | $16^3$ voxels ($1\text{ m}^3$) |
| **Level 2** | Meso Node | $16 \times 16 \times 16$ | $1\text{ m} \times 1\text{ m} \times 1\text{ m}$ ($1\text{ m}^3$) | 4,096 | $4^3$ voxels ($0.25\text{ m}^3$) |
| **Level 3** | Leaf Brick | $4 \times 4 \times 4$ | $0.25\text{ m} \times 0.25\text{ m} \times 0.25\text{ m}$ ($0.015625\text{ m}^3$) | Up to 262,144 | Individual $(1/16\text{ m})^3$ voxels |

---

## 2. Bit Index Mapping ($4 \times 4 \times 4 \leftrightarrow 64$-Bit Mask)

Within any node or leaf brick, local sub-cell coordinates $x, y, z \in \{0, 1, 2, 3\}$ map bijectively to a 6-bit child bit index in $[0, 63]$ using 2-bit coordinate packing:

$$\text{index}(x, y, z) = (z \ll 4) \mid (y \ll 2) \mid x$$

$$\text{mask}(x, y, z) = 1\text{u64} \ll \text{index}(x, y, z)$$

```rust
// In crates/arcadia-core/src/coord.rs
#[inline(always)]
pub const fn child_bit_index(x: u32, y: u32, z: u32) -> u32 {
    debug_assert!(x < 4 && y < 4 && z < 4);
    (z << 4) | (y << 2) | x
}

#[inline(always)]
pub const fn child_bit_mask(x: u32, y: u32, z: u32) -> u64 {
    1u64 << child_bit_index(x, y, z)
}
```

This bit packing ensures that consecutive $X$ coordinates lie on adjacent bits, while $Y$ steps by 4 bits and $Z$ steps by 16 bits.

---

## 3. `Node64` Binary Layout (16 Bytes, CPU & GPU Shared)

To maintain strict alignment and zero serialization overhead between Rust host buffers and WGSL compute storage buffers without requiring 64-bit integer hardware shader extensions, the 64-bit occupancy mask is split into two `u32` words:

```rust
// GPU-SYNC: Mirrors `Node64` in `shaders/common/types.wgsl` (16 bytes, 16-byte aligned)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Node64 {
    pub mask_low: u32,       // Bits 0..31 of child occupancy
    pub mask_high: u32,      // Bits 32..63 of child occupancy
    pub child_ptr: u32,      // Bit 31: IS_LEAF_FLAG; Bit 30: IS_UNIFORM_FLAG; Bits 0..29: base index
    pub lod_payload: u32,    // Bits 0..23: RGB8 average color; Bits 24..31: primary MaterialId / emissive flags
}
```

### Pointer Encoding in `child_ptr`
- **Bit 31 (`IS_LEAF_FLAG` = `0x8000_0000`):** Indicates whether target children are `LeafBrick64` entries in `BrickPool` (`1`) or intermediate `Node64` entries in `NodePool` (`0`).
- **Bit 30 (`IS_UNIFORM_FLAG` = `0x4000_0000`):** Homogeneous subtree flag. When set, all solid voxels under this node share the uniform material stored in `lod_payload`, allocating **0 bytes** in `BrickPool` or child `NodePool`.
- **Bits 0..29 (`INDEX_MASK` = `0x3FFF_FFFF`):** Base index into either the chunk-local/global `NodePool` or `BrickPool`. Up to $2^{30} \approx 1.07\times 10^9$ addressable entries.

### Single-Cycle POPCNT Child Indexing
All active children of a node are stored in contiguous memory slices. The offset of active child $b \in [0, 63]$ is determined by counting set bits below $b$:

$$\text{offset}(b) = \operatorname{POPCNT}(\text{mask} \ \& \ ((1 \ll b) - 1))$$

**Rust Implementation:**
```rust
#[inline(always)]
pub fn child_offset_u64(mask: u64, b: u32) -> usize {
    (mask & ((1u64 << b) - 1)).count_ones() as usize
}
```

**WGSL Implementation:**
```wgsl
fn child_offset(mask_low: u32, mask_high: u32, b: u32) -> u32 {
    if (b < 32u) {
        return countOneBits(mask_low & ((1u << b) - 1u));
    } else {
        return countOneBits(mask_low) + countOneBits(mask_high & ((1u << (b - 32u)) - 1u));
    }
}
```

---

## 4. Leaf Brick Layout (`LeafBrick64`) & In-Place Zero-Repack Mutation

### `LeafBrick64` Memory Layout (64 Bytes)
When a leaf node is not uniform, its $4 \times 4 \times 4 = 64$ voxels store individual `u8` Material IDs inside a 64-byte payload:

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LeafBrick64 {
    pub packed_materials: [u32; 16], // 16 * 4 bytes = 64 bytes (1 byte per voxel)
}
```

### Homogeneous Solid Compression
If all solid voxels in a subtree share the identical material:
1. `child_ptr` sets `IS_UNIFORM_FLAG`.
2. The `u8` material ID is packed directly into `lod_payload`.
3. `child_ptr` base index is set to `0`.
4. **Memory footprint:** 0 bytes in `BrickPool` and 0 child nodes allocated.

### In-Place Leaf Brick Mutation Engine
Located in `crates/arcadia-core/src/chunk_tree/edit/`. Sculpting operations (carving, placing, painting, clearing) modify the tree in-place without converting to recursive `BuildNode` trees or triggering BFS repacks during gameplay:

1. **Zero-Repack Sculpting:** Carve and paint operations directly mutate existing `self.nodes` and `self.bricks` vectors.
2. **Lazy Branch Expansion (`uncollapse_if_needed` & `ensure_child`):**
   - When editing a uniform solid node, the branch uncollapses on demand into explicit children.
   - If a child node does not yet exist, a new contiguous sibling slice of size $\text{old\_count} + 1$ is appended to the pool.
   - Existing siblings are copied in `POPCNT` rank order, and the parent's base index and bitmask are updated.
3. **Leaf Mutation (`mutate_brick_in_place`):**
   - **Clearing to Air:** Uniform leaf node clears the corresponding bit in `Node64` bitmask (0 bytes allocated).
   - **Uniform to Mixed:** When setting a different material inside a uniform leaf, a `LeafBrick64` is lazily allocated in `BrickPool`, initialized with the uniform material, and mutated.
   - **Mixed Leaf:** Directly updates `self.bricks[brick_idx].set_material(b, mat)` in-place.
4. **Tree Pruning & In-Place Sibling Compaction (`remove_child`):**
   - When all bits in a leaf become zero (`mask == 0`), the node is pruned.
   - Sibling compaction shifts remaining sibling nodes within their contiguous allocation (`nodes[old_base + rank .. old_base + new_count]`) and zeroes the trailing slot without reallocating the vector.
   - Pruning cascades recursively up to Level 0 if all children of a parent become empty.
5. **Fine-Grained Staging (`ChunkEditReport`):**
   - Tracks bounded ranges `[min_dirty_node..=max_dirty_node]` and `[min_dirty_brick..=max_dirty_brick]`.
   - Used by `WorldMap::sync_dirty_chunk_partial` to stream localized GPU buffer sub-allocations via `wgpu::Queue::write_buffer` in sub-milliseconds.
6. **Compaction Pass (`ChunkTree64::compact()`):**
   - Garbage-collects orphaned sibling slices resulting from extensive destructive carving, restoring contiguous level-order BFS packing during idle frames or chunk serialization.

---

## 5. Breadth-First (BFS / Level-Order) Node Packing within Chunks

During initial generation or complete compaction (`ChunkTree64::pack_build_tree` in `crates/arcadia-core/src/chunk_tree/builder.rs`), nodes are laid out in strict Level-Order (BFS):

```text
[Slot 0: Chunk Root (Level 0, 16m)]
[Slots 1..K1: Level 1 Macro Nodes (4m) - contiguous]
[Slots K1..K2: Level 2 Meso Nodes (1m) - contiguous]
[Slots K2..N: Level 3 Leaf Nodes (0.25m) - contiguous]
```

### Cache & Traversal Benefits
- **L1/L2 Cache Locality:** Coarse ray march queries and LOD tests stay hot in GPU cache lines.
- **Contiguous Sibling Slices:** Sibling nodes remain strictly adjacent (`nodes[base_ptr .. base_ptr + count]`), ensuring single-cycle hardware `POPCNT` addressing is 100% valid.
- **Predictable Streaming:** Memory transfers to GPU VRAM are packed sequential writes.

---

## 6. 3D Morton (Z-Order) Space-Filling Chunk Ordering & Dynamic Registration

### Morton (Z-Order) Space-Filling Curve
In `WorldMap::rebuild_all_gpu_pools` (`crates/arcadia-world/src/world_map.rs`), chunks are packed into global GPU storage buffers (`gpu_nodes`, `gpu_bricks`) ordered along a 3D Morton curve:

$$\operatorname{morton3d}(x, y, z) = \operatorname{spread}(x) \mid (\operatorname{spread}(y) \ll 1) \mid (\operatorname{spread}(z) \ll 2)$$

Where `spread(v)` interleaves bits of integer coordinates.

- **Benefits:** Maximizes 3D spatial coherence in 1D GPU buffer addresses. Rays marching across chunk boundaries frequently hit neighboring chunks already cached in GPU L2 texture/buffer cache.

### Dynamic Tree Registration Bridge (`WorldMap::register_dynamic_tree`)
Dynamic physics bodies (`RigidVoxelBody`) own their local `ChunkTree64` trees. To allow the GPU raymarcher to render dynamic bodies with full LOD:
- Dynamic bodies register their local trees into global GPU pools.
- Reuses freed spans from destroyed chunks or appends to the pool tail.
- Emits a `GpuPatch` tagged with sentinel `chunk_flat_idx == EMPTY_CHUNK_SENTINEL` (`0xFFFF_FFFF`), ensuring GPU storage buffer upload without corrupting static 3D world clipmap roots.

---

## Related Specifications & Implementation Files
- **Rendering & Raymarching:** [02_rendering_and_lighting.md](02_rendering_and_lighting.md)
- **Physics Compound Colliders:** [03_physics_and_destruction.md](03_physics_and_destruction.md)
- **Master Entry Point:** [ARCHITECTURE.md](../../ARCHITECTURE.md)
- **Source Code:**
  - Bitwise coordinates: [`crates/arcadia-core/src/coord.rs`](../../crates/arcadia-core/src/coord.rs)
  - Node layout: [`crates/arcadia-core/src/node64.rs`](../../crates/arcadia-core/src/node64.rs)
  - Chunk Tree: [`crates/arcadia-core/src/chunk_tree.rs`](../../crates/arcadia-core/src/chunk_tree.rs)
  - Mutation Engine: [`crates/arcadia-core/src/chunk_tree/edit/`](../../crates/arcadia-core/src/chunk_tree/edit/)
  - World Map & Morton Packing: [`crates/arcadia-world/src/world_map.rs`](../../crates/arcadia-world/src/world_map.rs)

