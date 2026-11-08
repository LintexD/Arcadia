# 01. Sparse Voxel Octrees (SVO) & N-ary Trees

## 1. Overview & Theoretical Motivation

A naive 3D voxel array covering a $256 \times 256 \times 256$ chunk requires $16,777,216$ cells. At 1 byte per voxel (material ID), that is $16\text{ MB}$ per $16\text{ m}$ chunk. For an open-world scene spanning $1\text{ km} \times 1\text{ km} \times 128\text{ m}$, a dense grid would demand over **$500\text{ GB}$ of VRAM**.

Real-world environments exhibit vast empty spaces (air) and large uniform volumes (subterranean bedrock, ocean water). Hierarchical tree structures exploit this spatial coherence by replacing empty or uniform regions with single coarse descriptor nodes.

---

## 2. Sparse Voxel Octrees (SVO)

### 2.1 The Traditional SVO
In an Octree, each internal node subdivides a 3D bounding box into $2 \times 2 \times 2 = 8$ octants:
- **Branching Factor:** $B = 8$.
- **Tree Depth for $N^3$ volume:** $\log_2(N)$. For $256^3$, depth is $8$.
- **Node Descriptor:** 
  - An 8-bit child validity mask indicating which octants are non-empty.
  - A 32-bit base child pointer indexing into an array of child nodes.

```text
       [ Root (Depth 0) ]
        /   |   |   \  ... (8 octants)
    [Child 0] [Child 1] ... [Child 7] (Depth 1)
       / ... \
   [Leaf Bricks] (Depth 8)
```

### 2.2 Memory Alignment & Cache Bottlenecks in Classic SVOs
Classic SVOs suffer from pointer chasing:
1. **Divergent Ray Stepping:** Each level of depth requires reading a node from memory, checking the child bitmask, and issuing an indirect memory fetch to `base_ptr + popcnt(mask, child_index)`.
2. **Deep Trees Mean High Memory Latency:** Traversing an 8-level SVO requires up to 8 dependent memory lookups per ray, stalling GPU warps on VRAM latency ($200\text{--}400\text{ cycles}$).

---

## 3. Higher-Order N-ary Trees: The Chunked Sparse 64-Tree (CS64)

To reduce tree depth and minimize dependent memory accesses, modern real-time engines utilize higher branching factors:
- **$B = 64$ ($4 \times 4 \times 4$ subdivision):** Tree depth drops by half compared to an octree ($\log_{64}(N) = \frac{1}{2}\log_8(N) = \frac{1}{6}\log_2(N)$).
- A $256^3$ voxel chunk is fully resolved in **exactly 4 levels** (Level 0: $256^3$, Level 1: $64^3$, Level 2: $16^3$, Level 3: $4^3$ leaf brick).

### 3.1 Node64 Bitmask Math & Hardware `POPCNT`
A 64-slot subdivision maps onto an exact 64-bit integer (`u64`), perfectly matching 64-bit CPU registers and single-cycle GPU hardware intrinsics (`count_ones` in Rust / `countOneBits` in WGSL/SPIR-V).

#### Local Coordinate to Bit Index
For $x, y, z \in \{0, 1, 2, 3\}$:
$$\text{bit\_index}(x, y, z) = (z \ll 4) \mid (y \ll 2) \mid x$$

$$\text{bit\_mask}(x, y, z) = 1\text{ULL} \ll \text{bit\_index}(x, y, z)$$

```rust
#[inline(always)]
pub const fn child_bit_index(x: u32, y: u32, z: u32) -> u32 {
    (z << 4) | (y << 2) | x
}

#[inline(always)]
pub const fn child_bit_mask(x: u32, y: u32, z: u32) -> u64 {
    1u64 << child_bit_index(x, y, z)
}
```

#### Child Offset Indexing via Population Count
Instead of allocating 64 pointers per node (which would consume $64 \times 4 = 256\text{ bytes}$ per node!), the node stores:
1. A 64-bit occupancy mask (`mask_low: u32`, `mask_high: u32`).
2. A single 32-bit `child_base_offset` pointer.

To find the array offset of active child $k \in [0, 63]$:
$$\text{active\_slot}(k) = \text{popcnt}(\text{mask} \ \& \ ((1\text{ULL} \ll k) - 1))$$
$$\text{child\_address} = \text{child\_base\_offset} + \text{active\_slot}(k)$$

On modern GPUs (NVIDIA Ampere/Ada, AMD RDNA2/3), `countOneBits` compiles to a single hardware instruction executed in **1 clock cycle** with zero branching.

### 3.2 16-Byte Compact Node64 Memory Layout

```text
┌──────────────────────────────────────────────────────────────┐
│                      Node64 (16 Bytes)                       │
├───────────────────────────────┬──────────────────────────────┤
│ Offset 0..3:  mask_low  (u32) │ Offset 4..7:  mask_high (u32)│
├───────────────────────────────┼──────────────────────────────┤
│ Offset 8..11: child_base (u32)│ Offset 12..15: meta/mat (u32)│
└───────────────────────────────┴──────────────────────────────┘
```

The 32-bit metadata/material field contains flags:
- `IS_LEAF_FLAG (1 << 31)`: Child offset points into the `BrickPool` (material array) rather than `NodePool`.
- `IS_UNIFORM_FLAG (1 << 30)`: The entire node is completely solid with a single material; no child nodes or bricks are allocated.
- `MATERIAL_ID (bits 0..7)`: The dominant material ID for LOD rendering or uniform solid representation.

---

## 4. Homogeneous Subtree Collapsing

### 4.1 The Mechanism
When generating or modifying the world:
1. If all 64 children of a node are empty (`mask == 0`), the parent clears its bit for this child.
2. If all 64 children are identical solid materials with 100% occupancy:
   - The node sets `IS_UNIFORM_FLAG | material_id`.
   - Its `child_base` is set to `0`.
   - **Zero child nodes or leaf bricks are allocated**, saving memory and terminating ray marching instantly at Level 1 or Level 2!

### 4.2 Impact on World Memory
In an open-world terrain, over 70% of underground volume consists of solid continuous stone, bedrock, or water. Subtree collapsing reduces the required node count by **85% to 92%** compared to a raw voxel tree.

---

## 5. Memory Locality: Breadth-First (BFS) vs Depth-First (DFS) Layout

### 5.1 The DFS Problem on GPUs
When a tree is serialized via standard Depth-First Search (DFS) recursion, a parent node is located far away from its siblings in the linear buffer. When a group of neighboring threads in a GPU warp step through adjacent octants, each thread accesses a completely different cache line, causing memory divergence and cache thrashing.

### 5.2 Breadth-First (BFS / Level-Order) Packing
In a BFS chunk layout:
- The Chunk Root (Level 0) is at index 0.
- All Level 1 nodes are packed contiguously in indices `1 .. K_1`.
- All Level 2 nodes follow contiguously in indices `(K_1 + 1) .. K_2`.

```text
Buffer: [ L0 Root (16B) ] [ L1 Nodes (Contiguous) ] [ L2 Nodes (Contiguous) ] [ Leaf Bricks ]
Cache:  |--- Line 0 ----| |------ Lines 1..N ------| |------ Lines N..M ------|
```

**Benefits:**
1. **L1/L2 Cache Hit Rate:** During top-down ray marching, all rays in a $8 \times 8$ screen tile traverse Level 0 and Level 1 simultaneously. If Level 0 and Level 1 are contiguous, they remain permanently pinned in the GPU L1 texture cache.
2. **Predictable Address Arithmetic:** Offsets between sibling nodes are strictly consecutive (`slot + i`).

---

## 6. Real-Time Performance & Resource Budgets

| Metric | Classic SVO (B=8) | CS64 Tree (B=64) |
|---|---|---|
| **Max Depth for $256^3$** | 8 | 4 |
| **Dependent Memory Reads / Ray** | Up to 8 | Up to 4 |
| **Node Size** | 8–16 Bytes | 16 Bytes |
| **Child Indexing Math** | Bit shift + `popcnt` | Dual `u32` mask + `countOneBits` |
| **Memory for $16\text{m}^3$ Chunk (Terrain)** | $80\text{--}180\text{ KB}$ | **$25\text{--}65\text{ KB}$** |
| **GPU L1 Cache Hit Rate (Primary Rays)** | ~62% | **~91%** |
| **Average Ray Traversal Steps** | 35–65 | **12–22** |

---

## 7. Implementation Best Practices & Gotchas

1. **Avoid 64-bit integer types in older WGSL / SPIR-V targets:** Many mobile GPUs and older Vulkan 1.1 drivers lack native `u64` shader support (`shaderInt64`). Splitting the mask into two `u32` words (`mask_low` for bits $0..31$, `mask_high` for bits $32..63$) provides 100% hardware compatibility across all Vulkan, Metal, and DirectX 12 hardware.
2. **64-bit Bit Shift Overflow:** In Rust and WGSL, shifting a 32-bit or 64-bit integer by $\ge 64$ bits is undefined behavior or wraps around. Ensure $k < 64$ before shifting `1u64 << k`.
3. **Ray-Cone Footprint LOD:** Internal nodes must store pre-filtered volumetric attributes (color, roughness, emissive intensity) so distant rays can terminate at Level 1 or Level 2 without artifact-inducing popping.
