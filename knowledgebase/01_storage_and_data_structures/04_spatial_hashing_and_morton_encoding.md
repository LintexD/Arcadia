# 04. Spatial Hashing & Morton / Hilbert Locality

## 1. Overview & Motivation

When organizing billions of potential 3D voxels in memory, linear coordinate indexing (`index = x + y * WIDTH + z * WIDTH * HEIGHT`) creates catastrophic cache locality problems for modern GPUs:
- Adjacent voxels along the Z-axis or Y-axis are separated by millions of bytes in memory.
- When an $8 \times 8$ pixel GPU workgroup traces rays into a 3D scene, nearby rays access coordinates that map to distant cache lines, causing **VRAM cache thrashing** and serializing memory controllers.

To solve this, modern voxel engines employ two critical mathematical tools:
1. **Space-Filling Curves (Morton Z-Order & Hilbert Curves)** to enforce 3D spatial locality in 1D linear memory buffers.
2. **GPU Spatial Hash Tables** for unbounded sparse coordinate lookups.

---

## 2. Space-Filling Curves: 3D Morton (Z-Order) Encoding

### 2.1 The Bit-Interleaving Principle
The 3D Morton curve (Z-order curve) interleaves the binary bits of the coordinate components $(x, y, z)$ into a single integer $M$:
$$x = \dots x_2 x_1 x_0, \quad y = \dots y_2 y_1 y_0, \quad z = \dots z_2 z_1 z_0$$
$$M(x, y, z) = \dots z_2 y_2 x_2 \; z_1 y_1 x_1 \; z_0 y_0 x_0$$

```text
3D Coordinates: (x, y, z)
  x:  0  0  0  0  x3  x2  x1  x0
  y:  0  0  0  0  y3  y2  y1  y0
  z:  0  0  0  0  z3  z2  z1  z0
Interleaved Morton Index (32-bit):
  M:  0  0  0  0  ...  z2 y2 x2  z1 y1 x1  z0 y0 x0
```

### 2.2 Why Morton Order Maximizes GPU L1/L2 Cache Hits
When chunks or voxel bricks are stored in 1D memory sorted by their 3D Morton code:
- Voxels that are close together in 3D Euclidean space are **guaranteed to be close together in linear 1D memory**.
- A $2 \times 2 \times 2$ block of adjacent chunks maps to 8 consecutive entries in the buffer!
- Cache lines fetched by one GPU thread in a warp automatically contain the data needed by neighboring threads in the same warp.

### 2.3 Fast Bit-Twiddling Implementation
Using 64-bit integer bit shifts and masks (or the x86 `PDEP` / `PEXT` instructions where available):

```rust
#[inline(always)]
pub fn part1by2_u32(mut n: u32) -> u64 {
    n &= 0x000003ff; // Mask to 10 bits (fits in 30-bit Morton code)
    let mut x = n as u64;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8))  & 0x0300f00f;
    x = (x | (x << 4))  & 0x030c30c3;
    x = (x | (x << 2))  & 0x09249249;
    x
}

#[inline(always)]
pub fn morton3d_u32(x: u32, y: u32, z: u32) -> u64 {
    (part1by2_u32(z) << 2) | (part1by2_u32(y) << 1) | part1by2_u32(x)
}
```

In WGSL compute shaders, Morton encoding of local chunk coordinates can be implemented with compact shift masks or direct bitwise lookups.

---

## 3. Hilbert Curves: Superior Locality at Higher Compute Cost

The **Hilbert Curve** provides even better locality than the Morton curve because it never makes long diagonal jumps across space; every step along the 1D Hilbert index corresponds to an adjacent neighbor in 3D.
- **Locality:** Hilbert achieves ~10–15% fewer cache line misses than Morton in extreme traversal patterns.
- **Drawback:** Encoding and decoding 3D Hilbert indices requires iterative state-machine transformations or recursive matrix rotations, making it significantly slower to evaluate on the GPU than Morton interleaving.
- **Practical Real-Time Rule:** Use **Morton (Z-order)** for GPU runtime structures; reserve Hilbert curves for pre-baked offline spatial indices.

---

## 4. Sparse GPU Spatial Hash Tables

When an environment is sparse and unbounded (e.g. flying particles, disconnected dynamic rigid bodies, or sparse voxel debris), storing chunks in an octree or grid introduces memory overhead. A **GPU Spatial Hash Table** stores only active coordinates.

### 4.1 Large Prime Coordinate Hashing
Given integer cell coordinates $(x, y, z) \in \mathbb{Z}^3$:
$$\text{hash}(x, y, z) = ((x \cdot 73856093) \oplus (y \cdot 19349663) \oplus (z \cdot 83492791)) \pmod{\text{TABLE\_SIZE}}$$

### 4.2 Handling Collisions on the GPU: Cuckoo Hashing vs Robin Hood Hashing
1. **Linear Probing with Open Addressing:** Simple, but degrades rapidly above 60% load factor.
2. **Cuckoo Hashing:** Uses 2 or 3 distinct hash functions and disjoint tables. Lookups are strictly $O(1)$ (worst case 2 or 3 memory reads), which is ideal for branchless GPU compute shaders.
3. **Compact Spatial Hash Maps (Nießner et al., Real-Time 3D Reconstruction):**
   - Stores a flat bucket array of size $M$.
   - Each bucket holds 8 entries plus a linked list pointer for overflow.
   - Buckets map directly to $8^3$ voxel bricks.

---

## 5. Performance Comparison

| Technique | Random Read Latency | GPU Branch Divergence | Dynamic Insertion Cost | Cache Hit Efficiency |
|---|---|---|---|---|
| **Linear Grid (`x+y*W+z*W*H`)** | 1 clock (flat arithmetic) | Zero | Instant | ⭐ Poor (High cache thrashing) |
| **Morton 3D (Z-Order) Buffer** | 3–6 clocks (bit-twiddle) | Zero | Instant | ⭐⭐⭐⭐⭐ Outstanding |
| **GPU Hash Table (Cuckoo)** | 20–40 clocks (indirection) | Moderate (probe loop) | Medium (atomic swap) | ⭐⭐⭐ Good (Active cells only) |
| **Hierarchical Octree Index** | 60–180 clocks (dependent reads)| High | High (Tree node alloc) | ⭐⭐⭐⭐ Very Good |

> **Arcadia Architecture Note:** Arcadia organizes its global GPU chunk buffers (`gpu_nodes`, `gpu_bricks`) along a 3D Morton space-filling curve. When rays trace across chunk boundaries, memory fetches land in contiguous GPU cache lines, eliminating VRAM bandwidth choke points.
