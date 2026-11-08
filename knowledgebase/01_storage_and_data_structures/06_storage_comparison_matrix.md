# 06. Storage Comparison & Selection Matrix

## 1. Quantitative Benchmark & Metric Comparison

The following matrix compares primary voxel data structures evaluated for real-time applications (60+ FPS, targeting $16\text{ m}$ chunks with $(1/16\text{ m})^3 = 6.25\text{ cm}$ resolution, $256^3 \approx 16.7\text{ million}$ potential voxels per chunk).

| Voxel Data Structure | Memory / Chunk (Terrain) | Random Read Latency | Dynamic Edit (Carving) | GPU Compute Ray Marching | Max World Scale |
|---|---|---|---|---|---|
| **Dense 3D Array (`u8`)** | $16.78\text{ MB}$ | $O(1)$ (1 memory read) | $O(1)$ Instant | ⭐ Poor (Massive VRAM footprint) | Tiny ($\le 128\text{ m}$) |
| **Sparse Voxel Octree (SVO, B=8)** | $80\text{--}180\text{ KB}$ | $O(\log_8 N) = 8$ reads | $O(\log_8 N)$ Node splits | ⭐⭐⭐ Good (Deep traversal latency) | Large ($>10\text{ km}$) |
| **Chunked Sparse 64-Tree (CS64)** | **$25\text{--}65\text{ KB}$** | **$O(\log_{64} N) = 4$ reads**| **$O(1)$ local mask patch**| **⭐⭐⭐⭐⭐ Exceptional (Single-cycle POPCNT)** | **Massive / Infinite** |
| **Sparse Voxel DAG (SVDAG)** | $3\text{--}12\text{ KB}$ | $O(\log_8 N) = 8$ reads | ❌ Very Slow (CoW re-hashing) | ⭐⭐⭐⭐ Very Good | Infinite (Static) |
| **NanoVDB / Hierarchical Grid** | $40\text{--}120\text{ KB}$ | $O(\text{Tree Depth}) = 3\text{--}4$ | ❌ Slow (Buffer repacking) | ⭐⭐⭐⭐ Very Good (HDDA) | Medium ($1\text{--}5\text{ km}$) |
| **Toroidal 3D Clipmap** | Bounded (e.g. $64\text{ MB}$) | $O(1)$ Modulo arithmetic | $O(1)$ Instant | ⭐⭐⭐⭐⭐ Fast (Uniform texture DDA) | Camera-relative |
| **GPU Spatial Hash Map** | $50\text{--}150\text{ KB}$ (Sparse) | $O(1)$ Cuckoo probe (2–3 reads)| $O(1)$ Atomic write | ⭐⭐⭐ Moderate (Hash jumps hurt DDA)| Sparse dynamic |

---

## 2. In-Depth Trade-Off Analysis

### 2.1 Traversal Latency vs Tree Depth
On modern GPUs, memory latency is the single greatest bottleneck:
- Each pointer dereference costs $200\text{--}400\text{ clock cycles}$ if the cache line is missed.
- An 8-level SVO traversing empty space near surfaces can stall warps up to 8 times per ray.
- A **4-level CS64 tree** cuts pointer dereferences in half. Because the first 2 levels (Root and Macro nodes) fit in the GPU L1 texture cache, **over 80% of rays resolve within 1 or 2 memory transactions**.

### 2.2 The Dynamic Editing Dilemma
If your game requires real-time destruction (explosions, digging, building, structural collapse):
- **Avoid Global SVDAGs:** Modifying one node requires re-hashing the DAG or branching into expensive Copy-on-Write trees.
- **Avoid Dense Linear Buffers:** Pushing $16\text{ MB}$ per edited chunk across PCIe chokes the host-to-device bus ($16\text{ GB/s}$ max bandwidth allows only a few dozen updates per frame).
- **Choose Chunked Tree + Leaf Bricks (CS64):**
  - Carving a sphere only modifies the 64-bit occupancy mask of intersected leaf bricks.
  - Only dirty chunks re-upload their compact node/brick buffers (typically $2\text{--}8\text{ KB}$ transferred via staging buffers).

---

## 3. Recommended Architectural Configurations by Engine Type

### 3.1 Fully Destructible Micro-Voxel World (Arcadia / Teardown Style)
- **Primary Structure:** Toroidal Chunk Grid + Chunked Sparse 64-Tree (CS64).
- **Leaf Representation:** $4 \times 4 \times 4$ voxel Leaf Bricks holding 1-byte material IDs.
- **Optimization:** Breadth-first node packing, Morton 3D chunk addressing, homogeneous subtree collapsing.

### 3.2 Massive Static Architectural / City Showcase (Nanite / Euclideon Style)
- **Primary Structure:** Clustered Sparse Voxel DAG (SVDAG) with separated attribute trees.
- **Leaf Representation:** 1-bit geometric occupancy with geometry deduplication.
- **Optimization:** 8-way cubic symmetry exploitation, relative offset pointer compression.

### 3.3 Dynamic Particle / Fluid / Debris Simulation
- **Primary Structure:** GPU Spatial Hash Table (Cuckoo hashing) or 3D Toroidal Dense Simulation Grid.
- **Optimization:** Atomic linked-list insertion per spatial bucket.
