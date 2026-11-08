# 02. Sparse Voxel DAGs (SVDAG) & Deduplication

## 1. Overview & Motivation

While a Sparse Voxel Octree (SVO) or 64-Tree eliminates empty space, identical solid structures (e.g., repeating architectural pillars, foliage patterns, identical terrain layers, or mirrored geometric features) are still duplicated across memory.

A **Sparse Voxel Directed Acyclic Graph (SVDAG)** merges identical subtrees across the entire world into shared nodes. A node in an SVDAG can have multiple parents pointing to it, transforming the tree into a directed acyclic graph.

```text
       [ Tree: Redundant ]                     [ DAG: Deduplicated ]
            [ Root ]                                  [ Root ]
           /        \                                /        \
     [ Subtree A ]  [ Subtree A ]                   \          /
       (Copy 1)       (Copy 2)                       [ Subtree A ] (Shared)
```

By deduplicating identical subtrees, SVDAGs achieve unprecedented geometric compression ratios—often representing **billions of voxels in dozens of megabytes**.

---

## 2. Theoretical Deduplication: How SVDAGs Work

### 2.1 Bottom-Up Hashing & Canonical Node Pools
Building an SVDAG requires a bottom-up hash consolidation:
1. **Level $K$ (Leaves):** Hash the bitmask of every leaf node. Insert into a hash set. Replace duplicate leaf nodes with a reference to the canonical leaf node.
2. **Level $K-1$ (Parents):** A parent node's identity is defined by its child validity mask and the **canonical memory addresses** of its children. Hash this combination:
   $$\text{Hash}(Node) = \text{XXH3}(\text{mask} \parallel \text{child\_ptr}_0 \parallel \dots \parallel \text{child\_ptr}_m)$$
3. **Consolidation:** If an identical parent already exists in the pool, point the grandparent to the existing canonical node and discard the duplicate.
4. Repeat upward until the root is reached.

### 2.2 SVDAG Compression Ratios
In urban environments, repetitive architectural structures (such as European cathedrals, Minecraft builds, or modular spaceship interiors), SVDAGs achieve **compression ratios between 10:1 and 100:1** over standard SVOs.
- *Kämpe et al. (High-Resolution Sparse Voxel DAGs, 2013)* demonstrated an entire $131,072^3$ voxel Minecraft world ($2.25$ trillion voxels) encoded in **$1.14\text{ GB}$ of RAM** ($0.004$ bits per voxel).

---

## 3. The Challenge of Attributes: Color & Material DAGs

The fundamental limitation of pure geometric SVDAGs is that **materials and colors break symmetry**:
- Two medieval towers may have identical geometry, but one is mossy cobblestone and the other is dry brick.
- In a pure SVDAG, if the material differs, the entire subtree must diverge, collapsing the deduplication ratio.

### 3.1 Separated Color Trees
To maintain geometric compression while supporting materials, modern architectures decouple geometry from attributes:
1. **Geometry SVDAG:** Encodes 1-bit occupancy geometry with maximum deduplication.
2. **Attribute Tree / Attribute Grid:** Stores material IDs, colors, and normals.
3. **Weight-Indexed Mapping:** A compact index table maps geometry leaf traversal paths to attribute entries.

### 3.2 Clustered DAG of Trees (Dolonius et al., 2017)
Instead of a single global DAG, partition the world into a coarse grid of chunks (e.g., $16\text{ m}^3$). Each chunk contains its own local DAG, with a global dictionary for shared high-frequency detail blocks. This allows local dynamic editing without having to re-hash the entire universe.

---

## 4. Symmetry & Reflection Exploitation

Standard DAG deduplication only detects identical translation-invariant subtrees. However, 3D worlds are rich in **reflections and rotations** (e.g., 8-way cubic symmetry):
- By storing a 3-bit transformation code (axis reflections $X, Y, Z$) in the parent-to-child pointer, a single corner node can represent all 4 corners of a building, or all 8 corners of a cube.
- **Symmetry-Aware DAGs** increase compression by an additional **$2\times\text{--}4\times$** at the cost of a 3-bit transformation transform during ray traversal.

---

## 5. Pointerless & Offset-Encoded DAG Formats

Classic pointers are 32-bit or 64-bit absolute addresses. To pack DAGs for GPU caches:
1. **Relative Node Offsets:** Since parents tend to reference children placed nearby in the buffer during bottom-up allocation, store relative signed offsets (`offset = child_addr - current_addr`).
2. **Variable-Length Node Encoding:** High-frequency small nodes use 8-bit or 16-bit relative offsets; rare deep jumps use a full 32-bit escape word.

---

## 6. Real-Time Dynamic Modification: The Achilles' Heel

While SVDAGs are unbeatable for static scenes, **real-time world destruction (carving/placing voxels) is extremely expensive in a monolithic DAG**:
- Modifying a single voxel in a shared subtree modifies that node.
- Modifying that node invalidates the hashes of all its parents, grandparents, and every shared instance across the entire world!
- Without careful handling, modifying one ruin arch would inadvertently carve holes in all duplicate ruin arches across the map.

### 6.1 Real-Time Hybrid Solutions for Dynamic Games
1. **Copy-on-Write (CoW) Unsharing:**
   - When a voxel edit occurs, traverse down the DAG.
   - Clone the modified path into unique nodes (turning the DAG locally back into a unique tree).
   - Apply the edit to the cloned path.
   - Run asynchronous background garbage collection and re-deduplication.
2. **Chunked Local DAGs:**
   - Keep DAG deduplication local to individual $16\text{ m}$ chunks.
   - Editing a chunk only dirties that single chunk's local node buffer.

---

## 7. Real-Time Suitability Verdict

| Criteria | Score (1–5) | Analysis |
|---|---|---|
| **Static Geometry Compression** | ⭐⭐⭐⭐⭐ (5/5) | Unrivaled. Billions of micro-voxels fit into standard GPU VRAM. |
| **GPU Ray Traversal Performance**| ⭐⭐⭐⭐ (4/5) | Fast (same traversal step count as SVO), but pointer redirection can slightly reduce cache locality. |
| **Material / Color Support** | ⭐⭐⭐ (3/5) | Requires attribute decoupling or color DAGs, which reduces deduplication efficiency. |
| **Dynamic Carving / Destruction**| ⭐⭐ (2/5) | Copy-on-Write is required. Re-hashing the DAG in real time at 60 FPS is prohibitive for large edits. |
| **Streaming & Network Bandwidth**| ⭐⭐⭐⭐⭐ (5/5) | Outstanding for transmitting large worlds over slow networks. |

> **Conclusion for Arcadia:** SVDAGs are optimal for static procedural asset libraries (e.g. pre-baking detailed ruin prefabs, giant statues, and foliage models into tiny memory footprints). For dynamic, destructible terrain, a **Chunked Sparse 64-Tree (CS64)** with homogeneous subtree collapsing provides significantly superior dynamic editing performance while maintaining near-zero CPU/GPU synchronization overhead.
