# 05. RLE, Bitpacking & Palette Compression

## 1. Overview & Compression Hierarchy

In a real-time voxel engine, voxel data exists across three distinct performance tiers:
1. **GPU L1/L2 Execution Tier:** Uncompressed or hardware-bitmasked format (Node64, single-cycle `POPCNT`, direct byte addressing) for nanosecond ray marching.
2. **Host RAM & Streaming Staging Tier:** Paletted, slice-run-length compressed, or block-quantized format for streaming chunks between disk, network, and GPU staging buffers.
3. **Cold Storage / Network Serialization Tier:** Entropy-coded (LZ4, Zstandard) chunk archives.

---

## 2. Paletted Voxel Storage

### 2.1 The Entropy of Voxel Materials
While an engine might support thousands of unique block types (different types of stone, dirt, vegetation, minerals, metals, glass), any individual $16\text{ m}^3$ chunk rarely contains more than 4 to 16 distinct materials.

Storing a full 16-bit or 32-bit material ID per voxel consumes $2\text{--}4\text{ MB}$ per chunk.

### 2.2 Global vs Local Chunk Palettes
1. **Global Material Palette:** A fixed array of 256 PBR material definitions (albedo, roughness, metallic, emission, subsurface) stored in a shared GPU uniform or storage buffer (consuming only $256 \times 32\text{ bytes} = 8\text{ KB}$).
2. **Local Chunk Palette:**
   - Each chunk stores a tiny array of the materials actually present (e.g. $[2, 14, 55, 102]$).
   - If a chunk contains $\le 2$ materials: 1 bit per voxel.
   - If $\le 4$ materials: 2 bits per voxel.
   - If $\le 16$ materials: 4 bits per voxel.
   - If $\le 256$ materials: 8 bits per voxel.

```text
Chunk Local Material Table: [ 0: Air, 1: Granite, 2: MossyStone, 3: GoldVein ]
Bits per Voxel: 2 bits
Voxel Array (u64 packed): [ 01 01 01 10 10 00 11 01 ... ]
```

**Memory Reduction:** Packing 8-bit material indices down to 2-bit or 4-bit indices cuts brick memory by **$50\%\text{--}75\%$**.

---

## 3. Run-Length Encoding (RLE) Variants for Voxels

### 3.1 1D Linear RLE
Replaces runs of identical voxels with a `(count, value)` pair:
- Highly efficient for solid bedrock or empty air.
- Catastrophic for high-frequency foliage or noisy terrain (a run of length 1 requires 2 bytes instead of 1, inflating memory by 100%).

### 3.2 2D Slice RLE & 1D Column RLE
Terrain generation is anisotropic: vertical columns of voxels exhibit extreme homogeneity (e.g., 50 voxels of deep bedrock, followed by 10 voxels of dirt, 1 voxel of grass, and 67 voxels of air).
- **Column RLE:** Compress along the Y-axis (vertical height).
- Storing `(height_span, material_id)` allows instant vertical heightfield evaluation and ray marching skips.

### 3.3 Bitmask-RLE Hybrids
Instead of storing counts, store a 64-bit occupancy bitmask followed by a compact array of non-empty values:
$$\text{Memory} = 8\text{ bytes (Mask)} + \text{popcnt}(\text{Mask}) \times \text{sizeof}(\text{Value})$$
This combines $O(1)$ random access (via hardware `POPCNT`) with zero storage overhead for empty cells.

---

## 4. Hardware-Accelerated Bitpacking

Modern CPUs (AVX2, AVX-512, ARM NEON) and GPUs (WGSL compute) provide high-throughput bit-manipulation instructions:
- `PEXT` / `PDEP` (Parallel Bit Extract / Deposit): Extract arbitrary bitfield widths across 64-bit words in 1 clock cycle.
- `bitfieldExtract` / `unpack4x8unorm` in shaders: Decompress packed 8-bit attributes directly inside shader registers without memory lookups.

---

## 5. Streaming & Disk Serialization Pipeline

For streaming infinite worlds over disk or network:
```text
[ Raw Voxel Chunk (16MB) ]
        │
        ▼ (Homogeneous Node Collapsing & Palette Extraction)
[ Sparse CS64 Chunk (35KB) ]
        │
        ▼ (Delta Encoding of Pointers + Bit-Packed Masks)
[ Staging Buffer (22KB) ]
        │
        ▼ (LZ4 / Zstandard Block Compression)
[ Serialized Network Packet (6–9 KB) ]
```

At $8\text{ KB}$ per $16\text{ m}^3$ chunk, a player flying at high speeds ($50\text{ m/s}$) consumes less than **$1.5\text{ MB/s}$ of disk/network bandwidth**, well within standard limits.
