# 06. Atomontage: Volumetric Matter & Streaming

## 1. Overview & Core Philosophy

Developed by **Branislav Siles** and Atomontage Inc. across more than a decade of research originating in the demoscene, **Atomontage** is a proprietary micro-voxel graphics and simulation platform designed to abandon triangle-mesh surfaces in favor of solid, volumetric geometry ("Virtual Matter"):
- **The Core Thesis:** Traditional 3D graphics represent objects as hollow surface boundaries composed of thin 2D polygon shells. Atomontage treats 3D objects as solid, atom-like volumetric matter with continuous internal material properties (density, color, elasticity, electrical conductivity).
- **Scale & Detail:** Renders scenes containing hundreds of billions to trillions of microscopic voxels (sub-millimeter resolution) running interactively on consumer laptops and mobile devices.
- **Key Breakthroughs:** Extreme geometry compression (AVF format), out-of-core cloud streaming, real-time boolean micro-sculpting, and seamless hybrid raster-voxel composition.

---

## 2. Spatial Data Representation & Extreme Compression

A naive grid of $100\text{ billion}$ voxels at 1 byte per voxel would require $100\text{ GB}$ of uncompressed memory. Atomontage makes this interactive through a proprietary multi-resolution data structure:

### 2.1 Multi-Resolution Sparse Voxel Hierarchies
- **Spatial Subdivision:** Combines Sparse Voxel Octrees (SVO) with bounded Directed Acyclic Graph (DAG) pointer deduplication.
- **Homogeneous Node Folding:** Identical solid or empty cubic blocks are deduplicated across the hierarchy.
- **Atomontage Volumetric Format (AVF):**
  - A compressed volumetric file format that compresses raw voxel models by **$90\%\text{ to }99\%$** using hierarchical run-length encoding (RLE), palette quantization, and differential Morton-order bit-packing.
  - A $2\text{ GB}$ raw scan of an ancient statue or geological landscape compresses down to $20\text{--}50\text{ MB}$, fitting comfortably in mobile memory.

---

## 3. Rendering Pipeline: Ray Casting & Hybrid Composition

Atomontage does not generate triangle meshes for its voxel geometry (avoiding mesh generation bottlenecks during destruction):

### 3.1 Adaptive Screen-Space Ray Casting
- **1 Ray Per Pixel:** Traces primary rays from the camera into the hierarchical voxel tree.
- **Adaptive LOD Footprint:** Ray stepping adjusts dynamically according to camera distance and screen pixel footprint:
  - Foreground geometry resolves to individual sub-millimeter micro-voxels.
  - Distant terrain terminates early against pre-filtered parent nodes carrying volumetric mipmapped color and opacity.
- **Multi-Core CPU & GPU Execution:** Early versions ran entirely in software on multi-core CPUs via SIMD vectorization (AVX-512), proving that hierarchical ray casting can run without specialized hardware. Modern versions leverage GPU compute shaders.

### 3.2 Seamless Hybrid Mesh & Voxel Composition
A notable practical capability of Atomontage is its ability to render **traditional polygon meshes inside the voxel world**:
- Traditional rasterized characters (e.g. animated skeletal meshes) render their depth into a shared depth buffer.
- The voxel ray caster tests against the rasterized Z-buffer, allowing characters to walk behind voxel pillars, submerge into voxel water, or kick up dynamic voxel debris without depth sorting artifacts.

---

## 4. Progressive Out-of-Core Cloud Streaming

Atomontage features a client-server streaming architecture:
1. **Hierarchical Spatial Paging:** The world is partitioned into spatial volumetric blocks stored on local SSDs or remote cloud servers.
2. **View-Dependent Streaming:** As the camera moves, the client predicts the camera frustum and requests only visible high-LOD nodes.
3. **Progressive Refinement:** Distant terrain renders at coarse LOD immediately, streaming in fine micro-voxel bricks asynchronously as bandwidth permits.
4. **Interactive Remote Collaboration:** Multiple users can sculpt, dig, and build in the same persistent world across web browsers and mobile clients simultaneously.

---

## 5. Real-Time Physical Simulation & Sculpting

Because the entire world is solid volumetric matter:
- **Instant Boolean Operations:** Carving a hole, slicing an object, or drilling a tunnel clears or alters voxel bits directly in memory. There is **zero polygon recalculation, zero UV re-mapping, and zero normal smoothing stall**.
- **Volumetric Mass Properties:** Centers of mass, volume, and moments of inertia are exact discrete integrations over the occupied voxel grid.
- **Material-Driven Multi-Physics:** Voxels carry physical parameters allowing thermal propagation, granular sand flow, and electrical conductivity to simulate on the identical spatial grid.

---

## 6. Architectural Lessons for Arcadia

| Atomontage Technique | Arcadia Equivalent / Opportunity | Strategic Value for Arcadia |
|---|---|---|
| **Multi-Resolution SVO / DAG Compression** | Arcadia's **Chunked Sparse 64-Tree (CS64)** with homogeneous solid node collapsing. | Arcadia achieves $4\times$ shallower tree depth (4 levels vs 8–10 levels in SVO), cutting dependent memory latency on GPUs. |
| **Hybrid Mesh + Voxel Depth Composition** | Future character / weapon rasterization pass writing into G-buffer depth. | Allows player hands, animated weapons, and third-party skeletal meshes to interact with CS64 raymarched terrain. |
| **Progressive Spatial Chunk Streaming** | Arcadia's planned **Infinite Toroidal 3D Clipmap** ([`ROADMAP_AND_TODO.md` D1](../../docs/plans/ROADMAP_AND_TODO.md)). | Background worker thread streaming of CS64 chunk pools as player flies across procedural terrain. |
| **Micro-Voxel Boolean Sculpting** | Arcadia's **Phase 7 In-Place Leaf Brick Mutation** ([`ROADMAP_AND_TODO.md` E2](../../docs/plans/ROADMAP_AND_TODO.md)). | Bypassing full-tree repacking to perform $O(1)$ single-voxel edits and stream $64\text{ B}$ patches to GPU. |
