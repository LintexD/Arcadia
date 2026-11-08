# Arcadia Voxel Engine — Session 1 Engineering Report

**Date:** October 25, 2026  
**Status:** Completed & Pre-Built (`target/debug/arcadia.exe`)  
**Target Architecture:** Rust 2024, `wgpu` 23.0 (Vulkan 1.3), WGSL Compute Shaders, Glam 0.29, Winit 0.30, Rayon 1.10  

---

## 1. Executive Summary

In a single intensive engineering session, the Arcadia Voxel Engine was designed, implemented, debugged, and optimized from a blank repository into a high-performance, fully playable micro-voxel sandbox:
- **Scale:** $(1/16\text{ m})^3 = 6.25\text{ cm}$ voxels ($4,096\text{ voxels/m}^3$).
- **World Volume:** $160\text{ m} \times 64\text{ m} \times 160\text{ m}$ ($10 \times 4 \times 10 = 400$ chunks, $2,560 \times 1,024 \times 2,560 \approx 6.71\text{ billion}$ potential micro-voxels).
- **Renderer:** Pure GPU compute ray marcher via WGSL compute shaders (zero polygon meshes, zero vertex buffers), featuring ReSTIR GI (spatio-temporal reservoir reuse), 2-regime voxel edge-stopping wavelet denoiser, and Halton sub-pixel DLAA/TSR.
- **Interactivity:** Full kinematic first-person player physics (walking, swimming, auto-step-up, jump, double-tap `Space` / `F` flight mode), real-time destructive carving and placement with brush scaling, 16k GPU particle debris, day/night cycles, and dynamic point light beacons.

---

## 2. Key Accomplishments & Systems Built

### 2.1 Core Data Structures (`arcadia-core`)
- **Chunked Sparse 64-Tree (CS64):**
  - 4-level tree representing $256^3$ voxel chunks ($16\text{ m}^3$).
  - `Node64` (16 bytes): Dual `u32` words (`mask_low`, `mask_high`) encoding 64-bit occupancy with single-cycle hardware `POPCNT` (`countOneBits`) child slot indexing.
  - `LeafBrick64` (64 bytes): Flat 64-byte array holding 1-byte material IDs for $4 \times 4 \times 4$ voxel leaf blocks.
  - **Homogeneous Subtree Collapsing:** Uniform solid nodes (e.g., deep rock, limestone, air, or pure water) collapse into a single `Uniform` leaf node with `IS_UNIFORM_FLAG`, consuming 0 bytes in the brick pool.
  - **Breadth-First (BFS / Level-Order) Node Packing:** Replaced DFS recursion with level-order queue packing so Levels 0, 1, and 2 reside contiguously at the start of each chunk’s node buffer for high GPU L1/L2 cache hit rates.
  - **Volumetric Coverage-Weighted Mipmapping:** Internal nodes store coverage-weighted averages of color and emission, eliminating single-voxel emissive hijacking.

### 2.2 Procedural World Generation & Memory Locality (`arcadia-world`)
- **John Lin-Inspired "Fantasy Nature & Ruins" Landscape:**
  - Multi-octave FBM terrain with lush green meadows, golden grass, and dark moss.
  - Central Ancient Ruin Sanctuary: Raised stone platforms, 3-voxel ($18.75\text{ cm}$) walkable staircases, perimeter colonnade pillars, vaulted stone archways, and a glowing central obelisk.
  - Cliffside Crystal Grotto: An open-fronted cliffside amphitheater with glowing cyan, magenta, and amber crystal spires, ceiling stalactite clusters, and a deep northern magma fissure.
  - Fantasy Trees: Twisted oak trunks with flared roots and multi-lobed leaf canopies (emerald, autumn gold, cherry blossom).
  - Crystal Water Lagoon: Shallow beach transitions, sand riverbeds, and translucent water volumes.
- **3D Morton (Z-Order) Space-Filling Chunk Layout:**
  - Chunks in global GPU storage buffers (`gpu_nodes`, `gpu_bricks`) are packed along a 3D Morton curve (`morton3d(cx, cy, cz)`), guaranteeing that chunks adjacent in 3D space map to contiguous cache lines in GPU VRAM.

### 2.3 Physics & Player Movement (`arcadia-physics`)
- **Kinematic Character Controller:**
  - AABB vs. CS64 bitwise collision detection testing directly against 64-bit occupancy bitmasks in nanoseconds.
  - Smooth wall-sliding, gravity, jumping, and 3-voxel ($18.75\text{ cm}$) auto-step-up.
  - Water swimming with buoyancy, fluid drag, and vertical swim controls.
  - **Effortless Fly Mode:** Toggled with `F` or **double-tapping `Space`** (within $350\text{ ms}$). Features smooth camera-oriented flight, dedicated vertical lift (`Space` for up, `Ctrl` or `C` for down), high-speed sprint boost ($38\text{ m/s}$ vs. $15\text{ m/s}$), and unstuck safety.

### 2.4 GPU Rendering & Reconstruction Pipeline (`arcadia-render` + WGSL)
- **4-Pass Compute Architecture:**
  - **Pass 1 (`primary_gbuffer.wgsl`):** Primary 1080p ray marching, analytical 3D motion vectors, sub-pixel Halton camera jitter, ray-traced water refraction with Beer-Lambert absorption and Fresnel reflections, ray-traced mirror chrome (`MAT_MIRROR_CHROME`), and 16,384 GPU-simulated flying voxel debris particles.
  - **Pass 2 (`lighting_restir_gi.wgsl`):** Directional sun shadows with blue-noise soft penumbra, 1-bounce stochastic diffuse GI, and ReSTIR spatio-temporal reservoir reuse.
  - **Pass 3 (`denoiser_reconstruct.wgsl`):** Edge-stopping wavelet denoiser with plane-distance weighting and stair-step tolerances, followed by PBR shading (sunlight, indirect GI, ambient sky, analytical point lights).
  - **Pass 4 (`composite_dlaa.wgsl`):** Sub-pixel Halton temporal super-resolution / DLAA with YCoCg variance bounding boxes, atmospheric distance fog, and ACES filmic tonemapping.
- **GigaVoxels Ray-Cone Footprint LOD:**
  - Ray cone expansion $d(t) = t \cdot \kappa$ automatically terminates rays at coarser octree levels over distance.
  - Diffuse GI rays bypass `brick_pool` entirely, operating purely out of the 16-byte `Node64` index hierarchy for massive memory bandwidth savings.

---

## 3. Critical Bugs Diagnosed & Solved

| Issue | Root Cause | Resolution |
|---|---|---|
| **Vulkan Validation Error on AMD iGPU** | SPIR-V SSA identifier scoping bug in `naga` when indexing local `let` arrays in WGSL. | Replaced local let-arrays in shader kernels with compact, branchless `switch` functions. |
| **Buffer Overflow Panic on Carving** | Dynamic chunk edits exceeded initial staging allocation. | Implemented 2x growth slack reallocation in `WorldMap::sync_dirty_chunk`. |
| **Looking-Down FOV Distortion** | Camera basis vectors were non-orthogonal during pitch reprojection. | Enforced orthonormal Gram-Schmidt camera basis (`up = right.cross(forward).normalize()`) in CPU controller and GPU uniforms. |
| **Bright-to-Dark Temporal Ghosting** | Temporal denoiser accumulated stale high-radiance history across camera rotations. | Tightened YCoCg variance clipping box to $0.85\sigma$ and validated spatial neighbors against world positions. |
| **Hillside Sky Streaks** | Grazing-angle rays parallel to terrace steps had small $dir_y$, causing fixed $\epsilon$ to round back into the same cell until `max_steps` expired. | Introduced axis-proportional exit epsilon `exit_aabb_t` based on the exiting slab plane. |
| **Buried Light Bleeding** | Octree irradiance was accumulated during empty-space traversal before checking `node_has_child`, and internal nodes allowed single-voxel emissive overrides. | Gated irradiance strictly on occupied hits, and enforced $\ge 50\%$ child occupancy threshold for internal emissive classifications. |
| **Underwater Stepped / Sliced Floor** | Primary rays through water stepped with a fixed `t += 4.0` voxels ($0.25\text{m}$), jumping inside the riverbed floor. | Replaced blind jumps with exact AABB node/cell exits using `exit_aabb_t`. |
| **6-Cardinal-Direction Light Swirls** | `interleaved_gradient_noise` used identical linear coefficients for $u_1$ and $u_2$, locking hemisphere samples onto a 1D Fermat spiral $\phi = 2\pi r^2 + C$. | Replaced with orthogonal spatial gradients, dual 32-bit PCG hashes, and Martin Roberts' 2D $R_2$ quasi-random sequence. |

---

## 4. Roadmap & Future Work (Next Sessions)

### 4.1 High Priority: Phase 6 — Structural Island Collapse (*Teardown*-Style Physics)
- **Goal:** Detached structures (e.g. shooting out the base of a temple pillar or carving a ring around an arch) must dynamically collapse and fall under gravity rather than floating in mid-air.
- **Architecture (`arcadia-world/src/islands.rs` & `arcadia-physics/src/rigid_voxel.rs`):**
  1. **Connectivity Flood Fill:** When a chunk is carved, dispatch a background bitwise 3D flood fill starting from bedrock ($y \le 4$).
  2. **Island Extraction:** Voxels disconnected from bedrock are copied into a dynamic `RigidVoxelBody` struct holding a local CS64 tree, center of mass, inertia tensor, linear/angular velocity, and world-space transform.
  3. **Rigid Body Integration:** Integrate rigid body dynamics (gravity, angular velocity) and collide against the static world CS64 tree using bitwise OBB-AABB penetration tests.
  4. **GPU Rendering of Rigid Bodies:** Pass dynamic OBB transforms to `primary_gbuffer.wgsl` to ray march dynamic objects into the primary depth buffer before shading.

### 4.2 Engine & Visual Enhancements
- **Multi-Candidate ReSTIR DI (Direct Illumination):** Expand analytical point lights from 8 to 64+ using reservoir-sampled light candidate selection.
- **Screen-Space / Cone-Traced Glossy Reflections:** Enhance `MAT_MIRROR_CHROME` and water surfaces with roughness-dependent cone tracing for brushed metals and frosted glass.
- **Continuous World Streaming:** Implement a toroidal chunk ring-buffer around the player to stream chunks dynamically as the player flies across infinite landscapes.
- **Volumetric Caustics & Water God Rays:** Add light shaft scattering beneath the lagoon surface using shadow ray marching through the water volume.
