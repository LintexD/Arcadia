# Arcadia Real-Time Voxel Engine Knowledgebase

Welcome to the **Arcadia Real-Time Voxel Knowledgebase**. This repository of technical knowledge, mathematical formalisms, hardware benchmarks, and architectural designs covers state-of-the-art real-time voxel technologies suitable for high-performance interactive 3D applications (60–120+ FPS on modern GPUs).

---
## 🧭 Navigation & Table of Contents

### [00. Master Research Bibliography](./00_master_research_bibliography.md)
*Curated repository of foundational academic papers, GDC engineering talks, and technical publications across rendering, physics, ReSTIR, and spatial data structures.*

---

### [01. Storage & Data Structures](./01_storage_and_data_structures/)
*How to represent billions of potential voxels compactly with nanosecond random-access and GPU-friendly memory alignment.*
- **[01. Sparse Voxel Octrees & N-ary Trees](./01_storage_and_data_structures/01_sparse_voxel_octrees_and_nary_trees.md)**: SVOs, Chunked Sparse 64-Trees (CS64), bitmask occupancy, hardware `POPCNT` child indexing, breadth-first contiguous layouts, and homogeneous subtree collapsing.
- **[02. Sparse Voxel DAGs & Deduplication](./01_storage_and_data_structures/02_sparse_voxel_dags_and_compression.md)**: SVDAGs, sub-tree pointer sharing, geometric hashing, color DAGs, pointerless DAG formats, and symmetry exploitation.
- **[03. Clipmaps, Brickmaps & Virtual Grids](./01_storage_and_data_structures/03_clipmaps_brickmaps_and_virtual_grids.md)**: Toroidal rolling clipmaps, OpenVDB/NanoVDB hierarchical grids, GigaVoxels page table brick caching, and memory virtualization.
- **[04. Spatial Hashing & Morton / Hilbert Locality](./01_storage_and_data_structures/04_spatial_hashing_and_morton_encoding.md)**: GPU cuckoo & compact spatial hash tables, 3D Morton (Z-order) and Hilbert curves, GPU L1/L2 cache hit optimization.
- **[05. RLE, Bitpacking & Palette Compression](./01_storage_and_data_structures/05_rle_bitpacking_and_palette_compression.md)**: Slice/Column run-length encoding, paletted attribute tables, entropy encoding for disk/network serialization.
- **[06. Storage Comparison & Selection Matrix](./01_storage_and_data_structures/06_storage_comparison_matrix.md)**: Comprehensive quantitative matrix comparing memory footprint, random read/write latency, dynamic carving overhead, and GPU ray traversal suitability.

---

### [02. Rendering Techniques](./02_rendering_techniques/)
*How to turn discrete 3D spatial grids into sub-millisecond pixels without blowing GPU compute budgets.*
- **[01. Ray Marching & Hierarchical DDA](./02_rendering_techniques/01_ray_marching_and_hierarchical_dda.md)**: 3D Amanatides & Woo DDA, Hierarchical N-tree skip-traversal, ray cone footprint LOD, slab-intersection floating-point edge handling.
- **[02. GPU Compute Ray Tracing Architectures](./02_rendering_techniques/02_gpu_compute_ray_tracing_architectures.md)**: Screen-tiled compute workgroups vs persistent threads, 1/4-res coarse tile depth pre-passes, stream compaction, divergence reduction.
- **[03. Mesh Extraction: Greedy & Blocky](./02_rendering_techniques/03_mesh_extraction_greedy_and_blocky.md)**: Binary greedy meshing, bitmask slice sweep-and-prune, face culling, per-vertex bitwise ambient occlusion, texturing and UV atlas strategies.
- **[04. Mesh Extraction: Smooth Surfaces](./02_rendering_techniques/04_mesh_extraction_smooth_surfaces.md)**: Surface Nets, Dual Contouring (QEF solvers and sharp feature preservation), Marching Cubes, and Transvoxel transition cells across LOD boundaries.
- **[05. Software Compute Rasterization & Splatting](./02_rendering_techniques/05_software_rasterization_and_splatting.md)**: Nanite-inspired 64-bit atomic integer depth buffer rasterizers in compute shaders, surfel / point splatting, and hierarchical Z-culling.
- **[06. Hybrid Rendering & SDF Approximations](./02_rendering_techniques/06_hybrid_rendering_and_sdf_approaches.md)**: Primary compute ray march + secondary rasterization, voxel-approximated Signed Distance Fields (SDFs), hybrid visibility buffers.
- **[07. Rendering Comparison & Decision Matrix](./02_rendering_techniques/07_rendering_comparison_matrix.md)**: Trade-off comparison between compute ray marching, polygon meshing, and software rasterization.

---

### [03. Simulation & Physics](./03_simulation_and_physics/)
*How to create fully interactive, destructible, flowing worlds at 60+ FPS.*
- **[01. Cellular Automata & Granular Materials](./03_simulation_and_physics/01_cellular_automata_and_falling_materials.md)**: Falling sand, granular physics, liquids, and gases in 3D. Active chunk tracking, sleep islands, bitwise CA updates.
- **[02. Structural Integrity & Teardown Collapse](./03_simulation_and_physics/02_structural_integrity_and_teardown_collapse.md)**: Bitwise 3D flood fill, disjoint set / union-find, island extraction, and instant dynamic rigid body spawning.
- **[03. Dynamic Rigid Body Voxels](./03_simulation_and_physics/03_dynamic_rigid_body_voxels.md)**: Voxel OBB collision, discrete voxel-vs-voxel penetration depth, center of mass and inertia tensor derivation from bitmasks.
- **[04. Real-Time Voxel Fluids & LBM](./03_simulation_and_physics/04_realtime_voxel_fluids_and_lbm.md)**: GPU Euler grid fluids (Stable Fluids), real-time Lattice Boltzmann Method (LBM D3Q19), boundary conditions, and PBD/SPH voxel rasterization.
- **[05. Material Point Method (MLS-MPM)](./03_simulation_and_physics/05_material_point_method_mls_mpm.md)**: Real-time GPU MLS-MPM for snow, mud, sand, and clay. Particle-to-Grid (P2G) and Grid-to-Particle (G2P) transfers on voxel backgrounds.
- **[06. Environmental & Thermal Simulation](./03_simulation_and_physics/06_environmental_and_thermal_simulation.md)**: Fire propagation, heat diffusion, scorch rims, moisture diffusion, and vegetation growth automata.
- **[07. Simulation Comparison Matrix](./03_simulation_and_physics/07_simulation_comparison_matrix.md)**: Computational budget, memory footprint, and scaling limits for real-time physics paradigms.

---

### [04. Lighting & Illumination](./04_lighting_and_illumination/)
*Real-time global illumination, ambient occlusion, shadows, and reconstruction for dynamic voxel environments.*
- **[01. Voxel Cone Tracing (VCT)](./04_lighting_and_illumination/01_voxel_cone_tracing_vct.md)**: Cascaded 3D mipmapped clipmaps, diffuse & specular cone stepping, radiance injection, and light leakage mitigations.
- **[02. ReSTIR GI & ReSTIR DI for Voxels](./04_lighting_and_illumination/02_restir_gi_and_restir_di.md)**: Spatio-temporal reservoir resampling, handling millions of dynamic emissive voxels without BVHs, ray cone LOD for indirect bounce acceleration.
- **[03. Radiance Cascades (2024–2026)](./04_lighting_and_illumination/03_radiance_cascades.md)**: The penumbra hypothesis, angular vs spatial intervals, merge operators, adaptation to 3D sparse voxel probe grids, and noiseless real-time GI.
- **[04. Probe Grids & DDGI](./04_lighting_and_illumination/04_probe_grids_and_ddgi.md)**: Dynamic Diffuse Global Illumination, ray-traced irradiance/depth probe grids, octahedral mapping, and dynamic probe relocation.
- **[05. Light Propagation Volumes (LPV)](./04_lighting_and_illumination/05_light_propagation_volumes_and_cascades.md)**: Spherical Harmonics (SH) radiance injection, GPU propagation iterations, and cascaded grids.
- **[06. Screen-Space GI with Voxel Fallbacks](./04_lighting_and_illumination/06_screen_space_gi_with_voxel_fallback.md)**: High-frequency screen-space ray tracing with hierarchical voxel DDA fallback for off-screen geometry.
- **[07. Ambient Occlusion: Bitwise & Screen Space](./04_lighting_and_illumination/07_ambient_occlusion_bitwise_and_ssao.md)**: Nanosecond directional bitwise AO directly from 64-bit occupancy masks, bent normals, and GTAO.
- **[08. Volumetric Lighting & Participating Media](./04_lighting_and_illumination/08_volumetric_lighting_and_participating_media.md)**: Froxel grids (frustum voxel grids), Beer-Lambert extinction, Henyey-Greenstein phase functions, god rays, and volumetric fog.
- **[09. Spatiotemporal Denoising & Reconstruction](./04_lighting_and_illumination/09_spatiotemporal_denoising_and_reconstruction.md)**: À-trous wavelet filters, SVGF, plane-distance aware edge-stopping geometry weights, and YCoCg variance clipping.
- **[10. Lighting Comparison Matrix](./04_lighting_and_illumination/10_lighting_comparison_matrix.md)**: Frame time costs (ms), VRAM requirements, dynamic scene responsiveness, and visual fidelity ratings.

---

### [05. Real-Time Engine Case Studies](./05_realtime_engine_case_studies/)
*Dissecting commercial and notable indie voxel engines to extract battle-tested design patterns.*
- **[01. Teardown Architecture Deep Dive](./05_realtime_engine_case_studies/01_teardown_architecture_deep_dive.md)**: Dual-depth ray marching, destructive connectivity flood-fill, spatial hashing, and rigid body voxelization.
- **[02. The Tomorrow Children: Cascaded VCT](./05_realtime_engine_case_studies/02_the_tomorrow_children_cascaded_vct.md)**: SVDAG geometry compression, compute cascaded voxel cone tracing, and directional occlusion.
- **[03. John Lin Microvoxel World](./05_realtime_engine_case_studies/03_john_lin_microvoxel_world.md)**: CS64 data layout, hardware `POPCNT`, GPU compute DDA, ReSTIR GI, and physical water.
- **[04. Noita: Cellular Automata at Scale](./05_realtime_engine_case_studies/04_noita_cellular_automata_at_scale.md)**: Chunk threading, falling everything, reaction matrices, and liquid/gas state machines.
- **[05. Enshrouded: Modern Smooth Voxels](./05_realtime_engine_case_studies/05_enshrouded_and_modern_smooth_voxels.md)**: Continuous density fields, Transvoxel dual contouring, dynamic editing, and texture blending.
- **[06. Atomontage: Volumetric Matter & Streaming](./05_realtime_engine_case_studies/06_atomontage_volumetric_matter_and_streaming.md)**: SVO/DAG micro-voxels, AVF compression, progressive cloud/client streaming, and hybrid raster-voxel depth compositing.
- **[07. IOLITE: Real-Time Path-Traced Voxel Engine](./05_realtime_engine_case_studies/07_iolite_path_traced_voxel_engine.md)**: Real-time Vulkan compute path tracing, blue noise stochastic sampling, AgX polynomial tonemapping, Reverse-Z depth, and variable-dt symplectic numerical integration.

---

### [06. Mobs, Pathfinding, Animation & Destruction](./06_mobs_pathfinding_animation_and_destruction/)
*Real-time AI, dynamic 3D voxel navigation, articulated rigging, and volumetric destructible demon anatomy.*
- **[01. Real-Time Voxel Navigation & Dynamic 3D Pathfinding](./06_mobs_pathfinding_animation_and_destruction/01_hierarchical_voxel_pathfinding_and_navigation.md)**: Three-tier Hierarchical Voxel Graph (HVG), $O(1)$ dynamic invalidation from `ChunkEditReport`, 3D Jump Point Search (JPS-3D), Dijkstra surface flow fields, auto-step-up, and wall-crawling.
- **[02. Mob Behavior Architectures, Perception & Spatial Reasoning](./06_mobs_pathfinding_animation_and_destruction/02_mob_behavior_architectures_and_spatial_perception.md)**: Behavior Trees vs Utility AI, zero-brick CPU DDA line-of-sight, acoustic diffusion through voxel porosity, procedural half/full cover finding, and dynamic destructible wall breach attacks.
- **[03. Voxel Skeletal Rigging, Forward Kinematics & GPU Ray Marching](./06_mobs_pathfinding_animation_and_destruction/03_voxel_skeletal_animation_and_articulation.md)**: Articulated kinematic multi-body hierarchy, ball-and-socket voxel overlap (zero gaps/seams), GPU inverse-space HDDA raymarching in `primary_gbuffer.wgsl`, analytic 3D motion vectors for Halton DLAA, and 2-bone analytical IK.
- **[04. Volumetric Anatomy & Dynamic Carving: The Next-Gen "Doom Eternal" System](./06_mobs_pathfinding_animation_and_destruction/04_destructible_demons_and_volumetric_anatomy.md)**: Transcending id Tech 7's mesh limits with true 3D voxel matter: 4-layer anatomy (Chitin, Flesh, Bone Core, Emissive Hellfire Organ), weapon carve signatures, real-time dismemberment via bone island detection, dynamic 6-DOF limb drops, and post-mutilation behavioral adaptation.
- **[05. Horde Scaling, Multi-Tier LODs & Hybrid GPU Simulation](./06_mobs_pathfinding_animation_and_destruction/05_horde_simulation_gpu_culling_and_lod.md)**: Scaling to 100+ active demons under 2.5 ms CPU / 1.5 ms GPU: 4-tier simulation/render LODs, Copy-On-Write (COW) anatomy templating, 1D compute shader frustum/depth limb culling, and stream compaction.
- **[06. Architectural Synthesis & Engine Integration Roadmap](./06_mobs_pathfinding_animation_and_destruction/06_architectural_synthesis_and_arcadia_roadmap.md)**: Implementation blueprint for `crates/arcadia-mob/`, GPU uniform layouts (`MobLimbInstance`), WGSL shader contracts, and a 6-phase milestone roadmap.

---

### [07. Ideas & Architectural Synthesis for Arcadia](./07_ideas_and_architectural_synthesis_for_arcadia/)
*Concrete next steps, high-leverage architectural upgrades, and experimental ideas tailored for Arcadia.*
- **[01. High-Leverage Upgrades for CS64](./07_ideas_and_architectural_synthesis_for_arcadia/01_high_leverage_upgrades_for_cs64.md)**: Drop-in enhancements for Arcadia's Rust + wgpu compute pipeline.
- **[02. Experimental Mechanics & Next-Gen Concepts](./07_ideas_and_architectural_synthesis_for_arcadia/02_experimental_mechanics_and_next_gen_ideas.md)**: Radiance Cascades on sparse 64-trees, bitwise cellular automata, and hybrid visibility buffers.

---

## ⚡ The Real-Time Paradigm Constraint

Every technique in this knowledgebase is strictly vetted for **real-time performance**:
- **Target Framerate:** 60 to 144 FPS (Frame budgets: **6.9 ms to 16.6 ms** total; rendering/lighting budget typically **4.0 ms to 8.0 ms**).
- **Target Hardware:** Desktop PC GPUs (NVIDIA RTX 3060/3080/4070, AMD RX 6700/7800) and current-gen consoles (PS5 / Xbox Series X).
- **Exclusion Rule:** Offline Monte Carlo path tracing with hundreds of samples per pixel, unaccelerated volume marching, and offline FEM mechanics that take seconds or minutes per frame are excluded or presented strictly in their real-time, compute-shader-adapted variants.
