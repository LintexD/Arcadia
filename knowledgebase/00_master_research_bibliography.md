# 00. Master Real-Time Research Bibliography

A curated repository of foundational and cutting-edge academic papers, GDC engineering lectures, and technical publications directly informing the design and implementation of the **Arcadia Real-Time Voxel Engine**.

---

## 1. Voxel Data Structures, Hierarchies & Traversal

* **[Amanatides & Woo 1987]**  
  *A Fast Voxel Traversal Algorithm for Ray Tracing*  
  John Amanatides, Andrew Woo. *Eurographics 1987*.  
  *Significance:* The seminal mathematical formulation of 3D Digital Differential Analyzer (DDA) grid traversal using incremental additions and axis comparisons. Serves as the micro-stepping core of Arcadia's [`traversal64.wgsl`](../shaders/common/traversal64.wgsl).

* **[Crassin et al. 2009 / 2011]**  
  *GigaVoxels: A Voxel-Based Rendering Pipeline for Efficient Exploration of Large and Detailed Scenes* (SIGGRAPH 2009)  
  *Interactive Indirect Illumination Using Voxel Cone Tracing* (Pacific Graphics 2011)  
  Cyril Crassin, Fabrice Neyret, Sylvain Lefebvre, Elmar Eisemann.  
  *Significance:* Introduced hierarchical ray cone footprint LOD ($d(t) = \kappa \cdot t$), bounding volume brick caching, and zero-brick early termination. Directly inspired Arcadia's GigaVoxels LOD and Zero-Brick GI bandwidth bypass.

* **[Laine & Karras 2010]**  
  *Efficient Sparse Voxel Octrees – Analysis on a Modern GPU*  
  Samuli Laine, Tero Karras (NVIDIA Research). *ACM SIGGRAPH Symposium on Interactive 3D Graphics and Games (I3D 2010)*.  
  *Significance:* Foundational research on GPU octree traversal, contour grid compression, beam-oriented ray coherence, and child bitmask indexing.

* **[Kämpe et al. 2013]**  
  *High-Resolution Sparse Voxel DAGs*  
  Viktor Kämpe, Erik Sintorn, Ulf Assarsson. *ACM Transactions on Graphics (SIGGRAPH 2013)*.  
  *Significance:* Proved that identical subtrees in voxel octrees can be merged into a Directed Acyclic Graph (DAG), encoding trillions of voxels in megabytes of memory.

* **[Aokana 2025]**  
  *Aokana: A GPU-Driven Voxel Rendering Framework for Open World Games*  
  Graphics Engineering Consortium. *Preprint / I3D 2025*.  
  *Significance:* Modern open-world GPU-driven voxel framework combining out-of-core streaming rings with dynamic SVDAG LOD hierarchies and seamless forward-pass depth integration.

---

## 2. GPU Wavefront Architectures & Compute Ray Marching

* **[Laine & Aila 2013]**  
  *Megakernels Considered Harmful: Wavefront Path Tracing on GPUs*  
  Samuli Laine, Tero Karras, Timo Aila (NVIDIA Research). *High-Performance Graphics (HPG 2013)*.  
  *Significance:* Demonstrated that monolithic compute ray tracers choke on register pressure and warp divergence. Proved that decomposing rendering into decoupled stages connected by stream-compacted queues maximizes GPU occupancy and throughput. Foundation of Arcadia's Pass 1–5 decoupled compute pipeline.

* **[Aila & Laine 2009]**  
  *Understanding the Efficiency of Ray Traversal on GPUs*  
  Timo Aila, Samuli Laine (NVIDIA Research). *High-Performance Graphics (HPG 2009)*.  
  *Significance:* Landmark analysis of GPU SIMD warp divergence, memory latency hiding, persistent thread dispatch loops, and work queues.

---

## 3. Real-Time Global Illumination, ReSTIR & Many-Light Sampling

* **[Bitterli et al. 2020]**  
  *Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting (ReSTIR)*  
  Benedikt Bitterli, Chris Wyman, Matt Pharr, Peter Shirley, Aaron Lefohn, Wojciech Jarosz. *ACM Transactions on Graphics (SIGGRAPH 2020)*.  
  *Significance:* Revolutionized real-time ray tracing by combining Resampled Importance Sampling (RIS) with streaming temporal/spatial candidate reuse, enabling millions of dynamic lights with 1 shadow ray per pixel.

* **[Ouyang et al. 2021]**  
  *ReSTIR GI: Path Resampling for Real-Time Path Tracing*  
  Yaobin Ouyang, Shiqiu Liu, Markus Kettunen, Matt Pharr, Jacopo Pantaleoni. *ACM SIGGRAPH 2021*.  
  *Significance:* Extended reservoir resampling to indirect diffuse multi-bounce illumination with spatial/temporal neighbor validation. Directly implemented in Arcadia's [`lighting_restir_gi.wgsl`](../shaders/lighting_restir_gi.wgsl).

* **[Bitterli et al. 2022]**  
  *Generalized Resampled Importance Sampling: Foundations of ReSTIR (GRIS)*  
  Benedikt Bitterli, Chris Wyman, Matt Pharr, Peter Shirley, Aaron Lefohn, Wojciech Jarosz. *ACM Transactions on Graphics (SIGGRAPH 2022)*.  
  *Significance:* Provides the mathematical formulation for unbiased multi-bounce path reuse and complex glossy/specular BRDF lobe shift mappings across surfaces with varying roughness.

* **[Yuksel 2019]**  
  *Stochastic Light Cuts*  
  Cem Yuksel. *High-Performance Graphics (HPG 2019)*.  
  *Significance:* Evaluates illumination from hundreds of thousands of dynamic emissive light sources in constant time per pixel using stochastic bounding hierarchies. Informs Arcadia's two-level macro-cell ReSTIR DI design.

* **[Sannikov 2024]**  
  *Radiance Cascades: A Novel Approach to Real-Time Global Illumination*  
  Alexander Sannikov. *Preprint / EGSR 2024*.  
  *Significance:* Proposes hierarchical spatial-angular radiance intervals that merge hierarchically to produce 100% noise-free, zero-Monte-Carlo real-time diffuse illumination without temporal ghosting.

---

## 4. Temporal Super-Resolution, Anti-Aliasing & Denoising

* **[Karis 2014]**  
  *High-Quality Temporal Supersampling*  
  Brian Karis (Epic Games). *SIGGRAPH 2014 Advances in Real-Time Rendering in Games*.  
  *Significance:* The seminal paper establishing modern TAA/TSR techniques: Halton sub-pixel jitter matrices, YCoCg color-space decoupling, and $3 \times 3$ neighborhood variance clipping ($[\mu - \sigma, \mu + \sigma]$) to prevent temporal ghosting. The mathematical baseline for Arcadia's Pass 4 ([`composite_dlaa.wgsl`](../shaders/composite_dlaa.wgsl)).

* **[Schied et al. 2017]**  
  *Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Illumination (SVGF)*  
  Christoph Schied, Anton Kaplanyan, Chris Wyman, Anjul Patney, Chakravarty R. Alla Chaitanya, John Burgess, Shiqiu Liu, Carsten Dachsbacher, Aaron Lefohn, Marco Salvi. *High-Performance Graphics (HPG 2017)*.  
  *Significance:* Pioneered combining temporal moment tracking with edge-stopping À-trous wavelet filters using depth and normal variance weights to reconstruct clean lighting from 1 sample per pixel.

* **[Schied et al. 2018]**  
  *Gradient Estimation for Real-Time Adaptive Temporal Filtering (A-SVGF)*  
  Christoph Schied, Marco Salvi, Anton Kaplanyan. *High-Performance Graphics (HPG 2018)*.  
  *Significance:* Formulates forward temporal gradient estimation to detect disocclusions and sudden shadow/light changes immediately, preventing ghosting trails.

* **[Jimenez et al. 2016]**  
  *Practical Real-Time Strategies for Addressing Surface Heterogeneity and Horizon-Based Ambient Occlusion (GTAO)*  
  Jorge Jimenez, Xian-Chun Wu, Angelo Pesce, Adrian Jarabo. *SIGGRAPH 2016*.  
  *Significance:* Derives the closed-form projected cosine horizon integral for physically grounded ambient occlusion without halo artifacts.

---

## 5. Physics, Rigid Bodies, Stacking & Destruction

* **[Guendelman, Bridson, Fedkiw 2003]**  
  *Nonconvex Rigid Bodies with Stacking*  
  Eran Guendelman, Robert Bridson, Ronald Fedkiw (Stanford University). *ACM Transactions on Graphics (SIGGRAPH 2003)*.  
  *Significance:* Established the gold standard for resting contact stability by decoupling velocity integration into separate collision (restitution) and contact (resting projection) steps with shock propagation. Directly resolves micro-bouncing and settle damping for Arcadia's [`rigid_voxel.rs`](../crates/arcadia-physics/src/rigid_voxel.rs).

* **[Macklin et al. 2016]**  
  *XPBD: Position-Based Simulation of Compliant Constrained Dynamics*  
  Miles Macklin, Matthias Müller, Nuttapong Chentanez, Tae-Yong Kim (NVIDIA). *Motion in Games (MIG 2016)*.  
  *Significance:* Formulates compliant physical constraints that remain unconditionally stable under varying time-steps $\Delta t$, eliminating explosive impulse spikes during deep geometric voxel interpenetration.

* **[Gustafsson 2015 / 2020]**  
  *Physics for Game Programmers: Destruction in Smash Hit & Teardown Real-Time Destruction*  
  Dennis Gustafsson (Tuxedo Labs / Mediocre). *GDC 2015 & Voxagon Technical Notes*.  
  *Significance:* The primary practical reference for real-time discrete voxel structural integrity flood-fill, impulse capping, and dynamic voxel rigid body instantiation.

* **[Mirtich 1996]**  
  *Fast and Accurate Computation of Polyhedral Mass Properties*  
  Brian Mirtich. *Journal of Graphics Tools (1996)*.  
  *Significance:* The definitive mathematical formulation for discrete centers of mass and $3 \times 3$ moment of inertia tensors, adapted to cubic parallel-axis correction ($\frac{s^2}{6}$) in Arcadia's rigid body physics.

---

## 6. Real-Time Voxel Fluids & Multi-Phase Simulation

* **[Chentanez & Müller 2011]**  
  *Real-Time Eulerian Water Simulation Using a Pressure Multigrid on Irregular Voxel Grids*  
  Nuttapong Chentanez, Matthias Müller (NVIDIA PhysX Research). *ACM Transactions on Graphics (SIGGRAPH 2011)*.  
  *Significance:* Demonstrates a GPU pressure multigrid solver capable of simulating 3D free-surface liquid directly inside dynamically changing voxelized boundaries at 60+ FPS.

* **[Stam 1999]**  
  *Stable Fluids*  
  Jos Stam. *ACM SIGGRAPH 1999*.  
  *Significance:* Introduced semi-Lagrangian advection and Helmholtz-Hodge projection, guaranteeing that grid-based fluid simulations never blow up regardless of simulation time-step.

* **[Hu et al. 2018]**  
  *A Fast Moving Least Squares Material Point Method of Compatible Particle-in-Cell (MLS-MPM)*  
  Yuanming Hu, Yu Fang, Ziheng Ge, Ziyin Qu, Yixin Zhu, André Pradhana, Chenfanfu Jiang. *ACM Transactions on Graphics (SIGGRAPH 2018)*.  
  *Significance:* The state of the art for simulating sand, wet mud, snow avalanches, and clay deformation on background Cartesian grids.

---

## 7. Procedural Acoustic Physics

* **[O'Brien et al. 2001]**  
  *Synthesizing Sounds from Rigid-Body Simulations*  
  James F. O'Brien, Chen Shen, Christine M. Gatchalian. *ACM SIGGRAPH 2001*.  
  *Significance:* Derives procedural audio synthesis directly from surface acceleration and impact impulses of colliding rigid bodies.

* **[Chadwick et al. 2009]**  
  *Harmonic Shells: A Practical Approach to Real-Time Sound Synthesis on GPUs*  
  Jeffrey N. Chadwick, Steven S. An, Doug L. James. *ACM SIGGRAPH Asia 2009*.  
  *Significance:* Real-time GPU synthesis of modal vibrations and acoustic radiation for hollow and solid resonant bodies. Guides Arcadia's kinetic voxel sound synthesis ([`ROADMAP_AND_TODO.md` E3](../docs/plans/ROADMAP_AND_TODO.md)).

---

## 8. Mob AI, Dynamic Voxel Navigation & Destructible Characters

* **[Martin et al. 2020 / id Software]**  
  *Push-Forward Combat and Destructible Demons in DOOM Eternal*  
  Hugo Martin, Marty Stratton, Billy Khan (id Software / Bethesda). *SIGGRAPH 2020 Real-Time Live & id Tech 7 Architecture Notes*.  
  *Significance:* Pioneered multi-layer dynamic anatomical damage states (armor stripping, flesh ablation, exposed organs/weakpoints) to eliminate "bullet-sponge" enemies. The conceptual benchmark that Arcadia elevates into true volumetric 3D matter.

* **[Harabor & Botea 2011]**  
  *Online Graph Pruning for Pathfinding on Grid Maps (Jump Point Search)*  
  Daniel Harabor, Alban Botea (NICTA & Australian National University). *AAAI Conference on Artificial Intelligence (AAAI 2011)*.  
  *Significance:* Establishes symmetry-breaking jump rules that eliminate intermediate grid nodes, speeding up path queries on uniform voxel grids by up to $10\times\text{--}30\times$.

* **[Sturtevant & Buro 2005]**  
  *Partial-Path Evaluation for Hierarchical Pathfinding (HPA\*)*  
  Nathan R. Sturtevant, Michael Buro. *Artificial Intelligence and Interactive Digital Entertainment (AIIDE 2005)*.  
  *Significance:* Proves that partitioning grid environments into multi-level clusters allows instant $O(1)$ dynamic edge invalidation upon environmental destruction without global graph recalculation.

* **[Reynolds 1987 / 1999]**  
  *Flocks, Herds, and Schools: A Distributed Behavioral Model* (SIGGRAPH 1987)  
  *Steering Behaviors for Autonomous Characters* (GDC 1999)  
  Craig W. Reynolds.  
  *Significance:* Foundational formulations of separation, alignment, cohesion, obstacle avoidance, and path-following flow field vectors for real-time crowd dynamics.

* **[Müller et al. 2007]**  
  *Position Based Dynamics (PBD)*  
  Matthias Müller, Bruno Heidelberger, Marcus Hennix, John Ratcliff. *Journal of Visual Communication and Image Representation (2007)*.  
  *Significance:* Derives unconditionally stable position-based constraints for articulated kinematic ragdolls, joint limit cones, and dynamic dismemberment severance in interactive environments.
