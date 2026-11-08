# 02. Experimental Mechanics & Next-Gen Concepts

## 1. Overview & Creative Horizons

Beyond standard rendering and physics optimizations, a micro-voxel engine like Arcadia provides a playground for mechanics and visual techniques that are impossible in traditional polygon engines.

This document compiles cutting-edge, experimental concepts vetted strictly for **real-time viability** in interactive 3D game engines (60–120+ FPS).

---

## 2. Concept 1: 3D Radiance Cascades on Sparse 64-Trees

### The Idea
Replace the stochastic diffuse indirect bounce in ReSTIR GI with **Split 3D Radiance Cascades**:
- Store cascade radiance in a sparse GPU hash map keyed by surface voxel coordinates.
- Cascades $c \in \{0, 1, 2, 3\}$ correspond to distance shells $[0.25\text{m}, 1\text{m}, 4\text{m}, 16\text{m}]$.
- Primary rays segment intervals during CS64 DDA traversal.
- **The Payoff:** **Zero Monte Carlo noise.** No fireflies, no boiling artifacts, and no temporal ghosting during fast camera rotations.

---

## 3. Concept 2: Hydraulic Erosion Automata with Living Water

### The Idea
Implement John Lin-style hydraulic erosion directly on the CS64 tree:
1. **Water Inflow:** Water springs spawn at mountain summits.
2. **Velocity & Sediment Capacity:** Water voxels track velocity $\mathbf{v}$ and sediment load $S$.
3. **Erosion:** Fast water over dirt/sand clears bits in the CS64 occupancy mask, converting solid terrain into dissolved sediment ($S \mathrel{+}= \Delta S$).
4. **Sediment Deposition:** When water slows down in lakes or lagoons, it deposits sand voxels, forming natural underwater sandbars and river deltas.
5. **Emergent Gameplay:** The player can dam up rivers, redirect waterfalls through enemy camps, or dig trenches to drain stagnant swamps.

---

## 4. Concept 3: Dynamic Voxel Rigid-Body Shattering [COMPLETED]

### Implementation Summary
Implemented in Phase 6A2 in [`rigid_voxel.rs`](../../crates/arcadia-physics/src/rigid_voxel.rs):
- Evaluates kinetic impact energy $E_k = \frac{1}{2} M |\mathbf{v}|^2$ across rigid body contact points.
- When $E_k > E_{\text{fracture}}$ (impact velocity $> 6.5\text{ m/s}$), boundary contact voxels fracture into physical GPU debris particles ([`DebrisParticle`](../../crates/arcadia-world/src/world_map.rs)).
- Small clusters ($\le 96$ voxels) impacting at terminal velocity are completely disintegrated into physicalized debris splinters.

---

## 5. Concept 4: Thermal Conduction & Firestorms

### The Idea
Couple voxel materials with thermal state machines:
- Lava fissures heat adjacent rock and wood.
- When dry wooden beams reach $300^\circ\text{C}$, they ignite, transitioning to `EMBER_RIM` with high blackbody emission.
- Burning wood consumes its structural integrity; when the support burns through, the ceiling collapses via structural integrity flood-fill!
- Fire injects radiant orange light into the ReSTIR pipeline and spawns smoke particles into the Froxel volumetric fog grid.

---

## 6. Concept 5: Hybrid Nanite-Style Compute Micro-Rasterizer

### The Idea
For scenes featuring massive swarms of flying voxel debris or dense foliage:
- Software-rasterize micro-voxels into a 64-bit atomic integer depth buffer (`atomicMin` on `(depth << 32) | payload`) using compute shaders.
- Sub-pixel debris particles are splatted as 1-pixel atomics in single clock cycles, bypassing ray marching loops entirely.
- Scales up to **hundreds of thousands of simultaneous debris fragments** during colossal building collapses.

---

## 7. Concept 6: Wavelet Specular ReSTIR & Variable Roughness Voxels [NEW]

### The Idea
Extend Arcadia's mirror reflections and water refraction to support **continuous surface roughness** (e.g. wet mossy rock, brushed gold relics, frosted quartz crystals):
1. **Roughness-Aware Ray-Cone LOD:** In [`primary_gbuffer.wgsl`](../../shaders/primary_gbuffer.wgsl), modulate the secondary reflection ray cone opening angle $\kappa_{\text{refl}}$ by surface roughness $\alpha$:
   $$\kappa_{\text{refl}} = \kappa_0 + 2 \alpha^2$$
2. **Rough Surfaces Skip Fine Bricks:** A rough stone floor terminates reflection traversal at Level 1 ($4\text{ m}$) or Level 2 ($1\text{ m}$), making rough glossy reflections **faster to march** than mirror reflections!
3. **Specular Wavelet Denoising:** Filter rough reflections in [`denoiser_reconstruct.wgsl`](../../shaders/denoiser_reconstruct.wgsl) with roughness-guided variance envelopes, providing smooth reflections across fantasy materials at $< 0.8\text{ ms}$.

---

## 8. Concept 7: Discrete Voxel Kinetic Audio Synthesis [NEW]

### The Idea
Game engines typically rely on pre-recorded sound clips triggered on bounding box collisions. Because Arcadia computes **exact discrete voxel contact manifolds**:
1. **Material-Acoustic Impulse Mapping:** Every voxel contact point carries a `MaterialId` (e.g. `MAT_LIMESTONE`, `MAT_DARK_SLATE`, `MAT_WATER`, `MAT_WOOD`).
2. **Procedural Impulse Audio:** Synthesize pitch, resonance, and crunch envelopes proportional to impact energy $E_k$ and material acoustic impedance:
   - High $E_k$ limestone contact $\to$ sharp resonant snap + grinding gravel rattle.
   - Low $E_k$ stone roll $\to$ continuous low-frequency tumbling rumble.
   - Debris falling into water $\to$ micro-splashes scaled by submerged debris volume.
3. **Zero Recording Overhead:** Dynamic destruction produces realistic, unrepeated audio soundscapes automatically.

---

## 9. Concept 8: GPU Indirect Dispatch for Divergence-Free Secondary Wavefronts [NEW]

### The Idea
Currently, Pass 2 ([`lighting_restir_gi.wgsl`](../../shaders/lighting_restir_gi.wgsl)) dispatches a fixed 2D grid covering all screen pixels. On sky pixels or smooth shadows, threads remain idle or exit early while neighboring threads march complex bounce paths, causing warp execution divergence.
1. **Stream Compaction:** In Pass 1 or a 64-thread workgroup prefix-sum pass, write pixels requiring secondary GI or reflection rays into an append buffer.
2. **Indirect Dispatch:** Use `wgpu::ComputePass::dispatch_workgroups_indirect` to launch compute threads strictly for active secondary rays.
3. **Impact:** Maximizes GPU SIMD efficiency and compute core saturation on both integrated and discrete architectures.
