# 04. Noita: Cellular Automata at Scale

## 1. Overview & Conceptual Vision

Developed by Nolla Games (Petri Purho, Olli Harjola, Arvi Teikari; released 2020), *Noita* achieved cult status with its core engineering motto: **"Every Pixel is Simulated"**.

While set in a 2D side-scrolling perspective, *Noita's* core engineering architecture offers masterclass insights for 3D voxel engine designers seeking to scale cellular automata and dynamic material reactions to millions of simulated elements.

---

## 2. The Material Reaction Matrix

In *Noita*, voxels are not static color tags; they are reactive physical materials:
- **States of Matter:** Solid (rock, wood, ice), Granular Powder (sand, coal, gunpowder, bone), Liquid (water, acid, lava, oil, blood, whiskey), and Gas (steam, smoke, toxic fumes, fire).
- **The Reaction Table:** When two materials touch:
  $$\text{Lava} + \text{Water} \longrightarrow \text{Rock} + \text{Steam}$$
  $$\text{Acid} + \text{Anything Organic} \longrightarrow \text{Flammable Toxic Gas}$$
  $$\text{Fire} + \text{Oil} \longrightarrow \text{Massive Firestorm} + \text{Smoke}$$
  $$\text{Water} + \text{Toxic Sludge} \longrightarrow \text{Purified Water}$$

Every frame, boundary cells evaluate this $N \times N$ reaction lookup matrix, producing emergent physical gameplay cascades.

---

## 3. High-Performance Multithreading: Chunk Partitioning

Evaluating millions of cellular automata cells on the CPU at 60 FPS requires aggressive parallelization:
- The world is partitioned into a 2D/3D grid of chunks.
- **Checkerboard Sub-Stepping:** Chunks are updated in 4 alternating passes (even/odd row and column parity). This guarantees that no two threads update adjacent boundary chunks simultaneously, eliminating thread locking and race conditions.
- **Sleep Islands:** Settled sand dunes and static bedrock chunks enter a dormant sleep state, consuming 0 CPU cycles until awakened by an explosion, player touch, or dynamic fluid contact.

---

## 4. Coupling Cellular Automata with Rigid Bodies

*Noita's* engine dynamically transitions materials between cellular automata and Box2D rigid bodies:
1. **Loose to Rigid:** When an explosion cuts a chunk of stone loose from the ceiling, the connected component is extracted, converted into a rigid body polygon, and falls under physics.
2. **Rigid to Loose:** When a heavy stone stalactite slams into the floor at high velocity, the impact force shatters the rigid body, vaporizing it back into thousands of dynamic falling sand and gravel pixels.

---

## 5. Key Lessons for 3D Voxel Engines

1. **Reaction Tables in Compute Shaders:** The material reaction matrix can be evaluated directly on the GPU in a secondary compute dispatch, enabling chemical chain reactions in 3D (e.g. lava melting subterranean ice caverns into steam).
2. **Sleeping Is Essential:** Over $95\%$ of a voxel world is at rest at any given second. Active-chunk dirty lists are mandatory to keep simulation times below $1\text{ ms}$.
3. **Emergent Gameplay:** Simple local transition rules produce infinitely richer player experiences than pre-scripted cinematic events.
