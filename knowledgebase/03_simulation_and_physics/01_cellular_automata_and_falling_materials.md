# 01. Cellular Automata & Granular Materials

## 1. Overview & Conceptual Model

Cellular Automata (CA) simulate complex macroscopic behaviors (falling sand, avalanches, liquid spreading, gas diffusion, fire propagation) through simple, local transition rules applied across a discrete grid.

Made famous in 2D by games like *Noita* and *The Powder Toy*, extending Cellular Automata to 3D real-time micro-voxel worlds provides unprecedented physical dynamism without the prohibitive cost of full continuous fluid or rigid-body solvers.

---

## 2. 3D Granular & Falling Sand Automata

### 2.1 The Falling Sand Transition Rule
For a granular voxel (e.g. Sand, Gravel, Powder) at coordinate $(x, y, z)$:
1. **Vertical Fall:** Check cell directly below $(x, y-1, z)$. If empty:
   $$\text{Swap}(\text{Voxel}(x, y, z), \text{Voxel}(x, y-1, z))$$
2. **Diagonal Slide (Angle of Repose):** If $(x, y-1, z)$ is solid, check the 8 diagonal downward neighbors $(x \pm 1, y-1, z \pm 1)$:
   - If empty, slide diagonally with probability $P_{\text{slide}}$.
   - This creates natural conical sand dunes with realistic angles of repose ($\approx 30^\circ\text{--}35^\circ$).

```text
       [ Sand ]
          │
          ├── (1. If empty, fall down)
          ▼
       [ Empty ]
          │
     (2. If blocked, slide diagonally)
       ↙  ↓  ↘
    [Air] [Solid] [Air]
```

### 2.2 Fluid Spread Rules (Water, Acid, Lava)
Liquids follow granular gravity rules, but if downward and diagonal paths are blocked:
1. **Lateral Spread:** Check the 4 cardinal horizontal neighbors $(x \pm 1, y, z)$ and $(x, y, z \pm 1)$.
2. If empty, move horizontally.
3. **Viscosity Factor:** Lava moves 1 step every $N$ ticks, while water spreads every tick.

---

## 3. Real-Time Optimization: Active Chunk & Sleep State Tracking

Evaluating CA rules across every voxel in a $256^3$ chunk ($16.7\text{ million}$ cells) at 60 FPS would require **1 billion cell updates per second**, overwhelming CPU and GPU pipelines.

### 3.1 Bounding Box & Active Island Tracking
In real-time engines:
1. **Sleep States:** Sand that has settled into a stable dune goes to sleep. Inactive chunks or sub-blocks are completely skipped during tick updates.
2. **Awakening Triggers:** A sleeping cell is awakened if:
   - A neighboring voxel is removed (carved, mined, or destroyed).
   - An active moving voxel collides with or lands on it.
   - An explosion or kinematic player controller penetrates its bounding box.
3. **Dirty Bitmasks:** Track active cells using a 64-bit mask per leaf brick. If `active_mask == 0`, the entire brick is skipped in 1 instruction.

---

## 4. Avoiding Directional Bias & Race Conditions

### 4.1 The Ping-Pong Double Buffer
To prevent an updating voxel from falling through multiple cells in a single frame (infinite speed artifact), CA updates must read from buffer $A$ (tick $t$) and write to buffer $B$ (tick $t+1$).

### 4.2 Checkerboard / Red-Black Sub-Stepping on GPUs
In parallel GPU compute implementations:
- If two neighboring threads simultaneously try to move sand into the same empty cell, an atomic collision race occurs.
- **Solution:** 3D Red-Black (Checkerboard) sub-stepping:
  - Step 1: Update cells where $(x + y + z) \pmod 2 == 0$ (Red cells).
  - Step 2: Update cells where $(x + y + z) \pmod 2 == 1$ (Black cells).
  - Guarantees that no two adjacent threads read and write to the same cell simultaneously, eliminating race conditions with **zero atomic locks**.

---

## 5. Performance Budget on Modern Hardware

| Implementation | Update Rate | Max Active Dynamic Voxels | Frame Budget (ms) |
|---|---|---|---|
| **CPU Single-Threaded** | 60 Hz | 50,000–100,000 | 4.0–8.0 ms |
| **CPU Multi-Threaded (Rayon)** | 60 Hz | 500,000–1,200,000 | 1.5–3.0 ms |
| **GPU Compute (Red-Black)** | 60 Hz | **10,000,000+** | **0.4–0.9 ms** |
