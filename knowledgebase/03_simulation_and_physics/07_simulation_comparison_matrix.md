# 07. Simulation Comparison Matrix

## 1. Quantitative Benchmark & Performance Matrix

The following table compares simulation methods across key real-time engineering parameters targeting interactive 60+ FPS performance:

| Simulation Method | Memory Overhead | GPU Compute Cost (ms) | Scaling Limit (Active Units) | Best Use Case in Voxel Engines |
|---|---|---|---|---|
| **3D Cellular Automata (CA)** | $1\text{--}4\text{ bytes}$ / voxel | **$0.4\text{--}0.9\text{ ms}$** | $10,000,000+$ cells | Falling sand, gravel, granular liquids, fire spread |
| **Bitwise Flood-Fill (BFS)** | Temporary bitmask buffer | **$0.2\text{--}0.8\text{ ms}$** | $50,000+$ bricks / frame | Structural integrity check, Teardown-style collapse |
| **Dynamic Voxel Rigid Bodies**| Local CS64 tree per body | **$0.5\text{--}1.5\text{ ms}$** | 100–500 rigid bodies | Large architectural debris, falling pillars |
| **Lattice Boltzmann (LBM D3Q19)**| $80\text{ bytes}$ / cell | **$1.1\text{--}1.8\text{ ms}$** | $128^3$ grid ($2\text{M}$ cells) | Cinematic smoke, gas, turbulent water vortices |
| **Eulerian Fluid (Stable Fluids)**| $20\text{ bytes}$ / cell | **$2.5\text{--}4.5\text{ ms}$** | $128^3$ grid | Smooth water bodies, pressure-driven liquid flow |
| **Real-Time MLS-MPM** | $64\text{ bytes}$ / particle | **$1.8\text{--}3.2\text{ ms}$** | $300,000$ particles | Realistic snow avalanches, wet mud, clay deformation|
| **Thermal Diffusion Automata**| $2\text{ bytes}$ / voxel | **$0.1\text{--}0.3\text{ ms}$** | Active burning regions | Fire propagation, scorched ember rims |

---

## 2. Architectural Synthesis & Recommendations

### 2.1 The Two-Phase Destruction Paradigm
For real-time destructive micro-voxel games:
1. **Macro Collapse (Structural Integrity):** Use **Two-Tiered Coarse-to-Fine Bitwise Flood-Fill** to instantly detect unsupported structures. Detached geometry converts into dynamic rigid bodies with bitwise OBB collision.
2. **Micro Debris (High-Speed Fragments):** Explosions blast surface voxels into lightweight **GPU compute debris particles** (with simple AABB bounce physics) rather than full rigid bodies, scaling up to $65,000+$ simultaneous flying fragments without frame drops.

### 2.2 Granular Sand & Liquid Integration
For falling sand, gravel, and fluid:
- Use **3D Red-Black Checkerboard Cellular Automata** directly on the CS64 occupancy masks.
- Active brick tracking ensures inactive, sleeping dunes consume zero processing power.
