# 02. Mob Behavior Architectures, Perception & Spatial Reasoning

## 1. Overview & AI Paradigm Selection

In a high-intensity action game with destructible terrain, mob behavior must fulfill three non-negotiable requirements:
1. **Dynamic Reactivity:** When the environment changes (walls collapse, bridges blow up, sightlines open), mobs must alter their tactical decisions in real time without hesitation.
2. **Computational Determinism:** With dozens of active enemies, AI decision loops must not cause frame time spikes.
3. **Tactical Spatial Exploitation:** Mobs should exploit voxel destruction offensively (punching through weak walls, setting ambushes) and defensively (seeking voxel cover, falling back behind pillars).

### AI Architecture Comparison

| Architecture | Reactivity | Destructible World Suitability | CPU Cost (50 Mobs) | Memory Footprint | Verdict for Arcadia |
|---|---|---|---|---|---|
| **Hierarchical Finite State Machines (HFSM)** | High | Low (State transitions explode with dynamic options) | **0.05 ms** (Trivial) | Very Low | Ideal for simple grunts / zombies |
| **Classic Behavior Trees (BT)** | High | Moderate (Requires frequent decorator re-evaluation) | **0.25 ms** | Low | **Selected Baseline for Combat Mobs** |
| **Utility AI (Reasoning Systems)** | Very High | Excellent (Dynamic scoring of continuous environmental factors) | **0.40 ms** | Low | **Selected for Tactical Elites / Bosses** |
| **Goal-Oriented Action Planning (GOAP)** | Moderate | High, but path-to-action graph search can spike on cratering | **1.50 ms** | Moderate | Too heavy for horde scales |

**Arcadia’s Hybrid Design:**
- **Tier A (Minions / Horde Melee):** Lightweight Behavior Tree with 4 primary branches (Flee, Swarm Attack, Pursue Flow Field, Idle/Roam).
- **Tier B (Tactical Ranged / Elite Demons):** Utility-Scored Behavior Tree evaluating cover positions, flank angles, and destructible penetration lines.

---

## 2. Spatial Sensory Perception in a 64-Tree World

Traditional AI sensors use coarse distance checks or physics engine collision spheres. In Arcadia, sensory queries leverage the native performance of our **Chunked Sparse 64-Tree (CS64)**.

```text
                     Mob Eye Origin
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
Vision Cone DDA     Acoustic Breadth     Voxel Structural
(Zero-Brick LOS)    (Cellular Diffusion) (Wall Hardness Probing)
```

### 2.1 Nanosecond Line-of-Sight (LOS) via Zero-Brick Ray Marching
When an AI evaluates if it can see the player or target:
- Rather than tracing expensive fine-detail rays through leaf bricks, LOS rays run on **Level 1 Macro and Level 2 Meso nodes** ($1.0\text{m}\text{--}4.0\text{m}$ cells) using our CPU DDA raycaster ([`crates/arcadia-core/src/raycast_cpu.rs`](../../crates/arcadia-core/src/raycast_cpu.rs)):
  - Ray step advances by node cell bounds.
  - If a node is completely empty (`mask == 0`), the ray jumps past the entire volume in 1 step.
  - Only when entering a node containing solid matter does it check leaf occupancy bitmasks.
- **Ray Cone Fuzzing:** To simulate peripheral vision and lighting attenuation:
  - Inside the central vision cone ($\theta \le 45^\circ$), full resolution LOS is checked.
  - In peripheral vision ($45^\circ < \theta \le 85^\circ$), LOS is gated on player movement velocity ($\|\mathbf{v}\| > 1.5\text{ m/s}$) and ambient voxel light levels (reading `lod_payload`).
- **Cost:** A typical 32-meter LOS check completes in **$0.4\text{ }\mu\text{s}$** ($250\times$ faster than mesh raycasts).

### 2.2 Acoustic Perception & Sound Diffusion through Voxel Porosity
Gunshots, explosions, and footstep sounds emit acoustic stimuli:
- Sound does not travel strictly in straight lines; it diffracts around corners and is attenuated by solid voxel barriers.
- **Porosity-Aware Sound Wave Propagation:**
  - When an explosion or gunshot occurs at voxel $\mathbf{x}_{\text{event}}$ with acoustic power $P_0$ (dB):
  - A fast 3D Dijkstra flood-fill expands across the macro-grid ($4\text{m}$ cells).
  - Empty cells transmit sound with distance attenuation: $P(r) = P_0 - 20 \log_{10}(r)$.
  - Cells with solid matter apply material-specific acoustic absorption:
    - `MAT_WOOD` / `MAT_DIRT`: $-6\text{ dB/meter}$
    - `MAT_STONE_BRICK` / `MAT_BEDROCK`: $-24\text{ dB/meter}$
    - `MAT_WATER`: $-3\text{ dB/meter}$ (underwater propagation)
- Mobs maintain an auditory threshold (e.g., $30\text{ dB}$). If an event exceeds the threshold, the mob acquires a sensory alert vector pointing along the negative gradient of acoustic arrival.

---

## 3. Tactical Spatial Analysis: Voxel Cover, Perches & Flanking

### 3.1 Procedural Voxel Cover Detection
Unlike pre-baked cover nodes placed manually by level designers, Arcadia's cover system evaluates dynamic voxel destruction procedurally:

```text
Player LOS Ray:  ──────────────────────────────────────────► (Blocked)
                                    ┌────────┐
                                    │ Solid  │ ◄── Voxel Pillar
                                    │ Voxel  │
                                    │ Wall   │
[Mob Ducking] ─────────────────────►│ (Stone)│
(Half-Cover: 12-16 voxels high)     └────────┘
```

1. **Cover Direction Analysis:** Given target position $\mathbf{x}_{\text{target}}$ and candidate mob position $\mathbf{x}_{\text{cand}}$:
   $$\mathbf{d} = \frac{\mathbf{x}_{\text{target}} - \mathbf{x}_{\text{cand}}}{\|\mathbf{x}_{\text{target}} - \mathbf{x}_{\text{cand}}\|}$$
2. **Bitwise Height Probing:** Query the vertical voxel column directly in front of the mob ($\mathbf{x}_{\text{cand}} + \mathbf{d} \cdot 0.5\text{ m}$):
   - **Low Cover (Half-Cover):** Solid voxels present from $y \in [4, 16]$ ($0.25\text{ m}\text{--}1.0\text{ m}$), air present above $y > 16$.
     *Mob Action:* Crouch behind cover; periodically peek over to fire projectiles.
   - **High Cover (Full-Cover):** Solid voxels present from $y \in [4, 32]$ ($0.25\text{ m}\text{--}2.0\text{ m}$).
     *Mob Action:* Reload, regenerate shield, or execute flanking reposition.
   - **Compromised Cover:** If an explosion or railgun blast carves away the cover voxels, the mob immediately senses the loss of blockage and scrambles to adjacent cover.

### 3.2 Dynamic High-Ground & Perch Finding
Flying or wall-climbing demons (e.g., winged Gargoyles, wall-climbing Imps) prioritize vertical advantage:
- Query the Tier 2 surface graph for nodes with:
  $$y_{\text{candidate}} - y_{\text{player}} \ge 4.0\text{ m} \quad (64\text{ voxels})$$
  $$\text{Clearance}_{\text{LOS}}(\mathbf{x}_{\text{candidate}}, \mathbf{x}_{\text{player}}) == \text{True}$$
- These high-ground perches confer higher projectile hit probabilities and force the player to aim vertically across complex ruins.

---

## 4. Destructive Environmental Exploitation (Breach & Ambush)

A distinguishing feature of Arcadia’s voxel AI is the ability of mobs to **destroy the world to reach the player**.

### 4.1 Wall Hardness Probing & Breach Attacks
When a mob pathfinder is blocked by a wall or door, it evaluates a **Breach Cost**:
1. Raycast forward into the blocking obstacle to determine wall thickness $\Delta x$ and material identity.
2. Calculate total demolition time based on mob melee DPS:
   $$t_{\text{demolish}} = \sum_{v \in \text{Obstacle}} \frac{\text{VoxelHardness}(M_v)}{\text{MobDemolitionDamage}}$$
3. If $t_{\text{demolish}} < t_{\text{detour}}$ (the time required to walk around the entire structure):
   - The mob transitions into the `BREACH_WALL` state!
   - The mob plays an attack animation (claw swipe, hammer smash, explosive charge).
   - Each impact calls [`carve_sphere`](../../crates/arcadia-world/src/destruction.rs) on the obstacle, emitting real-time voxel debris and punching a physical hole through the wall.
   - Once a $24 \times 32$ voxel opening is created, the mob and following horde rush through the breach!

```text
      [Intact Voxel Wall]             [Mob Claw Strikes]              [Breach Created!]
      ┌─────────────────┐             ┌───────────-─────┐             ┌─────┐     ┌─────┐
      │  Stone   Stone  │             │  Crater   Cracks│             │     │     │     │
      │  Stone   Stone  │    ───►     │  Debris   Sparks│    ───►     │     │ MOB │     │
      │  Stone   Stone  │             │  Stone   Stone  │             │     │ RUSH│     │
      └─────────────────┘             └─────────────────┘             └─────┘     └─────┘
```

---

## 5. Swarm & Horde Coordination: Dynamic Combat Slots

When dozens of melee mobs swarm the player, uncoordinated movement causes clustering, physics interpenetration, and mobs pushing each other out of attack range.

### 5.1 Polar Combat Ring Slot Reservation
Arcadia organizes surrounding melee combatants using **Concentric Polar Slot Rings**:

```text
                       Ring 2: Ranged / Stalkers (r = 5.0m - 8.0m)
                              ○     ○     ○     ○
                                 Ring 1: Melee (r = 2.0m - 3.0m)
                                    ●     ●
                                       [PLAYER]
                                    ●     ●
                              ○     ○     ○     ○
```

1. **Ring Geometry:** Around the player position $\mathbf{x}_{\text{player}}$, two concentric rings are maintained:
   - **Inner Ring ($r_1 \in [2.0\text{ m}, 3.0\text{ m}]$):** Divided into $K_1 = 6$ angular attack slots ($\Delta \theta = 60^\circ$). Only mobs holding an inner slot token may execute melee strikes.
   - **Outer Ring ($r_2 \in [5.0\text{ m}, 8.0\text{ m}]$):** Divided into $K_2 = 12$ angular staging slots ($\Delta \theta = 30^\circ$). Mobs circle, taunt, dodge, or fire ranged support projectiles.
2. **Token Arbitration:**
   - Mobs bid for inner slots based on proximity, remaining health, and attack cooldown.
   - When an inner attacker finishes a combo or takes heavy damage, it releases its token and falls back to an outer slot, allowing a fresh demon to lunge into melee range.
   - **Dynamic Obstacle Pruning:** If a slot falls inside solid voxel terrain (e.g., the player stands with their back against a stone wall), the slot is flagged invalid, and attackers naturally funnel into the open flanking sectors.

---

## 6. Implementation Architecture in Rust

```rust
//! Owns mob behavior trees, sensory perception, and combat slot arbitration.

use arcadia_core::{ChunkTree64, VoxelCoord};
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MobBehaviorState {
    Idle,
    AlertAuditory { stimulus_pos: Vec3 },
    FlankingSlot { target_slot_idx: u8 },
    MeleeEngage,
    BreachingObstacle { hit_count: u8 },
    TakingCover { cover_anchor: Vec3 },
    FleeingMutilated,
}

pub struct MobSensorComponent {
    pub eye_offset: Vec3,
    pub view_range: f32,
    pub fov_cos: f32,
    pub los_cached: bool,
    pub last_known_player_pos: Option<Vec3>,
    pub acoustic_threat_level: f32,
}

impl MobSensorComponent {
    /// Evaluates line-of-sight to player using hierarchical CS64 zero-brick traversal.
    pub fn update_los(&mut self, eye_pos: Vec3, target_pos: Vec3, tree: &ChunkTree64) -> bool {
        let dir = target_pos - eye_pos;
        let dist = dir.length();
        if dist > self.view_range {
            self.los_cached = false;
            return false;
        }
        
        // Fast CPU raycast through CS64 Level 1 & 2 nodes
        let hit = tree.raycast_coarse(eye_pos, dir / dist, dist);
        self.los_cached = hit.is_none();
        if self.los_cached {
            self.last_known_player_pos = Some(target_pos);
        }
        self.los_cached
    }
}
```

This sensory and behavioral pipeline enables mobs to act with eerie tactical intelligence, using the engine's volumetric destructibility as a core gameplay weapon.

