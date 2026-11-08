# 05. Horde Scaling, Multi-Tier LODs & Hybrid GPU Simulation

## 1. The Scaling Bottleneck: 1 Mob vs 100 Mobs

Supporting a single articulated voxel demon is straightforward. However, modern action combat demands swarms and hordes:
- **Scenario:** 50 to 100 active demons swarming through ruins.
- **Complexity Explosion:**
  - $100\text{ mobs} \times 20\text{ limbs} = 2,000\text{ dynamic limb OBBs}$.
  - If every primary camera ray (at $1920 \times 1080 = 2.07\text{ million pixels}$) performed 2,000 ray-box slab tests, rendering would take $> 45\text{ ms}$ per frame!
  - If 100 mobs updated forward kinematics, sensory vision rays, and 2-bone IK at 60 Hz on the CPU, simulation would consume $> 12\text{ ms}$.

To achieve our strict performance budget (**$< 2.5\text{ ms}$ CPU update**, **$< 1.5\text{ ms}$ GPU raymarch overhead**), Arcadia implements a **Hierarchical Simulation & Rendering LOD Architecture**.

---

## 2. Multi-Tier Simulation & Rendering LODs

```text
Distance from Camera:
0m                     15m                    40m                    100m+
├───────────────────────┼──────────────────────┼───────────────────────┤
│ Tier 0: Close Combat  │ Tier 1: Mid-Range    │ Tier 2: Far Horde     │ Tier 3: Dormant
│ • 60 Hz Sim & IK      │ • 30 Hz Sim (Stagger)│ • 10 Hz Flow Field    │ • Sleep State
│ • Full 20-Limb OBBs   │ • 5-OBB Compound     │ • 1-AABB Coarse Box   │ • Frustum Culled
│ • Micro-Voxel Carving │ • Meso-Node LOD Ray  │ • Macro-Node LOD Ray  │ • Zero GPU Cost
│ • ReSTIR GI Emissive  │ • Zero-Brick Bypass  │ • Zero-Brick Bypass   │
```

### Detailed Tier Specifications

| Feature | Tier 0: Close Combat ($< 15\text{ m}$) | Tier 1: Mid-Range ($15\text{--}40\text{ m}$) | Tier 2: Far Horde ($40\text{--}100\text{ m}$) | Tier 3: Background ($> 100\text{ m}$) |
|---|---|---|---|---|
| **Max Active Count** | Up to 12 mobs | Up to 40 mobs | Up to 150 mobs | Unlimited |
| **Sim Tick Rate** | 60 Hz (Every frame) | 30 Hz (Even/Odd interleaved) | 10 Hz (Staggered 6-frame loop) | 1 Hz or Event-driven |
| **Kinematic Rig** | Full 20-bone skeleton | 5 compound bones | 1 root transform | 0 (Static anchor) |
| **Procedural IK** | 2-bone foot & look-at IK | Disabled (Pose animation only) | Disabled | Disabled |
| **GPU Raymarch Geometry** | 20 local CS64 limb trees | 5 compound CS64 volumes | 1 coarse $1.0\text{ m}$ bounding tree | Frustum culled ($0\text{ rays}$) |
| **Ray Cone LOD** | Full leaf micro-voxels | Meso-node early exit (`shift = 2`) | Macro-node early exit (`shift = 4`) | — |
| **Destruction Model** | Full in-place micro-carving | Bounded radius carving | Health-bar reduction only | Invulnerable |

---

## 3. VRAM Optimization: Copy-On-Write (COW) Anatomy Templating

Storing 100 unique CS64 voxel models in GPU memory would waste hundreds of megabytes of VRAM. Arcadia uses a **Copy-On-Write (COW) Instancing Model**:

```text
Global GPU VRAM:
┌────────────────────────────────────────────────────────┐
│ Shared Read-Only Anatomy Template (Hellknight #0)      │
│ Base Head, Torso, Limbs, Bones, Internal Organs        │
└──────────────────────────┬─────────────────────────────┘
                           │ Referenced by 49 undamaged mobs
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
[Mob 1 (Undamaged)] [Mob 2 (Undamaged)] [Mob 3 (Undamaged)]
(0 extra VRAM)      (0 extra VRAM)      (0 extra VRAM)

                                                │ Shotgun Blast Strikes!
                                                ▼
                                        [Mob 4 (Damaged Torso)]
                                        Lazily clones Torso into
                                        Dynamic Slab Allocator!
```

### 3.1 Template Instancing
- All mobs of a given archetype (e.g., "Hellknight", "Gargoyle", "Stalker") share a single set of base node slices in `gpu_nodes` and `gpu_bricks`.
- Each mob instance simply uploads a uniform array of `MobLimbInstance` transforms pointing to the shared template node offsets:
  $$\text{VRAM per intact mob} = 20\text{ limbs} \times 128\text{ bytes} = 2.56\text{ KB}$$
- 100 intact mobs consume a negligible **$256\text{ KB}$** of GPU buffer memory!

### 3.2 Lazy Copy-On-Write on First Carve
- The moment a projectile, shotgun blast, or explosion damages a mob limb:
  1. The engine checks `limb.is_unique_instance`.
  2. If `false`, allocate a fresh contiguous span from `SlabAllocator<Node64>` ([`slab.rs`](../../crates/arcadia-core/src/slab.rs)).
  3. Clone the template limb’s nodes and bricks into the newly allocated span.
  4. Mark `limb.is_unique_instance = true` and update the GPU uniform `node_pool_offset`.
  5. Apply the carve operation to the private copy.
- **Result:** Undamaged limbs remain 100% instanced. Only the specific limbs that suffer physical damage consume dynamic VRAM.

---

## 4. GPU-Driven Limb Culling & Compacted Worklists

To prevent primary camera rays from testing against all 2,000 limb OBBs, Arcadia introduces a **GPU Compute Pre-Pass (Limb Culling & Compaction)** before the primary ray march:

```text
All 2,000 Limb OBBs in Scene
              │
              ▼
[Compute Pass: Frustum & Coarse Tile Depth Culling] (1D Dispatch: 256 threads/group)
              │
              ▼ (Pass: atomicAdd to visible_count)
Compacted Visible Limb Array (e.g., 64 - 120 limbs)
              │
              ▼
Primary Raymarch (Pass 1) tests ONLY Compacted Visible Limbs!
```

### 4.1 Culling Criteria in Compute Shader
Each thread evaluates one limb OBB:
1. **Frustum Test:** Transform the 8 OBB corner vertices to clip space; test against the 6 camera frustum planes.
2. **Coarse Tile Depth Culling:** Compare the limb's nearest depth $t_{\text{near}}$ against the coarse tile depth buffer ([`01_high_leverage_upgrades_for_cs64.md` Upgrade 4](../07_ideas_and_architectural_synthesis_for_arcadia/01_high_leverage_upgrades_for_cs64.md)). If the entire limb lies behind a solid stone cliff, discard it.
3. **Stream Compaction:** If visible, append the limb’s uniform index to `visible_limb_indices` via a fast workgroup-shared atomic counter:
   ```wgsl
   var<workgroup> local_count: u32;
   // Shared prefix sum -> single atomicAdd to global buffer
   ```

### 4.2 Impact on Primary Ray Marching
Instead of 2,000 slab tests per pixel, primary rays test only **$\sim 80\text{--}120$ visible limbs**. Furthermore, rays only test limbs whose screen-projected bounding boxes overlap the ray's current $16 \times 16$ screen tile, slashing limb traversal overhead to **$< 0.8\text{ ms}$**.

---

## 5. Performance Budget Summary (100 Active Mobs)

```text
CPU Simulation Budget (16.6 ms target):
├── Pathfinding (Tier 2 Dijkstra Flow Field):   0.80 ms (Worker Thread)
├── Behavior & Sensory Updates:                0.35 ms (Worker Thread)
├── Forward Kinematics & Procedural IK:        0.40 ms (Worker Thread)
├── In-Place Damage Carving (5 active events): 0.15 ms (Main Thread)
└── Uniform Buffer Upload (write_buffer):      0.08 ms (Main Thread)
TOTAL CPU COST:                                1.78 ms (Well under 2.5 ms budget!)

GPU Render Budget (8.0 ms target):
├── Limb Culling & Compaction Pre-Pass:        0.12 ms
├── Primary G-Buffer Raymarch (Compacted Limbs):0.75 ms
├── ReSTIR GI & Emissive Weakpoint Gathering:   0.30 ms
└── Temporal Motion Vector Resolve:            0.10 ms
TOTAL GPU OVERHEAD:                            1.27 ms (Within 1.5 ms budget!)
```

This multi-tier pipeline guarantees smooth 60+ FPS performance even during apocalyptic demon horde encounters.

