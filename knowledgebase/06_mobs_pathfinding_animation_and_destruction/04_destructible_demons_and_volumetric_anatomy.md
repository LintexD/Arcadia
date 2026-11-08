# 04. Volumetric Anatomy & Dynamic Carving: The Next-Gen "Doom Eternal" System

## 1. The Paradigm Shift: id Tech 7 vs Native Volumetric CS64

In *DOOM Eternal* (id Tech 7), id Software introduced the **"Destructible Demons"** system, providing unprecedented kinetic feedback. However, because id Tech 7 is a traditional rasterization engine bound to triangle meshes, the implementation was constrained by structural limitations:

```text
DOOM Eternal (id Tech 7)                   Arcadia Voxel Engine (CS64)
─────────────────────────────────────      ─────────────────────────────────────
• Multi-layered surface triangle meshes    • True volumetric continuous 3D matter
• Pre-authored damage zone boundaries      • Pixel-precise arbitrary carving
• Alpha cutout blending & texture swaps    • Real multi-material core (Bone, Meat, Core)
• Fixed, pre-rigged dismemberment points   • Arbitrary topological severing via islands
• Hollow interior artifacts upon deep cut  • Solid interior with multi-bounce GI
```

In Arcadia, **voxels are inherently volumetric**. When a projectile strikes an enemy, we do not fade in a red bloody texture or swap a LOD mesh. **We physically excise 3D matter from the mob's anatomy**, revealing deeper biological layers and dynamically triggering physical dismemberment.

---

## 2. Layered Multi-Material Voxel Anatomy

Every demon or mob in Arcadia is authored with **Concentric Anatomical Layers** mapped to our 256-entry PBR + Emissive Material Palette ([`material.rs`](../../crates/arcadia-core/src/material.rs)):

```text
                  Demonic Torso Cross-Section (Top-Down View)
      
                     ┌─── Outer Carapace: Chitin / Armor Plate
                     │      ┌─── Subcutaneous Muscle & Flesh
                     │      │      ┌─── Endoskeleton: Bone Core
                     ▼      ▼      ▼
                  ┌──────┬──────┬──────┬──────┬──────┐
                  │ ████ │ ░░░░ │ ▓▓▓▓ │ ░░░░ │ ████ │
                  │ ████ │ ░░░░ │ ▓▓▓▓ │ ░░░░ │ ████ │
                  │ ████ │ ░░░░ │  ●   │ ░░░░ │ ████ │ ◄── Pressurized Hellfire Core
                  │ ████ │ ░░░░ │ ▓▓▓▓ │ ░░░░ │ ████ │     (Emissive Organ)
                  │ ████ │ ░░░░ │ ▓▓▓▓ │ ░░░░ │ ████ │
                  └──────┴──────┴──────┴──────┴──────┘
```

### 2.1 The Four Anatomical Strata

| Layer | Material Names | Material Properties | Kinetic Function | Damage Reaction |
|---|---|---|---|---|
| **Layer 1: Carapace / Armor** | `MAT_DEMON_CHITIN`, `MAT_OBSIDIAN_PLATE` | Hardness: $0.85$, Metallic: $0.2$, Roughness: $0.4$ | Absorbs light ballistic impacts; deflects low-caliber rounds | Chips off in brittle fragments; exposes Layer 2 |
| **Layer 2: Muscle & Flesh** | `MAT_DEMON_FLESH`, `MAT_GORE_SOFT` | Hardness: $0.20$, Metallic: $0.0$, Roughness: $0.8$, Subsurface scattering | Connective soft tissue; bleeds upon puncture | Readily carved by buckshot and explosives; spurts voxel blood |
| **Layer 3: Endoskeleton** | `MAT_DEMON_BONE`, `MAT_STEEL_BONE` | Hardness: $0.95$, Dense structural support | Structural backbone of kinematic limb | High resistance; when severed, limb physics activates |
| **Layer 4: Pressurized Organs** | `MAT_HELL_CORE`, `MAT_MAGMA_GLAND` | Emissive: $8.0\text{--}15.0\text{ W/m}^2$, High volatility | Vital core / Weak point | Piercing exposes core to ReSTIR GI; detonation triggers gibbing |

---

## 3. Real-Time Volumetric Weapon Carving

When a projectile collides with a mob limb, Arcadia applies localized **Zero-Repack Sculpting** ([`crates/arcadia-core/src/chunk_tree/edit/carve.rs`](../../crates/arcadia-core/src/chunk_tree/edit/carve.rs)) directly to the limb's local `ChunkTree64`:

```text
Shotgun Blast (Multiple Spherical Cavities):
     Before Impact                          After Impact
   ┌────────────────┐                     ┌───┐        ┌───┐
   │ Chitin Carapace│                     │   │ Raw    │   │
   │ Flesh & Muscle │   ──────►           │   │ Muscle │   │
   │ Bone Core      │                     └───┤        ├───┘
   └────────────────┘                         │ Bone   │ ◄── Visible Bone!
                                              └────────┘
```

### 3.1 Weapon Carve Signatures

1. **Shotgun Buckshot (Conical Pellet Cluster):**
   - Fires 8–12 pellets, each carving a small sphere of radius $r \in [1, 2.5]$ voxels ($6.25\text{ cm}\text{--}15.6\text{ cm}$).
   - **Result:** Shreds external chitin plating over a broad area, shredding skin and exposing weeping demon muscle.
2. **High-Caliber Railgun / Sniper (Cylindrical Ray Carve):**
   - Carves a continuous cylinder of radius $r = 1.5\text{ voxels}$ straight along the projectile trajectory.
   - **Result:** Punches a hole clean through a demon’s torso. The player can literally look through the entrance wound and see terrain or enemies behind it!
3. **Plasma / Energy Bolt (Thermal Melting):**
   - Carves a sphere and sets surrounding boundary voxels to `MAT_EMBER_SCORCH` with active emissive heat values that cool over time.
   - **Result:** Glowing, molten charred rims around gaping wounds.
4. **Rocket / High-Explosive Shell (Blast Cavitation):**
   - Large sphere carve ($r = 4\text{--}8\text{ voxels}$) accompanied by radial impulse that strips away soft tissue and fractures bone.

---

## 4. Real-Time Structural Dismemberment via Bitwise Island Detection

The crowning achievement of Arcadia’s voxel anatomy is **Dynamic Topological Severance**:

In traditional games, dismemberment can only occur at pre-rigged joint seams (wrist, elbow, shoulder). In Arcadia, **a demon can be severed at any arbitrary voxel plane along its body**.

```text
                 Intact Bone Core                       Severed Bone Core
                 ┌──────────────┐                       ┌─────┐      ┌─────┐
                 │  Bone Core   │                       │Bone │ Void │Bone │
                 │  (Continuous)│      ──────►          │Core │ Crater│Core│
                 └──────────────┘                       └─────┘      └─────┘
                        │                                          │
                        ▼                                          ▼
                Kinematic Joint Valid                      STRUCTURAL COLLAPSE!
                                                           Distal Part Detaches
                                                           as 6-DOF Rigid Body!
```

### 4.1 Bone Anchor Flood-Fill Protocol
We adapt the structural island detection algorithm from [`islands.rs`](../../crates/arcadia-world/src/islands.rs):
1. **Anchor Definition:** In each limb’s local `ChunkTree64`, the proximal voxel interface connected to the parent bone joint is tagged as the **Structural Anchor**.
2. **Post-Carve Connectivity Check:** Whenever carving touches a limb:
   - Run a localized bitwise 6-connected flood fill starting from the proximal anchor voxels.
   - Traverse exclusively through structural matter (`MAT_DEMON_BONE` and dense muscle).
3. **Island Identification:**
   - If the flood-fill successfully reaches the distal end of the limb, structural integrity holds (the arm is damaged, but remains attached).
   - If a continuous slice of void/air separates the distal segment from the anchor, an **Orphaned Voxel Island** is detected!

### 4.2 Dynamic Physical Decoupling
Once an island is confirmed:
1. **Extract Island Voxels:** Clear the disconnected voxels from the mob’s kinematic limb tree in a single turn.
2. **Spawn Dynamic Rigid Body:** Convert the orphaned voxels into a dynamic [`RigidVoxelBody`](../../crates/arcadia-physics/src/rigid_voxel.rs):
   - Center of mass and discrete $3 \times 3$ moment of inertia tensor are evaluated via parallel-axis math ([`03_dynamic_rigid_body_voxels.md`](../03_simulation_and_physics/03_dynamic_rigid_body_voxels.md)).
   - Initial linear velocity: Inherits parent bone velocity $\mathbf{v}_{\text{bone}} + \mathbf{v}_{\text{projectile\_impulse}}$.
   - Initial angular velocity: $\boldsymbol{\omega}_{\text{bone}} + (\mathbf{r} \times \mathbf{J}) \mathbf{I}^{-1}$.
3. **Kinetic Decoupling:** The severed forearm, horn, or lower leg spins through the air, collides with walls and floors, tumbles down staircases, and eventually bakes into resting world rubble.
4. **GPU Debris Emission:** The severance plane emits a burst of $150\text{--}300$ blood and flesh debris voxels using Arcadia's existing 16k GPU debris particle pipeline.

---

## 5. Dynamic Weak Points & Combat Feedback

### 5.1 Destructible Demon Weaponry
Heavy demons carry mounted weapons (e.g., cybernetic plasma turrets or oversized obsidian cleavers):
- Weapon components are modeled as peripheral bones/limbs with their own local voxel trees.
- By concentrating fire on the weapon mounts, players carve away the structural brackets.
- Upon destruction, the weapon falls to the ground as a dynamic physics prop, permanently stripping the demon of its primary ranged attack!

### 5.2 Emissive Hellfire Core Meltdown
When heavy armor is blasted off a demon’s chest, its internal `MAT_HELL_CORE` is exposed to world space:
1. **Dynamic Lighting Integration:**
   - As soon as the outer opaque voxels are carved away, the core’s high emissive radiance ($\Phi = 12.0\text{ W/m}^2$) escapes into the world.
   - Arcadia’s **ReSTIR GI pass** ([`lighting_restir_gi.wgsl`](../../shaders/lighting_restir_gi.wgsl)) instantly picks up the newly exposed emissive surface, casting dramatic pulsing red and orange multi-bounce illumination across the surrounding colonnade and floor.
2. **Catastrophic Detonation:**
   - Hitting an exposed core triggers a critical hit multiplier ($3.5\times$).
   - If destroyed, the core explodes with a 2.5-meter radius spherical carve, vaporizing the remaining torso and instantly gibbing the demon into dozens of kinetic voxel fragments!

### 5.3 Anatomical Vitality: The Zero-HP Organ Destruction Paradigm (Brain & Heart Lethality)

A revolutionary advantage of native volumetric voxels is the ability to **completely eliminate abstract "Health Points" (HP bars)** and artificial damage-sponge scaling. Instead, mob lethality is governed strictly by **Direct Physical Organ Failure**:

```text
                  [Precision Sniper / Railgun Shot]
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ Cranial Bone Armor: MAT_DEMON_BONE (Carved)   │
         │       ┌──────────────────────────────┐       │
         │       │ Brain Voxel Core:            │       │
         │       │ MAT_VITAL_BRAIN              │       │
         │       │ (Direct Piercing Strike!)    │       │
         │       └──────────────┬───────────────┘       │
         └──────────────────────┼───────────────────────┘
                                │
                                ▼
         Neurological Integrity Drops Below Threshold (≤ 40%)
                                │
                                ▼
             INSTANT SHUTDOWN: XPBD Ragdoll Collapse!
```

1. **Vital Organ Voxel Clusters:**
   - **Cranial Brain (`MAT_VITAL_BRAIN`):** A dense cluster of $30\text{--}80$ voxels housed centrally within the `Head` bone volume, protected by a thick cranium of `MAT_DEMON_BONE`.
   - **Thoracic Cardiac Core (`MAT_VITAL_HEART` / `MAT_HELL_CORE`):** A cluster of $60\text{--}150$ voxels positioned behind the sternum and rib cage in the `Torso` bone volume.
2. **Deterministic Lethality Condition:**
   - Rather than subtracting abstract numerical damage from a floating scalar health pool:
     $$\text{Integrity}_{\text{organ}} = \frac{N_{\text{voxels}}(\text{MAT\_VITAL}, t)}{N_{\text{voxels}}(\text{MAT\_VITAL}, t=0)}$$
   - **Instant Brain Death ($\text{Integrity}_{\text{brain}} \le 0.40$ or Brainstem Severance):**
     - Immediate cessation of motor signals. The kinematic skeleton disables all joint motor drives, instantly converting the mob into an active articulated **XPBD ragdoll** that collapses under gravity, tumbling down steps or slamming against walls.
   - **Cardiac Hemorrhagic Shock ($\text{Integrity}_{\text{heart}} \le 0.30$):**
     - Massive arterial pressure loss. The chest wound continuously emits high-velocity voxel blood streams. The demon enters an agonizing $1.5\text{--}3.0\text{ s}$ terminal stagger state before succumbing to complete circulatory collapse.
3. **Decoupled Kinetic Damage:**
   - Damage to non-vital structures (shoulders, thighs, horns, armor) causes **mechanical disablement** (amputation, limping, loss of weapon handling) rather than chipping away at an artificial health bar.
   - A player can shoot a demon in the calves or biceps 50 times without killing it, whereas a single high-velocity railgun slug or buckshot core that punctures the skull or chest cavity drops the demon instantly.

---

## 6. Post-Mutilation Behavioral Transitions

Unlike traditional games where damaged enemies maintain standard run cycles, Arcadia’s mobs adapt their behavior dynamically based on anatomical mutilation:

```text
       Damage Event                         Behavioral Adaptation
───────────────────────────       ──────────────────────────────────────────────
Severed Right Arm (Weapon)  ───►  Drops weapon; switches to left-hand claw or bite
Severed Left Leg            ───►  Loses bipedal locomotion; collapses to floor;
                                  enters 1-legged crawling state
Exposed Hellfire Core       ───►  Enters berserk state; sprints directly at player
                                  to detonate in suicide blast
Bilateral Leg Amputation    ───►  Stationary turret mode; spits ranged venom
```

This dynamic coupling of **volumetric carving**, **structural physics**, and **tactical AI** elevates combat far beyond scripted animations, delivering an intensely physical, kinetic gameplay experience.

