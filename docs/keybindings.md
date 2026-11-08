# Arcadia Controls & Keybindings Reference

> Fast-reference guide for all keyboard and mouse controls in the Arcadia Voxel Engine.  
> Cross-reference: **[ARCHITECTURE.md](../ARCHITECTURE.md)** | **[docs/architecture/04_benchmarking_and_hud.md](architecture/04_benchmarking_and_hud.md)**

---

## 1. Quick Start: Essential Shortcuts

| Shortcut | Description |
|---|---|
| **`[F1]`, `[Tab]`, ``[`]``** | Toggle immediate-mode `DebugUi` overlay & release/lock mouse cursor |
| **`[F3]`** | Toggle rolling Frame Time Graph HUD overlay (180-frame pacing, jitter, 1% lows) |
| **`[Double-Space]`** | Toggle between Kinematic Walk Mode and Free Fly Mode (double-tap window < 0.35s) |
| **`[Left Click]`** | Explosive spherical voxel carve (or grab mouse cursor when unlocked) |
| **`[Right Click]`** | Spherical voxel brush placement using currently selected material |
| **`[Mouse Wheel]`** | Adjust sculpting brush radius ($0.25\text{ m}$ to $4.50\text{ m}$) |
| **`[1] – [8]`** | Instant material palette selection (Crystals, Lava, Slate, Moss, Point Light, Mirror) |
| **`[Escape]`** | Release cursor lock if grabbed; exit application if already unlocked |

---

## 2. Categorized Controls Reference

### Global Navigation & Diagnostic Overlays
These controls toggle real-time diagnostic tooling, the profiler, and cursor capture states.

| Key / Control | Context | Action | Implementation Reference |
|---|---|---|---|
| `[F1]` | Global | Toggle immediate-mode `DebugUi` overlay; unlocks mouse cursor when shown | [`crates/arcadia-app/src/gui.rs`](../crates/arcadia-app/src/gui.rs) |
| `[Tab]` | Global | Alternate shortcut to toggle immediate-mode `DebugUi` overlay | [`crates/arcadia-app/src/input.rs`](../crates/arcadia-app/src/input.rs) |
| ``[`]`` (Backquote) | Global | Alternate shortcut to toggle immediate-mode `DebugUi` overlay | [`crates/arcadia-app/src/input.rs`](../crates/arcadia-app/src/input.rs) |
| `[F3]` | Global | Toggle rolling Frame Time Graph HUD overlay (180-frame pacing, FPS, jitter rating) | [`crates/arcadia-app/src/frame_graph.rs`](../crates/arcadia-app/src/frame_graph.rs) |

---

### Player Kinematics & Motion
Arcadia features a dual-mode character controller: a physics-driven kinematic walker (with 3-voxel auto-step-up, gravity, and water buoyancy) and an unrestricted 6-DOF fly mode.

| Key / Control | Context | Action | Details |
|---|---|---|---|
| `[W]` | Motion | Move forward / Fly forward | In walk mode, projects along horizontal forward vector |
| `[S]` | Motion | Move backward / Fly backward | In walk mode, projects along horizontal backward vector |
| `[A]` | Motion | Strafe left / Fly left | Orthogonal lateral strafe |
| `[D]` | Motion | Strafe right / Fly right | Orthogonal lateral strafe |
| `[Double-Space]` | Motion | Toggle Walk / Fly mode | Double-tap within `< 0.35s` switches mode; resets velocity to zero |
| `[Space]` | Motion | Jump / Fly upward | Jump in Walk Mode; ascend vertically in Fly Mode |
| `[Shift]` | Motion | Fly downward | Descend vertically along `-Y` (Fly Mode only) |
| `[Ctrl]` | Motion | Sprint boost | Increases movement velocity (up to $38\text{ m/s}$ in Fly Mode) |

> [!TIP]
> **Auto-Step-Up:** When walking, the kinematic controller automatically steps up obstacles between $1$ and $3$ voxels high ($6.25\text{ cm} - 18.75\text{ cm}$) without jumping. Moving into water automatically activates buoyancy and hydrodynamic viscous drag.

---

### World Sculpting & Destruction
Arcadia supports real-time destruction via direct in-place CS64 leaf brick mutations with zero full-tree re-allocations.

| Key / Control | Context | Action | Details |
|---|---|---|---|
| `[Left Click]` | Sculpting | Explosive spherical voxel carve | Carves a spherical crater; triggers rim scorch, GPU debris, and structural island checks |
| `[Right Click]` | Sculpting | Spherical voxel brush placement | Places solid spheres of the selected material at the raycast hit point |
| `[Mouse Wheel]` | Sculpting | Adjust brush radius | Scaled dynamically in $[0.25\text{ m}, 4.50\text{ m}]$ ($4$ to $72$ voxels diameter) |
| `[G]` | Sculpting | Toggle scorch / ember blast rim | Enables/disables scorched stone and glowing ember rings around carve craters |
| `[I]` | Sculpting / Physics | Toggle structural island detachment | Toggles `island_detection_enabled` (severed geometry detaches into 6-DOF rigid bodies) |

> [!NOTE]
> When `island_detection_enabled` is active (`[I]`), carving through supports triggers a targeted 2D surface perimeter flood-fill around the crater. Unanchored geometry detaches into dynamic 6-DOF rigid voxel bodies. If disabled, small crater rim crumbs ($\le 8$ voxels) still convert into GPU debris to keep crater edges crisp.

---

### Material Palette Selection
Select active voxel materials for brush placement (`[Right Click]`). Materials mirror the 256-entry GPU material palette.

| Key | Material Name | Material ID | Visual & Physical Characteristics |
|---|---|---|---|
| `[1]` | Cyan Crystal | `MAT_CRYSTAL_CYAN` | Emissive cyan glow, ReSTIR GI dynamic light source |
| `[2]` | Magenta Crystal | `MAT_CRYSTAL_MAGENTA` | Emissive magenta glow, ReSTIR GI dynamic light source |
| `[3]` | Amber Sunstone | `MAT_CRYSTAL_AMBER` | Warm radiant amber glow, ReSTIR GI dynamic light source |
| `[4]` | Molten Lava | `MAT_LAVA_MAGMA` | High-temperature emissive liquid stone, dynamic GI emitter |
| `[5]` | Carved Ruin Slate | `MAT_SLATE_CARVED` | Ancient carved stone masonry, rough diffuse PBR |
| `[6]` | Lush Ruin Moss | `MAT_MOSS_VIBRANT` | Organic vegetation layer with rich green diffuse reflectance |
| `[7]` | Point Light Beacon | `MAT_POINT_LIGHT` | Spawns an analytical shadowed point light source at placement |
| `[8]` | Mirror Chrome | `MAT_MIRROR_CHROME` | 100% specular mirror reflectance with continuation ray tracing |

---

### Diurnal Lighting & Sun Controls
Arcadia computes real-time soft sun shadows and multi-bounce ReSTIR global illumination driven by dynamic sun vectors.

| Key / Control | Context | Action | Details |
|---|---|---|---|
| `[Q]` | Lighting | Rotate sun counter-clockwise | Continuous smooth diurnal orbit progression along azimuth/elevation |
| `[E]` | Lighting | Rotate sun clockwise | Continuous smooth diurnal orbit progression along azimuth/elevation |
| `[N]` | Lighting | Cycle discrete Time-of-Day presets | Cycles through **Day** ($\sin \theta > 0.25$), **Sunset** ($\sin \theta \approx 0.04$), and **Midnight** ($\sin \theta \approx -0.2$) |

---

### Quality & Presentation Controls
Adjust temporal accumulation, anti-aliasing, and dithering pipelines on the fly.

| Key / Control | Context | Action | Details |
|---|---|---|---|
| `[C]` | Quality | Run 40-frame hardware auto-calibration | Measures GPU frame throughput and calibrates optimal render scale & ray budgets for 60 FPS |
| `[L]` | Quality | Toggle 30 FPS / 60 FPS calibrated presets | Cycles between **Calibrated 60 FPS Fluid** (48% scale) and **Calibrated 30 FPS Fidelity** (58% scale) |
| `[K]` | Presentation | Cycle dithering modes | Cycles: **PCG 2D Decorrelated TPDF** (default) $\to$ **Spatial Integer Bit-Mix** $\to$ **Disabled** |
| `[J]` | Presentation | Cycle camera jitter modes | Cycles: **Freeze when Still** (zero-flicker dynamic TSR) $\to$ **Always Off** $\to$ **Continuous Halton** |

> [!TIP]
> **Dithering Recommendation:** Keep dither mode on `PCG 2D Decorrelated TPDF` (`[K]`). This mode temporally rotates blue noise over time to completely eliminate dark banding in nocturnal sky gradients, specular glints, and deep caverns without visible pattern crawlers.

---

### Benchmarking & Hardware Profiling

| Key / Control | Context | Action | Details |
|---|---|---|---|
| `[B]` | Benchmark | Start / Restart Sanctuary Benchmark | Launches deterministic pure render benchmark with Catmull-Rom spline camera flythrough; logs performance report to stdout and JSON |

---

### Window & System Controls

| Key / Control | Context | Action | Details |
|---|---|---|---|
| `[Escape]` | Window | Release cursor / Exit | Releases cursor lock if grabbed; exits the application immediately if cursor is already unlocked |

---

## 3. Master Keybindings Quick Reference Table

| Key / Control | Context | Action |
|---|---|---|
| `[F1]`, `[Tab]`, ``[`]`` | Global | Toggle immediate-mode `DebugUi` overlay & release/lock mouse cursor |
| `[F3]` | Global | Toggle rolling Frame Time Graph HUD overlay |
| `[W, A, S, D]` | Motion | Walk / Strafe / Fly forward, back, left, right |
| `[Double-Space]` (< 0.35s) | Motion | Toggle between Kinematic Walk Mode and Free Fly Mode |
| `[Space]` | Motion | Jump (Walk Mode) / Fly Up (Fly Mode) |
| `[Shift]` | Motion | Fly Down (Fly Mode) |
| `[Ctrl]` | Motion | Sprint boost (up to $38\text{ m/s}$ in Fly Mode) |
| `[Left Click]` | Sculpting | Explosive spherical voxel carve (or grab cursor if unlocked) |
| `[Right Click]` | Sculpting | Spherical voxel brush placement of selected material |
| `[Mouse Wheel]` | Sculpting | Adjust brush radius ($0.25\text{ m}$ to $4.50\text{ m}$) |
| `[1] - [8]` | Palette | Select material (Cyan/Magenta/Amber Crystal, Lava, Slate, Moss, Point Light, Mirror) |
| `[Q] / [E]` | Lighting | Smooth continuous diurnal sun rotation |
| `[N]` | Lighting | Cycle discrete Time-of-Day lighting presets (Day / Sunset / Midnight) |
| `[B]` | Benchmark | Initiate or re-run the pure render benchmark |
| `[G]` | Sculpting | Toggle scorch / ember blast rim around craters |
| `[C]` | Quality | Trigger live 40-frame hardware auto-calibration |
| `[I]` | Physics | Toggle structural island detachment (`island_detection_enabled`) |
| `[K]` | Presentation | Cycle dithering modes (PCG 2D Decorrelated TPDF $\to$ Spatial Bit-Mix $\to$ Off) |
| `[J]` | Presentation | Cycle camera jitter modes (Freeze when Still $\to$ Always Off $\to$ Continuous Halton) |
| `[L]` | Quality | Toggle between Calibrated 60 FPS Fluid and Calibrated 30 FPS Fidelity presets |
| `[Escape]` | Window | Release cursor lock if grabbed; exit application if already unlocked |

---

## 4. Subsystem Cross-References

- **Executive Architecture:** [ARCHITECTURE.md](../ARCHITECTURE.md)
- **Benchmarking & HUD Profiler:** [docs/architecture/04_benchmarking_and_hud.md](architecture/04_benchmarking_and_hud.md)
- **Physics, Destruction & Rigid Bodies:** [docs/architecture/03_physics_and_destruction.md](architecture/03_physics_and_destruction.md)
- **Rendering & Lighting Pipeline:** [docs/architecture/02_rendering_and_lighting.md](architecture/02_rendering_and_lighting.md)
- **Storage & In-Place Mutation Engine:** [docs/architecture/01_storage_and_memory.md](architecture/01_storage_and_memory.md)
- **Phased Implementation History:** [docs/plans/implementation_history.md](plans/implementation_history.md)
- **Comprehensive Project Roadmap:** [docs/plans/ROADMAP_AND_TODO.md](plans/ROADMAP_AND_TODO.md)
