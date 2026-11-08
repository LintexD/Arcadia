# Benchmarking, Quality Presets & HUD Profiler Architecture

## Overview
Arcadia incorporates a standardized profiling suite, deterministic benchmark engine, dynamic hardware auto-calibration routine, and real-time rolling frame pacing diagnostics. These tools enable scientific measurement of compute shader optimizations and provide users and developers with interactive control over rendering fidelity and performance budgets.

---

## 1. Pure Render Benchmark Suite & Standardized Profiler

To measure and iterate on GPU ray marching and ReSTIR compute shader optimizations with scientific rigor:

### 1.1 Benchmark Sanctuary Demo Scene (`crates/arcadia-world/src/benchmark_scene.rs`)
The benchmark executes inside a dedicated $160\text{m} \times 64\text{m} \times 160\text{m}$ architectural environment engineered to stress all rendering subsystems simultaneously:
1. **Light Stress:** Soft sun shadows, 6 analytical 3D point lights, multi-bounce ReSTIR GI, molten lava cascades, and multi-colored emissive crystal spires.
2. **Water Stress:** Sunken crystal reflecting basin (`MAT_WATER_CRYSTAL`) with caustics on mosaic bed, Beer-Lambert light extinction, submerged crystal glow, and wave-perturbed Fresnel reflections.
3. **Geometry Stress:** Grand fluted Romanesque colonnade with capitals, cross-vaulted gothic arch ceilings, twin helical/spiral staircases with walkable 3-voxel steps, cantilevered skybridge galleries, and natural cliff amphitheater. Central north and south portals are carved with open arched threshold voids for seamless camera flythrough.
4. **Reflection Stress:** Massive polished mirror chrome pillars (`MAT_MIRROR_CHROME`), a 16m wide flat mirror feature wall reflecting the entire complex in real-time ray tracing, and suspended mirror chrome orb.

### 1.2 Deterministic Camera Spline & Diurnal Lighting (`crates/arcadia-app/src/benchmark.rs`)
- **Catmull-Rom Trajectory:** Interpolates position, look-at, and FOV across 9 curated collision-verified waypoints (guaranteed $> 2\text{m}$ clearance from all solids, obelisks, and ceiling ribs).
- **Dynamic Time-of-Day Progression:** Smoothly cycles through 4 distinct lighting environments:
  - Midday sun ($t \in [0.0, 0.28]$)
  - Golden-hour sunset on reflecting pool ($t \in [0.28, 0.50]$)
  - Pitch-dark midnight in the magma grotto testing pure ReSTIR multi-bounce GI ($t \in [0.50, 0.78]$)
  - Dawn sunrise breaking over the mountain peaks ($t \in [0.78, 1.00]$)
- **Isolated Fixed Modes:** Supports `--tod <cycle|day|sunset|night|sunrise>`.
- **Determinism:**
  - 60 warmup frames prime GPU pipelines, ReSTIR reservoirs, and temporal history.
  - 900 benchmark frames (15s at 60 FPS) with fixed simulation step $dt = 1/60\text{ s}$ ensure 100% determinism across runs.

### 1.3 Statistical Analysis & Composite Score
The benchmark analyzes frametimes and generates comprehensive distribution statistics:
- **Metrics Computed:** Average FPS, 1% Low FPS, 0.1% Low FPS, Min/Max/Median/p90/p95/p99 frametimes, standard deviation, and frametime consistency percentage.
- **Standardized Composite Score:**

$$\text{Score} = \operatorname{round}(\text{Avg FPS} \times 100 \times \text{Consistency})$$

- **Export Formats:**
  - Machine-readable JSON: `benchmark_results.json`
  - Human-readable Markdown: `benchmark_report.md`
- **Execution Interface:**
  - PowerShell runner: `./run.ps1 -Benchmark [-TimeOfDay <mode>] [-Frames N] [-AutoExit]`
  - Cargo CLI: `cargo run -p arcadia-app -- --bench [--tod <mode>] [--frames N] [--auto-exit]`
  - In-engine hotkey: Press `[B]` to initiate or re-run the benchmark; press `[N]` to toggle Day/Sunset/Midnight in interactive mode.

---

## 2. Interactive Raymarcher Quality Tuning & Real-Time Debug Interface

### 2.1 `RaymarchQualitySettings` (`crates/arcadia-render/src/quality.rs`)
Encapsulates all raymarch loop bounds, draw distances, cone LOD thresholds, and reconstruction settings:

| Category | Field Name | Default Value | Description |
|---|---|---|---|
| **Raymarch Steps** | `primary_max_steps` | 200 | Maximum steps for primary camera rays |
| | `sec_max_steps` | 72 | Maximum steps for secondary reflection/refraction rays |
| | `gi_max_steps` | 48 | Maximum steps for diffuse indirect GI rays |
| | `shadow_max_steps` | 72 | Maximum steps for directional sun shadow rays |
| **Draw Distances** | `water_refraction_dist` | 140.0m | Max distance for underwater refraction rays |
| | `mirror_reflection_dist` | 120.0m | Max distance for mirror reflection rays |
| | `gi_bounce_dist` | 26.0m | Maximum reach for indirect GI gather |
| | `shadow_max_dist` | 88.0m | Maximum reach for direct shadow evaluation |
| **LOD & Penumbra** | `primary_lod_dist` | 120.0m | Distance where primary rays transition to coarse LOD |
| | `sec_lod_dist` | 28.0m | Distance where secondary reflections drop to coarse LOD |
| | `gi_cone_angle` | 0.038 rad | Angular width of diffuse GI sampling cone |
| | `sun_penumbra_radius` | 0.035 rad | Directional sun soft shadow penumbra jitter radius |
| **Features** | `enable_sec_cone_lod` | true | Cone LOD acceleration on reflections |
| | `enable_adaptive_primary_lod` | true | Adaptive primary LOD based on distance |
| | `shadow_traversal_mode` | 0 | 0 = Accurate 64-Tree, 1 = Fast Any-Hit |
| | `denoiser_taps` | 16 | Vogel-disk reconstruction filter tap count (4..16) |
| **Island Physics** | `island_detection_enabled` | true | Structural collapse processing toggle |
| | `island_max_depth_m` | 12.0m | Max search depth before assuming bedrock support |
| | `island_max_voxels` | 32,768 | Max voxel threshold before grounding island |
| | `object_bake_timeout_s` | 45.0s | Rest duration required before static world baking |

### 2.2 Uniform Buffer Contract (`shaders/common/types.wgsl`)
These parameters are packed into four 16-byte aligned `vec4` vectors inside `CameraUniforms`:
- `ray_steps: vec4<u32>` $\to$ `(primary, reflection, gi, shadow)`
- `ray_distances: vec4<f32>` $\to$ `(water_refraction, mirror_reflection, gi_bounce, shadow_ray)`
- `lod_settings: vec4<f32>` $\to$ `(primary_lod_dist, reflection_lod_dist, gi_cone_angle, sun_penumbra_radius)`
- `quality_flags: vec4<u32>` $\to$ `(enable_refl_lod, enable_adaptive_primary_lod, shadow_mode, denoiser_taps)`

Pushed to GPU every frame with zero pipeline re-creation or compilation stalls.

### 2.3 Immediate-Mode Debug Overlay (`crates/arcadia-app/src/gui.rs`)
- Built using `egui` (0.30) integrated into `arcadia-render`'s output pass via `wgpu::RenderPass::forget_lifetime()`.
- Toggled via `[F1]`, `[Tab]`, or ``[`]`` (backquote), releasing mouse capture and opening an interactive tuning window.
- **Curated One-Click Presets:**
  - **High Quality:** (Reference default — max fidelity, 100% scale, 200/72/48/72 steps, 16 denoiser taps, 140m water/120m mirrors).
  - **🚀 60 FPS Fluid:** (Hardware-calibrated for integrated GPUs — 48% scale, 110/28/12/36 steps, zero jitter, ~50 FPS on Radeon 660M).
  - **🎨 30 FPS Fidelity:** (Hardware-calibrated — 58% scale, 112/36/16/42 steps, balanced ReSTIR GI).
  - **Yesterday Perf:** (110/64/30/34 steps, 12 taps, 100% scale).
  - **✨ Ultra Fidelity:** (260/96/64/96 steps, 160m water/150m mirrors, 32m GI, 16 taps).
- **6 Collapsible Sections:**
  1. *Primary Raymarcher*: Step budget, LOD distance, adaptive step trigger.
  2. *Secondary Rays (Water & Mirrors)*: Step budget, draw distances, distance-adaptive mirror cone LOD.
  3. *ReSTIR GI & Sun Shadows*: GI bounce distance/steps, shadow distance/steps, sun penumbra radius, ambient floor slider, shadow mode.
  4. *Denoising & Reconstruction*: Vogel-disk filter tap count (4..16).
  5. *Upscaling & Presentation*: Render scale slider, Dithering mode radio (`[K]`), Camera Jitter radio (`[J]`).
  6. *Destruction & Island Physics*: Island detachment toggle (`[I]`), Max Depth slider (0..30m), Max Voxels logarithmic slider (64..262k), Rest Timeout slider (`object_bake_timeout_s`, 5..180s).

---

## 3. Hardware-Calibrated Presets & Auto-Calibration Engine

### 3.1 Decoupled Render Scaling
- `params.quality.render_scale` dynamically reconfigures internal `ScreenResources` (GBuffers, Reservoirs, Lit Color, History) between 35% and 100% of native swapchain resolution with zero pipeline churn.
- Internal compute passes march at reduced resolution; Pass 5 (Catmull-Rom + RCAS) upsamples to native swapchain pixels.

### 3.2 Hardware Calibration Routine (`crates/arcadia-app/src/app/`)
- Initiated by pressing `[C]` or clicking "Run Auto-Calibration" in the GUI.
- Measures real-time GPU throughput over a 40-frame window.
- Dynamically calculates the optimal render scale using quadratic fillrate scaling:

$$\text{scale} = \operatorname{clamp}\left(\text{scale} \cdot \sqrt{\frac{\text{fps\_ratio}}{1.08}}, 0.40, 0.90\right)$$

- Classifies hardware into one of 3 performance tiers ($\ge 55\text{ FPS}$, $\ge 40\text{ FPS}$, $<40\text{ FPS}$) and adjusts ray step budgets accordingly.
- **Quick Switch:** Hotkey `[L]` toggles instantly between Calibrated 60 FPS Fluid and Calibrated 30 FPS Fidelity presets.

---

## 4. Rolling Frame Pacing & Stability Tracking (`crates/arcadia-app/src/frame_graph.rs`)

To monitor micro-stutter, GC pauses, and GPU scheduling jitter in real time:

- **Ring Buffer:** Rolling 180-frame ring buffer (`VecDeque<f32>`) tracking the past 3 seconds at 60 FPS.
- **Evaluated Statistics (`FramePacingStats`):** Current frame time, rolling average, min/max ms, standard deviation, 1% low FPS, and consistency purity percentage.

### Custom Egui Bottom-Right Overlay HUD (`[F3]` Hotkey)
- **Framerate Readout:** Color-coded FPS ($\ge 58$ green, $\ge 29$ yellow, $< 29$ red).
- **Jitter Badge:** Dynamic stability rating (`STABLE`, `MODERATE`, `JITTER`).
- **Target Guidelines:** Reference lines at 16.7ms (60 FPS) and 33.3ms (30 FPS).
- **Rolling Histogram:** Bar graph displaying the duration of each individual frame.
- **Footer Diagnostics:** Displays min/max frametimes, 1% low FPS, and consistency purity.

---

## Related Specifications & Implementation Files
- **Rendering Architecture & Upscaler:** [02_rendering_and_lighting.md](02_rendering_and_lighting.md)
- **Physics Architecture:** [03_physics_and_destruction.md](03_physics_and_destruction.md)
- **Master Entry Point:** [ARCHITECTURE.md](../../ARCHITECTURE.md)
- **Source Code:**
  - Quality settings: [`crates/arcadia-render/src/quality.rs`](../../crates/arcadia-render/src/quality.rs)
  - Benchmark scene: [`crates/arcadia-world/src/benchmark_scene.rs`](../../crates/arcadia-world/src/benchmark_scene.rs)
  - Benchmark runner: [`crates/arcadia-app/src/benchmark.rs`](../../crates/arcadia-app/src/benchmark.rs)
  - Frame graph HUD: [`crates/arcadia-app/src/frame_graph.rs`](../../crates/arcadia-app/src/frame_graph.rs)
  - Debug GUI: [`crates/arcadia-app/src/gui.rs`](../../crates/arcadia-app/src/gui.rs)

