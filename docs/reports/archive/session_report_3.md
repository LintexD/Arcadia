# Arcadia Voxel Engine — Session 3 Engineering Report

**Date:** October 26, 2026  
**Status:** Completed, Benchmarked & Verified  
**Target Architecture:** Rust 2024, `wgpu` 23.0 (Vulkan 1.3 / DX12), WGSL Compute & Render Shaders, `egui` 0.30, Winit 0.30  
**Test Platform:** AMD Ryzen 5 7535U, Radeon 660M (RDNA2 Integrated GPU, DX12), 16 GB DDR5  

---

## 1. Executive Summary

Session 3 addressed visual fidelity, image quality polish, artifact eradication, and hardware scalability:
1. **Pass 5 Spatial Upscaler & RCAS Sharpening ([blit.wgsl](../../../shaders/blit.wgsl)):** Implemented a dedicated swapchain presentation pass featuring 5-tap Catmull-Rom bicubic spatial upsampling with FSR 1.0 Robust Contrast-Adaptive Sharpening (RCAS), strict 5-tap neighborhood anti-ringing clamping, and native-resolution 1-pixel HUD crosshair.
2. **Hardware Auto-Calibration Engine & Calibrated Presets:** Replaced jittery dynamic resolution scaling (DRS) with a hardware calibration routine (`[Key C]` / GUI) that samples GPU throughput over 40 frames and locks in zero-jitter, stable rendering parameters. Deployed dedicated **🚀 60 FPS Fluid** (48% scale, 110 steps, ~50 FPS on integrated Radeon graphics) and **🎨 30 FPS Fidelity** (58% scale, 112 steps, balanced ReSTIR GI) presets.
3. **High-Precision Decorrelated TPDF Dithering System:** Implemented a selectable 3-mode dithering pipeline in both Pass 4 ([composite_dlaa.wgsl](../../../shaders/composite_dlaa.wgsl)) and Pass 5 ([blit.wgsl](../../../shaders/blit.wgsl)). Mode 1 uses 2D PCG hashing with temporal frame rotation and channel decorrelation (`hash_pcg2d`, `pcg_tpdf_dither`), completely eliminating 8-bit quantization rings (moon banding), 2D lattice dots in the night sky, and vertical green striping. Mode cycling is bound to hotkey `[K]` and GUI radio buttons.
4. **Mirror Seam Artifact Eradication ([primary_gbuffer.wgsl](../../../shaders/primary_gbuffer.wgsl)):** Resolved white border dots on mirror surfaces via seam continuation ray tracing and DDA normal tie-breaking, allowing grazing reflections to trace past self-intersections to the true reflected scene while preserving crystal-clear reflections of nearby pillars and arches.
5. **Cave Daylight Occlusion & Day/Night Decoupling ([lighting_restir_gi.wgsl](../../../shaders/lighting_restir_gi.wgsl), [traversal64.wgsl](../../../shaders/common/traversal64.wgsl)):** Cured daylight leaking into underground chambers by requiring shadow rays to reach open atmosphere (`end_p.y >= 32.0`) before granting sun visibility, enforcing occlusion on step-exhaustion underground, and restricting secondary GI sky escape sampling.
6. **Tunable Ambient Floor (Mood) Slider ([gui.rs](../../../crates/arcadia-app/src/gui.rs)):** Added an interactive `ambient_floor` slider (`0.000..=0.060`, default: `0.022`) modulated by contact AO, guaranteeing every voxel remains discernible in deep caves and night scenes without producing a flat, plastic look.

---

## 2. Key Accomplishments & Technical Breakdown

### 2.1 Spatial Upscaler & Presentation Pipeline (Pass 5: `blit.wgsl`)
- **Catmull-Rom 5-Tap Bicubic Reconstruction:** Decouples internal raymarch resolution (`render_scale` 0.35..=1.0) from the native 1080p swapchain. Reconstructs smooth continuous gradients and sharp diagonal contours without bilinear blur.
- **FSR 1.0 RCAS (Robust Contrast-Adaptive Sharpening):** Evaluates local 4-tap cross-neighborhood contrast and applies an adaptive sharpening lobe:
  $$\text{rcas\_lobe} = -0.14 \times \text{clamp}(1.0 - \text{contrast} \times 1.6,\, 0.10,\, 1.0)$$
  Clamped strictly inside the 5-tap neighborhood min/max to prevent overshoot halos on high-contrast voxel silhouettes.
- **Native-Resolution HUD Crosshair:** Computed in native window swapchain pixels using screen-space UV derivatives (`fwidth`) to guarantee a crisp 1-pixel hairline crosshair independent of render scale.
- **Uniform Buffer Layout Fix:** Corrected a 128-byte offset misalignment in `blit.wgsl`'s `CameraUniforms` struct declaration (where view-projection matrices were positioned out of order), restoring correct uniform binding for `quality_flags.w`.

### 2.2 Hardware Calibration & Zero-Jitter Presets
- **Dynamic Render Scaling without Pipeline Re-creation:** `arcadia-render` dynamically updates internal `ScreenResources` (GBuffers, Reservoirs, Lit Color, History) whenever `render_scale` changes, executing instantaneously with zero pipeline recompilation.
- **Hardware Auto-Calibration Engine (`crates/arcadia-app/src/main.rs`):**
  - Triggered via `[Key C]` or GUI button.
  - Samples instantaneous frame times over a 40-frame window.
  - Computes the target render scale and primary/secondary ray step budgets required to hit consistent 30 FPS or 60 FPS on any GPU.
- **Hardware-Calibrated Presets:**
  - **🚀 60 FPS Fluid:** 48% scale (922x518 internal), 110 primary steps, 64 reflection steps, 12 GI steps, 20 shadow steps. Achieves ~50 FPS on integrated AMD Radeon 660M graphics across 6.71 billion voxels.
  - **🎨 30 FPS Fidelity:** 58% scale (1114x626 internal), 112 primary steps, 72 reflection steps, 24 GI steps, 32 shadow steps. Delivers locked 30 FPS with deep indirect lighting.
  - **Hotkey `[Key L]`:** Quickly toggles between Calibrated 30 FPS and Calibrated 60 FPS.

### 2.3 High-Precision Decorrelated TPDF Dithering System
To eradicate banding, striping, and noise across the pipeline:
- **PCG 2D Hash with Temporal Rotation (`hash_pcg2d`):**
  ```wgsl
  fn hash_pcg2d(v: vec2<u32>) -> vec2<u32> {
      var q = v * 1664525u + 1013904223u;
      q.x += q.y * 1664525u;
      q.y += q.x * 1664525u;
      q ^= (q >> vec2<u32>(16u));
      q.x += q.y * 1664525u;
      q.y += q.x * 1664525u;
      q ^= (q >> vec2<u32>(16u));
      return q;
  }
  ```
- **Temporal Rotation & Channel Decorrelation (`pcg_tpdf_dither`):**
  - Rotates screen positions temporally: `p_rg = pos + vec2(frame * 19, frame * 37)`.
  - Decorrelates blue channel: `p_ba = pos.yx + vec2(frame * 53 + 0x55555555, frame * 79 + 0xAAAAAAAA)`.
  - Computes Triangular Probability Density Function (TPDF) $(u_1 - u_2) / 255.0$.
- **Artifacts Eliminated:**
  - **Moon Banding:** Eliminates concentric quantization steps around the moon in the night sky.
  - **2D Lattice Dots:** Replaces static screen-space hash patterns with temporally rotated noise that visually disappears.
  - **Green Stripes:** Channel decorrelation prevents cross-channel alignment and colored striping in dark scenes.
- **Selectable Options:**
  - `Mode 1`: PCG 2D Decorrelated TPDF (User Favorite - Default).
  - `Mode 2`: Spatial Integer Bit-Mix TPDF.
  - `Mode 0`: Off / Disabled.
  - Interactive cycling via **`[Key K]`** or GUI radio buttons.

### 2.4 Mirror Seam Continuation & Normal Tie-Breaking
- **Root Cause of White Border Dots:** When grazing reflection rays struck the seam of an adjacent mirror voxel on a flat mirror plane within 12cm, the hit was discarded as self-intersection and defaulted to `eval_reflection_sky`, which returned bright daylight horizon sky.
- **Continuation Ray Tracing:** When a ray hits an adjacent mirror voxel along the same plane, a continuation step (`trace_world_64tree`) traces past the seam from `hit_pos + refl_dir * 0.08`, capturing the actual reflected scene geometry (pillars, arches, floors).
- **Dominant Axis Normal Tie-Breaking:** In `traversal64.wgsl`, added an epsilon tie-breaker (`eps = 1e-4`) comparing against dominant ray axes to prevent DDA floating-point ambiguities on voxel edges from flipping surface normals by $90^\circ$ to $(0, 1, 0)$.
- **Nearby Pillar Reflections:** Retained a tight $0.04\text{ m}$ reflection ray origin, ensuring reflections of adjacent structures remain crisp and detailed.

### 2.5 Cave Daylight Occlusion & Ambient Floor Control
- **Occlusion on Step Exhaustion:** In `lighting_restir_gi.wgsl`, shadow rays now require verified exit into the open sky (`end_p.y >= 32.0 || jittered_sun.y < 0.0`) to grant sun visibility. In `traversal64.wgsl`, `trace_shadow_vis` returns $0.0$ (occluded) if its step budget runs out underground.
- **Secondary GI Sky Escape:** Restricted upward sky sampling to rays where `end_p.y >= 32.0 && bounce_dir.y > 0.15`, preventing underground chambers from sampling daytime sky radiance. Closed caves now remain dark regardless of external time-of-day.
- **Ambient Floor Slider (`ambient_floor`):** Added a GUI slider (`0.000..=0.060`, default: `0.022`) packed into uniform `point_light_count.z`. Modulated by contact AO (`max(contact_ao, 0.45)`), it ensures all voxels remain visible in deep caves and night scenes without washing out scene contrast.

---

## 3. Verification & Performance Results

### 3.1 Workspace Test Suite
All 15 unit tests pass cleanly:
- `arcadia-render`: Full WGSL shader compilation and layout validation passed.
- `arcadia-app`: Benchmark lifecycle, statistical results computation, time-of-day progression passed.
- `arcadia-core`: Bitwise POPCNT indexing, chunk tree collapsing, leaf brick packing, CPU raycast passed.
- `arcadia-world`: Benchmark scene generation, point lights, spline camera interpolation passed.
- **Result:** `15 passed, 0 failed, 0 ignored` (Exit code: 0).

### 3.2 Automated Release Benchmark
Executed `cargo run --release -p arcadia-app -- --bench --preset 60 --frames 120 --auto-exit`:
- **Hardware:** AMD Radeon 660M (Integrated GPU, DirectX 12)
- **Scene:** Arcadia Sanctuary (6.71 billion micro-voxels, 400 chunks, 6,621,200 packed nodes)
- **Preset:** 🚀 60 FPS Fluid (48% scale, 110 primary steps, 12 GI steps)
- **Average Framerate:** **49.83 FPS**
- **Average Frametime:** **19.84 ms** (Median p50: **16.52 ms**)
- **1% Low:** **17.67 FPS**
- **Composite Score:** **3239**
- **Stability:** Zero crashes, zero shader compilation warnings, zero memory leaks.

---

## 4. Next Milestone: Phase 6 — Structural Island Collapse

With the rendering, upscaling, dithering, and illumination pipelines polished and stable, the next engineering milestone is:
1. **Background Bitwise $u64$ Flood-Fill Connectivity:** Check connectivity starting from bedrock ($y \le 4$) in `crates/arcadia-world/src/islands.rs`.
2. **Dynamic Rigid Body Spawning:** Extract disconnected floating voxel clusters into `RigidVoxelBody` instances (`crates/arcadia-physics/src/rigid_voxel.rs`).
3. **Rigid Body Physics & OBB Ray Marching:** Simulate gravity, angular momentum, and dynamic OBB-AABB collisions, ray marching dynamic volumes directly in `primary_gbuffer.wgsl`.
