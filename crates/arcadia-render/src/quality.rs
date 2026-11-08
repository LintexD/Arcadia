//! Raymarching quality parameters, LOD thresholds, and hardware performance presets.
//! Owns CPU raymarching configuration and dynamic framerate calibration state.

/// Tunable raymarcher and reconstruction quality parameters exposed for real-time debugging.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaymarchQualitySettings {
    // --- Step Counts ---
    /// Primary ray maximum steps (default: 200, aggressive perf: 110). Range: 16..=300
    pub primary_max_steps: u32,
    /// Secondary ray (reflections/refractions) max steps (default: 72, aggressive perf: 64). Range: 16..=150
    pub sec_max_steps: u32,
    /// Indirect diffuse GI bounce ray max steps (default: 48, aggressive perf: 30). Range: 8..=100
    pub gi_max_steps: u32,
    /// Sun shadow ray max steps (default: 72, aggressive perf: 34). Range: 16..=120
    pub shadow_max_steps: u32,

    // --- Max Distances (Meters) ---
    /// Water refraction ray max distance (default: 140.0m, aggressive perf: 16.0m). Range: 4.0..=200.0m
    pub water_refr_dist: f32,
    /// Mirror reflection ray max distance (default: 120.0m, aggressive perf: 72.0m). Range: 10.0..=200.0m
    pub mirror_refl_dist: f32,
    /// Indirect diffuse GI bounce max distance (default: 26.0m, aggressive perf: 24.0m). Range: 6.0..=50.0m
    pub gi_max_dist: f32,
    /// Sun shadow ray max distance (default: 88.0m, aggressive perf: 72.0m). Range: 10.0..=160.0m
    pub shadow_max_dist: f32,

    // --- Adaptive Ray-Cone LOD & Sampling ---
    /// Primary ray distance threshold (meters) before switching to meso-cells (LOD 2) (default: 120.0m, aggressive perf: 70.0m). Range: 20.0..=200.0m
    pub primary_lod_dist: f32,
    /// Primary ray step count threshold before switching to meso-cells (default: 155, aggressive perf: 80). Range: 30..=250
    pub primary_lod_step: u32,
    /// Secondary ray distance threshold (meters) before switching to LOD 2 (default: 28.0m, aggressive perf: 48.0m). Range: 10.0..=80.0m
    pub sec_lod_dist: f32,
    /// Secondary ray step count threshold before switching to LOD 2 (default: 60, aggressive perf: 52). Range: 20..=100
    pub sec_lod_step: u32,
    /// Indirect GI bounce distance threshold (meters) before switching to LOD 2 (default: 24.0m, aggressive perf: 14.0m). Range: 6.0..=40.0m
    pub gi_lod_dist: f32,
    /// Indirect GI bounce step count threshold (default: 48, aggressive perf: 22). Range: 10..=60
    pub gi_lod_step: u32,

    // --- Shadows & Reconstruction ---
    /// Sun penumbra angular jitter radius (default: 0.018, aggressive perf: 0.005). Range: 0.000..=0.040
    pub shadow_jitter: f32,
    /// Shadow mode: 0 = High-fidelity 64-tree shadow traversal, 1 = Fast any-hit shadow raymarch (default: 0)
    pub shadow_mode: u32,
    /// Denoiser Vogel-disk reconstruct filter tap count: 12 or 16 (default: 16, aggressive perf: 12)
    pub denoiser_taps: u32,
    /// Mirror reflection adaptive LOD mode: 0 = constant LOD (default 1u), 1 = distance-adaptive (0u close, 2u far)
    pub mirror_adaptive: u32,
    /// Internal render resolution scale (`0.40 ..= 1.00`) before Catmull-Rom TSR + FSR RCAS upscaling.
    pub render_scale: f32,
    /// Dithering mode: 0 = Off, 1 = PCG 2D Decorrelated TPDF (User Favorite), 2 = Spatial Bit-Mix TPDF (default: 1).
    pub dither_mode: u32,
    /// Display tonemapper transform: 0 = ACES Filmic, 1 = Minimalist Polynomial AgX (default: 0).
    pub tonemapper_mode: u32,
    /// Minimum ambient illumination floor power (default: 0.0055, range: 0.000..=0.060).
    pub ambient_floor: f32,
    /// Distance in meters beyond which micro-voxel bitwise contact AO is smoothly disabled to eliminate distant Moiré rings (default: 24.0m, range: 4.0..=64.0m).
    pub contact_ao_dist: f32,
    /// Camera ray jitter mode for TSR/DLAA:
    /// 0 = Off (Zero jitter, static ray centers)
    /// 1 = Freeze when Still (Jitter during motion, zero jitter when stationary) [Default & Recommended]
    /// 2 = Always On (Continuous 16-phase Halton sequence)
    pub jitter_mode: u32,
    // --- Camera & Presentation ---
    /// Camera vertical field of view in degrees for interactive play (default: 65.0, range: 40.0..=120.0).
    pub fov_deg: f32,
    // --- Destruction & Island Physics ---
    /// Whether structural island detection and collapse into dynamic rigid bodies is enabled.
    /// When false, severed voxels remain floating in mid-air (Minecraft style). Default: true.
    pub island_detection_enabled: bool,
    /// Maximum vertical search depth/height in meters for island detection passes (default: 15.0m, range: 0.0..=30.0m).
    /// If 0.0, island detection is disabled.
    pub island_max_depth_m: f32,
    /// Maximum number of voxels checked per connected component before treating it as grounded terrain (default: 16_384, range: 64..=262_144).
    pub island_max_voxels: u32,
    /// Maximum surface voxels explored during Phase 1 boundary shell BFS before tagging as grounded terrain (default: 16_384, range: 64..=131_072).
    pub island_max_surface_voxels: u32,
    /// Duration in seconds a resting/stationary rigid body must remain undisturbed before baking back into the static world map (default: 60.0s, range: 5.0..=180.0s).
    pub object_bake_timeout_s: f32,
}

impl Default for RaymarchQualitySettings {
    fn default() -> Self {
        Self::high_quality()
    }
}

impl RaymarchQualitySettings {
    /// SHARC path-seed tile edge in pixels: 16 in low-tap Tier 1 mode (`denoiser_taps <= 12`), else 8.
    // GPU-SYNC: Mirrors `sharc_tile_size()` in `shaders/sharc_update.wgsl`.
    pub fn sharc_tile_size(&self) -> u32 {
        if self.denoiser_taps <= 12 {
            16
        } else {
            8
        }
    }

    /// Restores high-quality raymarching defaults (full fidelity).
    pub fn high_quality() -> Self {
        Self {
            primary_max_steps: 200,
            sec_max_steps: 72,
            gi_max_steps: 48,
            shadow_max_steps: 72,
            water_refr_dist: 140.0,
            mirror_refl_dist: 120.0,
            gi_max_dist: 26.0,
            shadow_max_dist: 88.0,
            primary_lod_dist: 120.0,
            primary_lod_step: 155,
            sec_lod_dist: 28.0,
            sec_lod_step: 60,
            gi_lod_dist: 24.0,
            gi_lod_step: 48,
            shadow_jitter: 0.018,
            shadow_mode: 0,
            denoiser_taps: 16,
            mirror_adaptive: 0,
            render_scale: 1.0,
            dither_mode: 3,
            tonemapper_mode: 0,
            ambient_floor: 0.005,
            contact_ao_dist: 24.0,
            jitter_mode: 1,
            fov_deg: 65.0,
            island_detection_enabled: true,
            island_max_depth_m: 15.0,
            island_max_voxels: 262_144,
            island_max_surface_voxels: 32_768,
            object_bake_timeout_s: 60.0,
        }
    }

    /// Calibrated 30 FPS Preset (Prioritizing rich visual detail on integrated graphics).
    pub fn calibrated_30fps() -> Self {
        Self {
            primary_max_steps: 112,
            sec_max_steps: 36,
            gi_max_steps: 16,
            shadow_max_steps: 42,
            water_refr_dist: 16.0,
            mirror_refl_dist: 48.0,
            gi_max_dist: 16.0,
            shadow_max_dist: 56.0,
            primary_lod_dist: 60.0,
            primary_lod_step: 72,
            sec_lod_dist: 28.0,
            sec_lod_step: 32,
            gi_lod_dist: 12.0,
            gi_lod_step: 16,
            shadow_jitter: 0.008,
            shadow_mode: 1,
            denoiser_taps: 8,
            mirror_adaptive: 1,
            render_scale: 0.58,
            dither_mode: 3,
            tonemapper_mode: 0,
            ambient_floor: 0.0055,
            contact_ao_dist: 20.0,
            jitter_mode: 1,
            fov_deg: 65.0,
            island_detection_enabled: true,
            island_max_depth_m: 15.0,
            island_max_voxels: 131_072,
            island_max_surface_voxels: 32_768,
            object_bake_timeout_s: 60.0,
        }
    }

    /// Calibrated 60 FPS Preset (Maximum fluid frame rate for relaxed testing on integrated graphics).
    pub fn calibrated_60fps() -> Self {
        Self {
            primary_max_steps: 110,
            sec_max_steps: 28,
            gi_max_steps: 12,
            shadow_max_steps: 36,
            water_refr_dist: 14.0,
            mirror_refl_dist: 36.0,
            gi_max_dist: 12.0,
            shadow_max_dist: 48.0,
            primary_lod_dist: 50.0,
            primary_lod_step: 64,
            sec_lod_dist: 24.0,
            sec_lod_step: 28,
            gi_lod_dist: 10.0,
            gi_lod_step: 12,
            shadow_jitter: 0.006,
            shadow_mode: 1,
            denoiser_taps: 8,
            mirror_adaptive: 1,
            render_scale: 0.48,
            dither_mode: 3,
            tonemapper_mode: 0,
            ambient_floor: 0.0055,
            contact_ao_dist: 18.0,
            jitter_mode: 1,
            fov_deg: 65.0,
            island_detection_enabled: true,
            island_max_depth_m: 15.0,
            island_max_voxels: 131_072,
            island_max_surface_voxels: 32_768,
            object_bake_timeout_s: 60.0,
        }
    }

    /// Dynamically calibrates all settings to hit `target_fps` based on a measured baseline framerate.
    pub fn calibrate_for_target_fps(&mut self, measured_fps: f32, target_fps: f32) {
        if measured_fps <= 0.0 || target_fps <= 0.0 {
            return;
        }

        let ratio = measured_fps / target_fps;

        if target_fps >= 55.0 {
            self.primary_max_steps = 110.max(self.primary_max_steps.min(130));
            self.gi_max_steps = 12;
            self.gi_max_dist = 12.0;
            self.gi_lod_step = 12;
            self.shadow_max_steps = 36;
            self.shadow_max_dist = 48.0;
            self.sec_max_steps = 28;
            self.water_refr_dist = 14.0;
            self.mirror_refl_dist = 36.0;
            self.denoiser_taps = 8;
            self.shadow_mode = 1;
            self.mirror_adaptive = 1;
        } else if target_fps >= 40.0 {
            self.primary_max_steps = 112.max(self.primary_max_steps.min(140));
            self.gi_max_steps = 14;
            self.gi_max_dist = 14.0;
            self.gi_lod_step = 14;
            self.shadow_max_steps = 40;
            self.shadow_max_dist = 52.0;
            self.sec_max_steps = 32;
            self.water_refr_dist = 15.0;
            self.mirror_refl_dist = 42.0;
            self.denoiser_taps = 8;
            self.shadow_mode = 1;
            self.mirror_adaptive = 1;
        } else {
            self.primary_max_steps = 112.max(self.primary_max_steps.min(160));
            self.gi_max_steps = 18;
            self.gi_max_dist = 18.0;
            self.gi_lod_step = 18;
            self.shadow_max_steps = 46;
            self.shadow_max_dist = 60.0;
            self.sec_max_steps = 36;
            self.water_refr_dist = 18.0;
            self.mirror_refl_dist = 50.0;
            self.denoiser_taps = 8;
            self.shadow_mode = 1;
            self.mirror_adaptive = 1;
        }

        // WHY: Fillrate scaling is quadratic O(scale^2) with an 8% safety margin against camera motion dips.
        let target_scale = (self.render_scale * (ratio / 1.08).sqrt()).clamp(0.40, 0.90);
        self.render_scale = (target_scale * 100.0).round() / 100.0;
    }

    /// Laptop iGPU Mode (58% internal resolution = 33.6% pixel count + Catmull-Rom TSR + FSR RCAS upscaling).
    /// Delivers ~3x higher FPS on integrated graphics while keeping crisp upscaled visuals.
    pub fn laptop_igpu() -> Self {
        Self::calibrated_30fps()
    }

    /// Aggressive performance preset with reduced step sizes and distances.
    pub fn yesterday_perf() -> Self {
        Self {
            primary_max_steps: 110,
            sec_max_steps: 64,
            gi_max_steps: 30,
            shadow_max_steps: 34,
            water_refr_dist: 16.0,
            mirror_refl_dist: 72.0,
            gi_max_dist: 24.0,
            shadow_max_dist: 72.0,
            primary_lod_dist: 70.0,
            primary_lod_step: 80,
            sec_lod_dist: 48.0,
            sec_lod_step: 52,
            gi_lod_dist: 14.0,
            gi_lod_step: 22,
            shadow_jitter: 0.005,
            shadow_mode: 1,
            denoiser_taps: 12,
            mirror_adaptive: 1,
            render_scale: 1.0,
            dither_mode: 3,
            tonemapper_mode: 0,
            ambient_floor: 0.0055,
            contact_ao_dist: 16.0,
            jitter_mode: 1,
            fov_deg: 65.0,
            island_detection_enabled: true,
            island_max_depth_m: 15.0,
            island_max_voxels: 131_072,
            island_max_surface_voxels: 32_768,
            object_bake_timeout_s: 60.0,
        }
    }

    /// Ultra fidelity preset for maximum visual perfection.
    pub fn ultra() -> Self {
        Self {
            primary_max_steps: 260,
            sec_max_steps: 96,
            gi_max_steps: 64,
            shadow_max_steps: 96,
            water_refr_dist: 160.0,
            mirror_refl_dist: 150.0,
            gi_max_dist: 32.0,
            shadow_max_dist: 110.0,
            primary_lod_dist: 160.0,
            primary_lod_step: 200,
            sec_lod_dist: 40.0,
            sec_lod_step: 80,
            gi_lod_dist: 32.0,
            gi_lod_step: 60,
            shadow_jitter: 0.022,
            shadow_mode: 0,
            denoiser_taps: 16,
            mirror_adaptive: 0,
            render_scale: 1.0,
            dither_mode: 3,
            tonemapper_mode: 0,
            ambient_floor: 0.005,
            contact_ao_dist: 28.0,
            jitter_mode: 1,
            fov_deg: 65.0,
            island_detection_enabled: true,
            island_max_depth_m: 15.0,
            island_max_voxels: 524_288,
            island_max_surface_voxels: 32_768,
            object_bake_timeout_s: 60.0,
        }
    }
}
