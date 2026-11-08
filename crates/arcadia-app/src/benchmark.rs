//! Arcadia Voxel Engine — Pure Render Benchmark Runner & Statistical Analyzer.
//! Owns CPU deterministic camera trajectory execution, exact frame-time instrumentation,
//! percentile/jitter analysis (including 1% and 0.1% lows), standardized score calculation,
//! and automated result export (JSON and Markdown).

use std::fs::File;
use std::io::Write;
use std::time::Instant;

use glam::{Vec2, Vec3};

use arcadia_world::{get_benchmark_camera_path, BenchmarkCameraPath};

/// Time-of-day mode for the benchmark run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeOfDayMode {
    /// Smoothly transitions from bright midday -> golden hour sunset -> dark midnight with glowing GI -> dawn sunrise.
    Cycle,
    /// Bright sunny midday (altitude ~65°), crisp soft shadows and clear water reflection.
    Day,
    /// Warm golden hour sunset (altitude ~2°), long sweeping shadows and warm reflections.
    Sunset,
    /// Deep midnight (sun under horizon, intensity 0), testing pure ReSTIR multi-bounce GI from emissives & point lights.
    Night,
    /// Crisp dawn sunrise (low morning sun breaking over mountain peaks).
    Sunrise,
}

impl TimeOfDayMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cycle" | "dynamic" | "all" => Some(Self::Cycle),
            "day" | "noon" | "midday" => Some(Self::Day),
            "sunset" | "dusk" | "golden" => Some(Self::Sunset),
            "night" | "midnight" | "dark" => Some(Self::Night),
            "sunrise" | "dawn" | "morning" => Some(Self::Sunrise),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cycle => "Cycle (Midday -> Sunset -> Midnight -> Sunrise)",
            Self::Day => "Midday (Bright Sun, Crisp Shadows)",
            Self::Sunset => "Sunset (Golden Hour, Long Shadows)",
            Self::Night => "Midnight (Emissive & ReSTIR GI Only)",
            Self::Sunrise => "Sunrise (Dawn Horizon Light)",
        }
    }
}

/// Configuration options for a benchmark execution run.
#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    /// Number of untimed warmup frames to pre-heat GPU pipelines, ReSTIR reservoirs, and temporal history.
    pub warmup_frames: u32,
    /// Number of timed benchmark frames to measure along the deterministic camera trajectory.
    pub test_frames: u32,
    /// Automatically exit the application with code 0 after benchmark completion and file export.
    pub auto_exit: bool,
    /// Save structured JSON results to disk (`benchmark_results.json`).
    pub save_json: bool,
    /// Save readable Markdown summary to disk (`benchmark_report.md`).
    pub save_markdown: bool,
    /// Configured time-of-day lighting mode (default: dynamic cycle).
    pub time_of_day: TimeOfDayMode,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_frames: 60,
            test_frames: 900,
            auto_exit: false,
            save_json: true,
            save_markdown: true,
            time_of_day: TimeOfDayMode::Cycle,
        }
    }
}

/// Internal lifecycle state of the benchmark test.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BenchmarkState {
    /// Benchmark is inactive (interactive exploration).
    Idle,
    /// Warmup phase: executing frames to warm caches and temporal history (untimed).
    Warmup { current: u32, total: u32 },
    /// Active measurement phase: timing every individual frame along the deterministic trajectory.
    Running { current: u32, total: u32 },
    /// Benchmark has completed and final report has been generated.
    Complete,
}

/// Comprehensive statistical results calculated from an executed benchmark.
#[derive(Clone, Debug)]
pub struct BenchmarkResults {
    pub gpu_name: String,
    pub gpu_backend: String,
    pub resolution: [u32; 2],
    pub time_of_day: String,
    pub total_frames: u32,
    pub duration_secs: f32,
    pub avg_fps: f32,
    pub avg_frame_time_ms: f32,
    pub min_frame_time_ms: f32,
    pub max_frame_time_ms: f32,
    pub median_frame_time_ms: f32,
    pub p90_frame_time_ms: f32,
    pub p95_frame_time_ms: f32,
    pub p99_frame_time_ms: f32,
    pub low_1_pct_fps: f32,
    pub low_0_1_pct_fps: f32,
    pub std_dev_ms: f32,
    pub consistency_pct: f32,
    pub score: u32,
}

impl BenchmarkResults {
    /// Computes statistical metrics from a recorded list of frame times in milliseconds.
    pub fn compute(
        frame_times_ms: &[f32],
        total_duration_secs: f32,
        gpu_name: String,
        gpu_backend: String,
        resolution: [u32; 2],
        time_of_day: String,
    ) -> Self {
        let n = frame_times_ms.len();
        if n == 0 {
            return Self {
                gpu_name,
                gpu_backend,
                resolution,
                time_of_day,
                total_frames: 0,
                duration_secs: total_duration_secs,
                avg_fps: 0.0,
                avg_frame_time_ms: 0.0,
                min_frame_time_ms: 0.0,
                max_frame_time_ms: 0.0,
                median_frame_time_ms: 0.0,
                p90_frame_time_ms: 0.0,
                p95_frame_time_ms: 0.0,
                p99_frame_time_ms: 0.0,
                low_1_pct_fps: 0.0,
                low_0_1_pct_fps: 0.0,
                std_dev_ms: 0.0,
                consistency_pct: 0.0,
                score: 0,
            };
        }

        let mut sorted = frame_times_ms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f32 = sorted.iter().sum();
        let avg_frame_time_ms = sum / n as f32;
        let avg_fps = if total_duration_secs > 0.0 {
            n as f32 / total_duration_secs
        } else if avg_frame_time_ms > 0.0 {
            1000.0 / avg_frame_time_ms
        } else {
            0.0
        };

        let min_frame_time_ms = sorted[0];
        let max_frame_time_ms = sorted[n - 1];
        let median_frame_time_ms = sorted[(n * 50 / 100).min(n - 1)];
        let p90_frame_time_ms = sorted[(n * 90 / 100).min(n - 1)];
        let p95_frame_time_ms = sorted[(n * 95 / 100).min(n - 1)];
        let p99_frame_time_ms = sorted[(n * 99 / 100).min(n - 1)];

        // 1% Low FPS: Average framerate of the slowest 1% of frames
        let count_1_pct = n.div_ceil(100).clamp(1, n);
        let worst_1_pct_slice = &sorted[n - count_1_pct..];
        let worst_1_pct_avg_ms: f32 = worst_1_pct_slice.iter().sum::<f32>() / count_1_pct as f32;
        let low_1_pct_fps = if worst_1_pct_avg_ms > 0.0 {
            1000.0 / worst_1_pct_avg_ms
        } else {
            0.0
        };

        // 0.1% Low FPS: Average framerate of the slowest 0.1% of frames
        let count_0_1_pct = n.div_ceil(1000).clamp(1, n);
        let worst_0_1_pct_slice = &sorted[n - count_0_1_pct..];
        let worst_0_1_pct_avg_ms: f32 =
            worst_0_1_pct_slice.iter().sum::<f32>() / count_0_1_pct as f32;
        let low_0_1_pct_fps = if worst_0_1_pct_avg_ms > 0.0 {
            1000.0 / worst_0_1_pct_avg_ms
        } else {
            0.0
        };

        // Standard Deviation of frame times (ms):
        let variance: f32 = sorted
            .iter()
            .map(|t| (t - avg_frame_time_ms).powi(2))
            .sum::<f32>()
            / n as f32;
        let std_dev_ms = variance.sqrt();

        // Frametime Consistency percentage:
        let cv = if avg_frame_time_ms > 0.0 {
            std_dev_ms / avg_frame_time_ms
        } else {
            1.0
        };
        let consistency_pct = ((1.0 - cv).clamp(0.0, 1.0)) * 100.0;

        // Standardized composite score:
        let score = ((avg_fps * 100.0) * (consistency_pct / 100.0).clamp(0.65, 1.0)).round() as u32;

        Self {
            gpu_name,
            gpu_backend,
            resolution,
            time_of_day,
            total_frames: n as u32,
            duration_secs: total_duration_secs,
            avg_fps,
            avg_frame_time_ms,
            min_frame_time_ms,
            max_frame_time_ms,
            median_frame_time_ms,
            p90_frame_time_ms,
            p95_frame_time_ms,
            p99_frame_time_ms,
            low_1_pct_fps,
            low_0_1_pct_fps,
            std_dev_ms,
            consistency_pct,
            score,
        }
    }

    /// Prints a beautiful ASCII benchmark result card to standard output.
    pub fn print_summary(&self) {
        println!();
        println!(
            "================================================================================"
        );
        println!(
            "             ARCADIA VOXEL ENGINE — PURE RENDER BENCHMARK RESULTS                "
        );
        println!(
            "================================================================================"
        );
        println!("  Hardware & Scene:");
        println!("    GPU Device:              {}", self.gpu_name);
        println!("    Graphics Backend:        {}", self.gpu_backend);
        println!(
            "    Render Resolution:       {} x {}",
            self.resolution[0], self.resolution[1]
        );
        println!("    Lighting / Time of Day:  {}", self.time_of_day);
        println!("    Scene Name:              Arcadia Benchmark Sanctuary");
        println!("    Voxel Precision:         (1/16m)³ = 6.25 cm micro-voxels");
        println!("    Tested Pipeline:         CS64 Ray Marcher + ReSTIR GI + Mirrors + Water");
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!("  Framerate Performance:");
        println!("    OVERALL BENCHMARK SCORE: {}", self.score);
        println!("    Average Framerate:       {:.2} FPS", self.avg_fps);
        println!("    1% Low Framerate:        {:.2} FPS", self.low_1_pct_fps);
        println!(
            "    0.1% Low Framerate:      {:.2} FPS",
            self.low_0_1_pct_fps
        );
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!("  Frametime Statistics (Lower is Better):");
        println!(
            "    Average Frametime:       {:.2} ms ({:.1} FPS)",
            self.avg_frame_time_ms,
            if self.avg_frame_time_ms > 0.0 {
                1000.0 / self.avg_frame_time_ms
            } else {
                0.0
            }
        );
        println!(
            "    Minimum Frametime:       {:.2} ms ({:.1} Peak FPS)",
            self.min_frame_time_ms,
            if self.min_frame_time_ms > 0.0 {
                1000.0 / self.min_frame_time_ms
            } else {
                0.0
            }
        );
        println!(
            "    Maximum Frametime:       {:.2} ms ({:.1} Min FPS)",
            self.max_frame_time_ms,
            if self.max_frame_time_ms > 0.0 {
                1000.0 / self.max_frame_time_ms
            } else {
                0.0
            }
        );
        println!(
            "    Median (p50):            {:.2} ms",
            self.median_frame_time_ms
        );
        println!(
            "    90th Percentile (p90):   {:.2} ms",
            self.p90_frame_time_ms
        );
        println!(
            "    95th Percentile (p95):   {:.2} ms",
            self.p95_frame_time_ms
        );
        println!(
            "    99th Percentile (p99):   {:.2} ms",
            self.p99_frame_time_ms
        );
        println!(
            "    Frametime Std Dev:       {:.2} ms ({:.1}% Consistency)",
            self.std_dev_ms, self.consistency_pct
        );
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!("  Execution Details:");
        println!("    Sampled Frames:          {} frames", self.total_frames);
        println!("    Elapsed Time:            {:.2} s", self.duration_secs);
        println!(
            "================================================================================"
        );
        println!();
    }

    /// Formats the benchmark results as a JSON document.
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"engine\": \"Arcadia Voxel Engine\",\n",
                "  \"benchmark_version\": \"1.0.0\",\n",
                "  \"gpu\": {{\n",
                "    \"name\": \"{}\",\n",
                "    \"backend\": \"{}\",\n",
                "    \"resolution\": [{}, {}]\n",
                "  }},\n",
                "  \"lighting\": \"{}\",\n",
                "  \"score\": {},\n",
                "  \"metrics\": {{\n",
                "    \"avg_fps\": {:.3},\n",
                "    \"low_1_pct_fps\": {:.3},\n",
                "    \"low_0_1_pct_fps\": {:.3},\n",
                "    \"avg_frame_time_ms\": {:.3},\n",
                "    \"min_frame_time_ms\": {:.3},\n",
                "    \"max_frame_time_ms\": {:.3},\n",
                "    \"median_frame_time_ms\": {:.3},\n",
                "    \"p90_frame_time_ms\": {:.3},\n",
                "    \"p95_frame_time_ms\": {:.3},\n",
                "    \"p99_frame_time_ms\": {:.3},\n",
                "    \"std_dev_ms\": {:.3},\n",
                "    \"consistency_pct\": {:.2}\n",
                "  }},\n",
                "  \"execution\": {{\n",
                "    \"total_frames\": {},\n",
                "    \"duration_secs\": {:.3}\n",
                "  }}\n",
                "}}"
            ),
            self.gpu_name.replace('\"', "\\\""),
            self.gpu_backend,
            self.resolution[0],
            self.resolution[1],
            self.time_of_day.replace('\"', "\\\""),
            self.score,
            self.avg_fps,
            self.low_1_pct_fps,
            self.low_0_1_pct_fps,
            self.avg_frame_time_ms,
            self.min_frame_time_ms,
            self.max_frame_time_ms,
            self.median_frame_time_ms,
            self.p90_frame_time_ms,
            self.p95_frame_time_ms,
            self.p99_frame_time_ms,
            self.std_dev_ms,
            self.consistency_pct,
            self.total_frames,
            self.duration_secs
        )
    }

    /// Formats the benchmark results as a GitHub Flavored Markdown report.
    pub fn to_markdown(&self) -> String {
        format!(
            concat!(
                "# Arcadia Voxel Engine — Benchmark Report\n\n",
                "## 1. System & Test Configuration\n",
                "- **GPU Device:** {}\n",
                "- **Graphics API / Backend:** {}\n",
                "- **Target Resolution:** {} × {} (1080p)\n",
                "- **Time of Day / Lighting:** {}\n",
                "- **Scene:** Arcadia Benchmark Sanctuary (Sunken Pool, Colonnade, Mirrors, GI, Point Lights)\n",
                "- **Voxel Grid Scale:** $(1/16\\text{{ m}})^3 = 6.25\\text{{ cm}}$ micro-voxels\n\n",
                "## 2. Performance Summary\n\n",
                "| Metric | Value | Rating / Context |\n",
                "| :--- | :--- | :--- |\n",
                "| **Composite Score** | **{}** | Standardized render index |\n",
                "| **Average Framerate** | **{:.2} FPS** | {:.2} ms frame budget |\n",
                "| **1% Low Framerate** | **{:.2} FPS** | 99th percentile stutter resistance |\n",
                "| **0.1% Low Framerate** | **{:.2} FPS** | Extreme frame spike tolerance |\n",
                "| **Frametime Std Dev** | **{:.2} ms** | {:.1}% frame pacing consistency |\n\n",
                "## 3. Detailed Frametime Distribution\n\n",
                "| Percentile | Frametime (ms) | Equivalent FPS |\n",
                "| :--- | :--- | :--- |\n",
                "| Minimum (Peak) | {:.2} ms | {:.1} FPS |\n",
                "| 50th (Median) | {:.2} ms | {:.1} FPS |\n",
                "| 90th | {:.2} ms | {:.1} FPS |\n",
                "| 95th | {:.2} ms | {:.1} FPS |\n",
                "| 99th | {:.2} ms | {:.1} FPS |\n",
                "| Maximum (Worst) | {:.2} ms | {:.1} FPS |\n\n",
                "## 4. Execution Metadata\n",
                "- **Measured Sample Count:** {} frames (excluding 60 warmup frames)\n",
                "- **Total Test Duration:** {:.2} seconds\n",
                "- **Shader Passes Active:** Primary G-Buffer + Ray-Traced Mirrors + Translucent Water + ReSTIR GI + Wavelet Denoiser + DLAA Resolve\n"
            ),
            self.gpu_name,
            self.gpu_backend,
            self.resolution[0],
            self.resolution[1],
            self.time_of_day,
            self.score,
            self.avg_fps,
            self.avg_frame_time_ms,
            self.low_1_pct_fps,
            self.low_0_1_pct_fps,
            self.std_dev_ms,
            self.consistency_pct,
            self.min_frame_time_ms,
            if self.min_frame_time_ms > 0.0 { 1000.0 / self.min_frame_time_ms } else { 0.0 },
            self.median_frame_time_ms,
            if self.median_frame_time_ms > 0.0 { 1000.0 / self.median_frame_time_ms } else { 0.0 },
            self.p90_frame_time_ms,
            if self.p90_frame_time_ms > 0.0 { 1000.0 / self.p90_frame_time_ms } else { 0.0 },
            self.p95_frame_time_ms,
            if self.p95_frame_time_ms > 0.0 { 1000.0 / self.p95_frame_time_ms } else { 0.0 },
            self.p99_frame_time_ms,
            if self.p99_frame_time_ms > 0.0 { 1000.0 / self.p99_frame_time_ms } else { 0.0 },
            self.max_frame_time_ms,
            if self.max_frame_time_ms > 0.0 { 1000.0 / self.max_frame_time_ms } else { 0.0 },
            self.total_frames,
            self.duration_secs
        )
    }

    /// Saves the benchmark results to disk.
    pub fn save_to_disk(&self, json_path: &str, md_path: &str) -> std::io::Result<()> {
        let mut json_file = File::create(json_path)?;
        json_file.write_all(self.to_json().as_bytes())?;

        let mut md_file = File::create(md_path)?;
        md_file.write_all(self.to_markdown().as_bytes())?;

        println!(
            "  [Benchmark] Exported structured results to: {}",
            json_path
        );
        println!("  [Benchmark] Exported readable report to:     {}", md_path);
        Ok(())
    }
}

/// Orchestrates benchmark state transitions, deterministic camera positioning, and frame timing.
pub struct BenchmarkRunner {
    pub config: BenchmarkConfig,
    pub state: BenchmarkState,
    pub camera_path: BenchmarkCameraPath,
    pub frame_times_ms: Vec<f32>,
    pub test_start_instant: Option<Instant>,
    pub frame_start_instant: Option<Instant>,
    pub latest_results: Option<BenchmarkResults>,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig, center_xz: Vec2) -> Self {
        let camera_path = get_benchmark_camera_path(center_xz);
        Self {
            config,
            state: BenchmarkState::Idle,
            camera_path,
            frame_times_ms: Vec::new(),
            test_start_instant: None,
            frame_start_instant: None,
            latest_results: None,
        }
    }

    /// Initiates a benchmark run beginning with the warmup phase.
    pub fn start(&mut self) {
        self.frame_times_ms.clear();
        self.test_start_instant = None;
        self.frame_start_instant = None;
        self.latest_results = None;

        if self.config.warmup_frames > 0 {
            self.state = BenchmarkState::Warmup {
                current: 0,
                total: self.config.warmup_frames,
            };
            println!(
                ">>> Starting Benchmark Warmup ({} frames to prime GPU pipelines & ReSTIR reservoirs)...",
                self.config.warmup_frames
            );
        } else {
            self.state = BenchmarkState::Running {
                current: 0,
                total: self.config.test_frames,
            };
            self.test_start_instant = Some(Instant::now());
            println!(
                ">>> Starting Pure Render Benchmark Test ({} frames across deterministic camera path)...",
                self.config.test_frames
            );
        }
    }

    /// Returns true if a benchmark is currently executing (either warmup or active timing).
    #[inline]
    pub fn is_active(&self) -> bool {
        matches!(
            self.state,
            BenchmarkState::Warmup { .. } | BenchmarkState::Running { .. }
        )
    }

    /// Returns the sun angle (in radians) corresponding to the benchmark progress `t in [0.0, 1.0]`.
    pub fn get_sun_angle(&self, progress: f32) -> f32 {
        match self.config.time_of_day {
            TimeOfDayMode::Day => 1.15,
            TimeOfDayMode::Sunset => 0.04,
            TimeOfDayMode::Night => -1.57,
            TimeOfDayMode::Sunrise => 3.10,
            TimeOfDayMode::Cycle => {
                let t = progress.clamp(0.0, 1.0);
                if t < 0.28 {
                    let phase = t / 0.28;
                    1.20 - phase * 0.75
                } else if t < 0.50 {
                    let phase = (t - 0.28) / 0.22;
                    0.45 - phase * 0.50
                } else if t < 0.78 {
                    let phase = (t - 0.50) / 0.28;
                    -0.05 - phase * 1.52
                } else {
                    let phase = (t - 0.78) / 0.22;
                    -1.57 + phase * 2.42
                }
            }
        }
    }

    /// Evaluates camera position, look_at target, and FOV for the current frame along the deterministic spline.
    pub fn prepare_frame_camera(&mut self) -> (Vec3, Vec3, f32) {
        let t = match self.state {
            BenchmarkState::Warmup { current, total } => {
                if total > 0 {
                    (current as f32 / total as f32) * 0.05
                } else {
                    0.0
                }
            }
            BenchmarkState::Running { current, total } if total > 0 => {
                current as f32 / total as f32
            }
            _ => 0.0,
        };

        // WHY: Timestamp recording frame dispatch to measure GPU rendering duration.
        self.frame_start_instant = Some(Instant::now());
        self.camera_path.sample(t)
    }

    /// Records the rendered frame and advances the benchmark state.
    /// Returns `Some(BenchmarkResults)` upon completion of the test.
    pub fn finish_frame(
        &mut self,
        gpu_name: &str,
        gpu_backend: &str,
        resolution: [u32; 2],
    ) -> Option<BenchmarkResults> {
        let now = Instant::now();
        let frame_duration_ms = if let Some(start) = self.frame_start_instant {
            (now - start).as_secs_f32() * 1000.0
        } else {
            16.667
        };

        match self.state {
            BenchmarkState::Warmup { current, total } => {
                let next = current + 1;
                if next >= total {
                    self.state = BenchmarkState::Running {
                        current: 0,
                        total: self.config.test_frames,
                    };
                    self.test_start_instant = Some(Instant::now());
                    self.frame_start_instant = None;
                    self.frame_times_ms.clear();
                    println!(
                        ">>> Warmup complete! Now running pure render benchmark across {} frames...",
                        self.config.test_frames
                    );
                } else {
                    self.state = BenchmarkState::Warmup {
                        current: next,
                        total,
                    };
                }
                None
            }
            BenchmarkState::Running { current, total } => {
                self.frame_times_ms.push(frame_duration_ms);
                let next = current + 1;

                if next >= total {
                    let total_duration_secs = if let Some(start) = self.test_start_instant {
                        (now - start).as_secs_f32()
                    } else {
                        self.frame_times_ms.iter().sum::<f32>() / 1000.0
                    };

                    let results = BenchmarkResults::compute(
                        &self.frame_times_ms,
                        total_duration_secs,
                        gpu_name.to_string(),
                        gpu_backend.to_string(),
                        resolution,
                        self.config.time_of_day.as_str().to_string(),
                    );

                    results.print_summary();
                    if self.config.save_json || self.config.save_markdown {
                        let _ =
                            results.save_to_disk("benchmark_results.json", "benchmark_report.md");
                    }

                    self.latest_results = Some(results.clone());
                    self.state = BenchmarkState::Complete;
                    Some(results)
                } else {
                    self.state = BenchmarkState::Running {
                        current: next,
                        total,
                    };
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_results_compute() {
        let mut times = Vec::new();
        // 100 frames with controlled distribution around 16.6ms (60 FPS)
        for i in 0..100 {
            if i >= 99 {
                times.push(33.3); // 1% extreme spike
            } else if i >= 95 {
                times.push(22.0); // 5% moderate spikes
            } else {
                times.push(16.0);
            }
        }

        let results = BenchmarkResults::compute(
            &times,
            1.66,
            "NVIDIA RTX 3080".to_string(),
            "Vulkan".to_string(),
            [1920, 1080],
            "Cycle (Midday -> Sunset -> Midnight -> Sunrise)".to_string(),
        );

        assert_eq!(results.total_frames, 100);
        assert!(results.avg_fps >= 55.0 && results.avg_fps <= 65.0);
        assert!(results.avg_frame_time_ms >= 16.0 && results.avg_frame_time_ms <= 17.5);
        assert_eq!(results.min_frame_time_ms, 16.0);
        assert_eq!(results.max_frame_time_ms, 33.3);
        assert!(results.low_1_pct_fps <= 35.0); // 1% low is affected by the spike
        assert!(results.std_dev_ms > 0.0);
        assert!(results.score > 0);

        let json = results.to_json();
        assert!(json.contains("\"engine\": \"Arcadia Voxel Engine\""));
        assert!(json.contains("\"avg_fps\":"));
        assert!(json.contains("\"lighting\": \"Cycle"));
        assert!(json.contains("\"score\":"));

        let md = results.to_markdown();
        assert!(md.contains("# Arcadia Voxel Engine — Benchmark Report"));
        assert!(md.contains("NVIDIA RTX 3080"));
        assert!(md.contains("Time of Day / Lighting"));
    }

    #[test]
    fn test_benchmark_time_of_day_progression() {
        let config = BenchmarkConfig {
            time_of_day: TimeOfDayMode::Cycle,
            ..Default::default()
        };
        let runner = BenchmarkRunner::new(config, Vec2::new(80.0, 80.0));

        let angle_start = runner.get_sun_angle(0.0);
        let angle_sunset = runner.get_sun_angle(0.40);
        let angle_night = runner.get_sun_angle(0.75);
        let angle_sunrise = runner.get_sun_angle(1.0);

        assert!(angle_start > 1.0); // Bright midday sun
        assert!(angle_sunset < 0.3 && angle_sunset > -0.2); // Horizon sunset
        assert!(angle_night < -1.0); // Pitch-dark midnight
        assert!(angle_sunrise > 0.5); // Morning sun restored
    }

    #[test]
    fn test_benchmark_runner_lifecycle() {
        let config = BenchmarkConfig {
            warmup_frames: 2,
            test_frames: 3,
            auto_exit: false,
            save_json: false,
            save_markdown: false,
            time_of_day: TimeOfDayMode::Cycle,
        };

        let mut runner = BenchmarkRunner::new(config, Vec2::new(80.0, 80.0));
        assert!(!runner.is_active());

        runner.start();
        assert!(runner.is_active());

        // Frame 1 (warmup 0/2)
        let _ = runner.prepare_frame_camera();
        let res = runner.finish_frame("GPU", "Vulkan", [1920, 1080]);
        assert!(res.is_none());
        assert!(runner.is_active());

        // Frame 2 (warmup 1/2 -> transitions to Running)
        let _ = runner.prepare_frame_camera();
        let res = runner.finish_frame("GPU", "Vulkan", [1920, 1080]);
        assert!(res.is_none());
        assert!(runner.is_active());

        // Running Frame 1/3
        let _ = runner.prepare_frame_camera();
        let res = runner.finish_frame("GPU", "Vulkan", [1920, 1080]);
        assert!(res.is_none());

        // Running Frame 2/3
        let _ = runner.prepare_frame_camera();
        let res = runner.finish_frame("GPU", "Vulkan", [1920, 1080]);
        assert!(res.is_none());

        // Final frame 3/3 completes benchmark run.
        let _ = runner.prepare_frame_camera();
        let res = runner.finish_frame("GPU", "Vulkan", [1920, 1080]);
        assert!(res.is_some());
        assert!(!runner.is_active());
        assert_eq!(runner.state, BenchmarkState::Complete);

        let report = res.expect("INVARIANT: Valid benchmark run must return results");
        assert_eq!(report.total_frames, 3);
    }
}
