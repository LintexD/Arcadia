//! Command-line argument parsing and application launch configuration.
//! Owns CPU CLI configuration state.

use std::env;

use arcadia_render::RaymarchQualitySettings;

use crate::benchmark::{BenchmarkConfig, TimeOfDayMode};

/// Parsed command-line configuration for the Arcadia application.
#[derive(Clone, Debug)]
pub struct LaunchConfig {
    pub scene_name: String,
    pub auto_start_bench: bool,
    pub benchmark_config: BenchmarkConfig,
    pub initial_quality: Option<RaymarchQualitySettings>,
}

impl LaunchConfig {
    /// Parses application options from standard environment arguments.
    pub fn parse() -> Result<Option<Self>, Box<dyn std::error::Error>> {
        Self::parse_from(env::args())
    }

    /// Parses application options from an arbitrary iterator of string arguments.
    pub fn parse_from(
        args: impl IntoIterator<Item = String>,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let args: Vec<String> = args.into_iter().collect();
        let mut scene_name = "benchmark".to_string();
        let mut auto_start_bench = false;
        let mut warmup_frames = 60u32;
        let mut test_frames = 900u32;
        let mut auto_exit = false;
        let mut time_of_day = TimeOfDayMode::Cycle;
        let mut initial_quality: Option<RaymarchQualitySettings> = None;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--bench" | "--benchmark" => {
                    scene_name = "benchmark".to_string();
                    auto_start_bench = true;
                }
                "--interactive" | "--no-bench" => {
                    auto_start_bench = false;
                }
                "--calibrated60" | "--60fps" => {
                    initial_quality = Some(RaymarchQualitySettings::calibrated_60fps());
                }
                "--calibrated30" | "--30fps" | "--laptop" => {
                    initial_quality = Some(RaymarchQualitySettings::calibrated_30fps());
                }
                "--preset" => {
                    if i + 1 < args.len() {
                        match args[i + 1].to_lowercase().as_str() {
                            "60" | "60fps" | "calibrated60" | "fluid" => {
                                initial_quality = Some(RaymarchQualitySettings::calibrated_60fps());
                            }
                            "30" | "30fps" | "calibrated30" | "laptop" => {
                                initial_quality = Some(RaymarchQualitySettings::calibrated_30fps());
                            }
                            "high" | "default" => {
                                initial_quality = Some(RaymarchQualitySettings::high_quality());
                            }
                            "ultra" => {
                                initial_quality = Some(RaymarchQualitySettings::ultra());
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                }
                "--scale" | "--render-scale" => {
                    if i + 1 < args.len() {
                        if let Ok(scale) = args[i + 1].parse::<f32>() {
                            let mut q = initial_quality
                                .unwrap_or_else(RaymarchQualitySettings::laptop_igpu);
                            q.render_scale = scale.clamp(0.40, 1.0);
                            initial_quality = Some(q);
                        }
                        i += 1;
                    }
                }
                "--scene" => {
                    if i + 1 < args.len() {
                        scene_name = args[i + 1].to_lowercase();
                        i += 1;
                    }
                }
                "--tod" | "--time-of-day" => {
                    if i + 1 < args.len() {
                        if let Some(mode) = TimeOfDayMode::parse(&args[i + 1]) {
                            time_of_day = mode;
                        } else {
                            eprintln!(
                                "Unknown time of day '{}'. Options: cycle, day, sunset, night, sunrise",
                                args[i + 1]
                            );
                        }
                        i += 1;
                    }
                }
                "--frames" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<u32>() {
                            test_frames = val;
                        }
                        i += 1;
                    }
                }
                "--warmup" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<u32>() {
                            warmup_frames = val;
                        }
                        i += 1;
                    }
                }
                "--auto-exit" | "--exit" => {
                    auto_exit = true;
                }
                "--help" | "-h" => {
                    print_help();
                    return Ok(None);
                }
                _ => {}
            }
            i += 1;
        }

        let benchmark_config = BenchmarkConfig {
            warmup_frames,
            test_frames,
            auto_exit,
            save_json: true,
            save_markdown: true,
            time_of_day,
        };

        Ok(Some(Self {
            scene_name,
            auto_start_bench,
            benchmark_config,
            initial_quality,
        }))
    }
}

fn print_help() {
    println!("Arcadia Voxel Engine — Options:");
    println!("  --bench, --benchmark    Launch Benchmark Scene & start pure render test (default)");
    println!("  --interactive           Launch without starting benchmark test automatically");
    println!("  --scene <name>          Select world scene: 'benchmark' (default) or 'fantasy'");
    println!("  --tod <mode>            Time of day: 'cycle' (default), 'day', 'sunset', 'night', 'sunrise'");
    println!("  --frames <N>            Number of benchmark frames (default: 900 = 15s @ 60 FPS)");
    println!("  --warmup <N>            Number of warmup frames (default: 60)");
    println!("  --auto-exit, --exit     Automatically exit after benchmark completes");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_launch_config_defaults() {
        let args = vec!["arcadia".to_string()];
        let config = LaunchConfig::parse_from(args)
            .expect("INVARIANT: Valid parse")
            .expect("INVARIANT: Expected LaunchConfig");
        assert_eq!(config.scene_name, "benchmark");
        assert!(!config.auto_start_bench);
        assert_eq!(config.benchmark_config.warmup_frames, 60);
        assert_eq!(config.benchmark_config.test_frames, 900);
        assert!(!config.benchmark_config.auto_exit);
        assert!(config.initial_quality.is_none());
    }

    #[test]
    fn test_launch_config_flags() {
        let args = vec![
            "arcadia".to_string(),
            "--bench".to_string(),
            "--scene".to_string(),
            "fantasy".to_string(),
            "--frames".to_string(),
            "300".to_string(),
            "--preset".to_string(),
            "60".to_string(),
            "--auto-exit".to_string(),
        ];
        let config = LaunchConfig::parse_from(args)
            .expect("INVARIANT: Valid parse")
            .expect("INVARIANT: Expected LaunchConfig");
        assert_eq!(config.scene_name, "fantasy");
        assert!(config.auto_start_bench);
        assert_eq!(config.benchmark_config.test_frames, 300);
        assert!(config.benchmark_config.auto_exit);
        assert!(config.initial_quality.is_some());
    }

    #[test]
    fn test_launch_config_help() {
        let args = vec!["arcadia".to_string(), "--help".to_string()];
        let result = LaunchConfig::parse_from(args).expect("INVARIANT: Valid parse");
        assert!(result.is_none());
    }
}
