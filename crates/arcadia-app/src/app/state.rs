//! Application core state definitions, startup world construction, and cursor grabbing.
//! Owns CPU application state and handles initialization.

use std::sync::{mpsc, Arc};
use std::time::Instant;

use glam::{IVec3, UVec3, Vec2, Vec3};
use winit::window::{CursorGrabMode, Window};

use arcadia_core::{
    ChunkTree64, MaterialId, MaterialPalette, CHUNK_DIM, MAT_CRYSTAL_CYAN, VOXEL_SIZE,
};
use arcadia_physics::{CharacterController, RigidVoxelBody};
use arcadia_render::{RaymarchQualitySettings, Renderer};
use arcadia_world::{generate_benchmark_world, generate_fantasy_ruins_world, WorldMap};

use crate::benchmark::BenchmarkRunner;
use crate::cli::LaunchConfig;
use crate::gui;
use crate::input::KeyState;

/// Active hardware calibration task that measures frame throughput over a fixed sample window.
#[derive(Clone, Debug)]
pub struct CalibrationTask {
    pub target_fps: f32,
    pub frames_remaining: u32,
    pub accum_frame_time_ms: f32,
    pub sample_count: u32,
}

impl CalibrationTask {
    /// Creates a new calibration task targeting the requested FPS over a frame count.
    pub fn new(target_fps: f32, frames: u32) -> Self {
        Self {
            target_fps,
            frames_remaining: frames,
            accum_frame_time_ms: 0.0,
            sample_count: 0,
        }
    }
}

/// Result of background rigid body rasterization ready to be committed into the static world map.
pub struct BakeResult {
    /// ID of the dynamic rigid body that was baked.
    pub body_id: u32,
    /// Rasterized world-space voxels.
    pub voxels: Vec<(IVec3, MaterialId)>,
}

/// Application state implementing the `winit 0.30` `ApplicationHandler`.
pub struct App {
    pub world: WorldMap,
    pub player: CharacterController,
    pub(crate) window: Option<Arc<Window>>,
    pub(crate) renderer: Option<Renderer>,
    pub(crate) cursor_grabbed: bool,
    pub(crate) keys: KeyState,
    pub brush_radius: f32,
    pub selected_mat: MaterialId,
    pub scorch_rim: bool,
    pub(crate) sun_angle: f32,
    /// Active user-placed analytical 3D point lights (`[x, y, z, radius]`, up to 8).
    pub(crate) point_lights: Vec<[f32; 4]>,
    /// Active 6-DOF dynamic rigid body voxel islands (`RigidVoxelBody`).
    pub rigid_bodies: Vec<RigidVoxelBody>,
    pub(crate) bake_tx: mpsc::Sender<BakeResult>,
    pub(crate) bake_rx: mpsc::Receiver<BakeResult>,
    pub(crate) last_frame_instant: Instant,
    pub(crate) last_space_press: Option<Instant>,
    pub(crate) total_time: f32,
    pub(crate) fps_timer: Instant,
    pub(crate) frame_count: u32,
    pub(crate) last_fps_frame_count: u32,
    pub(crate) benchmark: BenchmarkRunner,
    #[allow(dead_code)]
    pub(crate) is_benchmark_scene: bool,
    pub(crate) gui: Option<gui::DebugUi>,
    pub quality: RaymarchQualitySettings,
    pub(crate) user_quality: Option<RaymarchQualitySettings>,
    pub(crate) calibration: Option<CalibrationTask>,
}

impl App {
    /// Creates a headless test application instance for simulation and sculpting verification.
    pub fn new_test() -> Self {
        let palette = MaterialPalette::default();
        let world = WorldMap::from_chunks(UVec3::ONE, vec![ChunkTree64::new_empty()], palette);
        let player = CharacterController::new(Vec3::new(0.5, 0.5, -2.0), 0.0, 0.0);
        let (bake_tx, bake_rx) = mpsc::channel();
        App {
            world,
            player,
            window: None,
            renderer: None,
            cursor_grabbed: false,
            keys: KeyState::default(),
            brush_radius: 0.15,
            selected_mat: MAT_CRYSTAL_CYAN,
            scorch_rim: false,
            sun_angle: 0.0,
            point_lights: Vec::new(),
            rigid_bodies: Vec::new(),
            bake_tx,
            bake_rx,
            last_frame_instant: Instant::now(),
            last_space_press: None,
            total_time: 0.0,
            fps_timer: Instant::now(),
            frame_count: 0,
            last_fps_frame_count: 0,
            benchmark: BenchmarkRunner::new(Default::default(), Vec2::ZERO),
            is_benchmark_scene: false,
            gui: None,
            quality: Default::default(),
            user_quality: None,
            calibration: None,
        }
    }
    /// Constructs the application from parsed launch configuration.
    pub fn new(config: LaunchConfig) -> Self {
        let t0 = Instant::now();
        let grid_dims = UVec3::new(10, 4, 10);
        let world_size_m = grid_dims.as_vec3() * (CHUNK_DIM as f32 * VOXEL_SIZE);
        let center_xz = Vec2::new(world_size_m.x * 0.5, world_size_m.z * 0.5);

        let is_benchmark = config.scene_name == "benchmark";

        println!(
            "================================================================================"
        );
        println!("  Arcadia Voxel Engine — (1/16m)³ CS64 Ray Marcher + ReSTIR GI");
        println!(
            "================================================================================"
        );

        let mut benchmark = BenchmarkRunner::new(config.benchmark_config, center_xz);
        if config.auto_start_bench {
            benchmark.start();
        }

        let (world, point_lights, sun_angle, spawn_pos) = if is_benchmark {
            println!(
                "Generating Benchmark Sanctuary Scene (Colonnade, Mirrors, Water, Crystals, GI)..."
            );
            println!(
                "World volume: {}x{}x{} chunks = {}m x {}m x {}m (6.71 billion micro-voxels)",
                grid_dims.x,
                grid_dims.y,
                grid_dims.z,
                grid_dims.x * 16,
                grid_dims.y * 16,
                grid_dims.z * 16
            );
            println!(
                "Benchmark lighting mode: {}",
                benchmark.config.time_of_day.as_str()
            );
            let w = generate_benchmark_world(grid_dims);
            let lights = Vec::new();
            let initial_sun = benchmark.get_sun_angle(0.0);
            (
                w,
                lights,
                initial_sun,
                Vec3::new(center_xz.x, 13.5, center_xz.y - 18.0),
            )
        } else {
            println!(
                "Generating Fantasy Nature & Ruins world ({}x{}x{} chunks = {}m x {}m x {}m)...",
                grid_dims.x,
                grid_dims.y,
                grid_dims.z,
                grid_dims.x * 16,
                grid_dims.y * 16,
                grid_dims.z * 16
            );
            let w = generate_fantasy_ruins_world(grid_dims);
            (w, Vec::new(), 0.85f32, Vec3::new(80.0, 13.0, 62.0))
        };

        let gen_duration = t0.elapsed();
        println!("World generation complete in {:.2?}!", gen_duration);
        println!(
            "  Chunks: {} total ({} non-empty roots)",
            world.chunks.len(),
            world
                .gpu_chunk_roots
                .iter()
                .filter(|&&r| r != arcadia_world::EMPTY_CHUNK_SENTINEL)
                .count()
        );
        println!(
            "  Packed GPU Nodes:  {} ({:.2} MB)",
            world.gpu_nodes.len(),
            (world.gpu_nodes.len() * std::mem::size_of::<arcadia_core::Node64>()) as f64
                / (1024.0 * 1024.0)
        );
        println!(
            "  Packed GPU Bricks: {} ({:.2} MB)",
            world.gpu_bricks.len(),
            (world.gpu_bricks.len() * std::mem::size_of::<arcadia_core::LeafBrick64>()) as f64
                / (1024.0 * 1024.0)
        );
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!("Controls:");
        println!("  [Click Window]    Lock cursor / FPS mouse look");
        println!("  [Escape]          Unlock cursor / Exit application");
        println!("  [W, A, S, D]      Walk / Strafe / Fly");
        println!("  [2x Space]        Toggle Walk Mode / Free Fly Mode");
        println!("  [Space]           Jump (Walk) / Fly Up (Fly mode)");
        println!("  [Shift]           Fly Down (Fly mode)");
        println!("  [Ctrl]            Sprint");
        println!(
            "  [B]               Run Pure Render Benchmark Test (Warmup -> Trajectory -> Report)"
        );
        println!("  [Left Click]      Explosive sphere carve");
        println!("  [Right Click]     Place sphere of selected material");
        println!("  [Mouse Wheel]     Adjust brush radius (0.25m - 4.50m)");
        println!("  [1 - 8]           Select build material:");
        println!("                    1: Cyan Crystal, 2: Magenta Crystal, 3: Amber Sunstone,");
        println!("                    4: Molten Lava, 5: Carved Slate, 6: Lush Moss,");
        println!("                    7: Point Light Beacon (Shadowed Point Light),");
        println!("                    8: Polished Mirror Chrome (Real-Time Reflection)");
        println!("  [G]               Toggle scorch / ember blast rim");
        println!("  [Q / E]           Rotate sun angle (Day -> Sunset -> Night -> Sunrise)");
        println!("  [N]               Cycle Time-of-Day preset (Day / Sunset / Midnight)");
        println!("  [C]               Auto-Calibrate settings for target 60 FPS (1-second hardware test)");
        println!("  [L]               Cycle Calibrated 60 FPS Fluid / 30 FPS Fidelity");
        println!("  [K]               Cycle Dithering Mode (PCG Decorrelated [Liked] / Spatial Bit-Mix / Off)");
        println!("  [J]               Cycle Camera Jitter Mode (Freeze when Still [Zero Flicker] / Off / Always On)");
        println!("  [F1 or Tab]       Toggle Raymarcher Quality & Debug Interface");
        println!(
            "================================================================================"
        );

        let player = CharacterController::new(spawn_pos, std::f32::consts::PI, 0.0);

        let (bake_tx, bake_rx) = mpsc::channel();

        Self {
            world,
            player,
            window: None,
            renderer: None,
            cursor_grabbed: false,
            keys: KeyState::default(),
            brush_radius: 1.25,
            selected_mat: MAT_CRYSTAL_CYAN,
            scorch_rim: true,
            sun_angle,
            point_lights,
            rigid_bodies: Vec::new(),
            bake_tx,
            bake_rx,
            last_frame_instant: Instant::now(),
            last_space_press: None,
            total_time: 0.0,
            fps_timer: Instant::now(),
            frame_count: 0,
            last_fps_frame_count: 0,
            benchmark,
            is_benchmark_scene: is_benchmark,
            gui: None,
            quality: config
                .initial_quality
                .unwrap_or_else(RaymarchQualitySettings::calibrated_30fps),
            user_quality: config.initial_quality,
            calibration: None,
        }
    }

    /// Toggles or updates cursor lock and visibility mode.
    pub(crate) fn set_cursor_grab(&mut self, grab: bool) {
        let Some(window) = &self.window else {
            return;
        };

        if grab {
            let grab_result = window.set_cursor_grab(CursorGrabMode::Locked);
            if grab_result.is_err() {
                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
            }
            window.set_cursor_visible(false);
            self.cursor_grabbed = true;
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
            self.cursor_grabbed = false;
        }
    }
}
