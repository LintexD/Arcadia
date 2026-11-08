//! GPU frame rendering dispatch, egui overlay rendering, and window HUD title profiling.
//! Owns GPU pass execution and swapchain presentation orchestration.

use std::time::Instant;

use glam::Vec3;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use arcadia_core::MAT_WATER_CRYSTAL;
use arcadia_physics::GpuRigidBodyInstance;
use arcadia_render::FrameParams;
use arcadia_world::DebrisParticle;

use crate::app::{App, CalibrationTask};
use crate::benchmark::BenchmarkState;
use crate::input::material_name;

impl App {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_render_frame(
        &mut self,
        event_loop: &ActiveEventLoop,
        eye_pos: Vec3,
        forward: Vec3,
        right: Vec3,
        up: Vec3,
        fov_y_radians: f32,
        sim_dt: f32,
        sun_dir: Vec3,
        sun_intensity: f32,
        target_hit_pos: Option<Vec3>,
        impact_debris: &[DebrisParticle],
        gpu_rigid_instances: Vec<GpuRigidBodyInstance>,
    ) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let Some(renderer) = &mut self.renderer else {
            return;
        };

        renderer.apply_world_patches(&mut self.world);

        if !impact_debris.is_empty() {
            renderer.spawn_debris(impact_debris);
        }
        let is_underwater = {
            let vx = (eye_pos.x * 16.0).floor() as i32;
            let vy = (eye_pos.y * 16.0).floor() as i32;
            let vz = (eye_pos.z * 16.0).floor() as i32;
            self.world.get_voxel(vx, vy, vz) == MAT_WATER_CRYSTAL
        };

        if let Some(gui) = &self.gui {
            self.quality = gui.quality;
        }

        let frame_params = FrameParams {
            eye_pos,
            forward,
            right,
            up,
            fov_y_radians,
            elapsed_time: self.total_time,
            dt: sim_dt,
            sun_dir,
            sun_intensity,
            brush_radius: self.brush_radius,
            target_pos: target_hit_pos,
            point_lights: self.point_lights.clone(),
            is_underwater,
            rigid_bodies: gpu_rigid_instances,
            quality: self.quality,
        };

        let mut gui = self.gui.take();
        let render_result = if let Some(gui_ref) = &mut gui {
            let gpu_name = renderer.adapter_name().to_string();
            renderer.render_with_gui(
                &frame_params,
                Some(&mut |dev, q, enc, view| {
                    gui_ref.render_frame(&window, dev, q, enc, view, &gpu_name);
                }),
            )
        } else {
            renderer.render(&frame_params)
        };
        if let Some(gui_ref) = &mut gui {
            self.quality = gui_ref.quality;
            if let Some(target) = gui_ref.calibration_requested.take() {
                println!(
                    ">>> Starting Hardware Auto-Calibration for {:.0} FPS (sampling 40 frames)... <<<",
                    target
                );
                self.calibration = Some(CalibrationTask::new(target, 40));
                gui_ref.calibration_status =
                    Some(format!("Measuring GPU throughput for {:.0} FPS...", target));
            }
        }
        self.gui = gui;

        match render_result {
            Ok(()) => {}
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let size = window.inner_size();
                renderer.resize(size.width, size.height);
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                log::error!("GPU Surface out of memory! Exiting.");
                event_loop.exit();
                return;
            }
            Err(wgpu::SurfaceError::Timeout) => {
                log::warn!("GPU Surface timeout");
            }
        }

        if self.benchmark.is_active() {
            let res = [renderer.width(), renderer.height()];
            let gpu_name = renderer.adapter_name();
            let gpu_backend = renderer.backend_name();

            if let Some(results) = self.benchmark.finish_frame(gpu_name, gpu_backend, res) {
                let title = format!(
                    "Arcadia [BENCHMARK COMPLETE] Score: {} | Avg: {:.1} FPS ({:.2}ms) | 1% Low: {:.1} FPS | [B] Re-test | Esc to exit",
                    results.score, results.avg_fps, results.avg_frame_time_ms, results.low_1_pct_fps
                );
                window.set_title(&title);

                if self.benchmark.config.auto_exit {
                    println!(
                        "Benchmark test completed with auto-exit enabled. Closing application."
                    );
                    event_loop.exit();
                    return;
                }
            }
        }

        self.update_window_title_hud(&window);
    }

    pub(crate) fn update_window_title_hud(&mut self, window: &Window) {
        self.frame_count += 1;
        let elapsed = self.fps_timer.elapsed();
        if elapsed.as_secs_f32() < 0.25 {
            return;
        }

        let frames_in_period = self.frame_count - self.last_fps_frame_count;
        let fps = frames_in_period as f32 / elapsed.as_secs_f32();
        let frame_time_ms = if fps > 0.0 { 1000.0 / fps } else { 0.0 };

        if let Some(gui) = &mut self.gui {
            gui.current_fps = fps;
            gui.current_frame_time_ms = frame_time_ms;
        }

        match &self.benchmark.state {
            BenchmarkState::Warmup { current, total } => {
                let title = format!(
                    "Arcadia [WARMUP: {}/{} frames] | {:.1} FPS ({:.2} ms)",
                    current, total, fps, frame_time_ms
                );
                window.set_title(&title);
            }
            BenchmarkState::Running { current, total } => {
                let pct = (*current as f32 / *total as f32) * 100.0;
                let title = format!(
                    "Arcadia [BENCHMARK: {}/{} frames ({:.0}%)] | {:.1} FPS ({:.2} ms)",
                    current, total, pct, fps, frame_time_ms
                );
                window.set_title(&title);
            }
            BenchmarkState::Complete => {
                if let Some(results) = &self.benchmark.latest_results {
                    let title = format!(
                        "Arcadia [BENCHMARK COMPLETE] Score: {} | Avg: {:.1} FPS ({:.2}ms) | 1% Low: {:.1} FPS | [B] Re-test",
                        results.score, results.avg_fps, results.avg_frame_time_ms, results.low_1_pct_fps
                    );
                    window.set_title(&title);
                }
            }
            BenchmarkState::Idle => {
                let mode_str = if self.player.fly_mode { "Fly" } else { "Walk" };
                let pos = self.player.foot_pos;
                let mat_str = material_name(self.selected_mat);
                let scorch_str = if self.scorch_rim {
                    " [Scorch: ON]"
                } else {
                    " [Scorch: OFF]"
                };
                let title = format!(
                    "Arcadia — {:.1} FPS ({:.2} ms) | Mode: {} | Pos: ({:.1}, {:.1}, {:.1}) | Brush: {:.2}m{} | Mat: {} | [F1] Debug | [F3] Graph | [B] Benchmark",
                    fps,
                    frame_time_ms,
                    mode_str,
                    pos.x,
                    pos.y,
                    pos.z,
                    self.brush_radius,
                    scorch_str,
                    mat_str
                );
                window.set_title(&title);
            }
        }

        self.fps_timer = Instant::now();
        self.last_fps_frame_count = self.frame_count;
    }
}
