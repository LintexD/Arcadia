//! Per-frame simulation stepping, camera input resolution, sun movement, and rigid body dynamics.
//! Owns CPU simulation step state and coordinates physics with rendering.

use std::time::Instant;

use glam::Vec3;
use winit::event_loop::ActiveEventLoop;

use arcadia_physics::{GpuRigidBodyInstance, MovementInput, RigidVoxelBody};
use arcadia_world::DebrisParticle;

use crate::app::{App, BakeResult};
use crate::benchmark::BenchmarkState;

/// Camera view parameters for visibility culling and baking gates.
#[derive(Clone, Copy, Debug)]
pub struct CameraView {
    pub eye_pos: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub fov_y_rad: f32,
    pub aspect_ratio: f32,
}

impl App {
    /// Advances simulation state and dispatches the GPU render pass.
    pub(crate) fn update_and_render(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let frame_delta_ms = (now - self.last_frame_instant).as_secs_f32() * 1000.0;
        let wall_dt = (now - self.last_frame_instant).as_secs_f32().min(0.05);
        self.last_frame_instant = now;
        self.total_time += wall_dt;

        if let Some(gui) = &mut self.gui {
            gui.push_frame_time(frame_delta_ms);
        }

        self.update_calibration(wall_dt);

        let (sun_dir, sun_intensity) = self.update_sun(wall_dt);

        let (eye_pos, forward, right, up, fov_y_radians, sim_dt) =
            self.determine_camera_and_input(wall_dt);

        let hit = self.raycast_scene(eye_pos, forward, 60.0);
        let target_hit_pos = hit.map(|h| h.hit.hit_pos);

        let aspect_ratio = self
            .renderer
            .as_ref()
            .map(|r| r.width() as f32 / r.height().max(1) as f32)
            .unwrap_or(16.0 / 9.0);
        let camera_view = CameraView {
            eye_pos,
            forward,
            right,
            up,
            fov_y_rad: fov_y_radians,
            aspect_ratio,
        };

        let (impact_debris, gpu_rigid_instances) =
            self.step_and_collect_rigid_bodies(sim_dt, Some(camera_view));

        self.dispatch_render_frame(
            event_loop,
            eye_pos,
            forward,
            right,
            up,
            fov_y_radians,
            sim_dt,
            sun_dir,
            sun_intensity,
            target_hit_pos,
            &impact_debris,
            gpu_rigid_instances,
        );
    }

    pub(crate) fn update_calibration(&mut self, wall_dt: f32) {
        let Some(calib) = &mut self.calibration else {
            return;
        };

        calib.accum_frame_time_ms += wall_dt * 1000.0;
        calib.sample_count += 1;
        calib.frames_remaining = calib.frames_remaining.saturating_sub(1);
        if calib.frames_remaining == 0 {
            let avg_ms = calib.accum_frame_time_ms / calib.sample_count.max(1) as f32;
            let measured_fps = if avg_ms > 0.0 { 1000.0 / avg_ms } else { 30.0 };
            let target_fps = calib.target_fps;

            self.quality
                .calibrate_for_target_fps(measured_fps, target_fps);
            let status_msg = format!(
                "Calibrated for {:.0} FPS: {:.0}% Scale, {} Primary Steps, {} GI Steps (Measured: {:.1} FPS)",
                target_fps,
                self.quality.render_scale * 100.0,
                self.quality.primary_max_steps,
                self.quality.gi_max_steps,
                measured_fps
            );
            println!(">>> {} <<<", status_msg);
            if let Some(gui) = &mut self.gui {
                gui.quality = self.quality;
                gui.calibration_status = Some(status_msg);
            }
            self.calibration = None;
        }
    }

    pub(crate) fn update_sun(&mut self, wall_dt: f32) -> (Vec3, f32) {
        if self.benchmark.is_active() {
            let progress = match self.benchmark.state {
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
            self.sun_angle = self.benchmark.get_sun_angle(progress);
        } else {
            if self.keys.q {
                self.sun_angle -= 0.65 * wall_dt;
            }
            if self.keys.e {
                self.sun_angle += 0.65 * wall_dt;
            }
        }
        self.sun_angle = self.sun_angle.rem_euclid(std::f32::consts::TAU);
        let sun_dir =
            Vec3::new(self.sun_angle.cos() * 0.92, self.sun_angle.sin(), 0.38).normalize();
        let sun_intensity = ((sun_dir.y + 0.04) / 0.25).clamp(0.0, 1.0) * 1.15;
        (sun_dir, sun_intensity)
    }

    pub(crate) fn determine_camera_and_input(
        &mut self,
        wall_dt: f32,
    ) -> (Vec3, Vec3, Vec3, Vec3, f32, f32) {
        if self.benchmark.is_active() {
            let (cam_pos, look_at, fov_deg) = self.benchmark.prepare_frame_camera();
            let forward = (look_at - cam_pos).normalize();
            let right = forward.cross(Vec3::Y).normalize();
            let up = right.cross(forward).normalize();
            (
                cam_pos,
                forward,
                right,
                up,
                fov_deg.to_radians(),
                1.0 / 60.0_f32,
            )
        } else {
            let mut move_dir = Vec3::ZERO;
            if self.keys.w {
                move_dir.z += 1.0;
            }
            if self.keys.s {
                move_dir.z -= 1.0;
            }
            if self.keys.d {
                move_dir.x += 1.0;
            }
            if self.keys.a {
                move_dir.x -= 1.0;
            }
            if self.player.fly_mode {
                if self.keys.space {
                    move_dir.y += 1.0;
                }
                if self.keys.shift {
                    move_dir.y -= 1.0;
                }
            }
            let input = MovementInput {
                move_dir,
                sprint: self.keys.ctrl,
                jump: self.keys.space,
            };
            self.player
                .update(wall_dt, input, &self.world, &mut self.rigid_bodies);

            (
                self.player.eye_pos(),
                self.player.forward(),
                self.player.right(),
                self.player.up(),
                self.quality.fov_deg.to_radians(),
                wall_dt,
            )
        }
    }

    /// Advances physics for dynamic rigid bodies and asynchronously bakes resting, off-screen bodies into the static world.
    pub fn step_and_collect_rigid_bodies(
        &mut self,
        sim_dt: f32,
        camera: Option<CameraView>,
    ) -> (Vec<DebrisParticle>, Vec<GpuRigidBodyInstance>) {
        // Atomic handshake: consume at most 1 background bake completion per frame,
        // updating static world octrees and retiring the dynamic rigid body seamlessly.
        if let Ok(result) = self.bake_rx.try_recv() {
            if let Some(pos) = self
                .rigid_bodies
                .iter()
                .position(|b| b.id == result.body_id)
            {
                self.rigid_bodies.swap_remove(pos);
                if !result.voxels.is_empty() {
                    self.world.set_voxels_global(&result.voxels);
                }
            }
        }

        let mut impact_debris = Vec::new();
        let system_debris =
            RigidVoxelBody::step_system(&mut self.rigid_bodies, sim_dt, &self.world);
        if !system_debris.is_empty() {
            impact_debris.extend(system_debris);
        }

        let bake_timeout = self.quality.object_bake_timeout_s;
        let forced_timeout = bake_timeout * 3.0;
        let mut active_baking_count = self.rigid_bodies.iter().filter(|b| b.is_baking).count();
        let mut gpu_rigid_instances: Vec<GpuRigidBodyInstance> =
            Vec::with_capacity(self.rigid_bodies.len());

        self.rigid_bodies.retain_mut(|body| {
            if body.position.y < -8.0 {
                return false;
            }
            if body.is_baking {
                gpu_rigid_instances.push(body.to_gpu_instance());
                return true;
            }
            if body.rest_time >= bake_timeout {
                let is_visible = camera.is_some_and(|cam| {
                    body.is_visible_to_camera(
                        cam.eye_pos,
                        cam.forward,
                        cam.right,
                        cam.up,
                        cam.fov_y_rad,
                        cam.aspect_ratio,
                        1.15, // WHY: 15% peripheral margin prevents popping at screen edge.
                    )
                });
                let forced_bake = body.rest_time >= forced_timeout;
                if (!is_visible || forced_bake) && active_baking_count < 1 {
                    body.is_baking = true;
                    body.sleeping = true;
                    body.linear_velocity = Vec3::ZERO;
                    body.angular_velocity = Vec3::ZERO;
                    active_baking_count += 1;

                    let body_clone = body.clone();
                    let tx = self.bake_tx.clone();
                    rayon::spawn(move || {
                        let voxels = body_clone.to_world_voxels();
                        let _ = tx.send(BakeResult {
                            body_id: body_clone.id,
                            voxels,
                        });
                    });
                }
            }
            gpu_rigid_instances.push(body.to_gpu_instance());
            true
        });

        (impact_debris, gpu_rigid_instances)
    }
}
