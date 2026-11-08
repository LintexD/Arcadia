//! `Renderer`: Vulkan compute-driven path tracer & rasterizer orchestrator.
//! Owns CPU render orchestration and GPU swapchain presentation.

use glam::{Mat4, UVec3, Vec3};
use std::sync::Arc;
use winit::window::Window;

use crate::camera::{compute_subpixel_jitter, reverse_z_perspective_matrix, CameraUniforms};
use crate::context::GpuContext;
use crate::di::DiResources;
use crate::gbuffer::ScreenResources;
use crate::gpu_pools::{create_group0_resources, Group0Resources};
use crate::pipeline::{self, PassLayouts, Pipelines};
use crate::quality::RaymarchQualitySettings;
use crate::sharc::{self, SharcResources};

/// Parameters supplied per frame from the game/character loop.
#[derive(Clone, Debug)]
pub struct FrameParams {
    pub eye_pos: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub elapsed_time: f32,
    pub dt: f32,
    pub sun_dir: Vec3,
    pub sun_intensity: f32,
    pub brush_radius: f32,
    pub target_pos: Option<Vec3>,
    /// Up to 8 user-placed analytical 3D point lights (`xyz` = position in meters, `w` = brush radius/intensity).
    pub point_lights: Vec<[f32; 4]>,
    pub is_underwater: bool,
    /// Active Phase 6 dynamic rigid body voxel instances (up to `MAX_RIGID_BODIES = 32`).
    pub rigid_bodies: Vec<arcadia_physics::GpuRigidBodyInstance>,
    /// Real-time tunable raymarching and quality settings.
    pub quality: RaymarchQualitySettings,
}

pub type GuiCallback<'a> =
    &'a mut dyn FnMut(&wgpu::Device, &wgpu::Queue, &mut wgpu::CommandEncoder, &wgpu::TextureView);

/// Vulkan compute-driven renderer for the Arcadia voxel world.
pub struct Renderer {
    pub context: GpuContext,
    pub window: Arc<Window>,
    pub gpu_pools: Group0Resources,
    pub pass_layouts: PassLayouts,
    pub pipelines: Pipelines,
    pub screen: ScreenResources,
    /// World-space SHARC voxel-surfel cache (4.5 MB); persists across resizes.
    pub sharc: SharcResources,
    /// ReGIR spatial grid and emissive light pool buffers/bind groups (1.14 MB).
    pub di: DiResources,

    render_width: u32,
    render_height: u32,
    frame_index: u64,
    prev_view_proj: Mat4,
    prev_eye: Vec3,
    prev_forward: Vec3,
    stationary_frames: u32,
    world_grid_dims: UVec3,
    debris_cursor: usize,
    active_debris: usize,
}

impl Renderer {
    /// Initializes the Vulkan high-performance GPU device, surface, storage buffers, and compute pipelines.
    pub async fn new(window: Arc<Window>, world: &arcadia_world::WorldMap) -> Self {
        let context = GpuContext::new(window.clone()).await;

        let gpu_pools = create_group0_resources(
            &context.device,
            &context.queue,
            world,
            context.max_storage_bytes,
        );

        let pass_layouts = pipeline::create_pass_layouts(&context.device);
        let pipelines = pipeline::create_pipelines(
            &context.device,
            context.surface_config.format,
            &gpu_pools.group0_bgl,
            &pass_layouts,
        );

        let screen = ScreenResources::new(
            &context.device,
            context.width,
            context.height,
            &pass_layouts.pass1_bgl,
            &pass_layouts.pass2_bgl,
            &pass_layouts.pass3_bgl,
            &pass_layouts.pass4_bgl,
            &pass_layouts.blit_bgl,
            &pass_layouts.blit_sampler,
        );

        let sharc = SharcResources::new(
            &context.device,
            &pass_layouts.sharc,
            &screen.gbuf_depth_mat_buf,
            &screen.gbuf_norm_motion_buf,
        );

        let mut di = DiResources::new(&context.device, &pass_layouts.di);
        di.update_buffers(&context.queue, &world.emissive);

        Self {
            context,
            window,
            gpu_pools,
            pass_layouts,
            pipelines,
            screen,
            sharc,
            di,
            render_width: 0,
            render_height: 0,
            frame_index: 0,
            prev_view_proj: Mat4::IDENTITY,
            prev_eye: Vec3::ZERO,
            prev_forward: Vec3::NEG_Z,
            stationary_frames: 0,
            world_grid_dims: world.grid_dims,
            debris_cursor: 0,
            active_debris: 0,
        }
    }

    /// Handles window resizing by reconfiguring the swapchain and rebuilding screen buffers.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if self.context.width == width && self.context.height == height {
            return;
        }
        self.context.resize(width, height);

        self.render_width = 0;
        self.render_height = 0;
    }

    /// Drains queued incremental GPU patches from `world` and writes them into GPU storage buffers.
    pub fn apply_world_patches(&mut self, world: &mut arcadia_world::WorldMap) {
        world.refresh_emissive_registry();
        self.di.update_buffers(&self.context.queue, &world.emissive);

        if world.pending_patches.is_empty() {
            return;
        }

        let max_storage_bytes = self.context.max_storage_bytes;
        let req_node_bytes =
            (world.gpu_nodes.len() * std::mem::size_of::<arcadia_core::Node64>()) as u64;
        let req_brick_bytes =
            (world.gpu_bricks.len() * std::mem::size_of::<arcadia_core::LeafBrick64>()) as u64;

        let old_node_size = self.gpu_pools.node_pool_buf.size();
        let old_brick_size = self.gpu_pools.brick_pool_buf.size();
        let node_needs_growth = req_node_bytes > old_node_size;
        let brick_needs_growth = req_brick_bytes > old_brick_size;

        if node_needs_growth || brick_needs_growth {
            let mut copy_encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("pool_growth_copy_encoder"),
                    });

            if node_needs_growth {
                let new_size = (req_node_bytes + 32 * 1024 * 1024).min(max_storage_bytes);
                let new_node_buf = self.context.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("node_pool_buf_grown"),
                    size: new_size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                if old_node_size > 0 {
                    copy_encoder.copy_buffer_to_buffer(
                        &self.gpu_pools.node_pool_buf,
                        0,
                        &new_node_buf,
                        0,
                        old_node_size,
                    );
                }
                self.gpu_pools.node_pool_buf = new_node_buf;
            }

            if brick_needs_growth {
                let new_size = (req_brick_bytes + 32 * 1024 * 1024).min(max_storage_bytes);
                let new_brick_buf = self.context.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("brick_pool_buf_grown"),
                    size: new_size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                if old_brick_size > 0 {
                    copy_encoder.copy_buffer_to_buffer(
                        &self.gpu_pools.brick_pool_buf,
                        0,
                        &new_brick_buf,
                        0,
                        old_brick_size,
                    );
                }
                self.gpu_pools.brick_pool_buf = new_brick_buf;
            }

            // GPU-SYNC: Submit zero-staging GPU-to-GPU copies before subsequent write_buffer staging copies.
            self.context
                .queue
                .submit(std::iter::once(copy_encoder.finish()));
            self.gpu_pools.rebuild_bind_group(&self.context.device);
        }

        let mut staged_bytes_this_call: u64 = 0;

        if node_needs_growth {
            let all_nodes_bytes = bytemuck::cast_slice(&world.gpu_nodes);
            let upload_end =
                (all_nodes_bytes.len() as u64).min(self.gpu_pools.node_pool_buf.size());
            if upload_end > old_node_size {
                let delta_slice = &all_nodes_bytes[old_node_size as usize..upload_end as usize];
                write_buffer_chunked(
                    &self.context.device,
                    &self.context.queue,
                    &self.gpu_pools.node_pool_buf,
                    old_node_size,
                    delta_slice,
                    &mut staged_bytes_this_call,
                );
            }
        }

        if brick_needs_growth {
            let all_bricks_bytes = bytemuck::cast_slice(&world.gpu_bricks);
            let upload_end =
                (all_bricks_bytes.len() as u64).min(self.gpu_pools.brick_pool_buf.size());
            if upload_end > old_brick_size {
                let delta_slice = &all_bricks_bytes[old_brick_size as usize..upload_end as usize];
                write_buffer_chunked(
                    &self.context.device,
                    &self.context.queue,
                    &self.gpu_pools.brick_pool_buf,
                    old_brick_size,
                    delta_slice,
                    &mut staged_bytes_this_call,
                );
            }
        }

        for patch in world.pending_patches.drain(..) {
            if patch.chunk_flat_idx != arcadia_world::EMPTY_CHUNK_SENTINEL {
                let root_offset_bytes =
                    (patch.chunk_flat_idx as u64) * (std::mem::size_of::<u32>() as u64);
                if root_offset_bytes + (std::mem::size_of::<u32>() as u64)
                    <= self.gpu_pools.chunk_roots_buf.size()
                {
                    write_buffer_chunked(
                        &self.context.device,
                        &self.context.queue,
                        &self.gpu_pools.chunk_roots_buf,
                        root_offset_bytes,
                        bytemuck::bytes_of(&patch.new_root_node_idx),
                        &mut staged_bytes_this_call,
                    );
                }
            }

            if !patch.nodes.is_empty() {
                let node_offset_bytes = (patch.node_offset as u64)
                    * (std::mem::size_of::<arcadia_core::Node64>() as u64);
                let slice_bytes = bytemuck::cast_slice(&patch.nodes);
                if node_offset_bytes + (slice_bytes.len() as u64)
                    <= self.gpu_pools.node_pool_buf.size()
                {
                    write_buffer_chunked(
                        &self.context.device,
                        &self.context.queue,
                        &self.gpu_pools.node_pool_buf,
                        node_offset_bytes,
                        slice_bytes,
                        &mut staged_bytes_this_call,
                    );
                }
            }

            if !patch.bricks.is_empty() {
                let brick_offset_bytes = (patch.brick_offset as u64)
                    * (std::mem::size_of::<arcadia_core::LeafBrick64>() as u64);
                let slice_bytes = bytemuck::cast_slice(&patch.bricks);
                if brick_offset_bytes + (slice_bytes.len() as u64)
                    <= self.gpu_pools.brick_pool_buf.size()
                {
                    write_buffer_chunked(
                        &self.context.device,
                        &self.context.queue,
                        &self.gpu_pools.brick_pool_buf,
                        brick_offset_bytes,
                        slice_bytes,
                        &mut staged_bytes_this_call,
                    );
                }
            }
        }
    }

    /// Writes newly spawned 6.25cm voxel debris particles into the GPU ring buffer.
    pub fn spawn_debris(&mut self, particles: &[arcadia_world::DebrisParticle]) {
        if particles.is_empty() {
            return;
        }
        let max = arcadia_world::MAX_DEBRIS_PARTICLES;
        let particle_size = std::mem::size_of::<arcadia_world::DebrisParticle>();

        let mut remaining = particles;
        while !remaining.is_empty() {
            let start = self.debris_cursor % max;
            let count_to_end = (max - start).min(remaining.len());
            let chunk = &remaining[..count_to_end];

            let offset = (start * particle_size) as u64;
            self.context.queue.write_buffer(
                &self.gpu_pools.debris_pool_buf,
                offset,
                bytemuck::cast_slice(chunk),
            );

            self.debris_cursor = (start + count_to_end) % max;
            self.active_debris = (self.active_debris + chunk.len()).min(max);
            remaining = &remaining[count_to_end..];
        }
    }

    /// Advances the frame: updates uniforms, dispatches all compute passes, and renders the fullscreen blit.
    pub fn render(&mut self, params: &FrameParams) -> Result<(), wgpu::SurfaceError> {
        self.render_with_gui(params, None)
    }

    /// Advances the frame with an optional GUI overlay render callback.
    pub fn render_with_gui(
        &mut self,
        params: &FrameParams,
        mut gui_cb: Option<GuiCallback<'_>>,
    ) -> Result<(), wgpu::SurfaceError> {
        if self.context.width == 0 || self.context.height == 0 {
            return Ok(());
        }

        let scale = self.ensure_render_size(params.quality.render_scale);
        let uniforms = self.prepare_camera_uniforms(params, scale);

        self.context.queue.write_buffer(
            &self.gpu_pools.camera_uniform_buf,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        let surface_texture = match self.context.surface.get_current_texture() {
            Ok(tex) => tex,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.context
                    .surface
                    .configure(&self.context.device, &self.context.surface_config);
                return Ok(());
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(()),
            Err(e) => return Err(e),
        };

        let view_target = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("surface_view_target"),
                ..Default::default()
            });

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame_command_encoder"),
                });

        let ping_pong = (self.frame_index & 1) as usize;
        let workgroups_x = self.render_width.div_ceil(8);
        let workgroups_y = self.render_height.div_ceil(8);

        self.encode_render_passes(
            &mut encoder,
            &view_target,
            ping_pong,
            workgroups_x,
            workgroups_y,
            params.quality.sharc_tile_size(),
        );

        if let Some(cb) = gui_cb.as_mut() {
            cb(
                &self.context.device,
                &self.context.queue,
                &mut encoder,
                &view_target,
            );
        }

        self.context.queue.submit(Some(encoder.finish()));
        surface_texture.present();

        self.frame_index = self.frame_index.wrapping_add(1);

        Ok(())
    }

    #[inline]
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.context.surface_config.format
    }

    #[inline]
    pub fn window(&self) -> &Window {
        &self.window
    }

    #[inline]
    pub fn device(&self) -> &wgpu::Device {
        &self.context.device
    }

    #[inline]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.context.queue
    }

    #[inline]
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.context.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.context.height
    }

    #[inline]
    pub fn adapter_name(&self) -> &str {
        &self.context.adapter_name
    }

    #[inline]
    pub fn backend_name(&self) -> &str {
        &self.context.backend_name
    }

    #[inline]
    pub fn is_integrated_gpu(&self) -> bool {
        self.context.is_integrated_gpu
    }

    #[inline]
    pub fn render_width(&self) -> u32 {
        self.render_width
    }

    #[inline]
    pub fn render_height(&self) -> u32 {
        self.render_height
    }

    fn ensure_render_size(&mut self, render_scale: f32) -> f32 {
        let scale = render_scale.clamp(0.35, 1.0);
        let target_rw = (((self.context.width as f32 * scale).round() as u32) & !1).max(64);
        let target_rh = (((self.context.height as f32 * scale).round() as u32) & !1).max(64);
        if target_rw != self.render_width || target_rh != self.render_height {
            self.render_width = target_rw;
            self.render_height = target_rh;
            self.screen = ScreenResources::new(
                &self.context.device,
                target_rw,
                target_rh,
                &self.pass_layouts.pass1_bgl,
                &self.pass_layouts.pass2_bgl,
                &self.pass_layouts.pass3_bgl,
                &self.pass_layouts.pass4_bgl,
                &self.pass_layouts.blit_bgl,
                &self.pass_layouts.blit_sampler,
            );
            self.sharc.rebuild_gbuf_bind_group(
                &self.context.device,
                &self.pass_layouts.sharc,
                &self.screen.gbuf_depth_mat_buf,
                &self.screen.gbuf_norm_motion_buf,
            );
            self.frame_index = 0;
        }
        scale
    }

    fn prepare_camera_uniforms(&mut self, params: &FrameParams, scale: f32) -> CameraUniforms {
        let aspect = (self.context.width as f32) / (self.context.height as f32);
        let tan_half_fov = (params.fov_y_radians * 0.5).tan();

        let eye = params.eye_pos;
        let forward = params.forward.normalize();
        let right = params.right.normalize();
        let up = right.cross(forward).normalize();

        let view = Mat4::look_to_rh(eye, forward, up);
        let proj = reverse_z_perspective_matrix(params.fov_y_radians, aspect, 0.1);
        let curr_view_proj = proj * view;
        let prev_view_proj = if self.frame_index == 0 {
            curr_view_proj
        } else {
            self.prev_view_proj
        };
        self.prev_view_proj = curr_view_proj;

        let prev_eye_for_uniform = if self.frame_index == 0 {
            eye
        } else {
            self.prev_eye
        };
        let is_moving = (eye - self.prev_eye).length_squared() > 1e-6
            || (forward - self.prev_forward).length_squared() > 1e-6;
        if is_moving {
            self.stationary_frames = 0;
        } else {
            self.stationary_frames = self.stationary_frames.saturating_add(1);
        }
        self.prev_eye = eye;
        self.prev_forward = forward;

        let (jx, jy) = compute_subpixel_jitter(
            params.quality.jitter_mode,
            self.frame_index,
            scale,
            is_moving,
            self.stationary_frames,
        );

        let target_info = if let Some(target) = params.target_pos {
            [target.x, target.y, target.z, 1.0]
        } else {
            [0.0, 0.0, 0.0, 0.0]
        };

        let mut pl = [[0.0f32; 4]; 8];
        let pl_count = params.point_lights.len().min(8);
        for (i, &light) in params.point_lights.iter().take(8).enumerate() {
            pl[i] = light;
        }

        let rb_count = params
            .rigid_bodies
            .len()
            .min(arcadia_physics::MAX_RIGID_BODIES);
        if rb_count > 0 {
            self.context.queue.write_buffer(
                &self.gpu_pools.rigid_bodies_buf,
                0,
                bytemuck::cast_slice(&params.rigid_bodies[..rb_count]),
            );
        }

        CameraUniforms {
            cam_pos_time: [eye.x, eye.y, eye.z, params.elapsed_time],
            cam_forward_fov: [forward.x, forward.y, forward.z, tan_half_fov],
            cam_right_aspect: [right.x, right.y, right.z, aspect],
            cam_up_frame: [up.x, up.y, up.z, f32::from_bits(self.frame_index as u32)],
            prev_view_proj: prev_view_proj.to_cols_array_2d(),
            curr_view_proj: curr_view_proj.to_cols_array_2d(),
            world_grid_dims: [
                self.world_grid_dims.x,
                self.world_grid_dims.y,
                self.world_grid_dims.z,
                (self.render_width & 0xFFFF) | ((self.render_height & 0xFFFF) << 16),
            ],
            sun_dir_intensity: [
                params.sun_dir.x,
                params.sun_dir.y,
                params.sun_dir.z,
                params.sun_intensity,
            ],
            jitter_brush_dt: [jx, jy, params.brush_radius, params.dt.clamp(0.001, 0.05)],
            target_voxel_info: target_info,
            point_light_count: [
                pl_count as u32,
                (if params.is_underwater { 1u32 } else { 0u32 }) | ((rb_count as u32) << 16),
                params.quality.ambient_floor.to_bits(),
                params.quality.contact_ao_dist.to_bits(),
            ],
            point_light_0: pl[0],
            point_light_1: pl[1],
            point_light_2: pl[2],
            point_light_3: pl[3],
            point_light_4: pl[4],
            point_light_5: pl[5],
            point_light_6: pl[6],
            point_light_7: pl[7],
            ray_steps: [
                params.quality.primary_max_steps,
                params.quality.sec_max_steps,
                params.quality.gi_max_steps,
                params.quality.shadow_max_steps,
            ],
            ray_distances: [
                params.quality.water_refr_dist,
                params.quality.mirror_refl_dist,
                params.quality.gi_max_dist,
                params.quality.shadow_max_dist,
            ],
            lod_settings: [
                params.quality.primary_lod_dist,
                params.quality.sec_lod_dist,
                params.quality.gi_lod_dist,
                params.quality.shadow_jitter,
            ],
            quality_flags: [
                params.quality.primary_lod_step,
                params.quality.sec_lod_step,
                params.quality.shadow_mode,
                params.quality.denoiser_taps
                    | (params.quality.mirror_adaptive << 8)
                    | (params.quality.gi_lod_step << 16)
                    | ((params.quality.dither_mode & 0x7) << 24)
                    | ((params.quality.tonemapper_mode & 0x1) << 27),
            ],
            prev_cam_pos: [
                prev_eye_for_uniform.x,
                prev_eye_for_uniform.y,
                prev_eye_for_uniform.z,
                0.0,
            ],
        }
    }

    fn encode_render_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view_target: &wgpu::TextureView,
        ping_pong: usize,
        workgroups_x: u32,
        workgroups_y: u32,
        sharc_tile_size: u32,
    ) {
        pipeline::record_debris_sim_pass(
            encoder,
            &self.pipelines.debris,
            &self.gpu_pools.group0_bind_group,
            &self.screen.pass1_bind_group,
        );

        pipeline::record_primary_raymarch_pass(
            encoder,
            &self.pipelines.primary,
            &self.gpu_pools.group0_bind_group,
            &self.screen.pass1_bind_group,
            workgroups_x,
            workgroups_y,
        );

        if self.active_debris > 0 {
            pipeline::record_debris_splat_pass(
                encoder,
                &self.pipelines.splat,
                &self.gpu_pools.group0_bind_group,
                &self.screen.pass1_bind_group,
            );
        }

        // WHY: SHARC update must see this frame's G-Buffer and finish resolving before Pass 2 queries the cache.
        sharc::record_sharc_update_pass(
            encoder,
            &self.pipelines.sharc_update,
            &self.gpu_pools.group0_bind_group,
            &self.sharc,
            sharc::sharc_update_workgroups(self.render_width, self.render_height, sharc_tile_size),
        );

        sharc::record_sharc_resolve_pass(
            encoder,
            &self.pipelines.sharc_resolve,
            &self.sharc.resolve_bind_group,
        );

        // WHY: Pass 2A (Temporal RIS) writes candidates into scratch reservoir_b_buf and di_reservoir_b_buf.
        pipeline::record_lighting_temporal_pass(
            encoder,
            &self.pipelines.lighting_temporal,
            &self.gpu_pools.group0_bind_group,
            &self.screen.pass2_bind_groups[0],
            &self.sharc.read_bind_group,
            &self.di.read_bind_group,
            workgroups_x,
            workgroups_y,
        );

        // WHY: Pass 2B (Spatial Reuse) reads current-frame scratch buffers and writes final candidates to reservoir_a_buf and di_reservoir_a_buf.
        pipeline::record_lighting_spatial_pass(
            encoder,
            &self.pipelines.lighting_spatial,
            &self.gpu_pools.group0_bind_group,
            &self.screen.pass2_bind_groups[1],
            &self.sharc.read_bind_group,
            &self.di.read_bind_group,
            workgroups_x,
            workgroups_y,
        );

        // WHY: Pass 3 (Denoiser) permanently reads final candidates from reservoir_a_buf and di_reservoir_a_buf.
        pipeline::record_denoiser_pass(
            encoder,
            &self.pipelines.denoiser,
            &self.gpu_pools.group0_bind_group,
            &self.screen.pass3_bind_group,
            &self.di.pass3_bind_group,
            workgroups_x,
            workgroups_y,
        );

        pipeline::record_composite_dlaa_pass(
            encoder,
            &self.pipelines.composite,
            &self.gpu_pools.group0_bind_group,
            &self.screen.pass4_bind_groups[ping_pong],
            workgroups_x,
            workgroups_y,
        );

        pipeline::record_blit_pass(
            encoder,
            &self.pipelines.blit,
            &self.screen.blit_bind_group,
            &self.gpu_pools.group0_bind_group,
            view_target,
        );
    }
}

// WHY: Breaks large uploads into 4MB chunks and flushes every 16MB to avoid staging buffer OOM.
fn write_buffer_chunked(
    _device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: u64,
    data: &[u8],
    staged_bytes: &mut u64,
) {
    if data.is_empty() {
        return;
    }

    const CHUNK_SIZE: usize = 4 * 1024 * 1024;
    const MAX_STAGED_THRESHOLD: u64 = 16 * 1024 * 1024;

    let mut current_offset = offset;
    for chunk in data.chunks(CHUNK_SIZE) {
        let chunk_len = chunk.len() as u64;
        if *staged_bytes + chunk_len > MAX_STAGED_THRESHOLD {
            // GPU-SYNC: Flush pending staging buffers to prevent wgpu-core staging belt exhaustion.
            queue.submit(std::iter::empty());
            *staged_bytes = 0;
        }

        queue.write_buffer(buffer, current_offset, chunk);
        *staged_bytes += chunk_len;
        current_offset += chunk_len;
    }
}
