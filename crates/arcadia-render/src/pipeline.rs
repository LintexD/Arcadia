//! `Pipelines` and `PassLayouts`: Compute & Render pipeline compilation and dispatch orchestration.
//! Owns GPU compute and render pipeline objects, shader modules, and pass layouts.

use crate::di::{create_di_layouts, DiLayouts};
use crate::sharc::{
    create_sharc_layouts, create_sharc_resolve_pipeline, create_sharc_update_pipeline, SharcLayouts,
};

pub const BLIT_WGSL: &str = include_str!("../../../shaders/blit.wgsl");

pub const SHADER1_SRC: &str = concat!(
    include_str!("../../../shaders/common/types.wgsl"),
    "\n",
    include_str!("../../../shaders/common/traversal64.wgsl"),
    "\n",
    include_str!("../../../shaders/primary_gbuffer.wgsl")
);

pub const SHADER2_SRC: &str = concat!(
    include_str!("../../../shaders/common/types.wgsl"),
    "\n",
    include_str!("../../../shaders/common/traversal64.wgsl"),
    "\n",
    include_str!("../../../shaders/common/sharc.wgsl"),
    "\n",
    include_str!("../../../shaders/common/sharc_query.wgsl"),
    "\n",
    include_str!("../../../shaders/common/di.wgsl"),
    "\n",
    include_str!("../../../shaders/lighting_restir_gi.wgsl")
);

pub const SHADER3_SRC: &str = concat!(
    include_str!("../../../shaders/common/types.wgsl"),
    "\n",
    include_str!("../../../shaders/common/traversal64.wgsl"),
    "\n",
    include_str!("../../../shaders/common/di.wgsl"),
    "\n",
    include_str!("../../../shaders/denoiser_reconstruct.wgsl")
);

pub const SHADER4_SRC: &str = concat!(
    include_str!("../../../shaders/common/types.wgsl"),
    "\n",
    include_str!("../../../shaders/common/traversal64.wgsl"),
    "\n",
    include_str!("../../../shaders/composite_dlaa.wgsl")
);

/// Pre-created bind group layouts for compute passes 1-4 and fullscreen blit.
pub struct PassLayouts {
    pub pass1_bgl: wgpu::BindGroupLayout,
    pub pass2_bgl: wgpu::BindGroupLayout,
    pub pass3_bgl: wgpu::BindGroupLayout,
    pub pass4_bgl: wgpu::BindGroupLayout,
    pub blit_bgl: wgpu::BindGroupLayout,
    pub blit_sampler: wgpu::Sampler,
    /// SHARC cache layouts, including the read-only `@group(2)` consumed by Pass 2.
    pub sharc: SharcLayouts,
    /// DI layouts, including Pass 2 read and Pass 3 read.
    pub di: DiLayouts,
}

/// Compiled compute and render pipelines across all rendering stages.
pub struct Pipelines {
    pub debris: wgpu::ComputePipeline,
    pub primary: wgpu::ComputePipeline,
    pub splat: wgpu::ComputePipeline,
    pub lighting_temporal: wgpu::ComputePipeline,
    pub lighting_spatial: wgpu::ComputePipeline,
    pub denoiser: wgpu::ComputePipeline,
    pub composite: wgpu::ComputePipeline,
    pub blit: wgpu::RenderPipeline,
    pub sharc_update: wgpu::ComputePipeline,
    pub sharc_resolve: wgpu::ComputePipeline,
}

/// Creates all bind group layouts and the presentation sampler for passes 1-5.
pub fn create_pass_layouts(device: &wgpu::Device) -> PassLayouts {
    PassLayouts {
        pass1_bgl: create_pass1_bgl(device),
        pass2_bgl: create_pass2_bgl(device),
        pass3_bgl: create_pass3_bgl(device),
        pass4_bgl: create_pass4_bgl(device),
        blit_bgl: create_blit_bgl(device),
        blit_sampler: create_blit_sampler(device),
        sharc: create_sharc_layouts(device),
        di: create_di_layouts(device),
    }
}

/// Compiles all compute and render pipelines.
pub fn create_pipelines(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    group0_bgl: &wgpu::BindGroupLayout,
    pass_layouts: &PassLayouts,
) -> Pipelines {
    let (debris, primary, splat) =
        create_primary_gbuffer_pipelines(device, group0_bgl, &pass_layouts.pass1_bgl);
    let (lighting_temporal, lighting_spatial) = create_lighting_restir_pipelines(
        device,
        group0_bgl,
        &pass_layouts.pass2_bgl,
        &pass_layouts.sharc.read_bgl,
        &pass_layouts.di.read_bgl,
    );
    let denoiser = create_denoiser_pipeline(
        device,
        group0_bgl,
        &pass_layouts.pass3_bgl,
        &pass_layouts.di.pass3_bgl,
    );
    let composite = create_composite_dlaa_pipeline(device, group0_bgl, &pass_layouts.pass4_bgl);
    let blit = create_blit_pipeline(device, surface_format, group0_bgl, &pass_layouts.blit_bgl);
    let sharc_update = create_sharc_update_pipeline(device, group0_bgl, &pass_layouts.sharc);
    let sharc_resolve = create_sharc_resolve_pipeline(device, &pass_layouts.sharc);

    Pipelines {
        debris,
        primary,
        splat,
        lighting_temporal,
        lighting_spatial,
        denoiser,
        composite,
        blit,
        sharc_update,
        sharc_resolve,
    }
}

pub fn record_debris_sim_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group0: &wgpu::BindGroup,
    pass1: &wgpu::BindGroup,
) {
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("debris_update_pass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, group0, &[]);
    cpass.set_bind_group(1, pass1, &[]);
    cpass.dispatch_workgroups(256, 1, 1);
}

pub fn record_primary_raymarch_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group0: &wgpu::BindGroup,
    pass1: &wgpu::BindGroup,
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("primary_raymarch_pass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, group0, &[]);
    cpass.set_bind_group(1, pass1, &[]);
    cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

pub fn record_debris_splat_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group0: &wgpu::BindGroup,
    pass1: &wgpu::BindGroup,
) {
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("splat_debris_pass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, group0, &[]);
    cpass.set_bind_group(1, pass1, &[]);
    cpass.dispatch_workgroups(256, 1, 1);
}

#[allow(clippy::too_many_arguments)]
pub fn record_lighting_temporal_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group0: &wgpu::BindGroup,
    pass2: &wgpu::BindGroup,
    sharc_read: &wgpu::BindGroup,
    di_read: &wgpu::BindGroup,
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("lighting_temporal_pass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, group0, &[]);
    cpass.set_bind_group(1, pass2, &[]);
    cpass.set_bind_group(2, sharc_read, &[]);
    cpass.set_bind_group(3, di_read, &[]);
    cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

#[allow(clippy::too_many_arguments)]
pub fn record_lighting_spatial_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group0: &wgpu::BindGroup,
    pass2: &wgpu::BindGroup,
    sharc_read: &wgpu::BindGroup,
    di_read: &wgpu::BindGroup,
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("lighting_spatial_pass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, group0, &[]);
    cpass.set_bind_group(1, pass2, &[]);
    cpass.set_bind_group(2, sharc_read, &[]);
    cpass.set_bind_group(3, di_read, &[]);
    cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

pub fn record_denoiser_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group0: &wgpu::BindGroup,
    pass3: &wgpu::BindGroup,
    di_pass3: &wgpu::BindGroup,
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("denoiser_pass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, group0, &[]);
    cpass.set_bind_group(1, pass3, &[]);
    cpass.set_bind_group(2, di_pass3, &[]);
    cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

pub fn record_composite_dlaa_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group0: &wgpu::BindGroup,
    pass4: &wgpu::BindGroup,
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("composite_dlaa_pass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, group0, &[]);
    cpass.set_bind_group(1, pass4, &[]);
    cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

pub fn record_blit_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    blit_bg: &wgpu::BindGroup,
    group0_bg: &wgpu::BindGroup,
    view_target: &wgpu::TextureView,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("blit_render_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: view_target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    rpass.set_pipeline(pipeline);
    rpass.set_bind_group(0, blit_bg, &[]);
    rpass.set_bind_group(1, group0_bg, &[]);
    rpass.draw(0..3, 0..1);
}

/// Creates a standard Reverse-Z `wgpu::DepthStencilState` (`near = 1.0, far = 0.0`).
///
/// Configures `Depth32Float` format with `CompareFunction::GreaterEqual`, matching
/// the infinite reverse perspective projection matrix and eliminating distant Z-fighting.
pub fn create_reverse_z_depth_stencil_state(depth_write_enabled: bool) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled,
        depth_compare: wgpu::CompareFunction::GreaterEqual,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Helper creating the Reverse-Z depth attachment descriptor with 0.0 far-plane clear.
pub fn create_reverse_z_depth_attachment<'a>(
    view: &'a wgpu::TextureView,
    clear: bool,
) -> wgpu::RenderPassDepthStencilAttachment<'a> {
    wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(wgpu::Operations {
            load: if clear {
                wgpu::LoadOp::Clear(0.0) // Far plane is 0.0 in Reverse-Z
            } else {
                wgpu::LoadOp::Load
            },
            store: wgpu::StoreOp::Store,
        }),
        stencil_ops: None,
    }
}

// Private Bind Group Layout & Pipeline Descriptor Builders

fn create_pass1_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pass1_bgl"),
        entries: &[
            // GPU-SYNC: Mirrors `gbuf_depth_mat` in `shaders/primary_gbuffer.wgsl` (@group(1) @binding(0)).
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `gbuf_norm_motion` in `shaders/primary_gbuffer.wgsl` (@group(1) @binding(1)).
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `gbuf_albedo_ao` in `shaders/primary_gbuffer.wgsl` (@group(1) @binding(2)).
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_pass2_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pass2_bgl"),
        entries: &[
            // GPU-SYNC: Mirrors `gbuf_depth_mat` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(0)).
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `gbuf_norm_motion` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(1)).
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `reservoirs_in` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(3)).
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `reservoirs_out` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(4)).
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `di_reservoir_in` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(5)).
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `di_reservoir_out` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(6)).
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_pass3_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pass3_bgl"),
        entries: &[
            // GPU-SYNC: Mirrors `gbuf_depth_mat` in `shaders/denoiser_reconstruct.wgsl` (@group(1) @binding(0)).
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `gbuf_norm_motion` in `shaders/denoiser_reconstruct.wgsl` (@group(1) @binding(1)).
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `gbuf_albedo_ao` in `shaders/denoiser_reconstruct.wgsl` (@group(1) @binding(2)).
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `reservoirs_in` in `shaders/denoiser_reconstruct.wgsl` (@group(1) @binding(4)).
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `lit_color_out` in `shaders/denoiser_reconstruct.wgsl` (@group(1) @binding(5)).
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `di_reservoir_curr` in `shaders/denoiser_reconstruct.wgsl` (@group(1) @binding(6)).
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_pass4_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pass4_bgl"),
        entries: &[
            // GPU-SYNC: Mirrors `gbuf_norm_motion` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(2)).
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `lit_color_in` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(5)).
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `history_in` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(6)).
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `history_out` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(7)).
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `out_tex` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(8)).
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    })
}

fn create_blit_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blit_bgl"),
        entries: &[
            // GPU-SYNC: Mirrors `t_diffuse` in `shaders/blit.wgsl` (@group(0) @binding(0)).
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `s_diffuse` in `shaders/blit.wgsl` (@group(0) @binding(1)).
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn create_blit_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("blit_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    })
}

fn create_primary_gbuffer_pipelines(
    device: &wgpu::Device,
    group0_bgl: &wgpu::BindGroupLayout,
    pass1_bgl: &wgpu::BindGroupLayout,
) -> (
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
) {
    let shader1 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("primary_gbuffer_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER1_SRC)),
    });
    let pass1_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pass1_layout"),
        bind_group_layouts: &[group0_bgl, pass1_bgl],
        push_constant_ranges: &[],
    });

    let pipeline_debris = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_debris"),
        layout: Some(&pass1_layout),
        module: &shader1,
        entry_point: Some("update_debris"),
        compilation_options: Default::default(),
        cache: None,
    });

    let pipeline_primary = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_primary"),
        layout: Some(&pass1_layout),
        module: &shader1,
        entry_point: Some("primary_raymarch"),
        compilation_options: Default::default(),
        cache: None,
    });

    let pipeline_splat = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_splat"),
        layout: Some(&pass1_layout),
        module: &shader1,
        entry_point: Some("splat_debris"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline_debris, pipeline_primary, pipeline_splat)
}

fn create_lighting_restir_pipelines(
    device: &wgpu::Device,
    group0_bgl: &wgpu::BindGroupLayout,
    pass2_bgl: &wgpu::BindGroupLayout,
    sharc_read_bgl: &wgpu::BindGroupLayout,
    di_read_bgl: &wgpu::BindGroupLayout,
) -> (wgpu::ComputePipeline, wgpu::ComputePipeline) {
    let shader2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lighting_restir_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER2_SRC)),
    });
    let pass2_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pass2_layout"),
        bind_group_layouts: &[group0_bgl, pass2_bgl, sharc_read_bgl, di_read_bgl],
        push_constant_ranges: &[],
    });

    let pipeline_temporal = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_lighting_temporal"),
        layout: Some(&pass2_layout),
        module: &shader2,
        entry_point: Some("temporal_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let pipeline_spatial = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_lighting_spatial"),
        layout: Some(&pass2_layout),
        module: &shader2,
        entry_point: Some("spatial_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    (pipeline_temporal, pipeline_spatial)
}

fn create_denoiser_pipeline(
    device: &wgpu::Device,
    group0_bgl: &wgpu::BindGroupLayout,
    pass3_bgl: &wgpu::BindGroupLayout,
    di_pass3_bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader3 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("denoiser_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER3_SRC)),
    });
    let pass3_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pass3_layout"),
        bind_group_layouts: &[group0_bgl, pass3_bgl, di_pass3_bgl],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_denoiser"),
        layout: Some(&pass3_layout),
        module: &shader3,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn create_composite_dlaa_pipeline(
    device: &wgpu::Device,
    group0_bgl: &wgpu::BindGroupLayout,
    pass4_bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader4 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("composite_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER4_SRC)),
    });
    let pass4_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pass4_layout"),
        bind_group_layouts: &[group0_bgl, pass4_bgl],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_composite"),
        layout: Some(&pass4_layout),
        module: &shader4,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn create_blit_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    group0_bgl: &wgpu::BindGroupLayout,
    blit_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader_blit = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blit_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BLIT_WGSL)),
    });
    let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("blit_layout"),
        bind_group_layouts: &[blit_bgl, group0_bgl],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pipeline_blit"),
        layout: Some(&blit_layout),
        vertex: wgpu::VertexState {
            module: &shader_blit,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader_blit,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview: None,
        cache: None,
    })
}
