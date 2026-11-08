//! `ScreenResources`: G-Buffer, Temporal History textures, and Ping-Pong Bind Groups.
//! Owns screen-resolution-dependent GPU textures, storage buffers, and pass 1-5 bind groups.

use bytemuck::{Pod, Zeroable};

/// GPU-shared ReSTIR GI reservoir buffer mirror (32 bytes, 16-byte aligned).
// GPU-SYNC: Mirrors `GiReservoir` in `shaders/common/types.wgsl` (32 bytes, 16-byte aligned).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GiReservoir {
    pub sample_pos: [f32; 3],
    pub norm_m_flags: u32,
    pub lo_rg: u32,
    pub lo_b_w: u32,
    pub primary_depth: f32,
    pub shadow_vis_m: u32,
}

/// GPU-shared ReSTIR DI reservoir buffer mirror (16 bytes, 16-byte aligned).
// GPU-SYNC: Mirrors `DiReservoir` in `shaders/common/di.wgsl` (16 bytes, 16-byte aligned).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct DiReservoir {
    pub light_idx_seed: u32,
    pub w_sum: f32,
    pub m: f32,
    pub w: f32,
}

/// Screen-size-dependent GPU storage buffers, textures, and bind groups recreated on resize.
pub struct ScreenResources {
    pub gbuf_depth_mat_buf: wgpu::Buffer,
    pub gbuf_norm_motion_buf: wgpu::Buffer,
    pub gbuf_albedo_ao_buf: wgpu::Buffer,
    /// Hardware Reverse-Z depth texture (`wgpu::TextureFormat::Depth32Float`).
    pub gbuf_depth_tex: wgpu::Texture,
    /// Hardware Reverse-Z depth texture view.
    pub gbuf_depth_tex_view: wgpu::TextureView,
    pub reservoir_a_buf: wgpu::Buffer,
    pub reservoir_b_buf: wgpu::Buffer,
    pub di_reservoir_a_buf: wgpu::Buffer,
    pub di_reservoir_b_buf: wgpu::Buffer,
    pub lit_color_buf: wgpu::Buffer,
    pub history_a_buf: wgpu::Buffer,
    pub history_b_buf: wgpu::Buffer,
    pub out_tex: wgpu::Texture,
    pub out_tex_view: wgpu::TextureView,

    pub pass1_bind_group: wgpu::BindGroup,
    pub pass2_bind_groups: [wgpu::BindGroup; 2],
    pub pass3_bind_group: wgpu::BindGroup,
    pub pass4_bind_groups: [wgpu::BindGroup; 2],
    pub blit_bind_group: wgpu::BindGroup,
}

impl ScreenResources {
    // WHY: Allocates all screen-dependent textures, buffers, and bind groups for passes 1-5.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        pass1_bgl: &wgpu::BindGroupLayout,
        pass2_bgl: &wgpu::BindGroupLayout,
        pass3_bgl: &wgpu::BindGroupLayout,
        pass4_bgl: &wgpu::BindGroupLayout,
        blit_bgl: &wgpu::BindGroupLayout,
        blit_sampler: &wgpu::Sampler,
    ) -> Self {
        let buffers = create_gbuffer_buffers(device, width, height);
        let textures = create_screen_textures(device, width, height);

        let pass1_bind_group = create_pass1_bind_group(
            device,
            pass1_bgl,
            &buffers.gbuf_depth_mat_buf,
            &buffers.gbuf_norm_motion_buf,
            &buffers.gbuf_albedo_ao_buf,
        );

        let pass2_bind_groups = create_pass2_bind_groups(
            device,
            pass2_bgl,
            &buffers.gbuf_depth_mat_buf,
            &buffers.gbuf_norm_motion_buf,
            &buffers.reservoir_a_buf,
            &buffers.reservoir_b_buf,
            &buffers.di_reservoir_a_buf,
            &buffers.di_reservoir_b_buf,
        );

        let pass3_bind_group = create_pass3_bind_group(
            device,
            pass3_bgl,
            &buffers.gbuf_depth_mat_buf,
            &buffers.gbuf_norm_motion_buf,
            &buffers.gbuf_albedo_ao_buf,
            &buffers.reservoir_a_buf,
            &buffers.di_reservoir_a_buf,
            &buffers.lit_color_buf,
        );

        let pass4_bind_groups = create_pass4_bind_groups(
            device,
            pass4_bgl,
            &buffers.gbuf_norm_motion_buf,
            &buffers.lit_color_buf,
            &buffers.history_a_buf,
            &buffers.history_b_buf,
            &textures.out_tex_view,
        );

        let blit_bind_group =
            create_blit_bind_group(device, blit_bgl, &textures.out_tex_view, blit_sampler);

        Self {
            gbuf_depth_mat_buf: buffers.gbuf_depth_mat_buf,
            gbuf_norm_motion_buf: buffers.gbuf_norm_motion_buf,
            gbuf_albedo_ao_buf: buffers.gbuf_albedo_ao_buf,
            gbuf_depth_tex: textures.gbuf_depth_tex,
            gbuf_depth_tex_view: textures.gbuf_depth_tex_view,
            reservoir_a_buf: buffers.reservoir_a_buf,
            reservoir_b_buf: buffers.reservoir_b_buf,
            di_reservoir_a_buf: buffers.di_reservoir_a_buf,
            di_reservoir_b_buf: buffers.di_reservoir_b_buf,
            lit_color_buf: buffers.lit_color_buf,
            history_a_buf: buffers.history_a_buf,
            history_b_buf: buffers.history_b_buf,
            out_tex: textures.out_tex,
            out_tex_view: textures.out_tex_view,
            pass1_bind_group,
            pass2_bind_groups,
            pass3_bind_group,
            pass4_bind_groups,
            blit_bind_group,
        }
    }
}

pub fn create_pass1_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    gbuf_depth_mat_buf: &wgpu::Buffer,
    gbuf_norm_motion_buf: &wgpu::Buffer,
    gbuf_albedo_ao_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pass1_bind_group"),
        layout,
        entries: &[
            // GPU-SYNC: Mirrors `gbuf_depth_mat` in `shaders/primary_gbuffer.wgsl` (@group(1) @binding(0)).
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gbuf_depth_mat_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `gbuf_norm_motion` in `shaders/primary_gbuffer.wgsl` (@group(1) @binding(1)).
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gbuf_norm_motion_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `gbuf_albedo_ao` in `shaders/primary_gbuffer.wgsl` (@group(1) @binding(2)).
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gbuf_albedo_ao_buf.as_entire_binding(),
            },
        ],
    })
}

#[allow(clippy::too_many_arguments)]
pub fn create_pass2_bind_groups(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    gbuf_depth_mat_buf: &wgpu::Buffer,
    gbuf_norm_motion_buf: &wgpu::Buffer,
    reservoir_a_buf: &wgpu::Buffer,
    reservoir_b_buf: &wgpu::Buffer,
    di_reservoir_a_buf: &wgpu::Buffer,
    di_reservoir_b_buf: &wgpu::Buffer,
) -> [wgpu::BindGroup; 2] {
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pass2_bind_group_ping"),
        layout,
        entries: &[
            // GPU-SYNC: Mirrors `gbuf_depth_mat` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(0)).
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gbuf_depth_mat_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `gbuf_norm_motion` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(1)).
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gbuf_norm_motion_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `reservoirs_in` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(3)).
            wgpu::BindGroupEntry {
                binding: 3,
                resource: reservoir_a_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `reservoirs_out` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(4)).
            wgpu::BindGroupEntry {
                binding: 4,
                resource: reservoir_b_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `di_reservoirs_in` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(5)).
            wgpu::BindGroupEntry {
                binding: 5,
                resource: di_reservoir_a_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `di_reservoirs_out` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(6)).
            wgpu::BindGroupEntry {
                binding: 6,
                resource: di_reservoir_b_buf.as_entire_binding(),
            },
        ],
    });

    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pass2_bind_group_pong"),
        layout,
        entries: &[
            // GPU-SYNC: Mirrors `gbuf_depth_mat` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(0)).
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gbuf_depth_mat_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `gbuf_norm_motion` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(1)).
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gbuf_norm_motion_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `reservoirs_in` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(3)).
            wgpu::BindGroupEntry {
                binding: 3,
                resource: reservoir_b_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `reservoirs_out` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(4)).
            wgpu::BindGroupEntry {
                binding: 4,
                resource: reservoir_a_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `di_reservoirs_in` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(5)).
            wgpu::BindGroupEntry {
                binding: 5,
                resource: di_reservoir_b_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `di_reservoirs_out` in `shaders/lighting_restir_gi.wgsl` (@group(1) @binding(6)).
            wgpu::BindGroupEntry {
                binding: 6,
                resource: di_reservoir_a_buf.as_entire_binding(),
            },
        ],
    });

    [bg0, bg1]
}

#[allow(clippy::too_many_arguments)]
pub fn create_pass3_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    gbuf_depth_mat_buf: &wgpu::Buffer,
    gbuf_norm_motion_buf: &wgpu::Buffer,
    gbuf_albedo_ao_buf: &wgpu::Buffer,
    reservoir_a_buf: &wgpu::Buffer,
    di_reservoir_a_buf: &wgpu::Buffer,
    lit_color_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pass3_bind_group"),
        layout,
        entries: &[
            // GPU-SYNC: Mirrors `gbuf_depth_mat` in `shaders/denoiser_reconstruct.wgsl` (@group(1) @binding(0)).
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gbuf_depth_mat_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `gbuf_norm_motion` in `shaders/denoiser_reconstruct.wgsl` (@group(1) @binding(1)).
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gbuf_norm_motion_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `gbuf_albedo_ao` in `shaders/denoiser_reconstruct.wgsl` (@group(1) @binding(2)).
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gbuf_albedo_ao_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `reservoirs_in` in `shaders/denoiser_reconstruct.wgsl` (@group(1) @binding(4)).
            wgpu::BindGroupEntry {
                binding: 4,
                resource: reservoir_a_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `lit_color_out` in `shaders/denoiser_reconstruct.wgsl` (@group(1) @binding(5)).
            wgpu::BindGroupEntry {
                binding: 5,
                resource: lit_color_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `di_reservoirs_in` in `shaders/denoiser_reconstruct.wgsl` (@group(1) @binding(6)).
            wgpu::BindGroupEntry {
                binding: 6,
                resource: di_reservoir_a_buf.as_entire_binding(),
            },
        ],
    })
}

pub fn create_pass4_bind_groups(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    gbuf_norm_motion_buf: &wgpu::Buffer,
    lit_color_buf: &wgpu::Buffer,
    history_a_buf: &wgpu::Buffer,
    history_b_buf: &wgpu::Buffer,
    out_tex_view: &wgpu::TextureView,
) -> [wgpu::BindGroup; 2] {
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pass4_bind_group_ping"),
        layout,
        entries: &[
            // GPU-SYNC: Mirrors `gbuf_norm_motion` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(2)).
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gbuf_norm_motion_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `lit_color_in` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(5)).
            wgpu::BindGroupEntry {
                binding: 5,
                resource: lit_color_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `history_in` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(6)).
            wgpu::BindGroupEntry {
                binding: 6,
                resource: history_a_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `history_out` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(7)).
            wgpu::BindGroupEntry {
                binding: 7,
                resource: history_b_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `out_tex` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(8)).
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(out_tex_view),
            },
        ],
    });

    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pass4_bind_group_pong"),
        layout,
        entries: &[
            // GPU-SYNC: Mirrors `gbuf_norm_motion` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(2)).
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gbuf_norm_motion_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `lit_color_in` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(5)).
            wgpu::BindGroupEntry {
                binding: 5,
                resource: lit_color_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `history_in` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(6)).
            wgpu::BindGroupEntry {
                binding: 6,
                resource: history_b_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `history_out` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(7)).
            wgpu::BindGroupEntry {
                binding: 7,
                resource: history_a_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `out_tex` in `shaders/composite_dlaa.wgsl` (@group(1) @binding(8)).
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(out_tex_view),
            },
        ],
    });

    [bg0, bg1]
}

pub fn create_blit_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    out_tex_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit_bind_group"),
        layout,
        entries: &[
            // GPU-SYNC: Mirrors `t_diffuse` in `shaders/blit.wgsl` (@group(0) @binding(0)).
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(out_tex_view),
            },
            // GPU-SYNC: Mirrors `s_diffuse` in `shaders/blit.wgsl` (@group(0) @binding(1)).
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

struct GbufferBuffers {
    gbuf_depth_mat_buf: wgpu::Buffer,
    gbuf_norm_motion_buf: wgpu::Buffer,
    gbuf_albedo_ao_buf: wgpu::Buffer,
    reservoir_a_buf: wgpu::Buffer,
    reservoir_b_buf: wgpu::Buffer,
    di_reservoir_a_buf: wgpu::Buffer,
    di_reservoir_b_buf: wgpu::Buffer,
    lit_color_buf: wgpu::Buffer,
    history_a_buf: wgpu::Buffer,
    history_b_buf: wgpu::Buffer,
}

fn create_gbuffer_buffers(device: &wgpu::Device, width: u32, height: u32) -> GbufferBuffers {
    let num_pixels = (width.max(1) * height.max(1)) as usize;
    let gbuf_depth_mat_size = ((num_pixels * 8).max(256)) as u64;
    let gbuf_norm_motion_size = ((num_pixels * 8).max(256)) as u64;
    let gbuf_albedo_ao_size = ((num_pixels * 4).max(256)) as u64;
    let reservoir_size = ((num_pixels * 32).max(256)) as u64;
    let di_reservoir_size = ((num_pixels * 16).max(256)) as u64;
    let color_history_size = ((num_pixels * 16).max(256)) as u64;

    let gbuf_depth_mat_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gbuf_depth_mat_buf"),
        size: gbuf_depth_mat_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let gbuf_norm_motion_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gbuf_norm_motion_buf"),
        size: gbuf_norm_motion_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let gbuf_albedo_ao_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gbuf_albedo_ao_buf"),
        size: gbuf_albedo_ao_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let reservoir_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reservoir_a_buf"),
        size: reservoir_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let reservoir_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reservoir_b_buf"),
        size: reservoir_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let di_reservoir_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("di_reservoir_a_buf"),
        size: di_reservoir_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let di_reservoir_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("di_reservoir_b_buf"),
        size: di_reservoir_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let lit_color_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lit_color_buf"),
        size: color_history_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let history_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("history_a_buf"),
        size: color_history_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let history_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("history_b_buf"),
        size: color_history_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    GbufferBuffers {
        gbuf_depth_mat_buf,
        gbuf_norm_motion_buf,
        gbuf_albedo_ao_buf,
        reservoir_a_buf,
        reservoir_b_buf,
        di_reservoir_a_buf,
        di_reservoir_b_buf,
        lit_color_buf,
        history_a_buf,
        history_b_buf,
    }
}

struct ScreenTextures {
    out_tex: wgpu::Texture,
    out_tex_view: wgpu::TextureView,
    gbuf_depth_tex: wgpu::Texture,
    gbuf_depth_tex_view: wgpu::TextureView,
}

fn create_screen_textures(device: &wgpu::Device, width: u32, height: u32) -> ScreenTextures {
    let out_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("out_tex"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let out_tex_view = out_tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("out_tex_view"),
        ..Default::default()
    });

    // WHY: Hardware Reverse-Z depth texture (D32_SFLOAT) for hybrid raster passes and Z-buffering.
    let gbuf_depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gbuf_depth_tex_d32"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let gbuf_depth_tex_view = gbuf_depth_tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("gbuf_depth_tex_view"),
        ..Default::default()
    });

    ScreenTextures {
        out_tex,
        out_tex_view,
        gbuf_depth_tex,
        gbuf_depth_tex_view,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gi_reservoir_layout() {
        assert_eq!(std::mem::size_of::<GiReservoir>(), 32);
        assert_eq!(std::mem::align_of::<GiReservoir>(), 16);
    }

    #[test]
    fn test_di_reservoir_layout() {
        assert_eq!(std::mem::size_of::<DiReservoir>(), 16);
        assert_eq!(std::mem::align_of::<DiReservoir>(), 16);
    }
}
