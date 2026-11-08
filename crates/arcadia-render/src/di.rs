//! ReSTIR DI: GPU buffers and bind groups for emissive sources, alias table, and parameters.
//! Owns GPU resources for emissive sources and Vose alias table for Pass 2 and Pass 3.

use bytemuck::{Pod, Zeroable};

/// Size of `sources_buf`: 16,384 `EmissiveSource` slots (512 KB).
pub const SOURCES_BUF_BYTES: u64 = (arcadia_world::MAX_EMISSIVE_SOURCES
    * std::mem::size_of::<arcadia_world::EmissiveSource>())
    as u64;

/// Size of `alias_buf`: 16,384 `AliasEntry` slots (128 KB).
pub const ALIAS_BUF_BYTES: u64 =
    (arcadia_world::MAX_EMISSIVE_SOURCES * std::mem::size_of::<arcadia_world::AliasEntry>()) as u64;

/// Size of `params_buf`: one `EmissiveParams` uniform (16 bytes).
pub const PARAMS_BUF_BYTES: u64 = std::mem::size_of::<EmissiveParams>() as u64;

/// Host uniform mirror for `u_emissive` in DI shaders (16 bytes, 16-byte aligned).
// GPU-SYNC: Mirrors `EmissiveParams` in `shaders/common/di.wgsl` (16 bytes, 16-byte aligned).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct EmissiveParams {
    pub count: u32,
    pub generation: u32,
    pub total_flux: f32,
    pub pad: u32,
}

/// Bind group layouts for Pass 2 (lighting_restir_gi) and Pass 3 (denoiser_reconstruct).
pub struct DiLayouts {
    /// `@group(3)` for Pass 2 (`lighting_restir_gi.wgsl`): read_only sources, alias + uniform params.
    pub read_bgl: wgpu::BindGroupLayout,
    /// `@group(2)` for Pass 3 (`denoiser_reconstruct.wgsl`): read_only sources + uniform params.
    pub pass3_bgl: wgpu::BindGroupLayout,
}

/// Emissive light pool buffers with their bind groups.
pub struct DiResources {
    pub sources_buf: wgpu::Buffer,
    pub alias_buf: wgpu::Buffer,
    pub params_buf: wgpu::Buffer,
    pub read_bind_group: wgpu::BindGroup,
    pub pass3_bind_group: wgpu::BindGroup,
    pub generation: u32,
}

impl DiResources {
    /// Allocates initial GPU buffers and builds all DI bind groups.
    pub fn new(device: &wgpu::Device, layouts: &DiLayouts) -> Self {
        let sources_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("di_emissive_sources_buf"),
            size: SOURCES_BUF_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let alias_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("di_emissive_alias_buf"),
            size: ALIAS_BUF_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("di_emissive_params_buf"),
            size: PARAMS_BUF_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let read_bind_group = create_read_bind_group(
            device,
            &layouts.read_bgl,
            &sources_buf,
            &alias_buf,
            &params_buf,
        );

        let pass3_bind_group =
            create_pass3_bind_group(device, &layouts.pass3_bgl, &sources_buf, &params_buf);

        Self {
            sources_buf,
            alias_buf,
            params_buf,
            read_bind_group,
            pass3_bind_group,
            generation: 0,
        }
    }

    /// Uploads CPU emissive sources and Vose alias table to the GPU when `emissive.generation()` changes.
    pub fn update_buffers(
        &mut self,
        queue: &wgpu::Queue,
        emissive: &arcadia_world::EmissiveRegistry,
    ) {
        let gen = emissive.generation();
        if gen == self.generation && self.generation > 0 {
            return;
        }

        let sources = emissive.sources();
        let alias = emissive.alias();
        let count = sources.len().min(arcadia_world::MAX_EMISSIVE_SOURCES) as u32;

        if count > 0 {
            queue.write_buffer(
                &self.sources_buf,
                0,
                bytemuck::cast_slice(&sources[..count as usize]),
            );
            queue.write_buffer(
                &self.alias_buf,
                0,
                bytemuck::cast_slice(&alias[..count as usize]),
            );
        }

        let params = EmissiveParams {
            count,
            generation: gen,
            total_flux: emissive.total_flux(),
            pad: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        self.generation = gen;
    }
}

/// Creates the bind group layouts required for ReSTIR DI.
pub fn create_di_layouts(device: &wgpu::Device) -> DiLayouts {
    let read_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("di_read_bgl"),
        entries: &[
            // GPU-SYNC: Mirrors `emissive_sources` in `shaders/lighting_restir_gi.wgsl` (@group(3) @binding(0)).
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
            // GPU-SYNC: Mirrors `emissive_alias` in `shaders/lighting_restir_gi.wgsl` (@group(3) @binding(1)).
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
            // GPU-SYNC: Mirrors `u_emissive` in `shaders/lighting_restir_gi.wgsl` (@group(3) @binding(2)).
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pass3_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("di_pass3_bgl"),
        entries: &[
            // GPU-SYNC: Mirrors `emissive_sources` in `shaders/denoiser_reconstruct.wgsl` (@group(2) @binding(0)).
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
            // GPU-SYNC: Mirrors `u_emissive` in `shaders/denoiser_reconstruct.wgsl` (@group(2) @binding(1)).
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    DiLayouts {
        read_bgl,
        pass3_bgl,
    }
}

fn create_read_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sources_buf: &wgpu::Buffer,
    alias_buf: &wgpu::Buffer,
    params_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("di_read_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sources_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: alias_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    })
}

fn create_pass3_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sources_buf: &wgpu::Buffer,
    params_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("di_pass3_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sources_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emissive_params_size() {
        assert_eq!(std::mem::size_of::<EmissiveParams>(), 16);
        assert_eq!(std::mem::align_of::<EmissiveParams>(), 16);
    }
}
