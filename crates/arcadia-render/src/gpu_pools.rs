//! `Group0Resources`: Global storage buffer pools and `@group(0)` engine bind groups.
//! Owns GPU memory allocations for Chunk Roots, Node Pool, Brick Pool, Debris, and Rigid Bodies.

use crate::camera::CameraUniforms;
use arcadia_world::WorldMap;

/// Engine-wide storage buffers, uniform buffer, and `@group(0)` bind group shared by passes 1-4.
pub struct Group0Resources {
    pub group0_bgl: wgpu::BindGroupLayout,
    pub group0_bind_group: wgpu::BindGroup,
    pub camera_uniform_buf: wgpu::Buffer,
    pub chunk_roots_buf: wgpu::Buffer,
    pub node_pool_buf: wgpu::Buffer,
    pub brick_pool_buf: wgpu::Buffer,
    pub material_palette_buf: wgpu::Buffer,
    pub debris_pool_buf: wgpu::Buffer,
    pub rigid_bodies_buf: wgpu::Buffer,
}

impl Group0Resources {
    /// Recreates `group0_bind_group` pointing to updated/grown buffer resources.
    pub fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        self.group0_bind_group = create_group0_bind_group(
            device,
            &self.group0_bgl,
            &self.camera_uniform_buf,
            &self.chunk_roots_buf,
            &self.node_pool_buf,
            &self.brick_pool_buf,
            &self.material_palette_buf,
            &self.debris_pool_buf,
            &self.rigid_bodies_buf,
        );
    }
}

/// Creates the `@group(0)` bind group layout shared across all primary and lighting compute passes.
pub fn create_group0_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("group0_bgl"),
        entries: &[
            // GPU-SYNC: Mirrors `CameraUniforms` in `shaders/common/types.wgsl` (@group(0) @binding(0)).
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // GPU-SYNC: Mirrors `chunk_roots` in `shaders/common/types.wgsl` (@group(0) @binding(1)).
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
            // GPU-SYNC: Mirrors `node_pool` in `shaders/common/types.wgsl` (@group(0) @binding(2)).
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
            // GPU-SYNC: Mirrors `brick_pool` in `shaders/common/types.wgsl` (@group(0) @binding(3)).
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
            // GPU-SYNC: Mirrors `material_palette` in `shaders/common/types.wgsl` (@group(0) @binding(4)).
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
            // GPU-SYNC: Mirrors `debris_pool` in `shaders/common/types.wgsl` (@group(0) @binding(5)).
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
            // GPU-SYNC: Mirrors `rigid_bodies` in `shaders/common/types.wgsl` (@group(0) @binding(6)).
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

/// Allocates initial GPU storage buffers from `world` and constructs `Group0Resources`.
pub fn create_group0_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    world: &WorldMap,
    max_storage_bytes: u64,
) -> Group0Resources {
    let group0_bgl = create_group0_bgl(device);

    let camera_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("camera_uniform_buf"),
        size: std::mem::size_of::<CameraUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let roots_size_bytes =
        ((world.gpu_chunk_roots.len() * std::mem::size_of::<u32>()).max(256)) as u64;
    let chunk_roots_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chunk_roots_buf"),
        size: roots_size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut staged_bytes = 0u64;
    write_buffer_chunked(
        device,
        queue,
        &chunk_roots_buf,
        0,
        bytemuck::cast_slice(&world.gpu_chunk_roots),
        &mut staged_bytes,
    );

    let node_cap = world.gpu_nodes.len() + 2_000_000;
    let node_buf_bytes =
        ((node_cap * std::mem::size_of::<arcadia_core::Node64>()) as u64).min(max_storage_bytes);
    let node_pool_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("node_pool_buf"),
        size: node_buf_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    write_buffer_chunked(
        device,
        queue,
        &node_pool_buf,
        0,
        bytemuck::cast_slice(&world.gpu_nodes),
        &mut staged_bytes,
    );

    let brick_cap = world.gpu_bricks.len() + 1_000_000;
    let brick_buf_bytes = ((brick_cap * std::mem::size_of::<arcadia_core::LeafBrick64>()) as u64)
        .min(max_storage_bytes);
    let brick_pool_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brick_pool_buf"),
        size: brick_buf_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    write_buffer_chunked(
        device,
        queue,
        &brick_pool_buf,
        0,
        bytemuck::cast_slice(&world.gpu_bricks),
        &mut staged_bytes,
    );

    let palette_size = (world.palette.materials.len()
        * std::mem::size_of::<arcadia_core::Material>())
    .max(256) as u64;
    let material_palette_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("material_palette_buf"),
        size: palette_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    write_buffer_chunked(
        device,
        queue,
        &material_palette_buf,
        0,
        bytemuck::cast_slice(&world.palette.materials),
        &mut staged_bytes,
    );
    if staged_bytes > 0 {
        // GPU-SYNC: Flush initial startup writes to the GPU queue.
        queue.submit(std::iter::empty());
    }

    let debris_size = (arcadia_world::MAX_DEBRIS_PARTICLES
        * std::mem::size_of::<arcadia_world::DebrisParticle>()) as u64;
    let debris_pool_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debris_pool_buf"),
        size: debris_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let rigid_bodies_size = (arcadia_physics::MAX_RIGID_BODIES
        * std::mem::size_of::<arcadia_physics::GpuRigidBodyInstance>())
        as u64;
    let rigid_bodies_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rigid_bodies_buf"),
        size: rigid_bodies_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let group0_bind_group = create_group0_bind_group(
        device,
        &group0_bgl,
        &camera_uniform_buf,
        &chunk_roots_buf,
        &node_pool_buf,
        &brick_pool_buf,
        &material_palette_buf,
        &debris_pool_buf,
        &rigid_bodies_buf,
    );

    Group0Resources {
        group0_bgl,
        camera_uniform_buf,
        chunk_roots_buf,
        node_pool_buf,
        brick_pool_buf,
        material_palette_buf,
        debris_pool_buf,
        rigid_bodies_buf,
        group0_bind_group,
    }
}

#[allow(clippy::too_many_arguments)]
fn create_group0_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera_uniform_buf: &wgpu::Buffer,
    chunk_roots_buf: &wgpu::Buffer,
    node_pool_buf: &wgpu::Buffer,
    brick_pool_buf: &wgpu::Buffer,
    material_palette_buf: &wgpu::Buffer,
    debris_pool_buf: &wgpu::Buffer,
    rigid_bodies_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("group0_bind_group"),
        layout,
        entries: &[
            // GPU-SYNC: Mirrors `CameraUniforms` in `shaders/common/types.wgsl` (@group(0) @binding(0)).
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `chunk_roots` in `shaders/common/types.wgsl` (@group(0) @binding(1)).
            wgpu::BindGroupEntry {
                binding: 1,
                resource: chunk_roots_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `node_pool` in `shaders/common/types.wgsl` (@group(0) @binding(2)).
            wgpu::BindGroupEntry {
                binding: 2,
                resource: node_pool_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `brick_pool` in `shaders/common/types.wgsl` (@group(0) @binding(3)).
            wgpu::BindGroupEntry {
                binding: 3,
                resource: brick_pool_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `material_palette` in `shaders/common/types.wgsl` (@group(0) @binding(4)).
            wgpu::BindGroupEntry {
                binding: 4,
                resource: material_palette_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `debris_pool` in `shaders/common/types.wgsl` (@group(0) @binding(5)).
            wgpu::BindGroupEntry {
                binding: 5,
                resource: debris_pool_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `rigid_bodies` in `shaders/common/types.wgsl` (@group(0) @binding(6)).
            wgpu::BindGroupEntry {
                binding: 6,
                resource: rigid_bodies_buf.as_entire_binding(),
            },
        ],
    })
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
