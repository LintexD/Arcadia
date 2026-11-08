//! SHARC voxel-surfel radiance cache: buffers, bind group layouts, compute pipelines, and pass recording.
//! Owns GPU resources only (4.5 MB of storage buffers); world-space keyed, so it survives screen resizes.

use bytemuck::{Pod, Zeroable};

/// Number of 4-way set-associative buckets (`shaders/common/sharc.wgsl` `SHARC_BUCKETS`).
pub const SHARC_BUCKETS: u32 = 32_768;
/// Slots per bucket; one bucket of resolved slots is a single 64-byte cache line.
pub const SHARC_WAYS: u32 = 4;
/// Total surfel slots (`shaders/common/sharc.wgsl` `SHARC_SLOTS`).
pub const SHARC_SLOTS: u32 = SHARC_BUCKETS * SHARC_WAYS;

/// Size of `sharc_keys`: one `atomic<u32>` key per slot (512 KB).
pub const SHARC_KEYS_BYTES: u64 = SHARC_SLOTS as u64 * 4;
/// Size of `sharc_accum`: `[r_fixed, g_fixed, b_fixed, sample_count]` per slot (2.0 MB).
pub const SHARC_ACCUM_BYTES: u64 = SHARC_SLOTS as u64 * 16;
/// Size of `sharc_resolved`: one `SharcResolvedSlot` per slot (2.0 MB).
pub const SHARC_RESOLVED_BYTES: u64 =
    SHARC_SLOTS as u64 * std::mem::size_of::<SharcResolvedSlot>() as u64;

/// Workgroup edge of `sharc_update.wgsl` (`@workgroup_size(8, 8, 1)`); one thread seeds one screen tile.
const UPDATE_WORKGROUP_EDGE: u32 = 8;
/// Workgroup width of `sharc_resolve.wgsl` (`@workgroup_size(256, 1, 1)`); one thread resolves one slot.
const RESOLVE_WORKGROUP_SIZE: u32 = 256;

pub const SHARC_UPDATE_SRC: &str = concat!(
    include_str!("../../../shaders/common/types.wgsl"),
    "\n",
    include_str!("../../../shaders/common/traversal64.wgsl"),
    "\n",
    include_str!("../../../shaders/common/sharc.wgsl"),
    "\n",
    include_str!("../../../shaders/common/sharc_query.wgsl"),
    "\n",
    include_str!("../../../shaders/sharc_update.wgsl")
);

pub const SHARC_RESOLVE_SRC: &str = concat!(
    include_str!("../../../shaders/common/sharc.wgsl"),
    "\n",
    include_str!("../../../shaders/sharc_resolve.wgsl")
);

/// Host mirror of one resolved surfel slot as read by Pass 2 (`query_sharc`).
// GPU-SYNC: Mirrors `SharcResolvedSlot` in `shaders/common/sharc.wgsl` (16 bytes, 4-byte aligned).
// `key == 0` marks an empty slot; `rad_b_meta` = f16 blue | (hist_frames << 16) | (stale_age << 24).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct SharcResolvedSlot {
    pub key: u32,
    pub rad_rg: u32,
    pub rad_b_meta: u32,
    pub pad: u32,
}

/// Bind group layouts shared by the SHARC passes and by Pass 2's read-only `@group(2)`.
pub struct SharcLayouts {
    /// `@group(1)` of `sharc_update.wgsl`: read-only `gbuf_depth_mat` and `gbuf_norm_motion`.
    pub gbuf_bgl: wgpu::BindGroupLayout,
    /// `@group(2)` of `sharc_update.wgsl`: `sharc_resolved` (read), `sharc_keys` and `sharc_accum` (read_write).
    pub update_bgl: wgpu::BindGroupLayout,
    /// `@group(2)` of Pass 2: `sharc_resolved` only, protecting the per-stage storage buffer budget.
    pub read_bgl: wgpu::BindGroupLayout,
    /// `@group(0)` of `sharc_resolve.wgsl`: keys, accumulators and resolved slots (all read_write).
    pub resolve_bgl: wgpu::BindGroupLayout,
}

/// SHARC cache buffers and the bind groups that expose them to each pass.
pub struct SharcResources {
    pub keys_buf: wgpu::Buffer,
    pub accum_buf: wgpu::Buffer,
    pub resolved_buf: wgpu::Buffer,
    /// `@group(2)` for `sharc_update.wgsl`.
    pub update_bind_group: wgpu::BindGroup,
    /// `@group(2)` for Pass 2 (`lighting_restir_gi.wgsl`).
    pub read_bind_group: wgpu::BindGroup,
    /// `@group(0)` for `sharc_resolve.wgsl`.
    pub resolve_bind_group: wgpu::BindGroup,
    /// `@group(1)` for `sharc_update.wgsl`; screen-size dependent, see `rebuild_gbuf_bind_group`.
    pub gbuf_bind_group: wgpu::BindGroup,
}

impl SharcResources {
    /// Allocates the zero-initialised cache (all slots empty) and builds every bind group.
    pub fn new(
        device: &wgpu::Device,
        layouts: &SharcLayouts,
        gbuf_depth_mat_buf: &wgpu::Buffer,
        gbuf_norm_motion_buf: &wgpu::Buffer,
    ) -> Self {
        let keys_buf = create_cache_buffer(device, "sharc_keys_buf", SHARC_KEYS_BYTES);
        let accum_buf = create_cache_buffer(device, "sharc_accum_buf", SHARC_ACCUM_BYTES);
        let resolved_buf = create_cache_buffer(device, "sharc_resolved_buf", SHARC_RESOLVED_BYTES);

        let update_bind_group = create_update_bind_group(
            device,
            &layouts.update_bgl,
            &keys_buf,
            &accum_buf,
            &resolved_buf,
        );
        let read_bind_group = create_read_bind_group(device, &layouts.read_bgl, &resolved_buf);
        let resolve_bind_group = create_resolve_bind_group(
            device,
            &layouts.resolve_bgl,
            &keys_buf,
            &accum_buf,
            &resolved_buf,
        );
        let gbuf_bind_group = create_gbuf_bind_group(
            device,
            &layouts.gbuf_bgl,
            gbuf_depth_mat_buf,
            gbuf_norm_motion_buf,
        );

        Self {
            keys_buf,
            accum_buf,
            resolved_buf,
            update_bind_group,
            read_bind_group,
            resolve_bind_group,
            gbuf_bind_group,
        }
    }

    /// Re-points the update pass at freshly allocated G-Buffer buffers; must follow every `ScreenResources::new`.
    pub fn rebuild_gbuf_bind_group(
        &mut self,
        device: &wgpu::Device,
        layouts: &SharcLayouts,
        gbuf_depth_mat_buf: &wgpu::Buffer,
        gbuf_norm_motion_buf: &wgpu::Buffer,
    ) {
        self.gbuf_bind_group = create_gbuf_bind_group(
            device,
            &layouts.gbuf_bgl,
            gbuf_depth_mat_buf,
            gbuf_norm_motion_buf,
        );
    }
}

/// Creates the four SHARC bind group layouts.
pub fn create_sharc_layouts(device: &wgpu::Device) -> SharcLayouts {
    SharcLayouts {
        gbuf_bgl: create_gbuf_bgl(device),
        update_bgl: create_update_bgl(device),
        read_bgl: create_read_bgl(device),
        resolve_bgl: create_resolve_bgl(device),
    }
}

/// Compiles the view-seeded tile path update pipeline (`sharc_update.wgsl`).
pub fn create_sharc_update_pipeline(
    device: &wgpu::Device,
    group0_bgl: &wgpu::BindGroupLayout,
    layouts: &SharcLayouts,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sharc_update_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHARC_UPDATE_SRC)),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sharc_update_layout"),
        bind_group_layouts: &[group0_bgl, &layouts.gbuf_bgl, &layouts.update_bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_sharc_update"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Compiles the 0-ray EMA resolve and aging pipeline (`sharc_resolve.wgsl`).
pub fn create_sharc_resolve_pipeline(
    device: &wgpu::Device,
    layouts: &SharcLayouts,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sharc_resolve_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHARC_RESOLVE_SRC)),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sharc_resolve_layout"),
        bind_group_layouts: &[&layouts.resolve_bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_sharc_resolve"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Screen tile grid `(tiles_x, tiles_y)`; each tile seeds exactly one SHARC path per frame.
pub fn sharc_tile_grid(width: u32, height: u32, tile_size: u32) -> (u32, u32) {
    let tile = tile_size.max(1);
    (width.div_ceil(tile), height.div_ceil(tile))
}

/// Workgroup counts for `sharc_update.wgsl` covering the tile grid with 8x8 threads per workgroup.
pub fn sharc_update_workgroups(width: u32, height: u32, tile_size: u32) -> (u32, u32) {
    let (tiles_x, tiles_y) = sharc_tile_grid(width, height, tile_size);
    (
        tiles_x.div_ceil(UPDATE_WORKGROUP_EDGE),
        tiles_y.div_ceil(UPDATE_WORKGROUP_EDGE),
    )
}

/// Workgroup count for `sharc_resolve.wgsl`: one thread per slot.
pub const fn sharc_resolve_workgroups() -> u32 {
    SHARC_SLOTS / RESOLVE_WORKGROUP_SIZE
}

/// Records the tile-seeded path trace and deposit pass; reads the current frame's G-Buffer.
pub fn record_sharc_update_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group0: &wgpu::BindGroup,
    sharc: &SharcResources,
    workgroups: (u32, u32),
) {
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("sharc_update_pass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, group0, &[]);
    cpass.set_bind_group(1, &sharc.gbuf_bind_group, &[]);
    cpass.set_bind_group(2, &sharc.update_bind_group, &[]);
    cpass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
}

/// Records the resolve and aging pass; a separate compute pass orders it after every update deposit.
pub fn record_sharc_resolve_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    resolve_bind_group: &wgpu::BindGroup,
) {
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("sharc_resolve_pass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, resolve_bind_group, &[]);
    cpass.dispatch_workgroups(sharc_resolve_workgroups(), 1, 1);
}

// Private Buffer, Bind Group Layout & Bind Group Builders

fn create_cache_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_gbuf_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sharc_gbuf_bgl"),
        entries: &[
            // GPU-SYNC: Mirrors `gbuf_depth_mat` in `shaders/sharc_update.wgsl` (@group(1) @binding(0)).
            storage_entry(0, true),
            // GPU-SYNC: Mirrors `gbuf_norm_motion` in `shaders/sharc_update.wgsl` (@group(1) @binding(1)).
            storage_entry(1, true),
        ],
    })
}

fn create_update_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sharc_update_bgl"),
        entries: &[
            // GPU-SYNC: Mirrors `sharc_resolved` in `shaders/common/sharc_query.wgsl` (@group(2) @binding(0)).
            storage_entry(0, true),
            // GPU-SYNC: Mirrors `sharc_keys` in `shaders/sharc_update.wgsl` (@group(2) @binding(1)).
            storage_entry(1, false),
            // GPU-SYNC: Mirrors `sharc_accum` in `shaders/sharc_update.wgsl` (@group(2) @binding(2)).
            storage_entry(2, false),
        ],
    })
}

fn create_read_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sharc_read_bgl"),
        entries: &[
            // GPU-SYNC: Mirrors `sharc_resolved` in `shaders/common/sharc_query.wgsl` (@group(2) @binding(0)).
            storage_entry(0, true),
        ],
    })
}

fn create_resolve_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sharc_resolve_bgl"),
        entries: &[
            // GPU-SYNC: Mirrors `sharc_keys` in `shaders/sharc_resolve.wgsl` (@group(0) @binding(0)).
            storage_entry(0, false),
            // GPU-SYNC: Mirrors `sharc_accum` in `shaders/sharc_resolve.wgsl` (@group(0) @binding(1)).
            storage_entry(1, false),
            // GPU-SYNC: Mirrors `sharc_resolved` in `shaders/sharc_resolve.wgsl` (@group(0) @binding(2)).
            storage_entry(2, false),
        ],
    })
}

fn create_gbuf_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    gbuf_depth_mat_buf: &wgpu::Buffer,
    gbuf_norm_motion_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sharc_gbuf_bind_group"),
        layout,
        entries: &[
            // GPU-SYNC: Mirrors `gbuf_depth_mat` in `shaders/sharc_update.wgsl` (@group(1) @binding(0)).
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gbuf_depth_mat_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `gbuf_norm_motion` in `shaders/sharc_update.wgsl` (@group(1) @binding(1)).
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gbuf_norm_motion_buf.as_entire_binding(),
            },
        ],
    })
}

fn create_update_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    keys_buf: &wgpu::Buffer,
    accum_buf: &wgpu::Buffer,
    resolved_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sharc_update_bind_group"),
        layout,
        entries: &[
            // GPU-SYNC: Mirrors `sharc_resolved` in `shaders/common/sharc_query.wgsl` (@group(2) @binding(0)).
            wgpu::BindGroupEntry {
                binding: 0,
                resource: resolved_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `sharc_keys` in `shaders/sharc_update.wgsl` (@group(2) @binding(1)).
            wgpu::BindGroupEntry {
                binding: 1,
                resource: keys_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `sharc_accum` in `shaders/sharc_update.wgsl` (@group(2) @binding(2)).
            wgpu::BindGroupEntry {
                binding: 2,
                resource: accum_buf.as_entire_binding(),
            },
        ],
    })
}

fn create_read_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    resolved_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sharc_read_bind_group"),
        layout,
        entries: &[
            // GPU-SYNC: Mirrors `sharc_resolved` in `shaders/common/sharc_query.wgsl` (@group(2) @binding(0)).
            wgpu::BindGroupEntry {
                binding: 0,
                resource: resolved_buf.as_entire_binding(),
            },
        ],
    })
}

fn create_resolve_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    keys_buf: &wgpu::Buffer,
    accum_buf: &wgpu::Buffer,
    resolved_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sharc_resolve_bind_group"),
        layout,
        entries: &[
            // GPU-SYNC: Mirrors `sharc_keys` in `shaders/sharc_resolve.wgsl` (@group(0) @binding(0)).
            wgpu::BindGroupEntry {
                binding: 0,
                resource: keys_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `sharc_accum` in `shaders/sharc_resolve.wgsl` (@group(0) @binding(1)).
            wgpu::BindGroupEntry {
                binding: 1,
                resource: accum_buf.as_entire_binding(),
            },
            // GPU-SYNC: Mirrors `sharc_resolved` in `shaders/sharc_resolve.wgsl` (@group(0) @binding(2)).
            wgpu::BindGroupEntry {
                binding: 2,
                resource: resolved_buf.as_entire_binding(),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_layout_matches_wgsl_contract() {
        assert_eq!(std::mem::size_of::<SharcResolvedSlot>(), 16);
        assert_eq!(SHARC_SLOTS, 131_072);
        assert_eq!(
            SHARC_KEYS_BYTES + SHARC_ACCUM_BYTES + SHARC_RESOLVED_BYTES,
            4_718_592,
            "4.5 MiB total SHARC VRAM"
        );
    }

    #[test]
    fn tile_grid_matches_plan_path_budgets_at_1080p() {
        let (tx, ty) = sharc_tile_grid(1920, 1080, 16);
        assert_eq!(tx * ty, 8_160);
        let (tx, ty) = sharc_tile_grid(1920, 1080, 8);
        assert_eq!(tx * ty, 32_400);
    }

    #[test]
    fn update_workgroups_cover_every_tile() {
        for tile in [8u32, 16] {
            let (tiles_x, tiles_y) = sharc_tile_grid(1920, 1080, tile);
            let (wx, wy) = sharc_update_workgroups(1920, 1080, tile);
            assert!(wx * UPDATE_WORKGROUP_EDGE >= tiles_x);
            assert!(wy * UPDATE_WORKGROUP_EDGE >= tiles_y);
        }
    }

    #[test]
    fn resolve_workgroups_cover_every_slot_exactly() {
        assert_eq!(
            sharc_resolve_workgroups() * RESOLVE_WORKGROUP_SIZE,
            SHARC_SLOTS
        );
    }
}
