//! Headless GPU integration test for the SHARC cache: concurrent deposits, resolve EMA/aging, and `query_sharc`.
//! Skips (passes) when no GPU adapter is available; drives the real `sharc_update.wgsl` / `sharc_resolve.wgsl` code.

#![allow(clippy::chunks_exact_to_as_chunks)]

use arcadia_render::sharc::{
    create_sharc_layouts, create_sharc_resolve_pipeline, record_sharc_resolve_pass, SharcResources,
    SHARC_SLOTS, SHARC_UPDATE_SRC,
};

/// Extra entry points appended to the real update module so tests reuse `sharc_deposit` and `query_sharc` verbatim.
const TEST_KERNELS: &str = r#"
@group(3) @binding(0) var<storage, read_write> test_out: array<vec4<f32>>;

const TEST_POS: vec3<f32> = vec3<f32>(40.13, 12.07, 55.31);
const TEST_CAM: vec3<f32> = vec3<f32>(41.0, 13.0, 56.0);

@compute @workgroup_size(16, 1, 1)
fn test_deposit_first(@builtin(global_invocation_id) gid: vec3<u32>) {
    sharc_deposit(sharc_make_key(TEST_POS, vec3<f32>(1.0, 0.0, 0.0), TEST_CAM, vec2<f32>(0.0)), vec3<f32>(1.0, 2.0, 3.0));
    if (gid.x % 2u == 0u) {
        sharc_deposit(sharc_make_key(TEST_POS, vec3<f32>(-1.0, 0.0, 0.0), TEST_CAM, vec2<f32>(0.0)), vec3<f32>(0.25, 0.5, 0.75));
    }
}

@compute @workgroup_size(16, 1, 1)
fn test_deposit_second(@builtin(global_invocation_id) gid: vec3<u32>) {
    sharc_deposit(sharc_make_key(TEST_POS, vec3<f32>(1.0, 0.0, 0.0), TEST_CAM, vec2<f32>(0.0)), vec3<f32>(3.0, 2.0, 1.0));
}

@compute @workgroup_size(1, 1, 1)
fn test_query() {
    test_out[0] = query_sharc(TEST_POS, vec3<f32>(1.0, 0.0, 0.0), TEST_CAM, vec2<f32>(0.0));
    test_out[1] = query_sharc(TEST_POS, vec3<f32>(1.0, 0.0, 0.0), TEST_CAM, vec2<f32>(0.9, -0.8));
    test_out[2] = query_sharc(TEST_POS, vec3<f32>(-1.0, 0.0, 0.0), TEST_CAM, vec2<f32>(0.0));
    test_out[3] = query_sharc(TEST_POS + vec3<f32>(50.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), TEST_CAM, vec2<f32>(0.0));
    test_out[4] = query_sharc(TEST_POS, vec3<f32>(0.0, 1.0, 0.0), TEST_CAM, vec2<f32>(0.0));
}
"#;

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    sharc: SharcResources,
    resolve_pipeline: wgpu::ComputePipeline,
    deposit_first: wgpu::ComputePipeline,
    deposit_second: wgpu::ComputePipeline,
    query: wgpu::ComputePipeline,
    empty_bind_group: wgpu::BindGroup,
    test_bind_group: wgpu::BindGroup,
    test_out_buf: wgpu::Buffer,
}

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let limits = adapter.limits();
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("sharc_test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: Default::default(),
        },
        None,
    ))
    .ok()
}

impl Harness {
    fn new() -> Option<Self> {
        let (device, queue) = headless_device()?;
        let layouts = create_sharc_layouts(&device);

        let dummy_gbuf = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let sharc = SharcResources::new(
            &device,
            &layouts,
            &dummy_gbuf("test_gbuf_depth_mat"),
            &dummy_gbuf("test_gbuf_norm_motion"),
        );
        let resolve_pipeline = create_sharc_resolve_pipeline(&device, &layouts);

        let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_empty_bgl"),
            entries: &[],
        });
        let empty_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test_empty_bind_group"),
            layout: &empty_bgl,
            entries: &[],
        });
        let test_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_out_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let test_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_out_buf"),
            size: 8 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let test_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test_out_bind_group"),
            layout: &test_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: test_out_buf.as_entire_binding(),
            }],
        });

        let source = format!("{SHARC_UPDATE_SRC}\n{TEST_KERNELS}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sharc_test_module"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(source)),
        });
        let test_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sharc_test_layout"),
            bind_group_layouts: &[&empty_bgl, &empty_bgl, &layouts.update_bgl, &test_bgl],
            push_constant_ranges: &[],
        });
        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&test_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let deposit_first = make_pipeline("test_deposit_first");
        let deposit_second = make_pipeline("test_deposit_second");
        let query = make_pipeline("test_query");

        Some(Self {
            device,
            queue,
            sharc,
            resolve_pipeline,
            deposit_first,
            deposit_second,
            query,
            empty_bind_group,
            test_bind_group,
            test_out_buf,
        })
    }

    fn run_test_kernel(&self, pipeline: &wgpu::ComputePipeline, workgroups: u32) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sharc_test_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sharc_test_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &self.empty_bind_group, &[]);
            cpass.set_bind_group(1, &self.empty_bind_group, &[]);
            cpass.set_bind_group(2, &self.sharc.update_bind_group, &[]);
            cpass.set_bind_group(3, &self.test_bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    fn resolve(&self, passes: u32) {
        for _ in 0..passes {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sharc_test_resolve_encoder"),
                });
            record_sharc_resolve_pass(
                &mut encoder,
                &self.resolve_pipeline,
                &self.sharc.resolve_bind_group,
            );
            self.queue.submit(Some(encoder.finish()));
        }
    }

    fn read_buffer(&self, src: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sharc_test_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sharc_test_readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res)
                .expect("INVARIANT: receiver outlives map_async callback");
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("INVARIANT: map_async callback always fires after poll(Wait)")
            .expect("INVARIANT: mapping a MAP_READ staging buffer must succeed");
        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        data
    }

    fn read_u32s(&self, src: &wgpu::Buffer, count: u64) -> Vec<u32> {
        let bytes = self.read_buffer(src, count * 4);
        bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
    }

    fn query_results(&self) -> Vec<[f32; 4]> {
        self.run_test_kernel(&self.query, 1);
        let bytes = self.read_buffer(&self.test_out_buf, 8 * 16);
        bytemuck::cast_slice::<u8, f32>(&bytes)
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect()
    }
}

fn assert_vec4(actual: [f32; 4], expected: [f32; 4], what: &str) {
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e).abs() < 1e-3,
            "{what}: expected {expected:?}, got {actual:?}"
        );
    }
}

#[test]
fn sharc_deposit_resolve_query_roundtrip_and_eviction() {
    let Some(h) = Harness::new() else {
        eprintln!("sharc_gpu: no GPU adapter available, skipping");
        return;
    };

    // Frame 1: 64 racing threads claim the +X surfel (32 also claim the -X face of the same cell).
    h.run_test_kernel(&h.deposit_first, 4);
    let keys = h.read_u32s(&h.sharc.keys_buf, u64::from(SHARC_SLOTS));
    assert_eq!(
        keys.iter().filter(|&&k| k != 0).count(),
        2,
        "concurrent CAS claims must create exactly one slot per surfel face"
    );
    let accum = h.read_u32s(&h.sharc.accum_buf, u64::from(SHARC_SLOTS) * 4);
    let mut counts: Vec<u32> = accum
        .chunks_exact(4)
        .map(|c| c[3])
        .filter(|&c| c > 0)
        .collect();
    counts.sort_unstable();
    assert_eq!(
        counts,
        vec![32, 64],
        "every deposit must be counted exactly once"
    );

    h.resolve(1);
    let accum = h.read_u32s(&h.sharc.accum_buf, u64::from(SHARC_SLOTS) * 4);
    assert!(
        accum.iter().all(|&v| v == 0),
        "resolve must clear every accumulator"
    );

    let q = h.query_results();
    assert_vec4(q[0], [1.0, 2.0, 3.0, 1.0], "unjittered +X query");
    assert_vec4(
        q[1],
        [1.0, 2.0, 3.0, 1.0],
        "jittered +X query (with unjittered fallback)",
    );
    assert_vec4(
        q[2],
        [0.25, 0.5, 0.75, 1.0],
        "-X face must not share the +X surfel",
    );
    assert_vec4(q[3], [0.0; 4], "distant position misses");
    assert_vec4(q[4], [0.0; 4], "+Y face of the same cell misses");

    // Frame 2: EMA with hist_frames = 2 gives alpha = 1/2 -> mix((1,2,3), (3,2,1), 0.5) = (2,2,2).
    h.run_test_kernel(&h.deposit_second, 1);
    h.resolve(1);
    let q = h.query_results();
    assert_vec4(q[0], [2.0, 2.0, 2.0, 1.0], "EMA blend of second frame");
    assert_vec4(
        q[2],
        [0.25, 0.5, 0.75, 1.0],
        "-X surfel aged but still resolved",
    );

    // Aging: after 63 more resolves the -X surfel (last touched in frame 1) sits at stale_age 64 and survives.
    h.resolve(63);
    let q = h.query_results();
    assert_vec4(
        q[2],
        [0.25, 0.5, 0.75, 1.0],
        "-X surfel retained at stale_age 64",
    );
    assert_vec4(q[0], [2.0, 2.0, 2.0, 1.0], "+X surfel retained");
    // One more resolve pushes the -X surfel to stale_age 65 (> 64) and evicts it; +X (stale_age 64) stays.
    h.resolve(1);
    let q = h.query_results();
    assert_vec4(
        q[2],
        [0.0; 4],
        "-X surfel evicted once stale_age exceeds 64",
    );
    assert_vec4(
        q[0],
        [2.0, 2.0, 2.0, 1.0],
        "+X surfel unaffected by -X eviction",
    );
    let keys = h.read_u32s(&h.sharc.keys_buf, u64::from(SHARC_SLOTS));
    assert_eq!(
        keys.iter().filter(|&&k| k != 0).count(),
        1,
        "evicted surfels must release their key slot"
    );
}
