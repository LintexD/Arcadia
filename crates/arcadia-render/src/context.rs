//! `GpuContext`: Vulkan device, adapter, surface, and queue initialization.
//! Owns wgpu instance, surface configuration, and low-level GPU device context.

use std::sync::Arc;
use winit::window::Window;

/// Low-level GPU device context, surface configuration, and adapter capabilities.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub width: u32,
    pub height: u32,
    pub adapter_name: String,
    pub backend_name: String,
    pub is_integrated_gpu: bool,
    pub max_storage_bytes: u64,
}

impl GpuContext {
    /// Initializes the Vulkan high-performance GPU device, surface, and queues.
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("INVARIANT: Valid window reference must create surface successfully");

        let default_adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("INVARIANT: Must locate compatible GPU adapter");

        // WHY: Prefer discrete GPU over integrated GPU, and prefer Vulkan over DirectX 12
        // because standard WGSL 64-bit/atomic compute shaders are fully native in SPIR-V/Vulkan.
        let all_adapters = instance.enumerate_adapters(wgpu::Backends::all());
        let adapter = all_adapters
            .into_iter()
            .filter(|a| a.is_surface_supported(&surface))
            .max_by_key(|a| {
                let info = a.get_info();
                let type_score = match info.device_type {
                    wgpu::DeviceType::DiscreteGpu => 30,
                    wgpu::DeviceType::IntegratedGpu => 20,
                    wgpu::DeviceType::VirtualGpu => 10,
                    _ => 0,
                };
                let backend_score = match info.backend {
                    wgpu::Backend::Vulkan => 3,
                    wgpu::Backend::Metal => 2,
                    wgpu::Backend::Dx12 => 1,
                    _ => 0,
                };
                type_score + backend_score
            })
            .unwrap_or(default_adapter);

        let adapter_info = adapter.get_info();
        let adapter_name = adapter_info.name.clone();
        let backend_name = format!("{:?}", adapter_info.backend);
        let is_integrated_gpu = adapter_info.device_type == wgpu::DeviceType::IntegratedGpu;
        println!(
            "  Active GPU:        {} ({:?}, {:?})",
            adapter_info.name, adapter_info.device_type, adapter_info.backend
        );
        log::info!("Arcadia Renderer GPU: {}", adapter_info.name);

        let max_storage_bytes = adapter.limits().max_storage_buffer_binding_size as u64;
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("arcadia_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .expect("INVARIANT: GPU device request with supported limits must succeed");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: if surface_caps
                .present_modes
                .contains(&wgpu::PresentMode::Mailbox)
            {
                wgpu::PresentMode::Mailbox
            } else if surface_caps
                .present_modes
                .contains(&wgpu::PresentMode::Immediate)
            {
                wgpu::PresentMode::Immediate
            } else {
                wgpu::PresentMode::Fifo
            },
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        Self {
            instance,
            adapter,
            device,
            queue,
            surface,
            surface_config,
            width,
            height,
            adapter_name,
            backend_name,
            is_integrated_gpu,
            max_storage_bytes,
        }
    }

    /// Reconfigures swapchain surface dimensions upon window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
    }
}
