//! `arcadia-render`: Real-time wgpu Vulkan compute-driven path tracer & rasterizer for Arcadia.
//! Owns GPU pipelines, shaders, bind groups, intermediate textures, and CPU render orchestration.

pub mod camera;
pub mod context;
pub mod di;
pub mod gbuffer;
pub mod gpu_pools;
pub mod pipeline;
pub mod quality;
pub mod renderer;
pub mod sharc;

pub use camera::{
    compute_subpixel_jitter, halton, linear_distance_to_reverse_z, reverse_z_perspective_matrix,
    CameraUniforms,
};
pub use context::GpuContext;
pub use di::{DiLayouts, DiResources, EmissiveParams};
pub use gbuffer::{DiReservoir, GiReservoir, ScreenResources};
pub use gpu_pools::{create_group0_bgl, create_group0_resources, Group0Resources};
pub use pipeline::{
    create_pass_layouts, create_pipelines, create_reverse_z_depth_attachment,
    create_reverse_z_depth_stencil_state, PassLayouts, Pipelines, BLIT_WGSL, SHADER1_SRC,
    SHADER2_SRC, SHADER3_SRC, SHADER4_SRC,
};
pub use quality::RaymarchQualitySettings;
pub use renderer::{FrameParams, GuiCallback, Renderer};
pub use sharc::{SharcLayouts, SharcResources, SHARC_RESOLVE_SRC, SHARC_UPDATE_SRC};
