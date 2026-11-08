//! egui immediate-mode debug overlay and quality tuning panel.
//! Owns CPU UI state and egui-wgpu GPU render pass descriptors.

use egui::{Color32, Context, RichText, Slider, Ui, ViewportId, Window as EguiWindow};
use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use std::sync::Arc;
use winit::event::WindowEvent;
use winit::window::Window;

use arcadia_render::RaymarchQualitySettings;

use crate::frame_graph::FrameTimeTracker;

/// Real-time Debug & Quality Tuning Interface for Arcadia's Voxel Raymarcher.
pub struct DebugUi {
    pub ctx: Context,
    pub winit_state: EguiWinitState,
    pub renderer: EguiRenderer,
    pub visible: bool,
    pub frame_graph_visible: bool,
    pub frame_tracker: FrameTimeTracker,
    pub quality: RaymarchQualitySettings,
    pub calibration_requested: Option<f32>,
    pub calibration_status: Option<String>,
    pub current_fps: f32,
    pub current_frame_time_ms: f32,
}

impl DebugUi {
    pub fn new(
        window: &Arc<Window>,
        device: &wgpu::Device,
        output_color_format: wgpu::TextureFormat,
    ) -> Self {
        let ctx = Context::default();
        let viewport_id = ViewportId::ROOT;
        let winit_state = EguiWinitState::new(
            ctx.clone(),
            viewport_id,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let renderer = EguiRenderer::new(device, output_color_format, None, 1, false);

        Self {
            ctx,
            winit_state,
            renderer,
            visible: true,
            frame_graph_visible: true,
            frame_tracker: FrameTimeTracker::new(),
            quality: RaymarchQualitySettings::high_quality(),
            calibration_requested: None,
            calibration_status: None,
            current_fps: 60.0,
            current_frame_time_ms: 16.67,
        }
    }

    /// Records a frame duration in milliseconds for the rolling frame pacing graph.
    pub fn push_frame_time(&mut self, dt_ms: f32) {
        self.frame_tracker.push(dt_ms);
    }

    /// Handles winit window events, passing them to egui. Returns true if consumed by the UI.
    pub fn handle_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let response = self.winit_state.on_window_event(window, event);
        if self.visible {
            response.consumed
        } else {
            false
        }
    }

    /// Renders the debug interface overlay.
    pub fn render_frame(
        &mut self,
        window: &Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view_target: &wgpu::TextureView,
        gpu_name: &str,
    ) {
        if !self.visible && !self.frame_graph_visible {
            return;
        }

        let raw_input = self.winit_state.take_egui_input(window);
        let ctx = self.ctx.clone();
        let mut quality = self.quality;
        let mut calibration_requested = self.calibration_requested.take();
        let calibration_status = self.calibration_status.clone();
        let fps = self.current_fps;
        let ft_ms = self.current_frame_time_ms;
        let is_visible = self.visible;
        let mut frame_graph_visible = self.frame_graph_visible;

        let full_output = ctx.run(raw_input, |egui_ctx| {
            if is_visible {
                Self::draw_window(
                    egui_ctx,
                    gpu_name,
                    &mut quality,
                    &mut calibration_requested,
                    &calibration_status,
                    fps,
                    ft_ms,
                    &mut frame_graph_visible,
                );
            }
            if frame_graph_visible {
                self.frame_tracker.draw(egui_ctx);
            }
        });

        self.quality = quality;
        self.frame_graph_visible = frame_graph_visible;
        if calibration_requested.is_some() {
            self.calibration_requested = calibration_requested;
        }

        self.winit_state
            .handle_platform_output(window, full_output.platform_output);

        let tris = self
            .ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }

        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        self.renderer
            .update_buffers(device, queue, encoder, &tris, &screen_descriptor);

        {
            let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: view_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut rpass = rpass.forget_lifetime();
            self.renderer.render(&mut rpass, &tris, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_window(
        ctx: &Context,
        gpu_name: &str,
        quality: &mut RaymarchQualitySettings,
        calibration_requested: &mut Option<f32>,
        calibration_status: &Option<String>,
        current_fps: f32,
        current_frame_time_ms: f32,
        frame_graph_visible: &mut bool,
    ) {
        EguiWindow::new("⚙ Raymarcher Quality & Debug Interface")
            .default_pos([24.0, 24.0])
            .default_width(430.0)
            .resizable(true)
            .collapsible(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("{:.1} FPS", current_fps))
                            .color(if current_fps >= 55.0 {
                                Color32::from_rgb(80, 240, 140)
                            } else if current_fps >= 30.0 {
                                Color32::from_rgb(250, 210, 80)
                            } else {
                                Color32::from_rgb(250, 90, 80)
                            })
                            .strong()
                            .size(16.0),
                    );
                    ui.label(format!("({:.2} ms)", current_frame_time_ms));
                    ui.separator();
                    ui.label(RichText::new(gpu_name).weak().size(11.0));
                });

                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Toggle UI: [F1] / [Tab] | Free cursor to adjust sliders")
                            .weak()
                            .size(11.0),
                    );
                    ui.separator();
                    ui.checkbox(frame_graph_visible, "Frame Time Graph [F3]");
                });

                ui.separator();

                Self::draw_presets_section(ui, quality, calibration_requested, calibration_status);

                ui.separator();

                egui::ScrollArea::vertical()
                    .max_height(480.0)
                    .show(ui, |ui| {
                        Self::draw_primary_raymarcher_section(ui, quality);
                        ui.separator();
                        Self::draw_secondary_rays_section(ui, quality);
                        ui.separator();
                        Self::draw_restir_gi_section(ui, quality);
                        ui.separator();
                        Self::draw_denoising_section(ui, quality);
                        ui.separator();
                        Self::draw_upscaling_section(ui, quality);
                        ui.separator();
                        Self::draw_island_physics_section(ui, quality);
                    });
            });
    }

    fn draw_presets_section(
        ui: &mut Ui,
        quality: &mut RaymarchQualitySettings,
        calibration_requested: &mut Option<f32>,
        calibration_status: &Option<String>,
    ) {
        ui.label(
            RichText::new("Hardware Calibrated Presets (Zero-Jitter Stable Settings):").strong(),
        );
        ui.horizontal_wrapped(|ui| {
            if ui
                .button(
                    RichText::new("🚀 60 FPS Fluid")
                        .color(Color32::from_rgb(90, 240, 175))
                        .strong(),
                )
                .on_hover_text(
                    "Calibrated for 60 FPS on integrated graphics: 48% scale, 110 primary steps, trimmed GI, zero resolution jitter.",
                )
                .clicked()
            {
                *quality = RaymarchQualitySettings::calibrated_60fps();
            }

            if ui
                .button(
                    RichText::new("🎨 30 FPS Fidelity")
                        .color(Color32::from_rgb(120, 210, 255))
                        .strong(),
                )
                .on_hover_text(
                    "Calibrated for 30 FPS: 58% scale, 112 primary steps, balanced GI & shadows.",
                )
                .clicked()
            {
                *quality = RaymarchQualitySettings::calibrated_30fps();
            }

            if ui
                .button(RichText::new("💎 100% Native High Quality").strong())
                .on_hover_text("Original 100% native high-fidelity raymarcher settings.")
                .clicked()
            {
                *quality = RaymarchQualitySettings::high_quality();
            }

            if ui
                .button(RichText::new("✨ Ultra Fidelity").color(Color32::from_rgb(180, 160, 255)))
                .on_hover_text("Maximum quality: higher step limits and longer draw distances.")
                .clicked()
            {
                *quality = RaymarchQualitySettings::ultra();
            }
        });

        ui.separator();

        ui.label(
            RichText::new("⚡ Live Auto-Calibration (Evaluates Hardware for Static Lock):")
                .strong(),
        );
        ui.horizontal_wrapped(|ui| {
            if ui
                .button(RichText::new("Calibrate for 60 FPS [C]").strong())
                .on_hover_text("Measures live frame throughput for 1 second and automatically tunes all settings to hit 60 FPS with maximum primary ray fidelity.")
                .clicked()
            {
                *calibration_requested = Some(60.0);
            }

            if ui
                .button(RichText::new("Calibrate for 45 FPS").strong())
                .on_hover_text("Measures live frame throughput and tunes settings to hit 45 FPS.")
                .clicked()
            {
                *calibration_requested = Some(45.0);
            }

            if ui
                .button(RichText::new("Calibrate for 30 FPS").strong())
                .on_hover_text("Measures live frame throughput and tunes settings to hit 30 FPS.")
                .clicked()
            {
                *calibration_requested = Some(30.0);
            }
        });

        if let Some(status) = calibration_status {
            ui.label(
                RichText::new(format!("✔ {}", status))
                    .color(Color32::from_rgb(100, 240, 150))
                    .size(11.0),
            );
        }

        ui.separator();

        ui.horizontal(|ui| {
            ui.label(RichText::new("Render Scale (TSR + RCAS):").strong());
            let mut pct = (quality.render_scale * 100.0).round() as u32;
            if ui
                .add(Slider::new(&mut pct, 40..=100).text("%"))
                .on_hover_text(
                    "Internal raymarching resolution scale before 5-tap Catmull-Rom TSR + FSR RCAS contrast-adaptive sharpening.",
                )
                .changed()
            {
                quality.render_scale = (pct as f32) / 100.0;
            }
        });
    }

    fn draw_primary_raymarcher_section(ui: &mut Ui, quality: &mut RaymarchQualitySettings) {
        ui.collapsing(RichText::new("1. Primary Raymarcher").strong(), |ui| {
            ui.horizontal(|ui| {
                ui.label("Max Steps:");
                ui.add(Slider::new(&mut quality.primary_max_steps, 20..=300).text("steps"));
            });
            ui.horizontal(|ui| {
                ui.label("LOD Distance:");
                ui.add(Slider::new(&mut quality.primary_lod_dist, 20.0..=200.0).text("m"));
            });
            ui.horizontal(|ui| {
                ui.label("LOD Step Trigger:");
                ui.add(Slider::new(&mut quality.primary_lod_step, 30..=250).text("steps"));
            });
        });
    }

    fn draw_secondary_rays_section(ui: &mut Ui, quality: &mut RaymarchQualitySettings) {
        ui.collapsing(
            RichText::new("2. Secondary Rays (Water & Mirrors)").strong(),
            |ui| {
                ui.horizontal(|ui| {
                    ui.label("Secondary Max Steps:");
                    ui.add(Slider::new(&mut quality.sec_max_steps, 16..=150).text("steps"));
                });
                ui.horizontal(|ui| {
                    ui.label("Water Refraction Distance:");
                    ui.add(Slider::new(&mut quality.water_refr_dist, 4.0..=200.0).text("m"));
                });
                ui.horizontal(|ui| {
                    ui.label("Mirror Reflection Distance:");
                    ui.add(Slider::new(&mut quality.mirror_refl_dist, 10.0..=200.0).text("m"));
                });
                ui.horizontal(|ui| {
                    ui.label("Secondary LOD Distance:");
                    ui.add(Slider::new(&mut quality.sec_lod_dist, 10.0..=80.0).text("m"));
                });
                ui.horizontal(|ui| {
                    ui.label("Secondary LOD Steps:");
                    ui.add(Slider::new(&mut quality.sec_lod_step, 20..=100).text("steps"));
                });
                let mut adaptive = quality.mirror_adaptive != 0;
                if ui
                    .checkbox(&mut adaptive, "Distance-Adaptive Mirror Cone LOD")
                    .changed()
                {
                    quality.mirror_adaptive = if adaptive { 1 } else { 0 };
                }
            },
        );
    }

    fn draw_restir_gi_section(ui: &mut Ui, quality: &mut RaymarchQualitySettings) {
        ui.collapsing(RichText::new("3. ReSTIR GI & Sun Shadows").strong(), |ui| {
            ui.horizontal(|ui| {
                ui.label("GI Bounce Distance:");
                ui.add(Slider::new(&mut quality.gi_max_dist, 6.0..=50.0).text("m"));
            });
            ui.horizontal(|ui| {
                ui.label("GI Bounce Max Steps:");
                ui.add(Slider::new(&mut quality.gi_max_steps, 8..=100).text("steps"));
            });
            ui.horizontal(|ui| {
                ui.label("GI LOD Distance:");
                ui.add(Slider::new(&mut quality.gi_lod_dist, 6.0..=40.0).text("m"));
            });
            ui.horizontal(|ui| {
                ui.label("GI LOD Step Trigger:");
                ui.add(Slider::new(&mut quality.gi_lod_step, 10..=60).text("steps"));
            });

            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Shadow Max Distance:");
                ui.add(Slider::new(&mut quality.shadow_max_dist, 10.0..=160.0).text("m"));
            });
            ui.horizontal(|ui| {
                ui.label("Shadow Max Steps:");
                ui.add(Slider::new(&mut quality.shadow_max_steps, 16..=120).text("steps"));
            });
            ui.horizontal(|ui| {
                ui.label("Sun Penumbra Jitter:");
                ui.add(Slider::new(&mut quality.shadow_jitter, 0.000..=0.040));
            });
            ui.horizontal(|ui| {
                ui.label("Ambient Floor (Mood):");
                ui.add(Slider::new(&mut quality.ambient_floor, 0.000..=0.060).text("floor"));
            });
            ui.horizontal(|ui| {
                ui.label("Contact AO Fade Dist:");
                ui.add(Slider::new(&mut quality.contact_ao_dist, 4.0..=64.0).text("m"));
            });
            ui.horizontal(|ui| {
                ui.label("Shadow Traversal Mode:");
                ui.radio_value(&mut quality.shadow_mode, 0, "Accurate 64-Tree");
                ui.radio_value(&mut quality.shadow_mode, 1, "Fast Any-Hit");
            });
        });
    }

    fn draw_denoising_section(ui: &mut Ui, quality: &mut RaymarchQualitySettings) {
        ui.collapsing(
            RichText::new("4. Denoising & Reconstruction").strong(),
            |ui| {
                ui.horizontal(|ui| {
                    ui.label("Vogel-Disk Filter Taps:");
                    ui.add(Slider::new(&mut quality.denoiser_taps, 4..=16).text("taps"));
                });
            },
        );
    }

    fn draw_upscaling_section(ui: &mut Ui, quality: &mut RaymarchQualitySettings) {
        ui.collapsing(
            RichText::new("5. Upscaling & Presentation").strong(),
            |ui| {
                ui.horizontal(|ui| {
                    ui.label("Field of View [FOV]:");
                    ui.add(Slider::new(&mut quality.fov_deg, 40.0..=120.0).text("°"));
                });
                ui.horizontal(|ui| {
                    ui.label("Render Scale:");
                    ui.add(Slider::new(&mut quality.render_scale, 0.40..=1.00).text("scale"));
                });
                ui.horizontal(|ui| {
                    ui.label("Tonemapper [T]:");
                    ui.radio_value(&mut quality.tonemapper_mode, 0, "ACES Filmic");
                    ui.radio_value(&mut quality.tonemapper_mode, 1, "Polynomial AgX");
                });
                ui.horizontal(|ui| {
                    ui.label("Dithering [K]:");
                    ui.radio_value(&mut quality.dither_mode, 3, "IGN (Blue-Noise, Default)");
                    ui.radio_value(&mut quality.dither_mode, 1, "PCG Temporal");
                    ui.radio_value(&mut quality.dither_mode, 2, "Spatial Bit-Mix");
                    ui.radio_value(&mut quality.dither_mode, 0, "Off");
                });
                ui.horizontal(|ui| {
                    ui.label("Camera Jitter [J]:");
                    ui.radio_value(&mut quality.jitter_mode, 1, "Freeze when Still");
                    ui.radio_value(&mut quality.jitter_mode, 0, "Always Off");
                    ui.radio_value(&mut quality.jitter_mode, 2, "Always On");
                });
            },
        );
    }

    fn draw_island_physics_section(ui: &mut Ui, quality: &mut RaymarchQualitySettings) {
        ui.collapsing(
            RichText::new("6. Destruction & Island Physics").strong(),
            |ui| {
                ui.checkbox(
                    &mut quality.island_detection_enabled,
                    "Enable Structural Island Detachment [I]",
                );
                ui.label(
                    RichText::new("When disabled, severed voxels remain floating in mid-air (Minecraft-style).")
                        .weak()
                        .size(11.0),
                );

                ui.horizontal(|ui| {
                    ui.label("Island Max Depth:");
                    ui.add(
                        Slider::new(&mut quality.island_max_depth_m, 0.0..=30.0)
                            .text("m")
                            .custom_formatter(|v, _| {
                                if v <= 0.0 {
                                    "0.0 m (Floating)".to_string()
                                } else {
                                    format!("{:.1} m", v)
                                }
                            }),
                    );
                });
                ui.label(
                    RichText::new("Controls max search depth for both synchronous & async island detection passes (0m disables).")
                        .weak()
                        .size(11.0),
                );

                ui.horizontal(|ui| {
                    ui.label("Island Max Voxels:");
                    ui.add(
                        Slider::new(&mut quality.island_max_voxels, 1024..=1_048_576)
                            .logarithmic(true)
                            .text("vox"),
                    );
                });
                ui.label(
                    RichText::new("Max voxels checked per island before staying anchored as static terrain (<= 8 vox become debris particles).")
                        .weak()
                        .size(11.0),
                );

                ui.horizontal(|ui| {
                    ui.label("Surface Check Max Voxels:");
                    ui.add(
                        Slider::new(&mut quality.island_max_surface_voxels, 1024..=131_072)
                            .logarithmic(true)
                            .text("vox"),
                    );
                });
                ui.label(
                    RichText::new("Max boundary surface voxels explored during Phase 1 BFS before tagging as grounded terrain.")
                        .weak()
                        .size(11.0),
                );

                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Rest Timeout before Re-Adding:");
                    ui.add(Slider::new(&mut quality.object_bake_timeout_s, 5.0..=180.0).text("s"));
                });
                ui.label(
                    RichText::new("Stationary/uninteracted objects bake back into the static world map after this time (Default: 60s).")
                        .weak()
                        .size(11.0),
                );
            },
        );
    }
}
