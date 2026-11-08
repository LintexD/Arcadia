//! Winit application event handling and window lifecycle callbacks.
//! Owns CPU window and device event translation for the application event loop.

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{WindowAttributes, WindowId};

use arcadia_render::{RaymarchQualitySettings, Renderer};

use crate::app::App;
use crate::gui;

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = WindowAttributes::default()
                .with_title("Arcadia Voxel Engine — (1/16m)³ CS64 Ray Marcher + ReSTIR GI")
                .with_inner_size(LogicalSize::new(1920.0, 1080.0));

            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .expect("INVARIANT: Window creation with supported attributes must succeed"),
            );

            let renderer = pollster::block_on(Renderer::new(window.clone(), &self.world));
            let mut gui = gui::DebugUi::new(&window, renderer.device(), renderer.surface_format());

            if let Some(user_q) = self.user_quality {
                println!(
                    "  [Custom Preset]    Applying user preset: {:.0}% scale, {} primary steps, {} GI steps",
                    user_q.render_scale * 100.0,
                    user_q.primary_max_steps,
                    user_q.gi_max_steps
                );
                self.quality = user_q;
                gui.quality = self.quality;
            } else if renderer.is_integrated_gpu() {
                println!(
                    "  [Laptop Mode]      Integrated GPU detected — Calibrated 30 FPS active (Press [C] to Auto-Calibrate, [L] for 60 FPS, [F1] for menu)"
                );
                self.quality = RaymarchQualitySettings::calibrated_30fps();
                gui.quality = self.quality;
            }

            self.window = Some(window);
            self.renderer = Some(renderer);
            self.gui = Some(gui);
            self.last_frame_instant = Instant::now();
            self.fps_timer = Instant::now();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let (Some(window), Some(gui)) = (&self.window, &mut self.gui) {
            if gui.handle_event(window, &event) {
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(new_size.width, new_size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                self.update_and_render(event_loop);
            }
            WindowEvent::Focused(focused) => {
                if !focused && self.cursor_grabbed {
                    self.set_cursor_grab(false);
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                self.handle_mouse_input(button, state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.handle_mouse_wheel(delta);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard_input(&event, event_loop);
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            let gui_open = self.gui.as_ref().is_some_and(|g| g.visible);
            if self.cursor_grabbed && !self.benchmark.is_active() && !gui_open {
                let sensitivity = 0.0022;
                self.player.rotate(dx as f32, dy as f32, sensitivity);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}
