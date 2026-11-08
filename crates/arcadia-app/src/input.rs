//! First-person movement key tracking and user input dispatch.
//! Owns CPU input state and translates winit events to simulation actions.

use std::time::Instant;

use glam::Vec3;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};

use arcadia_core::{
    MaterialId, MAT_CRYSTAL_AMBER, MAT_CRYSTAL_CYAN, MAT_CRYSTAL_MAGENTA, MAT_LAVA_MAGMA,
    MAT_MIRROR_CHROME, MAT_MOSS_VIBRANT, MAT_POINT_LIGHT, MAT_SLATE_CARVED,
};
use arcadia_render::RaymarchQualitySettings;

use crate::app::{App, CalibrationTask};

/// Tracks active movement and control keys.
#[derive(Default, Debug)]
pub struct KeyState {
    pub w: bool,
    pub a: bool,
    pub s: bool,
    pub d: bool,
    pub space: bool,
    pub ctrl: bool,
    pub shift: bool,
    pub q: bool,
    pub e: bool,
}

/// Friendly display name for selectable materials.
pub fn material_name(mat_id: MaterialId) -> &'static str {
    match mat_id {
        MAT_CRYSTAL_CYAN => "1: Cyan Crystal (Emissive)",
        MAT_CRYSTAL_MAGENTA => "2: Magenta Crystal (Emissive)",
        MAT_CRYSTAL_AMBER => "3: Amber Sunstone (Emissive)",
        MAT_LAVA_MAGMA => "4: Molten Lava (Emissive)",
        MAT_SLATE_CARVED => "5: Carved Ruin Slate",
        MAT_MOSS_VIBRANT => "6: Lush Ruin Moss",
        MAT_POINT_LIGHT => "7: Point Light Beacon (Shadowed Point Light)",
        MAT_MIRROR_CHROME => "8: Mirror Chrome (Ray-Traced Reflection)",
        _ => "Custom Material",
    }
}

impl App {
    pub(crate) fn handle_mouse_input(&mut self, button: MouseButton, state: ElementState) {
        if state != ElementState::Pressed {
            return;
        }
        let gui_active = self
            .gui
            .as_ref()
            .is_some_and(|g| g.visible && g.ctx.wants_pointer_input());
        if gui_active {
            return;
        }
        match button {
            MouseButton::Left => {
                if !self.cursor_grabbed {
                    self.set_cursor_grab(true);
                } else if !self.benchmark.is_active() {
                    self.carve_action();
                }
            }
            MouseButton::Right => {
                if !self.cursor_grabbed {
                    self.set_cursor_grab(true);
                } else if !self.benchmark.is_active() {
                    self.place_action();
                }
            }
            _ => {}
        }
    }

    pub(crate) fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        let gui_active = self
            .gui
            .as_ref()
            .is_some_and(|g| g.visible && g.ctx.wants_pointer_input());
        if gui_active {
            return;
        }
        let scroll = match delta {
            MouseScrollDelta::LineDelta(_x, y) => y,
            MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) * 0.05,
        };
        self.brush_radius = (self.brush_radius + scroll * 0.125).clamp(0.25, 4.5);
    }

    pub(crate) fn handle_keyboard_input(&mut self, event: &KeyEvent, event_loop: &ActiveEventLoop) {
        let is_pressed = event.state == ElementState::Pressed;
        let repeat = event.repeat;

        let PhysicalKey::Code(key_code) = event.physical_key else {
            return;
        };

        match key_code {
            KeyCode::F1 | KeyCode::Tab | KeyCode::Backquote => {
                if is_pressed && !repeat {
                    if let Some(gui) = &mut self.gui {
                        gui.visible = !gui.visible;
                        if gui.visible {
                            self.set_cursor_grab(false);
                        }
                    }
                }
            }
            KeyCode::F3 => {
                if is_pressed && !repeat {
                    if let Some(gui) = &mut self.gui {
                        gui.frame_graph_visible = !gui.frame_graph_visible;
                        println!(
                            ">>> Frame Time Graph: {} [Key F3] <<<",
                            if gui.frame_graph_visible {
                                "ENABLED"
                            } else {
                                "DISABLED"
                            }
                        );
                    }
                }
            }
            KeyCode::KeyW => self.keys.w = is_pressed,
            KeyCode::KeyA => self.keys.a = is_pressed,
            KeyCode::KeyS => self.keys.s = is_pressed,
            KeyCode::KeyD => self.keys.d = is_pressed,
            KeyCode::Space => {
                if is_pressed && !repeat {
                    let now = Instant::now();
                    if let Some(prev) = self.last_space_press {
                        if now.duration_since(prev).as_secs_f32() < 0.35 {
                            self.player.fly_mode = !self.player.fly_mode;
                            self.player.velocity = Vec3::ZERO;
                            println!(
                                ">>> Switched to {} mode (Double-Space) <<<",
                                if self.player.fly_mode { "FLY" } else { "WALK" }
                            );
                            self.last_space_press = None;
                        } else {
                            self.last_space_press = Some(now);
                        }
                    } else {
                        self.last_space_press = Some(now);
                    }
                }
                self.keys.space = is_pressed;
            }
            KeyCode::ControlLeft | KeyCode::ControlRight => {
                self.keys.ctrl = is_pressed;
            }
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys.shift = is_pressed,
            KeyCode::KeyQ => self.keys.q = is_pressed,
            KeyCode::KeyE => self.keys.e = is_pressed,
            KeyCode::KeyB => {
                if is_pressed && !repeat {
                    self.benchmark.start();
                }
            }
            KeyCode::KeyG => {
                if is_pressed && !repeat {
                    self.scorch_rim = !self.scorch_rim;
                    log::info!("Scorch rim: {}", self.scorch_rim);
                }
            }
            KeyCode::KeyC => {
                if is_pressed && !repeat {
                    println!(">>> Starting Hardware Auto-Calibration for 60 FPS (sampling 40 frames)... <<<");
                    self.calibration = Some(CalibrationTask::new(60.0, 40));
                    if let Some(gui) = &mut self.gui {
                        gui.calibration_status =
                            Some("Measuring GPU throughput (40 frames)...".to_string());
                    }
                }
            }
            KeyCode::KeyI => {
                if is_pressed && !repeat {
                    self.quality.island_detection_enabled = !self.quality.island_detection_enabled;
                    println!(
                        ">>> Structural Island Detachment: {} [Key I] <<<",
                        if self.quality.island_detection_enabled {
                            "ENABLED"
                        } else {
                            "DISABLED"
                        }
                    );
                    if let Some(gui) = &mut self.gui {
                        gui.quality.island_detection_enabled =
                            self.quality.island_detection_enabled;
                    }
                }
            }
            KeyCode::KeyT => {
                if is_pressed && !repeat {
                    self.quality.tonemapper_mode = if self.quality.tonemapper_mode == 0 { 1 } else { 0 };
                    let mode_name = if self.quality.tonemapper_mode == 1 {
                        "Polynomial AgX (Wrensch / Sobotka)"
                    } else {
                        "ACES Filmic"
                    };
                    println!(">>> Tonemapper: {} [Key T] <<<", mode_name);
                    if let Some(gui) = &mut self.gui {
                        gui.quality.tonemapper_mode = self.quality.tonemapper_mode;
                    }
                }
            }
            KeyCode::KeyK => {
                if is_pressed && !repeat {
                    self.quality.dither_mode = match self.quality.dither_mode {
                        3 => 1,
                        1 => 2,
                        2 => 0,
                        _ => 3,
                    };
                    let mode_name = match self.quality.dither_mode {
                        1 => "PCG 2D Monochrome Sub-LSB TPDF",
                        2 => "Spatial Integer Bit-Mix Sub-LSB TPDF",
                        3 => "Interleaved Gradient Noise (IGN) Sub-LSB",
                        _ => "DISABLED (Off)",
                    };
                    println!(">>> Dithering Mode: {} [Key K] <<<", mode_name);
                    if let Some(gui) = &mut self.gui {
                        gui.quality.dither_mode = self.quality.dither_mode;
                    }
                }
            }
            KeyCode::KeyJ => {
                if is_pressed && !repeat {
                    self.quality.jitter_mode = match self.quality.jitter_mode {
                        1 => 0,
                        0 => 2,
                        _ => 1,
                    };
                    let mode_name = match self.quality.jitter_mode {
                        1 => "Freeze when Still (Zero Flicker Dynamic TSR) [Default]",
                        0 => "ALWAYS OFF (Zero Jitter)",
                        _ => "ALWAYS ON (Continuous Halton)",
                    };
                    println!(">>> Camera Jitter Mode: {} [Key J] <<<", mode_name);
                    if let Some(gui) = &mut self.gui {
                        gui.quality.jitter_mode = self.quality.jitter_mode;
                    }
                }
            }
            KeyCode::KeyL => {
                if is_pressed && !repeat {
                    if self.quality.render_scale < 0.52 {
                        self.quality = RaymarchQualitySettings::calibrated_30fps();
                        println!(">>> Switched to CALIBRATED 30 FPS (58% Scale, Balanced Fidelity) [Key L] <<<");
                    } else if self.quality.render_scale < 0.95 {
                        self.quality = RaymarchQualitySettings::calibrated_60fps();
                        println!(">>> Switched to CALIBRATED 60 FPS (48% Scale, Fluid Motion) [Key L] <<<");
                    } else {
                        self.quality = RaymarchQualitySettings::calibrated_30fps();
                        println!(">>> Switched to CALIBRATED 30 FPS [Key L] <<<");
                    }
                    if let Some(gui) = &mut self.gui {
                        gui.quality = self.quality;
                    }
                }
            }
            KeyCode::KeyN => {
                if is_pressed && !repeat {
                    if self.sun_angle.sin() > 0.25 {
                        self.sun_angle = 0.04;
                    } else if self.sun_angle.sin() > -0.2 {
                        self.sun_angle = 4.71;
                    } else {
                        self.sun_angle = 0.75;
                    }
                }
            }
            KeyCode::Digit1 => {
                if is_pressed {
                    self.selected_mat = MAT_CRYSTAL_CYAN;
                }
            }
            KeyCode::Digit2 => {
                if is_pressed {
                    self.selected_mat = MAT_CRYSTAL_MAGENTA;
                }
            }
            KeyCode::Digit3 => {
                if is_pressed {
                    self.selected_mat = MAT_CRYSTAL_AMBER;
                }
            }
            KeyCode::Digit4 => {
                if is_pressed {
                    self.selected_mat = MAT_LAVA_MAGMA;
                }
            }
            KeyCode::Digit5 => {
                if is_pressed {
                    self.selected_mat = MAT_SLATE_CARVED;
                }
            }
            KeyCode::Digit6 => {
                if is_pressed {
                    self.selected_mat = MAT_MOSS_VIBRANT;
                }
            }
            KeyCode::Digit7 => {
                if is_pressed {
                    self.selected_mat = MAT_POINT_LIGHT;
                }
            }
            KeyCode::Digit8 => {
                if is_pressed {
                    self.selected_mat = MAT_MIRROR_CHROME;
                }
            }
            KeyCode::Escape if is_pressed && !repeat => {
                if self.cursor_grabbed {
                    self.set_cursor_grab(false);
                } else {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}
