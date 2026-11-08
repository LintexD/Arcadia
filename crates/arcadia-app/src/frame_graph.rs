//! Rolling frame time history and real-time GPU/CPU frame pacing graph.
//! Owns frame duration ring buffers, percentile jitter analysis, and egui graph rendering.

use std::collections::VecDeque;

use egui::{pos2, vec2, Align2, Color32, FontId, Id, Painter, Rect, Rounding, Stroke};

/// Number of frames retained in the rolling frame time history.
pub const FRAME_HISTORY_CAPACITY: usize = 180;

/// Statistical summary of recent frame pacing stability.
#[derive(Clone, Copy, Debug)]
pub struct FramePacingStats {
    pub current_ms: f32,
    pub avg_ms: f32,
    pub min_ms: f32,
    pub max_ms: f32,
    pub low_1_pct_fps: f32,
    pub std_dev_ms: f32,
    pub consistency_pct: f32,
}

impl Default for FramePacingStats {
    fn default() -> Self {
        Self {
            current_ms: 16.67,
            avg_ms: 16.67,
            min_ms: 16.67,
            max_ms: 16.67,
            low_1_pct_fps: 60.0,
            std_dev_ms: 0.0,
            consistency_pct: 100.0,
        }
    }
}

/// Rolling frame time history and stability analyzer.
pub struct FrameTimeTracker {
    history: VecDeque<f32>,
    pub stats: FramePacingStats,
}

impl Default for FrameTimeTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameTimeTracker {
    /// Creates a new frame time tracker with pre-allocated capacity.
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(FRAME_HISTORY_CAPACITY),
            stats: FramePacingStats::default(),
        }
    }

    /// Records a new frame duration in milliseconds and refreshes rolling statistics.
    pub fn push(&mut self, frame_time_ms: f32) {
        // Sanity clamp extreme anomalies (e.g. window pause/resize) to 250ms
        let clamped = frame_time_ms.clamp(0.1, 250.0);

        if self.history.len() >= FRAME_HISTORY_CAPACITY {
            self.history.pop_front();
        }
        self.history.push_back(clamped);

        self.recompute_stats(clamped);
    }

    /// Recomputes stability metrics over the current sample buffer.
    fn recompute_stats(&mut self, latest_ms: f32) {
        let n = self.history.len();
        if n == 0 {
            return;
        }

        let mut sum = 0.0;
        let mut min_ms = f32::MAX;
        let mut max_ms = 0.0f32;

        for &val in &self.history {
            sum += val;
            if val < min_ms {
                min_ms = val;
            }
            if val > max_ms {
                max_ms = val;
            }
        }

        let avg_ms = sum / n as f32;

        // Standard deviation (frame time jitter):
        let mut variance_sum = 0.0;
        for &val in &self.history {
            let diff = val - avg_ms;
            variance_sum += diff * diff;
        }
        let std_dev_ms = (variance_sum / n as f32).sqrt();

        // 1% Low calculation:
        let mut sorted = self.history.clone().into_iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99_idx = ((n as f32 * 0.99).ceil() as usize).min(n) - 1;
        let p99_ms = sorted[p99_idx];
        let low_1_pct_fps = if p99_ms > 0.0 { 1000.0 / p99_ms } else { 0.0 };

        // Consistency percentage:
        let cv = if avg_ms > 0.0 {
            std_dev_ms / avg_ms
        } else {
            1.0
        };
        let consistency_pct = (1.0 - cv).clamp(0.0, 1.0) * 100.0;

        self.stats = FramePacingStats {
            current_ms: latest_ms,
            avg_ms,
            min_ms,
            max_ms,
            low_1_pct_fps,
            std_dev_ms,
            consistency_pct,
        };
    }

    /// Draws the frame time graph HUD in the bottom-right corner.
    pub fn draw(&self, ctx: &egui::Context) {
        egui::Area::new(Id::new("frame_time_graph_hud"))
            .anchor(Align2::RIGHT_BOTTOM, vec2(-16.0, -16.0))
            .interactable(false)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                self.draw_ui(ui);
            });
    }

    fn draw_ui(&self, ui: &mut egui::Ui) {
        let panel_width = 320.0;
        let panel_height = 145.0;
        let (rect, _response) =
            ui.allocate_exact_size(vec2(panel_width, panel_height), egui::Sense::hover());
        let painter = ui.painter();

        // Background container:
        let bg_color = Color32::from_rgba_unmultiplied(12, 14, 20, 225);
        let border_stroke = Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(75, 85, 105, 180));
        painter.rect_filled(rect, Rounding::same(6.0), bg_color);
        painter.rect_stroke(rect, Rounding::same(6.0), border_stroke);

        let pad_x = 10.0;
        let pad_y = 8.0;

        // Title and toggle badge:
        let title_pos = pos2(rect.min.x + pad_x, rect.min.y + pad_y);
        painter.text(
            title_pos,
            Align2::LEFT_TOP,
            "FRAME PACING & STABILITY",
            FontId::proportional(11.0),
            Color32::from_gray(180),
        );

        let toggle_pos = pos2(rect.max.x - pad_x, rect.min.y + pad_y);
        painter.text(
            toggle_pos,
            Align2::RIGHT_TOP,
            "[F3] Toggle",
            FontId::monospace(10.0),
            Color32::from_rgb(100, 180, 255),
        );

        // Stats summary row:
        let avg_fps = if self.stats.avg_ms > 0.0 {
            1000.0 / self.stats.avg_ms
        } else {
            0.0
        };

        let fps_color = if avg_fps >= 58.0 {
            Color32::from_rgb(80, 240, 140)
        } else if avg_fps >= 29.0 {
            Color32::from_rgb(250, 210, 80)
        } else {
            Color32::from_rgb(250, 90, 80)
        };

        let stats_row_y = rect.min.y + 24.0;
        painter.text(
            pos2(rect.min.x + pad_x, stats_row_y),
            Align2::LEFT_TOP,
            format!("{:.1} FPS", avg_fps),
            FontId::proportional(15.0),
            fps_color,
        );

        painter.text(
            pos2(rect.min.x + pad_x + 75.0, stats_row_y + 2.0),
            Align2::LEFT_TOP,
            format!("{:.2} ms", self.stats.current_ms),
            FontId::proportional(12.0),
            Color32::WHITE,
        );

        let (rating_text, rating_color) = if self.stats.std_dev_ms < 1.0 {
            ("STABLE", Color32::from_rgb(80, 240, 140))
        } else if self.stats.std_dev_ms < 2.5 {
            ("MODERATE", Color32::from_rgb(250, 210, 80))
        } else {
            ("JITTER", Color32::from_rgb(255, 90, 80))
        };

        painter.text(
            pos2(rect.max.x - pad_x, stats_row_y + 2.0),
            Align2::RIGHT_TOP,
            format!("±{:.1}ms ({})", self.stats.std_dev_ms, rating_text),
            FontId::proportional(11.0),
            rating_color,
        );

        // Graph area:
        let graph_rect = Rect::from_min_max(
            pos2(rect.min.x + pad_x, rect.min.y + 46.0),
            pos2(rect.max.x - pad_x, rect.max.y - 18.0),
        );
        let graph_bg = Color32::from_rgba_unmultiplied(6, 8, 12, 190);
        painter.rect_filled(graph_rect, Rounding::same(3.0), graph_bg);
        painter.rect_stroke(
            graph_rect,
            Rounding::same(3.0),
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(50, 60, 75, 120)),
        );

        // Dynamic ceiling: at least 35ms, clamped up to 100ms for spikes
        let max_ms_scale = self.stats.max_ms.clamp(35.0, 100.0);

        // Target reference lines:
        self.draw_reference_line(
            painter,
            graph_rect,
            16.67,
            max_ms_scale,
            "16.7ms (60 FPS)",
            Color32::from_rgba_unmultiplied(80, 240, 140, 120),
        );
        self.draw_reference_line(
            painter,
            graph_rect,
            33.33,
            max_ms_scale,
            "33.3ms (30 FPS)",
            Color32::from_rgba_unmultiplied(250, 200, 60, 120),
        );

        // Draw histogram bars:
        let n_bars = self.history.len();
        if n_bars > 0 {
            let bar_width = graph_rect.width() / FRAME_HISTORY_CAPACITY as f32;
            let offset_x = graph_rect.max.x - (n_bars as f32 * bar_width);

            for (i, &ms) in self.history.iter().enumerate() {
                let bar_x = offset_x + (i as f32 * bar_width);
                let normalized_h = (ms / max_ms_scale).clamp(0.0, 1.0);
                let bar_height = (normalized_h * graph_rect.height()).max(1.0);
                let bar_y = graph_rect.max.y - bar_height;

                let bar_rect = Rect::from_min_max(
                    pos2(bar_x, bar_y),
                    pos2(bar_x + (bar_width - 0.5).max(1.0), graph_rect.max.y),
                );

                let bar_color = if ms <= 17.0 {
                    Color32::from_rgb(60, 220, 130)
                } else if ms <= 33.5 {
                    Color32::from_rgb(245, 195, 55)
                } else if ms <= 50.0 {
                    Color32::from_rgb(250, 110, 60)
                } else {
                    Color32::from_rgb(255, 60, 80)
                };

                painter.rect_filled(bar_rect, Rounding::ZERO, bar_color);
            }
        }

        // Footer detail row:
        let footer_y = rect.max.y - pad_y - 2.0;
        let footer_text = format!(
            "Min: {:.1}ms | Max: {:.1}ms | 1% Low: {:.1} FPS | Purity: {:.0}%",
            self.stats.min_ms,
            self.stats.max_ms,
            self.stats.low_1_pct_fps,
            self.stats.consistency_pct
        );
        painter.text(
            pos2(rect.min.x + pad_x, footer_y),
            Align2::LEFT_BOTTOM,
            footer_text,
            FontId::proportional(10.0),
            Color32::from_gray(160),
        );
    }

    fn draw_reference_line(
        &self,
        painter: &Painter,
        graph_rect: Rect,
        ms: f32,
        max_ms_scale: f32,
        label: &str,
        color: Color32,
    ) {
        if ms >= max_ms_scale {
            return;
        }
        let frac = ms / max_ms_scale;
        let y = graph_rect.max.y - (frac * graph_rect.height());
        let p1 = pos2(graph_rect.min.x, y);
        let p2 = pos2(graph_rect.max.x, y);

        painter.line_segment([p1, p2], Stroke::new(1.0_f32, color));
        painter.text(
            pos2(graph_rect.min.x + 3.0, y - 1.0),
            Align2::LEFT_BOTTOM,
            label,
            FontId::proportional(8.5),
            color,
        );
    }
}
