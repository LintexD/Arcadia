//! Application coordinator module managing lifecycle, simulation stepping, rendering, and winit events.
//! Owns CPU application state and orchestrates wgpu presentation.

pub mod event;
pub mod render;
pub mod simulation;
pub mod state;

pub use simulation::CameraView;
pub use state::{App, BakeResult, CalibrationTask};
