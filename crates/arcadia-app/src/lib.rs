//! `arcadia-app`: Application state machine, simulation loop, and interactive client.
//! Owns application orchestration, sculpting input, and frame graph profiling.

pub mod app;
pub mod benchmark;
pub mod cli;
pub mod frame_graph;
pub mod gui;
pub mod input;
pub mod sculpting;

pub use app::{App, CameraView};
