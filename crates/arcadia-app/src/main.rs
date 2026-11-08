//! Executable application entry point and event loop bootstrap.
//! Owns CPU process lifecycle, logging initialization, and winit event loop execution.

use winit::event_loop::{ControlFlow, EventLoop};

use arcadia_app::app::App;
use arcadia_app::cli::LaunchConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = env_logger::try_init();

    let Some(config) = LaunchConfig::parse()? else {
        return Ok(());
    };

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(config);
    event_loop.run_app(&mut app)?;

    Ok(())
}
