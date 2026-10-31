//! The live viewer. The same code drives a native window and a browser canvas.

mod viewer;

pub use viewer::{Control, Viewer, default_speed};

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native(start: Option<String>) {
    use std::sync::Arc;
    use winit::application::ApplicationHandler;
    use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::keyboard::{Key, NamedKey};
    use winit::window::{Window, WindowId};

    struct App {
        start: Option<String>,
        window: Option<Arc<Window>>,
        viewer: Option<Viewer>,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes().with_title("auto"))
                    .expect("window"),
            );
            let instance = auto_core::gpu::instance_with_display(event_loop.owned_display_handle());
            let viewer = pollster::block_on(Viewer::new(
                instance,
                window.clone(),
                self.start.as_deref().unwrap_or("life"),
            ));
            self.viewer = Some(viewer);
            self.window = Some(window);
        }

        fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
            let Some(viewer) = self.viewer.as_mut() else {
                return;
            };
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => viewer.resize(size.width, size.height),
                WindowEvent::RedrawRequested => {
                    viewer.frame();
                    if let Some(window) = &self.window {
                        window.set_title(&viewer.stats());
                        window.request_redraw();
                    }
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed =>
                {
                    let control = match event.logical_key.as_ref() {
                        Key::Named(NamedKey::Space) => Some(Control::TogglePause),
                        Key::Named(NamedKey::ArrowRight) => Some(Control::NextAutomaton),
                        Key::Named(NamedKey::ArrowLeft) => Some(Control::PreviousAutomaton),
                        Key::Named(NamedKey::ArrowUp) => Some(Control::Faster),
                        Key::Named(NamedKey::ArrowDown) => Some(Control::Slower),
                        Key::Character("r") => Some(Control::Reseed),
                        Key::Character("q") => {
                            event_loop.exit();
                            None
                        }
                        _ => None,
                    };
                    if let Some(c) = control {
                        viewer.control(c);
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                    };
                    viewer.zoom(0.85f32.powf(y));
                }
                _ => {}
            }
        }
    }

    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App {
        start,
        window: None,
        viewer: None,
    };
    event_loop.run_app(&mut app).expect("event loop");
}
