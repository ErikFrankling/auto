//! A window onto a world of soft cells: watch it, change it, look at it any way.
//!   ./soft
//!   ./soft --size 512 --seed 7
//!   ./soft --load world.bin

mod sim;
mod ui;

use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use auto_core::{Checks, Gpu};
use auto_life_soft::{Config, VRAM_MARGIN, free_vram, view};
use clap::Parser;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::platform::wayland::WindowAttributesExtWayland;
use winit::window::{Window, WindowId};

use sim::{Cmd, Sim, Status};

#[derive(Parser, Clone)]
#[command(about = "A window onto a world of soft cells")]
struct Args {
    /// Start from this saved world.
    #[arg(long)]
    load: Option<String>,
    /// How many squares wide a new world is.
    #[arg(long)]
    size: Option<u32>,
    #[arg(long, default_value_t = 1)]
    seed: u32,
    /// Directory with defaults.json and the .wgsl files, to run edited physics with no rebuild.
    #[arg(long)]
    shaders: Option<PathBuf>,
    /// Physics setting, as key=value. Repeat for more.
    #[arg(long = "set")]
    sets: Vec<String>,
}

struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Arc<Gpu>,
    egui_state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    ui: ui::Ui,
    tx: mpsc::Sender<Cmd>,
    worker: Option<std::thread::JoinHandle<()>>,
    next_redraw: Option<Instant>,
}

struct App {
    args: Args,
    proxy: EventLoopProxy<()>,
    running: Option<Running>,
}

/// What to open with: the file asked for, or a new world that fits what VRAM is free.
fn first_world(args: &Args, gpu: &Gpu) -> Result<Cmd, String> {
    if let Some(path) = &args.load {
        Config::from_save(path, args.shaders.clone())?;
        return Ok(Cmd::Load(path.clone()));
    }
    let software = gpu.name().to_lowercase().contains("llvmpipe");
    let mut size = args.size.unwrap_or(256);
    let mut physics = Config::new(size, args.seed, args.shaders.clone())?;
    for s in &args.sets {
        let (k, v) = s.split_once('=').ok_or_else(|| format!("--set {s} needs key=value"))?;
        physics.set(k, v)?;
    }
    let physics = physics.physics;
    loop {
        let mut cfg = Config::new(size, args.seed, args.shaders.clone())?;
        cfg.physics = physics.clone();
        let fits = software || free_vram().is_none_or(|free| cfg.vram_bytes() + VRAM_MARGIN <= free);
        if fits || size <= 64 {
            return Ok(Cmd::New {
                size,
                seed: args.seed,
                physics,
            });
        }
        size /= 2;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.running.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Soft life")
                        .with_name("soft-app", "soft-app")
                        .with_inner_size(winit::dpi::LogicalSize::new(1500.0, 950.0))
                        .with_min_inner_size(winit::dpi::LogicalSize::new(760.0, 480.0)),
                )
                .expect("window"),
        );
        let instance = auto_core::gpu::instance_with_display(event_loop.owned_display_handle());
        let surface = instance.create_surface(window.clone()).expect("window surface");
        let gpu = Arc::new(
            pollster::block_on(Gpu::for_surface(instance, &surface, Checks::Safe)).expect("gpu"),
        );
        eprintln!("gpu: {}", gpu.name());
        // A bad setting shouldn't take the whole window down. Errors are shown, not fatal.
        let status = Arc::new(Mutex::new(Status::default()));
        {
            let status = status.clone();
            gpu.device.on_uncaptured_error(Arc::new(move |e| {
                eprintln!("gpu error: {e}");
                if let Ok(mut st) = status.lock() {
                    st.message = Some((true, format!("GPU error: {e}")));
                }
            }));
        }

        let size = window.inner_size();
        let caps = surface.get_capabilities(&gpu.adapter);
        // Plain 8-bit output is what egui's colours are written for. A driver may list wider
        // formats first, and those would wash everything out.
        let format = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        ]
        .into_iter()
        .find(|f| caps.formats.contains(f))
        .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&gpu.device, &config);

        let ctx = egui::Context::default();
        ui::style(&ctx);
        let egui_state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            event_loop,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            format,
            egui_wgpu::RendererOptions {
                dithering: false,
                ..Default::default()
            },
        );

        let (tx, rx) = mpsc::channel();
        let view_layout = view::layout(&gpu.device);
        let first = first_world(&self.args, &gpu).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        let proxy = self.proxy.clone();
        let repaint: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            let _ = proxy.send_event(());
        });
        let worker = {
            let sim = Sim {
                gpu: gpu.clone(),
                status: status.clone(),
                view_layout: view_layout.clone(),
                shaders: self.args.shaders.clone(),
                rx,
                repaint,
            };
            std::thread::Builder::new()
                .name("world".into())
                .spawn(move || sim.run(first))
                .expect("thread")
        };
        let ui = ui::Ui::new(ctx, gpu.clone(), view_layout, status, tx.clone(), self.args.shaders.clone(), self.args.seed);
        self.running = Some(Running {
            window,
            surface,
            config,
            gpu,
            egui_state,
            renderer,
            ui,
            tx,
            worker: Some(worker),
            next_redraw: Some(Instant::now()),
        });
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(r) = &mut self.running else {
            return;
        };
        match r.next_redraw {
            Some(at) if at <= Instant::now() => {
                r.next_redraw = None;
                r.window.request_redraw();
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            Some(at) => event_loop.set_control_flow(ControlFlow::WaitUntil(at)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }

    fn user_event(&mut self, _: &ActiveEventLoop, _: ()) {
        if let Some(r) = &self.running {
            r.window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(r) = &mut self.running else {
            return;
        };
        let response = r.egui_state.on_window_event(&r.window, &event);
        if response.repaint {
            r.window.request_redraw();
        }
        match event {
            WindowEvent::CloseRequested => {
                let _ = r.tx.send(Cmd::Quit);
                if let Some(w) = r.worker.take() {
                    let _ = w.join();
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                r.config.width = size.width.max(1);
                r.config.height = size.height.max(1);
                r.surface.configure(&r.gpu.device, &r.config);
                r.window.request_redraw();
            }
            WindowEvent::RedrawRequested => r.frame(),
            _ => {}
        }
    }
}

impl Running {
    fn frame(&mut self) {
        let started = Instant::now();
        // Nothing else polls while the world is paused, and finished work needs noticing.
        let _ = self.gpu.device.poll(wgpu::PollType::Poll);
        let raw = self.egui_state.take_egui_input(&self.window);
        let ctx = self.ui.ctx.clone();
        let output = ctx.run_ui(raw, |root| {
            self.ui.show(root, &mut self.renderer);
        });
        self.egui_state
            .handle_platform_output(&self.window, output.platform_output);
        self.window.set_title(&self.ui.title());

        let ppp = output.pixels_per_point;
        let jobs = ctx.tessellate(output.shapes, ppp);
        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.config);
                self.window.request_redraw();
                return;
            }
            _ => {
                self.window.request_redraw();
                return;
            }
        };
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: ppp,
        };
        let (device, queue) = (&self.gpu.device, &self.gpu.queue);
        for (id, delta) in &output.textures_delta.set {
            for one in delta {
                self.renderer.update_texture(device, queue, *id, one);
            }
        }
        let mut enc = device.create_command_encoder(&Default::default());
        let extra = self.renderer.update_buffers(device, queue, &mut enc, &jobs, &screen);
        let view = texture.texture.create_view(&Default::default());
        {
            let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("window"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
        }
        queue.submit(extra.into_iter().chain([enc.finish()]));
        self.window.pre_present_notify();
        queue.present(texture);
        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }

        // Keep drawing at up to 60 frames a second while there is anything to see, and rest otherwise.
        let delay = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(Duration::from_secs(1), |v| v.repaint_delay);
        self.next_redraw = if self.ui.wants_frames() || delay.is_zero() {
            Some(started + Duration::from_millis(16))
        } else if delay < Duration::from_secs(1) {
            Some(Instant::now() + delay)
        } else {
            None
        };
    }
}

fn main() {
    let args = Args::parse();
    let event_loop = EventLoop::<()>::with_user_event().build().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let proxy = event_loop.create_proxy();
    let mut app = App {
        args,
        proxy,
        running: None,
    };
    event_loop.run_app(&mut app).expect("event loop");
}
