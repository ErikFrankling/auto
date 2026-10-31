//! One automaton running on a surface: step it, render it, blit it.

use auto_core::{Camera, Checks, Gpu, Sim, automata};

const BLIT: &str = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> srgb: vec4f;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(p[i], 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let c = textureLoad(tex, vec2i(pos.xy), 0).rgb;
    // On an sRGB surface the hardware will encode, so undo the gamma in the texture.
    return vec4f(mix(c, pow(c, vec3f(2.2)), srgb.x), 1.0);
}
"#;

#[derive(Clone, Copy, Debug)]
pub enum Control {
    TogglePause,
    NextAutomaton,
    PreviousAutomaton,
    Faster,
    Slower,
    Reseed,
}

/// Steps per displayed frame that make each automaton nice to watch.
pub fn default_speed(id: &str) -> u32 {
    match id {
        "life" => 1,
        "rule30" | "rule110" => 4,
        "brain" => 1,
        "wireworld" => 2,
        "cyclic" => 2,
        "hodgepodge" => 1,
        "ants" => 64,
        "forest" => 1,
        "sandpile" => 2000,
        "sand" => 8,
        "grayscott" => 32,
        "lenia" => 2,
        "smoothlife" => 1,
        "mnca" => 1,
        "physarum" => 2,
        _ => 1,
    }
}

struct Target {
    view: wgpu::TextureView,
    group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

pub struct Viewer {
    gpu: Gpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    blit: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    srgb: wgpu::Buffer,
    target: Target,
    sim: Sim,
    index: usize,
    seed: u32,
    speed: u32,
    paused: bool,
    zoom: f32,
    fps: f32,
    last_frame: f64,
}

fn now() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now() / 1000.0)
            .unwrap_or(0.0)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        use std::time::Instant;
        static START: OnceLock<Instant> = OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_secs_f64()
    }
}

fn make_target(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    srgb: &wgpu::Buffer,
    width: u32,
    height: u32,
) -> Target {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: srgb.as_entire_binding(),
            },
        ],
    });
    Target {
        view,
        group,
        width,
        height,
    }
}

fn make_sim(gpu: &Gpu, index: usize, seed: u32) -> Sim {
    let entry = &automata::ALL[index];
    let (w, h) = entry.default_size;
    Sim::new(
        gpu,
        automata::build(entry.id, w, h).expect("listed automaton builds"),
        seed,
    )
    .expect("sim")
}

impl Viewer {
    pub async fn new(
        instance: wgpu::Instance,
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        start: &str,
    ) -> Self {
        Self::try_new_target(instance, window, start)
            .await
            .expect("gpu setup")
    }

    pub async fn try_new_target(
        instance: wgpu::Instance,
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        start: &str,
    ) -> Result<Self, String> {
        let surface = instance.create_surface(window).map_err(|e| e.to_string())?;
        let gpu = Gpu::for_surface(instance, &surface, Checks::Safe).await?;
        let caps = surface.get_capabilities(&gpu.adapter);
        // The frame texture already holds display values, so prefer a plain 8-bit surface.
        // Float and 10-bit surfaces can be linear, which would wash the picture out.
        let format = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ]
        .into_iter()
        .find(|f| caps.formats.contains(f))
        .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: 1280,
            height: 720,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: Default::default(),
        };
        surface.configure(&gpu.device, &config);

        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("blit"),
                source: wgpu::ShaderSource::Wgsl(BLIT.into()),
            });
        let blit_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("blit"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("blit"),
                bind_group_layouts: &[Some(&blit_layout)],
                immediate_size: 0,
            });
        let blit = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("blit"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        let srgb = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("srgb"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let flag: [f32; 4] = [if format.is_srgb() { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0];
        gpu.queue
            .write_buffer(&srgb, 0, bytemuck::cast_slice(&flag));
        let target = make_target(&gpu, &blit_layout, &srgb, config.width, config.height);

        let index = automata::ALL
            .iter()
            .position(|e| e.id == start)
            .unwrap_or(0);
        let sim = make_sim(&gpu, index, 1);
        let speed = default_speed(automata::ALL[index].id);
        Ok(Self {
            gpu,
            surface,
            config,
            blit,
            blit_layout,
            srgb,
            target,
            sim,
            index,
            seed: 1,
            speed,
            paused: false,
            zoom: 1.0,
            fps: 0.0,
            last_frame: now(),
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        self.target = make_target(&self.gpu, &self.blit_layout, &self.srgb, width, height);
    }

    pub fn control(&mut self, control: Control) {
        match control {
            Control::TogglePause => self.paused = !self.paused,
            Control::Faster => self.speed = (self.speed * 2).min(1 << 16),
            Control::Slower => self.speed = (self.speed / 2).max(1),
            Control::Reseed => {
                self.seed = self.seed.wrapping_add(1);
                self.sim = make_sim(&self.gpu, self.index, self.seed);
            }
            Control::NextAutomaton => self.select((self.index + 1) % automata::ALL.len()),
            Control::PreviousAutomaton => {
                self.select((self.index + automata::ALL.len() - 1) % automata::ALL.len())
            }
        }
    }

    pub fn select(&mut self, index: usize) {
        self.index = index;
        self.speed = default_speed(automata::ALL[index].id);
        self.zoom = 1.0;
        self.sim = make_sim(&self.gpu, index, self.seed);
    }

    pub fn select_id(&mut self, id: &str) {
        if let Some(i) = automata::ALL.iter().position(|e| e.id == id) {
            self.select(i);
        }
    }

    pub fn set_speed(&mut self, speed: u32) {
        self.speed = speed.clamp(1, 1 << 16);
    }

    pub fn zoom(&mut self, factor: f32) {
        self.zoom = (self.zoom * factor).clamp(0.001, 1.0);
    }

    pub fn stats(&self) -> String {
        let entry = &automata::ALL[self.index];
        let cells = self.sim.program.width as f64 * self.sim.program.height as f64;
        format!(
            "{} | {}x{} = {:.1} M cells | tick {} | {} steps/frame | {:.0} fps | {:.1} G cell updates/s",
            entry.name,
            self.sim.program.width,
            self.sim.program.height,
            cells / 1e6,
            self.sim.tick,
            self.speed,
            self.fps,
            self.speed as f64 * self.fps as f64 * self.sim.program.cells_per_step as f64 / 1e9,
        )
    }

    pub fn frame(&mut self) {
        let t = now();
        let dt = (t - self.last_frame).max(1e-4) as f32;
        self.last_frame = t;
        self.fps = self.fps * 0.9 + (1.0 / dt) * 0.1;

        if !self.paused {
            self.sim.step(&self.gpu, self.speed);
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.config);
                return;
            }
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut camera = Camera::fit(
            self.sim.program.width,
            self.sim.program.height,
            self.target.width,
            self.target.height,
        );
        camera.scale *= self.zoom;
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.sim.render(
            &self.gpu,
            &mut encoder,
            &self.target.view,
            (self.target.width, self.target.height),
            camera,
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.blit);
            pass.set_bind_group(0, &self.target.group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.gpu.queue.submit([encoder.finish()]);
        self.gpu.queue.present(frame);
    }
}
