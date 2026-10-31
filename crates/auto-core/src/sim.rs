//! Turns a [`Program`] into GPU objects and runs it.

use std::collections::HashMap;

use crate::gpu::Gpu;
use crate::program::{Bind, Field, Pass, Program, WGSL_COMMON};

const FRAME_BYTES: u64 = 16;
/// Dispatches that can run in one batch before the frame ring has to be flushed.
const RING_ENTRIES: u64 = 4096;

const FRAME_WGSL: &str = r#"
struct Frame { tick: u32, phase: u32, seed: u32, cursor: u32 }
@group(0) @binding(0) var<uniform> frame: Frame;
"#;

const VIEW_WGSL: &str = r#"
struct View { origin: vec2f, scale: f32, samples: u32, out: vec2u, world: vec2u }
@group(1) @binding(0) var<uniform> V: View;
@group(1) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn view_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= V.out.x || id.y >= V.out.y) { return; }
    var acc = vec3f(0.0);
    let n = V.samples;
    for (var sy = 0u; sy < n; sy++) {
        for (var sx = 0u; sx < n; sx++) {
            let sub = (vec2f(f32(sx), f32(sy)) + 0.5) / f32(n);
            let p = V.origin + (vec2f(id.xy) + sub) * V.scale;
            // Outside the world stays black instead of showing the torus twice.
            if (p.x >= 0.0 && p.y >= 0.0 && p.x < f32(V.world.x) && p.y < f32(V.world.y)) {
                let c = color(min(u32(p.x), V.world.x - 1u), min(u32(p.y), V.world.y - 1u));
                // Average in light rather than in gamma, or sparse bright cells go dark when zoomed out.
                acc += c * c;
            }
        }
    }
    textureStore(out_tex, id.xy, vec4f(sqrt(acc / f32(n * n)), 1.0));
}
"#;

/// What part of the world to show. `scale` is cells per output pixel.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub center: [f32; 2],
    pub scale: f32,
}

impl Camera {
    pub fn fit(width: u32, height: u32, out_w: u32, out_h: u32) -> Self {
        let scale = (width as f32 / out_w as f32).max(height as f32 / out_h as f32);
        Self {
            center: [width as f32 / 2.0, height as f32 / 2.0],
            scale,
        }
    }
}

struct Compiled {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    pass: Pass,
}

pub struct Sim {
    pub program: Program,
    pub tick: u64,
    pub seed: u32,
    buffers: HashMap<&'static str, [Option<wgpu::Buffer>; 2]>,
    front: HashMap<&'static str, u8>,
    ring: wgpu::Buffer,
    ring_stride: u64,
    params: wgpu::Buffer,
    init: Vec<Compiled>,
    step: Vec<Compiled>,
    groups: HashMap<(usize, u8, u32), wgpu::BindGroup>,
    view: Option<ViewPipe>,
}

struct ViewPipe {
    compiled: Compiled,
    view_layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
}

fn preamble(program: &Program, binds: &[Bind]) -> String {
    let mut s = String::new();
    s.push_str(FRAME_WGSL);
    s.push_str(&format!("struct Params {{ {} }}\n", program.params_wgsl));
    s.push_str("@group(0) @binding(1) var<uniform> P: Params;\n");
    for (i, bind) in binds.iter().enumerate() {
        let field = program
            .fields
            .iter()
            .find(|f| f.name == bind.field())
            .expect("bind names a field");
        let (access, var) = match bind {
            Bind::Read(_, v) => ("read", *v),
            Bind::Write(_, v) | Bind::ReadWrite(_, v) => ("read_write", *v),
        };
        s.push_str(&format!(
            "@group(0) @binding({}) var<storage, {access}> {var}: array<{}>;\n",
            i + 2,
            field.elem
        ));
    }
    s.push_str(WGSL_COMMON);
    s
}

/// The full WGSL for a pass: bindings, shared helpers, then the pass itself.
pub fn wgsl_for(program: &Program, pass: &Pass) -> String {
    let mut source = preamble(program, &pass.binds);
    source.push_str(&pass.source);
    source
}

fn layout_entries(binds: &[Bind]) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(FRAME_BYTES),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ];
    for (i, bind) in binds.iter().enumerate() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: i as u32 + 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: matches!(bind, Bind::Read(..)),
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    entries
}

fn compile(
    gpu: &Gpu,
    program: &Program,
    pass: &Pass,
    entry: &str,
    extra: Option<&wgpu::BindGroupLayout>,
) -> Compiled {
    let device = &gpu.device;
    let source = wgsl_for(program, pass);
    let module = gpu.shader(pass.label, &source);
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(pass.label),
        entries: &layout_entries(&pass.binds),
    });
    let mut layouts = vec![Some(&layout)];
    if let Some(extra) = extra {
        layouts.push(Some(extra));
    }
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(pass.label),
        bind_group_layouts: &layouts,
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(pass.label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            // Kernels initialise what they read. WebGPU in the browser always zeroes.
            zero_initialize_workgroup_memory: gpu.checks != crate::Checks::Trusted,
            ..Default::default()
        },
        cache: None,
    });
    Compiled {
        pipeline,
        layout,
        pass: pass.clone(),
    }
}

impl Sim {
    pub fn new(gpu: &Gpu, program: Program, seed: u32) -> Result<Self, String> {
        let device = &gpu.device;
        let limits = device.limits();
        let mut buffers = HashMap::new();
        let mut front = HashMap::new();
        for field in &program.fields {
            if field.bytes()
                > limits
                    .max_storage_buffer_binding_size
                    .min(limits.max_buffer_size)
            {
                return Err(format!(
                    "field {} needs {} bytes, the GPU allows {} per buffer",
                    field.name,
                    field.bytes(),
                    limits.max_storage_buffer_binding_size
                ));
            }
            let make = |suffix: &str| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{}{suffix}", field.name)),
                    size: field.bytes(),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let pair = [Some(make("/a")), field.double.then(|| make("/b"))];
            buffers.insert(field.name, pair);
            front.insert(field.name, 0u8);
        }

        let ring_stride = (limits.min_uniform_buffer_offset_alignment as u64).max(FRAME_BYTES);
        let ring = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame ring"),
            size: ring_stride * RING_ENTRIES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_bytes = if program.params.is_empty() {
            vec![0u8; 16]
        } else {
            program.params.clone()
        };
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: (params_bytes.len() as u64).next_multiple_of(16),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&params, 0, &params_bytes);

        let init = program
            .init
            .iter()
            .map(|p| compile(gpu, &program, p, "main", None))
            .collect();
        let step = program
            .step
            .iter()
            .map(|p| compile(gpu, &program, p, "main", None))
            .collect();

        let mut sim = Self {
            program,
            tick: 0,
            seed,
            buffers,
            front,
            ring,
            ring_stride,
            params,
            init,
            step,
            groups: HashMap::new(),
            view: None,
        };
        sim.run_init(gpu);
        Ok(sim)
    }

    /// Fill the fields using the program's init passes, entirely on the GPU, so
    /// huge worlds never need a CPU side copy.
    pub fn run_init(&mut self, gpu: &Gpu) {
        self.tick = 0;
        for v in self.front.values_mut() {
            *v = 0;
        }
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.encode(gpu, &mut encoder, true, 1);
        gpu.queue.submit([encoder.finish()]);
    }

    /// Advance by `steps` and submit. Returns the generations advanced.
    pub fn step(&mut self, gpu: &Gpu, steps: u32) -> u64 {
        let per_step: u64 = self.step.iter().map(|c| c.pass.repeat as u64).sum();
        let max_steps = (RING_ENTRIES / per_step.max(1)).max(1) as u32;
        let mut left = steps;
        while left > 0 {
            let batch = left.min(max_steps);
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            self.encode(gpu, &mut encoder, false, batch);
            gpu.queue.submit([encoder.finish()]);
            left -= batch;
        }
        steps as u64 * self.program.generations_per_step as u64
    }

    fn encode(&mut self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, init: bool, steps: u32) {
        let per_step: usize = if init {
            self.init.iter().map(|c| c.pass.repeat as usize).sum()
        } else {
            self.step.iter().map(|c| c.pass.repeat as usize).sum()
        };
        let total = per_step * steps as usize;
        let stride = self.ring_stride as usize;
        let mut frames = vec![0u8; total * stride];
        let mut slot = 0usize;

        // Decide the bind group order first so the borrow of self stays simple.
        let mut plan = Vec::with_capacity(total);
        let count = if init {
            self.init.len()
        } else {
            self.step.len()
        };
        for s in 0..steps as u64 {
            for pass_index in 0..count {
                let compiled = if init {
                    &self.init[pass_index]
                } else {
                    &self.step[pass_index]
                };
                let repeat = compiled.pass.repeat;
                let swap = compiled.pass.swap.clone();
                for phase in 0..repeat {
                    // The cursor is the tick modulo the world's height, computed here because
                    // the shader's 32-bit tick would give the wrong answer once it wraps.
                    let tick = self.tick + s;
                    let frame = [
                        tick as u32,
                        phase,
                        self.seed,
                        (tick % self.program.height as u64) as u32,
                    ];
                    frames[slot * stride..slot * stride + 16]
                        .copy_from_slice(bytemuck::cast_slice(&frame));
                    let key = self.front_mask();
                    self.ensure_group(gpu, init, pass_index, key);
                    plan.push((pass_index, key, (slot * stride) as u32));
                    slot += 1;
                    for field in &swap {
                        let f = self.front.get_mut(field).expect("swap names a field");
                        *f ^= 1;
                    }
                }
            }
        }
        gpu.queue.write_buffer(&self.ring, 0, &frames);

        let kind = u8::from(init);
        let compiled = if init { &self.init } else { &self.step };
        let mut pass = encoder.begin_compute_pass(&Default::default());
        let mut current = usize::MAX;
        for (pass_index, key, offset) in plan {
            if current != pass_index {
                pass.set_pipeline(&compiled[pass_index].pipeline);
                current = pass_index;
            }
            pass.set_bind_group(0, &self.groups[&(pass_index, kind, key)], &[offset]);
            let [x, y, z] = compiled[pass_index].pass.dispatch;
            pass.dispatch_workgroups(x, y, z);
        }
        drop(pass);
        if !init {
            self.tick += steps as u64;
        }
    }

    fn front_mask(&self) -> u32 {
        let mut mask = 0;
        for (i, field) in self.program.fields.iter().enumerate() {
            mask |= (self.front[field.name] as u32) << i;
        }
        mask
    }

    fn buffer(&self, field: &Field, back: bool) -> &wgpu::Buffer {
        let which = self.front[field.name] ^ u8::from(back);
        self.buffers[field.name][which as usize]
            .as_ref()
            .expect("double buffer exists")
    }

    fn make_group(
        &self,
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        binds: &[Bind],
    ) -> wgpu::BindGroup {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.ring,
                    offset: 0,
                    size: wgpu::BufferSize::new(FRAME_BYTES),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.params.as_entire_binding(),
            },
        ];
        for (i, bind) in binds.iter().enumerate() {
            let field = self
                .program
                .fields
                .iter()
                .find(|f| f.name == bind.field())
                .expect("bind names a field");
            let buffer = self.buffer(field, matches!(bind, Bind::Write(..)));
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 2,
                resource: buffer.as_entire_binding(),
            });
        }
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        })
    }

    fn ensure_group(&mut self, gpu: &Gpu, init: bool, pass_index: usize, key: u32) {
        let full_key = (pass_index, u8::from(init), key);
        if self.groups.contains_key(&full_key) {
            return;
        }
        let compiled = if init {
            &self.init[pass_index]
        } else {
            &self.step[pass_index]
        };
        let group = self.make_group(gpu, &compiled.layout, &compiled.pass.binds);
        self.groups.insert(full_key, group);
    }

    /// Render the current state into `target`, a rgba8unorm storage texture.
    pub fn render(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size: (u32, u32),
        camera: Camera,
    ) {
        if self.view.is_none() {
            self.view = Some(self.make_view(gpu));
        }
        let vp = self.view.as_ref().expect("just built");
        let (w, h) = size;
        let origin = [
            camera.center[0] - w as f32 * camera.scale / 2.0,
            camera.center[1] - h as f32 * camera.scale / 2.0,
        ];
        let samples = (camera.scale.ceil() as u32).clamp(1, 4);
        let data: [u32; 8] = [
            origin[0].to_bits(),
            origin[1].to_bits(),
            camera.scale.to_bits(),
            samples,
            w,
            h,
            self.program.width,
            self.program.height,
        ];
        gpu.queue
            .write_buffer(&vp.uniform, 0, bytemuck::cast_slice(&data));

        // The render shows the latest tick, in the frame ring's last slot.
        let frame = [
            self.tick as u32,
            0u32,
            self.seed,
            (self.tick % self.program.height as u64) as u32,
        ];
        let offset = self.ring_stride * (RING_ENTRIES - 1);
        gpu.queue
            .write_buffer(&self.ring, offset, bytemuck::cast_slice(&frame));

        let group0 = self.make_group(gpu, &vp.compiled.layout, &vp.compiled.pass.binds);
        let group1 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view"),
            layout: &vp.view_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vp.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(target),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&vp.compiled.pipeline);
        pass.set_bind_group(0, &group0, &[offset as u32]);
        pass.set_bind_group(1, &group1, &[]);
        pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
    }

    fn make_view(&self, gpu: &Gpu) -> ViewPipe {
        let device = &gpu.device;
        let view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let source = format!("{}\n{}", self.program.view, VIEW_WGSL);
        let pass = Pass::new("view", source, self.program.view_binds.clone(), [1, 1, 1]);
        let compiled = compile(gpu, &self.program, &pass, "view_main", Some(&view_layout));
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ViewPipe {
            compiled,
            view_layout,
            uniform,
        }
    }

    /// Replace the front buffer of a field with CPU data. Used by tests and pattern loaders.
    pub fn write_field(&self, gpu: &Gpu, name: &str, bytes: &[u8]) {
        let field = self
            .program
            .fields
            .iter()
            .find(|f| f.name == name)
            .expect("known field");
        gpu.queue.write_buffer(self.buffer(field, false), 0, bytes);
    }

    /// Read the front buffer of a field back to the CPU.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_field(&self, gpu: &Gpu, name: &str) -> Vec<u8> {
        let field = self
            .program
            .fields
            .iter()
            .find(|f| f.name == name)
            .expect("known field");
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: field.bytes(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(self.buffer(field, false), 0, &staging, 0, field.bytes());
        gpu.queue.submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        gpu.wait();
        let out = staging.get_mapped_range(..).expect("mapped").to_vec();
        staging.unmap();
        out
    }
}
