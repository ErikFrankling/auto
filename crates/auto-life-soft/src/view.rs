//! Looking at a world: what cells and squares can be coloured by, and the pass that draws them.
//! Reads the world's buffers and changes nothing.

use bytemuck::{Pod, Zeroable};

use crate::dump::Palette;
use crate::world::{Access, World};

/// What cells can be coloured by, as (name, what it means, the shader's number).
pub const CELL_VIEWS: [(&str, &str, u32); 22] = [
    ("Genome", "A fixed random projection of the weights. Similar genomes get similar colours.", 0),
    ("Energy", "What it holds.", 1),
    ("Main action", "What most of its budget goes to.", 2),
    ("Contacts", "How many cells it's touching. Surface against inside.", 3),
    ("Age", "Ticks since it was born.", 4),
    ("Founder", "Which rained-in cell it descends from.", 5),
    ("Generation", "How many births back to its founder.", 6),
    ("Stick", "How hard it's sticking now.", 7),
    ("Womb", "Energy put by for a child.", 8),
    ("Mutation rate", "Its own, inherited with a nudge.", 9),
    ("Share: rest", "Share of the budget for resting.", 16),
    ("Share: thrust", "Share of the budget for moving itself.", 17),
    ("Share: eat food", "Share of the budget for eating food.", 18),
    ("Share: eat meat", "Share of the budget for eating meat.", 19),
    ("Share: grow", "Share of the budget for filling the womb.", 20),
    ("Share: scent", "Share of the budget for emitting scent.", 21),
    ("Share: heat", "Share of the budget for making heat on purpose.", 22),
    ("Share: bite", "Share of the budget for biting.", 23),
    ("Share: give", "Share of the budget for giving energy away.", 24),
    ("Share: enzyme", "Share of the budget for making enzyme.", 25),
    ("Share: stick", "Share of the budget for sticking.", 26),
    ("Share: shove", "Share of the budget for shoving.", 27),
];

/// What can be shown under the cells, as (name, what it means).
pub const FIELD_VIEWS: [(&str, &str); 11] = [
    ("Nothing", "Black, with only the rock."),
    ("Food", "Grows where the sun shines, eaten by cells."),
    ("Meat", "What dead bodies leave."),
    ("Heat", "Shed by everything a cell does. Hurts thin cells most."),
    ("Enzyme", "Secreted by cells. Unlocks food."),
    ("Scent 1", "Lingers and spreads slowly."),
    ("Scent 2", "Spreads a little faster."),
    ("Scent 3", "Spreads fast and fades quickly."),
    ("Scent 4", "Spreads fastest and fades within a few ticks."),
    ("All scents", "Each scent in its own colour, added together."),
    ("Sun", "How rich each square is."),
];

pub const SHOW_CELLS: u32 = 1;

/// What to draw. Laid out as `ViewU` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ViewParams {
    pub origin: [f32; 2],
    pub scale: f32,
    pub cell_mode: u32,
    pub out: [u32; 2],
    pub row0: u32,
    pub field_mode: u32,
    pub field_opacity: f32,
    pub cell_opacity: f32,
    pub age_top: f32,
    pub gen_top: f32,
    pub center: [f32; 2],
    pub e_top: f32,
    pub flags: u32,
    pub hi: [f32; 8],
    pub axes: [f32; 48],
    pub mean: [f32; 16],
    pub gain: [f32; 4],
}

impl ViewParams {
    /// Everything but where to look, at sensible settings.
    pub fn new(palette: &Palette, tops: [f32; 8]) -> ViewParams {
        let mut axes = [0.0; 48];
        for a in 0..3 {
            axes[a * 16..(a + 1) * 16].copy_from_slice(&palette.axes[a]);
        }
        ViewParams {
            origin: [0.0, 0.0],
            scale: 1.0,
            cell_mode: 0,
            out: [0, 0],
            row0: 0,
            field_mode: 0,
            field_opacity: 0.8,
            cell_opacity: 1.0,
            age_top: 5000.0,
            gen_top: 200.0,
            center: [0.0, 0.0],
            e_top: 20.0,
            flags: SHOW_CELLS,
            hi: tops,
            axes,
            mean: palette.mean,
            gain: [palette.gain[0], palette.gain[1], palette.gain[2], 0.0],
        }
    }
}

const STRIDE: u64 = 1024;
/// A frame goes to the GPU in bands this tall, so drawing doesn't hold it for long.
const BAND: u32 = 256;
const MAX_BANDS: u64 = 32;

/// The layout of the view pass's own bindings. Shared by every world's pipeline.
pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("soft view"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ViewParams>() as u64),
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
    })
}

/// The view pass for one world. Holds its buffers alive, so drop it before making a new world.
#[derive(Clone)]
pub struct ViewPipe {
    pipeline: wgpu::ComputePipeline,
    groups: [wgpu::BindGroup; 2],
    pub size: u32,
}

impl ViewPipe {
    pub fn new(world: &World, layout: &wgpu::BindGroupLayout) -> Result<ViewPipe, String> {
        let source = world.cfg.source("view.wgsl")?;
        let (pipeline, groups) = world.extra_pipe(
            "view.wgsl",
            &source,
            "main",
            &[
                ("bb", "BinBody", Access::R),
                ("cell", "Cell", Access::R),
                ("plan", "Plan", Access::R),
                ("sketch", "vec4u", Access::R),
                ("terrain", "u32", Access::R),
                ("fa0", "vec4i", Access::R),
                ("fb0", "vec4i", Access::R),
            ],
            Some(layout),
        )?;
        Ok(ViewPipe {
            pipeline,
            groups,
            size: world.cfg.size,
        })
    }
}

/// A texture the view pass draws into.
pub struct Canvas {
    pub texture: wgpu::Texture,
    /// For the shader to write.
    pub write: wgpu::TextureView,
    /// For showing on screen, with the colours read as they were meant.
    pub show: wgpu::TextureView,
    group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
}

impl Canvas {
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, width: u32, height: u32) -> Canvas {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("soft world view"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });
        let write = texture.create_view(&Default::default());
        let show = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("soft world view"),
            size: STRIDE * MAX_BANDS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soft world view"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<ViewParams>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&write),
                },
            ],
        });
        Canvas {
            texture,
            write,
            show,
            group,
            uniform,
            width,
            height,
        }
    }

    /// Draws a frame, in short submissions. Doesn't wait for the GPU. `flip` says which of the
    /// two field buffers is current.
    pub fn draw(&self, device: &wgpu::Device, queue: &wgpu::Queue, pipe: &ViewPipe, flip: usize, params: ViewParams) {
        let band = BAND.max(self.height.div_ceil(MAX_BANDS as u32));
        let mut k = 0u64;
        let mut row = 0;
        let mut p = params;
        p.out = [self.width, self.height];
        while row < self.height {
            p.row0 = row;
            queue.write_buffer(&self.uniform, k * STRIDE, bytemuck::bytes_of(&p));
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipe.pipeline);
                pass.set_bind_group(0, &pipe.groups[flip], &[0]);
                pass.set_bind_group(1, &self.group, &[(k * STRIDE) as u32]);
                pass.dispatch_workgroups(self.width.div_ceil(16), band.min(self.height - row).div_ceil(16), 1);
            }
            queue.submit([enc.finish()]);
            row += band;
            k += 1;
        }
    }

    /// Reads the picture back, as RGBA rows. For tests and tools.
    pub fn read(&self, gpu: &auto_core::Gpu) -> Vec<u8> {
        let row = (self.width * 4).next_multiple_of(256);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read picture"),
            size: row as u64 * self.height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([enc.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        gpu.wait();
        let view = staging.get_mapped_range(..).expect("mapped");
        let mut out = Vec::with_capacity((self.width * self.height * 4) as usize);
        for y in 0..self.height as usize {
            let at = y * row as usize;
            out.extend_from_slice(&view[at..at + self.width as usize * 4]);
        }
        drop(view);
        staging.unmap();
        out
    }
}
