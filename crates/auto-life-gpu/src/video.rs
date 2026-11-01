//! Records the world to video and flies the camera to wherever is most interesting.

use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};

use crate::report::{Palette, sample};
use crate::world::{Access2, Frame, World};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layout {
    Genome,
    Diet,
    Split,
}

pub struct RecorderOptions {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub layout: Layout,
    /// Cells per pixel to zoom in to. 0 keeps the whole world in view.
    pub zoom: f32,
    pub seconds_per_shot: f32,
    pub label: String,
    pub threads: u32,
}

struct Camera {
    x: f32,
    y: f32,
    scale: f32,
}

pub struct Recorder {
    opts: RecorderOptions,
    ffmpeg: Child,
    pipe: Option<ChildStdin>,
    texture: wgpu::Texture,
    readback: wgpu::Buffer,
    bytes_per_row: u32,
    pipe_view: wgpu::ComputePipeline,
    group0: wgpu::BindGroup,
    group1_layout: wgpu::BindGroupLayout,
    view_uniform: wgpu::Buffer,
    bake_uniform: wgpu::Buffer,
    palette: Palette,
    cam: Camera,
    goal: Camera,
    frames: u64,
    next_shot: u64,
    visited: Vec<(f32, f32)>,
    wide_next: bool,
    row: Vec<u8>,
}

const VIEW_STRIDE: u64 = 256;

impl Recorder {
    pub fn new(world: &World, path: &str, opts: RecorderOptions) -> Result<Recorder, String> {
        let device = &world.gpu.device;
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(32),
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
        let (pipe_view, group0) = world.extra_pipe(
            "view.wgsl",
            "main",
            &[
                ("occ", Access2::Read),
                ("cell", Access2::Read),
                ("stat", Access2::Read),
                ("fa", Access2::Read),
                ("fb", Access2::Read),
                ("ctr", Access2::ReadWrite),
            ],
            Some(&group1_layout),
        )?;
        let (w, h) = (opts.width, opts.height);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let bytes_per_row = (w * 4).next_multiple_of(256);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame readback"),
            size: bytes_per_row as u64 * h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let view_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view uniform"),
            size: VIEW_STRIDE * 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bake_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake uniform"),
            size: (12 + 4 + 1) * 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(dir) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let label = opts.label.replace(':', "\\:").replace('\'', "\u{2019}");
        let font = std::env::var("AUTO_FONT").unwrap_or_else(|_| "DejaVuSans".into());
        let filter = format!(
            "drawbox=x=0:y=ih-36:w=iw:h=36:color=black@0.55:t=fill,drawtext=font='{font}':text='{label}':x=14:y=h-26:fontsize=18:fontcolor=white"
        );
        let mut ffmpeg = Command::new("ffmpeg")
            .args([
                "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s",
            ])
            .arg(format!("{w}x{h}"))
            .args(["-r", &opts.fps.to_string(), "-i", "-", "-vf", &filter])
            .args([
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            ])
            .args([
                "-threads",
                &opts.threads.to_string(),
                "-crf",
                "22",
                "-movflags",
                "+faststart",
            ])
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("ffmpeg: {e}"))?;
        let pipe = ffmpeg.stdin.take();
        let size = world.cfg.size as f32;
        let fit = Self::fit_scale(world.cfg.size, &opts);
        Ok(Recorder {
            row: vec![0u8; (w * 4) as usize],
            opts,
            ffmpeg,
            pipe,
            texture,
            readback,
            bytes_per_row,
            pipe_view,
            group0,
            group1_layout,
            view_uniform,
            bake_uniform,
            palette: Palette::default(),
            cam: Camera {
                x: size / 2.0,
                y: size / 2.0,
                scale: fit,
            },
            goal: Camera {
                x: size / 2.0,
                y: size / 2.0,
                scale: fit,
            },
            frames: 0,
            next_shot: 0,
            visited: vec![],
            wide_next: true,
        })
    }

    fn panel_width(opts: &RecorderOptions) -> u32 {
        if opts.layout == Layout::Split {
            (opts.width - 4) / 2
        } else {
            opts.width
        }
    }

    fn fit_scale(size: u32, opts: &RecorderOptions) -> f32 {
        size as f32 / Self::panel_width(opts).min(opts.height) as f32
    }

    /// Picks the next place to look: lots of different genomes at once, with some of them eating others.
    fn choose_shot(&mut self, world: &mut World) {
        let fit = Self::fit_scale(world.cfg.size, &self.opts);
        let size = world.cfg.size as f32;
        if self.opts.zoom <= 0.0 {
            return;
        }
        if self.wide_next {
            self.goal = Camera {
                x: size / 2.0,
                y: size / 2.0,
                scale: fit,
            };
            self.wide_next = false;
            return;
        }
        let tiles = world.census();
        let side = (world.cfg.size / world.cfg.tile()) as usize;
        let tile = world.cfg.tile() as f32;
        let mut best = (-1.0f32, 0usize);
        for (i, t) in tiles.iter().enumerate() {
            let n = t[0];
            if n < tile * tile * 0.02 {
                continue;
            }
            let hunt = t[2] / n;
            let moving = t[3] / n;
            let mut spread = 0.0;
            for c in 0..3 {
                let mean = t[4 + c] / n;
                spread += (t[8 + c] / n - mean * mean).max(0.0);
            }
            let (x, y) = (
                ((i % side) as f32 + 0.5) * tile,
                ((i / side) as f32 + 0.5) * tile,
            );
            // Don't keep returning to the same spot.
            let near = self
                .visited
                .iter()
                .map(|(vx, vy)| ((vx - x).powi(2) + (vy - y).powi(2)).sqrt())
                .fold(f32::MAX, f32::min);
            let fresh = (near / (tile * 6.0)).min(1.0);
            let score = spread.sqrt() * (0.1 + hunt) * (0.2 + moving) * fresh * (n.sqrt() / tile);
            if score > best.0 {
                best = (score, i);
            }
        }
        if best.0 >= 0.0 {
            let i = best.1;
            let (x, y) = (
                ((i % side) as f32 + 0.5) * tile,
                ((i / side) as f32 + 0.5) * tile,
            );
            self.visited.push((x, y));
            self.goal = Camera {
                x,
                y,
                scale: self.opts.zoom.min(fit),
            };
            eprintln!("camera: tile at {x:.0},{y:.0} scored {:.3}", best.0);
        }
        self.wide_next = self.visited.len() % 4 == 0;
    }

    pub fn frame(&mut self, world: &mut World) {
        world.flush();
        if self.frames % 60 == 0 {
            let s = sample(world, 2000);
            self.palette.update(&s.cells);
        }
        world
            .gpu
            .queue
            .write_buffer(&self.bake_uniform, 0, &self.palette.bytes());
        world.bake(&self.bake_uniform);
        if self.frames >= self.next_shot {
            self.choose_shot(world);
            self.next_shot =
                self.frames + (self.opts.seconds_per_shot * self.opts.fps as f32) as u64;
        }

        // Ease towards the goal, panning less when zoomed in far.
        let k = 0.035;
        self.cam.scale =
            (self.cam.scale.ln() + (self.goal.scale.ln() - self.cam.scale.ln()) * k).exp();
        self.cam.x += (self.goal.x - self.cam.x) * k;
        self.cam.y += (self.goal.y - self.cam.y) * k;

        let (ow, oh) = (self.opts.width, self.opts.height);
        let pw = Self::panel_width(&self.opts);
        let panels: Vec<(u32, u32)> = match self.opts.layout {
            Layout::Genome => vec![(0, 0)],
            Layout::Diet => vec![(1, 0)],
            Layout::Split => vec![(0, 0), (1, pw + 4)],
        };
        let scale = self.cam.scale;
        let samples = (scale.ceil() as u32).clamp(1, 4);
        let origin = [
            self.cam.x - pw as f32 * scale / 2.0,
            self.cam.y - oh as f32 * scale / 2.0,
        ];
        let view = self.texture.create_view(&Default::default());
        let group1 = world
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("view"),
                layout: &self.group1_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.view_uniform,
                            offset: 0,
                            size: wgpu::BufferSize::new(32),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                ],
            });
        for (k, (mode, x0)) in panels.iter().enumerate() {
            let mut u = vec![];
            u.extend_from_slice(bytemuck::cast_slice(&[origin[0], origin[1], scale]));
            u.extend_from_slice(bytemuck::cast_slice(&[samples, pw, oh, *mode, *x0]));
            world
                .gpu
                .queue
                .write_buffer(&self.view_uniform, k as u64 * VIEW_STRIDE, &u);
        }
        // Short bands, so drawing a frame doesn't hold the GPU either.
        let band = (oh / 4).max(16).next_multiple_of(16);
        let mut row = 0;
        while row < oh {
            let frame = Frame {
                row0: row,
                ..world.current_frame()
            };
            world.write_frame(frame);
            let mut enc = world.gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipe_view);
                pass.set_bind_group(0, &self.group0, &[0]);
                for k in 0..panels.len() {
                    pass.set_bind_group(1, &group1, &[(k as u64 * VIEW_STRIDE) as u32]);
                    pass.dispatch_workgroups(pw.div_ceil(16), band.min(oh - row).div_ceil(16), 1);
                }
            }
            world.gpu.queue.submit([enc.finish()]);
            world.gpu.wait();
            row += band;
        }
        let mut enc = world.gpu.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: ow,
                height: oh,
                depth_or_array_layers: 1,
            },
        );
        world.gpu.queue.submit([enc.finish()]);
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        world.gpu.wait();
        {
            let data = self.readback.get_mapped_range(..).expect("mapped");
            let pipe = self.pipe.as_mut().expect("ffmpeg stdin");
            for y in 0..oh as usize {
                let start = y * self.bytes_per_row as usize;
                self.row
                    .copy_from_slice(&data[start..start + (ow * 4) as usize]);
                pipe.write_all(&self.row).expect("ffmpeg pipe");
            }
        }
        self.readback.unmap();
        self.frames += 1;
    }

    pub fn finish(mut self) {
        drop(self.pipe.take());
        let _ = self.ffmpeg.wait();
    }
}
