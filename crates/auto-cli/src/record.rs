//! Headless video recording: render to a texture, read back, pipe to ffmpeg.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use auto_core::{Camera, Gpu, GpuOptions, Sim, automata};
use clap::Parser;

#[derive(Parser)]
pub struct Args {
    pub id: String,
    #[arg(long, default_value_t = 1920)]
    pub width: u32,
    #[arg(long, default_value_t = 1080)]
    pub height: u32,
    /// World size as WxH. Defaults to the automaton's own default.
    #[arg(long)]
    pub world: Option<String>,
    #[arg(long, default_value_t = 20.0)]
    pub seconds: f64,
    #[arg(long, default_value_t = 60)]
    pub fps: u32,
    /// Steps between frames.
    #[arg(long, default_value_t = 1)]
    pub steps_per_frame: u32,
    /// Steps to run before the first frame.
    #[arg(long, default_value_t = 0)]
    pub warmup: u32,
    /// Output cells per pixel at the start and at the end. 0 means fit the world.
    #[arg(long, default_value_t = 0.0)]
    pub zoom_start: f32,
    #[arg(long, default_value_t = 0.0)]
    pub zoom_end: f32,
    #[arg(long, default_value_t = 1)]
    pub seed: u32,
    #[arg(long, default_value_t = 23)]
    pub crf: u32,
    /// Longest one GPU step may hold the card, since the desktop shares it.
    #[arg(long, default_value_t = 50.0)]
    pub max_step_ms: f64,
    /// Record even if it would stall the desktop or nearly fill VRAM.
    #[arg(long)]
    pub force: bool,
    #[arg(long)]
    pub title: Option<String>,
    #[arg(long)]
    pub out: String,
}

pub fn run(opts: GpuOptions, args: Args) {
    let gpu = pollster::block_on(Gpu::headless(opts)).expect("no gpu");
    let entry = automata::find(&args.id);
    let (w, h) = match &args.world {
        Some(s) => {
            let (a, b) = s.split_once('x').expect("world looks like WxH");
            (a.parse().expect("width"), b.parse().expect("height"))
        }
        None => entry.map(|e| e.default_size).unwrap_or((1920, 1080)),
    };
    let limits = crate::guard::Limits {
        max_step_ms: args.max_step_ms,
        force: args.force,
    };
    if let Err(why) = crate::guard::check(&gpu, &args.id, w, h, &limits) {
        eprintln!("refused: {why}");
        std::process::exit(3);
    }
    let program = automata::build(&args.id, w, h).expect("known id");
    let (pw, ph, cells) = (program.width, program.height, program.cells_per_step);
    let name = entry
        .map(|e| e.name.to_string())
        .unwrap_or_else(|| args.id.clone());
    let title = args.title.clone().unwrap_or(name);
    let mut sim = Sim::new(&gpu, program, args.seed).expect("sim");
    gpu.wait();

    let warm = Instant::now();
    let mut left = args.warmup;
    while left > 0 {
        let n = left.min(256);
        sim.step(&gpu, n);
        gpu.wait();
        left -= n;
    }
    if args.warmup > 0 {
        eprintln!(
            "warmup {} steps in {:.2}s",
            args.warmup,
            warm.elapsed().as_secs_f64()
        );
    }

    let (ow, oh) = (args.width, args.height);
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size: wgpu::Extent3d {
            width: ow,
            height: oh,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let bytes_per_row = (ow * 4).next_multiple_of(256);
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame readback"),
        size: bytes_per_row as u64 * oh as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    if let Some(dir) = std::path::Path::new(&args.out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let label = format!(
        "{title}   {pw} x {ph} = {} cells   {} step{} per frame",
        human(pw as u64 * ph as u64),
        args.steps_per_frame,
        if args.steps_per_frame == 1 { "" } else { "s" }
    )
    .replace(':', "\\:")
    .replace('\'', "\u{2019}");
    let font = std::env::var("AUTO_FONT").unwrap_or_else(|_| "DejaVuSans".into());
    let filter = format!(
        "drawbox=x=0:y=ih-44:w=iw:h=44:color=black@0.55:t=fill,drawtext=font='{font}':text='{label}':x=18:y=h-32:fontsize=22:fontcolor=white"
    );
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s",
        ])
        .arg(format!("{ow}x{oh}"))
        .args(["-r", &args.fps.to_string(), "-i", "-", "-vf", &filter])
        .args([
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-threads", "3",
        ])
        .args([
            "-crf",
            &args.crf.to_string(),
            "-maxrate",
            "40M",
            "-bufsize",
            "80M",
            "-movflags",
            "+faststart",
        ])
        .arg(&args.out)
        .stdin(Stdio::piped())
        .spawn()
        .expect("ffmpeg in PATH");
    let mut pipe = ffmpeg.stdin.take().expect("stdin");

    let total = (args.seconds * args.fps as f64) as u32;
    let fit = Camera::fit(pw, ph, ow, oh).scale;
    let z0 = if args.zoom_start > 0.0 {
        args.zoom_start
    } else {
        fit
    };
    let z1 = if args.zoom_end > 0.0 {
        args.zoom_end
    } else {
        fit
    };
    let mut row = vec![0u8; (ow * 4) as usize];
    let mut sim_time = 0.0;
    let started = Instant::now();
    for frame in 0..total {
        let t = frame as f32 / (total.max(2) - 1) as f32;
        // Eased exponential zoom.
        let e = t * t * (3.0 - 2.0 * t);
        let scale = z0 * (z1 / z0).powf(e);
        let cam = Camera {
            center: [pw as f32 / 2.0, ph as f32 / 2.0],
            scale,
        };

        let s = Instant::now();
        sim.step(&gpu, args.steps_per_frame);
        gpu.wait();
        sim_time += s.elapsed().as_secs_f64();

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        sim.render(&gpu, &mut encoder, &view, (ow, oh), cam);
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: ow,
                height: oh,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        gpu.wait();
        {
            let data = readback.get_mapped_range(..).expect("mapped");
            for y in 0..oh as usize {
                let start = y * bytes_per_row as usize;
                row.copy_from_slice(&data[start..start + (ow * 4) as usize]);
                pipe.write_all(&row).expect("ffmpeg pipe");
            }
        }
        readback.unmap();
    }
    drop(pipe);
    ffmpeg.wait().expect("ffmpeg");
    let steps = (total * args.steps_per_frame) as f64;
    eprintln!(
        "{}: {} frames in {:.1}s; sim {:.1} steps/s, {:.3} Gcell/s, {} ticks",
        args.out,
        total,
        started.elapsed().as_secs_f64(),
        steps / sim_time.max(1e-9),
        steps * cells as f64 / sim_time.max(1e-9) / 1e9,
        sim.tick
    );
}

fn human(n: u64) -> String {
    match n {
        n if n >= 1_000_000_000_000 => format!("{:.1} trillion", n as f64 / 1e12),
        n if n >= 1_000_000_000 => format!("{:.1} billion", n as f64 / 1e9),
        n if n >= 1_000_000 => format!("{:.1} million", n as f64 / 1e6),
        n => n.to_string(),
    }
}
