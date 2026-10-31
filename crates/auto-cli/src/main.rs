//! Command line front end: list automata, benchmark them, record videos.

mod bench;
mod cpu;
mod guard;
mod record;

use auto_core::{Checks, Gpu, GpuOptions, automata};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "auto", about = "Very fast cellular automata on wgpu")]
struct Cli {
    /// Adapter name to use, for example 7900 or llvmpipe.
    #[arg(long, global = true)]
    adapter: Option<String>,
    /// Compile shaders without runtime bounds checks (native only).
    #[arg(long, global = true)]
    trusted: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List automata.
    List,
    /// Show the GPU and the limits that matter.
    Probe,
    /// Measure throughput of one kernel at a list of sizes, appending JSON lines.
    Bench(bench::Args),
    /// Record a video through ffmpeg.
    Record(record::Args),
    /// Measure Life on the CPU, and Hashlife on patterns too big for memory.
    Cpu(cpu::Args),
}

fn main() {
    let cli = Cli::parse();
    let opts = GpuOptions {
        adapter_name: cli.adapter,
        checks: if cli.trusted {
            Checks::Trusted
        } else {
            Checks::Safe
        },
    };
    match cli.command {
        Command::List => {
            for e in automata::ALL {
                println!("{:<14} {:<34} {}", e.id, e.name, e.blurb);
            }
        }
        Command::Probe => {
            let gpu = pollster::block_on(Gpu::headless(opts)).expect("no gpu");
            let l = gpu.limits();
            println!("adapter: {:?}", gpu.adapter.get_info());
            println!(
                "max_storage_buffer_binding_size: {}",
                l.max_storage_buffer_binding_size
            );
            println!("max_buffer_size: {}", l.max_buffer_size);
            println!(
                "max_storage_buffers_per_shader_stage: {}",
                l.max_storage_buffers_per_shader_stage
            );
            println!(
                "max_compute_workgroup_storage_size: {}",
                l.max_compute_workgroup_storage_size
            );
            println!(
                "max_compute_invocations_per_workgroup: {}",
                l.max_compute_invocations_per_workgroup
            );
            println!(
                "max_compute_workgroups_per_dimension: {}",
                l.max_compute_workgroups_per_dimension
            );
            println!(
                "min_uniform_buffer_offset_alignment: {}",
                l.min_uniform_buffer_offset_alignment
            );
            println!("int64: {}", gpu.has_int64());
        }
        Command::Bench(args) => bench::run(opts, args),
        Command::Record(args) => record::run(opts, args),
        Command::Cpu(args) => cpu::run(args),
    }
}
