//! Runs a thermodynamic life world on the CPU and logs what happens.
//!   auto-life --size 128 --ticks 20000 --out run.jsonl
//!   auto-life --size 256 --ticks 60000 --video run.mp4 --frame-every 40
//!   auto-life --size 256 --ticks 60000 --ablate-at 60000 --ablate-ticks 4000

use auto_life::{Config, World};
use clap::Parser;
use serde_json::json;
use std::io::Write;
use std::process::{Command, Stdio};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 128)]
    size: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long, default_value_t = 20000)]
    ticks: u64,
    #[arg(long, default_value_t = 2000)]
    log_every: u64,
    #[arg(long)]
    out: Option<String>,
    #[arg(long, default_value_t = 0.05)]
    sun: f32,
    #[arg(long, default_value_t = 0.4)]
    compat: f32,
    #[arg(long, default_value_t = 20.0)]
    bite_power: f32,
    #[arg(long, default_value_t = 0.4)]
    bite_yield: f32,
    #[arg(long, default_value_t = 0.6)]
    meat_yield: f32,
    #[arg(long, default_value_t = 0.03)]
    commitment: f32,
    #[arg(long, default_value_t = 200)]
    rain_every: u32,
    #[arg(long, default_value_t = 0.0)]
    bloom: f32,
    #[arg(long, default_value_t = 8000)]
    season: u32,
    #[arg(long)]
    video: Option<String>,
    #[arg(long, default_value_t = 40)]
    frame_every: u64,
    #[arg(long, default_value_t = 0)]
    video_from: u64,
    #[arg(long, default_value_t = 2)]
    scale: usize,
    /// Comma list of things to switch off: pheromones,kin,voice,bite,heat,give,enzyme,sex,share,memory,rock,seasons
    #[arg(long, default_value = "")]
    off: String,
    /// At this tick, copy the world and run each copy with one ability removed.
    #[arg(long, default_value_t = 0)]
    ablate_at: u64,
    #[arg(long, default_value_t = 4000)]
    ablate_ticks: u64,
}

fn switch(cfg: &mut Config, name: &str) {
    match name {
        "pheromones" => cfg.on.pheromones = false,
        "kin" => cfg.on.kin = false,
        "voice" => cfg.on.voice = false,
        "bite" => cfg.on.bite = false,
        "heat" => cfg.on.heat = false,
        "give" => cfg.on.give = false,
        "enzyme" => cfg.on.enzyme = false,
        "sex" => cfg.on.sex = false,
        "share" => cfg.on.share = false,
        "memory" => cfg.on.memory = false,
        "rock" => cfg.rock = false,
        "exposure" => cfg.exposure = false,
        "odour" => cfg.odour = 0.0,
        "seasons" => cfg.season_ticks = 0,
        "" | "none" => {}
        other => panic!("unknown switch {other}"),
    }
}

fn main() {
    let args = Args::parse();
    let mut cfg = Config { size: args.size, seed: args.seed, sun: args.sun, compat: args.compat, bite_power: args.bite_power, bite_yield: args.bite_yield, meat_yield: args.meat_yield, commitment: args.commitment, rain_every: args.rain_every, bloom: args.bloom, season_ticks: args.season, ..Config::default() };
    for name in args.off.split(',') {
        switch(&mut cfg, name);
    }
    let mut world = World::new(cfg);
    let mut log: Box<dyn Write> = match &args.out {
        Some(p) => Box::new(std::fs::File::create(p).unwrap()),
        None => Box::new(std::io::stdout()),
    };
    let mut video = args.video.as_ref().map(|path| {
        let (w, h, _) = world.render(args.scale);
        Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{w}x{h}"), "-r", "30", "-i", "-"])
            .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22"])
            .args(["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2", path])
            .stdin(Stdio::piped())
            .spawn()
            .expect("ffmpeg")
    });
    let started = std::time::Instant::now();
    for t in 1..=args.ticks {
        world.step();
        if let Some(ff) = &mut video {
            if t >= args.video_from && t % args.frame_every == 0 {
                let (_, _, img) = world.render(args.scale);
                ff.stdin.as_mut().unwrap().write_all(&img).unwrap();
            }
        }
        if t % args.log_every == 0 || t == args.ticks {
            let mut r = world.report();
            r["seconds"] = json!(started.elapsed().as_secs_f32());
            r["seed"] = json!(args.seed);
            writeln!(log, "{r}").unwrap();
            log.flush().unwrap();
        }
        if args.ablate_at > 0 && t == args.ablate_at {
            ablate(&world, args.ablate_ticks, &mut log);
        }
    }
    if let Some(mut ff) = video {
        drop(ff.stdin.take());
        ff.wait().unwrap();
    }
}

/// Removes one ability at a time from a copy of the evolved world. If the copy
/// does no worse, the population wasn't really using that ability.
fn ablate(world: &World, ticks: u64, log: &mut Box<dyn Write>) {
    for name in ["none", "memory", "pheromones", "kin", "voice", "bite", "heat", "give", "enzyme", "share", "sex"] {
        let mut copy = world.clone();
        switch(&mut copy.cfg, name);
        copy.report();
        let (mut sum, mut least) = (0u64, u64::MAX);
        for _ in 0..ticks {
            copy.step();
            sum += copy.n as u64;
            least = least.min(copy.n as u64);
        }
        let end = copy.report();
        writeln!(
            log,
            "{}",
            json!({"ablate": name, "ticks": ticks, "mean_cells": sum / ticks.max(1), "least_cells": least, "cells": copy.n, "births": end["births"], "deaths": end["deaths"], "intake": end["intake"], "effective_species": end["effective_species"]})
        )
        .unwrap();
        log.flush().unwrap();
    }
}
