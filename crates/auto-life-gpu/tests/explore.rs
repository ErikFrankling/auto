//! Looking at a world and changing it while it runs. Colours are checked against the
//! same rules written as plain Rust.

use auto_life_gpu::config::Scope;
use auto_life_gpu::explore::{self, CELL_VIEWS, Canvas, FIELD_VIEWS, ViewParams, ViewPipe};
use auto_life_gpu::{Config, Pacing, World};

const SIZE: u32 = 64;

fn world(sets: &[(&str, &str)]) -> World {
    let mut cfg = Config::new(SIZE, 5, None).unwrap();
    for (k, v) in sets {
        cfg.set(k, v).unwrap();
    }
    World::new(
        cfg,
        Pacing {
            target_ms: 50.0,
            duty: 1.0,
        },
    )
    .unwrap()
}

/// Nothing grows or lands, so what's planted is all there is.
fn still() -> World {
    world(&[
        ("rock", "false"),
        ("rain_p", "0"),
        ("rain_burst_ticks", "0"),
        ("sun", "0"),
    ])
}

fn params(cell: &str, field: &str) -> ViewParams {
    let mut hi = [1.0; 16];
    hi[4] = 2.0;
    ViewParams {
        origin: [0.0, 0.0],
        scale: 1.0,
        samples: 1,
        out: [SIZE, SIZE],
        row0: 0,
        show: explore::SHOW_CELLS,
        cell_mode: CELL_VIEWS.iter().position(|v| v.name == cell).unwrap() as u32,
        field_mode: FIELD_VIEWS.iter().position(|v| v.name == field).unwrap() as u32,
        field_opacity: 1.0,
        cell_opacity: 1.0,
        cell_lo: 0.0,
        cell_hi: 40.0,
        tau: 0.4,
        clade: 0,
        hi,
        sketch: [0; 4],
        center: [SIZE as f32 / 2.0; 2],
        sun_k: 0.0,
        season: 0.0,
        season_on: 0.0,
        ground_cap: 4.0,
        sun_keep: 1.0,
        pad0: 0.0,
        drift: [0.0; 2],
        pad1: [0.0; 2],
    }
}

fn draw(w: &World, p: ViewParams) -> Vec<u8> {
    let layout = explore::layout(&w.gpu.device);
    let pipe = ViewPipe::new(w, &layout).unwrap();
    let canvas = Canvas::new(&w.gpu.device, &layout, SIZE, SIZE);
    canvas.draw(&w.gpu.device, &w.gpu.queue, &pipe, p);
    let staging = w.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * SIZE as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = w.gpu.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        canvas.texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    w.gpu.queue.submit([enc.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    w.gpu.wait();
    staging.get_mapped_range(..).expect("mapped").to_vec()
}

fn pixel(image: &[u8], x: u32, y: u32) -> [f32; 3] {
    let at = (y * 256 + x * 4) as usize;
    std::array::from_fn(|k| image[at + k] as f32 / 255.0)
}

fn close(a: [f32; 3], b: [f32; 3], what: &str) {
    for k in 0..3 {
        assert!(
            (a[k] - b[k]).abs() < 0.012,
            "{what}: got {a:?}, wanted {b:?}"
        );
    }
}

const BASE: [f32; 3] = [0.035, 0.04, 0.055];

fn blank_genome(w: &World) -> Vec<u8> {
    vec![0u8; w.cfg.blocks() as usize * 16]
}

#[test]
fn colours_match_the_plain_rules() {
    let mut w = still();
    let genome = blank_genome(&w);
    w.plant(10, 20, 12.0, 0.0, 3, &genome);
    w.write_field("fb", 30, 31, [0.5, 0.0, 0.0, 0.0]);

    let image = draw(&w, params("Energy", "Nothing"));
    close(pixel(&image, 10, 20), explore::seq(12.0 / 40.0), "energy");
    close(pixel(&image, 11, 20), BASE, "an empty square");

    let image = draw(&w, params("Heading", "Nothing"));
    close(
        pixel(&image, 10, 20),
        explore::hue(3.0 / 8.0, 0.85, 1.0),
        "heading",
    );

    // Scent 1's top of range here is 2, and levels are shown as a square root.
    let image = draw(&w, params("Energy", "Scent 1"));
    let k = (0.5f32 / 2.0).sqrt();
    let want = [BASE[0] + 1.0 * k, BASE[1] + 0.22 * k, BASE[2] + 0.28 * k];
    close(pixel(&image, 30, 31), want.map(|v| v.min(1.0)), "scent");

    // Half opacity lets half the field through.
    let mut half = params("Energy", "Scent 1");
    half.field_opacity = 0.5;
    let image = draw(&w, half);
    let want = [BASE[0] + 0.5 * k, BASE[1] + 0.11 * k, BASE[2] + 0.14 * k];
    close(pixel(&image, 30, 31), want, "half opacity");

    // Hidden cells leave the field showing where they stand.
    let mut hidden = params("Energy", "Nothing");
    hidden.show = 0;
    let image = draw(&w, hidden);
    close(pixel(&image, 10, 20), BASE, "a hidden cell");
}

#[test]
fn close_up_cells_are_squares_with_a_gap() {
    let mut w = still();
    let genome = blank_genome(&w);
    w.plant(2, 2, 12.0, 0.0, 0, &genome);
    let mut p = params("Energy", "Nothing");
    p.scale = 1.0 / 8.0;
    let image = draw(&w, p);
    let cell = explore::seq(12.0 / 40.0);
    // Square 2,2 covers pixels 16 to 23. Its first pixel is the gap, the next is cell.
    close(pixel(&image, 16, 20), BASE, "the gap");
    close(pixel(&image, 17, 20), cell, "the cell's edge");
    close(pixel(&image, 15, 20), BASE, "the empty square next to it");
}

#[test]
fn every_view_builds_and_draws() {
    let mut w = world(&[]);
    for _ in 0..30 {
        w.step();
    }
    w.flush();
    let scope = w.gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let layout = explore::layout(&w.gpu.device);
    let pipe = ViewPipe::new(&w, &layout).unwrap();
    let canvas = Canvas::new(&w.gpu.device, &layout, SIZE, SIZE);
    for cell in CELL_VIEWS {
        canvas.draw(
            &w.gpu.device,
            &w.gpu.queue,
            &pipe,
            params(cell.name, "Nothing"),
        );
    }
    for field in FIELD_VIEWS {
        canvas.draw(
            &w.gpu.device,
            &w.gpu.queue,
            &pipe,
            params("Genome", field.name),
        );
    }
    w.gpu.wait();
    let error = pollster::block_on(scope.pop());
    assert!(error.is_none(), "{error:?}");
}

#[test]
fn physics_changes_on_a_running_world() {
    let mut w = world(&[("rock", "false")]);
    for _ in 0..50 {
        w.step();
    }
    w.flush();
    let tick = w.tick;
    let terrain = w.read_u32s("terrain");
    assert!(w.alive() > 0, "a world with rain should have cells");

    // Certain death by old age, and no new cells: a rule only the new pipelines can know.
    let mut physics = w.cfg.physics.clone();
    physics.insert("hazard".into(), 2.0.into());
    physics.insert("rain_p".into(), 0.0.into());
    physics.insert("rain_burst_ticks".into(), 0.0.into());
    w.set_physics(physics).unwrap();
    assert_eq!(w.tick, tick, "changing physics doesn't pass time");
    assert_eq!(w.read_u32s("terrain"), terrain, "the land stays the same");
    assert!(w.alive() > 0, "changing physics doesn't kill anyone itself");
    w.step();
    w.flush();
    assert_eq!(w.alive(), 0, "the new rule applies from the next tick");
}

#[test]
fn sun_keeps_its_scale_when_changed() {
    let mut w = world(&[]);
    let k = w.sun_k;
    let mut physics = w.cfg.physics.clone();
    physics.insert("sun".into(), 0.1.into());
    w.set_physics(physics).unwrap();
    assert!((w.sun_k - k * 2.0).abs() < k * 1e-3, "{} vs {}", w.sun_k, k);
}

#[test]
fn a_saved_world_reopens_with_its_shape_and_physics() {
    let mut w = world(&[("bite_power", "17")]);
    for _ in 0..20 {
        w.step();
    }
    w.flush();
    let path = std::env::temp_dir().join(format!("explore-test-{}.bin", std::process::id()));
    let path = path.to_string_lossy().to_string();
    w.save(&path).unwrap();
    let cfg = Config::from_save(&path, None).unwrap();
    assert_eq!(cfg.size, SIZE);
    assert_eq!(cfg.f("bite_power"), 17.0);
    let opened = World::open_saved(
        w.gpu.clone(),
        &path,
        None,
        Pacing {
            target_ms: 50.0,
            duty: 1.0,
        },
        Default::default(),
    )
    .unwrap();
    assert_eq!(opened.tick, w.tick);
    assert_eq!(opened.alive(), w.alive());
    assert_eq!(opened.read_u32s("occ"), w.read_u32s("occ"));
    std::fs::remove_file(&path).unwrap();
    assert!(Config::from_save("/nonexistent/world.bin", None).is_err());
}

#[test]
fn settings_know_when_they_take_effect() {
    let cfg = Config::new(SIZE, 1, None).unwrap();
    assert_eq!(cfg.scope("rock"), Scope::NewWorld);
    assert_eq!(cfg.scope("sun"), Scope::Live);
    assert_eq!(cfg.scope("heat_decay"), Scope::Live);
    assert_eq!(cfg.scope("sex_on"), Scope::Live);
    assert_eq!(cfg.scope("x0_on"), Scope::Live);
    assert_eq!(cfg.scope("not_a_setting"), Scope::Unused);
}

#[test]
fn colour_ranges_ignore_the_extremes() {
    let mut v: Vec<f32> = (0..100).map(|k| k as f32).collect();
    v.push(1e9);
    let (lo, hi) = explore::fit_range(v).unwrap();
    assert!(lo < 3.0 && hi < 100.0, "{lo}..{hi}");
    assert!(explore::fit_range(vec![1.0, 2.0]).is_none());
    assert_eq!(explore::fit_top(vec![0.0, 0.0]), 1e-3);
}

#[test]
fn a_world_too_big_for_the_device_is_known_beforehand() {
    let mut cfg = Config::new(1024, 1, None).unwrap();
    let genome = cfg.blocks() as u64 * 16;
    // Half the squares can hold cells, so genomes are the biggest buffer here.
    cfg.slots = 1 << 19;
    let need = (1u64 << 19) * genome;
    assert!(cfg.check_buffers(need).is_ok());
    assert!(cfg.check_buffers(need - 1).is_err());

    // The cell limit that's offered always fits.
    let limit = 128 << 20;
    cfg.slots = cfg.slots_within(limit);
    assert!(cfg.check_buffers(limit).is_ok());
    assert!(cfg.check_buffers(cfg.slots as u64 * genome - 1).is_err());
}

#[test]
fn a_save_from_an_older_layout_is_refused_with_a_clear_reason() {
    use auto_life_gpu::config::read_save_head;
    // A header as the old engine wrote it, followed by nothing it could use.
    let path = std::env::temp_dir().join(format!("explore-old-{}.bin", std::process::id()));
    let head = br#"{"size":512,"slots":131072,"hidden_rows":2,"seed":1,"tick":5,"physics":{}}"#;
    let mut file = (head.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(head);
    std::fs::write(&path, &file).unwrap();
    let path = path.to_string_lossy().to_string();
    assert!(read_save_head(&path).is_ok());
    let why = Config::from_save(&path, None).err().unwrap();
    assert!(why.contains("older layout"), "{why}");
    std::fs::remove_file(&path).unwrap();
}
