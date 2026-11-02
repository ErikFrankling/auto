//! The window's contents: the world in the middle, what to colour it by on the left,
//! and on the right the picked cell, the charts and every physics setting.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use auto_core::Gpu;
use auto_life_gpu::config::{Scope, cell, stat};
use auto_life_gpu::explore::{
    self, CELL_VIEWS, Canvas, FIELD_VIEWS, HI_BASE, HI_SUN, HI_TRACKS, Paint, SCENT_TINT,
    SHOW_CELLS, SHOW_FOUNDER, SHOW_SIMILAR, ViewParams,
};
use auto_life_gpu::report::ACTIONS;
use auto_life_gpu::{Config, VRAM_MARGIN, free_vram};
use egui::{
    Align, Align2, Color32, Context, CornerRadius, Event, Frame, Key, Layout, Margin, ProgressBar,
    Rect, RichText, ScrollArea, Sense, Slider, Stroke, TextureId, Vec2, pos2, vec2,
};
use serde_json::{Map, Value};

use crate::files::{Browser, Purpose};
use crate::sim::{Cmd, NewWorld, Picked, Status};
use crate::vars::{GROUPS, Var, all_vars};
use crate::{SAVES_DIR, charts, legend};

const CELL_KEYS: [&str; 10] = [
    "Genome",
    "Diet",
    "Energy",
    "Age",
    "Heading",
    "Both signals",
    "Moved this tick",
    "Founder line",
    "Main spending",
    "Recent damage",
];
const FIELD_KEYS: [&str; 10] = [
    "All scents",
    "Scent 1",
    "Scent 2",
    "Scent 3",
    "Scent 4",
    "Ground food",
    "Meat",
    "Heat",
    "Noise",
    "Nothing",
];
const NUMBER_KEYS: [Key; 10] = [
    Key::Num1,
    Key::Num2,
    Key::Num3,
    Key::Num4,
    Key::Num5,
    Key::Num6,
    Key::Num7,
    Key::Num8,
    Key::Num9,
    Key::Num0,
];
const KEY_NAMES: [&str; 10] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"];

fn cell_index(name: &str) -> usize {
    CELL_VIEWS.iter().position(|v| v.name == name).unwrap_or(0)
}
fn field_index(name: &str) -> usize {
    FIELD_VIEWS.iter().position(|v| v.name == name).unwrap_or(0)
}
fn tint(c: [f32; 3]) -> Color32 {
    Color32::from_rgb(
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
    )
}
fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

pub fn style(ctx: &Context) {
    ctx.set_visuals(egui::Visuals::dark());
    ctx.global_style_mut(|s| {
        s.spacing.item_spacing = vec2(6.0, 3.0);
        s.spacing.slider_width = 130.0;
        s.spacing.interact_size.y = 20.0;
    });
}

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Inspect,
    Charts,
    Variables,
}

struct NewForm {
    open: bool,
    size: u32,
    seed: u32,
    slots: u32,
    hidden_rows: u32,
    keep_settings: bool,
    seeds: Vec<String>,
    seed_keep: f64,
    anyway: bool,
}

pub struct Ui {
    pub ctx: Context,
    gpu: Arc<Gpu>,
    layout: wgpu::BindGroupLayout,
    status: Arc<Mutex<Status>>,
    progress: Arc<AtomicU32>,
    tau_shared: Arc<AtomicU32>,
    tx: Sender<Cmd>,
    shaders: Option<PathBuf>,

    st: Status,
    canvas: Option<Canvas>,
    texture: Option<TextureId>,
    drawing: Arc<AtomicBool>,
    last_draw: Instant,
    drawn: Option<DrawKey>,

    cx: f64,
    cy: f64,
    scale: f64,
    fit_next: bool,
    /// The view still shows the whole world, so it should keep doing so when the window changes.
    fitted: bool,
    follow: bool,
    world_id: u64,

    cell_mode: usize,
    field_mode: usize,
    show_cells: bool,
    field_opacity: f32,
    cell_opacity: f32,
    fit_colours: bool,
    brightness: f32,
    similar: bool,
    same_founder: bool,
    tau: f32,
    cell_range: (f32, f32),
    sent_showing: Option<(usize, bool)>,

    running: bool,
    full_speed: bool,
    rate: f64,

    tab: Tab,
    vars: Vec<Var>,
    edits: Map<String, Value>,
    applied_seen: u64,
    dirty_at: Option<Instant>,
    search: String,

    new: NewForm,
    browser: Option<(Purpose, Browser)>,
    path_text: String,
    overwrite_ok: bool,
    help: bool,
    show_panels: bool,
    longest_frame: f32,
    good_news_since: Option<Instant>,
    pan: Option<Pan>,
    frame_clock: Instant,
}

/// A drag on the world in progress.
struct Pan {
    last: egui::Pos2,
    moved: f32,
}

#[derive(PartialEq, Clone, Copy)]
struct DrawKey {
    world: u64,
    tick: u64,
    params: [u8; std::mem::size_of::<ViewParams>()],
}

impl Ui {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: Context,
        gpu: Arc<Gpu>,
        layout: wgpu::BindGroupLayout,
        status: Arc<Mutex<Status>>,
        progress: Arc<AtomicU32>,
        tau_shared: Arc<AtomicU32>,
        tx: Sender<Cmd>,
        shaders: Option<PathBuf>,
        seed: u32,
    ) -> Ui {
        Ui {
            ctx,
            gpu,
            layout,
            status,
            progress,
            tau_shared,
            tx,
            shaders,
            st: Status::default(),
            canvas: None,
            texture: None,
            drawing: Arc::new(AtomicBool::new(false)),
            last_draw: Instant::now(),
            drawn: None,
            cx: 0.0,
            cy: 0.0,
            scale: 1.0,
            fit_next: true,
            fitted: true,
            follow: false,
            world_id: 0,
            cell_mode: 0,
            field_mode: field_index("Natural mix"),
            show_cells: true,
            field_opacity: 1.0,
            cell_opacity: 1.0,
            fit_colours: true,
            brightness: 1.0,
            similar: false,
            same_founder: false,
            tau: 0.4,
            cell_range: (0.0, 1.0),
            sent_showing: None,
            running: true,
            full_speed: true,
            rate: 60.0,
            tab: Tab::Charts,
            vars: vec![],
            edits: Map::new(),
            applied_seen: 0,
            dirty_at: None,
            search: String::new(),
            new: NewForm {
                open: false,
                size: 1024,
                seed,
                slots: 0,
                hidden_rows: 2,
                keep_settings: true,
                seeds: vec![],
                seed_keep: 1.0,
                anyway: false,
            },
            browser: None,
            path_text: String::new(),
            overwrite_ok: false,
            help: false,
            show_panels: true,
            longest_frame: 0.0,
            good_news_since: None,
            pan: None,
            frame_clock: Instant::now(),
        }
    }

    /// Remembers the slowest recent frame, for the world details and for debugging.
    pub fn note_frame(&mut self, spent: std::time::Duration) {
        let ms = spent.as_secs_f32() * 1000.0;
        self.longest_frame = self.longest_frame.max(ms);
        if self.frame_clock.elapsed().as_secs() >= 5 {
            if std::env::var_os("LIFE_DEBUG").is_some() {
                eprintln!(
                    "slowest frame {:.1} ms, tick {}, {:.1} ticks/s, gpu {:.2} ms/tick, longest submit {:.1} ms",
                    self.longest_frame,
                    self.st.tick,
                    self.st.tps,
                    self.st.gpu_ms_per_tick,
                    self.st.longest_ms
                );
            }
            self.longest_frame = 0.0;
            self.frame_clock = Instant::now();
        }
    }

    pub fn title(&self) -> String {
        if self.st.size == 0 {
            return "Life".into();
        }
        format!(
            "Life: {0} x {0}, tick {1}, {2} cells",
            self.st.size, self.st.tick, self.st.cells
        )
    }

    /// Whether to keep drawing frames with no input.
    pub fn wants_frames(&self) -> bool {
        self.st.busy.is_some()
            || self.drawing.load(Ordering::Relaxed)
            || (self.running && (self.full_speed || self.rate >= 4.0))
            || self.follow
    }

    fn send(&self, cmd: Cmd) {
        let _ = self.tx.send(cmd);
    }

    pub fn show(&mut self, root: &mut Ui2, renderer: &mut egui_wgpu::Renderer) {
        self.st = self.status.lock().unwrap().clone();
        // Good news goes away by itself. Errors stay until they're read.
        if matches!(self.st.message, Some((false, _))) {
            let since = *self.good_news_since.get_or_insert_with(Instant::now);
            if since.elapsed().as_secs() >= 6 {
                self.status.lock().unwrap().message = None;
                self.good_news_since = None;
            } else {
                root.ctx()
                    .request_repaint_after(std::time::Duration::from_secs(1));
            }
        } else {
            self.good_news_since = None;
        }
        if self.st.world_id != self.world_id {
            self.world_id = self.st.world_id;
            self.fit_next = true;
            self.follow = false;
            self.similar = false;
            self.same_founder = false;
            self.running = true;
            let defaults = Config::new(256, 1, self.shaders.clone());
            if let Ok(d) = defaults {
                self.vars = all_vars(&d, &self.st.physics);
            }
            self.edits = self.st.physics.clone();
            self.applied_seen = self.st.physics_applied;
            self.dirty_at = None;
            self.new.size = (self.st.size * 2).clamp(256, 8192);
            self.new.slots = 0;
        }
        self.tau_shared.store(self.tau.to_bits(), Ordering::Relaxed);
        self.apply_edits();
        self.keys(root.ctx().clone());
        let showing = (self.cell_mode, self.cell_mode == 0);
        if self.sent_showing != Some(showing) {
            self.sent_showing = Some(showing);
            self.send(Cmd::Showing {
                cell_mode: showing.0,
                needs_bake: showing.1,
            });
        }

        egui::Panel::top("controls")
            .frame(Frame::side_top_panel(root.style()).inner_margin(Margin::symmetric(8, 5)))
            .show(root, |ui| self.top_bar(ui));
        // Side panels take a share of the window, so a small window still shows the world.
        let window = root.ctx().content_rect().width();
        if self.show_panels {
            egui::Panel::left("views")
                .resizable(true)
                .default_size((window * 0.18).clamp(170.0, 236.0))
                .min_size(150.0)
                .max_size((window * 0.3).max(170.0))
                .show(root, |ui| self.views_panel(ui));
            egui::Panel::right("side")
                .resizable(true)
                .default_size((window * 0.26).clamp(240.0, 372.0))
                .min_size(220.0)
                .max_size((window * 0.45).max(240.0))
                .show(root, |ui| self.side_panel(ui));
        }
        egui::CentralPanel::default()
            .frame(Frame::NONE)
            .show(root, |ui| self.world_view(ui, renderer));

        self.new_world_window(root.ctx().clone());
        self.file_window(root.ctx().clone());
        self.help_window(root.ctx().clone());
    }

    fn apply_edits(&mut self) {
        if let Some(at) = self.dirty_at
            && at.elapsed().as_millis() > 180
        {
            self.dirty_at = None;
            self.send(Cmd::Physics(self.edits.clone()));
        }
    }

    fn set_running(&mut self, on: bool) {
        self.running = on;
        self.send(Cmd::Run(on));
    }

    fn set_speed(&mut self, full: bool, rate: f64) {
        self.full_speed = full;
        self.rate = rate.clamp(1.0, 50000.0);
        self.send(Cmd::Speed(if full { None } else { Some(self.rate) }));
    }

    fn fit(&mut self) {
        self.fit_next = true;
        self.follow = false;
    }

    fn keys(&mut self, ctx: Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let mut pressed: Vec<(Key, bool)> = vec![];
        ctx.input(|i| {
            for e in &i.events {
                if let Event::Key {
                    key,
                    physical_key,
                    pressed: true,
                    modifiers,
                    ..
                } = e
                {
                    pressed.push((physical_key.unwrap_or(*key), modifiers.shift));
                }
            }
        });
        for (key, shift) in pressed {
            if let Some(n) = NUMBER_KEYS.iter().position(|k| *k == key) {
                if shift {
                    self.field_mode = field_index(FIELD_KEYS[n]);
                } else {
                    self.cell_mode = cell_index(CELL_KEYS[n]);
                    self.show_cells = true;
                }
                continue;
            }
            match key {
                Key::Space => self.set_running(!self.running),
                Key::Period | Key::N => {
                    self.running = false;
                    self.send(Cmd::Step);
                }
                Key::ArrowDown => {
                    self.cell_mode = (self.cell_mode + 1) % CELL_VIEWS.len();
                    self.show_cells = true;
                }
                Key::ArrowUp => {
                    self.cell_mode = (self.cell_mode + CELL_VIEWS.len() - 1) % CELL_VIEWS.len();
                    self.show_cells = true;
                }
                Key::ArrowRight => self.field_mode = (self.field_mode + 1) % FIELD_VIEWS.len(),
                Key::ArrowLeft => {
                    self.field_mode = (self.field_mode + FIELD_VIEWS.len() - 1) % FIELD_VIEWS.len()
                }
                Key::F => self.fit(),
                Key::H => self.show_cells = !self.show_cells,
                Key::P => self.show_panels = !self.show_panels,
                Key::Plus | Key::Equals => {
                    if !self.full_speed {
                        self.set_speed(self.rate >= 25000.0, self.rate * 2.0);
                    }
                }
                Key::Minus => {
                    let from = if self.full_speed {
                        self.st.tps.max(2.0)
                    } else {
                        self.rate
                    };
                    self.set_speed(false, from / 2.0);
                }
                Key::Escape => self.clear_pick(),
                _ => {}
            }
        }
    }

    fn clear_pick(&mut self) {
        self.follow = false;
        self.similar = false;
        self.same_founder = false;
        self.send(Cmd::Unpick);
    }

    fn top_bar(&mut self, ui: &mut Ui2) {
        // Wide enough for everything on one row, or the numbers go on a row of their own.
        let one_row = ui.available_width() >= 1250.0;
        ui.horizontal(|ui| {
            self.top_controls(ui);
            if one_row {
                self.top_stats(ui);
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                self.top_right(ui)
            });
        });
        if !one_row {
            ui.horizontal(|ui| self.top_stats(ui));
        }
    }

    fn top_controls(&mut self, ui: &mut Ui2) {
        {
            let label = if self.running { "Pause" } else { "Play" };
            if ui
                .add_sized(
                    [62.0, 24.0],
                    egui::Button::new(RichText::new(label).strong()),
                )
                .on_hover_text("Space")
                .clicked()
            {
                self.set_running(!self.running);
            }
            if ui
                .add_sized([62.0, 24.0], egui::Button::new("Step"))
                .on_hover_text("Run one tick, then pause. Key: N or the full stop")
                .clicked()
            {
                self.running = false;
                self.send(Cmd::Step);
            }
            ui.separator();
            ui.label("Speed");
            let mut rate = self.rate;
            let slider = ui
                .add_enabled(
                    !self.full_speed,
                    Slider::new(&mut rate, 1.0..=20000.0)
                        .logarithmic(true)
                        .suffix(" ticks/s")
                        .fixed_decimals(0),
                )
                .on_hover_text("Ticks per second to aim for. Keys: + and -");
            if slider.changed() {
                self.set_speed(false, rate);
            }
            for (name, r) in [("1/s", 1.0), ("10/s", 10.0), ("100/s", 100.0)] {
                if ui
                    .selectable_label(!self.full_speed && self.rate == r, name)
                    .clicked()
                {
                    self.set_speed(false, r);
                }
            }
            if ui
                .selectable_label(self.full_speed, "Max")
                .on_hover_text("As fast as the GPU can go, while leaving it room for the desktop")
                .clicked()
            {
                let full = !self.full_speed;
                self.set_speed(full, self.rate);
            }
        }
    }

    fn top_stats(&mut self, ui: &mut Ui2) {
        {
            ui.separator();
            if self.st.size == 0 {
                return;
            }
            let tick = format!("{}", self.st.tick);
            stat(ui, "tick", &tick);
            stat(ui, "cells", &format!("{}", self.st.cells));
            stat(ui, "ticks/s", &rate_text(self.st.tps));
            stat(ui, "world", &format!("{0} x {0}", self.st.size))
                .on_hover_text(self.world_details());
            if self.st.sharing {
                ui.colored_label(Color32::from_rgb(230, 190, 90), "sharing the GPU")
                    .on_hover_text("Another GPU job is running, so this takes a smaller share.");
            }
        }
    }

    fn top_right(&mut self, ui: &mut Ui2) {
        if ui
            .selectable_label(self.show_panels, "Panels")
            .on_hover_text("Hide or show the side panels. Key: P")
            .clicked()
        {
            self.show_panels = !self.show_panels;
        }
        if ui.button("Help").clicked() {
            self.help = !self.help;
        }
        if ui.button("Save").clicked() {
            self.path_text = format!("{SAVES_DIR}/app-{}-tick{}.bin", self.st.size, self.st.tick);
            self.overwrite_ok = false;
            self.browser = Some((Purpose::Save, Browser::new(SAVES_DIR)));
        }
        if ui.button("Open").clicked() {
            self.path_text.clear();
            self.browser = Some((Purpose::Load, Browser::new(SAVES_DIR)));
        }
        if ui.button(RichText::new("New world").strong()).clicked() {
            self.new.open = true;
            self.new.anyway = false;
        }
        if ui
            .button("Fit")
            .on_hover_text("See the whole world. Key: F")
            .clicked()
        {
            self.fit();
        }
    }

    fn world_details(&self) -> String {
        let st = &self.st;
        let from = st.path.as_deref().unwrap_or("new");
        format!(
            "{from}\nSeed {}. Room for {} cells. {:.2} GB of VRAM.\nGPU: {}\nGPU time per tick: {:.2} ms. Longest submission lately: {:.1} ms.",
            st.seed,
            st.slots,
            gb(st.vram),
            st.adapter,
            st.gpu_ms_per_tick,
            st.longest_ms
        )
    }

    fn views_panel(&mut self, ui: &mut Ui2) {
        let total = ui.available_height();
        ui.add_space(4.0);
        ui.heading("Colour cells by");
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_cells, "Show")
                .on_hover_text("Hide cells to see what's under them. Key: H");
            ui.add_enabled(
                self.show_cells,
                Slider::new(&mut self.cell_opacity, 0.1..=1.0)
                    .show_value(false)
                    .text("opacity"),
            );
        });
        ui.checkbox(&mut self.fit_colours, "Fit colours to what's alive")
            .on_hover_text(
                "Stretches the colour scale over the range the population actually has now.",
            );
        ui.small("Keys 1 to 0, or up and down");
        ScrollArea::vertical()
            .id_salt("cell list")
            .max_height((total * 0.56).max(160.0))
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let mut group = "";
                for (i, v) in CELL_VIEWS.iter().enumerate() {
                    if v.group != group {
                        group = v.group;
                        ui.add_space(4.0);
                        ui.label(RichText::new(group).small().strong().weak());
                    }
                    let key = CELL_KEYS.iter().position(|n| *n == v.name);
                    if row(ui, self.cell_mode == i, v.name, key.map(|k| KEY_NAMES[k]))
                        .on_hover_text(v.help)
                        .clicked()
                    {
                        self.cell_mode = i;
                        self.show_cells = true;
                    }
                }
            });
        ui.separator();
        ui.heading("Under the cells");
        ui.add(
            Slider::new(&mut self.field_opacity, 0.0..=1.0)
                .show_value(true)
                .text("opacity"),
        );
        ui.add(
            Slider::new(&mut self.brightness, 0.25..=8.0)
                .logarithmic(true)
                .text("brightness"),
        )
        .on_hover_text(
            "Each field's colours are scaled to its own range. This pushes them further.",
        );
        ui.small("Shift + 1 to 0, or left and right");
        ScrollArea::vertical()
            .id_salt("field list")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let mut group = "";
                for (i, v) in FIELD_VIEWS.iter().enumerate() {
                    if v.group != group {
                        group = v.group;
                        ui.add_space(4.0);
                        ui.label(RichText::new(group).small().strong().weak());
                    }
                    let key = FIELD_KEYS.iter().position(|n| *n == v.name);
                    let hint = key.map(|k| format!("Shift {}", KEY_NAMES[k]));
                    if row(ui, self.field_mode == i, v.name, hint.as_deref())
                        .on_hover_text(v.help)
                        .clicked()
                    {
                        self.field_mode = i;
                    }
                }
            });
    }

    fn side_panel(&mut self, ui: &mut Ui2) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            for (tab, name) in [
                (Tab::Inspect, "Inspect"),
                (Tab::Charts, "Charts"),
                (Tab::Variables, "Variables"),
            ] {
                if ui
                    .selectable_label(self.tab == tab, RichText::new(name).size(15.0))
                    .clicked()
                {
                    self.tab = tab;
                }
            }
        });
        ui.separator();
        match self.tab {
            Tab::Inspect => {
                ScrollArea::vertical()
                    .id_salt("inspect")
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.inspect(ui));
            }
            Tab::Charts => {
                let history = self.st.history.clone();
                ScrollArea::vertical()
                    .id_salt("charts")
                    .auto_shrink([false, false])
                    .show(ui, |ui| charts::show(ui, &history));
            }
            Tab::Variables => self.variables(ui),
        }
    }

    fn inspect(&mut self, ui: &mut Ui2) {
        let Some(p) = self.st.picked.clone() else {
            ui.add_space(6.0);
            ui.label("Click any cell in the world to see its numbers.");
            ui.small("Zoom in with the scroll wheel until cells are squares. Clicking an empty square shows what's on it.");
            return;
        };
        if let Some(slot) = p.slot {
            ui.heading(format!("Cell at {}, {}", p.x, p.y));
            ui.small(format!("Slot {slot}"));
            ui.horizontal_wrapped(|ui| {
                ui.toggle_value(&mut self.follow, "Follow")
                    .on_hover_text("Keep this cell in the middle of the view.");
                ui.toggle_value(&mut self.similar, "Similar genomes")
                    .on_hover_text("Grey out everything but cells with a genome like this one, across the whole world.");
                if ui
                    .toggle_value(&mut self.same_founder, "Same founder")
                    .on_hover_text("Grey out everything but this cell's founder line.")
                    .changed()
                    && self.same_founder
                {
                    self.similar = false;
                }
                if ui.button("Clear").clicked() {
                    self.clear_pick();
                }
            });
            if self.similar {
                self.same_founder = false;
                ui.add(
                    Slider::new(&mut self.tau, 0.05..=2.0)
                        .logarithmic(true)
                        .text("how similar"),
                );
                if let Some(s) = p.similar_share {
                    ui.small(format!(
                        "About {:.1}% of the population is this close.",
                        s * 100.0
                    ));
                }
            }
            self.cell_numbers(ui, &p);
        } else {
            if p.rock {
                ui.heading(format!("Rock at {}, {}", p.x, p.y));
            } else {
                ui.heading(format!("Square at {}, {}", p.x, p.y));
            }
            if !p.note.is_empty() {
                ui.colored_label(Color32::from_rgb(230, 170, 120), &p.note);
            }
            if ui.button("Clear").clicked() {
                self.clear_pick();
            }
        }
        ui.add_space(6.0);
        ui.strong("On this square");
        let row_of = |ui: &mut Ui2, name: &str, v: f32| {
            ui.label(name);
            ui.label(legend::number(v));
            ui.end_row();
        };
        egui::Grid::new("square")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                row_of(ui, "Ground food", p.fa[0]);
                row_of(ui, "Meat", p.fa[1]);
                row_of(ui, "Heat", p.fa[2]);
                row_of(ui, "Noise", p.fa[3]);
                for k in 0..4 {
                    ui.colored_label(tint(SCENT_TINT[k]), format!("Scent {}", k + 1));
                    ui.label(legend::number(p.fb[k]));
                    ui.end_row();
                }
                row_of(ui, "Enzyme", p.fc[0]);
                row_of(ui, "Waste", p.fc[1]);
                row_of(ui, "Fire", p.fc[2]);
                row_of(ui, "Tracks", p.fd[0]);
            });
    }

    fn cell_numbers(&self, ui: &mut Ui2, p: &Picked) {
        let f = |k: usize| f32::from_bits(p.cell[k]);
        let flags = p.cell[cell::FLAGS];
        const DIRECTION: [&str; 8] = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"];
        ui.add_space(4.0);
        egui::Grid::new("cell")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                let mut line = |name: &str, value: String| {
                    ui.label(name);
                    ui.label(value);
                    ui.end_row();
                };
                line("Energy", legend::number(f(cell::E)));
                line("Womb", legend::number(f(cell::WOMB)));
                line("Age", format!("{:.0} ticks", f(cell::AGE)));
                line("Generation", format!("{}", p.cell[cell::GEN]));
                line("Founder line", format!("{:08x}", p.cell[cell::CLADE]));
                line("Recent damage", legend::number(f(cell::HIT)));
                // What the rules that are on keep on each cell.
                for (key, name, at) in [
                    ("armour_on", "Armour", cell::HARD),
                    ("pack_on", "Wound", cell::WOUND),
                    ("skill_on", "Practice grazing", cell::SKILL_G),
                    ("skill_on", "Practice at flesh", cell::SKILL_F),
                    ("biomes_on", "Place feels costly", cell::FEEL_U),
                    ("biomes_on", "Place feels slow", cell::FEEL_D),
                    ("mate_track", "Children by mating", cell::MATE_KIDS),
                ] {
                    if self.st.physics.get(key) == Some(&Value::Bool(true)) {
                        line(name, legend::number(f(at)));
                    }
                }
                line(
                    "Facing",
                    DIRECTION[(flags & cell::DIR_MASK) as usize].to_string(),
                );
                line("Signal 1", format!("{:+.2}", f(cell::VOICE)));
                line("Signal 2", format!("{:+.2}", f(cell::VOICE + 1)));
                line(
                    "Right now",
                    [
                        (flags & cell::F_MOVED != 0, "moved"),
                        (flags & cell::F_BIT != 0, "bit"),
                        (flags & cell::F_MATE != 0, "will mate"),
                        (flags & cell::F_SHARE != 0, "will share"),
                    ]
                    .iter()
                    .filter(|x| x.0)
                    .map(|x| x.1)
                    .collect::<Vec<_>>()
                    .join(", "),
                );
            });

        ui.add_space(6.0);
        ui.strong("Lives on");
        let d = &p.stat[stat::DIET..stat::DIET + 4];
        let total: f32 = d.iter().sum::<f32>().max(1e-6);
        let diet = [
            ("ground", Color32::from_rgb(0x19, 0x9e, 0x70)),
            ("meat", Color32::from_rgb(0xd9, 0x59, 0x26)),
            ("biting", Color32::from_rgb(0xe6, 0x67, 0x67)),
            ("gifts", Color32::from_rgb(0x39, 0x87, 0xe5)),
        ];
        for (k, (name, c)) in diet.iter().enumerate() {
            bar(ui, name, d[k] / total, *c);
        }
        ui.add_space(6.0);
        ui.strong("Spends its budget on");
        if f(cell::AGE) < 200.0 {
            ui.small("It is young, so these readings are still building up.");
        }
        for (k, name) in ACTIONS.iter().enumerate() {
            // Armour, cleaning and the spare actions only appear while they're in use.
            let share = if k < 12 {
                p.stat[stat::BUDGET + k]
            } else {
                p.stat[stat::SPARE + k - 12]
            };
            if k < 10 || share > 0.001 {
                bar(ui, name, share, Color32::from_rgb(0x8a, 0x93, 0xa8));
            }
        }
        ui.add_space(6.0);
        ui.strong("When it moves");
        let pct = |v: f32| format!("{:+.0}%", v * 100.0);
        egui::Grid::new("moves")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                let mut line = |name: &str, value: String| {
                    ui.label(name);
                    ui.label(value);
                    ui.end_row();
                };
                line(
                    "Moves this share of ticks",
                    format!("{:.0}%", p.stat[stat::CHASE + 3] * 100.0),
                );
                line("Towards food", pct(p.stat[stat::CHASE + 2]));
                line("Towards bigger cells", pct(p.stat[stat::CHASE]));
                line("Towards smaller cells", pct(p.stat[stat::CHASE + 1]));
                for k in 0..4 {
                    line(
                        &format!("Up scent {}", k + 1),
                        pct(p.stat[stat::FOLLOW + k]),
                    );
                }
                line("Crowding", format!("{:.0}%", p.stat[stat::MISC] * 100.0));
            });
    }

    fn variables(&mut self, ui: &mut Ui2) {
        let changed: usize = self
            .vars
            .iter()
            .filter(|v| self.edits.get(&v.key) != Some(&v.default))
            .count();
        ui.horizontal(|ui| {
            ui.label("Find");
            ui.add(egui::TextEdit::singleline(&mut self.search).desired_width(120.0));
            if ui
                .add_enabled(changed > 0, egui::Button::new("Reset all"))
                .on_hover_text("Everything back to its default")
                .clicked()
            {
                for v in &self.vars {
                    self.edits.insert(v.key.clone(), v.default.clone());
                }
                self.dirty_at = Some(Instant::now());
            }
        });
        let pending = self.dirty_at.is_some() || self.st.physics != self.edits;
        ui.small(if pending {
            "Applying to the running world..."
        } else if changed > 0 {
            "Changes apply to the running world straight away. Changed ones are marked."
        } else {
            "Changes apply to the running world straight away."
        });
        ui.add_space(2.0);
        let search = self.search.to_lowercase();
        let mut dirty = false;
        // Fixed once per frame: sizing from the space left in a row would stretch the panel.
        let slider_width = (ui.available_width() - LABEL_WIDTH - 140.0).clamp(50.0, 300.0);
        ScrollArea::vertical()
            .id_salt("variables")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for group in GROUPS {
                    let mut in_group: Vec<&Var> = self
                        .vars
                        .iter()
                        .filter(|v| v.group == group)
                        .filter(|v| search.is_empty() || v.key.contains(&search))
                        .collect();
                    if in_group.is_empty() {
                        continue;
                    }
                    in_group.sort_by_key(|v| !v.key.ends_with("_on"));
                    egui::CollapsingHeader::new(RichText::new(group).strong())
                        .default_open(true)
                        .show(ui, |ui| {
                            for v in in_group {
                                dirty |= var_row(ui, v, &mut self.edits, slider_width);
                            }
                        });
                }
            });
        if dirty {
            self.dirty_at = Some(Instant::now());
        }
    }

    fn view_params(&mut self, width: u32, height: u32) -> ViewParams {
        let view = &CELL_VIEWS[self.cell_mode];
        // Fitted ranges glide, so colours don't jump every time they're refitted.
        let mut want = view.range;
        if self.fit_colours
            && view.reading.is_some()
            && let Some((mode, r)) = self.st.cell_range
            && mode == self.cell_mode
        {
            want = if view.paint == Paint::Sides {
                let m = r.0.abs().max(r.1.abs()).max(0.02);
                (-m, m)
            } else {
                r
            };
        }
        self.cell_range.0 += (want.0 - self.cell_range.0) * 0.25;
        self.cell_range.1 += (want.1 - self.cell_range.1) * 0.25;
        let picked = self.st.picked.as_ref().filter(|p| p.slot.is_some());
        let mut show = 0;
        if self.show_cells {
            show |= SHOW_CELLS;
        }
        if self.similar && picked.is_some() {
            show |= SHOW_SIMILAR;
        }
        if self.same_founder && picked.is_some() {
            show |= SHOW_FOUNDER;
        }
        let b = self.brightness;
        let h = &self.st.hi;
        let mut hi = [1.0f32; 16];
        for k in 0..12 {
            hi[k] = h[k] / b;
        }
        hi[12] = h[HI_SUN] / b;
        hi[13] = h[HI_BASE] / b;
        hi[14] = h[HI_TRACKS] / b;
        ViewParams {
            origin: [
                (self.cx - width as f64 * 0.5 * self.scale) as f32,
                (self.cy - height as f64 * 0.5 * self.scale) as f32,
            ],
            scale: self.scale as f32,
            samples: explore::samples_for(self.scale as f32),
            out: [width, height],
            row0: 0,
            show,
            cell_mode: self.cell_mode as u32,
            field_mode: self.field_mode as u32,
            field_opacity: self.field_opacity,
            cell_opacity: self.cell_opacity,
            cell_lo: self.cell_range.0,
            cell_hi: self.cell_range.1,
            tau: self.tau,
            clade: picked.map_or(0, |p| p.cell[cell::CLADE]),
            hi,
            sketch: picked.map_or([0; 4], |p| p.sketch),
            center: [self.cx as f32, self.cy as f32],
            sun_k: self.st.sun_k,
            season: self.st.season,
            season_on: self.st.season_on as u32 as f32,
            ground_cap: self.st.ground_cap.max(1e-3),
            sun_keep: self.st.sun_keep,
            pad0: 0.0,
            drift: self.st.drift,
            pad1: [0.0, 0.0],
        }
    }

    fn world_view(&mut self, ui: &mut Ui2, renderer: &mut egui_wgpu::Renderer) {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let ppp = ui.ctx().pixels_per_point() as f64;
        let width = ((rect.width() as f64 * ppp).round() as u32).max(16);
        let height = ((rect.height() as f64 * ppp).round() as u32).max(16);
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, Color32::from_rgb(8, 9, 12));

        if self
            .canvas
            .as_ref()
            .is_none_or(|c| c.width != width || c.height != height)
        {
            let canvas = Canvas::new(&self.gpu.device, &self.layout, width, height);
            match self.texture {
                Some(id) => renderer.update_egui_texture_from_wgpu_texture(
                    &self.gpu.device,
                    &canvas.show,
                    wgpu::FilterMode::Nearest,
                    id,
                ),
                None => {
                    self.texture = Some(renderer.register_native_texture(
                        &self.gpu.device,
                        &canvas.show,
                        wgpu::FilterMode::Nearest,
                    ));
                }
            }
            self.canvas = Some(canvas);
            self.drawn = None;
            if self.fitted {
                self.fit_next = true;
            }
        }

        let size = self.st.size as f64;
        if self.st.view.is_some() && size > 0.0 {
            let fit = size / width.min(height) as f64;
            if self.fit_next {
                self.cx = size / 2.0;
                self.cy = size / 2.0;
                self.scale = fit;
                self.fit_next = false;
                self.fitted = true;
            }
            self.camera(ui, &response, rect, ppp, fit);
            let params = self.view_params(width, height);
            let key = DrawKey {
                world: self.world_id,
                tick: self.st.tick,
                params: bytemuck::cast(params),
            };
            let gap = if self.st.software { 60 } else { 15 };
            if self.drawn != Some(key)
                && self.last_draw.elapsed().as_millis() >= gap
                && !self.drawing.load(Ordering::Relaxed)
                && let (Some(view), Some(canvas)) = (&self.st.view, &self.canvas)
            {
                self.drawing.store(true, Ordering::Relaxed);
                canvas.draw(&self.gpu.device, &self.gpu.queue, view, params);
                let drawing = self.drawing.clone();
                self.gpu
                    .queue
                    .on_submitted_work_done(move || drawing.store(false, Ordering::Relaxed));
                self.drawn = Some(key);
                self.last_draw = Instant::now();
            } else if self.drawn != Some(key) {
                ui.ctx().request_repaint();
            }
            if let Some(id) = self.texture {
                ui.painter().image(
                    id,
                    rect,
                    Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
            self.mark_picked(ui, rect, ppp, size);
            self.legend_box(ui, rect);
            self.place_hint(ui, &response, rect, ppp, size);
        }
        self.busy_box(ui, rect);
    }

    fn camera(&mut self, ui: &mut Ui2, response: &egui::Response, rect: Rect, ppp: f64, fit: f64) {
        let size = self.st.size as f64;
        // Read from the raw pointer events, so a drag counts however few frames it spans. On a
        // slow frame a whole quick drag can arrive at once, which egui's own drag tracking ignores.
        let events = ui.ctx().input(|i| i.events.clone());
        for event in &events {
            match event {
                Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    ..
                } => {
                    let on_world = rect.contains(*pos)
                        && ui
                            .ctx()
                            .layer_id_at(*pos)
                            .is_none_or(|layer| layer.order == egui::Order::Background);
                    self.pan = (*pressed && on_world).then_some(Pan {
                        last: *pos,
                        moved: 0.0,
                    });
                }
                Event::PointerMoved(pos) => {
                    if let Some(pan) = &mut self.pan {
                        let step = *pos - pan.last;
                        pan.last = *pos;
                        pan.moved += step.length();
                        // A click wobbles a little. That shouldn't move the view.
                        if pan.moved > 3.0 {
                            self.cx -= step.x as f64 * ppp * self.scale;
                            self.cy -= step.y as f64 * ppp * self.scale;
                            self.follow = false;
                            self.fitted = false;
                        }
                    }
                }
                Event::PointerGone => self.pan = None,
                _ => {}
            }
        }
        if response.hovered() {
            let (scroll, pinch, cursor) = ui.ctx().input(|i| {
                (
                    i.smooth_scroll_delta.y,
                    i.zoom_delta(),
                    i.pointer.hover_pos(),
                )
            });
            let factor = (-(scroll as f64) * 0.0028).exp() / pinch as f64;
            if factor != 1.0
                && let Some(cursor) = cursor
            {
                let new = (self.scale * factor).clamp(1.0 / 64.0, fit * 2.0);
                // The square under the cursor stays under the cursor.
                let off_x = (cursor.x - rect.center().x) as f64 * ppp;
                let off_y = (cursor.y - rect.center().y) as f64 * ppp;
                self.cx += off_x * (self.scale - new);
                self.cy += off_y * (self.scale - new);
                self.scale = new;
                self.fitted = false;
            }
        }
        if self.follow
            && let Some(p) = self.st.picked.as_ref().filter(|p| p.slot.is_some())
        {
            let wrap = |d: f64| (d + size / 2.0).rem_euclid(size) - size / 2.0;
            self.cx += wrap(p.x as f64 + 0.5 - self.cx) * 0.25;
            self.cy += wrap(p.y as f64 + 0.5 - self.cy) * 0.25;
        }
        self.cx = self.cx.rem_euclid(size);
        self.cy = self.cy.rem_euclid(size);
        if response.clicked()
            && let Some(cursor) = response.interact_pointer_pos()
        {
            let (x, y) = self.square_at(cursor, rect, ppp, size);
            self.send(Cmd::Pick(x, y));
            self.tab = Tab::Inspect;
            self.similar = false;
            self.same_founder = false;
        }
        if response.secondary_clicked() {
            self.clear_pick();
        }
    }

    fn square_at(&self, at: egui::Pos2, rect: Rect, ppp: f64, size: f64) -> (u32, u32) {
        let q = |p: f32, centre: f32, c: f64| {
            (c + (p - centre) as f64 * ppp * self.scale)
                .floor()
                .rem_euclid(size) as u32
        };
        (
            q(at.x, rect.center().x, self.cx),
            q(at.y, rect.center().y, self.cy),
        )
    }

    fn mark_picked(&self, ui: &Ui2, rect: Rect, ppp: f64, size: f64) {
        let Some(p) = &self.st.picked else { return };
        let wrap = |d: f64| (d + size / 2.0).rem_euclid(size) - size / 2.0;
        let dx = wrap(p.x as f64 + 0.5 - self.cx) / self.scale;
        let dy = wrap(p.y as f64 + 0.5 - self.cy) / self.scale;
        let at = rect.center() + vec2((dx / ppp) as f32, (dy / ppp) as f32);
        if !rect.contains(at) {
            return;
        }
        let side = (1.0 / self.scale / ppp) as f32;
        let painter = ui.painter_at(rect);
        let ring = Stroke::new(2.0, Color32::WHITE);
        let halo = Stroke::new(4.0, Color32::from_black_alpha(160));
        if side >= 5.0 {
            let r = Rect::from_center_size(at, Vec2::splat(side + 3.0));
            painter.rect_stroke(r, 0.0, halo, egui::StrokeKind::Middle);
            painter.rect_stroke(r, 0.0, ring, egui::StrokeKind::Middle);
        } else {
            painter.circle_stroke(at, 9.0, halo);
            painter.circle_stroke(at, 9.0, ring);
        }
    }

    fn legend_box(&mut self, ui: &mut Ui2, rect: Rect) {
        let view = &CELL_VIEWS[self.cell_mode];
        let field = &FIELD_VIEWS[self.field_mode];
        let hi = field.hi.map_or(1.0, |k| self.st.hi[k] / self.brightness);
        let range = self.cell_range;
        let show_cells = self.show_cells;
        egui::Area::new(ui.scope_id().with("legend"))
            .order(egui::Order::Foreground)
            .pivot(Align2::LEFT_BOTTOM)
            .fixed_pos(rect.left_bottom() + vec2(10.0, -10.0))
            .show(ui.ctx(), |ui| {
                Frame::popup(ui.style())
                    .fill(Color32::from_rgba_unmultiplied(18, 20, 26, 225))
                    .show(ui, |ui| {
                        ui.set_width(250.0);
                        if show_cells {
                            legend::cell(ui, view, range);
                            ui.add_space(6.0);
                        }
                        legend::field(ui, field, hi);
                    });
            });
    }

    #[allow(clippy::too_many_arguments)]
    fn place_hint(&self, ui: &Ui2, response: &egui::Response, rect: Rect, ppp: f64, size: f64) {
        let zoom = 1.0 / self.scale / ppp;
        let mut text = if zoom >= 1.0 {
            format!("{zoom:.0} pixels per square")
        } else {
            format!("{:.1} squares per pixel", self.scale * ppp)
        };
        if let Some(at) = response.hover_pos() {
            let (x, y) = self.square_at(at, rect, ppp, size);
            text = format!("{x}, {y}    {text}");
        }
        let painter = ui.painter_at(rect);
        let at = rect.right_bottom() + vec2(-10.0, -8.0);
        let font = egui::FontId::proportional(12.0);
        let galley = painter.layout_no_wrap(text, font, Color32::from_gray(215));
        let r = Align2::RIGHT_BOTTOM
            .anchor_size(at, galley.size())
            .expand(4.0);
        painter.rect_filled(r, 3.0, Color32::from_black_alpha(150));
        painter.galley(r.min + vec2(4.0, 4.0), galley, Color32::WHITE);
    }

    fn busy_box(&mut self, ui: &mut Ui2, rect: Rect) {
        let busy = self.st.busy.clone();
        let message = self.st.message.clone();
        let no_world = self.st.view.is_none();
        if busy.is_none() && message.is_none() && !no_world {
            return;
        }
        let progress = self.progress.load(Ordering::Relaxed) as f32 / 1000.0;
        let mut clear = false;
        let mut open_new = false;
        egui::Area::new(ui.scope_id().with("notice"))
            .order(egui::Order::Foreground)
            .pivot(Align2::CENTER_TOP)
            .fixed_pos(rect.center_top() + vec2(0.0, 14.0))
            .show(ui.ctx(), |ui| {
                Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_max_width(520.0);
                    if let Some(b) = &busy {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(b);
                        });
                        if progress > 0.0 {
                            ui.add(ProgressBar::new(progress).desired_width(320.0));
                        }
                    }
                    if let Some((bad, m)) = &message {
                        let c = if *bad {
                            Color32::from_rgb(255, 150, 130)
                        } else {
                            Color32::from_rgb(160, 225, 170)
                        };
                        ui.colored_label(c, m);
                        if ui.small_button("OK").clicked() {
                            clear = true;
                        }
                    }
                    if no_world && busy.is_none() && ui.button("Start a new world").clicked() {
                        open_new = true;
                    }
                });
            });
        if clear {
            self.status.lock().unwrap().message = None;
        }
        if open_new {
            self.new.open = true;
        }
    }

    /// VRAM a world may use: what's free, plus what the world it replaces will give back, less the reserve.
    fn room_for_world(&self) -> Option<u64> {
        if self.st.software {
            return None;
        }
        free_vram().map(|free| (free + self.st.vram).saturating_sub(VRAM_MARGIN))
    }

    fn new_world_window(&mut self, ctx: Context) {
        if !self.new.open {
            return;
        }
        let mut open = true;
        let mut start = false;
        let mut seed_browser = self.browser.take();
        egui::Window::new("New world")
            .open(&mut open)
            .collapsible(false)
            .default_width(460.0)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(&ctx, |ui| {
                ui.label("Squares wide (and high)");
                ui.horizontal_wrapped(|ui| {
                    for size in [256u32, 512, 1024, 2048, 4096, 8192] {
                        if ui
                            .selectable_label(self.new.size == size, format!("{size}"))
                            .clicked()
                        {
                            self.new.size = size;
                            self.new.slots = 0;
                        }
                    }
                });
                // Read after the size buttons, so the numbers below are for the size just picked.
                let Ok(mut cfg) =
                    Config::new(self.new.size, self.new.seed, self.shaders.clone())
                else {
                    return;
                };
                cfg.hidden_rows = self.new.hidden_rows;
                cfg.shard_slots = cfg.max_shard_slots();
                let limit = crate::sim::buffer_limit(&self.gpu);
                let most = (cfg.max_shard_slots() as u64 * 2).min(cfg.slots_within(limit) as u64);
                let most = most.min(cfg.size as u64 * cfg.size as u64) as u32;
                if self.new.slots == 0 {
                    self.new.slots = cfg.default_slots().min(most);
                }
                self.new.slots = self.new.slots.clamp(1024, most.max(1024));
                cfg.slots = self.new.slots;
                ui.horizontal(|ui| {
                    ui.label("Seed");
                    ui.add(egui::DragValue::new(&mut self.new.seed));
                    if ui.button("Random").clicked() {
                        let t = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_or(1, |d| d.subsec_nanos());
                        self.new.seed = t % 100000;
                    }
                });
                ui.add(
                    Slider::new(&mut self.new.slots, 1024..=most.max(1025))
                        .logarithmic(true)
                        .text("most cells it can hold"),
                );
                ui.add(
                    Slider::new(&mut self.new.hidden_rows, 1..=4)
                        .text("memory rows per brain"),
                )
                .on_hover_text("Each row is four memory units. More rows make every genome bigger.");
                ui.checkbox(&mut self.new.keep_settings, "Keep my changes from Variables")
                    .on_hover_text("Off starts from the defaults.");
                ui.add_space(4.0);
                ui.strong("Seed it with evolved life (optional)");
                ui.small("Pick saved smaller worlds. They're laid out as tiles, so populations that evolved apart meet at the borders.");
                let mut browser = match seed_browser.take() {
                    Some((Purpose::Seeds, b)) => b,
                    other => {
                        seed_browser = other;
                        Browser::new(SAVES_DIR)
                    }
                };
                if let Some(path) = browser.list(ui, 120.0, Some(&self.new.seeds)) {
                    if let Some(at) = self.new.seeds.iter().position(|s| *s == path) {
                        self.new.seeds.remove(at);
                    } else {
                        self.new.seeds.push(path);
                    }
                }
                if seed_browser.is_none() {
                    seed_browser = Some((Purpose::Seeds, browser));
                }
                ui.add_enabled(
                    !self.new.seeds.is_empty(),
                    Slider::new(&mut self.new.seed_keep, 0.01..=1.0)
                        .text("share of their cells to keep"),
                );
                ui.separator();
                let need = cfg.vram_bytes();
                let room = self.room_for_world();
                let fits = room.is_none_or(|r| need <= r);
                match room {
                    Some(r) => {
                        let text = format!(
                            "Needs {:.2} GB of VRAM. {:.2} GB can be used, keeping 3 GB spare.",
                            gb(need),
                            gb(r)
                        );
                        if fits {
                            ui.label(text);
                        } else {
                            ui.colored_label(Color32::from_rgb(255, 150, 130), format!(
                                "{text} This won't fit, and could freeze the desktop."
                            ));
                            ui.checkbox(&mut self.new.anyway, "Start it anyway");
                        }
                    }
                    None => {
                        ui.label(format!("Needs {:.2} GB of memory.", gb(need)));
                    }
                }
                let too_big = cfg.check_buffers(limit).err();
                if let Some(why) = &too_big {
                    ui.colored_label(
                        Color32::from_rgb(255, 150, 130),
                        format!("{why}, so it can't be made."),
                    );
                }
                if self.new.size >= 4096 {
                    ui.small("Worlds this big run a few ticks per second at best.");
                }
                ui.add_space(4.0);
                if ui
                    .add_enabled(
                        too_big.is_none() && (fits || self.new.anyway),
                        egui::Button::new(RichText::new("Start this world").strong()),
                    )
                    .clicked()
                {
                    start = true;
                }
            });
        self.browser = seed_browser.filter(|(p, _)| *p != Purpose::Seeds || (open && !start));
        if start {
            let defaults = Config::new(self.new.size, self.new.seed, self.shaders.clone());
            let physics = if self.new.keep_settings {
                self.edits.clone()
            } else {
                defaults.map(|c| c.physics).unwrap_or_default()
            };
            self.send(Cmd::New(Box::new(NewWorld {
                size: self.new.size,
                seed: self.new.seed,
                slots: self.new.slots,
                hidden_rows: self.new.hidden_rows,
                physics,
                seed_from: self.new.seeds.clone(),
                seed_keep: self.new.seed_keep,
                note: None,
            })));
            self.new.open = false;
            self.drawn = None;
        } else {
            self.new.open = open;
        }
    }

    fn file_window(&mut self, ctx: Context) {
        let purpose = match &self.browser {
            Some((p, _)) if *p != Purpose::Seeds => *p,
            _ => return,
        };
        let Some((_, mut browser)) = self.browser.take() else {
            return;
        };
        let title = if purpose == Purpose::Load {
            "Open a saved world"
        } else {
            "Save this world"
        };
        let mut open = true;
        let mut done = false;
        egui::Window::new(title)
            .open(&mut open)
            .collapsible(false)
            .default_width(520.0)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(&ctx, |ui| {
                if let Some(path) = browser.list(ui, 240.0, None) {
                    self.path_text = path;
                    self.overwrite_ok = false;
                }
                ui.horizontal(|ui| {
                    ui.label("File");
                    if ui
                        .add(egui::TextEdit::singleline(&mut self.path_text).desired_width(420.0))
                        .changed()
                    {
                        self.overwrite_ok = false;
                    }
                });
                if purpose == Purpose::Load {
                    let info = (!self.path_text.is_empty())
                        .then(|| Config::from_save(&self.path_text, self.shaders.clone()));
                    let mut can = false;
                    if info.is_none() {
                        ui.label("Pick a file, or type its path.");
                    }
                    if let Some(info) = info {
                        match info {
                            Ok(cfg) => {
                                let need = cfg.vram_bytes();
                                let room = self.room_for_world();
                                let fits = room.is_none_or(|r| need <= r);
                                let too_big =
                                    cfg.check_buffers(crate::sim::buffer_limit(&self.gpu)).err();
                                can = fits && too_big.is_none();
                                if let Some(why) = too_big {
                                    ui.colored_label(
                                        Color32::from_rgb(255, 150, 130),
                                        format!("{why}, so it can't be opened."),
                                    );
                                }
                                if fits {
                                    ui.label(format!(
                                        "{0} x {0}, needs {1:.2} GB of VRAM. This replaces the world that's running.",
                                        cfg.size,
                                        gb(need)
                                    ));
                                } else {
                                    ui.colored_label(
                                        Color32::from_rgb(255, 150, 130),
                                        format!(
                                            "Needs {:.2} GB of VRAM and only {:.2} GB can be used, keeping 3 GB spare.",
                                            gb(need),
                                            gb(room.unwrap_or(0))
                                        ),
                                    );
                                    ui.checkbox(&mut self.overwrite_ok, "Open it anyway");
                                    can = self.overwrite_ok;
                                }
                            }
                            Err(e) => {
                                ui.colored_label(Color32::from_rgb(255, 150, 130), e);
                            }
                        }
                    }
                    if ui
                        .add_enabled(can, egui::Button::new(RichText::new("Open").strong()))
                        .clicked()
                    {
                        self.send(Cmd::Load(self.path_text.clone()));
                        done = true;
                    }
                } else {
                    ui.label(format!(
                        "About {:.2} GB on disk. The world keeps running afterwards.",
                        gb(self.st.vram)
                    ));
                    let exists = std::path::Path::new(&self.path_text).exists();
                    if exists {
                        ui.colored_label(
                            Color32::from_rgb(255, 150, 130),
                            "That file already exists.",
                        );
                        ui.checkbox(&mut self.overwrite_ok, "Replace it");
                    }
                    if ui
                        .add_enabled(
                            !self.path_text.is_empty() && (!exists || self.overwrite_ok),
                            egui::Button::new(RichText::new("Save").strong()),
                        )
                        .clicked()
                    {
                        self.send(Cmd::Save(self.path_text.clone()));
                        done = true;
                    }
                }
            });
        if open && !done {
            self.browser = Some((purpose, browser));
        }
    }

    fn help_window(&mut self, ctx: Context) {
        let mut open = self.help;
        egui::Window::new("How to use this")
            .open(&mut open)
            .collapsible(false)
            .default_width(380.0)
            .show(&ctx, |ui| {
                egui::Grid::new("help").num_columns(2).striped(true).show(ui, |ui| {
                    for (a, b) in [
                        ("Drag", "Pan"),
                        ("Scroll", "Zoom at the cursor"),
                        ("Click a cell", "See its numbers"),
                        ("Right click, Esc", "Let go of the picked cell"),
                        ("F", "Fit the whole world"),
                        ("Space", "Pause and play"),
                        ("N or .", "Step one tick"),
                        ("+ and -", "Faster and slower"),
                        ("1 to 0", "Quick colourings for cells"),
                        ("Up and down", "Every colouring for cells"),
                        ("Shift + 1 to 0", "Quick fields under the cells"),
                        ("Left and right", "Every field under the cells"),
                        ("H", "Hide or show the cells"),
                        ("P", "Hide or show the side panels"),
                    ] {
                        ui.label(a);
                        ui.label(b);
                        ui.end_row();
                    }
                });
                ui.add_space(6.0);
                ui.small("The world wraps at its edges. When a repeat of it comes into view, the repeat is dimmed.");
            });
        self.help = open;
    }
}

type Ui2 = egui::Ui;

fn rate_text(tps: f64) -> String {
    if tps >= 100.0 {
        format!("{tps:.0}")
    } else {
        format!("{tps:.1}")
    }
}

fn stat(ui: &mut Ui2, name: &str, value: &str) -> egui::Response {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 3.0;
        ui.label(RichText::new(name).small().weak());
        ui.label(RichText::new(value).monospace().strong());
    })
    .response
}

fn row(ui: &mut Ui2, on: bool, name: &str, hint: Option<&str>) -> egui::Response {
    let height = 18.0;
    let (rect, response) =
        ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::click());
    let visuals = ui.style().interact_selectable(&response, on);
    if on || response.hovered() {
        ui.painter().rect_filled(rect, 3.0, visuals.bg_fill);
    }
    let font = egui::FontId::proportional(13.0);
    ui.painter().text(
        rect.left_center() + vec2(5.0, 0.0),
        Align2::LEFT_CENTER,
        name,
        font,
        visuals.text_color(),
    );
    if let Some(h) = hint {
        ui.painter().text(
            rect.right_center() - vec2(5.0, 0.0),
            Align2::RIGHT_CENTER,
            h,
            egui::FontId::proportional(10.5),
            ui.visuals().weak_text_color(),
        );
    }
    response
}

fn bar(ui: &mut Ui2, name: &str, v: f32, colour: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width() - 10.0, 16.0), Sense::hover());
    let label_w = 62.0;
    let painter = ui.painter();
    painter.text(
        rect.left_center(),
        Align2::LEFT_CENTER,
        name,
        egui::FontId::proportional(12.5),
        ui.visuals().text_color(),
    );
    let track = Rect::from_min_max(rect.min + vec2(label_w, 3.0), rect.max - vec2(44.0, 3.0));
    painter.rect_filled(track, 2.0, Color32::from_white_alpha(14));
    let v = v.clamp(0.0, 1.0);
    if v > 0.0 {
        let fill = Rect::from_min_size(track.min, vec2(track.width() * v, track.height()));
        painter.rect_filled(fill, 2.0, colour);
    }
    painter.text(
        rect.right_center(),
        Align2::RIGHT_CENTER,
        format!("{:.0}%", v * 100.0),
        egui::FontId::proportional(12.0),
        ui.visuals().text_color(),
    );
}

/// One setting's control. Returns whether it changed.
const LABEL_WIDTH: f32 = 112.0;

fn var_row(ui: &mut Ui2, v: &Var, edits: &mut Map<String, Value>, slider_width: f32) -> bool {
    let mut changed = false;
    let current = edits.get(&v.key).cloned().unwrap_or(v.default.clone());
    let differs = current != v.default;
    let name = v.key.replace('_', " ");
    let mut text = RichText::new(&name);
    if differs {
        text = text.color(Color32::from_rgb(240, 200, 110)).strong();
    }
    if v.scope != Scope::Live {
        text = text.italics();
    }
    let help = match v.scope {
        Scope::Live => v.help.to_string(),
        Scope::NewWorld => format!("{} Only takes effect in a new world.", v.help),
        Scope::Unused => format!("{} No shader reads this at the moment.", v.help),
    };
    let badge = match v.scope {
        Scope::Live => None,
        Scope::NewWorld => Some("new world only"),
        Scope::Unused => Some("unused"),
    };
    ui.horizontal(|ui| {
        match current {
            Value::Bool(mut on) => {
                let r = ui.checkbox(&mut on, text);
                if !help.is_empty() {
                    r.clone().on_hover_text(&help);
                }
                if r.changed() {
                    edits.insert(v.key.clone(), Value::Bool(on));
                    changed = true;
                }
            }
            _ => {
                let mut value = current.as_f64().unwrap_or(0.0);
                let label = ui
                    .allocate_ui_with_layout(
                        vec2(LABEL_WIDTH, 18.0),
                        Layout::left_to_right(Align::Center),
                        |ui| {
                            ui.set_min_width(LABEL_WIDTH);
                            ui.add(egui::Label::new(text).truncate())
                        },
                    )
                    .inner;
                if !help.is_empty() {
                    label.on_hover_text(&help);
                }
                ui.spacing_mut().slider_width = slider_width;
                let mut slider = Slider::new(&mut value, v.range.0..=v.range.1)
                    .clamping(egui::SliderClamping::Never);
                if v.log {
                    slider = slider.logarithmic(true).smallest_positive(v.smallest);
                }
                if v.whole {
                    slider = slider.integer();
                } else {
                    slider = slider.max_decimals(5);
                }
                let r = ui.add(slider);
                if !help.is_empty() {
                    r.clone().on_hover_text(&help);
                }
                if r.changed() && value.is_finite() && value >= 0.0 {
                    edits.insert(v.key.clone(), Value::from(value));
                    changed = true;
                }
            }
        }
        if differs
            && ui
                .small_button("reset")
                .on_hover_text(format!("Back to {}", v.default))
                .clicked()
        {
            edits.insert(v.key.clone(), v.default.clone());
            changed = true;
        }
    });
    if let Some(b) = badge {
        ui.small(RichText::new(format!("{name}: {b}")).color(Color32::from_rgb(190, 170, 255)));
    }
    changed
}
