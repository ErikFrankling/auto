//! The window's contents: the world in the middle, what to colour it by on the left, and every
//! physics setting on the right.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use auto_core::Gpu;
use auto_life_gpu::config::Scope;
use auto_life_soft::dump::Palette;
use auto_life_soft::view::{CELL_VIEWS, Canvas, FIELD_VIEWS, SHOW_CELLS, ViewParams};
use auto_life_soft::{Config, VRAM_MARGIN, free_vram};
use egui::{
    Align, Color32, Context, CornerRadius, Event, Key, Layout, Rect, RichText, ScrollArea, Sense, Slider,
    TextureId, Ui as Ui2, pos2, vec2,
};
use serde_json::{Map, Value};

use crate::sim::{Cmd, Status};

pub fn style(ctx: &Context) {
    ctx.set_visuals(egui::Visuals::dark());
}

/// Settings, sorted into sections for a person to find.
const GROUPS: [&str; 10] = [
    "Land and sun",
    "Rain and genes",
    "Brain",
    "Budget and upkeep",
    "Body and forces",
    "Biting and giving",
    "Food and enzyme",
    "Meat",
    "Heat",
    "Scent and bookkeeping",
];

fn group_of(key: &str) -> &'static str {
    match key {
        "sun" | "food_cap" | "food_flow" | "food_start_ticks" | "sun_drift" | "rock" | "terrain_contrast"
        | "batch_tile" | "batch_wall" => "Land and sun",
        k if k.starts_with("rain_") || k.starts_with("mut_") || k.starts_with("gene_") => "Rain and genes",
        "brain_every" | "memory_on" | "voice_on" | "turn_max" => "Brain",
        "spend" | "spend_share" | "commitment" | "upkeep" | "upkeep_e" | "rest_saving" | "hazard" | "max_energy"
        | "body" | "womb_cap" | "split_on" | "split_share_max" => "Budget and upkeep",
        "r0" | "e_size" | "r_max" | "drag" | "step_max" | "crowd_on" | "k_push" | "gap" | "glue_sharp" => {
            "Body and forces"
        }
        k if k.starts_with("stick_") || k.starts_with("push_") || k.starts_with("thrust_") => "Body and forces",
        k if k.starts_with("bite_") || k.starts_with("give_") => "Biting and giving",
        "eat_rate" | "eat_max" | "locked" => "Food and enzyme",
        k if k.starts_with("enzyme_") => "Food and enzyme",
        k if k.starts_with("meat_") => "Meat",
        k if k.starts_with("heat_") => "Heat",
        _ => "Scent and bookkeeping",
    }
}

fn help_of(key: &str) -> &'static str {
    match key {
        "sun" => "Mean sunlight per square, per tick. How fast food grows.",
        "food_cap" => "The most food a square can hold.",
        "food_flow" => "How fast food spreads between squares.",
        "food_start_ticks" => "A new world's food, as this many ticks of sun.",
        "sun_drift" => "How fast the sun map slides across the world. 0 keeps patches still.",
        "rock" => "Whether new worlds have rock ridges.",
        "terrain_contrast" => "How much rich and poor land differ.",
        "batch_tile" => "Width of each walled-off world on one map. 0 means one world.",
        "batch_wall" => "How thick the rock between batched worlds is.",
        "rain_p" => "How often a random new cell lands.",
        "rain_energy" => "Energy a landing cell starts with.",
        "rain_burst_ticks" => "For this many ticks, cells land much faster until 1% of the world is covered.",
        "gene_init" => "Spread of the weights in a landing cell's random genome.",
        "mut_rate" => "Chance each weight mutates at birth, for a landing cell. Then it's inherited.",
        "mut_size" => "How big a mutation is.",
        "mut_heritable" => "Whether each cell's mutation rate is inherited with a nudge.",
        "mut_rate_step" => "How big the nudge to a child's mutation rate is.",
        "brain_every" => "A cell thinks once in this many ticks, staggered by slot.",
        "memory_on" => "Whether brains remember between thoughts.",
        "voice_on" => "Whether cells can show neighbours their two voice outputs.",
        "turn_max" => "The most a cell can turn in one thought, in radians.",
        "spend" => "The most energy a cell can spend per tick.",
        "spend_share" => "The share of its energy a cell can spend per tick, up to that limit.",
        "commitment" => "Spending a little on an action barely works. Bigger wastes more on dabbling.",
        "upkeep" => "Energy lost per tick just for being alive.",
        "upkeep_e" => "Extra upkeep per unit of stored energy.",
        "rest_saving" => "How much of its upkeep a fully resting cell saves.",
        "hazard" => "Chance per tick of dying of old age, at any age.",
        "max_energy" => "The most energy a cell can hold.",
        "body" => "Energy locked in a body, left as meat at death. A womb must cover this to split.",
        "womb_cap" => "The most a womb can hold.",
        "split_on" => "Whether cells can split at all.",
        "split_share_max" => "The most of its energy a parent can hand a child.",
        "r0" => "Radius of a cell with no energy.",
        "e_size" => "How much energy it takes to grow a cell's area by one more r0 disc.",
        "r_max" => "The largest radius a cell can have.",
        "drag" => "Velocity is force over this.",
        "step_max" => "The longest step in one tick.",
        "crowd_on" => "Whether overlapping cells push apart.",
        "k_push" => "How hard overlap pushes.",
        "gap" => "How far beyond contact adhesion still pulls.",
        "stick_on" => "Whether cells can stick. Off is the control for everything about bodies.",
        "stick_max" => "Adhesion between two fully sticking cells with the same glue.",
        "stick_half" => "Effort at which a cell's stick is half of its most.",
        "stick_fee" => "Energy each side pays per tick for each contact, at full strength.",
        "glue_sharp" => "How fast adhesion falls as two cells' glue differs.",
        "push_on" => "Whether cells can shove what's in front of them.",
        "push_power" => "Shoving force per unit of effort.",
        "thrust_on" => "Whether cells can move themselves.",
        "thrust_rate" => "Thrust per unit of effort.",
        "bite_on" => "Whether cells can bite.",
        "bite_power" => "How much energy a bite asks for, per unit of effort.",
        "bite_yield" => "The share of a bite the biter keeps. The rest is shed.",
        "give_on" => "Whether cells can give energy to contacts.",
        "give_power" => "How much energy is given, per unit of effort.",
        "give_yield" => "The share of a gift that arrives.",
        "eat_rate" => "How much food a cell wants, per unit of effort.",
        "eat_max" => "The most food a cell can take per tick.",
        "locked" => "The share of food that a cell can't reach without enzyme. 0 removes the rule.",
        "enzyme_half" => "Enzyme level that unlocks half of the locked share.",
        "enzyme_on" => "Whether cells can secrete enzyme.",
        "enzyme_power" => "Enzyme made per unit of effort.",
        "meat_on" => "Whether cells can eat meat.",
        "meat_rate" => "How much meat a cell wants, per unit of effort.",
        "meat_max" => "The most meat a cell can take per tick.",
        "meat_yield" => "The share of meat eaten that becomes energy.",
        "meat_rot" => "How fast meat rots into food.",
        "heat_on" => "Whether cells can make heat on purpose.",
        "heat_rate" => "Heat shed per unit of energy spent. Every action does this.",
        "heat_power" => "Heat made on purpose, per unit of effort.",
        "heat_harm" => "Energy lost per tick per unit of heat, by a cell with no reserves. 0 removes it.",
        "heat_scale" => "Energy at which the harm from heat has halved.",
        "scent_on" => "Whether cells can emit scent.",
        "emit_power" => "Scent made per unit spent.",
        "ledger" => "Keep an exact energy count, for checking the physics. Slower.",
        k if k.ends_with("_flow") => "How fast this spreads to each neighbouring square.",
        k if k.ends_with("_decay") => "The share of this lost per tick.",
        _ => "",
    }
}

fn range_of(key: &str, default: f64) -> (f64, f64, bool, bool) {
    if key.ends_with("_ticks") || key == "batch_tile" {
        return (0.0, (default * 8.0).max(10000.0), false, true);
    }
    if key == "brain_every" {
        return (1.0, 16.0, false, true);
    }
    if key == "batch_wall" {
        return (5.0, 32.0, false, true);
    }
    if key.ends_with("_flow") {
        // Jacobi with four neighbours overshoots at a quarter.
        return (0.0, 0.249, false, false);
    }
    if key.ends_with("_decay") || key.ends_with("_yield") || key.ends_with("_saving") || key == "locked" {
        return (0.0, 1.0, false, false);
    }
    if key == "r_max" {
        return (0.1, 0.9, false, false);
    }
    if key == "r0" {
        return (0.05, 0.9, false, false);
    }
    if key == "gap" {
        return (0.01, 0.2, false, false);
    }
    if default <= 0.0 {
        return (0.0, 1.0, false, false);
    }
    if default < 0.02 {
        return (0.0, (default * 40.0).max(1e-3), true, false);
    }
    (0.0, default * 5.0, default < 0.1, false)
}

struct Var {
    key: String,
    scope: Scope,
    default: Value,
    range: (f64, f64),
    log: bool,
    whole: bool,
    smallest: f64,
}

fn all_vars(cfg: &Config) -> Vec<Var> {
    cfg.physics
        .iter()
        .map(|(key, default)| {
            let d = default.as_f64().unwrap_or(0.0);
            let (lo, hi, log, whole) = range_of(key, d);
            Var {
                key: key.clone(),
                scope: cfg.scope(key),
                default: default.clone(),
                range: (lo, hi),
                log,
                whole,
                smallest: if d > 0.0 { (d / 100.0).min(1e-3) } else { 1e-3 },
            }
        })
        .collect()
}

fn var_row(ui: &mut Ui2, v: &Var, edits: &mut Map<String, Value>) -> bool {
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
        Scope::Live => help_of(&v.key).to_string(),
        Scope::NewWorld => format!("{} Only takes effect in a new world.", help_of(&v.key)),
        Scope::Unused => format!("{} No shader reads this at the moment.", help_of(&v.key)),
    };
    ui.horizontal(|ui| {
        match current {
            Value::Bool(mut on) => {
                let r = ui.checkbox(&mut on, text).on_hover_text(&help);
                if r.changed() {
                    edits.insert(v.key.clone(), Value::Bool(on));
                    changed = true;
                }
            }
            _ => {
                let mut value = current.as_f64().unwrap_or(0.0);
                ui.allocate_ui_with_layout(vec2(112.0, 18.0), Layout::left_to_right(Align::Center), |ui| {
                    ui.set_min_width(112.0);
                    ui.add(egui::Label::new(text).truncate()).on_hover_text(&help);
                });
                ui.spacing_mut().slider_width = 110.0;
                let mut slider = Slider::new(&mut value, v.range.0..=v.range.1).clamping(egui::SliderClamping::Never);
                if v.log {
                    slider = slider.logarithmic(true).smallest_positive(v.smallest);
                }
                slider = if v.whole { slider.integer() } else { slider.max_decimals(5) };
                let r = ui.add(slider).on_hover_text(&help);
                if r.changed() && value.is_finite() && value >= 0.0 {
                    edits.insert(v.key.clone(), Value::from(value));
                    changed = true;
                }
            }
        }
        if differs
            && ui.small_button("reset").on_hover_text(format!("Back to {}", v.default)).clicked()
        {
            edits.insert(v.key.clone(), v.default.clone());
            changed = true;
        }
    });
    changed
}

pub struct Ui {
    pub ctx: Context,
    gpu: Arc<Gpu>,
    layout: wgpu::BindGroupLayout,
    status: Arc<Mutex<Status>>,
    tx: Sender<Cmd>,
    shaders: Option<PathBuf>,
    st: Status,
    world_id: u64,
    canvas: Option<Canvas>,
    texture: Option<TextureId>,
    drawing: Arc<AtomicBool>,
    last_draw: Instant,
    drawn: Option<(u64, u64, [u8; 64])>,
    cx: f64,
    cy: f64,
    scale: f64,
    fit_next: bool,
    panning: Option<(egui::Pos2, f32)>,
    running: bool,
    full_speed: bool,
    rate: f64,
    cell_view: usize,
    field_view: usize,
    field_opacity: f32,
    cell_opacity: f32,
    show_cells: bool,
    vars: Vec<Var>,
    edits: Map<String, Value>,
    dirty_at: Option<Instant>,
    filter: String,
    new_size: u32,
    new_seed: u32,
    path: String,
    good_news_since: Option<Instant>,
}

impl Ui {
    pub fn new(
        ctx: Context,
        gpu: Arc<Gpu>,
        layout: wgpu::BindGroupLayout,
        status: Arc<Mutex<Status>>,
        tx: Sender<Cmd>,
        shaders: Option<PathBuf>,
        seed: u32,
    ) -> Ui {
        Ui {
            ctx,
            gpu,
            layout,
            status,
            tx,
            shaders,
            st: Status::default(),
            world_id: 0,
            canvas: None,
            texture: None,
            drawing: Arc::new(AtomicBool::new(false)),
            last_draw: Instant::now(),
            drawn: None,
            cx: 0.0,
            cy: 0.0,
            scale: 1.0,
            fit_next: true,
            panning: None,
            running: true,
            full_speed: true,
            rate: 200.0,
            cell_view: 0,
            field_view: 0,
            field_opacity: 0.8,
            cell_opacity: 1.0,
            show_cells: true,
            vars: vec![],
            edits: Map::new(),
            dirty_at: None,
            filter: String::new(),
            new_size: 256,
            new_seed: seed,
            path: "soft-world.bin".to_string(),
            good_news_since: None,
        }
    }

    pub fn title(&self) -> String {
        format!(
            "Soft life: {} wide, seed {}, tick {}, {} cells",
            self.st.size, self.st.seed, self.st.tick, self.st.cells
        )
    }

    pub fn wants_frames(&self) -> bool {
        self.running || self.drawing.load(Ordering::Relaxed)
    }

    fn send(&self, cmd: Cmd) {
        let _ = self.tx.send(cmd);
    }

    fn set_running(&mut self, on: bool) {
        self.running = on;
        self.send(Cmd::Run(on));
    }

    fn set_speed(&mut self) {
        self.send(Cmd::Speed(if self.full_speed { None } else { Some(self.rate) }));
    }

    pub fn show(&mut self, root: &mut Ui2, renderer: &mut egui_wgpu::Renderer) {
        self.st = self.status.lock().unwrap().clone();
        // Good news goes away by itself. Errors stay until they're read.
        if matches!(self.st.message, Some((false, _))) {
            let since = *self.good_news_since.get_or_insert_with(Instant::now);
            if since.elapsed().as_secs() >= 6 {
                self.status.lock().unwrap().message = None;
                self.good_news_since = None;
            }
        } else {
            self.good_news_since = None;
        }
        if self.st.world_id != self.world_id {
            self.world_id = self.st.world_id;
            self.fit_next = true;
            self.running = true;
            if let Ok(d) = Config::new(256, 1, self.shaders.clone()) {
                self.vars = all_vars(&d);
            }
            self.edits = self.st.physics.clone();
            self.dirty_at = None;
            self.new_size = self.st.size;
        }
        if let Some(at) = self.dirty_at
            && at.elapsed().as_millis() > 180
        {
            self.dirty_at = None;
            self.send(Cmd::Physics(self.edits.clone()));
        }
        self.keys(root.ctx().clone());

        egui::Panel::top("controls").show(root, |ui| self.top_bar(ui));
        let window = root.ctx().content_rect().width();
        egui::Panel::left("views")
            .resizable(true)
            .default_size((window * 0.16).clamp(170.0, 220.0))
            .min_size(150.0)
            .show(root, |ui| self.views_panel(ui));
        egui::Panel::right("settings")
            .resizable(true)
            .default_size((window * 0.24).clamp(280.0, 360.0))
            .min_size(240.0)
            .show(root, |ui| self.settings_panel(ui));
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(root, |ui| self.world_view(ui, renderer));
    }

    fn keys(&mut self, ctx: Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let (space, fit, step) = ctx.input(|i| (i.key_pressed(Key::Space), i.key_pressed(Key::F), i.key_pressed(Key::Period)));
        if space {
            self.set_running(!self.running);
        }
        if fit {
            self.fit_next = true;
        }
        if step {
            self.running = false;
            self.send(Cmd::Step);
        }
    }

    fn top_bar(&mut self, ui: &mut Ui2) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button(if self.running { "Pause" } else { "Play" })
                .on_hover_text("Space")
                .clicked()
            {
                self.set_running(!self.running);
            }
            if ui.button("Step").on_hover_text("One tick. Full stop key").clicked() {
                self.running = false;
                self.send(Cmd::Step);
            }
            if ui.checkbox(&mut self.full_speed, "Full speed").changed() {
                self.set_speed();
            }
            ui.add_enabled_ui(!self.full_speed, |ui| {
                ui.spacing_mut().slider_width = 130.0;
                if ui
                    .add(Slider::new(&mut self.rate, 1.0..=5000.0).logarithmic(true).suffix(" ticks/s").max_decimals(0))
                    .changed()
                {
                    self.set_speed();
                }
            });
            ui.separator();
            ui.label(format!("tick {}", self.st.tick));
            ui.label(format!("{} cells", self.st.cells));
            ui.label(format!("{:.0} ticks/s", self.st.tps));
            ui.label(format!("{:.2} ms GPU per tick", self.st.gpu_ms_per_tick));
            ui.label(format!("longest {:.0} ms", self.st.longest_ms))
                .on_hover_text("The longest the GPU was held at once. Over about 20 ms and the desktop can stutter.");
            if self.st.overflow > 0 {
                ui.label(format!("{} left out of full bins", self.st.overflow))
                    .on_hover_text("A bin holds 8. Any more sit out that tick. A few are fine.");
            }
            if self.st.sharing {
                ui.label(RichText::new("sharing the GPU").color(Color32::from_rgb(190, 170, 255)));
            }
            if self.st.software {
                ui.label(RichText::new("software GPU").color(Color32::from_rgb(240, 200, 110)));
            }
        });
        if let Some((bad, text)) = &self.st.message {
            let colour = if *bad { Color32::from_rgb(255, 130, 110) } else { Color32::from_rgb(140, 220, 140) };
            ui.label(RichText::new(text).color(colour));
        }
        if let Some(what) = &self.st.busy {
            ui.label(RichText::new(what).color(Color32::from_rgb(240, 200, 110)));
        }
    }

    fn views_panel(&mut self, ui: &mut Ui2) {
        ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Cells");
            ui.checkbox(&mut self.show_cells, "Show cells");
            ui.add(Slider::new(&mut self.cell_opacity, 0.0..=1.0).text("opacity"));
            for (i, (name, help, _)) in CELL_VIEWS.iter().enumerate() {
                if ui.selectable_label(self.cell_view == i, *name).on_hover_text(*help).clicked() {
                    self.cell_view = i;
                }
            }
            ui.separator();
            ui.heading("Under them");
            ui.add(Slider::new(&mut self.field_opacity, 0.0..=1.0).text("opacity"));
            for (i, (name, help)) in FIELD_VIEWS.iter().enumerate() {
                if ui.selectable_label(self.field_view == i, *name).on_hover_text(*help).clicked() {
                    self.field_view = i;
                }
            }
            ui.separator();
            ui.heading("World");
            ui.horizontal(|ui| {
                ui.label("Size");
                egui::ComboBox::from_id_salt("size")
                    .selected_text(format!("{}", self.new_size))
                    .show_ui(ui, |ui| {
                        for s in [64u32, 128, 256, 512, 1024, 2048, 4096] {
                            ui.selectable_value(&mut self.new_size, s, format!("{s}"));
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Seed");
                ui.add(egui::DragValue::new(&mut self.new_seed));
                if ui.small_button("next").clicked() {
                    self.new_seed = self.new_seed.wrapping_add(1);
                }
            });
            let need = Config::new(self.new_size, self.new_seed, self.shaders.clone())
                .map(|c| c.vram_bytes())
                .unwrap_or(0);
            ui.small(format!("About {:.1} GB of VRAM", need as f64 / 1e9));
            let free = free_vram().map(|f| f + self.st.vram);
            let fits = self.st.software || free.is_none_or(|f| need + VRAM_MARGIN <= f);
            if !fits {
                ui.small(RichText::new("That won't fit in what's free, keeping 3 GB spare.").color(Color32::from_rgb(255, 130, 110)));
            }
            if ui.add_enabled(fits, egui::Button::new("New world")).clicked() {
                self.send(Cmd::New {
                    size: self.new_size,
                    seed: self.new_seed,
                    physics: self.edits.clone(),
                });
            }
            ui.separator();
            ui.heading("File");
            ui.text_edit_singleline(&mut self.path);
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    self.send(Cmd::Save(self.path.clone()));
                }
                if ui.button("Load").clicked() {
                    self.send(Cmd::Load(self.path.clone()));
                }
            });
            ui.separator();
            ui.small("Drag to move, scroll to zoom, F to see it all. Zoom in to see cells as shapes.");
        });
    }

    fn settings_panel(&mut self, ui: &mut Ui2) {
        ui.heading("Physics");
        ui.horizontal(|ui| {
            ui.label("Find");
            ui.text_edit_singleline(&mut self.filter);
        });
        ui.small("Changes apply to the running world, except the slanted ones, which need a new world.");
        if ui.small_button("Back to defaults").clicked() {
            for v in &self.vars {
                self.edits.insert(v.key.clone(), v.default.clone());
            }
            self.dirty_at = Some(Instant::now());
        }
        ui.separator();
        let filter = self.filter.to_lowercase().replace(' ', "_");
        let mut changed = false;
        ScrollArea::vertical().show(ui, |ui| {
            for group in GROUPS {
                let vars: Vec<&Var> = self
                    .vars
                    .iter()
                    .filter(|v| group_of(&v.key) == group && v.key.contains(&filter))
                    .collect();
                if vars.is_empty() {
                    continue;
                }
                egui::CollapsingHeader::new(group)
                    .default_open(!filter.is_empty() || group == "Body and forces")
                    .show(ui, |ui| {
                        for v in vars {
                            changed |= var_row(ui, v, &mut self.edits);
                        }
                    });
            }
        });
        if changed {
            self.dirty_at = Some(Instant::now());
        }
    }

    fn view_params(&self, width: u32, height: u32) -> ViewParams {
        let fallback;
        let palette: &Palette = match &self.st.palette {
            Some(p) => p,
            None => {
                fallback = Palette::fit(&[]);
                &fallback
            }
        };
        let mut p = ViewParams::new(palette, self.st.tops);
        p.origin = [
            (self.cx - width as f64 * 0.5 * self.scale) as f32,
            (self.cy - height as f64 * 0.5 * self.scale) as f32,
        ];
        p.scale = self.scale as f32;
        p.cell_mode = CELL_VIEWS[self.cell_view].2;
        p.field_mode = self.field_view as u32;
        p.field_opacity = self.field_opacity;
        p.cell_opacity = self.cell_opacity;
        p.center = [self.cx as f32, self.cy as f32];
        p.flags = if self.show_cells { SHOW_CELLS } else { 0 };
        p
    }

    fn world_view(&mut self, ui: &mut Ui2, renderer: &mut egui_wgpu::Renderer) {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let ppp = ui.ctx().pixels_per_point() as f64;
        let width = ((rect.width() as f64 * ppp).round() as u32).max(16);
        let height = ((rect.height() as f64 * ppp).round() as u32).max(16);
        ui.painter().rect_filled(rect, CornerRadius::ZERO, Color32::BLACK);

        if self.canvas.as_ref().is_none_or(|c| c.width != width || c.height != height) {
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
        }

        let size = self.st.size as f64;
        if self.st.view.is_some() && size > 0.0 {
            let fit = size / width.min(height) as f64;
            if self.fit_next {
                self.cx = size / 2.0;
                self.cy = size / 2.0;
                self.scale = fit;
                self.fit_next = false;
            }
            self.camera(ui, &response, rect, ppp, fit);
            let params = self.view_params(width, height);
            let mut key_bytes = [0u8; 64];
            key_bytes.copy_from_slice(&bytemuck::bytes_of(&params)[..64]);
            let key = (self.world_id, self.st.tick, key_bytes);
            let gap = if self.st.software { 80 } else { 16 };
            if self.drawn != Some(key)
                && self.last_draw.elapsed().as_millis() >= gap
                && !self.drawing.load(Ordering::Relaxed)
                && let (Some(view), Some(canvas)) = (&self.st.view, &self.canvas)
            {
                self.drawing.store(true, Ordering::Relaxed);
                canvas.draw(&self.gpu.device, &self.gpu.queue, view, self.st.flip, params);
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
                ui.painter().image(id, rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            }
        }
    }

    fn camera(&mut self, ui: &mut Ui2, response: &egui::Response, rect: Rect, ppp: f64, fit: f64) {
        let size = self.st.size as f64;
        // Read from the raw pointer events, so a drag counts however few frames it spans.
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
                    self.panning = (*pressed && on_world).then_some((*pos, 0.0));
                }
                Event::PointerMoved(pos) => {
                    if let Some((last, moved)) = &mut self.panning {
                        let step = *pos - *last;
                        *last = *pos;
                        *moved += step.length();
                        if *moved > 3.0 {
                            self.cx -= step.x as f64 * ppp * self.scale;
                            self.cy -= step.y as f64 * ppp * self.scale;
                        }
                    }
                }
                Event::PointerGone => self.panning = None,
                _ => {}
            }
        }
        if response.hovered() {
            let (scroll, pinch, cursor) =
                ui.ctx().input(|i| (i.smooth_scroll_delta.y, i.zoom_delta(), i.pointer.hover_pos()));
            let factor = (-(scroll as f64) * 0.0028).exp() / pinch as f64;
            if factor != 1.0
                && let Some(cursor) = cursor
            {
                let new = (self.scale * factor).clamp(1.0 / 96.0, fit * 2.0);
                // The square under the cursor stays under the cursor.
                let off_x = (cursor.x - rect.center().x) as f64 * ppp;
                let off_y = (cursor.y - rect.center().y) as f64 * ppp;
                self.cx += off_x * (self.scale - new);
                self.cy += off_y * (self.scale - new);
                self.scale = new;
            }
        }
        self.cx = self.cx.rem_euclid(size);
        self.cy = self.cy.rem_euclid(size);
    }
}
