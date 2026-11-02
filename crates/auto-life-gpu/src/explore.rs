//! Looking at a world: the tables of what cells and squares can be coloured by, and
//! the pass that draws them. Reads only the cell, stat and field buffers. Adding a
//! view is one row in a table.

use bytemuck::{Pod, Zeroable};

use crate::report::Sample;
use crate::world::{Access2, World};

const SHADER: &str = include_str!("../shaders/explore.wgsl");

/// How a reading turns into a colour, which is also how its legend is drawn.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Paint {
    /// Low to high, one scale.
    Scale,
    /// Either side of a middle.
    Sides,
    /// A colour for each of these.
    Kinds(&'static [(&'static str, [f32; 3])]),
    /// Described in words, because the colour itself is what's being compared.
    Words(&'static str),
    /// A disc, for directions.
    Wheel,
    /// Two signals on two axes.
    Square,
}

pub struct CellView {
    pub name: &'static str,
    pub group: &'static str,
    pub help: &'static str,
    pub paint: Paint,
    /// WGSL statements that return the colour, with `s` the cell's slot.
    pub wgsl: &'static str,
    /// The range when not fitted to the population. High of 0 means there's no number to fit.
    pub range: (f32, f32),
    /// The same reading on a sampled cell, for fitting the range to what's alive.
    pub reading: Option<fn(&Sample) -> f32>,
}

#[allow(clippy::excessive_precision)]
const SEQ_C: [[f32; 3]; 7] = [
    [0.2777273, 0.0054073, 0.3340998],
    [0.1050930, 1.4046135, 1.3845901],
    [-0.3308618, 0.2148476, 0.0950952],
    [-4.6342305, -5.7991005, -19.3324410],
    [6.2282699, 14.1799333, 56.6905526],
    [4.7763850, -13.7451460, -65.3530326],
    [-5.4354559, 4.6458523, 26.3124352],
];

/// The same scale the shader uses, for legends.
pub fn seq(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let mut out = [0.0; 3];
    for (k, v) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for c in SEQ_C.iter().rev() {
            acc = acc * t + c[k];
        }
        *v = acc.clamp(0.0, 1.0);
    }
    out
}

/// The same two-sided scale the shader uses, for legends.
pub fn div(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0) * 2.0 - 1.0;
    let (mid, end, k) = if t < 0.0 {
        ([0.62, 0.62, 0.64], [0.13, 0.47, 0.95], -t)
    } else {
        ([0.62, 0.62, 0.64], [0.98, 0.42, 0.10], t)
    };
    std::array::from_fn(|i| mid[i] + (end[i] - mid[i]) * k)
}

/// The same colour wheel the shader uses, for legends.
pub fn hue(h: f32, s: f32, v: f32) -> [f32; 3] {
    let offset = [1.0, 2.0 / 3.0, 1.0 / 3.0];
    std::array::from_fn(|i| {
        let k = (h + offset[i]).fract() * 6.0;
        let x = ((k - 3.0).abs() - 1.0).clamp(0.0, 1.0);
        v * (1.0 + (x - 1.0) * s)
    })
}

const YES_NO: Paint = Paint::Kinds(&[("yes", [1.0, 0.86, 0.2]), ("no", [0.16, 0.18, 0.26])]);
const MAIN_ACTION: Paint = Paint::Kinds(&[
    ("rest", [0.45, 0.47, 0.55]),
    ("move", [0.95, 0.95, 0.95]),
    ("eat ground", [0.20, 0.80, 0.30]),
    ("eat meat", [0.85, 0.20, 0.25]),
    ("grow child", [0.95, 0.55, 0.75]),
    ("emit scent", [0.30, 0.65, 1.00]),
    ("heat", [1.00, 0.55, 0.10]),
    ("bite", [0.60, 0.05, 0.10]),
    ("give", [0.65, 0.40, 1.00]),
    ("enzyme", [1.00, 0.90, 0.20]),
    ("armour", [0.55, 0.80, 0.85]),
    ("clean waste", [0.55, 0.40, 0.20]),
]);

fn share(d: &[f32; 4], k: usize) -> f32 {
    d[k] / (d[0] + d[1] + d[2] + d[3]).max(1e-3)
}

macro_rules! scale {
    ($name:literal, $group:literal, $help:literal, $wgsl:literal, $range:expr, $read:expr) => {
        CellView {
            name: $name,
            group: $group,
            help: $help,
            paint: Paint::Scale,
            wgsl: concat!("return seq(norm(", $wgsl, "));"),
            range: $range,
            reading: Some($read),
        }
    };
}
macro_rules! sides {
    ($name:literal, $group:literal, $help:literal, $wgsl:literal, $read:expr) => {
        CellView {
            name: $name,
            group: $group,
            help: $help,
            paint: Paint::Sides,
            wgsl: concat!("return div(norm(", $wgsl, "));"),
            range: (-1.0, 1.0),
            reading: Some($read),
        }
    };
}
macro_rules! yes_no {
    ($name:literal, $group:literal, $help:literal, $wgsl:literal) => {
        CellView {
            name: $name,
            group: $group,
            help: $help,
            paint: YES_NO,
            wgsl: concat!("return flag((cell[s].flags & ", $wgsl, ") != 0u);"),
            range: (0.0, 0.0),
            reading: None,
        }
    };
}

/// Everything cells can be coloured by. Order is the order of the list.
pub const CELL_VIEWS: &[CellView] = &[
    CellView {
        name: "Genome",
        group: "Who",
        help: "Cells with similar genes get similar colours, along the main axes the population differs on.",
        paint: Paint::Words(
            "Similar colour means similar genes. The colours follow the population's main differences, so they drift slowly.",
        ),
        wgsl: "let g = cell[s].rgb; if (g == 0u) { return vec3f(0.5); } return unpack4x8unorm(g).xyz;",
        range: (0.0, 0.0),
        reading: None,
    },
    CellView {
        name: "Founder line",
        group: "Who",
        help: "Each cell that rained in starts a line, and its descendants keep the colour.",
        paint: Paint::Words(
            "One colour per founder that rained in. A big patch of one colour is one family.",
        ),
        wgsl: "return founder(cell[s].clade);",
        range: (0.0, 0.0),
        reading: None,
    },
    scale!(
        "Generation",
        "Who",
        "How many births back to the founder.",
        "f32(cell[s].gen)",
        (0.0, 2000.0),
        |c| c.generation as f32
    ),
    CellView {
        name: "Diet",
        group: "Lives on",
        help: "Where its energy has come from lately.",
        paint: Paint::Kinds(&[
            ("meat and biting", [1.0, 0.16, 0.16]),
            ("ground", [0.16, 1.0, 0.16]),
            ("gifts", [0.16, 0.16, 1.0]),
        ]),
        wgsl: "let d = stat[s].diet; let t = max(d.x + d.y + d.z + d.w, 1e-3); return vec3f(0.16) + 0.84 * vec3f((d.y + d.z) / t, d.x / t, d.w / t);",
        range: (0.0, 0.0),
        reading: None,
    },
    scale!(
        "Eats ground",
        "Lives on",
        "Share of recent energy from ground food.",
        "stat[s].diet.x / max(dot(stat[s].diet, vec4f(1.0)), 1e-3)",
        (0.0, 1.0),
        |c| share(&c.diet, 0)
    ),
    scale!(
        "Eats meat",
        "Lives on",
        "Share of recent energy from meat.",
        "stat[s].diet.y / max(dot(stat[s].diet, vec4f(1.0)), 1e-3)",
        (0.0, 1.0),
        |c| share(&c.diet, 1)
    ),
    scale!(
        "Eats by biting",
        "Lives on",
        "Share of recent energy from biting other cells.",
        "stat[s].diet.z / max(dot(stat[s].diet, vec4f(1.0)), 1e-3)",
        (0.0, 1.0),
        |c| share(&c.diet, 2)
    ),
    scale!(
        "Lives on gifts",
        "Lives on",
        "Share of recent energy from other cells' gifts.",
        "stat[s].diet.w / max(dot(stat[s].diet, vec4f(1.0)), 1e-3)",
        (0.0, 1.0),
        |c| share(&c.diet, 3)
    ),
    CellView {
        name: "Main spending",
        group: "Spends on",
        help: "The action it spends the most of its budget on.",
        paint: MAIN_ACTION,
        wgsl: "return main_action(s);",
        range: (0.0, 0.0),
        reading: None,
    },
    scale!(
        "Rest",
        "Spends on",
        "Share of budget spent resting. Resting nearly stops upkeep.",
        "stat[s].b0.x",
        (0.0, 1.0),
        |c| c.budget[0]
    ),
    scale!(
        "Move",
        "Spends on",
        "Share of budget spent moving.",
        "stat[s].b0.y",
        (0.0, 1.0),
        |c| c.budget[1]
    ),
    scale!(
        "Eat ground",
        "Spends on",
        "Share of budget spent eating ground food.",
        "stat[s].b0.z",
        (0.0, 1.0),
        |c| c.budget[2]
    ),
    scale!(
        "Eat meat",
        "Spends on",
        "Share of budget spent eating meat.",
        "stat[s].b0.w",
        (0.0, 1.0),
        |c| c.budget[3]
    ),
    scale!(
        "Grow child",
        "Spends on",
        "Share of budget going into the womb.",
        "stat[s].b1.x",
        (0.0, 1.0),
        |c| c.budget[4]
    ),
    scale!(
        "Emit scent",
        "Spends on",
        "Share of budget spent emitting scent.",
        "stat[s].b1.y",
        (0.0, 1.0),
        |c| c.budget[5]
    ),
    scale!(
        "Heat",
        "Spends on",
        "Share of budget spent burning the squares around it.",
        "stat[s].b1.z",
        (0.0, 1.0),
        |c| c.budget[6]
    ),
    scale!(
        "Bite",
        "Spends on",
        "Share of budget spent biting.",
        "stat[s].b1.w",
        (0.0, 1.0),
        |c| c.budget[7]
    ),
    scale!(
        "Give",
        "Spends on",
        "Share of budget spent giving energy away.",
        "stat[s].b2.x",
        (0.0, 1.0),
        |c| c.budget[8]
    ),
    scale!(
        "Enzyme",
        "Spends on",
        "Share of budget spent on enzyme, which helps eat ground food.",
        "stat[s].b2.y",
        (0.0, 1.0),
        |c| c.budget[9]
    ),
    scale!(
        "Armour",
        "Spends on",
        "Share of budget spent building armour. Needs armour_on.",
        "stat[s].b2.z",
        (0.0, 1.0),
        |c| c.budget[10]
    ),
    scale!(
        "Clean waste",
        "Spends on",
        "Share of budget spent cleaning up waste. Needs waste_clean.",
        "stat[s].b2.w",
        (0.0, 1.0),
        |c| c.budget[11]
    ),
    scale!(
        "Taking up mineral",
        "Spends on",
        "Share of budget on gathering mineral from the ground. Needs mineral_on.",
        "stat[s].b3.x",
        (0.0, 1.0),
        |c| c.budget[12]
    ),
    scale!(
        "Spare action x1",
        "Spends on",
        "Share of budget on spare action x1. Needs x1_on.",
        "stat[s].b3.y",
        (0.0, 1.0),
        |c| c.budget[13]
    ),
    scale!(
        "Spare action x2",
        "Spends on",
        "Share of budget on spare action x2. Needs x2_on.",
        "stat[s].b3.z",
        (0.0, 1.0),
        |c| c.budget[14]
    ),
    scale!(
        "Spare action x3",
        "Spends on",
        "Share of budget on spare action x3. Needs x3_on.",
        "stat[s].b3.w",
        (0.0, 1.0),
        |c| c.budget[15]
    ),
    scale!(
        "Energy",
        "Body",
        "Energy stored.",
        "cell[s].e",
        (0.0, 40.0),
        |c| c.e
    ),
    scale!(
        "Age",
        "Body",
        "Ticks since birth.",
        "cell[s].age",
        (0.0, 4000.0),
        |c| c.age
    ),
    scale!(
        "Womb fill",
        "Body",
        "Energy saved for the next child.",
        "cell[s].womb",
        (0.0, 5.0),
        |c| c.womb
    ),
    scale!(
        "Recent damage",
        "Body",
        "Energy lost to bites and heat in the last tick.",
        "cell[s].hit",
        (0.0, 5.0),
        |c| c.hit
    ),
    scale!(
        "Armour built",
        "Rules",
        "How hard its armour is. Needs armour_on.",
        "cell[s].hard",
        (0.0, 1.0),
        |c| c.x[0]
    ),
    scale!(
        "Wound",
        "Rules",
        "How wounded it is, which pack hunters can exploit. Needs pack_on.",
        "cell[s].wound",
        (0.0, 1.0),
        |c| c.x[1]
    ),
    scale!(
        "Practice at grazing",
        "Rules",
        "How practised it is at eating ground food. Needs skill_on.",
        "cell[s].skill_g",
        (0.0, 1.0),
        |c| c.x[2]
    ),
    scale!(
        "Practice at flesh",
        "Rules",
        "How practised it is at eating bodies. Needs skill_on.",
        "cell[s].skill_f",
        (0.0, 1.0),
        |c| c.x[3]
    ),
    scale!(
        "Place feels costly",
        "Rules",
        "How much upkeep this place costs it, as it feels it. Needs biomes_on and biome_feel.",
        "cell[s].feel_u",
        (0.0, 1.0),
        |c| c.feel[0]
    ),
    scale!(
        "Place feels slow",
        "Rules",
        "How much this place drags at it, as it feels it. Needs biomes_on and biome_feel.",
        "cell[s].feel_d",
        (0.0, 1.0),
        |c| c.feel[1]
    ),
    scale!(
        "Children by mating",
        "Rules",
        "How many children it has had with a mate. Needs mate_track.",
        "cell[s].mate_kids",
        (0.0, 8.0),
        |c| c.mate_kids
    ),
    scale!(
        "Mineral held",
        "Rules",
        "Mineral a cell has free for its next child. Needs mineral_on.",
        "cell[s].mineral",
        (0.0, 4.0),
        |c| c.spare[0]
    ),
    scale!(
        "Coat",
        "Rules",
        "The lock on this cell's body, a place on a circle. Only the right key bites well. Needs coat_on.",
        "cell[s].coat",
        (0.0, 1.0),
        |c| c.spare[1]
    ),
    scale!(
        "Mineral in the body",
        "Rules",
        "Mineral built into the body, which comes back when the cell dies. Needs mineral_on.",
        "cell[s].bound",
        (0.0, 4.0),
        |c| c.spare[2]
    ),
    scale!(
        "Key last used",
        "Rules",
        "The key this cell last bit with, a place on a circle. Needs coat_on.",
        "cell[s].key",
        (0.0, 1.0),
        |c| c.spare[3]
    ),
    sides!(
        "Signal 1",
        "Signals",
        "The first signal a cell chooses to show its neighbours.",
        "cell[s].voice.x",
        |c| c.voice[0]
    ),
    sides!(
        "Signal 2",
        "Signals",
        "The second signal a cell chooses to show its neighbours.",
        "cell[s].voice.y",
        |c| c.voice[1]
    ),
    CellView {
        name: "Both signals",
        group: "Signals",
        help: "Both signals as one colour. Cells showing the same colour are saying the same thing.",
        paint: Paint::Square,
        wgsl: "return vec3f(0.5 + 0.5 * cell[s].voice.x, 0.35, 0.5 + 0.5 * cell[s].voice.y);",
        range: (0.0, 0.0),
        reading: None,
    },
    CellView {
        name: "Heading",
        group: "Motion",
        help: "The direction it faces, as a hue. Flocks moving together show as one colour.",
        paint: Paint::Wheel,
        wgsl: "return hue(f32(cell[s].flags & 7u) / 8.0, 0.85, 1.0);",
        range: (0.0, 0.0),
        reading: None,
    },
    yes_no!(
        "Moved this tick",
        "Motion",
        "Whether it moved in the last tick.",
        "F_MOVED"
    ),
    scale!(
        "How often it moves",
        "Motion",
        "Share of recent ticks it moved.",
        "stat[s].chase.w",
        (0.0, 1.0),
        |c| c.chase[3]
    ),
    sides!(
        "Heads to food",
        "Motion",
        "When it moves, is that towards ground food or away from it?",
        "stat[s].chase.z",
        |c| c.chase[2]
    ),
    sides!(
        "Heads to bigger",
        "Motion",
        "When it moves, is that towards cells with more energy or away from them?",
        "stat[s].chase.x",
        |c| c.chase[0]
    ),
    sides!(
        "Heads to smaller",
        "Motion",
        "When it moves, is that towards cells with less energy or away from them?",
        "stat[s].chase.y",
        |c| c.chase[1]
    ),
    sides!(
        "Follows scent 1",
        "Motion",
        "When it moves, is that up the gradient of scent 1 or down it?",
        "stat[s].follow.x",
        |c| c.follow[0]
    ),
    sides!(
        "Follows scent 2",
        "Motion",
        "When it moves, is that up the gradient of scent 2 or down it?",
        "stat[s].follow.y",
        |c| c.follow[1]
    ),
    sides!(
        "Follows scent 3",
        "Motion",
        "When it moves, is that up the gradient of scent 3 or down it?",
        "stat[s].follow.z",
        |c| c.follow[2]
    ),
    sides!(
        "Follows scent 4",
        "Motion",
        "When it moves, is that up the gradient of scent 4 or down it?",
        "stat[s].follow.w",
        |c| c.follow[3]
    ),
    yes_no!(
        "Willing to mate",
        "With others",
        "Whether it would mate right now.",
        "F_MATE"
    ),
    yes_no!(
        "Willing to share",
        "With others",
        "Whether it would level out energy with a willing neighbour right now.",
        "F_SHARE"
    ),
    yes_no!(
        "Bit this tick",
        "With others",
        "Whether it bit someone in the last tick.",
        "F_BIT"
    ),
    scale!(
        "Crowding",
        "With others",
        "Share of its 8 neighbouring squares holding a cell.",
        "stat[s].misc.x",
        (0.0, 1.0),
        |c| c.misc[0]
    ),
    scale!(
        "Bites relatives",
        "With others",
        "How much it bites cells with a similar genome.",
        "stat[s].misc.y",
        (0.0, 1.0),
        |c| c.misc[1]
    ),
    scale!(
        "Bites strangers",
        "With others",
        "How much it bites cells with a different genome.",
        "stat[s].misc.z",
        (0.0, 1.0),
        |c| c.misc[2]
    ),
    scale!(
        "Faces a cell",
        "With others",
        "Share of recent ticks it was facing another cell.",
        "stat[s].misc.w",
        (0.0, 1.0),
        |c| c.misc[3]
    ),
];

/// Which number in `ViewParams::hi` a field's range is.
pub const HI_SUN: usize = 12;
pub const HI_BASE: usize = 13;
pub const HI_TRACKS: usize = 14;
pub const HI_TRAFFIC: usize = 15;
pub const HI_COUNT: usize = 16;

pub struct FieldView {
    pub name: &'static str,
    pub group: &'static str,
    pub help: &'static str,
    /// WGSL statements that return the colour, with `p` the square.
    pub wgsl: &'static str,
    /// For the legend: what it's tinted, and the range it's read against. None when it's a mix.
    pub tint: Option<[f32; 3]>,
    pub hi: Option<usize>,
}

const SCENT: [[f32; 3]; 4] = [
    [1.0, 0.22, 0.28],
    [0.2, 1.0, 0.32],
    [0.25, 0.55, 1.0],
    [1.0, 0.85, 0.1],
];
/// The colour of each scent, for labels elsewhere.
pub const SCENT_TINT: [[f32; 3]; 4] = SCENT;

macro_rules! field {
    ($name:literal, $group:literal, $help:literal, $src:literal, $hi:literal, $r:literal, $g:literal, $b:literal, $hi_n:expr) => {
        FieldView {
            name: $name,
            group: $group,
            help: $help,
            wgsl: concat!(
                "return vec3f(",
                $r,
                ", ",
                $g,
                ", ",
                $b,
                ") * level(",
                $src,
                ", V.hi[",
                $hi,
                ");"
            ),
            tint: Some([$r, $g, $b]),
            hi: Some($hi_n),
        }
    };
}

/// Everything that can be shown under the cells. Order is the order of the list.
pub const FIELD_VIEWS: &[FieldView] = &[
    FieldView {
        name: "Nothing",
        group: "Plain",
        help: "A dark background.",
        wgsl: "return vec3f(0.0);",
        tint: None,
        hi: None,
    },
    FieldView {
        name: "Natural mix",
        group: "Plain",
        help: "Ground food in green, meat in red, heat in orange, scent in blue.",
        wgsl: "let a = fa[p]; let b = fb[p]; let g = min(a.x / V.ground_cap, 1.0); let m = min(a.y / 3.0, 1.0); let h = min(a.z / 2.0, 1.0); let c = min((b.x + b.y + b.z + b.w) / 4.0, 1.0); return vec3f(m * 0.27 + h * 0.78, g * 0.15 + h * 0.35, c * 0.17 + g * 0.03) * 1.6;",
        tint: None,
        hi: None,
    },
    FieldView {
        name: "All scents",
        group: "Scent",
        help: "Each scent in its own colour, added together, so trails show at a glance.",
        wgsl: "let b = fb[p]; let sum = vec3f(1.0, 0.22, 0.28) * level(b.x, V.hi[1].x) + vec3f(0.2, 1.0, 0.32) * level(b.y, V.hi[1].y) + vec3f(0.25, 0.55, 1.0) * level(b.z, V.hi[1].z) + vec3f(1.0, 0.85, 0.1) * level(b.w, V.hi[1].w); return vec3f(1.0) - exp(-sum * 1.1);",
        tint: None,
        hi: None,
    },
    field!(
        "Scent 1",
        "Scent",
        "Lingers and spreads slowly.",
        "fb[p].x",
        "1].x",
        1.0,
        0.22,
        0.28,
        4
    ),
    field!(
        "Scent 2",
        "Scent",
        "Spreads a little faster.",
        "fb[p].y",
        "1].y",
        0.2,
        1.0,
        0.32,
        5
    ),
    field!(
        "Scent 3",
        "Scent",
        "Spreads fast and fades quickly.",
        "fb[p].z",
        "1].z",
        0.25,
        0.55,
        1.0,
        6
    ),
    field!(
        "Scent 4",
        "Scent",
        "Spreads fastest and fades within a few ticks.",
        "fb[p].w",
        "1].w",
        1.0,
        0.85,
        0.1,
        7
    ),
    field!(
        "Ground food",
        "Food",
        "Grows where the sun shines, eaten by cells.",
        "fa[p].x",
        "0].x",
        0.25,
        0.95,
        0.3,
        0
    ),
    field!(
        "Meat",
        "Food",
        "What dead bodies and spilled bites leave behind.",
        "fa[p].y",
        "0].y",
        1.0,
        0.25,
        0.3,
        1
    ),
    field!(
        "Enzyme",
        "Food",
        "Makes ground food quicker to eat where it lies.",
        "fc[p].x",
        "2].x",
        0.9,
        0.35,
        1.0,
        8
    ),
    field!(
        "Sun now",
        "Food",
        "How fast ground food grows here at this point in the seasons.",
        "sun_now(p)",
        "3].x",
        1.0,
        0.9,
        0.35,
        HI_SUN
    ),
    field!(
        "Sun on average",
        "Food",
        "How rich this square is, ignoring the seasons.",
        "f32(terrain[p] & 0xFFFFu) / 65535.0 * SUN_RANGE * V.sun_k",
        "3].y",
        1.0,
        0.75,
        0.2,
        HI_BASE
    ),
    field!(
        "Heat",
        "Danger",
        "Burns whatever stands on it.",
        "fa[p].z",
        "0].z",
        1.0,
        0.5,
        0.08,
        2
    ),
    field!(
        "Noise",
        "Danger",
        "Made by moving bodies and bites. Anyone can hear it.",
        "fa[p].w",
        "0].w",
        0.85,
        0.85,
        1.0,
        3
    ),
    field!(
        "Waste",
        "Rules",
        "What cells leave behind, harmful until it breaks down. Needs waste_on.",
        "fc[p].y",
        "2].y",
        0.65,
        0.5,
        0.2,
        9
    ),
    field!(
        "Fire",
        "Rules",
        "Squares alight. A square that has just burned can't light again for a while. Needs weather_fire_on.",
        "max(fc[p].z, 0.0)",
        "2].z",
        1.0,
        0.45,
        0.05,
        10
    ),
    field!(
        "Tracks",
        "Rules",
        "Worn paths, which are cheaper to walk. Needs roads_on.",
        "fd[p].x",
        "3].z",
        0.9,
        0.85,
        0.7,
        HI_TRACKS
    ),
    field!(
        "Traffic",
        "Rules",
        "Where cells have walked lately, so paths and trails show up. Needs trail_stats.",
        "fd[p].w",
        "3].w",
        0.95,
        0.75,
        0.95,
        HI_TRAFFIC
    ),
    FieldView {
        name: "Mineral on the ground",
        group: "Rules",
        help: "Mineral lying free, to be gathered. Needs mineral_on.",
        wgsl: "return vec3f(0.35, 0.8, 1.0) * level(fd[p].y, 2.0);",
        tint: None,
        hi: None,
    },
    FieldView {
        name: "Mineral-rich land",
        group: "Rules",
        help: "How much of the mineral flows towards each place. Needs mineral_on.",
        wgsl: "return vec3f(0.35, 0.55, 1.0) * level(fd[p].z, 4.0);",
        tint: None,
        hi: None,
    },
    FieldView {
        name: "Kind of place",
        group: "Rules",
        help: "Cold, hot, swamp or barren. Only means something with walls_on or biomes_on.",
        wgsl: "return place_tint(terrain[p]) * 0.7;",
        tint: None,
        hi: None,
    },
    field!(
        "Experiment channel fc.w",
        "Experiments",
        "Free field channel, unused unless the physics uses it.",
        "fc[p].w",
        "2].w",
        0.75,
        0.9,
        0.5,
        11
    ),
];

/// The view shader with every row of the tables in it.
pub fn shader_source() -> String {
    let mut cells = String::new();
    for (i, v) in CELL_VIEWS.iter().enumerate() {
        cells += &format!("case {i}u: {{ {} }}\n        ", v.wgsl);
    }
    let mut fields = String::new();
    for (i, v) in FIELD_VIEWS.iter().enumerate() {
        fields += &format!("case {i}u: {{ {} }}\n        ", v.wgsl);
    }
    SHADER
        .replace("//@CELLS", &cells)
        .replace("//@FIELDS", &fields)
}

pub const SHOW_CELLS: u32 = 1;
pub const SHOW_SIMILAR: u32 = 2;
pub const SHOW_FOUNDER: u32 = 4;

/// What to draw. Laid out as `ViewU` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ViewParams {
    pub origin: [f32; 2],
    pub scale: f32,
    pub samples: u32,
    pub out: [u32; 2],
    pub row0: u32,
    pub show: u32,
    pub cell_mode: u32,
    pub field_mode: u32,
    pub field_opacity: f32,
    pub cell_opacity: f32,
    pub cell_lo: f32,
    pub cell_hi: f32,
    pub tau: f32,
    pub clade: u32,
    pub hi: [f32; 16],
    pub sketch: [u32; 4],
    pub center: [f32; 2],
    pub sun_k: f32,
    pub season: f32,
    pub season_on: f32,
    pub ground_cap: f32,
    /// Share of the sun that isn't going into blooms.
    pub sun_keep: f32,
    pub pad0: f32,
    /// How far moving weather has slid the sun map, in squares.
    pub drift: [f32; 2],
    pub pad1: [f32; 2],
}

const STRIDE: u64 = 256;
/// A frame goes to the GPU in bands this tall, so drawing doesn't hold it for long.
const BAND: u32 = 256;
const MAX_BANDS: u64 = 32;

/// The layout the view pass's own bindings use. Shared by every world's pipeline.
pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("explore"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<ViewParams>() as u64
                    ),
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
    group0: wgpu::BindGroup,
    pub size: u32,
}

impl ViewPipe {
    pub fn new(world: &World, layout: &wgpu::BindGroupLayout) -> Result<ViewPipe, String> {
        let (pipeline, group0) = world.extra_pipe_source(
            "explore.wgsl",
            &shader_source(),
            "main",
            &[
                ("occ", Access2::Read),
                ("cell", Access2::Read),
                ("stat", Access2::Read),
                ("sketch", Access2::Read),
                ("terrain", Access2::Read),
                ("fa", Access2::Read),
                ("fb", Access2::Read),
                ("fc", Access2::Read),
                ("fd", Access2::Read),
                ("ctr", Access2::ReadWrite),
            ],
            Some(layout),
        )?;
        Ok(ViewPipe {
            pipeline,
            group0,
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
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
    ) -> Canvas {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("world view"),
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
            label: Some("world view"),
            size: STRIDE * MAX_BANDS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world view"),
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

    /// Draws a frame, in short submissions. Doesn't wait for the GPU.
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipe: &ViewPipe,
        params: ViewParams,
    ) {
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
                pass.set_bind_group(0, &pipe.group0, &[0]);
                pass.set_bind_group(1, &self.group, &[(k * STRIDE) as u32]);
                pass.dispatch_workgroups(
                    self.width.div_ceil(16),
                    band.min(self.height - row).div_ceil(16),
                    1,
                );
            }
            queue.submit([enc.finish()]);
            row += band;
            k += 1;
        }
    }
}

/// How many samples per pixel suit this many squares per pixel.
pub fn samples_for(scale: f32) -> u32 {
    (scale.ceil() as u32).clamp(1, 2)
}

/// Fits a range to sampled readings, ignoring the extremes.
pub fn fit_range(mut values: Vec<f32>) -> Option<(f32, f32)> {
    values.retain(|v| v.is_finite());
    if values.len() < 8 {
        return None;
    }
    values.sort_by(f32::total_cmp);
    let at = |q: f32| values[((values.len() - 1) as f32 * q) as usize];
    let (lo, hi) = (at(0.02), at(0.98));
    Some(if hi > lo { (lo, hi) } else { (lo, lo + 1e-3) })
}

/// A field's top of range from sampled squares: near the highest, so dense places still show
/// detail, but not thrown off by one hot square.
pub fn fit_top(mut values: Vec<f32>) -> f32 {
    values.retain(|v| v.is_finite() && *v > 0.0);
    if values.is_empty() {
        return 1e-3;
    }
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f32 * 0.995) as usize].max(1e-3)
}
