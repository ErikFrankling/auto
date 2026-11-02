//! What the colours mean, drawn from the same tables and scales the view shader uses.

use auto_life_gpu::explore::{self, CellView, FieldView, Paint};
use egui::{Align2, Color32, FontId, Mesh, Rect, Sense, Ui, Vec2, pos2, vec2};

fn colour(c: [f32; 3]) -> Color32 {
    Color32::from_rgb(
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
    )
}

/// Readable numbers for the ends of a scale.
pub fn number(v: f32) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".into()
    } else if a >= 1000.0 {
        format!("{v:.0}")
    } else if a >= 10.0 {
        format!("{v:.1}")
    } else if a >= 0.1 {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    }
}

fn bar(ui: &mut Ui, shade: impl Fn(f32) -> [f32; 3], lo: &str, hi: &str) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 12.0), Sense::hover());
    let mut mesh = Mesh::default();
    let n = 48;
    for k in 0..=n {
        let t = k as f32 / n as f32;
        let x = rect.left() + rect.width() * t;
        let c = colour(shade(t));
        mesh.colored_vertex(pos2(x, rect.top()), c);
        mesh.colored_vertex(pos2(x, rect.bottom()), c);
    }
    for k in 0..n as u32 {
        let i = k * 2;
        mesh.add_triangle(i, i + 1, i + 2);
        mesh.add_triangle(i + 1, i + 2, i + 3);
    }
    ui.painter().add(mesh);
    ui.horizontal(|ui| {
        ui.small(lo);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.small(hi);
        });
    });
}

fn swatches(ui: &mut Ui, kinds: &[(&str, [f32; 3])]) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        for (name, c) in kinds {
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(11.0), Sense::hover());
            ui.painter().rect_filled(rect, 2.0, colour(*c));
            ui.small(*name);
            ui.add_space(5.0);
        }
    });
}

fn wheel(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 58.0), Sense::hover());
    let c = pos2(rect.left() + 29.0, rect.center().y);
    let names = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"];
    let painter = ui.painter();
    for k in 0..8u32 {
        let a = k as f32 / 8.0 * std::f32::consts::TAU;
        let at = c + Vec2::new(a.cos(), a.sin()) * 19.0;
        painter.circle_filled(at, 5.5, colour(explore::hue(k as f32 / 8.0, 0.85, 1.0)));
    }
    let mut x = rect.left() + 66.0;
    for (k, n) in names.iter().enumerate().step_by(2) {
        let (r, _) = (Rect::from_center_size(pos2(x, c.y), Vec2::splat(10.0)), 0);
        painter.rect_filled(r, 2.0, colour(explore::hue(k as f32 / 8.0, 0.85, 1.0)));
        painter.text(
            pos2(x + 8.0, c.y),
            Align2::LEFT_CENTER,
            *n,
            FontId::proportional(11.0),
            ui.visuals().text_color(),
        );
        x += 40.0;
    }
}

fn square(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 62.0), Sense::hover());
    let box_ = Rect::from_min_size(rect.min, Vec2::splat(56.0));
    let mut mesh = Mesh::default();
    let n = 10u32;
    for j in 0..=n {
        for i in 0..=n {
            let (x, y) = (i as f32 / n as f32, j as f32 / n as f32);
            mesh.colored_vertex(box_.min + vec2(x, y) * 56.0, colour([x, 0.35, 1.0 - y]));
        }
    }
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            mesh.add_triangle(a, a + 1, a + n + 1);
            mesh.add_triangle(a + 1, a + n + 1, a + n + 2);
        }
    }
    let painter = ui.painter();
    painter.add(mesh);
    let ink = ui.visuals().text_color();
    let small = FontId::proportional(11.0);
    painter.text(
        pos2(box_.right() + 8.0, box_.top() + 4.0),
        Align2::LEFT_TOP,
        "up: signal 2 high",
        small.clone(),
        ink,
    );
    painter.text(
        pos2(box_.right() + 8.0, box_.top() + 22.0),
        Align2::LEFT_TOP,
        "right: signal 1 high",
        small.clone(),
        ink,
    );
    painter.text(
        pos2(box_.right() + 8.0, box_.top() + 40.0),
        Align2::LEFT_TOP,
        "same colour, same message",
        small,
        ink,
    );
}

pub fn cell(ui: &mut Ui, view: &CellView, range: (f32, f32)) {
    ui.strong(format!("Cells: {}", view.name));
    match view.paint {
        Paint::Scale => bar(ui, explore::seq, &number(range.0), &number(range.1)),
        Paint::Sides => bar(ui, explore::div, &number(range.0), &number(range.1)),
        Paint::Kinds(kinds) => swatches(ui, kinds),
        Paint::Words(words) => {
            ui.small(words);
        }
        Paint::Wheel => wheel(ui),
        Paint::Square => square(ui),
    }
}

pub fn field(ui: &mut Ui, view: &FieldView, hi: f32) {
    ui.strong(format!("Under the cells: {}", view.name));
    if view.name == "All scents" {
        swatches(
            ui,
            &[
                ("scent 1", explore::SCENT_TINT[0]),
                ("scent 2", explore::SCENT_TINT[1]),
                ("scent 3", explore::SCENT_TINT[2]),
                ("scent 4", explore::SCENT_TINT[3]),
            ],
        );
        ui.small("Mixed colours mean several scents on one square. Each scent is scaled to its own range.");
    } else if view.name == "Natural mix" {
        swatches(
            ui,
            &[
                ("ground food", [0.05, 0.3, 0.05]),
                ("meat", [0.5, 0.05, 0.05]),
                ("heat", [1.0, 0.55, 0.05]),
                ("scent", [0.05, 0.1, 0.4]),
            ],
        );
    } else if let Some(tint) = view.tint {
        let shade = move |t: f32| {
            let k = t.sqrt();
            let base = [0.035, 0.04, 0.055];
            std::array::from_fn(|i| base[i] + tint[i] * k)
        };
        bar(ui, shade, "0", &number(hi));
    } else {
        ui.small(view.help);
    }
}
