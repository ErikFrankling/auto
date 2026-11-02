//! Live charts, drawn plainly: one measure per chart, a legend, and a readout at the cursor.

use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Ui, Vec2, pos2};

use crate::sim::Point;

/// Each diet's colour, matching the world view: ground green, meat and biting red, gifts blue.
const GROUND: Color32 = Color32::from_rgb(0x19, 0x9e, 0x70);
const MEAT: Color32 = Color32::from_rgb(0xd9, 0x59, 0x26);
const BITE: Color32 = Color32::from_rgb(0xe6, 0x67, 0x67);
const GIFT: Color32 = Color32::from_rgb(0x39, 0x87, 0xe5);
const POPULATION: Color32 = Color32::from_rgb(0x39, 0x87, 0xe5);
const STARVED: Color32 = Color32::from_rgb(0xc9, 0x85, 0x00);
const BITTEN: Color32 = Color32::from_rgb(0xe6, 0x67, 0x67);
const BURNED: Color32 = Color32::from_rgb(0xd9, 0x59, 0x26);
const OLD: Color32 = Color32::from_rgb(0x90, 0x85, 0xe9);
const BIRTHS: Color32 = Color32::from_rgb(0x19, 0x9e, 0x70);

struct Series<'a> {
    name: &'a str,
    colour: Color32,
    read: fn(&Point) -> f64,
}

pub fn show(ui: &mut Ui, history: &[Point]) {
    if history.len() < 2 {
        ui.label("The charts fill in once the world has run for a couple of seconds.");
        return;
    }
    let height = ((ui.available_height() - 90.0) / 3.0).clamp(90.0, 220.0);
    chart(
        ui,
        "Cells alive",
        history,
        height,
        &[Series {
            name: "cells",
            colour: POPULATION,
            read: |p| p.cells,
        }],
        Scale::Count,
    );
    chart(
        ui,
        "Where energy comes from (share)",
        history,
        height,
        &[
            Series {
                name: "ground",
                colour: GROUND,
                read: |p| p.diet[0],
            },
            Series {
                name: "meat",
                colour: MEAT,
                read: |p| p.diet[1],
            },
            Series {
                name: "biting",
                colour: BITE,
                read: |p| p.diet[2],
            },
            Series {
                name: "gifts",
                colour: GIFT,
                read: |p| p.diet[3],
            },
        ],
        Scale::Share,
    );
    chart(
        ui,
        "Births and deaths per tick",
        history,
        height,
        &[
            Series {
                name: "born",
                colour: BIRTHS,
                read: |p| p.births,
            },
            Series {
                name: "starved",
                colour: STARVED,
                read: |p| p.deaths[0],
            },
            Series {
                name: "bitten",
                colour: BITTEN,
                read: |p| p.deaths[1],
            },
            Series {
                name: "burned",
                colour: BURNED,
                read: |p| p.deaths[2],
            },
            Series {
                name: "old age",
                colour: OLD,
                read: |p| p.deaths[3],
            },
        ],
        Scale::Count,
    );
}

#[derive(PartialEq)]
enum Scale {
    Count,
    Share,
}

fn number(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if v >= 1e6 {
        format!("{:.2}M", v / 1e6)
    } else if v >= 1e4 {
        format!("{:.0}k", v / 1e3)
    } else if v >= 100.0 {
        format!("{v:.0}")
    } else if v >= 1.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

fn chart(
    ui: &mut Ui,
    title: &str,
    history: &[Point],
    height: f32,
    series: &[Series],
    scale: Scale,
) {
    ui.add_space(4.0);
    ui.strong(title);
    if series.len() > 1 {
        ui.horizontal_wrapped(|ui| {
            for s in series {
                let (rect, _) = ui.allocate_exact_size(Vec2::new(14.0, 3.0), Sense::hover());
                ui.painter().rect_filled(rect, 1.0, s.colour);
                ui.label(s.name);
            }
        });
    }
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::hover());
    let painter = ui.painter_at(rect);
    let plot = Rect::from_min_max(
        rect.min + Vec2::new(44.0, 4.0),
        rect.max - Vec2::new(4.0, 18.0),
    );
    let t0 = history.first().unwrap().tick;
    let t1 = history.last().unwrap().tick.max(t0 + 1.0);
    let top = match scale {
        Scale::Share => 1.0,
        Scale::Count => {
            history
                .iter()
                .flat_map(|p| series.iter().map(move |s| (s.read)(p)))
                .fold(1e-6, f64::max)
                * 1.1
        }
    };
    let at = |t: f64, v: f64| -> Pos2 {
        pos2(
            plot.left() + ((t - t0) / (t1 - t0)) as f32 * plot.width(),
            plot.bottom() - (v / top) as f32 * plot.height(),
        )
    };
    let grid = Color32::from_white_alpha(18);
    let ink = ui.visuals().weak_text_color();
    let small = FontId::proportional(11.0);
    for k in 0..=2 {
        let v = top * k as f64 / 2.0;
        let y = at(t0, v).y;
        painter.hline(plot.x_range(), y, Stroke::new(1.0, grid));
        let text = if scale == Scale::Share {
            format!("{:.0}%", v * 100.0)
        } else {
            number(v)
        };
        painter.text(
            pos2(plot.left() - 5.0, y),
            Align2::RIGHT_CENTER,
            text,
            small.clone(),
            ink,
        );
    }
    painter.text(
        pos2(plot.left(), rect.bottom() - 1.0),
        Align2::LEFT_BOTTOM,
        format!("tick {t0:.0}"),
        small.clone(),
        ink,
    );
    painter.text(
        pos2(plot.right(), rect.bottom() - 1.0),
        Align2::RIGHT_BOTTOM,
        format!("tick {t1:.0}"),
        small.clone(),
        ink,
    );
    for s in series {
        let points: Vec<Pos2> = history.iter().map(|p| at(p.tick, (s.read)(p))).collect();
        painter.line(points, Stroke::new(1.5, s.colour));
    }
    if let Some(cursor) = response.hover_pos()
        && plot.contains(cursor)
    {
        let t = t0 + ((cursor.x - plot.left()) / plot.width()) as f64 * (t1 - t0);
        let nearest = history
            .iter()
            .min_by(|a, b| (a.tick - t).abs().total_cmp(&(b.tick - t).abs()))
            .unwrap();
        let x = at(nearest.tick, 0.0).x;
        painter.vline(
            x,
            plot.y_range(),
            Stroke::new(1.0, Color32::from_white_alpha(70)),
        );
        for s in series {
            painter.circle_filled(at(nearest.tick, (s.read)(nearest)), 3.0, s.colour);
        }
        let mut text = format!("tick {:.0}", nearest.tick);
        for s in series {
            let v = (s.read)(nearest);
            let shown = if scale == Scale::Share {
                format!("{:.1}%", v * 100.0)
            } else {
                number(v)
            };
            text += &format!("\n{}: {shown}", s.name);
        }
        response.on_hover_text_at_pointer(text);
    }
}
