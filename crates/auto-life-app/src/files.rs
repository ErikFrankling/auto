//! A small file browser inside the window, with what's known about each saved world.

use std::path::{Path, PathBuf};

use auto_life_gpu::config::{check_layout, read_save_head};
use egui::{Color32, RichText, Ui};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Purpose {
    Load,
    Save,
    Seeds,
}

#[derive(Clone)]
pub struct Entry {
    pub path: PathBuf,
    pub name: String,
    pub is_dir: bool,
    /// Width, tick and size in GB, if it's a saved world.
    pub info: Option<(u64, u64, f64)>,
    /// Whether this build can read it.
    pub current: bool,
}

pub struct Browser {
    pub dir: PathBuf,
    pub path_text: String,
    pub entries: Vec<Entry>,
    pub error: Option<String>,
}

fn scan(dir: &Path) -> Result<Vec<Entry>, String> {
    let mut out = vec![];
    let read = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for item in read.flatten() {
        let path = item.path();
        let name = item.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let is_dir = path.is_dir();
        let mut info = None;
        let mut current = false;
        if !is_dir {
            if !name.ends_with(".bin") {
                continue;
            }
            if let Ok(head) = read_save_head(&path.to_string_lossy()) {
                current = check_layout(&head, &name).is_ok();
                let bytes = item.metadata().map_or(0, |m| m.len());
                info = Some((
                    head["size"].as_u64().unwrap_or(0),
                    head["tick"].as_u64().unwrap_or(0),
                    bytes as f64 / 1e9,
                ));
            }
        }
        out.push(Entry {
            path,
            name,
            is_dir,
            info,
            current,
        });
    }
    out.sort_by(|a, b| (!a.is_dir, &a.name).cmp(&(!b.is_dir, &b.name)));
    Ok(out)
}

impl Browser {
    pub fn new(dir: &str) -> Browser {
        let dir = if Path::new(dir).is_dir() {
            PathBuf::from(dir)
        } else {
            std::env::var("HOME").map_or(PathBuf::from("/"), PathBuf::from)
        };
        let mut b = Browser {
            path_text: dir.to_string_lossy().to_string(),
            dir,
            entries: vec![],
            error: None,
        };
        b.refresh();
        b
    }

    pub fn refresh(&mut self) {
        match scan(&self.dir) {
            Ok(e) => {
                self.entries = e;
                self.error = None;
            }
            Err(e) => {
                self.entries.clear();
                self.error = Some(e);
            }
        }
    }

    fn go(&mut self, dir: PathBuf) {
        self.path_text = dir.to_string_lossy().to_string();
        self.dir = dir;
        self.refresh();
    }

    /// Draws the folder. Returns a file that was clicked.
    pub fn list(&mut self, ui: &mut Ui, height: f32, ticked: Option<&[String]>) -> Option<String> {
        let mut chosen = None;
        ui.horizontal(|ui| {
            if ui.button("Up").clicked()
                && let Some(parent) = self.dir.parent()
            {
                let parent = parent.to_path_buf();
                self.go(parent);
            }
            if ui.button("Saved worlds").clicked() {
                self.go(PathBuf::from(crate::SAVES_DIR));
            }
            if let Ok(home) = std::env::var("HOME")
                && ui.button("Home").clicked()
            {
                self.go(PathBuf::from(home));
            }
            ui.weak(self.dir.to_string_lossy());
        });
        if let Some(e) = &self.error {
            ui.colored_label(Color32::LIGHT_RED, e);
        }
        let mut enter = None;
        egui::ScrollArea::vertical()
            .max_height(height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if self.entries.is_empty() {
                    ui.weak("No saved worlds here.");
                }
                for e in &self.entries {
                    let label = if e.is_dir {
                        format!("{}/", e.name)
                    } else {
                        match e.info {
                            Some((w, t, gb)) if e.current => {
                                format!("{}    {w} wide, tick {t}, {gb:.2} GB", e.name)
                            }
                            Some((w, t, _)) => {
                                format!("{}    {w} wide, tick {t}, older version", e.name)
                            }
                            None => format!("{}    not a saved world", e.name),
                        }
                    };
                    let path = e.path.to_string_lossy().to_string();
                    let on = ticked.is_some_and(|t| t.contains(&path));
                    let text = if e.is_dir {
                        RichText::new(label).strong()
                    } else {
                        RichText::new(label)
                    };
                    if ui.selectable_label(on, text).clicked() {
                        if e.is_dir {
                            enter = Some(e.path.clone());
                        } else {
                            chosen = Some(path);
                        }
                    }
                }
            });
        if let Some(d) = enter {
            self.go(d);
        }
        chosen
    }
}
