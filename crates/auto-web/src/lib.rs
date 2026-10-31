#![cfg(target_arch = "wasm32")]
//! Browser entry points. The page calls these from JavaScript.

use std::cell::RefCell;

use auto_core::{Checks, Gpu, Sim, automata, gpu};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use auto_app::{Control, Viewer, default_speed};

thread_local! {
    static VIEWER: RefCell<Option<Viewer>> = const { RefCell::new(None) };
}

#[wasm_bindgen]
pub async fn start(canvas: web_sys::HtmlCanvasElement, id: String) -> Result<(), JsValue> {
    let target = wgpu::SurfaceTarget::Canvas(canvas);
    let viewer = Viewer::try_new_target(gpu::instance(), target, &id)
        .await
        .map_err(|e| JsValue::from_str(&e))?;
    VIEWER.with(|v| *v.borrow_mut() = Some(viewer));
    Ok(())
}

fn with<R>(f: impl FnOnce(&mut Viewer) -> R) -> Option<R> {
    VIEWER.with(|v| v.borrow_mut().as_mut().map(f))
}

#[wasm_bindgen]
pub fn frame() {
    with(|v| v.frame());
}

#[wasm_bindgen]
pub fn resize(width: u32, height: u32) {
    with(|v| v.resize(width, height));
}

#[wasm_bindgen]
pub fn select(id: String) {
    with(|v| v.select_id(&id));
}

#[wasm_bindgen]
pub fn set_speed(speed: u32) {
    with(|v| v.set_speed(speed));
}

#[wasm_bindgen]
pub fn control(name: String) {
    use Control::*;
    let c = match name.as_str() {
        "pause" => TogglePause,
        "reseed" => Reseed,
        "faster" => Faster,
        "slower" => Slower,
        _ => return,
    };
    with(|v| v.control(c));
}

#[wasm_bindgen]
pub fn zoom(factor: f32) {
    with(|v| v.zoom(factor));
}

#[wasm_bindgen]
pub fn stats() -> String {
    with(|v| v.stats()).unwrap_or_default()
}

/// The automata as JSON, for the page's menu.
#[wasm_bindgen]
pub fn automata_json() -> String {
    let rows: Vec<serde_json::Value> = automata::ALL
        .iter()
        .map(|e| {
            serde_json::json!({ "id": e.id, "name": e.name, "blurb": e.blurb, "speed": default_speed(e.id) })
        })
        .collect();
    serde_json::Value::Array(rows).to_string()
}

/// Wait until the GPU has finished what was submitted.
async fn idle(gpu: &Gpu) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        gpu.queue.on_submitted_work_done(move || {
            let _ = resolve.call0(&JsValue::NULL);
        });
    });
    let _ = JsFuture::from(promise).await;
}

fn now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now() / 1000.0)
        .unwrap_or(0.0)
}

/// The native benchmark's method, in the browser: grow a batch, time batches,
/// return the median as JSON. Batches run about 400 ms rather than the native
/// 20 ms, because Firefox only reports GPU completion about every 100 ms and
/// that would otherwise be the thing being measured.
#[wasm_bindgen]
pub async fn bench(id: String, width: u32, height: u32, seconds: f64) -> Result<String, JsValue> {
    let instance = gpu::instance();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let gpu = Gpu::from_adapter(instance, adapter, Checks::Safe)
        .await
        .map_err(|e| JsValue::from_str(&e))?;
    let program = automata::build(&id, width, height)
        .ok_or_else(|| JsValue::from_str("unknown automaton"))?;
    let (cells, memory, gens, pw, ph) = (
        program.cells_per_step,
        program.total_bytes(),
        program.generations_per_step as f64,
        program.width,
        program.height,
    );
    let mut sim = Sim::new(&gpu, program, 1).map_err(|e| JsValue::from_str(&e))?;
    idle(&gpu).await;
    sim.step(&gpu, 32);
    idle(&gpu).await;

    let mut batch = 1u32;
    loop {
        let t = now();
        sim.step(&gpu, batch);
        idle(&gpu).await;
        let dt = now() - t;
        if dt >= 0.4 || batch >= 1 << 22 {
            break;
        }
        let want = (batch as f64 * 0.45 / dt.max(1e-4)).ceil() as u32;
        batch = want.clamp(batch + 1, batch.saturating_mul(8)).min(1 << 22);
    }
    let mut rates = Vec::new();
    let start = now();
    while now() - start < seconds || rates.len() < 3 {
        let t = now();
        sim.step(&gpu, batch);
        idle(&gpu).await;
        rates.push(batch as f64 / (now() - t).max(1e-5));
    }
    rates.sort_by(|a, b| a.total_cmp(b));
    let median = rates[rates.len() / 2];
    let info = gpu.adapter.get_info();
    let limits = gpu.limits();
    Ok(serde_json::json!({
        "id": sim.program.id, "tag": "browser", "adapter": format!("{} {}", info.vendor, info.name),
        "checks": "Safe", "width": pw, "height": ph, "cells": cells, "memory_bytes": memory,
        "steps_per_second": median, "generations_per_second": median * gens,
        "cells_per_second": median * gens * cells as f64, "batch_steps": batch, "batches": rates.len(),
        "max_storage_binding": limits.max_storage_buffer_binding_size,
        "max_buffer": limits.max_buffer_size,
        "max_workgroup_storage": limits.max_compute_workgroup_storage_size,
        "max_invocations": limits.max_compute_invocations_per_workgroup,
    })
    .to_string())
}
