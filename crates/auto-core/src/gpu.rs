//! Adapter and device setup, shared by every target.

use wgpu::{Features, Limits};

/// Runtime bounds checks in generated shader code. WebGPU always has them on;
/// native can turn them off once a kernel is trusted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Checks {
    /// The default for WebGPU: every buffer access is clamped.
    #[default]
    Safe,
    /// No clamping. Only works on native.
    Trusted,
}

#[derive(Clone, Debug, Default)]
pub struct GpuOptions {
    /// Pick the adapter whose name contains this, for example "7900" or "llvmpipe".
    pub adapter_name: Option<String>,
    pub checks: Checks,
}

pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub checks: Checks,
}

impl Gpu {
    /// Headless setup, for benchmarks, recording and tests.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn headless(opts: GpuOptions) -> Result<Self, String> {
        let instance = instance();
        let adapter = match &opts.adapter_name {
            Some(name) => {
                let want = name.to_lowercase();
                instance
                    .enumerate_adapters(wgpu::Backends::all())
                    .await
                    .into_iter()
                    .find(|a| a.get_info().name.to_lowercase().contains(&want))
                    .ok_or_else(|| format!("no adapter matching {name:?}"))?
            }
            None => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    apply_limit_buckets: false,
                    compatible_surface: None,
                })
                .await
                .map_err(|e| e.to_string())?,
        };
        Self::from_adapter(instance, adapter, opts.checks).await
    }

    /// Setup that must also present to a window or canvas.
    pub async fn for_surface(
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
        checks: Checks,
    ) -> Result<Self, String> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
                compatible_surface: Some(surface),
            })
            .await
            .map_err(|e| e.to_string())?;
        Self::from_adapter(instance, adapter, checks).await
    }

    pub async fn from_adapter(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        checks: Checks,
    ) -> Result<Self, String> {
        // Ask for everything the hardware has. Kernels check for what they need.
        let wanted = Features::SHADER_INT64 | Features::SUBGROUP | Features::TIMESTAMP_QUERY;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("auto"),
                required_features: adapter.features() & wanted,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            checks,
        })
    }

    pub fn name(&self) -> String {
        self.adapter.get_info().name
    }

    pub fn limits(&self) -> Limits {
        self.device.limits()
    }

    pub fn has_int64(&self) -> bool {
        self.device.features().contains(Features::SHADER_INT64)
    }

    pub fn shader(&self, label: &str, source: &str) -> wgpu::ShaderModule {
        let desc = wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        };
        #[cfg(not(target_arch = "wasm32"))]
        if self.checks == Checks::Trusted {
            // SAFETY: Trusted is opt-in, and every kernel only indexes inside its
            // own buffers, which the test suite runs with checks on first.
            return unsafe {
                self.device
                    .create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
            };
        }
        self.device.create_shader_module(desc)
    }

    /// Block until everything submitted has finished.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("gpu wait failed");
    }
}

pub fn instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: if cfg!(target_arch = "wasm32") {
            wgpu::Backends::BROWSER_WEBGPU
        } else {
            wgpu::Backends::VULKAN
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

/// The instance a native window needs, which has to know about the display.
#[cfg(not(target_arch = "wasm32"))]
pub fn instance_with_display(
    display: impl wgpu::wgt::WgpuHasDisplayHandle + 'static,
) -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_with_display_handle(Box::new(display))
    })
}
