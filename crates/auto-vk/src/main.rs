//! The same programs as wgpu runs, driven by hand through raw Vulkan.
//!
//! This exists to measure one thing: what the wgpu layer costs. The WGSL goes
//! through the same compiler (naga) with bounds checks off, and the dispatch loop
//! is the minimum Vulkan allows: bind, dispatch, barrier.

use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use std::io::Write;
use std::time::Instant;

use ash::vk;
use auto_core::program::{Bind, Pass, Program};
use auto_core::{automata, sim};
use clap::Parser;

#[derive(Parser)]
struct Args {
    id: String,
    #[arg(long, default_value = "4096x4096")]
    sizes: String,
    #[arg(long, default_value_t = 3.0)]
    seconds: f64,
    #[arg(long, default_value = "7900")]
    adapter: String,
    #[arg(long, default_value = "results/vulkan.jsonl")]
    out: String,
    /// Check results against wgpu after running.
    #[arg(long)]
    verify: bool,
}

const FRAME_STRIDE: u64 = 256;
const RING: u64 = 4096;

struct Buf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

struct Compiled {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pass: Pass,
}

struct Vk {
    // Dropping the entry unloads the Vulkan library, so it has to live as long as we do.
    _entry: ash::Entry,
    device: ash::Device,
    props: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    pool: vk::CommandPool,
    descriptors: vk::DescriptorPool,
    name: String,
}

fn spirv(program: &Program, pass: &Pass) -> Vec<u32> {
    let source = sim::wgsl_for(program, pass);
    let module = naga::front::wgsl::parse_str(&source).expect("wgsl parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("wgsl validates");
    let options = naga::back::spv::Options {
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Unchecked,
            buffer: naga::proc::BoundsCheckPolicy::Unchecked,
            image_load: naga::proc::BoundsCheckPolicy::Unchecked,
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
        ..Default::default()
    };
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".into(),
    };
    naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline)).expect("spirv")
}

impl Vk {
    unsafe fn new(adapter: &str) -> Self {
        unsafe {
            let entry = ash::Entry::load().expect("vulkan loader");
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app),
                    None,
                )
                .expect("instance");
            let want = adapter.to_lowercase();
            let (physical, name) = instance
                .enumerate_physical_devices()
                .expect("devices")
                .into_iter()
                .find_map(|p| {
                    let props = instance.get_physical_device_properties(p);
                    let name = CStr::from_ptr(props.device_name.as_ptr() as *const c_char)
                        .to_string_lossy()
                        .to_string();
                    name.to_lowercase().contains(&want).then_some((p, name))
                })
                .expect("adapter not found");
            let family = instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .position(|q| {
                    q.queue_flags
                        .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS)
                })
                .expect("queue") as u32;
            let priorities = [1.0];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities)];
            let device = instance
                .create_device(
                    physical,
                    &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info),
                    None,
                )
                .expect("device");
            let queue = device.get_device_queue(family, 0);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .expect("pool");
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 256,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 256,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1024,
                },
            ];
            let descriptors = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(256)
                        .pool_sizes(&sizes),
                    None,
                )
                .expect("descriptors");
            let props = instance.get_physical_device_memory_properties(physical);
            Self {
                _entry: entry,
                device,
                props,
                queue,
                pool,
                descriptors,
                name,
            }
        }
    }

    unsafe fn buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        flags: vk::MemoryPropertyFlags,
    ) -> Buf {
        unsafe {
            let buffer = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(size).usage(usage),
                    None,
                )
                .expect("buffer");
            let req = self.device.get_buffer_memory_requirements(buffer);
            let index = (0..self.props.memory_type_count)
                .find(|i| {
                    req.memory_type_bits & (1 << i) != 0
                        && self.props.memory_types[*i as usize]
                            .property_flags
                            .contains(flags)
                })
                .expect("memory type");
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(index),
                    None,
                )
                .expect("memory");
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("bind");
            Buf { buffer, memory }
        }
    }

    unsafe fn compile(&self, program: &Program, pass: &Pass) -> Compiled {
        unsafe {
            let code = spirv(program, pass);
            let module = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .expect("module");
            let mut bindings = vec![
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            for i in 0..pass.binds.len() {
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32 + 2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }
            let set_layout = self
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .expect("set layout");
            let layouts = [set_layout];
            let layout = self
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                    None,
                )
                .expect("layout");
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            let pipeline = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &info, None)
                .expect("pipeline")[0];
            Compiled {
                pipeline,
                layout,
                set_layout,
                pass: pass.clone(),
            }
        }
    }
}

struct Run<'a> {
    vk: &'a Vk,
    program: &'a Program,
    buffers: HashMap<&'static str, [Option<Buf>; 2]>,
    front: HashMap<&'static str, usize>,
    ring: Buf,
    params: Buf,
    ring_ptr: *mut u8,
    sets: HashMap<(usize, u32), vk::DescriptorSet>,
    tick: u64,
}

impl<'a> Run<'a> {
    fn mask(&self) -> u32 {
        let mut m = 0;
        for (i, f) in self.program.fields.iter().enumerate() {
            m |= (self.front[f.name] as u32) << i;
        }
        m
    }

    unsafe fn set_for(&mut self, id: usize, c: &Compiled) -> vk::DescriptorSet {
        let key = (id, self.mask());
        if let Some(s) = self.sets.get(&key) {
            return *s;
        }
        unsafe {
            let layouts = [c.set_layout];
            let set = self
                .vk
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.vk.descriptors)
                        .set_layouts(&layouts),
                )
                .expect("set")[0];
            let frame = [vk::DescriptorBufferInfo {
                buffer: self.ring.buffer,
                offset: 0,
                range: 16,
            }];
            let params = [vk::DescriptorBufferInfo {
                buffer: self.params.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            }];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = c
                .pass
                .binds
                .iter()
                .map(|b| {
                    let name = b.field();
                    let which = self.front[name] ^ usize::from(matches!(b, Bind::Write(..)));
                    let buf = self.buffers[name][which].as_ref().expect("buffer");
                    [vk::DescriptorBufferInfo {
                        buffer: buf.buffer,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    }]
                })
                .collect();
            let mut writes = vec![
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .buffer_info(&frame),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&params),
            ];
            for (i, info) in infos.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32 + 2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info),
                );
            }
            self.vk.device.update_descriptor_sets(&writes, &[]);
            self.sets.insert(key, set);
            set
        }
    }

    /// Advance `steps`, submitting whenever the frame ring is full, like wgpu does.
    unsafe fn steps(
        &mut self,
        passes: &[Compiled],
        steps: u32,
        cmd: vk::CommandBuffer,
        fence: vk::Fence,
    ) {
        let per_step: u64 = passes.iter().map(|c| c.pass.repeat as u64).sum();
        let max = (RING / per_step.max(1)).max(1) as u32;
        let mut left = steps;
        while left > 0 {
            let batch = left.min(max);
            // SAFETY: `run` waits for this submission before returning.
            unsafe { self.run(1000, passes, batch, cmd, fence) };
            self.tick += batch as u64;
            left -= batch;
        }
    }

    /// Clear every buffer, since Vulkan memory starts undefined and wgpu's starts zeroed.
    unsafe fn clear(&self, cmd: vk::CommandBuffer, fence: vk::Fence) {
        unsafe {
            let d = &self.vk.device;
            d.reset_fences(&[fence]).expect("fence");
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .expect("begin");
            for pair in self.buffers.values() {
                for buf in pair.iter().flatten() {
                    d.cmd_fill_buffer(cmd, buf.buffer, 0, vk::WHOLE_SIZE, 0);
                }
            }
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
            d.end_command_buffer(cmd).expect("end");
            let cmds = [cmd];
            d.queue_submit(
                self.vk.queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                fence,
            )
            .expect("submit");
            d.wait_for_fences(&[fence], true, u64::MAX).expect("wait");
        }
    }

    /// Run `steps` of `passes` and wait.
    unsafe fn run(
        &mut self,
        base: usize,
        passes: &[Compiled],
        steps: u32,
        cmd: vk::CommandBuffer,
        fence: vk::Fence,
    ) {
        unsafe {
            let d = &self.vk.device.clone();
            d.reset_fences(&[fence]).expect("fence");
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin");
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            let mut slot = 0u64;
            let mut current = None;
            for s in 0..steps as u64 {
                for (i, c) in passes.iter().enumerate() {
                    for phase in 0..c.pass.repeat {
                        let tick = self.tick + s;
                        let frame = [
                            tick as u32,
                            phase,
                            1u32,
                            (tick % self.program.height as u64) as u32,
                        ];
                        let at = (slot % RING) * FRAME_STRIDE;
                        std::ptr::copy_nonoverlapping(
                            frame.as_ptr() as *const u8,
                            self.ring_ptr.add(at as usize),
                            16,
                        );
                        let set = self.set_for(base + i, c);
                        if current != Some(i) {
                            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, c.pipeline);
                            current = Some(i);
                        }
                        d.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            c.layout,
                            0,
                            &[set],
                            &[at as u32],
                        );
                        let [x, y, z] = c.pass.dispatch;
                        d.cmd_dispatch(cmd, x, y, z);
                        d.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &barrier,
                            &[],
                            &[],
                        );
                        for field in &c.pass.swap {
                            *self.front.get_mut(field).expect("field") ^= 1;
                        }
                        slot += 1;
                    }
                }
            }
            d.end_command_buffer(cmd).expect("end");
            let cmds = [cmd];
            d.queue_submit(
                self.vk.queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                fence,
            )
            .expect("submit");
            d.wait_for_fences(&[fence], true, u64::MAX).expect("wait");
        }
    }
}

fn main() {
    let args = Args::parse();
    let vk_ctx = unsafe { Vk::new(&args.adapter) };
    for size in args.sizes.split(',') {
        let (w, h) = size.split_once('x').expect("WxH");
        let program =
            automata::build(&args.id, w.parse().expect("w"), h.parse().expect("h")).expect("id");
        let sample = unsafe { measure(&vk_ctx, &program, &args) };
        println!("{sample}");
        if let Some(parent) = std::path::Path::new(&args.out).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.out)
            .expect("results");
        writeln!(f, "{sample}").expect("write");
    }
}

unsafe fn measure(vkc: &Vk, program: &Program, args: &Args) -> String {
    unsafe {
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let mut buffers = HashMap::new();
        let mut front = HashMap::new();
        for f in &program.fields {
            let make = || vkc.buffer(f.bytes(), storage, vk::MemoryPropertyFlags::DEVICE_LOCAL);
            buffers.insert(f.name, [Some(make()), f.double.then(make)]);
            front.insert(f.name, 0usize);
        }
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let ring = vkc.buffer(
            FRAME_STRIDE * RING,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            host,
        );
        let ring_ptr = vkc
            .device
            .map_memory(ring.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .expect("map") as *mut u8;
        let bytes = if program.params.is_empty() {
            vec![0u8; 16]
        } else {
            program.params.clone()
        };
        let params = vkc.buffer(
            (bytes.len() as u64).next_multiple_of(16),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            host,
        );
        let p = vkc
            .device
            .map_memory(
                params.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
            .expect("map") as *mut u8;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());

        let init: Vec<Compiled> = program
            .init
            .iter()
            .map(|p| vkc.compile(program, p))
            .collect();
        let step: Vec<Compiled> = program
            .step
            .iter()
            .map(|p| vkc.compile(program, p))
            .collect();
        let cmd = vkc
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(vkc.pool)
                    .command_buffer_count(1),
            )
            .expect("cmd")[0];
        let fence = vkc
            .device
            .create_fence(&Default::default(), None)
            .expect("fence");
        let mut run = Run {
            vk: vkc,
            program,
            buffers,
            front,
            ring,
            params,
            ring_ptr,
            sets: HashMap::new(),
            tick: 0,
        };
        run.clear(cmd, fence);
        run.run(0, &init, 1, cmd, fence);
        run.steps(&step, 32, cmd, fence);

        // Same batching policy as the wgpu benchmark.
        let mut batch = 1u32;
        loop {
            let t = Instant::now();
            run.steps(&step, batch, cmd, fence);
            let dt = t.elapsed().as_secs_f64();
            if dt >= 0.02 || batch >= 1 << 16 {
                break;
            }
            let want = (batch as f64 * 0.025 / dt.max(1e-5)).ceil() as u32;
            batch = want.clamp(batch + 1, batch.saturating_mul(8)).min(1 << 16);
        }
        let mut rates = Vec::new();
        let start = Instant::now();
        while start.elapsed().as_secs_f64() < args.seconds || rates.len() < 3 {
            let t = Instant::now();
            run.steps(&step, batch, cmd, fence);
            rates.push(batch as f64 / t.elapsed().as_secs_f64());
        }
        rates.sort_by(|a, b| a.total_cmp(b));
        let median = rates[rates.len() / 2];
        let gens = program.generations_per_step as f64;

        if args.verify {
            verify(vkc, program, &mut run, cmd, fence);
        }
        for f in &program.fields {
            for buf in run
                .buffers
                .remove(f.name)
                .expect("field")
                .into_iter()
                .flatten()
            {
                vkc.device.destroy_buffer(buf.buffer, None);
                vkc.device.free_memory(buf.memory, None);
            }
        }
        vkc.device
            .reset_descriptor_pool(vkc.descriptors, vk::DescriptorPoolResetFlags::empty())
            .expect("reset");

        serde_json::json!({
            "id": program.id, "tag": "raw-vulkan", "adapter": vkc.name, "checks": "None",
            "width": program.width, "height": program.height, "cells": program.cells_per_step,
            "memory_bytes": program.total_bytes(), "steps_per_second": median,
            "generations_per_second": median * gens,
            "cells_per_second": median * gens * program.cells_per_step as f64,
            "batch_steps": batch, "batches": rates.len(),
        })
        .to_string()
    }
}

/// Run a fresh Vulkan simulation and a fresh wgpu one, and compare the bytes.
unsafe fn verify(
    vkc: &Vk,
    program: &Program,
    run: &mut Run,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
) {
    unsafe {
        let init: Vec<Compiled> = program
            .init
            .iter()
            .map(|p| vkc.compile(program, p))
            .collect();
        let step: Vec<Compiled> = program
            .step
            .iter()
            .map(|p| vkc.compile(program, p))
            .collect();
        for v in run.front.values_mut() {
            *v = 0;
        }
        run.sets.clear();
        vkc.device
            .reset_descriptor_pool(vkc.descriptors, vk::DescriptorPoolResetFlags::empty())
            .expect("reset");
        run.tick = 0;
        run.clear(cmd, fence);
        run.run(0, &init, 1, cmd, fence);
        run.steps(&step, 10, cmd, fence);

        let field = &program.fields[0];
        let staging = vkc.buffer(
            field.bytes(),
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        vkc.device.reset_fences(&[fence]).expect("fence");
        vkc.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .expect("begin");
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
        vkc.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
        let src = run.buffers[field.name][run.front[field.name]]
            .as_ref()
            .expect("buffer")
            .buffer;
        vkc.device.cmd_copy_buffer(
            cmd,
            src,
            staging.buffer,
            &[vk::BufferCopy::default().size(field.bytes())],
        );
        vkc.device.end_command_buffer(cmd).expect("end");
        let cmds = [cmd];
        vkc.device
            .queue_submit(
                vkc.queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                fence,
            )
            .expect("submit");
        vkc.device
            .wait_for_fences(&[fence], true, u64::MAX)
            .expect("wait");
        let p = vkc
            .device
            .map_memory(
                staging.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
            .expect("map") as *const u8;
        let got = std::slice::from_raw_parts(p, field.bytes() as usize).to_vec();

        let gpu = pollster::block_on(auto_core::Gpu::headless(auto_core::GpuOptions {
            adapter_name: Some("7900".into()),
            checks: auto_core::Checks::Safe,
        }))
        .expect("wgpu");
        let mut sim = auto_core::Sim::new(&gpu, program.clone(), 1).expect("sim");
        gpu.wait();
        sim.step(&gpu, 10);
        gpu.wait();
        assert!(
            sim.read_field(&gpu, field.name) == got,
            "raw Vulkan and wgpu disagree on {}",
            program.id
        );
        eprintln!(
            "verified: raw Vulkan and wgpu give identical bytes for {}",
            program.id
        );
    }
}
