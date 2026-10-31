# Source-only code review

Reviewed README.md, the engine and automata, CPU implementations, native/browser/Vulkan measurement paths, tests, benchmark scripts and report generator. No cargo commands, tests, benchmarks or GPU work were run. Findings below are static analysis; severity reflects impact on correctness or reported claims, not exploitability.

## High — The scripted elementary-rule benchmarks race on their input

**Location:** `crates/auto-core/src/automata/bits.rs:287–295`; `scripts/bench_all.sh:42–43`.

The script benchmarks both Rule 30 and Rule 110 with height 1. Consequently `src == dst == 0`: each invocation writes a word while adjacent invocations read that word as a neighbour. There is no dispatch-wide snapshot or synchronization. Even the 1024-cell case has conflicting invocations, and larger cases cross workgroups. Results depend on execution order instead of synchronous elementary-rule evolution. The reported throughput therefore measures an invalid implementation for every scripted elementary-rule size. The test uses height 40 and never exercises this case. Use separate source/destination buffers when history has only one row, or benchmark with at least two history rows.

## High — Vulkan reuses frame slots before submitting their consumers

**Location:** `crates/auto-vk/src/main.rs:297–298`, `:325`; compare `crates/auto-core/src/sim.rs:260–270`.

`run` records the entire batch before submission, but writes each frame to `(slot % 4096)`. Once a batch contains more than 4096 dispatches, later host writes replace uniforms that earlier dispatches have not read yet. For example, an 8192-step elementary-rule batch makes the first dispatch read the frame intended for dispatch 4096. FFT programs can overwrite both tick and phase with different values because the pass sequence need not divide the ring length. Native wgpu splits batches to fit its ring; raw Vulkan does not. This also gives raw Vulkan fewer submissions/waits for large batches, so the small-grid API comparison includes different batching overhead. Life's time-independent shaders do not observe the corrupted ticks, but the batching advantage still applies. Split and submit at ring capacity, accounting for repeats.

## High — Vulkan never advances its tick after timed batches

**Location:** `crates/auto-vk/src/main.rs:296`, `:326`, `:377–396`.

`run` uses `self.tick + s` but never increments `self.tick`; the only measurement-side assignment sets it to 32 after warmup. Each calibration and timed batch therefore starts again at tick 32. For example, Rule 110 with height 40 restarts reading row 32 each batch instead of continuing from the newest row. Forest, sand and Physarum likewise reuse randomness or phases differently from wgpu. This invalidates the runner's same-program comparison for time-dependent rules, although the two Life kernels currently used in the API script do not depend on tick. Advance tick after step batches and distinguish initialization.

## High — Raw Vulkan reads uninitialized storage that wgpu zeroes

**Location:** `crates/auto-vk/src/main.rs:349–355`; `crates/auto-core/src/automata/agents.rs:119–126`, `:170–171`, `:214–228`.

Raw storage allocations are never cleared. Physarum initializes only agents, then immediately reads `trail` and performs atomics on `deposit`; brains initializes only genome, then reads `state`. In wgpu these buffers begin as zero, whereas Vulkan allocation contents are undefined. A fresh raw Physarum run can sense arbitrary trails and accumulate arbitrary deposits; a brains run can begin with NaNs and remain NaN. Thus the runners do not perform equivalent work, and state-dependent timing can differ. This is an unsafe Vulkan initialization-contract problem even without a CPU pointer overrun. Explicitly initialize all fields that kernels read before writing. The current API script runs Life only, so this does not by itself invalidate its existing Life samples.

## Medium — Small FFT worlds silently use a different neighbourhood kernel

**Location:** `crates/auto-core/src/automata/spectral.rs:61–68`, `:133–135`.

Normalization sums the complete spatial kernel, but the GPU stores only one minimum-image sample at each torus coordinate. When a dimension is smaller than the kernel diameter, multiple offsets that wrap onto one cell must contribute to that cell's kernel weight. They are discarded instead. For example, SmoothLife on 16×16 uses radius 21, yet no stored minimum-image distance exceeds about 11.3: much of the outer ring is missing while its normalization still includes the full ring. Even a uniform field no longer convolves to itself. A narrow non-square grid triggers the same issue. Reject undersized worlds or periodize the full kernel by summing wrapped contributions. Existing Lenia convolution coverage uses 128×64 with radius 13 and misses this case; scripted FFT sizes are large enough.

## Medium — Elementary-rule history breaks at u32 tick wrap

**Location:** `crates/auto-core/src/automata/bits.rs:287–288`; tick truncation in `crates/auto-core/src/sim.rs:295`.

For history heights that do not divide 2^32, truncating tick before taking modulo changes the history cursor. With height 3 and tick `u32::MAX`, both source and destination become row 0 because `tick + 1` wraps first; the mathematical next row is 1. This introduces the same read/write race as height 1 and loses the current evolution at wrap. The next step also selects the wrong history row relative to a u64 timeline. Pass a separately computed history cursor or compute modulo on the host before truncation. Current tests never approach wrap.

## Medium — Report selection biases reruns upward and can mix adapters

**Location:** `scripts/make_report.py:202–208`, `:213–215`.

`best` selects the maximum throughput across appended reruns, rather than the latest run or an aggregate. Its key also omits adapter. For example, appending a slower rerun after correcting a kernel leaves the old faster result in the report; appending another GPU under the same tag/size can retain that GPU's result in a chart labelled RX 7900 XT. Independent maximum selection for Vulkan and wgpu also makes the reported relative overhead depend on which side received more reruns. Preserve adapter and run identity, select an explicitly identified campaign, and aggregate repeated samples consistently.

## Low — Hashlife silently ignores the highest generation bit

**Location:** `crates/auto-cpu/src/hashlife.rs:190`.

`advance(gens: u128)` iterates `0..127`, excluding bit 127. Calling `HashLife::new().advance(1u128 << 127)` does nothing and leaves `generation == 0`; a mixed request advances only its lower 127 bits. This violates the exact-advance contract without reporting an unsupported range. Iterate all supported bits or reject out-of-range requests explicitly. The report's scripted exponents stop at 70, so this does not affect those samples.
