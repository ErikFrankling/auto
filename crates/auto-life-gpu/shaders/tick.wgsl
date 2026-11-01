// Free slots are a ring. Births only take slots that were free when the tick
// began, so a slot freed this tick can't be read while it is being written.

@compute @workgroup_size(1)
fn start() {
    let avail = ctl[0];
    var tail = atomicLoad(&ctr[C_TAIL]);
    // Births that asked for more than there was took nothing.
    if (i32(tail - avail) > 0) {
        tail = avail;
        atomicStore(&ctr[C_TAIL], avail);
    }
    let head = atomicLoad(&ctr[C_HEAD]);
    ctl[0] = head;
    ctl[1] = SLOTS - (head - tail);
    atomicStore(&ctr[C_JOBS], 0u);
}

@compute @workgroup_size(1)
fn finish() {
    let n = atomicLoad(&ctr[C_JOBS]);
    ctl[2] = n;
    args[0] = (n + 31u) / 32u;
    args[1] = 1u;
    args[2] = 1u;
}
