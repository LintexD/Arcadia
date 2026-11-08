// Bound by `arcadia-render` SHARC resolve pass (`sharc.rs`, runs after sharc_update.wgsl, 0 rays).
// Expected inputs: @group(0) SHARC cache (bindings 0..2); requires common/sharc.wgsl; touches no world pools.

// GPU-SYNC: Mirrors `SharcLayouts::resolve_bgl` entry 0 in `crates/arcadia-render/src/sharc.rs` (@group(0) @binding(0), 512 KB).
@group(0) @binding(0) var<storage, read_write> sharc_keys: array<atomic<u32>, 131072>;
// GPU-SYNC: Mirrors `SharcLayouts::resolve_bgl` entry 1 in `crates/arcadia-render/src/sharc.rs` (@group(0) @binding(1), 2.0 MB).
@group(0) @binding(1) var<storage, read_write> sharc_accum: array<atomic<u32>, 524288>;
// GPU-SYNC: Mirrors `SharcLayouts::resolve_bgl` entry 2 in `crates/arcadia-render/src/sharc.rs` (@group(0) @binding(2), 2.0 MB).
@group(0) @binding(2) var<storage, read_write> sharc_resolved: array<SharcResolvedSlot, 131072>;

// One thread per slot: folds this frame's accumulated deposits into the EMA, ages untouched surfels and
// evicts them past `SHARC_STALE_EVICT_AGE`, then clears the accumulator for the next update pass.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= SHARC_SLOTS) {
        return;
    }

    let key = atomicLoad(&sharc_keys[slot]);
    if (key == 0u) {
        sharc_resolved[slot] = SharcResolvedSlot(0u, 0u, 0u, 0u);
        return;
    }

    let a = slot * 4u;
    let count = atomicLoad(&sharc_accum[a + 3u]);
    let prev = sharc_resolved[slot];
    // INVARIANT: History is only valid when the resolved key matches; a re-claimed slot starts fresh.
    let prev_valid = prev.key == key;

    var rad = vec3<f32>(0.0);
    var hist_frames = 0u;
    var stale_age = 0u;
    if (prev_valid) {
        rad = sharc_unpack_radiance(prev);
        hist_frames = sharc_unpack_hist_frames(prev);
        stale_age = sharc_unpack_stale_age(prev);
    }

    if (count > 0u) {
        let inv = 1.0 / (f32(count) * SHARC_FIXED_SCALE);
        let sample_rgb = vec3<f32>(
            f32(atomicLoad(&sharc_accum[a])),
            f32(atomicLoad(&sharc_accum[a + 1u])),
            f32(atomicLoad(&sharc_accum[a + 2u])),
        ) * inv;
        hist_frames = min(hist_frames + 1u, 255u);
        // Fast-start EMA: the first sample (hist_frames == 1) fully replaces the empty value.
        let alpha = 1.0 / min(f32(hist_frames), SHARC_HISTORY_CAP);
        rad = mix(rad, sample_rgb, alpha);
        stale_age = 0u;
        atomicStore(&sharc_accum[a], 0u);
        atomicStore(&sharc_accum[a + 1u], 0u);
        atomicStore(&sharc_accum[a + 2u], 0u);
        atomicStore(&sharc_accum[a + 3u], 0u);
    } else {
        stale_age = stale_age + 1u;
        // WHY: Evicting untouched surfels auto-cleans carved or collapsed terrain without CPU invalidation.
        if (!prev_valid || stale_age > SHARC_STALE_EVICT_AGE) {
            atomicStore(&sharc_keys[slot], 0u);
            sharc_resolved[slot] = SharcResolvedSlot(0u, 0u, 0u, 0u);
            return;
        }
    }

    let packed = sharc_pack_resolved(rad, hist_frames, stale_age);
    sharc_resolved[slot] = SharcResolvedSlot(key, packed.x, packed.y, 0u);
}

