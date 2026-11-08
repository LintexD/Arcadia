// Binding-free SHARC voxel-surfel cache helpers (constants, key hashing, f16 packing).
// Concatenated by `arcadia-render` `sharc.rs` into sharc_update.wgsl, sharc_resolve.wgsl and Pass 2 (lighting_restir_gi.wgsl).

const SHARC_BUCKETS: u32 = 32768u;       // Rust: sharc::SHARC_BUCKETS
const SHARC_WAYS: u32 = 4u;              // Rust: sharc::SHARC_WAYS (slots per bucket = one 64-byte cache line of `SharcResolvedSlot`)
const SHARC_SLOTS: u32 = 131072u;        // Rust: sharc::SHARC_SLOTS (= SHARC_BUCKETS * SHARC_WAYS)
const SHARC_NO_SLOT: u32 = 0xFFFFFFFFu;  // Sentinel returned when a bucket is full of other surfels
const SHARC_FIXED_SCALE: f32 = 1024.0;   // Fixed-point radiance scale for `atomic<u32>` accumulation
const SHARC_MAX_RADIANCE: f32 = 32.0;    // Per-deposit clamp so 131,072 hits/frame cannot overflow a u32 accumulator
const SHARC_STALE_EVICT_AGE: u32 = 64u;  // Frames without a deposit before a surfel is evicted
const SHARC_HISTORY_CAP: f32 = 16.0;     // EMA window: alpha = 1 / min(hist_frames, 16)
const SHARC_KEY_VALID_BIT: u32 = 0x80000000u; // Bit 31 guarantees `key != 0u`; `0u` is reserved for empty slots

// GPU-SYNC: Mirrors `SharcResolvedSlot` in `crates/arcadia-render/src/sharc.rs` (16 bytes, 4-byte aligned).
// `rad_rg` = pack2x16float(r, g); `rad_b_meta` = f16 b | (hist_frames << 16) | (stale_age << 24).
struct SharcResolvedSlot {
    key: u32,
    rad_rg: u32,
    rad_b_meta: u32,
    pad: u32,
};

// Bucket index (`0..SHARC_BUCKETS`) and 32-bit key signature identifying one voxel-surfel:
struct SharcKey {
    bucket: u32,
    key: u32,
};

// Orthonormal tangent frame used for the sub-cell query dither:
struct SharcFrame {
    t: vec3<f32>,
    b: vec3<f32>,
};

// Self-contained PCG hash (no dependency on types.wgsl so sharc_resolve.wgsl stays world-binding free):
fn sharc_pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Cascade level `0..3` from camera distance: 0.25 m surfels under 8 m, then 0.5 / 1.0 / 2.0 m.
fn sharc_lod_for_distance(d: f32) -> u32 {
    return u32(clamp(floor(log2(max(d, 4.0) * 0.25)), 0.0, 3.0));
}

// Face id `0..5` for +X, -X, +Y, -Y, +Z, -Z; continuous rigid-body normals snap to their dominant axis.
fn sharc_norm_id(normal: vec3<f32>) -> u32 {
    let abs_n = abs(normal);
    if (abs_n.x > abs_n.y && abs_n.x > abs_n.z) {
        return select(1u, 0u, normal.x > 0.0);
    }
    if (abs_n.y > abs_n.z) {
        return select(3u, 2u, normal.y > 0.0);
    }
    return select(5u, 4u, normal.z > 0.0);
}

fn sharc_tangent_frame(n: vec3<f32>) -> SharcFrame {
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    var f: SharcFrame;
    f.t = normalize(cross(up, n));
    f.b = cross(n, f.t);
    return f;
}

// Builds the face-separated voxel-surfel key for a hit. `tangent_jitter` must be (0, 0) when depositing and
// blue-noise in [-1, 1]^2 when reading; the 2 cm inward anchor keeps face hits from flickering across cell borders.
fn sharc_make_key(pos: vec3<f32>, normal: vec3<f32>, cam_pos: vec3<f32>, tangent_jitter: vec2<f32>) -> SharcKey {
    let lod = sharc_lod_for_distance(distance(pos, cam_pos));
    let s = 0.25 * f32(1u << lod);

    var anchor = pos - normal * 0.02;
    if (tangent_jitter.x != 0.0 || tangent_jitter.y != 0.0) {
        let frame = sharc_tangent_frame(normal);
        anchor += (frame.t * tangent_jitter.x + frame.b * tangent_jitter.y) * (0.42 * s);
    }

    let cell = bitcast<vec3<u32>>(vec3<i32>(floor(anchor / s)) + vec3<i32>(16384));
    let norm_id = sharc_norm_id(normal);

    let bucket = sharc_pcg_hash(
        cell.x * 73856093u ^ cell.y * 19349663u ^ cell.z * 83492791u ^ norm_id * 2654435761u ^ lod * 38069u
    ) & (SHARC_BUCKETS - 1u);
    let key = sharc_pcg_hash(
        bucket ^ (cell.x * 1013904223u) ^ (cell.y * 1664525u) ^ (cell.z * 2891336453u) ^ (norm_id << 20u) ^ (lod << 24u)
    ) | SHARC_KEY_VALID_BIT;

    var out: SharcKey;
    out.bucket = bucket;
    out.key = key;
    return out;
}

// Packs radiance (clamped to the f16-safe range) and metadata into `(rad_rg, rad_b_meta)`:
fn sharc_pack_resolved(rgb: vec3<f32>, hist_frames: u32, stale_age: u32) -> vec2<u32> {
    let c = clamp(rgb, vec3<f32>(0.0), vec3<f32>(SHARC_MAX_RADIANCE));
    let rad_rg = pack2x16float(c.xy);
    let blue_f16 = pack2x16float(vec2<f32>(c.z, 0.0)) & 0xFFFFu;
    let rad_b_meta = blue_f16 | ((hist_frames & 0xFFu) << 16u) | ((stale_age & 0xFFu) << 24u);
    return vec2<u32>(rad_rg, rad_b_meta);
}

fn sharc_unpack_radiance(slot: SharcResolvedSlot) -> vec3<f32> {
    let rg = unpack2x16float(slot.rad_rg);
    let b = unpack2x16float(slot.rad_b_meta & 0xFFFFu).x;
    return vec3<f32>(rg.x, rg.y, b);
}

fn sharc_unpack_hist_frames(slot: SharcResolvedSlot) -> u32 {
    return (slot.rad_b_meta >> 16u) & 0xFFu;
}

fn sharc_unpack_stale_age(slot: SharcResolvedSlot) -> u32 {
    return (slot.rad_b_meta >> 24u) & 0xFFu;
}

