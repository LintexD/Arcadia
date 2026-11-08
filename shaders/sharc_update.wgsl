// Bound by `arcadia-render` SHARC update pass (`sharc.rs`, runs after Pass 1 and before sharc_resolve.wgsl).
// Expected inputs: @group(0) world pools (types.wgsl), @group(1) G-Buffer (bindings 0..1), @group(2) SHARC cache (bindings 0..2).

// GPU-SYNC: Mirrors `SharcLayouts::gbuf_bgl` entry 0 in `crates/arcadia-render/src/sharc.rs` (@group(1) @binding(0)).
@group(1) @binding(0) var<storage, read> gbuf_depth_mat: array<vec2<u32>>;
// GPU-SYNC: Mirrors `SharcLayouts::gbuf_bgl` entry 1 in `crates/arcadia-render/src/sharc.rs` (@group(1) @binding(1)).
@group(1) @binding(1) var<storage, read> gbuf_norm_motion: array<vec2<u32>>;
// GPU-SYNC: Mirrors `SharcLayouts::update_bgl` entry 1 in `crates/arcadia-render/src/sharc.rs` (@group(2) @binding(1), 512 KB).
// @group(2) @binding(0) is `sharc_resolved` (declared in common/sharc_query.wgsl).
@group(2) @binding(1) var<storage, read_write> sharc_keys: array<atomic<u32>, 131072>;
// GPU-SYNC: Mirrors `SharcLayouts::update_bgl` entry 2 in `crates/arcadia-render/src/sharc.rs` (@group(2) @binding(2), 2.0 MB).
// Layout per slot: [r_fixed, g_fixed, b_fixed, sample_count].
@group(2) @binding(2) var<storage, read_write> sharc_accum: array<atomic<u32>, 524288>;

// Surface hit with the material terms needed to shade it:
struct SharcSurface {
    hit: bool,
    t_m: f32,
    pos: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    emission: vec3<f32>,
};

// Tile edge in pixels: 16 in low-tap (Tier 1) mode, else 8. Rust: `RaymarchQualitySettings::sharc_tile_size`.
fn sharc_tile_size() -> u32 {
    let taps = u_cam.quality_flags.w & 0xFFu;
    return select(8u, 16u, taps <= 12u);
}

// Traces static terrain plus rigid bodies and resolves albedo/emission exactly like Pass 2's GI bounce:
fn sharc_trace_surface(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> SharcSurface {
    var out: SharcSurface;
    out.hit = false;

    var tr = trace_world_64tree(origin, dir, max_dist, 2u, true);
    let rb_max = select(max_dist, tr.t_voxels * VOXEL_SIZE, tr.hit);
    let rb_count = (u_cam.point_light_count.y >> 16u) & 0xFFFFu;
    var rb: RigidTraceResult;
    rb.hit = false;
    if (rb_count > 0u) {
        rb = trace_rigid_bodies(origin, dir, rb_max);
    }
    if (rb.hit) {
        tr.hit = true;
        tr.t_voxels = rb.t_voxels;
        tr.hit_pos_meters = rb.hit_pos_meters;
        tr.normal = rb.normal;
        tr.mat_id = rb.mat_id;
        tr.lod_rgb = material_palette[rb.mat_id].albedo_roughness.xyz;
    }
    if (!tr.hit) {
        return out;
    }

    let m = material_palette[tr.mat_id];
    out.hit = true;
    out.t_m = tr.t_voxels * VOXEL_SIZE;
    out.pos = tr.hit_pos_meters;
    out.normal = tr.normal;
    out.albedo = mix(m.albedo_roughness.xyz, tr.lod_rgb, 0.55);
    out.emission = m.emission_meta.xyz;
    return out;
}

// Outgoing diffuse radiance at `s` (emission excluded; Pass 2 adds it analytically):
// L_o = albedo * (sun direct + one cosine-sampled bounce gathered from last frame's cache or the sky).
fn sharc_shade_surface(
    s: SharcSurface,
    xi: vec2<f32>,
    sun_dir: vec3<f32>,
    sun_power: f32,
    bounce_max_dist: f32,
) -> vec3<f32> {
    let n = s.normal;

    var direct = vec3<f32>(0.0);
    let ndl = max(dot(n, sun_dir), 0.0);
    if (ndl > 0.001 && sun_power > 0.01) {
        let vis = trace_shadow_vis(s.pos + n * 0.10, sun_dir, 32.0);
        direct = vec3<f32>(1.0, 0.90, 0.75) * sun_power * ndl * vis * 0.55;
    }

    let bounce_dir = sample_cosine_hemisphere(xi.x, xi.y, n, make_basis(n));
    let bounce_origin = s.pos + n * 0.10;
    let s2 = sharc_trace_surface(bounce_origin, bounce_dir, bounce_max_dist);

    var gathered = vec3<f32>(0.0);
    if (s2.hit) {
        let em_atten = 1.0 / (1.0 + s2.t_m * 0.06 + s2.t_m * s2.t_m * 0.025);
        let ndl_em = max(dot(n, bounce_dir), 0.25);
        gathered += s2.emission * 1.15 * em_atten * ndl_em;
        // INVARIANT: Reads last frame's resolved cache (a different pass wrote it); zero on miss.
        gathered += query_sharc(s2.pos, s2.normal, u_cam.cam_pos_time.xyz, vec2<f32>(0.0)).xyz;
    } else {
        // WHY: Only rays reaching open sky above terrain count as sky, matching Pass 2's escape heuristic.
        let end_p = bounce_origin + bounce_dir * bounce_max_dist;
        if (bounce_dir.y > 0.15 && end_p.y >= 32.0) {
            let sky_escape = clamp((bounce_dir.y - 0.15) / 0.35, 0.0, 1.0);
            gathered += eval_sky_radiance(bounce_dir, sun_dir) * 0.45 * sky_escape;
        }
    }

    return s.albedo * (direct + gathered);
}

// Claims (or finds) the slot for `k` in its 4-way bucket; returns SHARC_NO_SLOT when the bucket is full.
fn sharc_claim_slot(k: SharcKey) -> u32 {
    let base = k.bucket * SHARC_WAYS;

    // Existing surfel first so a later slot is never duplicated by an earlier slot that was evicted.
    for (var i = 0u; i < SHARC_WAYS; i = i + 1u) {
        if (atomicLoad(&sharc_keys[base + i]) == k.key) {
            return base + i;
        }
    }

    for (var i = 0u; i < SHARC_WAYS; i = i + 1u) {
        let idx = base + i;
        var res = atomicCompareExchangeWeak(&sharc_keys[idx], 0u, k.key);
        // WHY: WGSL only offers the weak CAS; retry once on a spurious failure that still observed an empty slot.
        if (!res.exchanged && res.old_value == 0u) {
            res = atomicCompareExchangeWeak(&sharc_keys[idx], 0u, k.key);
        }
        if (res.exchanged || res.old_value == k.key) {
            return idx;
        }
    }
    return SHARC_NO_SLOT;
}

// Accumulates radiance into the claimed slot using fixed-point `atomicAdd`s consumed by sharc_resolve.wgsl:
fn sharc_deposit(k: SharcKey, radiance: vec3<f32>) {
    let slot = sharc_claim_slot(k);
    if (slot == SHARC_NO_SLOT) {
        return;
    }
    let fixed_rgb = vec3<u32>(clamp(radiance, vec3<f32>(0.0), vec3<f32>(SHARC_MAX_RADIANCE)) * SHARC_FIXED_SCALE);
    let a = slot * 4u;
    atomicAdd(&sharc_accum[a], fixed_rgb.x);
    atomicAdd(&sharc_accum[a + 1u], fixed_rgb.y);
    atomicAdd(&sharc_accum[a + 2u], fixed_rgb.z);
    atomicAdd(&sharc_accum[a + 3u], 1u);
}

// One thread per screen tile: a jittered G-Buffer pixel seeds a path x_0 -> x_1 (bounce ray) with a sun shadow
// ray and one more bounce at x_1; L_o(x_1) is deposited into the cache keyed by (x_1, n_1).
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = screen_dims();
    let tile = sharc_tile_size();
    let tiles = (dims + vec2<u32>(tile - 1u)) / vec2<u32>(tile);
    if (gid.x >= tiles.x || gid.y >= tiles.y) {
        return;
    }

    let frame_idx = bitcast<u32>(u_cam.cam_up_frame.w);
    let h = sharc_pcg_hash(gid.x * 1973u + gid.y * 9277u + frame_idx * 26699u + 0x51ED270Bu);
    // INVARIANT: `tile` is a power of two, so masking yields a uniform in-tile offset.
    let offset = vec2<u32>(h & (tile - 1u), (h >> 8u) & (tile - 1u));
    let px = min(gid.xy * tile + offset, dims - vec2<u32>(1u));

    let p_idx = pixel_index(px.x, px.y, dims.x);
    let d_mat = gbuf_depth_mat[p_idx];
    let depth = unpack_depth(d_mat);
    if (depth >= 1e4) {
        return;
    }
    let x0 = reconstruct_world_pos(px, depth, dims);
    let n0 = unpack_normal_oct(gbuf_norm_motion[p_idx].x);

    let sun_dir = u_cam.sun_dir_intensity.xyz;
    let sun_power = u_cam.sun_dir_intensity.w;
    let tier1 = tile == 16u;
    let bounce_max_dist = min(u_cam.ray_distances.z, select(16.0, 10.0, tier1));

    let px_f = vec2<f32>(f32(px.x), f32(px.y));
    let xi0 = interleaved_gradient_noise(px_f, frame_idx);
    let xi1 = interleaved_gradient_noise(px_f + vec2<f32>(37.0, 91.0), frame_idx + 4099u);

    let dir0 = sample_cosine_hemisphere(xi0.x, xi0.y, n0, make_basis(n0));
    let s1 = sharc_trace_surface(x0 + n0 * 0.10, dir0, bounce_max_dist);
    if (!s1.hit) {
        return;
    }

    let radiance = sharc_shade_surface(s1, xi1, sun_dir, sun_power, bounce_max_dist);
    // INVARIANT: Deposits use zero tangent jitter; only reads dither across cell borders.
    let k = sharc_make_key(s1.pos, s1.normal, u_cam.cam_pos_time.xyz, vec2<f32>(0.0));
    sharc_deposit(k, radiance);
}
