// Bound by `arcadia-render` passes 1-4 (concatenated as common shader header).
// Expected inputs: @group(0) engine uniform buffer & world storage pools (bindings 0..6).

const VOXEL_SIZE: f32 = 0.0625;          // 1/16 meter (6.25 cm)
const VOXELS_PER_METER: f32 = 16.0;
const CHUNK_DIM: u32 = 256u;
const CHUNK_DIM_F: f32 = 256.0;
const EMPTY_CHUNK_SENTINEL: u32 = 0xFFFFFFFFu;

const IS_LEAF_FLAG: u32 = 0x80000000u;
const IS_UNIFORM_FLAG: u32 = 0x40000000u;
const PTR_INDEX_MASK: u32 = 0x3FFFFFFFu;

const MAT_AIR: u32 = 0u;
const MAT_WATER_CRYSTAL: u32 = 18u;
const MAT_POINT_LIGHT: u32 = 27u;
const MAT_MIRROR_CHROME: u32 = 28u;

struct CameraUniforms {
    // xyz = eye position (meters), w = elapsed time (seconds)
    cam_pos_time: vec4<f32>,
    // xyz = unit forward, w = tan(fov_y * 0.5)
    cam_forward_fov: vec4<f32>,
    // xyz = unit right, w = aspect (width / height)
    cam_right_aspect: vec4<f32>,
    // xyz = unit up, w = bitcast<f32>(frame_index)
    cam_up_frame: vec4<f32>,
    // Previous frame unjittered view-projection matrix (infinite Reverse-Z: near=1.0, far=0.0)
    prev_view_proj: mat4x4<f32>,
    // Current frame unjittered view-projection matrix (infinite Reverse-Z: near=1.0, far=0.0)
    curr_view_proj: mat4x4<f32>,
    // xyz = world chunk grid dims (gx, gy, gz), w = screen width | (screen height << 16)
    world_grid_dims: vec4<u32>,
    // xyz = normalized sun direction (towards sun), w = sun intensity
    sun_dir_intensity: vec4<f32>,
    // xy = sub-pixel Halton jitter in [-0.5, 0.5], z = brush radius (meters), w = dt (seconds)
    jitter_brush_dt: vec4<f32>,
    // xyz = targeted voxel hit position (meters), w = 1.0 if valid hit else 0.0
    target_voxel_info: vec4<f32>,
    // x = active analytical point light count (0..=8)
    point_light_count: vec4<u32>,
    // Up to 8 user-placed analytical 3D point lights (`xyz` = world pos, `w` = brush radius)
    point_light_0: vec4<f32>,
    point_light_1: vec4<f32>,
    point_light_2: vec4<f32>,
    point_light_3: vec4<f32>,
    point_light_4: vec4<f32>,
    point_light_5: vec4<f32>,
    point_light_6: vec4<f32>,
    point_light_7: vec4<f32>,
    // Dynamic Raymarch Quality Uniforms:
    // x = primary_max_steps, y = sec_max_steps, z = gi_max_steps, w = shadow_max_steps
    ray_steps: vec4<u32>,
    // x = water_refr_dist, y = mirror_refl_dist, z = gi_max_dist, w = shadow_max_dist
    ray_distances: vec4<f32>,
    // x = primary_lod_dist, y = sec_lod_dist, z = gi_lod_dist, w = shadow_jitter
    lod_settings: vec4<f32>,
    // x = primary_lod_step, y = sec_lod_step, z = shadow_mode (0=64tree, 1=fast anyhit),
    // w = denoiser_taps | (mirror_adaptive << 8) | (gi_lod_step << 16)
    quality_flags: vec4<u32>,
    // xyz = previous frame eye position (meters), w = 0.0
    prev_cam_pos: vec4<f32>,
};

struct Node64 {
    mask_low: u32,
    mask_high: u32,
    child_ptr: u32,
    lod_payload: u32,
};

struct LeafBrick64 {
    packed_materials: array<u32, 16>,
};

struct Material {
    albedo_roughness: vec4<f32>,
    emission_meta: vec4<f32>,
};

struct DebrisParticle {
    // xyz = world position (meters), w = remaining life (seconds)
    pos_life: vec4<f32>,
    // xyz = velocity (m/s), w = bitcast<f32>(packed_lod_payload)
    vel_color: vec4<f32>,
};

// ReSTIR GI Reservoir (32 bytes, 16-byte aligned)
// GPU-SYNC: Mirrors `GiReservoir` in `arcadia-render::gbuffer` (32 bytes, 16-byte aligned).
struct GiReservoir {
    // World-space secondary hit x_s (or x_v + bounce_dir * 500.0 when is_sky == 1)
    sample_pos: vec3<f32>,
    // [0..15]: snorm16 octahedral secondary normal n_s
    // [16..23]: M (u8 sample count, capped <= 16)
    // [24..30]: sample_age (u7 frames since path origin, invalidated > 12)
    // [31]: is_sky (1 bit)
    norm_m_flags: u32,
    // pack2x16float(Lo.r, Lo.g): outgoing radiance at x_s (excluding primary BRDF/cosine)
    lo_rg: u32,
    // pack2x16float(Lo.b, min(W, 65500.0)): Lo.b and clamped unbiased RIS weight W
    lo_b_w: u32,
    // Linear ray distance `hit_dist` of the primary pixel (for exact 3D disocclusion check)
    primary_depth: f32,
    // pack2x16float(accum_sun_shadow, shadow_m): decoupled temporal sun-shadow EMA
    shadow_vis_m: u32,
};

fn oct_wrap(v: vec2<f32>) -> vec2<f32> {
    return (vec2<f32>(1.0) - abs(v.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), v.xy >= vec2<f32>(0.0));
}

fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let l1_norm = abs(n.x) + abs(n.y) + abs(n.z);
    var p = n.xy * (1.0 / max(l1_norm, 1e-6));
    if (n.z < 0.0) {
        p = oct_wrap(p);
    }
    return p;
}

fn oct_decode(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        let signs = select(vec2<f32>(-1.0), vec2<f32>(1.0), p.xy >= vec2<f32>(0.0));
        let wrapped = (vec2<f32>(1.0) - abs(p.yx)) * signs;
        n.x = wrapped.x;
        n.y = wrapped.y;
    }
    return normalize(n);
}

fn pack_norm_m_flags(n: vec3<f32>, m: u32, sample_age: u32, is_sky: bool) -> u32 {
    let oct = oct_encode(n);
    let oct_u = vec2<u32>(clamp(vec2<i32>(round(oct * 127.0)), vec2<i32>(-128), vec2<i32>(127)) + 128);
    return (oct_u.x & 0xFFu) | ((oct_u.y & 0xFFu) << 8u) | ((min(m, 16u) & 0xFFu) << 16u) | ((min(sample_age, 127u) & 0x7Fu) << 24u) | select(0u, 0x80000000u, is_sky);
}

fn unpack_normal(norm_m_flags: u32) -> vec3<f32> {
    let ux = f32(norm_m_flags & 0xFFu);
    let uy = f32((norm_m_flags >> 8u) & 0xFFu);
    let oct = (vec2<f32>(ux, uy) - 128.0) / 127.0;
    return oct_decode(oct);
}

fn unpack_m(norm_m_flags: u32) -> u32 {
    return (norm_m_flags >> 16u) & 0xFFu;
}

fn unpack_sample_age(norm_m_flags: u32) -> u32 {
    return (norm_m_flags >> 24u) & 0x7Fu;
}

fn unpack_is_sky(norm_m_flags: u32) -> bool {
    return (norm_m_flags & 0x80000000u) != 0u;
}

fn pack_lo_w(lo: vec3<f32>, w: f32) -> vec2<u32> {
    let clamped_lo = clamp(lo, vec3<f32>(0.0), vec3<f32>(65500.0));
    let lo_rg = pack2x16float(clamped_lo.xy);
    let lo_b_w = pack2x16float(vec2<f32>(clamped_lo.z, min(w, 65500.0)));
    return vec2<u32>(lo_rg, lo_b_w);
}

fn unpack_lo(lo_rg: u32, lo_b_w: u32) -> vec3<f32> {
    let rg = unpack2x16float(lo_rg);
    let bw = unpack2x16float(lo_b_w);
    return vec3<f32>(rg.x, rg.y, bw.x);
}

fn unpack_w(lo_b_w: u32) -> f32 {
    let bw = unpack2x16float(lo_b_w);
    return bw.y;
}

fn pack_shadow_m(shadow_vis: f32, shadow_m: f32) -> u32 {
    return pack2x16float(vec2<f32>(shadow_vis, shadow_m));
}

fn unpack_shadow_m(shadow_vis_m: u32) -> vec2<f32> {
    return unpack2x16float(shadow_vis_m);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn create_empty_reservoir() -> GiReservoir {
    var res: GiReservoir;
    res.sample_pos = vec3<f32>(0.0);
    res.norm_m_flags = 0u;
    res.lo_rg = 0u;
    res.lo_b_w = 0u;
    res.primary_depth = 1e5;
    res.shadow_vis_m = 0u;
    return res;
}

fn eval_reservoir_gi(res: GiReservoir, x_v: vec3<f32>, n_v: vec3<f32>) -> vec3<f32> {
    let m = unpack_m(res.norm_m_flags);
    if (m == 0u) {
        return vec3<f32>(0.0);
    }
    let Lo = unpack_lo(res.lo_rg, res.lo_b_w);
    let W = unpack_w(res.lo_b_w);
    if (W <= 0.0 || luma(Lo) <= 0.0) {
        return vec3<f32>(0.0);
    }
    let to_s = res.sample_pos - x_v;
    let d = length(to_s);
    if (d < 1e-4) {
        return vec3<f32>(0.0);
    }
    let dir = to_s / d;
    let smooth_n = normalize(n_v + vec3<f32>(0.0, 0.22, 0.0));
    let cos_theta = max(0.0, dot(smooth_n, dir));
    return Lo * (cos_theta / 3.14159265) * W;
}

// Phase 6: Dynamic 6-DOF Rigid Body Instance (224 bytes, 16-byte aligned)
struct RigidBodyInstance {
    world_to_local: mat4x4<f32>,
    local_to_world: mat4x4<f32>,
    prev_local_to_world: mat4x4<f32>,
    local_bounds_and_root: vec4<f32>,
    world_sphere: vec4<f32>,
};

// GPU-SYNC: Mirrors `CameraUniforms` in `arcadia-render` (@group(0) @binding(0), 480 bytes, 16-byte aligned).
@group(0) @binding(0) var<uniform> u_cam: CameraUniforms;
// GPU-SYNC: Mirrors `chunk_roots` in `arcadia-world` (@group(0) @binding(1), storage read).
@group(0) @binding(1) var<storage, read> chunk_roots: array<u32>;
// GPU-SYNC: Mirrors `node_pool` in `arcadia-core` (@group(0) @binding(2), storage read).
@group(0) @binding(2) var<storage, read> node_pool: array<Node64>;
// GPU-SYNC: Mirrors `brick_pool` in `arcadia-core` (@group(0) @binding(3), storage read).
@group(0) @binding(3) var<storage, read> brick_pool: array<LeafBrick64>;
// GPU-SYNC: Mirrors `material_palette` in `arcadia-core` (@group(0) @binding(4), storage read).
@group(0) @binding(4) var<storage, read> material_palette: array<Material>;
// GPU-SYNC: Mirrors `debris_pool` in `arcadia-world` (@group(0) @binding(5), storage read_write).
@group(0) @binding(5) var<storage, read_write> debris_pool: array<DebrisParticle>;
// GPU-SYNC: Mirrors `rigid_bodies` in `arcadia-physics` (@group(0) @binding(6), storage read).
@group(0) @binding(6) var<storage, read> rigid_bodies: array<RigidBodyInstance>;

// Fetches analytical point light `idx` (0..7) without dynamic local array indexing:
fn get_point_light(idx: u32) -> vec4<f32> {
    switch (idx & 7u) {
        case 0u: { return u_cam.point_light_0; }
        case 1u: { return u_cam.point_light_1; }
        case 2u: { return u_cam.point_light_2; }
        case 3u: { return u_cam.point_light_3; }
        case 4u: { return u_cam.point_light_4; }
        case 5u: { return u_cam.point_light_5; }
        case 6u: { return u_cam.point_light_6; }
        default: { return u_cam.point_light_7; }
    }
}

// Fast PCG 32-bit hash for blue-noise / Monte Carlo sampling:
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967295.0;
}

fn unpack_lod_rgb(payload: u32) -> vec3<f32> {
    let r = f32(payload & 0xFFu) / 255.0;
    let g = f32((payload >> 8u) & 0xFFu) / 255.0;
    let b = f32((payload >> 16u) & 0xFFu) / 255.0;
    return vec3<f32>(r, g, b);
}

fn screen_dims() -> vec2<u32> {
    let packed = u_cam.world_grid_dims.w;
    return vec2<u32>(packed & 0xFFFFu, packed >> 16u);
}

fn pixel_index(px: u32, py: u32, width: u32) -> u32 {
    return py * width + px;
}

// ============================================================================
// Compact G-Buffer 20 B/px Packing, Unpacking & World Position Reconstruction
// Slot 0: gbuf_depth_mat_buf (8 B / vec2<u32>): depth_f32, u16 material_id, u16 flags
// Slot 1: gbuf_norm_motion_buf (8 B / vec2<u32>): octahedral_normal (pack2x16snorm), motion_vector (pack2x16float)
// Slot 2: gbuf_albedo_ao_buf (4 B / u32): u24 albedo (R8G8B8), u8 contact_ao
// ============================================================================

fn pack_depth_mat(depth: f32, mat_id: u32, flags: u32) -> vec2<u32> {
    return vec2<u32>(bitcast<u32>(depth), (mat_id & 0xFFFFu) | ((flags & 0xFFFFu) << 16u));
}

fn unpack_depth(d_mat: vec2<u32>) -> f32 {
    return bitcast<f32>(d_mat.x);
}

fn unpack_mat_id(d_mat: vec2<u32>) -> u32 {
    return d_mat.y & 0xFFFFu;
}

fn unpack_flags(d_mat: vec2<u32>) -> u32 {
    return (d_mat.y >> 16u) & 0xFFFFu;
}

fn pack_norm_motion(normal: vec3<f32>, motion: vec2<f32>) -> vec2<u32> {
    let oct = oct_encode(normal);
    return vec2<u32>(pack2x16snorm(oct), pack2x16float(motion));
}

fn unpack_normal_oct(packed: u32) -> vec3<f32> {
    let oct = unpack2x16snorm(packed);
    return oct_decode(oct);
}

fn unpack_motion(packed: u32) -> vec2<f32> {
    return unpack2x16float(packed);
}

fn pack_albedo_ao(albedo: vec3<f32>, contact_ao: f32) -> u32 {
    return pack4x8unorm(vec4<f32>(albedo, clamp(contact_ao, 0.0, 1.0)));
}

fn unpack_albedo(packed: u32) -> vec3<f32> {
    return unpack4x8unorm(packed).xyz;
}

fn unpack_contact_ao(packed: u32) -> f32 {
    return unpack4x8unorm(packed).w;
}

fn reconstruct_ray_dir(px: vec2<u32>, dims: vec2<u32>) -> vec3<f32> {
    let jitter = u_cam.jitter_brush_dt.xy;
    let uv = (vec2<f32>(f32(px.x) + 0.5 + jitter.x, f32(px.y) + 0.5 + jitter.y)) / vec2<f32>(dims);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let tan_half_fov = u_cam.cam_forward_fov.w;
    let aspect = u_cam.cam_right_aspect.w;
    return normalize(
        u_cam.cam_forward_fov.xyz
        + u_cam.cam_right_aspect.xyz * (ndc.x * aspect * tan_half_fov)
        + u_cam.cam_up_frame.xyz * (ndc.y * tan_half_fov)
    );
}

fn reconstruct_world_pos(px: vec2<u32>, depth: f32, dims: vec2<u32>) -> vec3<f32> {
    let ray_dir = reconstruct_ray_dir(px, dims);
    return u_cam.cam_pos_time.xyz + ray_dir * depth;
}

// Hash helper for procedural Voronoi caustic lattice:
fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(443.897, 441.423, 437.195));
    p3 += dot(p3, p3.yzx + 19.19);
    return fract(vec2<f32>(p3.x * p3.y, p3.y * p3.z));
}

// Multi-harmonic Gerstner wave domain warping:
// Warps straight spatial coordinates into fluid, serpentine organic curves:
fn gerstner_warp(p: vec2<f32>, t: f32) -> vec2<f32> {
    var q = p;
    let w1 = t * 1.1;
    q.x += sin(p.y * 1.6 + w1) * 0.26 + cos(p.x * 1.2 - w1 * 0.7) * 0.18;
    q.y += cos(p.x * 1.7 - w1 * 0.8) * 0.26 + sin(p.y * 1.1 + w1 * 0.6) * 0.18;

    let w2 = t * 1.8;
    q.x += sin(q.x * 2.6 + q.y * 1.8 + w2) * 0.10;
    q.y += cos(q.x * 1.5 - q.y * 2.7 - w2 * 1.1) * 0.10;
    return q;
}

// Evaluates a single multi-channel cellular Voronoi caustic sample with chromatic dispersion:
fn eval_voronoi_caustic_rgb(uv: vec2<f32>, disp: vec2<f32>, t: f32) -> vec3<f32> {
    let n = floor(uv);
    let f = uv - n;

    var m1_r = 1e4; var m2_r = 1e4;
    var m1_g = 1e4; var m2_g = 1e4;
    var m1_b = 1e4; var m2_b = 1e4;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let node_offset = vec2<f32>(f32(dx), f32(dy));
            let h = hash22(n + node_offset);

            // Animate feature points organically across the cell:
            let pt = 0.5 + 0.38 * sin(vec2<f32>(t * 1.4 + h.x * 6.28318, t * 1.2 + h.y * 6.28318));
            let r = node_offset + pt - f;

            // Green channel (un-dispersed center):
            let dg = dot(r, r);
            if (dg < m1_g) { m2_g = m1_g; m1_g = dg; }
            else if (dg < m2_g) { m2_g = dg; }

            // Red channel (offset -disp):
            let rr = r + disp;
            let dr = dot(rr, rr);
            if (dr < m1_r) { m2_r = m1_r; m1_r = dr; }
            else if (dr < m2_r) { m2_r = dr; }

            // Blue channel (offset +disp):
            let rb = r - disp;
            let db = dot(rb, rb);
            if (db < m1_b) { m2_b = m1_b; m1_b = db; }
            else if (db < m2_b) { m2_b = db; }
        }
    }

    let del_r = sqrt(m2_r) - sqrt(m1_r);
    let del_g = sqrt(m2_g) - sqrt(m1_g);
    let del_b = sqrt(m2_b) - sqrt(m1_b);

    // Glowing filament sharpness (F2 - F1 ridge profile):
    let fil_r = pow(max(0.0, 1.0 - del_r * 4.2), 2.2);
    let fil_g = pow(max(0.0, 1.0 - del_g * 4.2), 2.2);
    let fil_b = pow(max(0.0, 1.0 - del_b * 4.2), 2.2);

    // Subtle ambient cell interior glow:
    let cell_glow = pow(max(0.0, 1.0 - sqrt(m1_g) * 1.4), 2.5) * 0.25;

    return vec3<f32>(fil_r, fil_g, fil_b) * 2.2 + vec3<f32>(cell_glow);
}

// Multi-directional fluid surface water normal for waves and reflections:
fn eval_surface_water_normal(xz: vec2<f32>, t_sec: f32) -> vec3<f32> {
    let p = xz * 1.8;
    let w1 = t_sec * 1.4;
    let w2 = t_sec * 2.2;
    let wx = sin(p.y * 2.2 + w1) * 0.024 + cos(p.x * 3.4 - w2) * 0.016 + sin(p.x * 5.6 + p.y * 4.2 + t_sec * 3.0) * 0.008;
    let wz = cos(p.x * 2.4 - w1 * 0.9) * 0.024 + sin(p.y * 3.1 + w2 * 0.8) * 0.016 + cos(p.x * 4.1 - p.y * 5.3 - t_sec * 2.7) * 0.008;
    return normalize(vec3<f32>(wx, 1.0, wz));
}

// Computes analytical water caustics with reverse-projected sunlight vector,
// Gerstner-warped cellular Voronoi filaments, directional chromatic dispersion, and vertical normal attenuation:
fn eval_water_caustics(
    bed_pos: vec3<f32>,
    water_y: f32,
    water_normal: vec3<f32>,
    sun_dir: vec3<f32>,
    t_sec: f32,
) -> vec3<f32> {
    let day_factor = clamp(sun_dir.y * 3.0, 0.0, 1.0);
    if (day_factor <= 0.001) {
        return vec3<f32>(1.0);
    }

    let sun_down = -normalize(sun_dir);
    let refr_down = refract(sun_down, vec3<f32>(0.0, 1.0, 0.0), 0.75);
    // INVARIANT: Air-to-water refraction guarantees refr_down.y is strictly negative.
    let l_refr = select(vec3<f32>(0.0, 1.0, 0.0), -normalize(refr_down), refr_down.y < -0.1);

    let depth = max(water_y - bed_pos.y, 0.0);
    let proj_dist = depth / max(l_refr.y, 0.1);
    let water_hit = bed_pos + l_refr * proj_dist;

    let uv = water_hit.xz * 2.0;
    let warped_uv1 = gerstner_warp(uv, t_sec);
    let warped_uv2 = gerstner_warp(uv * 1.8 + vec2<f32>(1.7, -2.4), t_sec * 1.3);

    // Directional Chromatic Dispersion offset aligned with wave displacement:
    let disp = clamp((warped_uv1 - uv) * 0.08, vec2<f32>(-0.025), vec2<f32>(0.025));

    let caustic1 = eval_voronoi_caustic_rgb(warped_uv1, disp, t_sec);
    let caustic2 = eval_voronoi_caustic_rgb(warped_uv2, disp * 1.4, t_sec * 1.25);
    let caustic_web = caustic1 * 0.65 + caustic2 * 0.40 + caustic1 * caustic2 * 0.45;

    // Depth focus modulation:
    let depth_focus = smoothstep(0.05, 0.6, depth) * exp(-depth * 0.06);
    let caustic_focused = mix(vec3<f32>(1.0), vec3<f32>(0.25) + caustic_web * 1.4, depth_focus);

    // Waterfall / Vertical Edge Guardrail (Section 4.4):
    let omega_plane = clamp((water_normal.y - 0.2) / 0.8, 0.0, 1.0);
    let caustic_plane = mix(vec3<f32>(1.0), caustic_focused, omega_plane);

    return mix(vec3<f32>(1.0), caustic_plane, day_factor);
}

// Truly independent 2D Spatial-Temporal Blue-Noise / R2 Hemisphere Sampler:
// (Fixes the 1D Fermat spiral bug where `ign1` and `ign2` shared the same linear gradient
// `(0.06711056, 0.00583715)`, which locked `u1 = r^2` and `u2 = phi / 2pi` onto a 1D spiral!)
fn interleaved_gradient_noise(px: vec2<f32>, frame: u32) -> vec2<f32> {
    let f = f32(frame);
    let ux = u32(px.x);
    let uy = u32(px.y);
    // Orthogonal spatial gradients + independent PCG hashes + 2D R2 quasi-random sequence:
    let ign1 = fract(52.9829189 * fract(0.06711056 * px.x + 0.00583715 * px.y));
    let ign2 = fract(52.9829189 * fract(0.00583715 * px.x - 0.06711056 * px.y + 0.5));
    let h1 = pcg_hash(ux * 1973u + uy * 9277u + frame * 26699u);
    let h2 = pcg_hash(h1 ^ 0x9E3779B9u);
    let r1 = f32(h1 & 0xFFFFu) / 65536.0;
    let r2 = f32(h2 & 0xFFFFu) / 65536.0;
    let u1 = fract(ign1 + f * 0.754877666 + r1);
    let u2 = fract(ign2 + f * 0.569840291 + r2);
    return vec2<f32>(u1, u2);
}

// Evaluates the atmospheric sky dome radiance in direction `dir` across Day, Sunset, and Night:
fn eval_sky_radiance(dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_elev = sun_dir.y;
    let day_t = clamp((sun_elev + 0.05) / 0.30, 0.0, 1.0);
    let twilight_t = clamp(1.0 - abs(sun_elev - 0.02) / 0.22, 0.0, 1.0);

    let zenith = clamp(dir.y, 0.0, 1.0);
    let horizon = pow(1.0 - zenith, 3.0);

    let day_zenith = vec3<f32>(0.08, 0.22, 0.55);
    let day_horizon = vec3<f32>(0.38, 0.50, 0.68);
    let night_zenith = vec3<f32>(0.0015, 0.0025, 0.0055);
    let night_horizon = vec3<f32>(0.0035, 0.0065, 0.0120);

    var sky_z = mix(night_zenith, day_zenith, day_t);
    var sky_h = mix(night_horizon, day_horizon, day_t);
    sky_h = mix(sky_h, vec3<f32>(0.65, 0.25, 0.10), twilight_t * 0.75);

    var sky = mix(sky_z, sky_h, horizon);
    let sun_cos = max(dot(dir, sun_dir), 0.0);
    if (sun_elev > -0.12) {
        sky += vec3<f32>(0.85, 0.52, 0.24) * pow(sun_cos, 14.0) * 0.35 * max(day_t, twilight_t);
    }
    return sky;
}

// Orthonormal tangent frame (tangent `t`, bitangent `b`) around a unit normal:
struct Basis {
    t: vec3<f32>,
    b: vec3<f32>,
}

fn make_basis(n: vec3<f32>) -> Basis {
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    var bs: Basis;
    bs.t = normalize(cross(up, n));
    bs.b = cross(n, bs.t);
    return bs;
}

// Cosine-weighted hemisphere sample around `n` using a precomputed frame; result is unit length by construction:
fn sample_cosine_hemisphere(u1: f32, u2: f32, n: vec3<f32>, bs: Basis) -> vec3<f32> {
    let r = sqrt(u1);
    let phi = 6.2831853 * u2;
    let z = sqrt(max(0.0, 1.0 - u1));
    return bs.t * (r * cos(phi)) + bs.b * (r * sin(phi)) + n * z;
}
