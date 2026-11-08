// Bound by `arcadia-render` Pass 4 (Internal TAA / DLAA temporal resolve and aerial fog).
// Expected inputs: @group(0) engine world pools, @group(1) Pass 4 G-Buffer, history, output texture (bindings 2, 5..8).

@group(1) @binding(2) var<storage, read> gbuf_norm_motion: array<vec2<u32>>; // Rust: pass4_bgl entry 0 (gbuf_norm_motion_buf)
@group(1) @binding(5) var<storage, read> lit_color_buf: array<vec4<f32>>; // Rust: pass4_bgl entry 1 (lit_color_buf)
@group(1) @binding(6) var<storage, read> history_prev: array<vec4<f32>>; // Rust: pass4_bgl entry 2 (history_prev_buf)
@group(1) @binding(7) var<storage, read_write> history_curr: array<vec4<f32>>; // Rust: pass4_bgl entry 3 (history_curr_buf)
@group(1) @binding(8) var out_tex: texture_storage_2d<rgba16float, write>; // Rust: pass4_bgl entry 4 (out_tex_view)

fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
    );
}

fn ycocg_to_rgb(y: vec3<f32>) -> vec3<f32> {
    let tmp = y.x - y.z;
    return max(vec3<f32>(0.0), vec3<f32>(tmp + y.y, y.x + y.z, tmp - y.y));
}

// Karis reversible perceptual HDR tone-weight (`c / (1 + luma)`) to prevent high-contrast
// emissive crystal silhouettes from flickering or firefly-aliasing during temporal resolve:
fn hdr_to_perceptual(c: vec3<f32>) -> vec3<f32> {
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    return c / (1.0 + luma);
}

fn perceptual_to_hdr(p: vec3<f32>) -> vec3<f32> {
    let luma = clamp(dot(p, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 0.985);
    return p / (1.0 - luma);
}

// Narkowicz ACES Filmic Tonemapping curve:
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Minimalist Polynomial AgX display transform (Benjamin Wrensch / Troy Sobotka):
fn agx_tonemap(val: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.84247928, 0.04232824, 0.04237565),
        vec3<f32>(0.07843360, 0.87846864, 0.07843360),
        vec3<f32>(0.07922372, 0.07922372, 0.87914297)
    );
    let outset = mat3x3<f32>(
        vec3<f32>(1.1968790, -0.0528969, -0.0529716),
        vec3<f32>(-0.0980209, 1.1519031, -0.0980435),
        vec3<f32>(-0.0990297, -0.0989612, 1.1510444)
    );
    let val_inset = max(inset * val, vec3<f32>(1e-5));
    let min_ev = -10.0;
    let max_ev = 6.5;
    let log_val = clamp((log2(val_inset) - min_ev) / (max_ev - min_ev), vec3<f32>(0.0), vec3<f32>(1.0));
    let x = log_val;
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let f = 15.5 * x6 - 40.14 * (x4 * x) + 31.96 * x4 - 6.868 * (x2 * x) + 0.4298 * x2 + 0.1191 * x - 0.00232;
    let agx_display = clamp(f, vec3<f32>(0.0), vec3<f32>(1.0));
    return clamp(outset * agx_display, vec3<f32>(0.0), vec3<f32>(1.0));
}

// High-Precision Decorrelated Triangular Probability Density Function (TPDF) Dither
fn hash_pcg2d(v: vec2<u32>) -> vec2<u32> {
    var q = v * 1664525u + 1013904223u;
    q.x += q.y * 1664525u;
    q.y += q.x * 1664525u;
    q ^= (q >> vec2<u32>(16u));
    q.x += q.y * 1664525u;
    q.y += q.x * 1664525u;
    q ^= (q >> vec2<u32>(16u));
    return q;
}

// Decorrelated Triangular Probability Density Function (TPDF) dither via 2D PCG hash.
fn pcg_tpdf_dither(pos: vec2<u32>, frame: u32) -> vec3<f32> {
    let p_rg = pos + vec2<u32>(frame * 19u, frame * 37u);
    let p_ba = pos.yx + vec2<u32>(frame * 53u + 0x55555555u, frame * 79u + 0xAAAAAAAAu);

    let h_rg = hash_pcg2d(p_rg);
    let h_ba = hash_pcg2d(p_ba);

    let u_r1 = f32(h_rg.x & 0xFFFFu) / 65535.0;
    let u_r2 = f32(h_rg.x >> 16u) / 65535.0;

    let u_g1 = f32(h_rg.y & 0xFFFFu) / 65535.0;
    let u_g2 = f32(h_rg.y >> 16u) / 65535.0;

    let u_b1 = f32(h_ba.x & 0xFFFFu) / 65535.0;
    let u_b2 = f32(h_ba.x >> 16u) / 65535.0;

    let d_r = (u_r1 - u_r2) / 255.0;
    let d_g = (u_g1 - u_g2) / 255.0;
    let d_b = (u_b1 - u_b2) / 255.0;

    return vec3<f32>(d_r, d_g, d_b);
}

// Mode 2: Spatial Integer Bit-Mix TPDF Dither
fn spatial_tpdf_dither(pos: vec2<u32>) -> vec3<f32> {
    var h1 = pos.x * 1597334677u ^ pos.y * 3812015801u;
    h1 = (h1 ^ (h1 >> 16u)) * 2246822519u;
    h1 = (h1 ^ (h1 >> 13u)) * 3266489917u;
    h1 = h1 ^ (h1 >> 16u);

    var h2 = pos.x * 2798796413u ^ pos.y * 1159711829u;
    h2 = (h2 ^ (h2 >> 16u)) * 2246822519u;
    h2 = (h2 ^ (h2 >> 13u)) * 3266489917u;
    h2 = h2 ^ (h2 >> 16u);

    let u1 = f32(h1 & 0xFFFFu) / 65535.0;
    let u2 = f32(h2 & 0xFFFFu) / 65535.0;
    let tpdf = (u1 - u2) / 255.0;
    return vec3<f32>(tpdf);
}

fn eval_dither(pos: vec2<u32>, frame: u32, mode: u32) -> vec3<f32> {
    if (mode == 1u) {
        return pcg_tpdf_dither(pos, frame);
    } else if (mode == 2u) {
        return spatial_tpdf_dither(pos);
    }
    return vec3<f32>(0.0);
}

// Bilinear fetch from `history_prev` storage buffer at continuous texel coordinates `pos_px`:
fn sample_history_bilinear(pos_px: vec2<f32>, dims: vec2<u32>) -> vec3<f32> {
    let max_coord = vec2<f32>(f32(dims.x) - 1.0, f32(dims.y) - 1.0);
    let p = clamp(pos_px - vec2<f32>(0.5), vec2<f32>(0.0), max_coord);
    let i0 = vec2<u32>(floor(p));
    let i1 = min(i0 + vec2<u32>(1u), dims - vec2<u32>(1u));
    let f = fract(p);

    let c00 = history_prev[pixel_index(i0.x, i0.y, dims.x)].xyz;
    let c10 = history_prev[pixel_index(i1.x, i0.y, dims.x)].xyz;
    let c01 = history_prev[pixel_index(i0.x, i1.y, dims.x)].xyz;
    let c11 = history_prev[pixel_index(i1.x, i1.y, dims.x)].xyz;

    let top = mix(c00, c10, f.x);
    let bot = mix(c01, c11, f.x);
    return mix(top, bot, f.y);
}

// Sub-pixel 5-tap Catmull-Rom bicubic history reconstruction (Filmic SMAA / TSR)
// Preserves razor-sharp sub-pixel detail across frames when rendering at reduced Laptop scale:
fn sample_history_catmull_rom(uv: vec2<f32>, dims: vec2<u32>) -> vec3<f32> {
    let sample_pos = uv * vec2<f32>(dims);
    let tex_pos1 = floor(sample_pos - vec2<f32>(0.5)) + vec2<f32>(0.5);
    let f = sample_pos - tex_pos1;
    let f2 = f * f;
    let f3 = f2 * f;

    let w0 = f2 - 0.5 * (f3 + f);
    let w1 = 1.5 * f3 - 2.5 * f2 + vec2<f32>(1.0);
    let w3 = 0.5 * (f3 - f2);
    let w2 = vec2<f32>(1.0) - w0 - w1 - w3;

    let w12 = w1 + w2;
    let offset12 = w2 / max(w12, vec2<f32>(1e-5));

    let tex_pos0 = tex_pos1 - vec2<f32>(1.0);
    let tex_pos3 = tex_pos1 + vec2<f32>(2.0);
    let tex_pos12 = tex_pos1 + offset12;

    let s0 = sample_history_bilinear(vec2<f32>(tex_pos12.x, tex_pos0.y), dims) * (w12.x * w0.y);
    let s1 = sample_history_bilinear(vec2<f32>(tex_pos0.x, tex_pos12.y), dims) * (w0.x * w12.y);
    let s2 = sample_history_bilinear(vec2<f32>(tex_pos12.x, tex_pos12.y), dims) * (w12.x * w12.y);
    let s3 = sample_history_bilinear(vec2<f32>(tex_pos3.x, tex_pos12.y), dims) * (w3.x * w12.y);
    let s4 = sample_history_bilinear(vec2<f32>(tex_pos12.x, tex_pos3.y), dims) * (w12.x * w3.y);

    let total_w = (w12.x * w0.y) + (w0.x * w12.y) + (w12.x * w12.y) + (w3.x * w12.y) + (w12.x * w3.y);
    return max(vec3<f32>(0.0), (s0 + s1 + s2 + s3 + s4) / max(total_w, 1e-5));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = screen_dims();
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let p_idx = pixel_index(gid.x, gid.y, dims.x);
    let raw_lit = lit_color_buf[p_idx];
    var curr_hdr = raw_lit.xyz;
    let hit_dist = raw_lit.w;

    // Atmospheric aerial perspective and warm solar scattering fog.
    let uv = (vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5)) / vec2<f32>(dims);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray_dir = normalize(
        u_cam.cam_forward_fov.xyz
        + u_cam.cam_right_aspect.xyz * (ndc.x * u_cam.cam_right_aspect.w * u_cam.cam_forward_fov.w)
        + u_cam.cam_up_frame.xyz * (ndc.y * u_cam.cam_forward_fov.w)
    );
    let sun_dir = u_cam.sun_dir_intensity.xyz;

    if (hit_dist < 1e4) {
        let day_t = clamp((sun_dir.y + 0.05) / 0.30, 0.0, 1.0);
        let fog_amount = 1.0 - exp(-max(hit_dist - 18.0, 0.0) * 0.0042);
        let sun_scatter = pow(max(dot(ray_dir, sun_dir), 0.0), 8.0) * day_t;
        let base_fog = mix(vec3<f32>(0.002, 0.0035, 0.0065), vec3<f32>(0.32, 0.44, 0.60), day_t);
        let fog_color = mix(
            base_fog,
            vec3<f32>(0.85, 0.58, 0.34),
            sun_scatter * 0.55,
        );
        curr_hdr = mix(curr_hdr, fog_color, clamp(fog_amount, 0.0, 0.78));
    }

    // Sub-pixel Halton temporal super-resolution (TSR) / DLAA.
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);

    let center_perc = hdr_to_perceptual(curr_hdr);

    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let sx = clamp(i32(gid.x) + dx, 0, i32(dims.x) - 1);
            let sy = clamp(i32(gid.y) + dy, 0, i32(dims.y) - 1);
            let s_lit = lit_color_buf[pixel_index(u32(sx), u32(sy), dims.x)];
            let s_perc = hdr_to_perceptual(s_lit.xyz);
            let s_yc = rgb_to_ycocg(s_perc);
            m1 += s_yc;
            m2 += s_yc * s_yc;
        }
    }

    let mean = m1 / 9.0;
    let stddev = sqrt(max(vec3<f32>(0.0), (m2 / 9.0) - mean * mean));
    // 1.35x stddev variance box ensures rock-solid anti-aliased temporal stability without sparkling:
    let box_min = mean - stddev * 1.35;
    let box_max = mean + stddev * 1.35;

    let motion = unpack_motion(gbuf_norm_motion[p_idx].y);
    let prev_uv = uv + motion;
    var resolved_perc = center_perc;

    let frame_idx = bitcast<u32>(u_cam.cam_up_frame.w);
    if (frame_idx > 0u && prev_uv.x >= 0.0 && prev_uv.x < 1.0 && prev_uv.y >= 0.0 && prev_uv.y < 1.0) {
        // Use 5-tap Catmull-Rom bicubic sampling for sub-pixel TSR history look-up:
        let hist_hdr = sample_history_catmull_rom(prev_uv, dims);
        let hist_perc = hdr_to_perceptual(hist_hdr);
        let clamped_ycocg = clamp(rgb_to_ycocg(hist_perc), box_min, box_max);
        let clamped_hist_perc = ycocg_to_rgb(clamped_ycocg);

        let motion_len = length(motion) * f32(dims.x);
        // Responsive temporal alpha: 0.18 when static (rapid convergence to unjittered ground truth), ramping to 0.50 under rapid motion:
        let blend_alpha = clamp(0.18 + motion_len * 0.08, 0.18, 0.50);
        resolved_perc = mix(clamped_hist_perc, center_perc, blend_alpha);
    }

    let resolved_hdr = perceptual_to_hdr(resolved_perc);
    history_curr[p_idx] = vec4<f32>(resolved_hdr, 1.0);

    // Write linear FP16 HDR output for native upscaling, tonemapping, and presentation in Pass 5:
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(resolved_hdr, 1.0));
}
