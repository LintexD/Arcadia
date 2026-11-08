// Bound by `arcadia-render` Pass 5 (Fullscreen blit and Catmull-Rom/RCAS spatial upscaler).
// Expected inputs: @group(0) blit bind group (bindings 0, 1), @group(1) engine uniforms (binding 0).

@group(0) @binding(0) var render_tex: texture_2d<f32>; // Rust: blit_bgl entry 0 (out_tex_view)
@group(0) @binding(1) var render_samp: sampler; // Rust: blit_bgl entry 1 (blit_sampler)

struct CameraUniforms {
    cam_pos_time: vec4<f32>,
    cam_forward_fov: vec4<f32>,
    cam_right_aspect: vec4<f32>,
    cam_up_frame: vec4<f32>,
    // Previous frame unjittered view-projection matrix (infinite Reverse-Z: near=1.0, far=0.0)
    prev_view_proj: mat4x4<f32>,
    // Current frame unjittered view-projection matrix (infinite Reverse-Z: near=1.0, far=0.0)
    curr_view_proj: mat4x4<f32>,
    world_grid_dims: vec4<u32>,
    sun_dir_intensity: vec4<f32>,
    jitter_brush_dt: vec4<f32>,
    target_voxel_info: vec4<f32>,
    point_light_count: vec4<u32>,
    point_light_0: vec4<f32>,
    point_light_1: vec4<f32>,
    point_light_2: vec4<f32>,
    point_light_3: vec4<f32>,
    point_light_4: vec4<f32>,
    point_light_5: vec4<f32>,
    point_light_6: vec4<f32>,
    point_light_7: vec4<f32>,
    ray_steps: vec4<u32>,
    ray_distances: vec4<f32>,
    lod_settings: vec4<f32>,
    quality_flags: vec4<u32>,
};

@group(1) @binding(0) var<uniform> u_cam: CameraUniforms; // Rust: group0_bgl entry 0 (camera_uniform_buf)

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
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

fn luma_rcas(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

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

// Mode 1: Monochrome Triangular Probability Density Function (TPDF) via 2D PCG hash.
// Full 2.0 LSB base (±1.0 / 255.0) eliminates 100% of 8-bit quantization banding
// without chromatic color scintillation.
fn pcg_tpdf_dither(pos: vec2<u32>, frame: u32) -> vec3<f32> {
    let p = pos + vec2<u32>(frame * 19u, frame * 37u);
    let h = hash_pcg2d(p);

    let u1 = f32(h.x & 0xFFFFu) / 65535.0;
    let u2 = f32(h.x >> 16u) / 65535.0;

    let tpdf = (u1 - u2) / 255.0;
    return vec3<f32>(tpdf);
}

// Mode 2: Spatial Integer Bit-Mix TPDF Dither (Static in screen space, zero temporal flicker).
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

// Mode 3: Interleaved Gradient Noise (IGN) Dither (Jorge Jimenez 2014 / Call of Duty / UE5).
// Blue-noise spatial spectrum concentrates high-frequency energy strictly at the Nyquist limit,
// giving seamless banding elimination with zero clustering and zero chromatic noise.
fn ign_dither(pos: vec2<u32>, frame: u32) -> vec3<f32> {
    let p = vec2<f32>(pos) + f32(frame % 64u) * vec2<f32>(5.588238, 5.588238);
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    let ign = fract(magic.z * fract(dot(p, magic.xy)));
    let dither = (ign - 0.5) / 255.0;
    return vec3<f32>(dither);
}

fn eval_dither(pos: vec2<u32>, frame: u32, mode: u32) -> vec3<f32> {
    if (mode == 1u) {
        return pcg_tpdf_dither(pos, frame);
    } else if (mode == 2u) {
        return spatial_tpdf_dither(pos);
    } else if (mode == 3u) {
        return ign_dither(pos, frame);
    }
    return vec3<f32>(0.0);
}

// 5-tap Catmull-Rom bicubic spatial upscaling + FSR 1.0 RCAS (Robust Contrast-Adaptive Sharpening)
// with zero-ringing min/max neighborhood clamping:
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex_dims = vec2<f32>(textureDimensions(render_tex));
    let inv_dims = 1.0 / tex_dims;

    // 1. 5-Tap Catmull-Rom Bicubic Upscaling Kernel:
    let sample_pos = in.uv * tex_dims;
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

    let uv0 = (tex_pos1 - vec2<f32>(1.0)) * inv_dims;
    let uv3 = (tex_pos1 + vec2<f32>(2.0)) * inv_dims;
    let uv12 = (tex_pos1 + offset12) * inv_dims;

    let s_top = textureSampleLevel(render_tex, render_samp, vec2<f32>(uv12.x, uv0.y), 0.0).xyz;
    let s_left = textureSampleLevel(render_tex, render_samp, vec2<f32>(uv0.x, uv12.y), 0.0).xyz;
    let s_center = textureSampleLevel(render_tex, render_samp, vec2<f32>(uv12.x, uv12.y), 0.0).xyz;
    let s_right = textureSampleLevel(render_tex, render_samp, vec2<f32>(uv3.x, uv12.y), 0.0).xyz;
    let s_bot = textureSampleLevel(render_tex, render_samp, vec2<f32>(uv12.x, uv3.y), 0.0).xyz;

    let cw0 = w12.x * w0.y;
    let cw1 = w0.x * w12.y;
    let cw2 = w12.x * w12.y;
    let cw3 = w3.x * w12.y;
    let cw4 = w12.x * w3.y;
    let total_w = cw0 + cw1 + cw2 + cw3 + cw4;

    var bicubic = (s_top * cw0 + s_left * cw1 + s_center * cw2 + s_right * cw3 + s_bot * cw4) / max(total_w, 1e-5);

    // 2. FSR RCAS (Robust Contrast-Adaptive Sharpening) + Anti-Ringing Clamp:
    // Fetch immediate 4-tap cross neighborhood around `in.uv`:
    let n = textureSampleLevel(render_tex, render_samp, in.uv + vec2<f32>(0.0, -inv_dims.y), 0.0).xyz;
    let s = textureSampleLevel(render_tex, render_samp, in.uv + vec2<f32>(0.0, inv_dims.y), 0.0).xyz;
    let w = textureSampleLevel(render_tex, render_samp, in.uv + vec2<f32>(-inv_dims.x, 0.0), 0.0).xyz;
    let e = textureSampleLevel(render_tex, render_samp, in.uv + vec2<f32>(inv_dims.x, 0.0), 0.0).xyz;
    let c = textureSampleLevel(render_tex, render_samp, in.uv, 0.0).xyz;

    let min_4 = min(min(n, s), min(w, e));
    let max_4 = max(max(n, s), max(w, e));
    let min_5 = min(min_4, c);
    let max_5 = max(max_4, c);

    // Clamp bicubic output strictly inside the 5-tap neighborhood to prevent overshoot halos:
    bicubic = clamp(bicubic, min_5, max_5);

    // Compute local contrast-adaptive RCAS lobe weight:
    let l_min = luma_rcas(min_4);
    let l_max = luma_rcas(max_4);
    let contrast = l_max - l_min;
    // Contrast-adaptive RCAS lobe weight.
    let rcas_lobe = -0.14 * clamp(1.0 - contrast * 1.6, 0.10, 1.0);
    let rcas_color = clamp(
        (bicubic + (n + s + w + e) * rcas_lobe) / (1.0 + 4.0 * rcas_lobe),
        min_5,
        max_5,
    );

    // 3. Selectable ACES Filmic or Polynomial AgX Display Transform (bit 27 in quality_flags.w):
    let tonemapper_mode = (u_cam.quality_flags.w >> 27u) & 1u;
    var out_col = select(aces_tonemap(rcas_color * 0.85), agx_tonemap(rcas_color * 0.85), tonemapper_mode != 0u);

    // 4. Crisp Native-Resolution Center Crosshair (drawn in display LDR using screen-space UV & derivatives):
    let d_uv = fwidth(in.uv);
    let px_dist = abs(in.uv - vec2<f32>(0.5, 0.5)) / max(d_uv, vec2<f32>(1e-5));
    let ix = i32(round(px_dist.x));
    let iy = i32(round(px_dist.y));
    if ((ix == 0 && iy >= 2 && iy <= 5) || (iy == 0 && ix >= 2 && ix <= 5)) {
        out_col = mix(out_col, vec3<f32>(0.95, 0.98, 1.0), 0.78);
    }

    // 5. Native Swapchain Presentation with Sub-LSB Dither (Mode 1: PCG Temporal, Mode 2: Spatial Bit-Mix, Mode 3: IGN):
    let dither_mode = (u_cam.quality_flags.w >> 24u) & 0x7u;
    if (dither_mode != 0u) {
        let frame_idx = bitcast<u32>(u_cam.cam_up_frame.w);
        let dither = eval_dither(vec2<u32>(in.pos.xy), frame_idx, dither_mode);
        out_col = clamp(out_col + dither, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    return vec4<f32>(out_col, 1.0);
}
