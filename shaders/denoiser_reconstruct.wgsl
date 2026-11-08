// Bound by `arcadia-render` Pass 3 (Wavelet reconstruct denoiser & PBR shading compute pass).
// Expected inputs: @group(0) engine world pools, @group(1) Pass 3 G-Buffer, reservoir, lit output (bindings 0..2, 4..6),
// @group(2) ReSTIR DI resources (sources, params).

@group(1) @binding(0) var<storage, read> gbuf_depth_mat: array<vec2<u32>>; // Rust: pass3_bgl entry 0 (gbuf_depth_mat_buf)
@group(1) @binding(1) var<storage, read> gbuf_norm_motion: array<vec2<u32>>; // Rust: pass3_bgl entry 1 (gbuf_norm_motion_buf)
@group(1) @binding(2) var<storage, read> gbuf_albedo_ao: array<u32>; // Rust: pass3_bgl entry 2 (gbuf_albedo_ao_buf)
@group(1) @binding(4) var<storage, read> reservoir_curr: array<GiReservoir>; // Rust: pass3_bgl entry 3 (reservoir_curr_buf)
@group(1) @binding(5) var<storage, read_write> lit_color_buf: array<vec4<f32>>; // Rust: pass3_bgl entry 4 (lit_color_buf)
@group(1) @binding(6) var<storage, read> di_reservoir_curr: array<DiReservoir>; // Rust: pass3_bgl entry 5 (di_reservoir_a_buf)

@group(2) @binding(0) var<storage, read> emissive_sources: array<EmissiveSource>; // Rust: di_pass3_bgl entry 0
@group(2) @binding(1) var<uniform> u_emissive: EmissiveParams; // Rust: di_pass3_bgl entry 1

fn eval_reservoir_di(res: DiReservoir, x_v: vec3<f32>, n_v: vec3<f32>) -> vec3<f32> {
    if (res.m <= 0.0 || res.w <= 0.0 || u_emissive.count == 0u) {
        return vec3<f32>(0.0);
    }
    let light_idx = res.light_idx_seed & 0xFFFFu;
    let seed = res.light_idx_seed >> 16u;
    if (light_idx >= u_emissive.count) {
        return vec3<f32>(0.0);
    }
    let em = emissive_sources[light_idx];
    let unshadowed = di_eval_unshadowed(em, seed, x_v, n_v);
    return unshadowed * res.w;
}

// 16-tap wide Vogel-disk filter kernel spanning up to 9 pixels in radius
// (uses a `switch` statement to avoid naga SPIR-V local let-array dynamic indexing bug):
fn get_denoiser_vogel_tap(idx: u32) -> vec2<f32> {
    switch (idx & 15u) {
        case 0u:  { return vec2<f32>(1.5, 0.0); }
        case 1u:  { return vec2<f32>(-1.8, 1.6); }
        case 2u:  { return vec2<f32>(0.6, -2.8); }
        case 3u:  { return vec2<f32>(2.8, 2.4); }
        case 4u:  { return vec2<f32>(-4.0, -0.8); }
        case 5u:  { return vec2<f32>(3.2, -3.5); }
        case 6u:  { return vec2<f32>(-0.6, 5.2); }
        case 7u:  { return vec2<f32>(-3.4, -4.8); }
        case 8u:  { return vec2<f32>(6.2, 1.4); }
        case 9u:  { return vec2<f32>(1.2, 8.2); }
        case 10u: { return vec2<f32>(6.4, 6.6); }
        case 11u: { return vec2<f32>(-9.5, 2.4); }
        case 12u: { return vec2<f32>(4.2, -9.8); }
        case 13u: { return vec2<f32>(-5.8, 10.4); }
        case 14u: { return vec2<f32>(12.2, -3.6); }
        default:  { return vec2<f32>(-11.4, -7.6); }
    }
}

// Sky dome helper for background pixels across Day, Sunset, and Night:
fn eval_sky_bg(dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_elev = sun_dir.y;
    let day_t = clamp((sun_elev + 0.05) / 0.30, 0.0, 1.0);
    let twilight_t = clamp(1.0 - abs(sun_elev - 0.02) / 0.22, 0.0, 1.0);

    let zenith = clamp(dir.y, 0.0, 1.0);
    let horizon = pow(1.0 - zenith, 3.2);

    let day_zenith = vec3<f32>(0.08, 0.22, 0.55);
    let day_horizon = vec3<f32>(0.38, 0.50, 0.68);
    let night_zenith = vec3<f32>(0.005, 0.008, 0.020);
    let night_horizon = vec3<f32>(0.012, 0.018, 0.036);

    var sky_z = mix(night_zenith, day_zenith, day_t);
    var sky_h = mix(night_horizon, day_horizon, day_t);
    sky_h = mix(sky_h, vec3<f32>(0.72, 0.28, 0.10), twilight_t * 0.80);

    var sky = mix(sky_z, sky_h, horizon);

    if (dir.y < 0.0) {
        let ground_mist = mix(vec3<f32>(0.008, 0.012, 0.025), vec3<f32>(0.16, 0.22, 0.28), day_t);
        sky = mix(sky_h, ground_mist, clamp(-dir.y * 4.0, 0.0, 1.0));
    }

    let sun_cos = max(dot(dir, sun_dir), 0.0);
    if (sun_elev > -0.15) {
        let sun_vis_t = clamp((sun_elev + 0.15) / 0.25, 0.0, 1.0);
        sky += vec3<f32>(0.95, 0.55, 0.25) * pow(sun_cos, 18.0) * 0.45 * sun_vis_t;
        if (sun_cos > 0.9993) {
            sky += vec3<f32>(6.0, 4.5, 2.8) * sun_vis_t;
        }
    }
    let moon_cos = max(dot(dir, -sun_dir), 0.0);
    if (sun_elev < 0.0) {
        sky += vec3<f32>(0.06, 0.08, 0.12) * pow(moon_cos, 48.0) * 0.18;
        if (moon_cos > 0.9994) {
            sky += vec3<f32>(1.6, 1.7, 1.8);
        }
    }
    return sky;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = screen_dims();
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let p_idx = pixel_index(gid.x, gid.y, dims.x);
    let d_mat = gbuf_depth_mat[p_idx];
    let hit_dist = unpack_depth(d_mat);
    let sun_dir = u_cam.sun_dir_intensity.xyz;
    let cam_underwater = (u_cam.point_light_count.y & 0xFFFFu) != 0u;

    // Sky background (with smooth turquoise underwater dome when camera is submerged):
    if (hit_dist >= 1e4) {
        let ray_d = reconstruct_ray_dir(gid.xy, dims);
        var sky = eval_sky_bg(ray_d, sun_dir);
        if (cam_underwater) {
            let up_trans = clamp(ray_d.y * 0.65 + 0.15, 0.0, 0.72);
            let deep_water = vec3<f32>(0.015, 0.16, 0.26);
            sky = mix(deep_water, sky * vec3<f32>(0.35, 0.82, 0.95), up_trans);
        }
        lit_color_buf[p_idx] = vec4<f32>(sky, 1e5);
        return;
    }

    let ray_dir = reconstruct_ray_dir(gid.xy, dims);
    let center_pos = u_cam.cam_pos_time.xyz + ray_dir * hit_dist;
    let center_norm = unpack_normal_oct(gbuf_norm_motion[p_idx].x);
    let mat_id = unpack_mat_id(d_mat);
    let mat = material_palette[mat_id];

    let raw_albedo_ao = gbuf_albedo_ao[p_idx];
    let albedo = unpack_albedo(raw_albedo_ao);
    let roughness = select(mat.albedo_roughness.w, 0.02, mat_id == MAT_MIRROR_CHROME);
    // Upgrade 3: Unpack Nanosecond Bitwise Contact AO from `gbuf_albedo_ao[p_idx]`
    // with smooth distance fadeout to completely eliminate distant Moiré interference on curved voxel surfaces:
    let raw_ao = clamp(unpack_contact_ao(raw_albedo_ao), 0.35, 1.0);
    let raw_ao_dist = bitcast<f32>(u_cam.point_light_count.w);
    let ao_max_dist = select(raw_ao_dist, 24.0, raw_ao_dist < 1.0);
    let ao_min_dist = ao_max_dist * 0.5;
    let ao_weight = 1.0 - smoothstep(ao_min_dist, ao_max_dist, hit_dist);
    let contact_ao = mix(1.0, raw_ao, ao_weight);

    // ------------------------------------------------------------------------
    // Two-Regime Voxel Edge-Stopping Wavelet Reconstruction Filter
    // (Zero ray-marching inside Pass 3 — 100% lag-free!)
    // ------------------------------------------------------------------------
    let center_res = reservoir_curr[p_idx];
    let center_gi = eval_reservoir_gi(center_res, center_pos, center_norm);
    let center_shd = unpack_shadow_m(center_res.shadow_vis_m).x;
    let center_di_res = di_reservoir_curr[p_idx];
    let center_di = eval_reservoir_di(center_di_res, center_pos, center_norm);

    let dist_scale = max(0.85, hit_dist * 0.08);
    let plane_falloff = 14.0 / dist_scale;
    let planar_dist_falloff = 0.85 / dist_scale;
    let stair_radius = 0.34 * dist_scale;
    let stair_r2 = stair_radius * stair_radius;
    let plane_thresh = 0.35 * dist_scale;

    let quad_id = (gid.x & 1u) + ((gid.y & 1u) << 1u);
    let cs = select(
        select(vec2<f32>(1.0, 0.0), vec2<f32>(0.70710678, 0.70710678), quad_id == 1u),
        select(vec2<f32>(0.0, 1.0), vec2<f32>(-0.70710678, 0.70710678), quad_id == 3u),
        quad_id >= 2u
    );
    let ca = cs.x;
    let sa = cs.y;

    var sum_gi = center_gi;
    var sum_shadow = center_shd;
    var sum_di = center_di;
    var total_w = 1.0;

    let max_taps = clamp(u_cam.quality_flags.w & 0xFFu, 4u, 16u);
    for (var i = 0u; i < max_taps; i = i + 1u) {
        let tap = get_denoiser_vogel_tap(i);
        let rx = i32(round(tap.x * ca - tap.y * sa));
        let ry = i32(round(tap.x * sa + tap.y * ca));
        let qx = i32(gid.x) + rx;
        let qy = i32(gid.y) + ry;
        if (qx >= 0 && qy >= 0 && qx < i32(dims.x) && qy < i32(dims.y)) {
            let q_idx = pixel_index(u32(qx), u32(qy), dims.x);
            let q_d_mat = gbuf_depth_mat[q_idx];
            let q_dist = unpack_depth(q_d_mat);
            if (q_dist < 1e4) {
                let q_pos = reconstruct_world_pos(vec2<u32>(u32(qx), u32(qy)), q_dist, dims);
                let d_pos = q_pos - center_pos;
                let plane_dist = abs(dot(d_pos, center_norm));
                let d2 = dot(d_pos, d_pos);

                // Geometric early rejection: skip reading gbuf_norm_motion if tap is far off-plane
                if (plane_dist > plane_thresh && d2 > stair_r2) {
                    continue;
                }

                let q_norm = unpack_normal_oct(gbuf_norm_motion[q_idx].x);
                let n_cos = dot(center_norm, q_norm);
                if (n_cos < -0.05) {
                    continue;
                }

                let world_dist = sqrt(d2);
                let n_clamp = max(0.0, n_cos);
                let n2 = n_clamp * n_clamp;
                let n6 = n2 * n2 * n2;

                let w_planar = n6 * exp(-(plane_dist * plane_falloff + world_dist * planar_dist_falloff));

                var w_stair = 0.0;
                if (d2 < stair_r2) {
                    let t_s = 1.0 - (world_dist / stair_radius);
                    w_stair = t_s * t_s * 0.75;
                }

                let w = max(w_planar, w_stair);
                if (w > 0.01) {
                    let q_res = reservoir_curr[q_idx];
                    let q_gi = eval_reservoir_gi(q_res, q_pos, q_norm);
                    let q_shd = unpack_shadow_m(q_res.shadow_vis_m).x;
                    let max_allowed = (sum_gi / total_w) * 2.2 + vec3<f32>(0.35);
                    let clamped_q_gi = min(q_gi, max_allowed);
                    sum_gi += clamped_q_gi * w;
                    sum_shadow += q_shd * w;

                    let q_di_res = di_reservoir_curr[q_idx];
                    let q_di = eval_reservoir_di(q_di_res, q_pos, q_norm);
                    let max_allowed_di = (sum_di / total_w) * 2.2 + vec3<f32>(0.35);
                    let clamped_q_di = min(q_di, max_allowed_di);
                    sum_di += clamped_q_di * w;

                    total_w += w;
                }
            }
        }
    }

    let denoised_gi = sum_gi / total_w;
    let denoised_shadow = sum_shadow / total_w;
    let denoised_di = sum_di / total_w;

    // ------------------------------------------------------------------------
    // Physically Based Shading (Direct Sunlight + Octree-Cached GI + Emissive DI + Mirror Chrome)
    // ------------------------------------------------------------------------
    let view_dir = normalize(u_cam.cam_pos_time.xyz - center_pos);
    let n_dot_l = max(dot(center_norm, sun_dir), 0.0);
    let sun_power = u_cam.sun_dir_intensity.w;
    let sun_color = mix(vec3<f32>(1.0, 0.48, 0.20), vec3<f32>(1.0, 0.92, 0.78), clamp(sun_dir.y * 2.5, 0.0, 1.0));
    let sun_radiance = sun_color * sun_power;

    let day_t = clamp((sun_dir.y + 0.05) / 0.30, 0.0, 1.0);

    // Direct and ambient moonlight: keeps nights beautifully visible (never pitch black)
    let moon_power = clamp((-sun_dir.y + 0.05) / 0.35, 0.0, 1.0) * 0.45;
    let moon_color = vec3<f32>(0.70, 0.82, 1.0);
    let moon_ndl = max(dot(center_norm, -sun_dir), 0.0);
    let moon_direct = moon_color * moon_power * moon_ndl * denoised_shadow;

    // Shadowed Analytical Point Lights (evaluated directly on visible surface):
    var point_light_direct = vec3<f32>(0.0);
    let pl_count = min(u_cam.point_light_count.x, 8u);
    if (pl_count > 0u) {
        for (var i = 0u; i < pl_count; i = i + 1u) {
            let pl = get_point_light(i);
            let to_pl = pl.xyz - center_pos;
            let d = length(to_pl);
            if (d > 0.05 && d < 28.0) {
                let l_d = to_pl / d;
                let ndl = max(dot(center_norm, l_d), 0.0);
                let falloff = clamp(1.0 - (d / 28.0), 0.0, 1.0);
                let w = (pl.w * ndl * falloff * falloff) / (1.0 + d * d * 0.14);
                if (w > 0.002) {
                    let shadow_d = max(0.05, d - pl.w * 1.15);
                    let pl_vis = trace_shadow_vis(center_pos + center_norm * 0.06, l_d, shadow_d);
                    point_light_direct += vec3<f32>(1.0, 0.88, 0.65) * w * 2.2 * pl_vis;
                }
            }
        }
    }

    // Specular highlight (switches between sun in daytime and moon at night):
    let active_light_dir = select(-sun_dir, sun_dir, day_t > 0.5);
    let active_light_rad = select(moon_color * moon_power, sun_radiance, day_t > 0.5);
    let half_vec = normalize(view_dir + active_light_dir);
    let n_dot_h = max(dot(center_norm, half_vec), 0.0);
    let spec_power = mix(128.0, 8.0, roughness * roughness);
    let spec_strength = (1.0 - roughness * 0.85) * 0.35;
    let active_ndl = select(moon_ndl, n_dot_l, day_t > 0.5);
    let specular = active_light_rad * pow(n_dot_h, spec_power) * spec_strength * denoised_shadow * active_ndl;

    // WHY: Pass 2 ReSTIR GI + SHARC integrate sky-escaping rays; synthetic day_sky_amb * denoised_shadow removed to ungate shadows.
    let ambient_floor = bitcast<f32>(u_cam.point_light_count.z);
    let min_ambient = vec3<f32>(0.85, 0.90, 1.0) * ambient_floor * contact_ao;

    var caustic_rgb = vec3<f32>(1.0);
    if (cam_underwater) {
        let water_y = max(11.25, u_cam.cam_pos_time.y + 0.35);
        caustic_rgb = eval_water_caustics(center_pos, water_y, center_norm, sun_dir, u_cam.cam_pos_time.w);
    }

    let facet_shade_raw = 0.88 + 0.12 * clamp(center_norm.y * 0.6 + abs(center_norm.x) * 0.25, -0.5, 1.0);
    let facet_shade = mix(facet_shade_raw, 1.0, smoothstep(ao_min_dist, ao_max_dist + 8.0, hit_dist));
    let diffuse_light = sun_radiance * n_dot_l * denoised_shadow * caustic_rgb
        + moon_direct
        + point_light_direct
        + denoised_di * 1.15 * facet_shade * contact_ao
        + denoised_gi * 1.35 * facet_shade * contact_ao
        + min_ambient;

    var color = albedo * diffuse_light + specular + mat.emission_meta.xyz;
    if (mat_id == MAT_MIRROR_CHROME) {
        // Metallic chrome mirror directly presents the raymarched scene reflection from Pass 1:
        // No Blinn-Phong specular glints, completely preventing white dots on voxel seams:
        color = albedo * 0.98;
    }
    if (cam_underwater) {
        let water_fog = 1.0 - exp(-hit_dist * 0.065);
        let deep_water = vec3<f32>(0.015, 0.18, 0.28) * max(day_t, 0.12);
        color = mix(color, deep_water, clamp(water_fog, 0.0, 0.88));
    }

    // ------------------------------------------------------------------------
    // Interactive 3D Sculpting/Destruction Brush Ring on the targeted surface
    // ------------------------------------------------------------------------
    if (u_cam.target_voxel_info.w > 0.5) {
        let brush_center = u_cam.target_voxel_info.xyz;
        let brush_r = u_cam.jitter_brush_dt.z;
        let d_brush = distance(center_pos, brush_center);
        let ring_thickness = 0.08;
        if (abs(d_brush - brush_r) < ring_thickness) {
            let pulse = 0.65 + 0.35 * sin(u_cam.cam_pos_time.w * 6.0);
            color = mix(color, vec3<f32>(0.25, 0.92, 1.0) * 2.2, pulse * 0.75);
        }
    }

    lit_color_buf[p_idx] = vec4<f32>(color, hit_dist);
}
