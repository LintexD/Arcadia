// Bound by `arcadia-render` Pass 2A/2B (Lighting & ReSTIR GI/DI temporal/spatial compute passes).
// Expected inputs: @group(0) engine world pools, @group(1) Pass 2 G-Buffer & reservoirs (bindings 0..6),
// @group(2) SHARC sharc_resolved (read-only, via common/sharc_query.wgsl),
// @group(3) ReSTIR DI resources (sources, alias, params).

@group(1) @binding(0) var<storage, read> gbuf_depth_mat: array<vec2<u32>>; // Rust: pass2_bgl entry 0 (gbuf_depth_mat_buf)
@group(1) @binding(1) var<storage, read> gbuf_norm_motion: array<vec2<u32>>; // Rust: pass2_bgl entry 1 (gbuf_norm_motion_buf)
@group(1) @binding(3) var<storage, read> reservoir_in: array<GiReservoir>; // Rust: pass2_bgl entry 3 (reads history in 2A, temporal scratch in 2B)
@group(1) @binding(4) var<storage, read_write> reservoir_out: array<GiReservoir>; // Rust: pass2_bgl entry 4 (writes temporal scratch in 2A, final history in 2B)
@group(1) @binding(5) var<storage, read> di_reservoir_in: array<DiReservoir>; // Rust: pass2_bgl entry 5 (reads history in 2A, temporal scratch in 2B)
@group(1) @binding(6) var<storage, read_write> di_reservoir_out: array<DiReservoir>; // Rust: pass2_bgl entry 6 (writes temporal scratch in 2A, final history in 2B)

@group(3) @binding(0) var<storage, read> emissive_sources: array<EmissiveSource>; // Rust: di_read_bgl entry 0
@group(3) @binding(1) var<storage, read> emissive_alias: array<AliasEntry>; // Rust: di_read_bgl entry 1
@group(3) @binding(2) var<uniform> u_emissive: EmissiveParams; // Rust: di_read_bgl entry 2

// Upgrade 2: Local 64-Tree Emissive Macro-Cell Importance Sampling (`01_high_leverage_upgrades_for_cs64.md`).
// Evaluates a stochastic candidate direction in `node_pool` at the 1.0m Macro-Node level (`shift = 4`)
// or 0.25m Meso-Node level (`shift = 2`) in a single register lookup to detect glowing crystals/lava:
fn probe_emissive_macro_candidate(pos_m: vec3<f32>, cand_dir: vec3<f32>, probe_dist_m: f32) -> f32 {
    let vp = (pos_m + cand_dir * probe_dist_m) * VOXELS_PER_METER;
    let g = u_cam.world_grid_dims.xyz;
    let world_max_v = vec3<f32>(g) * CHUNK_DIM_F;
    if (any(vp < vec3<f32>(0.0)) || any(vp >= world_max_v)) {
        return 0.0;
    }
    let vi = vec3<u32>(vp);
    let c = vi >> vec3<u32>(8u);
    let c_idx = c.x + c.y * g.x + c.z * g.x * g.y;
    let root_idx = chunk_roots[c_idx];
    if (root_idx == EMPTY_CHUNK_SENTINEL) {
        return 0.0;
    }
    let l = vi & vec3<u32>(255u);
    let root = node_pool[root_idx];
    let b0 = child_bit_index((l.x >> 6u) & 3u, (l.y >> 6u) & 3u, (l.z >> 6u) & 3u);
    if (!node_has_child(root.mask_low, root.mask_high, b0)) {
        return 0.0;
    }
    if ((root.child_ptr & IS_LEAF_FLAG) != 0u) {
        let m_id = (root.lod_payload >> 24u) & 0xFFu;
        let em = material_palette[m_id].emission_meta.xyz;
        return dot(em, vec3<f32>(0.333));
    }
    let l1_idx = node_child_offset(root, b0);
    let l1 = node_pool[l1_idx];
    let b1 = child_bit_index((l.x >> 4u) & 3u, (l.y >> 4u) & 3u, (l.z >> 4u) & 3u);
    if (!node_has_child(l1.mask_low, l1.mask_high, b1)) {
        return 0.0;
    }
    let m_id = (l1.lod_payload >> 24u) & 0xFFu;
    let em = material_palette[m_id].emission_meta.xyz;
    return dot(em, vec3<f32>(0.333));
}

// ----------------------------------------------------------------------------
// Pass 2A: Temporal Resampled Importance Sampling (RIS)
// Reads previous frame history from `reservoir_in` and `di_reservoir_in` (reservoir_a_buf, di_reservoir_a_buf),
// writes current frame temporal candidates to `reservoir_out` and `di_reservoir_out` (reservoir_b_buf, di_reservoir_b_buf).
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8, 1)
fn temporal_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = screen_dims();
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let p_idx = pixel_index(gid.x, gid.y, dims.x);
    let d_mat = gbuf_depth_mat[p_idx];
    let hit_dist = unpack_depth(d_mat);

    if (hit_dist >= 1e4) {
        reservoir_out[p_idx] = create_empty_reservoir();
        di_reservoir_out[p_idx] = create_empty_di_reservoir();
        return;
    }

    let ray_dir = reconstruct_ray_dir(gid.xy, dims);
    let hit_pos = u_cam.cam_pos_time.xyz + ray_dir * hit_dist;
    let n_mot = gbuf_norm_motion[p_idx];
    let normal = unpack_normal_oct(n_mot.x);
    let motion = unpack_motion(n_mot.y);
    let pixel_uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let prev_uv = pixel_uv + motion;

    let frame_idx = bitcast<u32>(u_cam.cam_up_frame.w);
    let xi = interleaved_gradient_noise(vec2<f32>(f32(gid.x), f32(gid.y)), frame_idx);

    let sun_dir = u_cam.sun_dir_intensity.xyz;
    let sun_power = u_cam.sun_dir_intensity.w;
    let n_dot_l = max(dot(normal, sun_dir), 0.0);

    // ------------------------------------------------------------------------
    // Geometric History Validation (Disocclusion-Safe Temporal Reuse via prev_cam_pos)
    // ------------------------------------------------------------------------
    var prev_valid = false;
    var prev_res: GiReservoir;
    var prev_idx = 0u;

    if (prev_uv.x >= 0.0 && prev_uv.x < 1.0 && prev_uv.y >= 0.0 && prev_uv.y < 1.0) {
        let prev_px = clamp(u32(prev_uv.x * f32(dims.x)), 0u, dims.x - 1u);
        let prev_py = clamp(u32(prev_uv.y * f32(dims.y)), 0u, dims.y - 1u);
        prev_idx = pixel_index(prev_px, prev_py, dims.x);
        prev_res = reservoir_in[prev_idx];
        let prev_m = unpack_m(prev_res.norm_m_flags);
        let prev_age = unpack_sample_age(prev_res.norm_m_flags);
        if (prev_m > 0u && prev_age <= 12u && prev_res.primary_depth < 1e4) {
            let prev_cam_dist = distance(hit_pos, u_cam.prev_cam_pos.xyz);
            let depth_tol = max(0.12, hit_dist * 0.05);
            if (abs(prev_cam_dist - prev_res.primary_depth) < depth_tol) {
                prev_valid = true;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Interleaved Checkerboard Half-Rate Mode (Category B1 Optimization)
    // When in Laptop / Low-Tap Mode (`quality_flags.w & 0xFFu <= 12u`), 50% of steady-state
    // pixels skip expensive secondary rays (diffuse GI + sun shadow + point light shadow).
    // ------------------------------------------------------------------------
    let denoiser_taps = u_cam.quality_flags.w & 0xFFu;
    let checkerboard_active = denoiser_taps <= 12u;
    let checkerboard_slot = (gid.x & 1u) ^ ((gid.y & 1u) << 1u);
    let frame_slot = frame_idx & 3u;
    let skip_secondary_rays = checkerboard_active && prev_valid && (((checkerboard_slot ^ frame_slot) & 1u) != 0u);

    // ------------------------------------------------------------------------
    // 1. Blue-Noise Jittered Soft Sun Shadow Ray & Decoupled Temporal EMA
    // ------------------------------------------------------------------------
    var sun_vis = 0.0;
    var shadow_m = 1.0;
    var accum_sun_shadow = 0.0;
    let prev_shadow_info = unpack_shadow_m(prev_res.shadow_vis_m);

    if (skip_secondary_rays) {
        accum_sun_shadow = prev_shadow_info.x;
        shadow_m = min(prev_shadow_info.y, 16.0);
    } else {
        if (n_dot_l > 0.001 && sun_power > 0.01) {
            let j1 = xi.x * 2.0 - 1.0;
            let j2 = xi.y * 2.0 - 1.0;
            let sun_up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(sun_dir.y) < 0.9);
            let s_tan = normalize(cross(sun_up, sun_dir));
            let s_bitan = cross(sun_dir, s_tan);
            let jitter_rad = u_cam.lod_settings.w;
            let jittered_sun = normalize(sun_dir + (s_tan * j1 + s_bitan * j2) * jitter_rad);
            let shadow_origin = hit_pos + normal * 0.08;
            let shadow_max_d = u_cam.ray_distances.w;
            let shadow_mode = u_cam.quality_flags.z;
            if (shadow_mode == 0u) {
                let shadow_trace = trace_world_64tree(shadow_origin, jittered_sun, shadow_max_d, 1u, true);
                if (!shadow_trace.hit) {
                    let end_p = shadow_origin + jittered_sun * shadow_max_d;
                    if (end_p.y >= 32.0 || jittered_sun.y < 0.0) {
                        sun_vis = 1.0;
                    }
                }
            } else {
                sun_vis = trace_shadow_vis(shadow_origin, jittered_sun, shadow_max_d);
            }
        }

        if (prev_valid) {
            let prev_shd_m = min(prev_shadow_info.y, 16.0);
            shadow_m = prev_shd_m + 1.0;
            accum_sun_shadow = (prev_shadow_info.x * prev_shd_m + sun_vis) / shadow_m;
        } else {
            accum_sun_shadow = sun_vis;
            shadow_m = 1.0;
        }
    }

    if (skip_secondary_rays) {
        var out_res = prev_res;
        out_res.primary_depth = hit_dist;
        out_res.shadow_vis_m = pack_shadow_m(accum_sun_shadow, shadow_m);
        let prev_norm = unpack_normal(prev_res.norm_m_flags);
        let prev_m = unpack_m(prev_res.norm_m_flags);
        let prev_age = unpack_sample_age(prev_res.norm_m_flags);
        let prev_is_sky = unpack_is_sky(prev_res.norm_m_flags);
        out_res.norm_m_flags = pack_norm_m_flags(prev_norm, prev_m, prev_age + 1u, prev_is_sky);
        reservoir_out[p_idx] = out_res;

        if (prev_valid) {
            di_reservoir_out[p_idx] = di_reservoir_in[prev_idx];
        } else {
            di_reservoir_out[p_idx] = create_empty_di_reservoir();
        }
        return;
    }

    // ------------------------------------------------------------------------
    // 2. Initial Candidate Generation: Secondary Ray + SHARC Cache Query
    // ------------------------------------------------------------------------
    let smooth_hemi_n = normalize(normal + vec3<f32>(0.0, 0.22, 0.0));
    let hemi_basis = make_basis(smooth_hemi_n);
    var bounce_dir = sample_cosine_hemisphere(xi.x, xi.y, smooth_hemi_n, hemi_basis);

    // Local 64-Tree Emissive Macro-Cell Importance Sampling:
    let cand_dir = sample_cosine_hemisphere(fract(xi.x + 0.5), fract(xi.y + 0.381966), smooth_hemi_n, hemi_basis);
    let em_near = probe_emissive_macro_candidate(hit_pos, cand_dir, 1.25);
    let em_mid = probe_emissive_macro_candidate(hit_pos, cand_dir, 3.0);
    let guided_pick = max(em_near, em_mid) > 0.25;
    if (guided_pick) {
        bounce_dir = cand_dir;
    }

    let gi_origin = hit_pos + normal * 0.10;
    var gi_hit = trace_world_64tree(gi_origin, bounce_dir, u_cam.ray_distances.z, 2u, true);
    let rb_gi_max = select(u_cam.ray_distances.z, gi_hit.t_voxels * VOXEL_SIZE, gi_hit.hit);
    let rb_count = (u_cam.point_light_count.y >> 16u) & 0xFFFFu;
    var rb_gi: RigidTraceResult;
    rb_gi.hit = false;
    if (rb_count > 0u) {
        rb_gi = trace_rigid_bodies(gi_origin, bounce_dir, rb_gi_max);
    }
    if (rb_gi.hit) {
        gi_hit.hit = true;
        gi_hit.t_voxels = rb_gi.t_voxels;
        gi_hit.hit_pos_meters = rb_gi.hit_pos_meters;
        gi_hit.normal = rb_gi.normal;
        gi_hit.mat_id = rb_gi.mat_id;
        gi_hit.lod_rgb = material_palette[rb_gi.mat_id].albedo_roughness.xyz;
    }

    var cand_pos = hit_pos + bounce_dir * 500.0;
    var cand_norm = -bounce_dir;
    var cand_is_sky = true;
    var cand_lo = vec3<f32>(0.0);

    if (gi_hit.hit) {
        cand_pos = gi_hit.hit_pos_meters;
        cand_norm = gi_hit.normal;
        cand_is_sky = false;

        let hit_m = material_palette[gi_hit.mat_id];
        let hit_albedo = mix(hit_m.albedo_roughness.xyz, gi_hit.lod_rgb, 0.55);
        let hit_emission = hit_m.emission_meta.xyz;
        let gi_dist_m = gi_hit.t_voxels * VOXEL_SIZE;

        // Outgoing radiance L_o(x_s) at secondary hit (excluding primary cosine):
        let em_atten = 1.0 / (1.0 + gi_dist_m * 0.06 + gi_dist_m * gi_dist_m * 0.025);
        let guided_weight = select(1.0, 0.55, guided_pick);
        let di_dedup = select(1.0, 0.25, u_emissive.count > 0u);
        cand_lo += hit_emission * em_atten * guided_weight * di_dedup;

        // SHARC returns cached reflected radiance at the hit (sun + sky + bounce 2..inf); misses use direct sun fallback.
        let sharc_jitter = interleaved_gradient_noise(vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(101.0, 57.0), frame_idx + 7919u) * 2.0 - vec2<f32>(1.0);
        let sharc_hit = query_sharc(gi_hit.hit_pos_meters, gi_hit.normal, u_cam.cam_pos_time.xyz, sharc_jitter);
        if (sharc_hit.w > 0.5) {
            cand_lo += sharc_hit.xyz;
        } else {
            let b_ndl = max(dot(gi_hit.normal, sun_dir), 0.0);
            var bounce_sun = vec3<f32>(0.0);
            if (b_ndl > 0.001 && sun_power > 0.01) {
                let bounce_sun_vis = trace_shadow_vis(gi_hit.hit_pos_meters + gi_hit.normal * 0.10, sun_dir, 32.0);
                if (bounce_sun_vis > 0.001) {
                    bounce_sun = vec3<f32>(1.0, 0.90, 0.75) * sun_power * b_ndl * bounce_sun_vis * 0.55;
                }
            }
            cand_lo += hit_albedo * bounce_sun;
        }

        let ao = clamp(gi_dist_m / 0.40, 0.25, 1.0);
        cand_lo *= ao;
    } else {
        // Escaped to sky:
        let end_p = gi_origin + bounce_dir * u_cam.ray_distances.z;
        if (bounce_dir.y > 0.15 && end_p.y >= 32.0) {
            let sky_escape = clamp((bounce_dir.y - 0.15) / 0.35, 0.0, 1.0);
            cand_lo += eval_sky_radiance(bounce_dir, sun_dir) * 0.45 * sky_escape;
        }
    }

    let gi_peak = max(cand_lo.x, max(cand_lo.y, cand_lo.z));
    if (gi_peak > 0.001) {
        cand_lo = cand_lo / (1.0 + gi_peak * 0.16);
    }

    // ------------------------------------------------------------------------
    // 3. Temporal Resampled Importance Sampling (RIS) for GI
    // ------------------------------------------------------------------------
    let cand_dir_to_s = bounce_dir;
    let cand_cos = max(0.0, dot(smooth_hemi_n, cand_dir_to_s));
    let cand_p_hat = luma(cand_lo) * (cand_cos / 3.14159265);
    let cand_w = select(0.0, luma(cand_lo), cand_p_hat > 1e-5);

    var w_sum = cand_w;
    var selected_pos = cand_pos;
    var selected_norm = cand_norm;
    var selected_lo = cand_lo;
    var selected_p_hat = cand_p_hat;
    var selected_age = 0u;
    var selected_is_sky = cand_is_sky;
    var m_total = 1u;

    if (prev_valid) {
        let max_m = select(12u, 8u, u_cam.target_voxel_info.w > 0.5);
        let prev_m = min(unpack_m(prev_res.norm_m_flags), max_m);
        let prev_w = unpack_w(prev_res.lo_b_w);
        let prev_lo = unpack_lo(prev_res.lo_rg, prev_res.lo_b_w);
        let prev_norm = unpack_normal(prev_res.norm_m_flags);
        let prev_age = unpack_sample_age(prev_res.norm_m_flags);
        let prev_is_sky = unpack_is_sky(prev_res.norm_m_flags);

        let to_prev = prev_res.sample_pos - hit_pos;
        let d_prev = length(to_prev);
        var prev_p_hat = 0.0;
        if (d_prev > 1e-4) {
            let dir_prev = to_prev / d_prev;
            let prev_cos = max(0.0, dot(smooth_hemi_n, dir_prev));
            prev_p_hat = luma(prev_lo) * (prev_cos / 3.14159265);
        }

        let effective_prev_m = min(prev_m, max_m - 1u);
        let prev_weight = select(0.0, prev_p_hat * prev_w * f32(effective_prev_m), prev_p_hat > 1e-5);
        w_sum += prev_weight;
        m_total = effective_prev_m + 1u;

        var seed = pcg_hash(gid.x + pcg_hash(gid.y + pcg_hash(frame_idx + 0x9E3779B9u)));
        let pick_prev = (rand_f32(&seed) * w_sum) < prev_weight;
        if (pick_prev) {
            selected_pos = prev_res.sample_pos;
            selected_norm = prev_norm;
            selected_lo = prev_lo;
            selected_p_hat = prev_p_hat;
            selected_age = prev_age + 1u;
            selected_is_sky = prev_is_sky;
        }

        // Smooth chromaticity accumulation across temporal history while preserving directional RIS winner:
        let blend_t = clamp(prev_weight / max(w_sum, 1e-5), 0.0, 1.0);
        let mixed_lo = mix(cand_lo, prev_lo, blend_t);
        let winner_luma = luma(selected_lo);
        let mixed_luma = luma(mixed_lo);
        if (mixed_luma > 1e-5 && winner_luma > 1e-5) {
            selected_lo = mixed_lo * (winner_luma / mixed_luma);
        }
    }

    let final_w = select(0.0, clamp(w_sum / (f32(m_total) * selected_p_hat), 0.0, 12.0), selected_p_hat > 1e-5 && w_sum > 0.0);
    let lo_packed = pack_lo_w(selected_lo, final_w);
    let norm_m = pack_norm_m_flags(selected_norm, m_total, selected_age, selected_is_sky);

    var out_res: GiReservoir;
    out_res.sample_pos = selected_pos;
    out_res.norm_m_flags = norm_m;
    out_res.lo_rg = lo_packed.x;
    out_res.lo_b_w = lo_packed.y;
    out_res.primary_depth = hit_dist;
    out_res.shadow_vis_m = pack_shadow_m(accum_sun_shadow, shadow_m);
    reservoir_out[p_idx] = out_res;

    // ------------------------------------------------------------------------
    // 4. ReSTIR DI: Candidate Sampling & Temporal Reuse
    // ------------------------------------------------------------------------
    var out_di = create_empty_di_reservoir();
    if (u_emissive.count > 0u && u_emissive.total_flux > 0.0) {
        var di_cand_res = create_empty_di_reservoir();
        var di_rng = pcg_hash(gid.x * 1973u + gid.y * 9277u + frame_idx * 26699u + 0xA341316Cu);

        var prev_di = create_empty_di_reservoir();
        var prev_di_idx = 0xFFFFu;
        var prev_di_seed = 0u;
        if (prev_valid) {
            prev_di = di_reservoir_in[prev_idx];
            prev_di_idx = prev_di.light_idx_seed & 0xFFFFu;
            prev_di_seed = prev_di.light_idx_seed >> 16u;
        }

        let avg_flux = max(1e-5, u_emissive.total_flux / f32(u_emissive.count));

        for (var a = 0u; a < 4u; a = a + 1u) {
            var cand_idx = 0u;
            if (a < 2u) {
                let u1 = rand_f32(&di_rng);
                let u2 = rand_f32(&di_rng);
                let col = min(u32(u1 * f32(u_emissive.count)), u_emissive.count - 1u);
                let alias_e = emissive_alias[col];
                cand_idx = select(alias_e.alias_idx, col, u2 < alias_e.prob);
            } else if (a == 2u) {
                cand_idx = pcg_hash((gid.x >> 2u) * 1973u + (gid.y >> 2u) * 9277u + frame_idx * 26699u) % u_emissive.count;
            } else {
                if (prev_valid && prev_di_idx < u_emissive.count) {
                    let offset = i32(rand_f32(&di_rng) * 33.0) - 16;
                    cand_idx = u32(clamp(i32(prev_di_idx) + offset, 0, i32(u_emissive.count) - 1));
                } else {
                    let u1 = rand_f32(&di_rng);
                    let u2 = rand_f32(&di_rng);
                    let col = min(u32(u1 * f32(u_emissive.count)), u_emissive.count - 1u);
                    let alias_e = emissive_alias[col];
                    cand_idx = select(alias_e.alias_idx, col, u2 < alias_e.prob);
                }
            }

            if (cand_idx < u_emissive.count) {
                let cand_seed = pcg_hash(di_rng + a * 7919u + 0x517CC1B7u) & 0xFFFFu;
                let em = emissive_sources[cand_idx];
                let cand_unshadowed = di_eval_unshadowed(em, cand_seed, hit_pos, normal);
                let p_hat = luma(cand_unshadowed);
                if (p_hat > 1e-5) {
                    let em_flux = max(1e-5, luma(em.intensity));
                    let q_rel = clamp(em_flux / avg_flux, 0.25, 4.0);
                    let cand_weight = p_hat / select(1.0, q_rel, a < 2u);

                    di_cand_res.w_sum += cand_weight;
                    di_cand_res.m += 1.0;
                    if (rand_f32(&di_rng) * di_cand_res.w_sum < cand_weight) {
                        di_cand_res.light_idx_seed = cand_idx | (cand_seed << 16u);
                    }
                }
            }
        }

        let cand_winner_idx = di_cand_res.light_idx_seed & 0xFFFFu;
        let cand_winner_seed = di_cand_res.light_idx_seed >> 16u;
        var cand_winner_p_hat = 0.0;
        if (cand_winner_idx < u_emissive.count) {
            cand_winner_p_hat = luma(di_eval_unshadowed(emissive_sources[cand_winner_idx], cand_winner_seed, hit_pos, normal));
        }

        var di_w_sum = di_cand_res.w_sum;
        var di_m_total = di_cand_res.m;
        var selected_di_idx_seed = di_cand_res.light_idx_seed;
        var selected_di_p_hat = cand_winner_p_hat;

        // WHY: Temporal reuse merges current frame candidates with history reservoir.
        if (prev_valid && prev_di.m > 0.0 && prev_di.w > 0.0 && prev_di_idx < u_emissive.count) {
            let prev_p_hat = luma(di_eval_unshadowed(emissive_sources[prev_di_idx], prev_di_seed, hit_pos, normal));
            if (prev_p_hat > 1e-5) {
                let prev_m_clamped = min(prev_di.m, 8.0);
                let prev_weight = prev_p_hat * prev_di.w * prev_m_clamped;
                di_w_sum += prev_weight;
                di_m_total += prev_m_clamped;
                if (rand_f32(&di_rng) * di_w_sum < prev_weight) {
                    selected_di_idx_seed = prev_di.light_idx_seed;
                    selected_di_p_hat = prev_p_hat;
                }
            }
        }

        let di_final_w = select(0.0, di_w_sum / (max(di_m_total, 1.0) * selected_di_p_hat), selected_di_p_hat > 1e-5 && di_w_sum > 0.0);
        out_di.light_idx_seed = selected_di_idx_seed;
        out_di.w_sum = di_w_sum;
        out_di.m = min(di_m_total, 8.0);
        out_di.w = clamp(di_final_w, 0.0, 2.5);
    }
    di_reservoir_out[p_idx] = out_di;
}

// ----------------------------------------------------------------------------
// Pass 2B: Spatial Reuse with Clamped Jacobian Reconnection
// Reads current frame temporal candidates from `reservoir_in` and `di_reservoir_in` (reservoir_b_buf, di_reservoir_b_buf),
// writes spatially resampled candidates to `reservoir_out` and `di_reservoir_out` (reservoir_a_buf, di_reservoir_a_buf).
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8, 1)
fn spatial_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = screen_dims();
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let p_idx = pixel_index(gid.x, gid.y, dims.x);
    let d_mat = gbuf_depth_mat[p_idx];
    let hit_dist = unpack_depth(d_mat);

    if (hit_dist >= 1e4) {
        reservoir_out[p_idx] = create_empty_reservoir();
        di_reservoir_out[p_idx] = create_empty_di_reservoir();
        return;
    }

    let ray_dir = reconstruct_ray_dir(gid.xy, dims);
    let hit_pos = u_cam.cam_pos_time.xyz + ray_dir * hit_dist;
    let normal = unpack_normal_oct(gbuf_norm_motion[p_idx].x);
    let center_res = reservoir_in[p_idx];
    let center_m = unpack_m(center_res.norm_m_flags);

    let center_di = di_reservoir_in[p_idx];
    let center_di_idx = center_di.light_idx_seed & 0xFFFFu;
    let center_di_seed = center_di.light_idx_seed >> 16u;

    if (center_m == 0u && center_di.m == 0.0) {
        reservoir_out[p_idx] = center_res;
        di_reservoir_out[p_idx] = center_di;
        return;
    }

    let smooth_hemi_n = normalize(normal + vec3<f32>(0.0, 0.22, 0.0));
    let center_lo = unpack_lo(center_res.lo_rg, center_res.lo_b_w);
    let center_w = unpack_w(center_res.lo_b_w);
    let center_to_s = center_res.sample_pos - hit_pos;
    let center_d = length(center_to_s);
    var center_dir = smooth_hemi_n;
    if (center_d > 1e-4) {
        center_dir = center_to_s / center_d;
    }
    let center_cos = max(0.0, dot(smooth_hemi_n, center_dir));
    let center_p_hat = luma(center_lo) * (center_cos / 3.14159265);

    var w_sum = select(0.0, center_p_hat * center_w * f32(center_m), center_p_hat > 1e-5);
    var accum_lo = center_lo * w_sum;
    let center_shd_info = unpack_shadow_m(center_res.shadow_vis_m);
    var shd_sum = center_shd_info.x * 2.0;
    var shd_w_sum = 2.0;
    var m_sum = center_m;
    var selected_res = center_res;
    var selected_p_hat = center_p_hat;
    var neighbor_won = false;
    var won_neighbor_dir = center_dir;
    var won_neighbor_dist = center_d;

    // Initialize DI spatial candidate state:
    var di_w_sum = 0.0;
    var di_m_sum = 0.0;
    var selected_di_idx_seed = center_di.light_idx_seed;
    var selected_di_p_hat = 0.0;
    var selected_di_idx = center_di_idx;
    var selected_di_seed = center_di_seed;

    if (u_emissive.count > 0u && center_di.m > 0.0 && center_di.w > 0.0 && center_di_idx < u_emissive.count) {
        let center_di_em = emissive_sources[center_di_idx];
        selected_di_p_hat = luma(di_eval_unshadowed(center_di_em, center_di_seed, hit_pos, normal));
        if (selected_di_p_hat > 1e-5) {
            di_w_sum = selected_di_p_hat * center_di.w * center_di.m;
            di_m_sum = center_di.m;
        }
    }

    let frame_idx = bitcast<u32>(u_cam.cam_up_frame.w);
    var seed = pcg_hash(gid.x + pcg_hash(gid.y + pcg_hash(frame_idx + 0x85EBCA6Bu)));

    let denoiser_taps = u_cam.quality_flags.w & 0xFFu;
    let num_taps = select(3u, 2u, denoiser_taps <= 12u);

    let angle = rand_f32(&seed) * 6.2831853;
    let ca = cos(angle);
    let sa = sin(angle);

    for (var i = 0u; i < num_taps; i = i + 1u) {
        var base_offset = vec2<f32>(3.0, 1.0);
        if (i == 1u) {
            base_offset = vec2<f32>(-3.5, 3.0);
        } else if (i == 2u) {
            base_offset = vec2<f32>(0.5, -5.0);
        }

        let rx = i32(round(base_offset.x * ca - base_offset.y * sa));
        let ry = i32(round(base_offset.x * sa + base_offset.y * ca));
        let qx = i32(gid.x) + rx;
        let qy = i32(gid.y) + ry;

        if (qx < 0 || qy < 0 || qx >= i32(dims.x) || qy >= i32(dims.y)) {
            continue;
        }

        let q_idx = pixel_index(u32(qx), u32(qy), dims.x);
        let q_d_mat = gbuf_depth_mat[q_idx];
        let q_dist = unpack_depth(q_d_mat);
        if (q_dist >= 1e4) {
            continue;
        }

        // Geometric early rejection on depth before fetching normal:
        if (abs(hit_dist - q_dist) / max(hit_dist, 0.001) > 0.10) {
            continue;
        }

        let q_norm = unpack_normal_oct(gbuf_norm_motion[q_idx].x);

        // Pre-reconnection bilateral normal reject:
        if (dot(normal, q_norm) < 0.80) {
            continue;
        }

        // 1. Spatial GI neighbor reuse:
        if (center_m > 0u) {
            let q_res = reservoir_in[q_idx];
            let q_shd_info = unpack_shadow_m(q_res.shadow_vis_m);
            if (q_shd_info.y > 0.0) {
                let shd_tap_w = min(q_shd_info.y, 4.0) * 0.35;
                shd_sum += q_shd_info.x * shd_tap_w;
                shd_w_sum += shd_tap_w;
            }

            let q_m = unpack_m(q_res.norm_m_flags);
            if (q_m > 0u) {
                let q_m_clamped = min(q_m, 4u);
                m_sum += q_m_clamped;

                let q_age = unpack_sample_age(q_res.norm_m_flags);
                if (q_age <= 4u) {
                    let q_lo = unpack_lo(q_res.lo_rg, q_res.lo_b_w);
                    let q_w = unpack_w(q_res.lo_b_w);
                    let q_n_s = unpack_normal(q_res.norm_m_flags);
                    let q_is_sky = unpack_is_sky(q_res.norm_m_flags);

                    var jacobian = 1.0;
                    var p_hat_target = 0.0;
                    var dir_v_to_s = vec3<f32>(0.0);
                    var dist_v_to_s = 0.0;

                    if (q_is_sky) {
                        let to_s = q_res.sample_pos - hit_pos;
                        dist_v_to_s = length(to_s);
                        if (dist_v_to_s >= 1e-4) {
                            dir_v_to_s = to_s / dist_v_to_s;
                            let cos_theta_v = max(0.0, dot(smooth_hemi_n, dir_v_to_s));
                            p_hat_target = luma(q_lo) * (cos_theta_v / 3.14159265);
                        }
                    } else {
                        let q_pos = reconstruct_world_pos(vec2<u32>(u32(qx), u32(qy)), q_dist, dims);
                        let v_new = hit_pos - q_res.sample_pos;
                        let v_old = q_pos - q_res.sample_pos;
                        let d_new = length(v_new);
                        let d_old = length(v_old);

                        if (d_new >= 0.25 && d_old >= 0.25) {
                            let hat_v_new = v_new / d_new;
                            let hat_v_old = v_old / d_old;

                            let cos_s_new = dot(q_n_s, hat_v_new);
                            let cos_s_old = dot(q_n_s, hat_v_old);

                            if (cos_s_new > 0.05 && cos_s_old > 0.05) {
                                let j_raw = (cos_s_new / cos_s_old) * ((d_old * d_old) / max(0.0625, d_new * d_new));
                                jacobian = clamp(j_raw, 0.2, 2.0);

                                dir_v_to_s = -hat_v_new;
                                dist_v_to_s = d_new;
                                let cos_theta_v = max(0.0, dot(smooth_hemi_n, dir_v_to_s));
                                p_hat_target = luma(q_lo) * (cos_theta_v / 3.14159265);
                            }
                        }
                    }

                    if (p_hat_target > 1e-5) {
                        let w_q = p_hat_target * jacobian * q_w * f32(q_m_clamped);
                        w_sum += w_q;
                        accum_lo += q_lo * w_q;

                        if ((rand_f32(&seed) * w_sum) < w_q) {
                            selected_res = q_res;
                            selected_p_hat = p_hat_target;
                            neighbor_won = true;
                            won_neighbor_dir = dir_v_to_s;
                            won_neighbor_dist = dist_v_to_s;
                        }
                    }
                }
            }
        }

        // 2. Spatial ReSTIR DI neighbor reuse:
        if (u_emissive.count > 0u) {
            let q_di = di_reservoir_in[q_idx];
            let q_di_idx = q_di.light_idx_seed & 0xFFFFu;
            let q_di_seed = q_di.light_idx_seed >> 16u;
            let q_di_m = min(q_di.m, 16.0);

            if (q_di_m > 0.0 && q_di.w > 0.0 && q_di_idx < u_emissive.count) {
                let q_em = emissive_sources[q_di_idx];
                let p_hat_target = luma(di_eval_unshadowed(q_em, q_di_seed, hit_pos, normal));
                if (p_hat_target > 1e-5) {
                    let w_q = p_hat_target * q_di.w * q_di_m;
                    di_w_sum += w_q;
                    di_m_sum += q_di_m;
                    if (rand_f32(&seed) * di_w_sum < w_q) {
                        selected_di_idx_seed = q_di.light_idx_seed;
                        selected_di_p_hat = p_hat_target;
                        selected_di_idx = q_di_idx;
                        selected_di_seed = q_di_seed;
                    }
                }
            }
        }
    }

    // Finalize GI reservoir:
    if (center_m > 0u) {
        if (neighbor_won && !unpack_is_sky(selected_res.norm_m_flags)) {
            let vis_origin = hit_pos + normal * 0.08;
            let shadow_d = max(0.05, won_neighbor_dist - 0.12);
            let vis = trace_shadow_vis(vis_origin, won_neighbor_dir, shadow_d);
            if (vis < 0.1) {
                selected_res = center_res;
                selected_p_hat = center_p_hat;
                w_sum = select(0.0, center_p_hat * center_w * f32(center_m), center_p_hat > 1e-5);
                accum_lo = center_lo * w_sum;
                m_sum = center_m;
            }
        }

        let max_m = select(12u, 8u, u_cam.target_voxel_info.w > 0.5);
        let out_m = min(m_sum, max_m);
        let final_w = select(0.0, clamp(w_sum / (f32(max(m_sum, 1u)) * selected_p_hat), 0.0, 12.0), selected_p_hat > 1e-5 && w_sum > 0.0);
        var sel_lo = unpack_lo(selected_res.lo_rg, selected_res.lo_b_w);
        let sel_luma = luma(sel_lo);
        let avg_lo = accum_lo / max(w_sum, 1e-5);
        let avg_luma = luma(avg_lo);
        if (avg_luma > 1e-5 && sel_luma > 1e-5) {
            sel_lo = avg_lo * (sel_luma / avg_luma);
        }
        let lo_packed = pack_lo_w(sel_lo, final_w);

        let sel_norm = unpack_normal(selected_res.norm_m_flags);
        let sel_age = unpack_sample_age(selected_res.norm_m_flags) + select(0u, 2u, neighbor_won);
        let sel_is_sky = unpack_is_sky(selected_res.norm_m_flags);
        let out_flags = pack_norm_m_flags(sel_norm, out_m, sel_age, sel_is_sky);

        var out_res = selected_res;
        out_res.norm_m_flags = out_flags;
        out_res.lo_b_w = lo_packed.y;
        out_res.shadow_vis_m = pack_shadow_m(shd_sum / shd_w_sum, center_shd_info.y);
        out_res.primary_depth = hit_dist;
        reservoir_out[p_idx] = out_res;
    } else {
        reservoir_out[p_idx] = center_res;
    }

    // ------------------------------------------------------------------------
    // Spatial ReSTIR DI: Radius-Trimmed Shadow Ray & Post-Visibility Zeroing
    // ------------------------------------------------------------------------
    var di_final_w = 0.0;
    if (selected_di_idx < u_emissive.count && selected_di_p_hat > 1e-5 && di_w_sum > 0.0) {
        let kept_center_di = (selected_di_idx_seed == center_di.light_idx_seed) && (center_di.w > 0.0);
        let checkerboard_active = denoiser_taps <= 12u;
        let checkerboard_slot = (gid.x & 1u) ^ ((gid.y & 1u) << 1u);
        let frame_slot = frame_idx & 3u;
        let skip_di_vis_ray = checkerboard_active && kept_center_di && (((checkerboard_slot ^ frame_slot) & 1u) != 0u);

        var vis = 1.0;
        if (!skip_di_vis_ray) {
            let em = emissive_sources[selected_di_idx];
            let un = di_unpack_normal_and_eta(em.normal_packed);
            let em_pos = di_emitter_surface_pos(em.pos, un.normal, em.radius, un.eta, selected_di_seed);
            let d_init = length(em_pos - hit_pos);
            let normal_offset = min(0.08, d_init * 0.4);
            let shadow_origin = hit_pos + normal * normal_offset;
            let to_em = em_pos - shadow_origin;
            let dist_to_em = length(to_em);
            let shadow_dir = select(vec3<f32>(0.0, 1.0, 0.0), to_em / max(dist_to_em, 1e-4), dist_to_em > 1e-4);
            let shadow_max_d = max(0.02, dist_to_em - max(em.radius * 0.35, 0.08));
            vis = trace_shadow_vis_di(shadow_origin, shadow_dir, shadow_max_d);
        }

        if (vis < 0.1) {
            // Post-visibility zeroing eliminates dark-halo boiling around occluded lights:
            di_w_sum = 0.0;
            di_final_w = 0.0;
        } else {
            di_final_w = clamp(di_w_sum / (max(di_m_sum, 1.0) * selected_di_p_hat), 0.0, 2.5) * vis;
        }
    }

    var out_di: DiReservoir;
    out_di.light_idx_seed = selected_di_idx_seed;
    out_di.w_sum = di_w_sum;
    out_di.m = min(di_m_sum, 12.0);
    out_di.w = di_final_w;
    di_reservoir_out[p_idx] = out_di;
}
