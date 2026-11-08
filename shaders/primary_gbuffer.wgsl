// Bound by `arcadia-render` Pass 1 (Primary G-Buffer & Debris compute passes).
// Expected inputs: @group(0) engine world pools, @group(1) Pass 1 G-Buffer RW storage buffers (bindings 0..2).

// GPU-SYNC: Compact G-Buffer (20 B/px)
// Slot 0: depth_f32 (u32), material_id (u16), flags (u16)
@group(1) @binding(0) var<storage, read_write> gbuf_depth_mat: array<vec2<u32>>; // Rust: pass1_bgl entry 0 (gbuf_depth_mat_buf)
// Slot 1: octahedral_normal (pack2x16snorm), motion_vector (pack2x16float)
@group(1) @binding(1) var<storage, read_write> gbuf_norm_motion: array<vec2<u32>>; // Rust: pass1_bgl entry 1 (gbuf_norm_motion_buf)
// Slot 2: albedo RGB (u24 / R8G8B8), contact AO (u8)
@group(1) @binding(2) var<storage, read_write> gbuf_albedo_ao: array<u32>; // Rust: pass1_bgl entry 2 (gbuf_albedo_ao_buf)

// Fast sky dome helper for mirror & water reflections across Day, Sunset, and Night:
fn eval_reflection_sky(dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
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
    sky_h = mix(sky_h, vec3<f32>(0.70, 0.28, 0.10), twilight_t * 0.78);

    var sky = mix(sky_z, sky_h, horizon);
    if (dir.y < 0.0) {
        let ground = mix(vec3<f32>(0.02, 0.03, 0.04), vec3<f32>(0.12, 0.15, 0.18), day_t);
        sky = mix(sky_h, ground, clamp(-dir.y * 3.0, 0.0, 1.0));
    }
    let sun_cos = max(dot(dir, sun_dir), 0.0);
    if (sun_elev > -0.12) {
        sky += vec3<f32>(0.90, 0.55, 0.25) * pow(sun_cos, 16.0) * 0.45 * max(day_t, twilight_t);
    }
    return sky;
}

// Entry Point 1A: Simulate 16,384 flying 6.25cm voxel debris cubes with 64-tree collisions
@compute @workgroup_size(64, 1, 1)
fn update_debris(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= 16384u) {
        return;
    }

    var p = debris_pool[idx];
    let life = p.pos_life.w;
    if (life <= 0.0) {
        return;
    }

    let dt = clamp(u_cam.jitter_brush_dt.w, 0.001, 0.05);
    let new_life = life - dt;
    if (new_life <= 0.0) {
        p.pos_life.w = 0.0;
        debris_pool[idx] = p;
        return;
    }

    var pos = p.pos_life.xyz;
    var vel = p.vel_color.xyz;

    // Gravity acceleration:
    vel.y = max(vel.y - 18.0 * dt, -35.0);
    let next_pos = pos + vel * dt;

    // Collide against the world 64-tree:
    if (query_voxel_world_meters(next_pos) != MAT_AIR) {
        // Test vertical vs horizontal bounce:
        if (query_voxel_world_meters(vec3<f32>(pos.x, next_pos.y, pos.z)) != MAT_AIR) {
            vel.y = -vel.y * 0.42;
            vel.x = vel.x * 0.72;
            vel.z = vel.z * 0.72;
        } else {
            vel.x = -vel.x * 0.5;
            vel.z = -vel.z * 0.5;
        }
    } else {
        pos = next_pos;
    }

    p.pos_life = vec4<f32>(pos, new_life);
    p.vel_color = vec4<f32>(vel, p.vel_color.w);
    debris_pool[idx] = p;
}

// Entry Point 1B: Primary 1080p Ray Marcher + Nanosecond Bitwise Contact AO (Upgrade 3)
// + Analytic 3D Motion Vectors + Water & Mirror Reflections
@compute @workgroup_size(8, 8, 1)
fn primary_raymarch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = screen_dims();
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let p_idx = pixel_index(gid.x, gid.y, dims.x);

    // Apply sub-pixel Halton jitter for Temporal Super-Resolution / DLAA:
    let jitter = u_cam.jitter_brush_dt.xy;
    let uv = (vec2<f32>(f32(gid.x) + 0.5 + jitter.x, f32(gid.y) + 0.5 + jitter.y)) / vec2<f32>(dims);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    let tan_half_fov = u_cam.cam_forward_fov.w;
    let aspect = u_cam.cam_right_aspect.w;

    // Orthonormal camera ray construction (`cam_up` is guaranteed `right × forward`):
    let ray_dir = normalize(
        u_cam.cam_forward_fov.xyz
        + u_cam.cam_right_aspect.xyz * (ndc.x * aspect * tan_half_fov)
        + u_cam.cam_up_frame.xyz * (ndc.y * tan_half_fov)
    );

    let eye = u_cam.cam_pos_time.xyz;
    let max_dist_m = 250.0;

    let cam_underwater = (u_cam.point_light_count.y & 0xFFFFu) != 0u;
    var primary = trace_world_64tree(eye, ray_dir, max_dist_m, 0u, cam_underwater);

    // Phase 6: Ray-march dynamic falling/tumbling RigidVoxelBody OBB instances:
    let rb_max_dist_m = select(max_dist_m, primary.t_voxels * VOXEL_SIZE, primary.hit);
    let rb_hit = trace_rigid_bodies(eye, ray_dir, rb_max_dist_m);
    var prev_world_pos = primary.hit_pos_meters;
    if (rb_hit.hit) {
        primary.hit = true;
        primary.t_voxels = rb_hit.t_voxels;
        primary.hit_pos_meters = rb_hit.hit_pos_meters;
        primary.normal = rb_hit.normal;
        primary.mat_id = rb_hit.mat_id;
        primary.voxel_hash = rb_hit.voxel_hash;
        primary.contact_ao = rb_hit.contact_ao;
        prev_world_pos = rb_hit.prev_hit_pos_meters;
    }

    if (!primary.hit) {
        gbuf_depth_mat[p_idx] = pack_depth_mat(1e5, MAT_AIR, 0u);
        gbuf_norm_motion[p_idx] = pack_norm_motion(ray_dir, vec2<f32>(0.0, 0.0));
        gbuf_albedo_ao[p_idx] = pack_albedo_ao(vec3<f32>(0.0), 1.0);
        return;
    }

    var hit_pos = primary.hit_pos_meters;
    var normal = primary.normal;
    var mat_id = primary.mat_id;
    var hit_dist = primary.t_voxels * VOXEL_SIZE;
    if (!rb_hit.hit) {
        prev_world_pos = hit_pos;
    }
    let mat = material_palette[mat_id];

    // Subtle per-voxel micro-variation so natural stone/grass/wood feels rich and organic,
    // faded at long range to eliminate high-frequency sub-pixel aliasing:
    let v_rand_weight = 1.0 - smoothstep(18.0, 48.0, hit_dist);
    let v_rand = ((f32(primary.voxel_hash & 0xFFu) / 255.0) * 2.0 - 1.0) * v_rand_weight;
    var albedo = clamp(
        mat.albedo_roughness.xyz * (1.0 + v_rand * mat.emission_meta.w),
        vec3<f32>(0.01),
        vec3<f32>(1.0),
    );
    let t_sec = u_cam.cam_pos_time.w;
    let sun_dir = u_cam.sun_dir_intensity.xyz;
    let sun_power = u_cam.sun_dir_intensity.w;

    if (cam_underwater) {
        // Underwater camera view: zero FOV distortion, smooth Beer-Lambert turquoise absorption & caustics
        let water_y = max(11.25, eye.y + 0.35);
        let caustic_rgb = eval_water_caustics(hit_pos, water_y, normal, sun_dir, t_sec);
        let transmittance = exp(-vec3<f32>(0.16, 0.045, 0.020) * min(hit_dist, 24.0));
        let deep_aqua = vec3<f32>(0.02, 0.14, 0.22);
        albedo = mix(deep_aqua, albedo * caustic_rgb, transmittance);
    } else if (mat_id == MAT_WATER_CRYSTAL) {
        // Fast translucent water with surface wave perturbation (no FOV distortion & 3x faster traversal!):
        let wave_n = eval_surface_water_normal(hit_pos.xz, t_sec);
        let wave_x = wave_n.x;
        let wave_z = wave_n.z;

        let refr_dir = normalize(ray_dir - vec3<f32>(wave_x, 0.0, wave_z) * 0.38);
        var refr_hit = trace_world_64tree(hit_pos + ray_dir * 0.12, refr_dir, u_cam.ray_distances.x, 1u, true);
        let rb_refr_max = select(u_cam.ray_distances.x, refr_hit.t_voxels * VOXEL_SIZE, refr_hit.hit);
        let rb_refr = trace_rigid_bodies(hit_pos + ray_dir * 0.12, refr_dir, rb_refr_max);
        if (rb_refr.hit) {
            refr_hit.hit = true;
            refr_hit.t_voxels = rb_refr.t_voxels;
            refr_hit.hit_pos_meters = rb_refr.hit_pos_meters;
            refr_hit.normal = rb_refr.normal;
            refr_hit.mat_id = rb_refr.mat_id;
            refr_hit.voxel_hash = rb_refr.voxel_hash;
        }
        var water_col = vec3<f32>(0.03, 0.16, 0.24);

        if (refr_hit.hit) {
            let bed_mat = material_palette[refr_hit.mat_id];
            let bed_rand = (f32(refr_hit.voxel_hash & 0xFFu) / 255.0) * 2.0 - 1.0;
            var bed_col = clamp(
                bed_mat.albedo_roughness.xyz * (1.0 + bed_rand * bed_mat.emission_meta.w),
                vec3<f32>(0.01),
                vec3<f32>(1.0),
            );
            let bp = refr_hit.hit_pos_meters;
            let caustic_rgb = eval_water_caustics(bp, hit_pos.y, primary.normal, sun_dir, t_sec);
            bed_col = bed_col * caustic_rgb + bed_mat.emission_meta.xyz * 0.5;

            let water_depth = distance(hit_pos, bp);
            let transmittance = exp(-vec3<f32>(0.32, 0.095, 0.042) * water_depth);
            let deep_scatter = vec3<f32>(0.02, 0.13, 0.20);
            water_col = mix(deep_scatter, bed_col, transmittance);
        }

        let refl_dir = reflect(ray_dir, wave_n);
        let n_dot_v = max(dot(-ray_dir, wave_n), 0.0);
        let fresnel = 0.03 + 0.42 * pow(clamp(1.0 - n_dot_v, 0.0, 1.0), 4.0);
        let day_factor = clamp(sun_dir.y * 2.5 + 0.15, 0.04, 1.0);
        var refl_col = eval_reflection_sky(refl_dir, sun_dir);

        // In Laptop/Perf mode, prioritize refraction and trace reflection ray only at steep grazing angles:
        let refl_thresh = select(0.08, 0.22, u_cam.quality_flags.z != 0u || (u_cam.quality_flags.w & 0xFFu) <= 12u);
        if (fresnel > refl_thresh) {
            var refl_hit = trace_world_64tree(hit_pos + vec3<f32>(0.0, 0.08, 0.0), refl_dir, 36.0, 2u, true);
            let rb_refl_max = select(36.0, refl_hit.t_voxels * VOXEL_SIZE, refl_hit.hit);
            let rb_refl = trace_rigid_bodies(hit_pos + vec3<f32>(0.0, 0.08, 0.0), refl_dir, rb_refl_max);
            if (rb_refl.hit) {
                refl_hit.hit = true;
                refl_hit.mat_id = rb_refl.mat_id;
            }
            if (refl_hit.hit) {
                let r_mat = material_palette[refl_hit.mat_id];
                refl_col = r_mat.albedo_roughness.xyz * day_factor + r_mat.emission_meta.xyz * 0.35;
            }
        }
        albedo = mix(water_col, clamp(refl_col, vec3<f32>(0.0), vec3<f32>(1.0)), fresnel);
        normal = vec3<f32>(0.0, 1.0, 0.0);
    } else if (mat_id == MAT_MIRROR_CHROME) {
        // Key 8: Real-Time Ray-Traced Mirror Chrome Material:
        let refl_dir = reflect(ray_dir, normal);
        let refl_origin = hit_pos + normal * 0.04;
        var refl_hit = trace_world_64tree(refl_origin, refl_dir, u_cam.ray_distances.y, 1u, false);
        var mirror_col = eval_reflection_sky(refl_dir, sun_dir);

        // If the reflection ray hits the originating mirror voxel itself (within 12cm on same monolith),
        // trace past the seam so we capture the true reflected scene instead of dropping to sky:
        if (refl_hit.hit && refl_hit.mat_id == 28u && (refl_hit.t_voxels * VOXEL_SIZE < 0.12)) {
            let re_hit = trace_world_64tree(refl_hit.hit_pos_meters + refl_dir * 0.08, refl_dir, u_cam.ray_distances.y, 1u, false);
            if (re_hit.hit) {
                refl_hit = re_hit;
            }
        }

        let is_self_mirror = refl_hit.hit && refl_hit.mat_id == 28u && (refl_hit.t_voxels * VOXEL_SIZE < 0.12);
        let rb_mirror_max = select(u_cam.ray_distances.y, refl_hit.t_voxels * VOXEL_SIZE, refl_hit.hit && !is_self_mirror);
        let rb_mirror = trace_rigid_bodies(refl_origin, refl_dir, rb_mirror_max);
        if (rb_mirror.hit) {
            refl_hit.hit = true;
            refl_hit.t_voxels = rb_mirror.t_voxels;
            refl_hit.hit_pos_meters = rb_mirror.hit_pos_meters;
            refl_hit.normal = rb_mirror.normal;
            refl_hit.mat_id = rb_mirror.mat_id;
            refl_hit.voxel_hash = rb_mirror.voxel_hash;
        }

        if (refl_hit.hit && !is_self_mirror) {
            let r_mat = material_palette[refl_hit.mat_id];
            let r_rand = (f32(refl_hit.voxel_hash & 0xFFu) / 255.0) * 2.0 - 1.0;
            let r_albedo = clamp(
                r_mat.albedo_roughness.xyz * (1.0 + r_rand * r_mat.emission_meta.w),
                vec3<f32>(0.01),
                vec3<f32>(1.0),
            );
            let r_em = r_mat.emission_meta.xyz;
            let r_ndl = max(dot(refl_hit.normal, sun_dir), 0.0);
            let r_sky = eval_reflection_sky(refl_hit.normal, sun_dir) * 0.45;
            var r_light = vec3<f32>(1.0, 0.92, 0.78) * sun_power * r_ndl * 0.75 + r_sky + vec3<f32>(0.04);

            // Also include any active analytical Point Lights (`Key 7`) in the mirror reflection:
            let pl_count = min(u_cam.point_light_count.x, 8u);
            for (var i = 0u; i < pl_count; i = i + 1u) {
                let pl = get_point_light(i);
                let to_pl = pl.xyz - refl_hit.hit_pos_meters;
                let d_pl = length(to_pl);
                if (d_pl > 0.01 && d_pl < 28.0) {
                    let pl_ndl = max(dot(refl_hit.normal, to_pl / d_pl), 0.0);
                    let atten = (pl.w * 3.5) / (1.0 + d_pl * d_pl * 0.22);
                    r_light += vec3<f32>(1.0, 0.92, 0.78) * pl_ndl * atten;
                }
            }

            let raw_refl = r_albedo * r_light + r_em * 0.85;
            // Compress HDR reflection smoothly into [0, 1] so it packs cleanly into the G-Buffer:
            mirror_col = raw_refl / (vec3<f32>(1.0) + raw_refl * 0.45);
        }

        albedo = clamp(mirror_col * vec3<f32>(0.93, 0.96, 1.0), vec3<f32>(0.01), vec3<f32>(1.0));
    }

    // Exact analytic 3D motion vector by projecting `prev_world_pos` with `prev_view_proj` (infinite Reverse-Z):
    let pixel_uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let prev_clip = u_cam.prev_view_proj * vec4<f32>(prev_world_pos, 1.0);
    var motion = vec2<f32>(-2.0, -2.0);
    if (prev_clip.w > 0.001) {
        let prev_ndc = prev_clip.xy / prev_clip.w;
        let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
        motion = prev_uv - pixel_uv;
    }

    let flags = select(0u, 1u, rb_hit.hit);
    gbuf_depth_mat[p_idx] = pack_depth_mat(hit_dist, mat_id, flags);
    gbuf_norm_motion[p_idx] = pack_norm_motion(normal, motion);
    // Upgrade 3: Store Nanosecond Bitwise Contact AO in `gbuf_albedo_ao`:
    gbuf_albedo_ao[p_idx] = pack_albedo_ao(albedo, primary.contact_ao);
}

// Entry Point 1C: Depth-tested splatting of flying 6.25cm 3D voxel debris cubes
@compute @workgroup_size(64, 1, 1)
fn splat_debris(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= 16384u) {
        return;
    }

    let p = debris_pool[idx];
    if (p.pos_life.w <= 0.0) {
        return;
    }

    let world_pos = p.pos_life.xyz;
    let clip = u_cam.curr_view_proj * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.15) {
        return;
    }

    let ndc = clip.xy / clip.w;
    if (abs(ndc.x) >= 0.995 || abs(ndc.y) >= 0.995) {
        return;
    }

    let dims = screen_dims();
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let center_px = vec2<i32>(uv * vec2<f32>(dims));

    let dist = distance(world_pos, u_cam.cam_pos_time.xyz);
    let half_size_px = clamp(i32((VOXEL_SIZE * 0.65 * f32(dims.y)) / (dist * u_cam.cam_forward_fov.w)), 1, 7);

    let payload = bitcast<u32>(p.vel_color.w);
    let col = unpack_lod_rgb(payload);
    let mat_id = (payload >> 24u) & 0xFFu;

    let prev_pos = world_pos - p.vel_color.xyz * u_cam.jitter_brush_dt.w;
    let prev_clip = u_cam.prev_view_proj * vec4<f32>(prev_pos, 1.0);
    var prev_uv = uv;
    if (prev_clip.w > 0.1) {
        let p_ndc = prev_clip.xy / prev_clip.w;
        prev_uv = vec2<f32>(p_ndc.x * 0.5 + 0.5, 0.5 - p_ndc.y * 0.5);
    }

    for (var dy = -half_size_px; dy <= half_size_px; dy = dy + 1) {
        for (var dx = -half_size_px; dx <= half_size_px; dx = dx + 1) {
            let sx = center_px.x + dx;
            let sy = center_px.y + dy;
            if (sx >= 0 && sy >= 0 && sx < i32(dims.x) && sy < i32(dims.y)) {
                let p_idx = pixel_index(u32(sx), u32(sy), dims.x);
                let cur_depth = unpack_depth(gbuf_depth_mat[p_idx]);
                if (dist < cur_depth) {
                    let n = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.7, 0.7), dy > 0);
                    let p_uv = (vec2<f32>(f32(sx), f32(sy)) + 0.5) / vec2<f32>(dims);
                    let motion = prev_uv - p_uv;
                    gbuf_depth_mat[p_idx] = pack_depth_mat(dist, mat_id, 2u);
                    gbuf_norm_motion[p_idx] = pack_norm_motion(n, motion);
                    gbuf_albedo_ao[p_idx] = pack_albedo_ao(col, 0.8);
                }
            }
        }
    }
}
