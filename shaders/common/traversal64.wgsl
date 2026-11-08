// Bound by `arcadia-render` passes 1-4 (concatenated as common traversal routines).
// Expected inputs: @group(0) engine uniform buffer & world storage pools (bindings 0..6).

struct TraceResult {
    hit: bool,
    t_voxels: f32,
    hit_pos_meters: vec3<f32>,
    normal: vec3<f32>,
    mat_id: u32,
    // For GI rays (`lod_bias >= 2u`), `lod_rgb` accumulates the 64-Tree Octree Irradiance Cache
    // harvested from emissive parent nodes (`shift = 4` 1m and `shift = 2` 25cm) along the ray!
    lod_rgb: vec3<f32>,
    voxel_hash: u32,
    // Upgrade 3: Nanosecond Bitwise Contact Ambient Occlusion evaluated in registers from the
    // intersected 64-bit LeafBrick64 occupancy mask (`[0.35 .. 1.0]`, 0 extra rays!).
    contact_ao: f32,
};

// Maps local 4x4x4 coordinates (0..3) to bit index 0..63:
fn child_bit_index(x: u32, y: u32, z: u32) -> u32 {
    return (z << 4u) | (y << 2u) | x;
}

// Tests whether bit `b` (0..63) is set in `(mask_low, mask_high)`:
fn node_has_child(mask_low: u32, mask_high: u32, b: u32) -> bool {
    // Branchless word select (WGSL masks shift counts to 5 bits, so `b & 31u` is explicit for clarity):
    let word = select(mask_high, mask_low, b < 32u);
    return ((word >> (b & 31u)) & 1u) != 0u;
}

// Hardware POPCNT (`countOneBits`) child slot offset calculation:
fn node_child_offset(node: Node64, b: u32) -> u32 {
    let base = node.child_ptr & PTR_INDEX_MASK;
    let low_half = b < 32u;
    let bit_prefix = (1u << (b & 31u)) - 1u;
    let prefix_low = select(0xFFFFFFFFu, bit_prefix, low_half);
    let prefix_high = select(bit_prefix, 0u, low_half);
    let rank = countOneBits(node.mask_low & prefix_low) + countOneBits(node.mask_high & prefix_high);
    return base + rank;
}

// Extracts 8-bit MaterialId from a 64-byte `LeafBrick64` at `bit_index` (0..63):
fn brick_get_material(brick_idx: u32, bit_index: u32) -> u32 {
    let word = brick_pool[brick_idx].packed_materials[bit_index >> 2u];
    let shift = (bit_index & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

// Unpacks pre-filtered volumetric RGB color/radiance from `Node64.lod_payload` (bits 0..23):
fn unpack_rgb_lod(lod_payload: u32) -> vec3<f32> {
    let r = f32(lod_payload & 0xFFu) / 255.0;
    let g = f32((lod_payload >> 8u) & 0xFFu) / 255.0;
    let b = f32((lod_payload >> 16u) & 0xFFu) / 255.0;
    return vec3<f32>(r, g, b);
}

// Upgrade 3: Nanosecond Bitwise Contact Ambient Occlusion (`01_high_leverage_upgrades_for_cs64.md`).
// Evaluates local neighbor occupancy inside the 4x4x4 leaf brick using pure register bit shifts
// (`±1` for X, `±4` for Y, `±16` for Z) and UV coordinates on the hit voxel face:
fn eval_leaf_bitwise_ao(
    mask_low: u32,
    mask_high: u32,
    cx: u32,
    cy: u32,
    cz: u32,
    normal: vec3<f32>,
    hit_p_v: vec3<f32>,
) -> f32 {
    let fx = clamp(fract(hit_p_v.x), 0.0, 1.0);
    let fy = clamp(fract(hit_p_v.y), 0.0, 1.0);
    let fz = clamp(fract(hit_p_v.z), 0.0, 1.0);

    // Step 1 voxel out along the surface normal to the air cell in front of the hit face:
    let nx = i32(cx) + i32(normal.x);
    let ny = i32(cy) + i32(normal.y);
    let nz = i32(cz) + i32(normal.z);

    // If the air neighbor across the face lies outside the 4x4x4 brick, fall back to subtle edge AO:
    if (nx < 0 || nx > 3 || ny < 0 || ny > 3 || nz < 0 || nz > 3) {
        return 1.0;
    }

    let base_b = child_bit_index(u32(nx), u32(ny), u32(nz));
    var occ = 0.0;

    if (abs(normal.y) > 0.5) {
        // Horizontal face (±Y): check ±X (bit shift ±1) and ±Z (bit shift ±16) neighbors in the air layer above/below:
        if (nx > 0 && node_has_child(mask_low, mask_high, base_b - 1u)) {
            occ += (1.0 - fx) * 0.42;
        }
        if (nx < 3 && node_has_child(mask_low, mask_high, base_b + 1u)) {
            occ += fx * 0.42;
        }
        if (nz > 0 && node_has_child(mask_low, mask_high, base_b - 16u)) {
            occ += (1.0 - fz) * 0.42;
        }
        if (nz < 3 && node_has_child(mask_low, mask_high, base_b + 16u)) {
            occ += fz * 0.42;
        }
    } else if (abs(normal.x) > 0.5) {
        // Vertical ±X face: check ±Y (bit shift ±4) and ±Z (bit shift ±16) neighbors:
        if (ny > 0 && node_has_child(mask_low, mask_high, base_b - 4u)) {
            occ += (1.0 - fy) * 0.42;
        }
        if (ny < 3 && node_has_child(mask_low, mask_high, base_b + 4u)) {
            occ += fy * 0.42;
        }
        if (nz > 0 && node_has_child(mask_low, mask_high, base_b - 16u)) {
            occ += (1.0 - fz) * 0.42;
        }
        if (nz < 3 && node_has_child(mask_low, mask_high, base_b + 16u)) {
            occ += fz * 0.42;
        }
    } else {
        // Vertical ±Z face: check ±X (bit shift ±1) and ±Y (bit shift ±4) neighbors:
        if (nx > 0 && node_has_child(mask_low, mask_high, base_b - 1u)) {
            occ += (1.0 - fx) * 0.42;
        }
        if (nx < 3 && node_has_child(mask_low, mask_high, base_b + 1u)) {
            occ += fx * 0.42;
        }
        if (ny > 0 && node_has_child(mask_low, mask_high, base_b - 4u)) {
            occ += (1.0 - fy) * 0.42;
        }
        if (ny < 3 && node_has_child(mask_low, mask_high, base_b + 4u)) {
            occ += fy * 0.42;
        }
    }

    return clamp(1.0 - occ, 0.35, 1.0);
}

// Slab ray-AABB intersection returning (t_enter, t_exit):
fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let enter_bound = select(bmax, bmin, inv_dir > vec3<f32>(0.0));
    let exit_bound = select(bmin, bmax, inv_dir > vec3<f32>(0.0));
    let tmin_v = (enter_bound - origin) * inv_dir;
    let tmax_v = (exit_bound - origin) * inv_dir;
    let tmin = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
    let tmax = min(min(tmax_v.x, tmax_v.y), tmax_v.z);
    return vec2<f32>(tmin, tmax);
}

struct DdaStep {
    t: f32,
    vi: vec3<u32>,
};

// Advances `(t, vi)` to the exact exit of the integer voxel box `[box_min, box_max)`
// without `step_eps` bias. Exited axes step deterministically to the adjacent voxel
// (`box_max` for +dir, `box_min - 1` for -dir); non-exited axes are clamped inside `[box_min, box_max - 1]`.
fn step_dda_box(
    v_origin: vec3<f32>,
    dir: vec3<f32>,
    inv_dir: vec3<f32>,
    neg_orig_inv: vec3<f32>,
    pos_dir: vec3<bool>,
    box_min: vec3<u32>,
    box_max: vec3<u32>,
    cur_t: f32,
) -> DdaStep {
    let exit_coord = select(box_min, box_max, pos_dir);
    let t_max = vec3<f32>(exit_coord) * inv_dir + neg_orig_inv;
    let t_exit = min(min(t_max.x, t_max.y), t_max.z);
    let next_t = max(cur_t, t_exit);

    let is_x = t_max.x <= t_max.y && t_max.x <= t_max.z;
    let is_y = !is_x && (t_max.y <= t_max.z);
    let is_z = !is_x && !is_y;
    let is_exit = vec3<bool>(is_x, is_y, is_z);
    let outside_vi = exit_coord - select(vec3<u32>(1u), vec3<u32>(0u), pos_dir);
    let inside_p = max(v_origin + dir * next_t, vec3<f32>(0.0));
    let inside_vi = clamp(vec3<u32>(inside_p), box_min, box_max - vec3<u32>(1u));

    var res: DdaStep;
    res.t = next_t;
    res.vi = select(inside_vi, outside_vi, is_exit);
    return res;
}

// Fast O(1) point occupancy query in world meters (used by GPU debris particle collisions & underwater detection):
fn query_voxel_world_meters(pos_m: vec3<f32>) -> u32 {
    let vp = pos_m * VOXELS_PER_METER;
    let world_max_v = vec3<f32>(u_cam.world_grid_dims.xyz) * CHUNK_DIM_F;
    if (any(vp < vec3<f32>(0.0)) || any(vp >= world_max_v)) {
        return MAT_AIR;
    }
    let vi = vec3<u32>(vp);
    let c = vi >> vec3<u32>(8u);
    let g = u_cam.world_grid_dims.xyz;
    let c_idx = c.x + c.y * g.x + c.z * g.x * g.y;
    let root_idx = chunk_roots[c_idx];
    if (root_idx == EMPTY_CHUNK_SENTINEL) {
        return MAT_AIR;
    }

    let l = vi & vec3<u32>(255u);
    var node_idx = root_idx;
    var shift = 6u;
    for (var level = 0u; level < 4u; level = level + 1u) {
        let node = node_pool[node_idx];
        let cx = (l.x >> shift) & 3u;
        let cy = (l.y >> shift) & 3u;
        let cz = (l.z >> shift) & 3u;
        let b = child_bit_index(cx, cy, cz);
        if (!node_has_child(node.mask_low, node.mask_high, b)) {
            return MAT_AIR;
        }
        if ((node.child_ptr & IS_LEAF_FLAG) != 0u) {
            if ((node.child_ptr & IS_UNIFORM_FLAG) != 0u) {
                return (node.lod_payload >> 24u) & 0xFFu;
            } else {
                return brick_get_material(node.child_ptr & PTR_INDEX_MASK, b);
            }
        }
        node_idx = node_child_offset(node, b);
        shift = shift - 2u;
    }
    return MAT_AIR;
}

// Hierarchical 64-Tree Ray Marcher with:
// 1. Custom Start Offset `start_t_m` (Upgrade 4: Tile Depth Pre-Pass integration)
// 2. 2-Way Move-to-Front (MTF) Chunk Root Cache + Ancestor Restart Stack (skips L0->L1->L2 re-walks!)
// 3. Nanosecond Bitwise Contact AO (Upgrade 3)
fn trace_world_64tree_from(
    origin_m: vec3<f32>,
    dir: vec3<f32>,
    start_t_m: f32,
    max_dist_m: f32,
    lod_bias: u32,
    ignore_water: bool,
) -> TraceResult {
    var res: TraceResult;
    res.hit = false;
    res.t_voxels = max_dist_m * VOXELS_PER_METER;
    res.hit_pos_meters = origin_m + dir * max_dist_m;
    res.normal = vec3<f32>(0.0, 1.0, 0.0);
    res.mat_id = MAT_AIR;
    res.lod_rgb = vec3<f32>(0.0);
    res.voxel_hash = 0u;
    res.contact_ao = 1.0;

    let v_origin = origin_m * VOXELS_PER_METER;
    let max_t_v = max_dist_m * VOXELS_PER_METER;
    let start_t_v = max(start_t_m * VOXELS_PER_METER, 0.0);

    let safe_dir = select(dir, vec3<f32>(1e-7), abs(dir) < vec3<f32>(1e-7));
    let inv_dir = 1.0 / safe_dir;

    let grid_dims = u_cam.world_grid_dims.xyz;
    let world_max_v = vec3<f32>(grid_dims) * CHUNK_DIM_F;

    let world_hit = intersect_aabb(v_origin, inv_dir, vec3<f32>(0.0), world_max_v);
    let t_enter = max(world_hit.x, start_t_v);
    if (world_hit.y < t_enter || t_enter >= max_t_v) {
        return res;
    }

    var t = t_enter;
    let t_limit = min(world_hit.y, max_t_v);
    var max_steps = u_cam.ray_steps.x;
    if (lod_bias == 1u) {
        max_steps = u_cam.ray_steps.y;
    } else if (lod_bias >= 2u) {
        max_steps = u_cam.ray_steps.z;
    }

    // Loop-invariant LOD thresholds (voxel units / step counts), hoisted out of the march loop:
    let prim_lod_t = u_cam.lod_settings.x * VOXELS_PER_METER;
    let prim_lod_step = u_cam.quality_flags.x;
    let sec_lod_t = u_cam.lod_settings.y * VOXELS_PER_METER;
    let sec_lod_step = u_cam.quality_flags.y;
    let gi_lod_t = u_cam.lod_settings.z * VOXELS_PER_METER;
    let gi_lod_step = (u_cam.quality_flags.w >> 16u) & 0xFFu;

    let pos_dir = inv_dir >= vec3<f32>(0.0);
    let neg_orig_inv = -v_origin * inv_dir;
    let world_max_vi = grid_dims << vec3<u32>(8u);

    let start_p = clamp(v_origin + dir * (t + 1e-4), vec3<f32>(0.0), world_max_v - vec3<f32>(1e-3));
    var vi = vec3<u32>(start_p);

    // ------------------------------------------------------------------------
    // Move-to-Front (MTF) 2-Slot Chunk Cache + Ancestor Node Restart Cache:
    // - `mtf_c0` / `mtf_c1`: 2-way MTF chunk cache (frequent chunk bubbles to slot 0, old falls off)
    // - `cached_l1_node` / `cached_l2_node` / `cached_l3_node`: caches active ancestor Node64s
    //   in registers so successive DDA steps directly restart without re-walking from root!
    // ------------------------------------------------------------------------
    var mtf_c0 = vec3<u32>(0xFFFFFFFFu);
    var mtf_root0 = EMPTY_CHUNK_SENTINEL;

    var mtf_c1 = vec3<u32>(0xFFFFFFFFu);
    var mtf_root1 = EMPTY_CHUNK_SENTINEL;

    var prev_vi = vec3<u32>(0xFFFFFFFFu);
    var cached_shift = 6u;
    var cached_l1_node: Node64;
    var cached_l2_node: Node64;
    var cached_l3_node: Node64;

    for (var step = 0u; step < max_steps; step = step + 1u) {
        if (t >= t_limit) {
            break;
        }
        if (any(vi >= world_max_vi)) {
            t = t_limit;
            break;
        }

        let diff = (prev_vi.x ^ vi.x) | (prev_vi.y ^ vi.y) | (prev_vi.z ^ vi.z);
        prev_vi = vi;

        var root_idx = mtf_root0;

        if (diff >= 256u) {
            let c = vi >> vec3<u32>(8u);
            cached_shift = 6u;
            if (all(c == mtf_c1)) {
                // Hit in MTF slot 1! Bubble slot 1 to slot 0 and push old slot 0 to slot 1:
                root_idx = mtf_root1;
                mtf_c1 = mtf_c0;
                mtf_root1 = mtf_root0;
                mtf_c0 = c;
                mtf_root0 = root_idx;
            } else {
                // Miss! Evict slot 1, shift slot 0 -> slot 1, and insert new chunk at slot 0:
                let c_idx = c.x + c.y * grid_dims.x + c.z * grid_dims.x * grid_dims.y;
                root_idx = chunk_roots[c_idx];
                mtf_c1 = mtf_c0;
                mtf_root1 = mtf_root0;
                mtf_c0 = c;
                mtf_root0 = root_idx;
            }
        }

        if (root_idx == EMPTY_CHUNK_SENTINEL) {
            let chunk_min_u = vi & vec3<u32>(~255u);
            let st = step_dda_box(
                v_origin,
                dir,
                inv_dir,
                neg_orig_inv,
                pos_dir,
                chunk_min_u,
                chunk_min_u + vec3<u32>(256u),
                t,
            );
            t = st.t;
            vi = st.vi;
            continue;
        }

        // Adaptive Ray-Cone LOD:
        var min_shift = 0u;
        if (lod_bias == 0u) {
            if (t > prim_lod_t || step > prim_lod_step) {
                min_shift = 2u; // Stop at 0.25m meso-cells
            }
        } else if (lod_bias == 1u) {
            if (t > (sec_lod_t * 2.2857) || step > sec_lod_step) {
                min_shift = 4u; // Stop at 1.0m macro-cells
            } else if (t > sec_lod_t) {
                min_shift = 2u; // Stop at 0.25m meso-cells
            }
        } else {
            if (t > gi_lod_t || step > gi_lod_step) {
                min_shift = 2u; // Stop at 0.25m meso-cells
            }
        }

        var node: Node64;
        var shift = 6u;
        if (diff < 4u && min_shift == 0u && cached_shift == 0u) {
            node = cached_l3_node;
            shift = 0u;
        } else if (diff < 16u && min_shift <= 2u && cached_shift <= 2u) {
            node = cached_l2_node;
            shift = 2u;
        } else if (diff < 64u && min_shift <= 4u && cached_shift <= 4u) {
            node = cached_l1_node;
            shift = 4u;
        } else {
            node = node_pool[root_idx];
            shift = 6u;
        }
        cached_shift = shift;

        let l = vi & vec3<u32>(255u);
        loop {
            let cx = (l.x >> shift) & 3u;
            let cy = (l.y >> shift) & 3u;
            let cz = (l.z >> shift) & 3u;
            let b = child_bit_index(cx, cy, cz);

            // Fast-path for `ignore_water`: if an entire 4x4x4 uniform leaf/internal node is pure
            // `MAT_WATER_CRYSTAL`, skip the whole 4x4x4 node box cleanly without stepping voxel-by-voxel!
            if (ignore_water && (node.child_ptr & IS_UNIFORM_FLAG) != 0u) {
                let uni_mat = (node.lod_payload >> 24u) & 0xFFu;
                if (uni_mat == MAT_WATER_CRYSTAL) {
                    let node_span = 4u << shift;
                    let node_min_u = vi & vec3<u32>(~(node_span - 1u));
                    let st = step_dda_box(
                        v_origin,
                        dir,
                        inv_dir,
                        neg_orig_inv,
                        pos_dir,
                        node_min_u,
                        node_min_u + vec3<u32>(node_span),
                        t,
                    );
                    t = st.t;
                    vi = st.vi;
                    break;
                }
            }

            if (!node_has_child(node.mask_low, node.mask_high, b)) {
                let cell_span = 1u << shift;
                let cell_min_u = vi & vec3<u32>(~(cell_span - 1u));
                let cell_max_u = cell_min_u + vec3<u32>(cell_span);
                let st = step_dda_box(
                    v_origin,
                    dir,
                    inv_dir,
                    neg_orig_inv,
                    pos_dir,
                    cell_min_u,
                    cell_max_u,
                    t,
                );
                t = st.t;
                vi = st.vi;
                break;
            }

            let is_leaf = (node.child_ptr & IS_LEAF_FLAG) != 0u;
            let stop_for_lod = shift <= min_shift;

            if (is_leaf || stop_for_lod) {
                let cell_span = 1u << shift;
                let cell_min_u = vi & vec3<u32>(~(cell_span - 1u));
                let cell_max_u = cell_min_u + vec3<u32>(cell_span);

                var mat_id = (node.lod_payload >> 24u) & 0xFFu;
                if (is_leaf && (node.child_ptr & IS_UNIFORM_FLAG) == 0u) {
                    mat_id = brick_get_material(node.child_ptr & PTR_INDEX_MASK, b);
                }

                if (ignore_water && mat_id == MAT_WATER_CRYSTAL) {
                    let st = step_dda_box(
                        v_origin,
                        dir,
                        inv_dir,
                        neg_orig_inv,
                        pos_dir,
                        cell_min_u,
                        cell_max_u,
                        t,
                    );
                    t = st.t;
                    vi = st.vi;
                    break;
                }

                if (mat_id != MAT_AIR) {
                    let enter_coord = select(cell_max_u, cell_min_u, pos_dir);
                    let tmin_v = vec3<f32>(enter_coord) * inv_dir + neg_orig_inv;
                    let cell_enter = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
                    let exact_t = max(cell_enter, 0.0);
                    let hit_p_v = v_origin + dir * exact_t;
                    res.hit = true;
                    res.t_voxels = exact_t;
                    res.hit_pos_meters = hit_p_v * VOXEL_SIZE;

                    // Robust edge/corner normal resolution: break floating-point ties within eps
                    // toward the face most directly opposing the ray direction (prevents grid line normal flips):
                    let max_t = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
                    let eps = 1e-4 * max(max_t * 0.01, 1.0);
                    let cand_x = select(0.0, abs(dir.x), abs(tmin_v.x - max_t) <= eps);
                    let cand_y = select(0.0, abs(dir.y), abs(tmin_v.y - max_t) <= eps);
                    let cand_z = select(0.0, abs(dir.z), abs(tmin_v.z - max_t) <= eps);

                    if (cand_x >= cand_y && cand_x >= cand_z && cand_x > 0.0) {
                        res.normal = vec3<f32>(-sign(dir.x), 0.0, 0.0);
                    } else if (cand_y >= cand_z && cand_y > 0.0) {
                        res.normal = vec3<f32>(0.0, -sign(dir.y), 0.0);
                    } else if (cand_z > 0.0) {
                        res.normal = vec3<f32>(0.0, 0.0, -sign(dir.z));
                    } else if (tmin_v.x >= tmin_v.y && tmin_v.x >= tmin_v.z) {
                        res.normal = vec3<f32>(-sign(dir.x), 0.0, 0.0);
                    } else if (tmin_v.y >= tmin_v.z) {
                        res.normal = vec3<f32>(0.0, -sign(dir.y), 0.0);
                    } else {
                        res.normal = vec3<f32>(0.0, 0.0, -sign(dir.z));
                    }
                    res.mat_id = mat_id;
                    res.lod_rgb = unpack_rgb_lod(node.lod_payload);
                    res.voxel_hash = pcg_hash(vi.x * 73856093u ^ vi.y * 19349663u ^ vi.z * 83492791u);
                    if (shift == 0u && lod_bias == 0u) {
                        let hit_dist_m = t * VOXEL_SIZE;
                        let raw_dist = bitcast<f32>(u_cam.point_light_count.w);
                        let ao_max_dist = select(raw_dist, 24.0, raw_dist < 1.0);
                        if (hit_dist_m < ao_max_dist) {
                            let raw_ao = eval_leaf_bitwise_ao(
                                node.mask_low,
                                node.mask_high,
                                cx,
                                cy,
                                cz,
                                res.normal,
                                hit_p_v,
                            );
                            let ao_min_dist = ao_max_dist * 0.5;
                            let ao_weight = 1.0 - smoothstep(ao_min_dist, ao_max_dist, hit_dist_m);
                            res.contact_ao = mix(1.0, raw_ao, ao_weight);
                        } else {
                            res.contact_ao = 1.0;
                        }
                    }
                    return res;
                }

                let st = step_dda_box(
                    v_origin,
                    dir,
                    inv_dir,
                    neg_orig_inv,
                    pos_dir,
                    cell_min_u,
                    cell_max_u,
                    t,
                );
                t = st.t;
                vi = st.vi;
                break;
            }

            let next_idx = node_child_offset(node, b);
            node = node_pool[next_idx];
            shift = shift - 2u;

            cached_shift = shift;
            if (shift == 4u) {
                cached_l1_node = node;
            } else if (shift == 2u) {
                cached_l2_node = node;
            } else if (shift == 0u) {
                cached_l3_node = node;
            }
        }
    }

    return res;
}

// Default wrapper starting at t = 0.0m:
fn trace_world_64tree(
    origin_m: vec3<f32>,
    dir: vec3<f32>,
    max_dist_m: f32,
    lod_bias: u32,
    ignore_water: bool,
) -> TraceResult {
    return trace_world_64tree_from(origin_m, dir, 0.0, max_dist_m, lod_bias, ignore_water);
}

// Fast any-hit shadow ray marcher returning 1.0 (unoccluded) or 0.0 (occluded)
// with Move-to-Front (MTF) Chunk & Ancestor Node Register Caching.
fn trace_shadow_vis(
    origin_m: vec3<f32>,
    dir: vec3<f32>,
    max_dist_m: f32,
) -> f32 {
    let v_origin = origin_m * VOXELS_PER_METER;
    let max_t_v = max_dist_m * VOXELS_PER_METER;

    let safe_dir = select(dir, vec3<f32>(1e-7), abs(dir) < vec3<f32>(1e-7));
    let inv_dir = 1.0 / safe_dir;

    let grid_dims = u_cam.world_grid_dims.xyz;
    let world_max_v = vec3<f32>(grid_dims) * CHUNK_DIM_F;

    let world_hit = intersect_aabb(v_origin, inv_dir, vec3<f32>(0.0), world_max_v);
    if (world_hit.y < max(world_hit.x, 0.0) || world_hit.x >= max_t_v) {
        return 1.0;
    }

    let t_enter = max(world_hit.x, 0.0);
    var t = t_enter;
    let t_limit = min(world_hit.y, max_t_v);
    let max_steps = u_cam.ray_steps.w;

    let pos_dir = inv_dir >= vec3<f32>(0.0);
    let neg_orig_inv = -v_origin * inv_dir;
    let world_max_vi = grid_dims << vec3<u32>(8u);

    let start_p = clamp(v_origin + dir * (t + 1e-4), vec3<f32>(0.0), world_max_v - vec3<f32>(1e-3));
    var vi = vec3<u32>(start_p);

    var prev_vi = vec3<u32>(0xFFFFFFFFu);
    var cached_shift = 6u;
    var cached_l1_node: Node64;
    var cached_l2_node: Node64;
    var cached_l3_node: Node64;

    var cached_root_idx = EMPTY_CHUNK_SENTINEL;

    for (var step = 0u; step < max_steps; step = step + 1u) {
        if (t >= t_limit) {
            break;
        }
        if (any(vi >= world_max_vi)) {
            t = t_limit;
            break;
        }

        let diff = (prev_vi.x ^ vi.x) | (prev_vi.y ^ vi.y) | (prev_vi.z ^ vi.z);
        prev_vi = vi;

        var root_idx = cached_root_idx;
        if (diff >= 256u) {
            let c = vi >> vec3<u32>(8u);
            cached_shift = 6u;
            let c_idx = c.x + c.y * grid_dims.x + c.z * grid_dims.x * grid_dims.y;
            root_idx = chunk_roots[c_idx];
            cached_root_idx = root_idx;
        }

        if (root_idx == EMPTY_CHUNK_SENTINEL) {
            let chunk_min_u = vi & vec3<u32>(~255u);
            let st = step_dda_box(
                v_origin,
                dir,
                inv_dir,
                neg_orig_inv,
                pos_dir,
                chunk_min_u,
                chunk_min_u + vec3<u32>(256u),
                t,
            );
            t = st.t;
            vi = st.vi;
            continue;
        }

        var min_shift = 0u; // Start at 0u so ray cleanly exits the surface without self-intersecting!
        if (t > 48.0 || step > 6u) {
            min_shift = 2u; // Beyond 3m or 6 steps, test at 0.25m meso-cells (16x faster traversal)
        }
        if (t > 192.0 || step > 16u) {
            min_shift = 4u; // Beyond 12m or 16 steps, test at 1.0m macro-cells
        }

        var node: Node64;
        var shift = 6u;
        if (diff < 4u && min_shift == 0u && cached_shift == 0u) {
            node = cached_l3_node;
            shift = 0u;
        } else if (diff < 16u && min_shift <= 2u && cached_shift <= 2u) {
            node = cached_l2_node;
            shift = 2u;
        } else if (diff < 64u && min_shift <= 4u && cached_shift <= 4u) {
            node = cached_l1_node;
            shift = 4u;
        } else {
            node = node_pool[root_idx];
            shift = 6u;
        }
        cached_shift = shift;

        let l = vi & vec3<u32>(255u);
        loop {
            let cx = (l.x >> shift) & 3u;
            let cy = (l.y >> shift) & 3u;
            let cz = (l.z >> shift) & 3u;
            let b = child_bit_index(cx, cy, cz);

            // Water fast-skip
            if ((node.child_ptr & IS_UNIFORM_FLAG) != 0u) {
                let uni_mat = (node.lod_payload >> 24u) & 0xFFu;
                if (uni_mat == MAT_WATER_CRYSTAL) {
                    let node_span = 4u << shift;
                    let node_min_u = vi & vec3<u32>(~(node_span - 1u));
                    let st = step_dda_box(
                        v_origin,
                        dir,
                        inv_dir,
                        neg_orig_inv,
                        pos_dir,
                        node_min_u,
                        node_min_u + vec3<u32>(node_span),
                        t,
                    );
                    t = st.t;
                    vi = st.vi;
                    break;
                }
            }

            if (!node_has_child(node.mask_low, node.mask_high, b)) {
                let cell_span = 1u << shift;
                let cell_min_u = vi & vec3<u32>(~(cell_span - 1u));
                let cell_max_u = cell_min_u + vec3<u32>(cell_span);
                let st = step_dda_box(
                    v_origin,
                    dir,
                    inv_dir,
                    neg_orig_inv,
                    pos_dir,
                    cell_min_u,
                    cell_max_u,
                    t,
                );
                t = st.t;
                vi = st.vi;
                break;
            }

            let is_leaf = (node.child_ptr & IS_LEAF_FLAG) != 0u;
            let stop_for_lod = shift <= min_shift;

            if (is_leaf || stop_for_lod) {
                let mat_id = (node.lod_payload >> 24u) & 0xFFu;
                if (mat_id == MAT_WATER_CRYSTAL) {
                    let cell_span = 1u << shift;
                    let cell_min_u = vi & vec3<u32>(~(cell_span - 1u));
                    let cell_max_u = cell_min_u + vec3<u32>(cell_span);
                    let st = step_dda_box(
                        v_origin,
                        dir,
                        inv_dir,
                        neg_orig_inv,
                        pos_dir,
                        cell_min_u,
                        cell_max_u,
                        t,
                    );
                    t = st.t;
                    vi = st.vi;
                    break;
                }
                // Solid obstacle occluded return.
                return 0.0;
            }

            let next_idx = node_child_offset(node, b);
            node = node_pool[next_idx];
            shift = shift - 2u;

            cached_shift = shift;
            if (shift == 4u) {
                cached_l1_node = node;
            } else if (shift == 2u) {
                cached_l2_node = node;
            } else if (shift == 0u) {
                cached_l3_node = node;
            }
        }
    }

    if (t < t_limit) {
        // Step budget was exhausted before ray reached open air!
        // In underground caves / deep terrain, assume occluded so daylight does not bleed into enclosed spaces:
        let curr_p = v_origin + dir * t;
        if (curr_p.y * VOXEL_SIZE < 32.0) {
            return 0.0;
        }
    }

    return 1.0;
}

fn is_emissive_mat(mat_id: u32) -> bool {
    return (mat_id >= 19u && mat_id <= 22u) || mat_id == 24u || mat_id == 27u;
}

// Fast any-hit shadow ray marcher for ReSTIR DI returning 1.0 (unoccluded or struck emitter) or 0.0 (occluded)
// with Move-to-Front (MTF) Chunk & Ancestor Node Register Caching.
fn trace_shadow_vis_di(
    origin_m: vec3<f32>,
    dir: vec3<f32>,
    max_dist_m: f32,
) -> f32 {
    let v_origin = origin_m * VOXELS_PER_METER;
    let max_t_v = max_dist_m * VOXELS_PER_METER;

    let safe_dir = select(dir, vec3<f32>(1e-7), abs(dir) < vec3<f32>(1e-7));
    let inv_dir = 1.0 / safe_dir;

    let grid_dims = u_cam.world_grid_dims.xyz;
    let world_max_v = vec3<f32>(grid_dims) * CHUNK_DIM_F;

    let world_hit = intersect_aabb(v_origin, inv_dir, vec3<f32>(0.0), world_max_v);
    if (world_hit.y < max(world_hit.x, 0.0) || world_hit.x >= max_t_v) {
        return 1.0;
    }

    let t_enter = max(world_hit.x, 0.0);
    var t = t_enter;
    let t_limit = min(world_hit.y, max_t_v);
    let max_steps = clamp(u_cam.ray_steps.w, 32u, 48u);

    let pos_dir = inv_dir >= vec3<f32>(0.0);
    let neg_orig_inv = -v_origin * inv_dir;
    let world_max_vi = grid_dims << vec3<u32>(8u);

    let start_p = clamp(v_origin + dir * (t + 1e-4), vec3<f32>(0.0), world_max_v - vec3<f32>(1e-3));
    var vi = vec3<u32>(start_p);

    var prev_vi = vec3<u32>(0xFFFFFFFFu);
    var cached_shift = 6u;
    var cached_l1_node: Node64;
    var cached_l2_node: Node64;
    var cached_l3_node: Node64;

    var cached_root_idx = EMPTY_CHUNK_SENTINEL;

    for (var step = 0u; step < max_steps; step = step + 1u) {
        if (t >= t_limit) {
            break;
        }
        if (any(vi >= world_max_vi)) {
            t = t_limit;
            break;
        }

        let diff = (prev_vi.x ^ vi.x) | (prev_vi.y ^ vi.y) | (prev_vi.z ^ vi.z);
        prev_vi = vi;

        var root_idx = cached_root_idx;
        if (diff >= 256u) {
            let c = vi >> vec3<u32>(8u);
            cached_shift = 6u;
            let c_idx = c.x + c.y * grid_dims.x + c.z * grid_dims.x * grid_dims.y;
            root_idx = chunk_roots[c_idx];
            cached_root_idx = root_idx;
        }

        if (root_idx == EMPTY_CHUNK_SENTINEL) {
            let chunk_min_u = vi & vec3<u32>(~255u);
            let st = step_dda_box(
                v_origin,
                dir,
                inv_dir,
                neg_orig_inv,
                pos_dir,
                chunk_min_u,
                chunk_min_u + vec3<u32>(256u),
                t,
            );
            t = st.t;
            vi = st.vi;
            continue;
        }

        var min_shift = 0u;
        let dist_rem = t_limit - t;
        if (t > 48.0 && dist_rem > 32.0 && step > 6u) {
            min_shift = 2u;
        }
        if (t > 160.0 && dist_rem > 64.0 && step > 14u) {
            min_shift = 4u;
        }

        var node: Node64;
        var shift = 6u;
        if (diff < 4u && min_shift == 0u && cached_shift == 0u) {
            node = cached_l3_node;
            shift = 0u;
        } else if (diff < 16u && min_shift <= 2u && cached_shift <= 2u) {
            node = cached_l2_node;
            shift = 2u;
        } else if (diff < 64u && min_shift <= 4u && cached_shift <= 4u) {
            node = cached_l1_node;
            shift = 4u;
        } else {
            node = node_pool[root_idx];
            shift = 6u;
        }
        cached_shift = shift;

        let l = vi & vec3<u32>(255u);
        loop {
            let cx = (l.x >> shift) & 3u;
            let cy = (l.y >> shift) & 3u;
            let cz = (l.z >> shift) & 3u;
            let b = child_bit_index(cx, cy, cz);

            // Water fast-skip
            if ((node.child_ptr & IS_UNIFORM_FLAG) != 0u) {
                let uni_mat = (node.lod_payload >> 24u) & 0xFFu;
                if (uni_mat == MAT_WATER_CRYSTAL) {
                    let node_span = 4u << shift;
                    let node_min_u = vi & vec3<u32>(~(node_span - 1u));
                    let st = step_dda_box(
                        v_origin,
                        dir,
                        inv_dir,
                        neg_orig_inv,
                        pos_dir,
                        node_min_u,
                        node_min_u + vec3<u32>(node_span),
                        t,
                    );
                    t = st.t;
                    vi = st.vi;
                    break;
                }
            }

            if (!node_has_child(node.mask_low, node.mask_high, b)) {
                let cell_span = 1u << shift;
                let cell_min_u = vi & vec3<u32>(~(cell_span - 1u));
                let cell_max_u = cell_min_u + vec3<u32>(cell_span);
                let st = step_dda_box(
                    v_origin,
                    dir,
                    inv_dir,
                    neg_orig_inv,
                    pos_dir,
                    cell_min_u,
                    cell_max_u,
                    t,
                );
                t = st.t;
                vi = st.vi;
                break;
            }

            let is_leaf = (node.child_ptr & IS_LEAF_FLAG) != 0u;
            let is_uniform = (node.child_ptr & IS_UNIFORM_FLAG) != 0u;
            let stop_for_lod = shift <= min_shift;

            if (is_leaf || is_uniform || stop_for_lod) {
                var mat_id = (node.lod_payload >> 24u) & 0xFFu;
                if (is_leaf && !is_uniform && shift == 0u) {
                    mat_id = brick_get_material(node.child_ptr & PTR_INDEX_MASK, b);
                }
                if (mat_id == MAT_WATER_CRYSTAL) {
                    let cell_span = 1u << shift;
                    let cell_min_u = vi & vec3<u32>(~(cell_span - 1u));
                    let cell_max_u = cell_min_u + vec3<u32>(cell_span);
                    let st = step_dda_box(
                        v_origin,
                        dir,
                        inv_dir,
                        neg_orig_inv,
                        pos_dir,
                        cell_min_u,
                        cell_max_u,
                        t,
                    );
                    t = st.t;
                    vi = st.vi;
                    break;
                }
                // WHY: Only treat an emissive hit as unoccluded if it is within 1m (16 voxels) of the target emitter distance.
                if (is_emissive_mat(mat_id) && t >= t_limit - 16.0) {
                    return 1.0;
                }
                return 0.0;
            }

            let next_idx = node_child_offset(node, b);
            node = node_pool[next_idx];
            shift = shift - 2u;

            cached_shift = shift;
            if (shift == 4u) {
                cached_l1_node = node;
            } else if (shift == 2u) {
                cached_l2_node = node;
            } else if (shift == 0u) {
                cached_l3_node = node;
            }
        }
    }

    // If the ray arrived at the target emitter distance without striking an obstacle, it is visible:
    if (t >= t_limit - 2.0) {
        return 1.0;
    }

    // Step budget was exhausted before reaching the target emitter.
    // Conservative occlusion: do NOT leak light through unmarched terrain:
    return 0.0;
}

// ============================================================================
// Dynamic rigid body OBB ray marcher (`trace_rigid_bodies`).
// ============================================================================

struct RigidTraceResult {
    hit: bool,
    t_voxels: f32,
    hit_pos_meters: vec3<f32>,
    prev_hit_pos_meters: vec3<f32>,
    normal: vec3<f32>,
    mat_id: u32,
    voxel_hash: u32,
    contact_ao: f32,
};

fn trace_single_rigid_body(
    rb_idx: u32,
    origin_m: vec3<f32>,
    dir: vec3<f32>,
    max_dist_m: f32,
) -> RigidTraceResult {
    var res: RigidTraceResult;
    res.hit = false;
    res.t_voxels = max_dist_m * VOXELS_PER_METER;
    res.hit_pos_meters = origin_m + dir * max_dist_m;
    res.prev_hit_pos_meters = res.hit_pos_meters;
    res.normal = vec3<f32>(0.0, 1.0, 0.0);
    res.mat_id = MAT_AIR;
    res.voxel_hash = 0u;
    res.contact_ao = 1.0;

    let rb = rigid_bodies[rb_idx];
    let root_idx = bitcast<u32>(rb.local_bounds_and_root.w);
    if (root_idx == EMPTY_CHUNK_SENTINEL) {
        return res;
    }

    // 1. Fast ray-sphere bounding pre-reject in world meters:
    let sphere_c = rb.world_sphere.xyz;
    let sphere_r = rb.world_sphere.w;
    let oc = origin_m - sphere_c;
    let b_proj = dot(oc, dir);
    let c_dist = dot(oc, oc) - sphere_r * sphere_r;
    if (c_dist > 0.0 && b_proj > 0.0) {
        return res;
    }
    let disc = b_proj * b_proj - c_dist;
    if (disc < 0.0) {
        return res;
    }
    let sphere_near_m = max(-b_proj - sqrt(disc), 0.0);
    if (sphere_near_m >= max_dist_m) {
        return res;
    }

    // 2. Transform ray into body's local CS64 coordinate system (in voxel units 1/16m):
    let local_o_m = (rb.world_to_local * vec4<f32>(origin_m, 1.0)).xyz;
    let local_d = (rb.world_to_local * vec4<f32>(dir, 0.0)).xyz;
    let v_origin = local_o_m * VOXELS_PER_METER;
    let local_max_v = rb.local_bounds_and_root.xyz * VOXELS_PER_METER;
    let max_t_v = max_dist_m * VOXELS_PER_METER;

    let safe_dir = select(local_d, vec3<f32>(1e-7), abs(local_d) < vec3<f32>(1e-7));
    let inv_dir = 1.0 / safe_dir;

    let box_hit = intersect_aabb(v_origin, inv_dir, vec3<f32>(0.0), local_max_v);
    let t_enter = max(box_hit.x, 0.0);
    if (box_hit.y < t_enter || t_enter >= max_t_v) {
        return res;
    }

    var t = t_enter;
    let t_limit = min(box_hit.y, max_t_v);

    let pos_dir = inv_dir >= vec3<f32>(0.0);
    let neg_orig_inv = -v_origin * inv_dir;
    let local_max_vi = vec3<u32>(local_max_v + vec3<f32>(0.5));

    let start_p = clamp(v_origin + local_d * (t + 1e-4), vec3<f32>(0.0), local_max_v - vec3<f32>(1e-3));
    var vi = vec3<u32>(start_p);

    var prev_vi = vec3<u32>(0xFFFFFFFFu);
    var cached_shift = 6u;
    var cached_l1_node: Node64;
    var cached_l2_node: Node64;
    var cached_l3_node: Node64;

    for (var step = 0u; step < 64u; step = step + 1u) {
        if (t >= t_limit) {
            break;
        }
        if (any(vi >= local_max_vi)) {
            t = t_limit;
            break;
        }

        let diff = (prev_vi.x ^ vi.x) | (prev_vi.y ^ vi.y) | (prev_vi.z ^ vi.z);
        prev_vi = vi;

        var node: Node64;
        var shift = 6u;
        if (diff < 4u && cached_shift == 0u) {
            node = cached_l3_node;
            shift = 0u;
        } else if (diff < 16u && cached_shift <= 2u) {
            node = cached_l2_node;
            shift = 2u;
        } else if (diff < 64u && cached_shift <= 4u) {
            node = cached_l1_node;
            shift = 4u;
        } else {
            node = node_pool[root_idx];
            shift = 6u;
        }
        cached_shift = shift;

        let l = vi & vec3<u32>(255u);
        loop {
            let cx = (l.x >> shift) & 3u;
            let cy = (l.y >> shift) & 3u;
            let cz = (l.z >> shift) & 3u;
            let b = child_bit_index(cx, cy, cz);

            if (!node_has_child(node.mask_low, node.mask_high, b)) {
                let cell_span = 1u << shift;
                let cell_min_u = vi & vec3<u32>(~(cell_span - 1u));
                let cell_max_u = cell_min_u + vec3<u32>(cell_span);
                let st = step_dda_box(
                    v_origin,
                    local_d,
                    inv_dir,
                    neg_orig_inv,
                    pos_dir,
                    cell_min_u,
                    cell_max_u,
                    t,
                );
                t = st.t;
                vi = st.vi;
                break;
            }

            let is_leaf = (node.child_ptr & IS_LEAF_FLAG) != 0u;
            if (is_leaf) {
                let cell_span = 1u << shift;
                let cell_min_u = vi & vec3<u32>(~(cell_span - 1u));
                let cell_max_u = cell_min_u + vec3<u32>(cell_span);

                var mat_id = (node.lod_payload >> 24u) & 0xFFu;
                if ((node.child_ptr & IS_UNIFORM_FLAG) == 0u) {
                    mat_id = brick_get_material(node.child_ptr & PTR_INDEX_MASK, b);
                }

                if (mat_id != MAT_AIR) {
                    let enter_coord = select(cell_max_u, cell_min_u, pos_dir);
                    let tmin_v = vec3<f32>(enter_coord) * inv_dir + neg_orig_inv;
                    let cell_enter = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
                    let exact_t = max(cell_enter, 0.0);
                    let hit_p_local_v = v_origin + local_d * exact_t;
                    let hit_p_local_m = hit_p_local_v * VOXEL_SIZE;

                    res.hit = true;
                    res.t_voxels = exact_t;
                    res.hit_pos_meters = (rb.local_to_world * vec4<f32>(hit_p_local_m, 1.0)).xyz;
                    res.prev_hit_pos_meters = (rb.prev_local_to_world * vec4<f32>(hit_p_local_m, 1.0)).xyz;

                    var local_n = vec3<f32>(0.0, 1.0, 0.0);
                    if (tmin_v.x >= tmin_v.y && tmin_v.x >= tmin_v.z) {
                        local_n = vec3<f32>(-sign(local_d.x), 0.0, 0.0);
                    } else if (tmin_v.y >= tmin_v.z) {
                        local_n = vec3<f32>(0.0, -sign(local_d.y), 0.0);
                    } else {
                        local_n = vec3<f32>(0.0, 0.0, -sign(local_d.z));
                    }
                    res.normal = normalize((rb.local_to_world * vec4<f32>(local_n, 0.0)).xyz);
                    res.mat_id = mat_id;
                    res.voxel_hash = pcg_hash(vi.x * 73856093u ^ vi.y * 19349663u ^ vi.z * 83492791u);
                    if (shift == 0u) {
                        res.contact_ao = eval_leaf_bitwise_ao(
                            node.mask_low,
                            node.mask_high,
                            cx,
                            cy,
                            cz,
                            local_n,
                            hit_p_local_v,
                        );
                    }
                    return res;
                }

                let st = step_dda_box(
                    v_origin,
                    local_d,
                    inv_dir,
                    neg_orig_inv,
                    pos_dir,
                    cell_min_u,
                    cell_max_u,
                    t,
                );
                t = st.t;
                vi = st.vi;
                break;
            }

            let next_idx = node_child_offset(node, b);
            node = node_pool[next_idx];
            shift = shift - 2u;

            cached_shift = shift;
            if (shift == 4u) {
                cached_l1_node = node;
            } else if (shift == 2u) {
                cached_l2_node = node;
            } else if (shift == 0u) {
                cached_l3_node = node;
            }
        }
    }

    return res;
}

fn trace_rigid_bodies(
    origin_m: vec3<f32>,
    dir: vec3<f32>,
    max_dist_m: f32,
) -> RigidTraceResult {
    var best: RigidTraceResult;
    best.hit = false;
    best.t_voxels = max_dist_m * VOXELS_PER_METER;
    best.hit_pos_meters = origin_m + dir * max_dist_m;
    best.prev_hit_pos_meters = best.hit_pos_meters;
    best.normal = vec3<f32>(0.0, 1.0, 0.0);
    best.mat_id = MAT_AIR;
    best.voxel_hash = 0u;
    best.contact_ao = 1.0;

    let rb_count = min((u_cam.point_light_count.y >> 16u) & 0xFFFFu, 32u);
    if (rb_count == 0u) {
        return best;
    }

    var cur_max_m = max_dist_m;
    for (var i = 0u; i < rb_count; i = i + 1u) {
        let cand = trace_single_rigid_body(i, origin_m, dir, cur_max_m);
        if (cand.hit && cand.t_voxels < best.t_voxels) {
            best = cand;
            cur_max_m = cand.t_voxels * VOXEL_SIZE;
        }
    }
    return best;
}
