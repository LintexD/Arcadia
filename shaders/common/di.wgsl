// Bound by `arcadia-render` Pass 2 (lighting_restir_gi) and Pass 3 (denoiser_reconstruct).
// Defines ReSTIR DI resources and types for Pass 2 and Pass 3.

const DI_MAX_SOURCES: u32 = 16384u;

// GPU-SYNC: Mirrors `EmissiveSource` in `crates/arcadia-world/src/emissive_registry.rs` (32 bytes, 16-byte aligned).
struct EmissiveSource {
    pos: vec3<f32>,
    radius: f32,
    intensity: vec3<f32>,
    normal_packed: u32,
};

// GPU-SYNC: Mirrors `AliasEntry` in `crates/arcadia-world/src/emissive_registry.rs` (8 bytes, 4-byte aligned).
struct AliasEntry {
    prob: f32,
    alias_idx: u32,
};

// GPU-SYNC: Mirrors `EmissiveParams` in `crates/arcadia-render/src/di.rs` (16 bytes, 16-byte aligned).
struct EmissiveParams {
    count: u32,
    generation: u32,
    total_flux: f32,
    pad: u32,
};

// GPU-SYNC: Mirrors `DiReservoir` in `crates/arcadia-render/src/gbuffer.rs` (16 bytes, 16-byte aligned).
struct DiReservoir {
    // [0..15]: light_idx (0..16383), [16..31]: deterministic sub-cluster jitter seed
    light_idx_seed: u32,
    w_sum: f32,
    m: f32,
    w: f32,
};

struct UnpackedEmitterNormal {
    normal: vec3<f32>,
    eta: f32,
};

fn di_unpack_normal_and_eta(normal_packed: u32) -> UnpackedEmitterNormal {
    let nx = f32(normal_packed & 1023u) / 1023.0 * 2.0 - 1.0;
    let ny = f32((normal_packed >> 10u) & 1023u) / 1023.0 * 2.0 - 1.0;
    let nz = f32((normal_packed >> 20u) & 1023u) / 1023.0 * 2.0 - 1.0;
    let n = vec3<f32>(nx, ny, nz);
    let len = length(n);
    var res: UnpackedEmitterNormal;
    res.eta = clamp(len, 0.0, 1.0);
    res.normal = select(vec3<f32>(0.0, 1.0, 0.0), n / max(len, 1e-4), len > 0.05);
    return res;
}

fn di_unpack_normal(normal_packed: u32) -> vec3<f32> {
    return di_unpack_normal_and_eta(normal_packed).normal;
}

fn di_emitter_surface_pos(centroid: vec3<f32>, normal: vec3<f32>, radius: f32, eta: f32, seed: u32) -> vec3<f32> {
    if (radius <= 0.04) {
        return centroid;
    }
    let u1 = f32(seed & 0xFFu) / 255.0;
    let u2 = f32((seed >> 8u) & 0xFFu) / 255.0;
    // Keep jitter small (within 1-2 voxels = ~0.08m) so samples remain within the physical cluster:
    let max_r = min(radius * 0.20, 0.08);
    if (eta >= 0.35) {
        // Planar surface (wall/slab): jitter in the tangent plane of the exposed surface
        let bs = make_basis(normal);
        let r = sqrt(u1) * max_r;
        let phi = 6.2831853 * u2;
        return centroid + (bs.t * (cos(phi) * r) + bs.b * (sin(phi) * r));
    } else {
        // Omnidirectional cluster (pillar/cube): 3D spherical surface jitter
        let phi = 6.2831853 * u1;
        let cos_theta = u2 * 2.0 - 1.0;
        let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
        let dir = vec3<f32>(cos(phi) * sin_theta, cos_theta, sin(phi) * sin_theta);
        return centroid + dir * max_r;
    }
}

fn create_empty_di_reservoir() -> DiReservoir {
    var res: DiReservoir;
    res.light_idx_seed = 0xFFFFFFFFu;
    res.w_sum = 0.0;
    res.m = 0.0;
    res.w = 0.0;
    return res;
}

fn di_eval_unshadowed(em: EmissiveSource, seed: u32, x_v: vec3<f32>, n_v: vec3<f32>) -> vec3<f32> {
    let un = di_unpack_normal_and_eta(em.normal_packed);
    let em_pos = di_emitter_surface_pos(em.pos, un.normal, em.radius, un.eta, seed);
    let to_l = em_pos - x_v;
    let d2 = dot(to_l, to_l);
    if (d2 < 1e-6 || d2 > 1296.0) {
        return vec3<f32>(0.0);
    }
    let d = sqrt(d2);
    let l_dir = to_l / d;
    let cos_theta_v = max(dot(n_v, l_dir), 0.0);
    if (cos_theta_v <= 0.0) {
        return vec3<f32>(0.0);
    }
    let dir_factor = (1.0 - un.eta) + un.eta * max(dot(un.normal, -l_dir), 0.0);
    let range_fade = clamp(1.0 - d2 / 1296.0, 0.0, 1.0);
    let geom = (cos_theta_v * dir_factor * 0.31830989 * range_fade * range_fade) / max(d2, 0.20);
    return em.intensity * geom;
}
