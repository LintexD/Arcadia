// Read-only SHARC lookup (`query_sharc`) bound by Pass 2 (lighting_restir_gi.wgsl) and sharc_update.wgsl.
// Expected inputs: @group(2) @binding(0) `sharc_resolved` (storage, read); requires common/sharc.wgsl.

// GPU-SYNC: Mirrors `SharcResources::resolved_buf` in `crates/arcadia-render/src/sharc.rs` (@group(2) @binding(0), 2.0 MB).
@group(2) @binding(0) var<storage, read> sharc_resolved: array<SharcResolvedSlot, 131072>;

// Probes the four slots of one bucket (a single 64-byte cache line); returns (radiance, 1.0) on hit, else 0.
fn sharc_lookup(k: SharcKey) -> vec4<f32> {
    let base = k.bucket * SHARC_WAYS;
    for (var i = 0u; i < SHARC_WAYS; i = i + 1u) {
        let slot = sharc_resolved[base + i];
        if (slot.key == k.key) {
            return vec4<f32>(sharc_unpack_radiance(slot), 1.0);
        }
    }
    return vec4<f32>(0.0);
}

// Returns cached outgoing diffuse radiance at a surface hit: xyz = radiance, w = 1.0 on hit and 0.0 on miss.
// A jittered probe dissolves cell seams; a miss falls back to the unjittered anchor to avoid edge speckles.
fn query_sharc(pos: vec3<f32>, normal: vec3<f32>, cam_pos: vec3<f32>, tangent_jitter: vec2<f32>) -> vec4<f32> {
    if (tangent_jitter.x != 0.0 || tangent_jitter.y != 0.0) {
        let jittered = sharc_lookup(sharc_make_key(pos, normal, cam_pos, tangent_jitter));
        if (jittered.w > 0.5) {
            return jittered;
        }
    }
    return sharc_lookup(sharc_make_key(pos, normal, cam_pos, vec2<f32>(0.0)));
}

