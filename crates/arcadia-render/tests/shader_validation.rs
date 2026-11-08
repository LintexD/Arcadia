//! Integration test for WGSL shader compilation and SPIR-V bytecode generation.
//! Owns CPU verification of shader compilation, entry points, and pipeline layouts via Naga.

use arcadia_render::{
    BLIT_WGSL, SHADER1_SRC, SHADER2_SRC, SHADER3_SRC, SHADER4_SRC, SHARC_RESOLVE_SRC,
    SHARC_UPDATE_SRC,
};

#[test]
fn test_wgsl_shaders_valid() {
    let shaders = [
        ("primary_gbuffer", SHADER1_SRC),
        ("lighting_restir_gi", SHADER2_SRC),
        ("denoiser_reconstruct", SHADER3_SRC),
        ("composite_dlaa", SHADER4_SRC),
        ("blit", BLIT_WGSL),
        ("sharc_update", SHARC_UPDATE_SRC),
        ("sharc_resolve", SHARC_RESOLVE_SRC),
    ];

    for (name, src) in shaders {
        let parsed = wgpu::naga::front::wgsl::parse_str(src);
        assert!(
            parsed.is_ok(),
            "WGSL parse error in shader '{}': {:?}",
            name,
            parsed.err()
        );
        let module =
            parsed.expect("INVARIANT: Valid WGSL shader source code must parse without errors");
        let mut validator = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        );
        let info = validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("WGSL validation error in shader '{}': {:?}", name, e));

        // WHY: Naga's SPIR-V backend cannot emit ReadZeroSkipWrite around atomics; wgpu's Vulkan backend uses Restrict.
        let atomic_policy = if name.starts_with("sharc") {
            wgpu::naga::proc::BoundsCheckPolicy::Restrict
        } else {
            wgpu::naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite
        };
        let policies = wgpu::naga::proc::BoundsCheckPolicies {
            index: atomic_policy,
            buffer: atomic_policy,
            image_load: wgpu::naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
            binding_array: wgpu::naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
        };
        let spv_options = wgpu::naga::back::spv::Options {
            lang_version: (1, 3),
            flags: wgpu::naga::back::spv::WriterFlags::DEBUG
                | wgpu::naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE
                | wgpu::naga::back::spv::WriterFlags::CLAMP_FRAG_DEPTH,
            bounds_check_policies: policies,
            zero_initialize_workgroup_memory:
                wgpu::naga::back::spv::ZeroInitializeWorkgroupMemoryMode::Polyfill,
            ..Default::default()
        };

        for ep in &module.entry_points {
            let pipe_opts = wgpu::naga::back::spv::PipelineOptions {
                entry_point: ep.name.clone(),
                shader_stage: ep.stage,
            };
            let spv_words =
                wgpu::naga::back::spv::write_vec(&module, &info, &spv_options, Some(&pipe_opts))
                    .unwrap_or_else(|e| {
                        panic!("SPIR-V codegen failed for {}::{}: {:?}", name, ep.name, e)
                    });

            let mut defined_vars = std::collections::HashSet::new();
            let mut pc = 5usize;
            while pc < spv_words.len() {
                let word0 = spv_words[pc];
                let word_count = (word0 >> 16) as usize;
                let opcode = word0 & 0xFFFF;
                if word_count == 0 {
                    break;
                }
                if opcode == 59 || opcode == 55 || opcode == 65 || opcode == 66 {
                    defined_vars.insert(spv_words[pc + 2]);
                }
                if opcode == 62 {
                    let ptr_id = spv_words[pc + 1];
                    let obj_id = spv_words[pc + 2];
                    assert!(
                        defined_vars.contains(&ptr_id),
                        "Undefined pointer ID %{} in OpStore %{} %{} in shader {}::{}",
                        ptr_id,
                        ptr_id,
                        obj_id,
                        name,
                        ep.name
                    );
                }
                pc += word_count;
            }
        }
    }
}
