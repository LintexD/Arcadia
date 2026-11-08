//! `arcadia-world`: Multi-chunk world management, parallel "Fantasy Nature & Ruins"
//! procedural generation, real-time sphere/beam destruction, incremental GPU buffer patching,
//! and bitwise structural island detection.
//! Owns high-level CPU world orchestration and chunk streaming state.

pub mod benchmark_scene;
pub mod destruction;
pub mod emissive_registry;
pub mod generator;
pub mod islands;
pub mod world_map;

pub use benchmark_scene::{
    generate_benchmark_world, get_benchmark_camera_path, BenchmarkCameraKeyframe,
    BenchmarkCameraPath,
};
pub use destruction::{DebrisParticle, DestructionEvent, MAX_DEBRIS_PARTICLES};
pub use emissive_registry::{
    build_alias_table, AliasEntry, EmissiveRegistry, EmissiveSource, MAX_EMISSIVE_SOURCES,
};
pub use generator::generate_fantasy_ruins_world;
pub use islands::{
    detect_disconnected_island_components, detect_disconnected_islands,
    detect_disconnected_islands_from_carve, is_externally_anchored, VoxelQuery,
    WorldRegionSnapshot,
};
pub use world_map::{GpuPatch, WorldMap, EMPTY_CHUNK_SENTINEL};
