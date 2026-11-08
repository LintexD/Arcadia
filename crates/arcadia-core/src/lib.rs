//! `arcadia-core`: Foundational types, bitwise 64-tree math, material palette,
//! and CPU reference DDA raycaster for the Arcadia (1/16 m)³ voxel engine.
//! Owns core CPU definitions and memory layouts.

pub mod chunk_tree;
pub mod coord;
pub mod material;
pub mod mtf_cache;
pub mod node64;
pub mod raycast_cpu;
pub mod slab;

pub use bytemuck;
pub use chunk_tree::{
    ChunkEditReport, ChunkTree64, LEVEL0_SCALE, LEVEL1_SCALE, LEVEL2_SCALE, LEVEL3_SCALE,
};
pub use coord::{
    child_bit_index, child_bit_mask, unpack_child_index, ChunkCoord, VoxelCoord, CHUNK_DIM,
    VOXELS_PER_METER, VOXEL_SIZE,
};
pub use material::{
    Material, MaterialId, MaterialPalette, MAT_AIR, MAT_BEDROCK, MAT_CRYSTAL_AMBER,
    MAT_CRYSTAL_CYAN, MAT_CRYSTAL_MAGENTA, MAT_DIRT, MAT_EMBER_GLOW, MAT_GOLD_TRIM,
    MAT_GRASS_GOLDEN, MAT_GRASS_LUSH, MAT_LAVA_MAGMA, MAT_LEAVES_AUTUMN, MAT_LEAVES_CHERRY,
    MAT_LEAVES_EMERALD, MAT_LIMESTONE, MAT_LIMESTONE_WEATHERED, MAT_MARBLE_TEMPLE,
    MAT_MIRROR_CHROME, MAT_MOSS_DARK, MAT_MOSS_VIBRANT, MAT_POINT_LIGHT, MAT_SAND_RIVER,
    MAT_SCORCHED_ROCK, MAT_SLATE_CARVED, MAT_SLATE_RUIN, MAT_STONE_DEEP, MAT_WATER_CRYSTAL,
    MAT_WOOD_OAK_BARK, MAT_WOOD_TIMBER,
};
pub use mtf_cache::MtfCache;
pub use node64::{LeafBrick64, Node64, IS_LEAF_FLAG, IS_UNIFORM_FLAG, PTR_INDEX_MASK};
pub use raycast_cpu::{raycast_chunk_tree, RayHit};
pub use slab::SlabAllocator;
