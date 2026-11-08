//! 256-entry PBR + Emissive Material Palette designed for the John Lin
//! "Fantasy Nature & Ruins" aesthetic. Owns CPU material definitions and GPU palette mirroring.

use bytemuck::{Pod, Zeroable};

pub type MaterialId = u8;

pub const MAT_AIR: MaterialId = 0;
pub const MAT_BEDROCK: MaterialId = 1;
pub const MAT_STONE_DEEP: MaterialId = 2;
pub const MAT_LIMESTONE: MaterialId = 3;
pub const MAT_LIMESTONE_WEATHERED: MaterialId = 4;
pub const MAT_SLATE_RUIN: MaterialId = 5;
pub const MAT_SLATE_CARVED: MaterialId = 6;
pub const MAT_DIRT: MaterialId = 7;
pub const MAT_GRASS_LUSH: MaterialId = 8;
pub const MAT_GRASS_GOLDEN: MaterialId = 9;
pub const MAT_MOSS_VIBRANT: MaterialId = 10;
pub const MAT_MOSS_DARK: MaterialId = 11;
pub const MAT_SAND_RIVER: MaterialId = 12;
pub const MAT_WOOD_OAK_BARK: MaterialId = 13;
pub const MAT_WOOD_TIMBER: MaterialId = 14;
pub const MAT_LEAVES_EMERALD: MaterialId = 15;
pub const MAT_LEAVES_AUTUMN: MaterialId = 16;
pub const MAT_LEAVES_CHERRY: MaterialId = 17;
pub const MAT_WATER_CRYSTAL: MaterialId = 18;
pub const MAT_CRYSTAL_CYAN: MaterialId = 19;
pub const MAT_CRYSTAL_MAGENTA: MaterialId = 20;
pub const MAT_CRYSTAL_AMBER: MaterialId = 21;
pub const MAT_LAVA_MAGMA: MaterialId = 22;
pub const MAT_SCORCHED_ROCK: MaterialId = 23;
pub const MAT_EMBER_GLOW: MaterialId = 24;
pub const MAT_MARBLE_TEMPLE: MaterialId = 25;
pub const MAT_GOLD_TRIM: MaterialId = 26;
pub const MAT_POINT_LIGHT: MaterialId = 27;
pub const MAT_MIRROR_CHROME: MaterialId = 28;

/// GPU-aligned material descriptor (32 bytes, `vec4` aligned for WGSL uniform/storage arrays).
// GPU-SYNC: Mirrors `Material` in `shaders/common/types.wgsl` (@group(0) @binding(4), 32 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Material {
    /// Linear RGB base color + roughness in `w`.
    pub albedo_roughness: [f32; 4],
    /// Linear RGB emission color (for crystals/lava/embers) + metallic/variation in `w`.
    pub emission_meta: [f32; 4],
}

impl Material {
    // WHY: Full PBR material specification requires albedo, roughness, emission, and scattering metadata.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        r: f32,
        g: f32,
        b: f32,
        roughness: f32,
        em_r: f32,
        em_g: f32,
        em_b: f32,
        color_variance: f32,
    ) -> Self {
        Self {
            albedo_roughness: [r, g, b, roughness],
            emission_meta: [em_r, em_g, em_b, color_variance],
        }
    }

    /// Packs RGB8 average color + emissive flag into a 32-bit `lod_payload` word for `Node64`.
    #[inline]
    pub fn pack_lod_payload(&self, mat_id: MaterialId) -> u32 {
        let r = (self.albedo_roughness[0].clamp(0.0, 1.0) * 255.0) as u32;
        let g = (self.albedo_roughness[1].clamp(0.0, 1.0) * 255.0) as u32;
        let b = (self.albedo_roughness[2].clamp(0.0, 1.0) * 255.0) as u32;
        (mat_id as u32) << 24 | (b << 16) | (g << 8) | r
    }

    #[inline]
    pub fn is_emissive(&self) -> bool {
        self.emission_meta[0] > 0.01 || self.emission_meta[1] > 0.01 || self.emission_meta[2] > 0.01
    }
}

/// Full 256-entry material table uploaded to the GPU.
// GPU-SYNC: Mirrors `material_palette` in `shaders/common/types.wgsl` (@group(0) @binding(4), 8192 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MaterialPalette {
    pub materials: [Material; 256],
}

impl Default for MaterialPalette {
    fn default() -> Self {
        Self::fantasy_nature_palette()
    }
}

impl MaterialPalette {
    /// Builds the curated John Lin "Fantasy Nature & Ruins" PBR + Emissive palette.
    pub fn fantasy_nature_palette() -> Self {
        let mut materials = [Material::default(); 256];

        // 0: Air
        materials[MAT_AIR as usize] = Material::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        // 1: Bedrock (dark basalt)
        materials[MAT_BEDROCK as usize] =
            Material::new(0.10, 0.11, 0.13, 0.90, 0.0, 0.0, 0.0, 0.03);
        // 2: Deep stone (cool grey-blue granite)
        materials[MAT_STONE_DEEP as usize] =
            Material::new(0.28, 0.30, 0.34, 0.85, 0.0, 0.0, 0.0, 0.05);
        // 3: Warm limestone (classic cliff & ruin stone)
        materials[MAT_LIMESTONE as usize] =
            Material::new(0.58, 0.54, 0.47, 0.80, 0.0, 0.0, 0.0, 0.06);
        // 4: Weathered limestone
        materials[MAT_LIMESTONE_WEATHERED as usize] =
            Material::new(0.46, 0.45, 0.40, 0.88, 0.0, 0.0, 0.0, 0.08);
        // 5: Ancient ruin slate (blue-grey dressed stone)
        materials[MAT_SLATE_RUIN as usize] =
            Material::new(0.36, 0.40, 0.45, 0.72, 0.0, 0.0, 0.0, 0.05);
        // 6: Carved ruin slate trim
        materials[MAT_SLATE_CARVED as usize] =
            Material::new(0.48, 0.52, 0.57, 0.65, 0.0, 0.0, 0.0, 0.04);
        // 7: Rich earth / soil
        materials[MAT_DIRT as usize] = Material::new(0.34, 0.23, 0.15, 0.92, 0.0, 0.0, 0.0, 0.06);
        // 8: Lush meadow grass
        materials[MAT_GRASS_LUSH as usize] =
            Material::new(0.22, 0.52, 0.18, 0.78, 0.0, 0.0, 0.0, 0.08);
        // 9: Sunlit golden highland grass
        materials[MAT_GRASS_GOLDEN as usize] =
            Material::new(0.46, 0.58, 0.20, 0.78, 0.0, 0.0, 0.0, 0.08);
        // 10: Vibrant ruin moss (creeping over stone)
        materials[MAT_MOSS_VIBRANT as usize] =
            Material::new(0.16, 0.46, 0.22, 0.85, 0.0, 0.0, 0.0, 0.09);
        // 11: Dark forest moss
        materials[MAT_MOSS_DARK as usize] =
            Material::new(0.11, 0.32, 0.15, 0.88, 0.0, 0.0, 0.0, 0.06);
        // 12: Riverbed sand / pebbles
        materials[MAT_SAND_RIVER as usize] =
            Material::new(0.68, 0.60, 0.46, 0.82, 0.0, 0.0, 0.0, 0.07);
        // 13: Ancient oak bark
        materials[MAT_WOOD_OAK_BARK as usize] =
            Material::new(0.27, 0.18, 0.12, 0.88, 0.0, 0.0, 0.0, 0.06);
        // 14: Hewn timber
        materials[MAT_WOOD_TIMBER as usize] =
            Material::new(0.48, 0.33, 0.20, 0.75, 0.0, 0.0, 0.0, 0.05);
        // 15: Emerald tree foliage
        materials[MAT_LEAVES_EMERALD as usize] =
            Material::new(0.15, 0.44, 0.18, 0.70, 0.0, 0.0, 0.0, 0.10);
        // 16: Golden-orange autumn foliage
        materials[MAT_LEAVES_AUTUMN as usize] =
            Material::new(0.76, 0.36, 0.11, 0.70, 0.0, 0.0, 0.0, 0.10);
        // 17: Blossom pink foliage
        materials[MAT_LEAVES_CHERRY as usize] =
            Material::new(0.85, 0.48, 0.58, 0.72, 0.0, 0.0, 0.0, 0.08);
        // 18: Crystal lagoon water
        materials[MAT_WATER_CRYSTAL as usize] =
            Material::new(0.06, 0.28, 0.38, 0.05, 0.0, 0.0, 0.0, 0.02);
        // 19: Glowing Cyan Crystal (rich chromatic emission without white clipping)
        materials[MAT_CRYSTAL_CYAN as usize] =
            Material::new(0.15, 0.85, 1.00, 0.15, 0.20, 2.40, 3.60, 0.05);
        // 20: Glowing Magenta Crystal
        materials[MAT_CRYSTAL_MAGENTA as usize] =
            Material::new(0.95, 0.20, 0.80, 0.15, 3.20, 0.25, 2.50, 0.05);
        // 21: Glowing Amber Sunstone Crystal
        materials[MAT_CRYSTAL_AMBER as usize] =
            Material::new(1.00, 0.68, 0.15, 0.15, 3.40, 1.75, 0.15, 0.05);
        // 22: Molten Magma / Lava
        materials[MAT_LAVA_MAGMA as usize] =
            Material::new(1.00, 0.30, 0.04, 0.30, 4.20, 1.10, 0.08, 0.10);
        // 23: Scorched blast rock (explosion crater rim)
        materials[MAT_SCORCHED_ROCK as usize] =
            Material::new(0.10, 0.09, 0.08, 0.95, 0.0, 0.0, 0.0, 0.05);
        // 24: Glowing hot ember (explosion rim)
        materials[MAT_EMBER_GLOW as usize] =
            Material::new(0.95, 0.25, 0.04, 0.50, 3.20, 0.75, 0.05, 0.12);
        // 25: Ivory temple marble
        materials[MAT_MARBLE_TEMPLE as usize] =
            Material::new(0.82, 0.80, 0.75, 0.35, 0.0, 0.0, 0.0, 0.03);
        // 26: Ancient gilded bronze/gold trim
        materials[MAT_GOLD_TRIM as usize] =
            Material::new(0.85, 0.68, 0.24, 0.25, 0.0, 0.0, 0.0, 0.04);
        // 27: Ultra-Bright Solar Beacon / Emissive Light Source (key 7)
        materials[MAT_POINT_LIGHT as usize] =
            Material::new(1.00, 0.97, 0.90, 0.05, 116.00, 114.50, 111.80, 0.02);
        // 28: Polished Mirror Chrome (key 8)
        materials[MAT_MIRROR_CHROME as usize] =
            Material::new(0.92, 0.95, 0.98, 0.02, 0.0, 0.0, 0.0, 0.01);

        // Fill remaining slots with procedural pastel/stone variations so any ID is valid
        for (i, mat) in materials.iter_mut().enumerate().skip(29) {
            let t = (i as f32) / 255.0;
            *mat = Material::new(
                0.3 + 0.4 * (t * std::f32::consts::TAU).sin().abs(),
                0.3 + 0.4 * (t * 3.0 * std::f32::consts::PI).cos().abs(),
                0.3 + 0.3 * t,
                0.75,
                0.0,
                0.0,
                0.0,
                0.05,
            );
        }

        Self { materials }
    }

    #[inline(always)]
    pub fn get(&self, id: MaterialId) -> &Material {
        &self.materials[id as usize]
    }
}
