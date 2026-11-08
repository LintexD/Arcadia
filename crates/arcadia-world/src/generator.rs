//! Parallel procedural generator for the John Lin "Fantasy Nature & Ruins" world
//! at (1/16 m)³ = 6.25 cm voxel resolution.
//! Owns CPU parallel procedural chunk terrain and ruins generation.
//!
//! Features generated at 6.25 cm micro-voxel precision:
//! 1. **Terraced Valley & Meadows**: Multi-octave FBM hills with natural stone ledges, mossy banks,
//!    and a crystal-clear central lagoon.
//! 2. **Ancient Stone Temple Ruins**: Walkable stone staircases (3 voxels = 18.75 cm step height),
//!    fluted columns with carved capitals, vaulted stone archways, broken courtyard walls,
//!    and glowing Cyan/Amber crystal monoliths + braziers.
//! 3. **Glowing Crystal Grotto & Magma Sanctuary**: An atmospheric cliffside cavern beneath the
//!    northern hill lit by giant cyan/magenta/amber crystal spires and a glowing lava fissure.
//! 4. **Lush Fantasy Trees**: Gnarled oak trunks with layered emerald, golden-autumn, and cherry canopies.

use crate::world_map::WorldMap;
use arcadia_core::{
    ChunkTree64, MaterialId, MaterialPalette, CHUNK_DIM, MAT_AIR, MAT_BEDROCK, MAT_CRYSTAL_AMBER,
    MAT_CRYSTAL_CYAN, MAT_CRYSTAL_MAGENTA, MAT_DIRT, MAT_GOLD_TRIM, MAT_GRASS_GOLDEN,
    MAT_GRASS_LUSH, MAT_LAVA_MAGMA, MAT_LEAVES_AUTUMN, MAT_LEAVES_CHERRY, MAT_LEAVES_EMERALD,
    MAT_LIMESTONE, MAT_LIMESTONE_WEATHERED, MAT_MARBLE_TEMPLE, MAT_MOSS_DARK, MAT_MOSS_VIBRANT,
    MAT_SAND_RIVER, MAT_SLATE_CARVED, MAT_SLATE_RUIN, MAT_STONE_DEEP, MAT_WATER_CRYSTAL,
    MAT_WOOD_OAK_BARK, VOXEL_SIZE,
};
use glam::{UVec3, Vec2, Vec3};
use rayon::prelude::*;

/// Generates the complete multi-chunk Fantasy Nature & Ruins world in parallel using `rayon`.
///
/// Default grid `UVec3::new(10, 4, 10)` covers `160m × 64m × 160m` = `2560 × 1024 × 2560`
/// ($\approx 6.71\text{ billion}$ potential $(1/16\text{ m})^3$ voxels!).
pub fn generate_fantasy_ruins_world(grid_dims: UVec3) -> WorldMap {
    let palette = MaterialPalette::fantasy_nature_palette();
    let total_chunks = (grid_dims.x * grid_dims.y * grid_dims.z) as usize;

    let world_size_m = grid_dims.as_vec3() * (CHUNK_DIM as f32 * VOXEL_SIZE);
    let center_xz = Vec2::new(world_size_m.x * 0.5, world_size_m.z * 0.5);

    // Precompute deterministic fantasy tree positions around the valley:
    let trees = build_tree_instances(center_xz, world_size_m);

    let chunks: Vec<ChunkTree64> = (0..total_chunks)
        .into_par_iter()
        .map(|flat_idx| {
            let cx = (flat_idx as u32) % grid_dims.x;
            let cy = ((flat_idx as u32) / grid_dims.x) % grid_dims.y;
            let cz = (flat_idx as u32) / (grid_dims.x * grid_dims.y);

            let chunk_base_vx = cx * CHUNK_DIM;
            let chunk_base_vy = cy * CHUNK_DIM;
            let chunk_base_vz = cz * CHUNK_DIM;

            // Filter trees whose bounding box overlaps this 16m × 16m × 16m chunk:
            let chunk_min_m = Vec3::new(
                chunk_base_vx as f32,
                chunk_base_vy as f32,
                chunk_base_vz as f32,
            ) * VOXEL_SIZE;
            let chunk_max_m = chunk_min_m + Vec3::splat(16.0);

            let local_trees: Vec<TreeInstance> = trees
                .iter()
                .copied()
                .filter(|t| {
                    t.base_pos.x + t.canopy_radius >= chunk_min_m.x
                        && t.base_pos.x - t.canopy_radius <= chunk_max_m.x
                        && t.base_pos.z + t.canopy_radius >= chunk_min_m.z
                        && t.base_pos.z - t.canopy_radius <= chunk_max_m.z
                        && t.base_pos.y + t.height + t.canopy_radius >= chunk_min_m.y
                        && t.base_pos.y - 1.0 <= chunk_max_m.y
                })
                .collect();

            ChunkTree64::from_generator(
                &palette,
                |ox, oy, oz, size| {
                    // Only apply coarse region skips at 64³ (4m) or smaller blocks so 16m chunk roots
                    // never clip local ridge peaks or terraced slopes:
                    if size > 64 {
                        return None;
                    }

                    let bmin_m = Vec3::new(
                        (chunk_base_vx + ox) as f32,
                        (chunk_base_vy + oy) as f32,
                        (chunk_base_vz + oz) as f32,
                    ) * VOXEL_SIZE;
                    let bmax_m = bmin_m + Vec3::splat(size as f32 * VOXEL_SIZE);

                    // Compute conservative 3x3 min/max terrain height over this XY footprint:
                    let (min_h, max_h) =
                        estimate_terrain_bounds(bmin_m.x, bmin_m.z, bmax_m.x, bmax_m.z, center_xz);

                    // Check if any ruin, grotto, or tree overlaps this block:
                    let near_ruin = (bmax_m.x >= center_xz.x - 28.0
                        && bmin_m.x <= center_xz.x + 28.0)
                        && (bmax_m.z >= center_xz.y - 28.0 && bmin_m.z <= center_xz.y + 32.0)
                        && bmin_m.y <= 34.0;

                    let near_grotto = (bmax_m.x >= center_xz.x - 22.0
                        && bmin_m.x <= center_xz.x + 22.0)
                        && (bmax_m.z >= center_xz.y + 10.0 && bmin_m.z <= center_xz.y + 46.0)
                        && bmin_m.y <= 28.0
                        && bmax_m.y >= 6.0;

                    let near_tree = local_trees.iter().any(|t| {
                        bmax_m.x >= t.base_pos.x - t.canopy_radius
                            && bmin_m.x <= t.base_pos.x + t.canopy_radius
                            && bmax_m.z >= t.base_pos.z - t.canopy_radius
                            && bmin_m.z <= t.base_pos.z + t.canopy_radius
                            && bmax_m.y >= t.base_pos.y
                            && bmin_m.y <= t.base_pos.y + t.height + t.canopy_radius
                    });

                    // 1. High sky skip (must also be above the 10.8m water lagoon level!):
                    if bmin_m.y > max_h.max(11.1) + 0.5 && !near_ruin && !near_tree {
                        return Some(MAT_AIR);
                    }

                    // 2. Deep solid rock / limestone collapse (dramatically accelerates generation & cuts VRAM by 10x!):
                    if bmax_m.y < min_h - 0.65 && bmin_m.y > 0.3 && !near_grotto && !near_ruin {
                        if bmin_m.y > min_h - 4.0 {
                            return Some(MAT_LIMESTONE);
                        } else {
                            return Some(MAT_STONE_DEEP);
                        }
                    }

                    None
                },
                |lx, ly, lz| {
                    let gx = chunk_base_vx + lx;
                    let gy = chunk_base_vy + ly;
                    let gz = chunk_base_vz + lz;
                    let pos_m = Vec3::new(
                        (gx as f32 + 0.5) * VOXEL_SIZE,
                        (gy as f32 + 0.5) * VOXEL_SIZE,
                        (gz as f32 + 0.5) * VOXEL_SIZE,
                    );
                    sample_world_voxel(pos_m, gx, gy, gz, center_xz, &local_trees)
                },
            )
        })
        .collect();

    WorldMap::from_chunks(grid_dims, chunks, palette)
}

#[derive(Clone, Copy)]
struct TreeInstance {
    base_pos: Vec3,
    height: f32,
    trunk_radius: f32,
    canopy_radius: f32,
    leaf_mat: MaterialId,
}

fn build_tree_instances(center_xz: Vec2, world_size_m: Vec3) -> Vec<TreeInstance> {
    let mut trees = Vec::new();
    let ring_radii = [23.0f32, 34.0, 46.0, 58.0];
    let leaf_cycle = [
        MAT_LEAVES_EMERALD,
        MAT_LEAVES_EMERALD,
        MAT_LEAVES_AUTUMN,
        MAT_LEAVES_CHERRY,
    ];

    for (r_idx, &radius) in ring_radii.iter().enumerate() {
        let count = 10 + r_idx * 6;
        for i in 0..count {
            let angle = (i as f32 / count as f32) * std::f32::consts::TAU + (r_idx as f32 * 0.7);
            let jitter_r = radius + (hash2(i as i32, r_idx as i32) * 6.0 - 3.0);
            let wx = center_xz.x + angle.cos() * jitter_r;
            let wz = center_xz.y + angle.sin() * jitter_r;

            if wx < 12.0 || wx > world_size_m.x - 12.0 || wz < 12.0 || wz > world_size_m.z - 12.0 {
                continue;
            }
            // Keep the southern approach path and northern grotto entrance clear:
            if (wx - center_xz.x).abs() < 6.0 && wz < center_xz.y {
                continue;
            }
            if (wx - center_xz.x).abs() < 10.0 && wz > center_xz.y + 12.0 && wz < center_xz.y + 32.0
            {
                continue;
            }

            let ground_y = eval_terrain_height(wx, wz, center_xz);
            if ground_y < 11.2 {
                // Skip underwater lagoon
                continue;
            }

            let h_var = hash2(i as i32 + 31, r_idx as i32 + 17);
            let height = 6.0 + h_var * 3.5;
            let canopy_radius = 3.2 + h_var * 1.6;
            let trunk_radius = 0.42 + h_var * 0.18;
            let leaf_mat = leaf_cycle[(i + r_idx) % leaf_cycle.len()];

            trees.push(TreeInstance {
                base_pos: Vec3::new(wx, ground_y - 0.3, wz),
                height,
                trunk_radius,
                canopy_radius,
                leaf_mat,
            });
        }
    }
    trees
}

/// Evaluates the procedural terraced fantasy terrain height (in meters) at `(wx, wz)`.
#[inline]
pub fn eval_terrain_height(wx: f32, wz: f32, center_xz: Vec2) -> f32 {
    let rel = Vec2::new(wx - center_xz.x, wz - center_xz.y);
    let dist_center = rel.length();

    // Central sanctuary plateau at y = 12.0m so the ruins and player spawn sit on level ground:
    let plateau_blend = ((dist_center - 16.0) / 14.0).clamp(0.0, 1.0);
    let smooth_plateau = plateau_blend * plateau_blend * (3.0 - 2.0 * plateau_blend);

    // Scenic lagoon depression to the west-southwest of the ruins:
    let lagoon_center = center_xz + Vec2::new(-18.0, -10.0);
    let d_lagoon = Vec2::new(wx, wz).distance(lagoon_center);
    let lagoon_dip = if d_lagoon < 14.0 {
        let t = 1.0 - (d_lagoon / 14.0);
        -3.2 * t * t
    } else {
        0.0
    };

    // Multi-octave 2D value noise for rolling fantasy hills:
    let n1 = fbm2(wx * 0.022, wz * 0.022, 3);
    let n2 = value_noise2(wx * 0.08, wz * 0.08);

    // Northern ridge rises higher to house the Crystal Grotto cliffside:
    let north_rise = ((wz - center_xz.y - 8.0).max(0.0) * 0.38).min(16.0);

    // Smooth natural rolling hills and lagoon basin (no artificial 0.75m stair-step contour ridges
    // that cause swirling shadow bands under point lights or concentric wedding-cake rings underwater):
    let raw_hills = 12.0 + (n1 * 11.0 + n2 * 2.2 + north_rise) * smooth_plateau + lagoon_dip;

    raw_hills.clamp(4.0, 44.0)
}

fn estimate_terrain_bounds(
    min_x: f32,
    min_z: f32,
    max_x: f32,
    max_z: f32,
    center_xz: Vec2,
) -> (f32, f32) {
    let mid_x = 0.5 * (min_x + max_x);
    let mid_z = 0.5 * (min_z + max_z);
    let samples = [
        eval_terrain_height(min_x, min_z, center_xz),
        eval_terrain_height(mid_x, min_z, center_xz),
        eval_terrain_height(max_x, min_z, center_xz),
        eval_terrain_height(min_x, mid_z, center_xz),
        eval_terrain_height(mid_x, mid_z, center_xz),
        eval_terrain_height(max_x, mid_z, center_xz),
        eval_terrain_height(min_x, max_z, center_xz),
        eval_terrain_height(mid_x, max_z, center_xz),
        eval_terrain_height(max_x, max_z, center_xz),
    ];
    let mut min_h = f32::MAX;
    let mut max_h = f32::MIN;
    for s in samples {
        min_h = min_h.min(s);
        max_h = max_h.max(s);
    }
    let margin = (max_x - min_x) * 0.65 + 1.15;
    (min_h - margin, max_h + margin)
}

#[inline]
fn sample_world_voxel(
    pos_m: Vec3,
    gx: u32,
    gy: u32,
    gz: u32,
    center_xz: Vec2,
    local_trees: &[TreeInstance],
) -> MaterialId {
    if gy < 4 {
        return MAT_BEDROCK;
    }

    let wx = pos_m.x;
    let wy = pos_m.y;
    let wz = pos_m.z;

    let ground_h = eval_terrain_height(wx, wz, center_xz);
    let voxel_hash =
        (gx.wrapping_mul(73856093)) ^ (gy.wrapping_mul(19349663)) ^ (gz.wrapping_mul(83492791));

    // -------------------------------------------------------------------------
    // 1. CLIFFSIDE CRYSTAL GROTTO & MAGMA SANCTUARY (North of the central ruins)
    // -------------------------------------------------------------------------
    let grotto_center = Vec3::new(center_xz.x, 13.5, center_xz.y + 26.0);
    let grotto_rel = Vec3::new(
        (wx - grotto_center.x) / 12.0,
        (wy - grotto_center.y) / 5.5,
        (wz - grotto_center.z) / 11.0,
    );
    let grotto_dist = grotto_rel.length();

    // Tunnel entrance connecting the ruin courtyard (z = center_z + 12) into the grotto:
    let in_tunnel = (wx - center_xz.x).abs() < 3.2
        && wz >= center_xz.y + 10.0
        && wz <= grotto_center.z
        && wy >= 12.0
        && wy <= 16.5 - ((wx - center_xz.x).abs() * 0.35);

    // Open the southern amphitheater mouth (`wz < grotto_center.z - 1.5`) into the courtyard while
    // keeping the deep northern cavern roof (`wz >= grotto_center.z - 1.5`) at least 1.8m under the hill:
    let roof_ok = wz < grotto_center.z - 1.5 || wy < ground_h - 1.8;
    if (grotto_dist < 1.0 && roof_ok) || in_tunnel {
        // Lava pool in the deep northern basin of the grotto:
        if wy < 10.2 && grotto_dist < 0.85 && wz > grotto_center.z + 2.0 {
            return MAT_LAVA_MAGMA;
        }
        // Walkable stone bridge / floor inside the grotto at y = 11.5..12.0:
        if wy < 11.8 && (wx - center_xz.x).abs() < 3.5 && wz <= grotto_center.z + 2.5 {
            return if (voxel_hash & 7) == 0 {
                MAT_MOSS_DARK
            } else {
                MAT_SLATE_RUIN
            };
        }

        // Giant glowing Crystal Spires inside the grotto!
        let spire_offsets = [
            (Vec2::new(-5.5, -2.0), MAT_CRYSTAL_CYAN, 4.2f32),
            (Vec2::new(5.5, -1.5), MAT_CRYSTAL_MAGENTA, 4.4f32),
            (Vec2::new(-4.2, 4.5), MAT_CRYSTAL_AMBER, 4.8f32),
            (Vec2::new(4.8, 4.2), MAT_CRYSTAL_CYAN, 4.6f32),
            (Vec2::new(0.0, 6.5), MAT_CRYSTAL_MAGENTA, 5.2f32),
        ];
        for &(offset, c_mat, spire_h) in &spire_offsets {
            let sx = grotto_center.x + offset.x;
            let sz = grotto_center.z + offset.y;
            // Tilted hexagonal-like crystal spire:
            let tilt_x = (wy - 11.0) * 0.18 * offset.x.signum();
            let dx = wx - (sx + tilt_x);
            let dz = wz - sz;
            let r = (dx * dx + dz * dz).sqrt();
            let max_r = (1.15 - (wy - 10.5) / spire_h).max(0.0);
            if wy >= 10.0 && wy <= 10.5 + spire_h && r < max_r {
                return c_mat;
            }
        }

        // Ceiling stalactite glowing crystal clusters:
        let ceil_y = grotto_center.y
            + 5.2
                * (1.0 - (grotto_rel.x * grotto_rel.x + grotto_rel.z * grotto_rel.z))
                    .max(0.0)
                    .sqrt();
        if wy > ceil_y - 1.1 && wy <= ceil_y {
            let cell_x = (wx * 0.7).floor() as i32;
            let cell_z = (wz * 0.7).floor() as i32;
            if ((cell_x * 31 + cell_z * 17) & 7) == 0 {
                let cx_l = (wx * 0.7).fract() - 0.5;
                let cz_l = (wz * 0.7).fract() - 0.5;
                if cx_l * cx_l + cz_l * cz_l < 0.08 {
                    return MAT_CRYSTAL_CYAN;
                }
            }
        }

        if wy >= 11.8 {
            return MAT_AIR;
        }
    }

    // -------------------------------------------------------------------------
    // 2. ANCIENT STONE TEMPLE RUINS & WALKABLE STAIRCASES (Central Sanctuary)
    // -------------------------------------------------------------------------
    if let Some(ruin_mat) = sample_temple_ruins(pos_m, center_xz, voxel_hash) {
        return ruin_mat;
    }

    // -------------------------------------------------------------------------
    // 3. LUSH FANTASY TREES (Trunks, Roots & Layered Canopies)
    // -------------------------------------------------------------------------
    for tree in local_trees {
        if let Some(tree_mat) = sample_tree(pos_m, tree, voxel_hash) {
            return tree_mat;
        }
    }

    // -------------------------------------------------------------------------
    // 4. NATURAL TERRAIN STRATA & WATER LAGOON
    // -------------------------------------------------------------------------
    let water_level = 10.8f32;
    if wy <= ground_h {
        let depth = ground_h - wy;
        // Make top grass layer 0.90m thick (thicker than the 0.75m terrace step) so sloped
        // terraced hillsides stay lush green without exposing thin dirt/limestone contour bands:
        if depth < 0.90 {
            if ground_h < water_level + 0.35 {
                return MAT_SAND_RIVER;
            }
            // Blend lush meadow grass, golden sunlit grass, and vibrant moss:
            let biome_n = value_noise2(wx * 0.09, wz * 0.09);
            if biome_n > 0.68 {
                return MAT_GRASS_GOLDEN;
            } else if biome_n < 0.25 {
                return MAT_MOSS_VIBRANT;
            } else {
                return MAT_GRASS_LUSH;
            }
        } else if depth < 1.1 {
            return if ground_h < water_level + 0.5 {
                MAT_SAND_RIVER
            } else {
                MAT_DIRT
            };
        } else if depth < 4.5 {
            return MAT_LIMESTONE;
        } else {
            return MAT_STONE_DEEP;
        }
    } else if wy <= water_level {
        return MAT_WATER_CRYSTAL;
    }

    MAT_AIR
}

/// Procedural generation of the Central Ancient Ruin Sanctuary at 6.25 cm precision:
/// - Raised stone platform (`y = 12.0m .. 13.5m`) with walkable 3-voxel ($18.75\text{ cm}$) steps on all 4 sides.
/// - Ring of 8 fluted temple pillars with base plinths, moss, and capitals supporting vaulted stone arches.
/// - Central glowing Sunstone / Cyan Crystal Obelisk casting indirect bounce light across the ruins.
fn sample_temple_ruins(pos_m: Vec3, center_xz: Vec2, voxel_hash: u32) -> Option<MaterialId> {
    let rx = pos_m.x - center_xz.x;
    let rz = pos_m.z - center_xz.y;
    let wy = pos_m.y;

    let abs_x = rx.abs();
    let abs_z = rz.abs();
    let max_xz = abs_x.max(abs_z);

    // Base ground level at the sanctuary is y = 12.0m; platform top is y = 13.5m (1.5m = 24 voxels high).
    let base_y = 12.0f32;
    let platform_top = 13.5f32;

    if max_xz > 16.0 || wy < base_y - 1.0 || wy > 23.5 {
        return None;
    }

    // 1. Walkable Grand Staircases on the South (-Z), North (+Z), East (+X), and West (-X) approaches:
    // Platform inner square is `max_xz <= 10.5m`. Stairs extend from `10.5m` out to `14.5m` (4.0m run for 1.5m rise).
    // Each step is 3 voxels high (0.1875m) and 8 voxels deep (0.5m) -> 8 steps total!
    let in_stair_corridor = (abs_x < 3.2 && (10.5..=14.5).contains(&abs_z))
        || (abs_z < 3.2 && (10.5..=14.5).contains(&abs_x));

    if in_stair_corridor && wy >= base_y {
        let dist_from_inner = (max_xz - 10.5).clamp(0.0, 4.0);
        let step_idx = (dist_from_inner / 0.5).floor(); // 0..7
        let step_top = platform_top - step_idx * (3.0 * VOXEL_SIZE);
        if wy <= step_top {
            return Some(if (voxel_hash & 15) == 0 {
                MAT_MOSS_VIBRANT
            } else {
                MAT_SLATE_CARVED
            });
        }
        // Low stone balustrade railings flanking the stairs:
        let on_railing = ((2.75..3.2).contains(&abs_x) && abs_z >= 10.5)
            || ((2.75..3.2).contains(&abs_z) && abs_x >= 10.5);
        if on_railing && wy <= step_top + 0.55 {
            return Some(MAT_LIMESTONE);
        }
    }

    // 2. Main Raised Temple Courtyard Platform (`max_xz <= 10.5m`, `wy <= 13.5m`):
    if max_xz <= 10.5 && wy >= base_y - 0.5 && wy <= platform_top {
        // Decorative inlaid mosaic pattern on the top floor (`wy > 13.35`):
        if wy > platform_top - 0.15 {
            let r_center = (rx * rx + rz * rz).sqrt();
            if (r_center - 4.5).abs() < 0.22 || (r_center - 8.2).abs() < 0.18 {
                return Some(MAT_GOLD_TRIM);
            }
            // Flagstone grid seams every 1.0m:
            let seam_x = (rx.rem_euclid(1.0) - 0.5).abs();
            let seam_z = (rz.rem_euclid(1.0) - 0.5).abs();
            if seam_x > 0.45 || seam_z > 0.45 {
                return Some(if (voxel_hash & 3) == 0 {
                    MAT_MOSS_VIBRANT
                } else {
                    MAT_SLATE_RUIN
                });
            }
            return Some(if (voxel_hash & 7) == 0 {
                MAT_LIMESTONE_WEATHERED
            } else {
                MAT_MARBLE_TEMPLE
            });
        }
        return Some(MAT_SLATE_RUIN);
    }

    // 3. Central Floating/Pedestal Crystal Monolith (`r <= 1.4m` around center):
    let r_center = (rx * rx + rz * rz).sqrt();
    if r_center <= 1.35 && wy > platform_top && wy <= platform_top + 4.8 {
        let rel_h = wy - platform_top;
        // Carved pedestal base (0.0 .. 1.1m):
        if rel_h <= 1.1 {
            if r_center <= 1.1 - rel_h * 0.25 {
                return Some(if (rel_h - 0.9).abs() < 0.12 {
                    MAT_GOLD_TRIM
                } else {
                    MAT_SLATE_CARVED
                });
            }
        } else if rel_h >= 1.35 {
            // Faceted glowing Cyan & Amber Crystal Spire above the pedestal:
            let crystal_r = (0.72 - (rel_h - 2.6).abs() * 0.32).max(0.0);
            if r_center <= crystal_r {
                return Some(if rel_h > 3.2 {
                    MAT_CRYSTAL_AMBER
                } else {
                    MAT_CRYSTAL_CYAN
                });
            }
        }
    }

    // 4. Perimeter Colonnade Pillars & Vaulted Stone Arches (at `x, z in ±8.5m`):
    // Place pillars at intervals around the 8.5m square perimeter:
    if wy > platform_top && wy <= platform_top + 7.2 {
        let rel_h = wy - platform_top;
        let pillar_coords = [-8.5f32, -3.5, 3.5, 8.5];

        for &px in &pillar_coords {
            for &pz in &pillar_coords {
                // Only place pillars along the outer perimeter ring (`|px| == 8.5 || |pz| == 8.5`):
                if px.abs() < 8.0 && pz.abs() < 8.0 {
                    continue;
                }
                let dx = rx - px;
                let dz = rz - pz;
                let dist_p = (dx * dx + dz * dz).sqrt();

                // Weathering: NE corner ruin is partially collapsed for dramatic silhouette!
                let max_pillar_h = if px > 3.0 && pz < -3.0 { 3.8 } else { 5.4 };

                if rel_h <= max_pillar_h {
                    // Square plinth base & capital, fluted cylindrical shaft in between:
                    if (rel_h < 0.55 || (rel_h > 4.85 && rel_h <= 5.4))
                        && dx.abs() <= 0.68
                        && dz.abs() <= 0.68
                    {
                        return Some(MAT_SLATE_CARVED);
                    }
                    let angle = dz.atan2(dx);
                    let flute = (angle * 8.0).cos() * 0.04;
                    if dist_p <= 0.54 + flute {
                        if rel_h < 1.4 && (voxel_hash & 7) == 0 {
                            return Some(MAT_MOSS_VIBRANT);
                        }
                        return Some(MAT_LIMESTONE);
                    }
                }
            }
        }

        // 5. Vaulted Stone Archways connecting adjacent perimeter pillars (`rel_h` between 4.6m and 6.8m):
        let on_x_wall = (abs_z - 8.5).abs() <= 0.55 && abs_x <= 9.0;
        let on_z_wall = (abs_x - 8.5).abs() <= 0.55 && abs_z <= 9.0;

        if (on_x_wall || on_z_wall) && (4.2..=6.8).contains(&rel_h) {
            // Skip the weathered NE corner arch so sunlight streams through:
            if !(rx > 3.0 && rz < -3.0) {
                let span_coord = if on_x_wall { abs_x } else { abs_z };
                // Arch spans between 0..3.5 (center gateway) and 3.5..8.5 (side bays):
                let (arch_center, arch_radius) = if span_coord <= 3.5 {
                    (0.0f32, 2.85f32)
                } else {
                    (6.0f32, 1.95f32)
                };
                let d_span = (span_coord - arch_center).abs();
                let arch_curve_y = 4.2
                    + (arch_radius * arch_radius - (d_span.min(arch_radius)).powi(2)).sqrt() * 0.75;

                if rel_h >= arch_curve_y && rel_h <= 6.5 {
                    if (rel_h - 6.35).abs() < 0.15 {
                        return Some(MAT_MOSS_VIBRANT);
                    }
                    return Some(if (voxel_hash & 3) == 0 {
                        MAT_LIMESTONE_WEATHERED
                    } else {
                        MAT_LIMESTONE
                    });
                }
            }
        }

        // 6. Glowing Corner Crystal Braziers atop the 4 corner pillars (`rel_h in 5.4..6.4`):
        if (abs_x - 8.5).abs() <= 0.5 && (abs_z - 8.5).abs() <= 0.5 && rel_h > 5.4 && rel_h <= 6.2 {
            return Some(MAT_CRYSTAL_AMBER);
        }
    }

    None
}

fn sample_tree(pos_m: Vec3, tree: &TreeInstance, voxel_hash: u32) -> Option<MaterialId> {
    let rel = pos_m - tree.base_pos;
    if rel.y < 0.0 || rel.y > tree.height + tree.canopy_radius {
        return None;
    }

    // Slight organic trunk lean/twist with height:
    let bend_x = (rel.y * 0.45).sin() * 0.35;
    let bend_z = (rel.y * 0.38).cos() * 0.35;
    let dx = rel.x - bend_x;
    let dz = rel.z - bend_z;
    let dist_xz = (dx * dx + dz * dz).sqrt();

    // Flared roots near `rel.y < 1.2m` + tapered trunk up to `tree.height`:
    if rel.y <= tree.height {
        let root_flare = if rel.y < 1.2 {
            (1.2 - rel.y) * 0.35 * ((dz.atan2(dx) * 5.0).cos().max(0.0))
        } else {
            0.0
        };
        let taper = 1.0 - (rel.y / tree.height) * 0.45;
        if dist_xz <= tree.trunk_radius * taper + root_flare {
            return Some(MAT_WOOD_OAK_BARK);
        }
    }

    // Multi-lobed foliage canopy cluster around the top of the trunk:
    let canopy_center = Vec3::new(bend_x, tree.height - 0.4, bend_z);
    let c_rel = rel - canopy_center;
    // Squash Y slightly for layered John Lin bonsai/oak look:
    let ellip_dist = (c_rel.x * c_rel.x + (c_rel.y * 1.45).powi(2) + c_rel.z * c_rel.z).sqrt();
    let lobe_mod =
        0.85 + 0.25 * ((pos_m.x * 1.3).sin() * (pos_m.y * 1.7).cos() * (pos_m.z * 1.3).sin());

    if ellip_dist <= tree.canopy_radius * lobe_mod {
        // Porous outer leaf shell so sunlight filters through the leaves!
        if ellip_dist > tree.canopy_radius * 0.65 && (voxel_hash & 3) == 0 {
            return None;
        }
        return Some(tree.leaf_mat);
    }

    None
}

// Fast deterministic 2D value noise & FBM helpers:
#[inline]
fn hash2(x: i32, z: i32) -> f32 {
    let mut n = x
        .wrapping_mul(374761393)
        .wrapping_add(z.wrapping_mul(668265263));
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    ((n ^ (n >> 16)) as u32 as f32) / (u32::MAX as f32)
}

#[inline]
fn value_noise2(x: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iz = z.floor() as i32;
    let fx = x - x.floor();
    let fz = z - z.floor();
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uz = fz * fz * (3.0 - 2.0 * fz);

    let v00 = hash2(ix, iz);
    let v10 = hash2(ix + 1, iz);
    let v01 = hash2(ix, iz + 1);
    let v11 = hash2(ix + 1, iz + 1);

    let a = v00 + ux * (v10 - v00);
    let b = v01 + ux * (v11 - v01);
    a + uz * (b - a)
}

#[inline]
fn fbm2(mut x: f32, mut z: f32, octaves: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 0.5f32;
    for _ in 0..octaves {
        sum += (value_noise2(x, z) * 2.0 - 1.0) * amp;
        x *= 2.03;
        z *= 2.03;
        amp *= 0.5;
    }
    sum
}
