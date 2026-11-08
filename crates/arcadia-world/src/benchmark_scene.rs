//! Benchmark Demo Scene for Arcadia Voxel Engine.
//! Owns CPU procedural benchmark scene and camera trajectory generation.
//!
//! Specifically engineered for rigorous performance benchmarking and profiling of:
//! 1. **Light**: Soft sun shadows, 6 analytical 3D point lights, multi-bounce ReSTIR GI,
//!    and high-intensity emissive crystal monoliths & molten lava fissures.
//! 2. **Water**: Sunken crystal reflecting basin (`MAT_WATER_CRYSTAL`), caustics on mosaic bed,
//!    subsurface light absorption, and Fresnel reflection off wave-perturbed water.
//! 3. **Complex Geometry**: Grand fluted Romanesque/Gothic colonnade with carved capitals,
//!    vaulted arch ceilings, twin helical/spiral staircases with 3-voxel steps, elevated
//!    skybridge galleries, carved obelisks, and natural cliff amphitheater.
//! 4. **Reflective Surfaces**: Massive polished mirror chrome pillars (`MAT_MIRROR_CHROME`),
//!    a 16m wide polished mirror feature wall reflecting the entire complex in real-time ray tracing,
//!    a suspended mirror chrome orb, and dynamic water Fresnel reflection.

use crate::world_map::WorldMap;
use arcadia_core::{
    ChunkTree64, MaterialId, MaterialPalette, CHUNK_DIM, MAT_AIR, MAT_BEDROCK, MAT_CRYSTAL_AMBER,
    MAT_CRYSTAL_CYAN, MAT_CRYSTAL_MAGENTA, MAT_DIRT, MAT_EMBER_GLOW, MAT_GOLD_TRIM,
    MAT_GRASS_GOLDEN, MAT_GRASS_LUSH, MAT_LAVA_MAGMA, MAT_LIMESTONE, MAT_LIMESTONE_WEATHERED,
    MAT_MARBLE_TEMPLE, MAT_MIRROR_CHROME, MAT_MOSS_VIBRANT, MAT_SAND_RIVER, MAT_SCORCHED_ROCK,
    MAT_SLATE_CARVED, MAT_SLATE_RUIN, MAT_STONE_DEEP, MAT_WATER_CRYSTAL, VOXEL_SIZE,
};
use glam::{UVec3, Vec2, Vec3};
use rayon::prelude::*;

/// Generates the dedicated Arcadia Benchmark World.
///
/// Default grid `UVec3::new(10, 4, 10)` covers `160m × 64m × 160m`
/// ($\approx 6.71\text{ billion}$ potential $(1/16\text{ m})^3$ voxels).
pub fn generate_benchmark_world(grid_dims: UVec3) -> WorldMap {
    let palette = MaterialPalette::fantasy_nature_palette();
    let total_chunks = (grid_dims.x * grid_dims.y * grid_dims.z) as usize;

    let world_size_m = grid_dims.as_vec3() * (CHUNK_DIM as f32 * VOXEL_SIZE);
    let center_xz = Vec2::new(world_size_m.x * 0.5, world_size_m.z * 0.5);

    let chunks: Vec<ChunkTree64> = (0..total_chunks)
        .into_par_iter()
        .map(|flat_idx| {
            let cx = (flat_idx as u32) % grid_dims.x;
            let cy = ((flat_idx as u32) / grid_dims.x) % grid_dims.y;
            let cz = (flat_idx as u32) / (grid_dims.x * grid_dims.y);

            let chunk_base_vx = cx * CHUNK_DIM;
            let chunk_base_vy = cy * CHUNK_DIM;
            let chunk_base_vz = cz * CHUNK_DIM;

            ChunkTree64::from_generator(
                &palette,
                |ox, oy, oz, size| {
                    if size > 64 {
                        return None;
                    }

                    let bmin_m = Vec3::new(
                        (chunk_base_vx + ox) as f32,
                        (chunk_base_vy + oy) as f32,
                        (chunk_base_vz + oz) as f32,
                    ) * VOXEL_SIZE;
                    let bmax_m = bmin_m + Vec3::splat(size as f32 * VOXEL_SIZE);

                    // 1. Bedrock floor collapse:
                    if bmax_m.y < 0.25 {
                        return Some(MAT_BEDROCK);
                    }

                    // 2. High sky skip:
                    if bmin_m.y > 36.0 {
                        return Some(MAT_AIR);
                    }

                    // 3. Deep underground solid stone collapse:
                    if bmax_m.y < 8.0 && bmin_m.y >= 0.25 {
                        let rel = Vec2::new(
                            (bmin_m.x + bmax_m.x) * 0.5 - center_xz.x,
                            (bmin_m.z + bmax_m.z) * 0.5 - center_xz.y,
                        );
                        if rel.length() > 38.0 {
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
                    sample_benchmark_voxel(pos_m, gx, gy, gz, center_xz)
                },
            )
        })
        .collect();

    WorldMap::from_chunks(grid_dims, chunks, palette)
}

/// Evaluates a single micro-voxel in the benchmark world.
#[inline]
fn sample_benchmark_voxel(pos_m: Vec3, gx: u32, gy: u32, gz: u32, center_xz: Vec2) -> MaterialId {
    if gy < 4 {
        return MAT_BEDROCK;
    }

    let wx = pos_m.x;
    let wy = pos_m.y;
    let wz = pos_m.z;

    let rx = wx - center_xz.x;
    let rz = wz - center_xz.y;
    let dist_center = (rx * rx + rz * rz).sqrt();

    let voxel_hash =
        (gx.wrapping_mul(73856093)) ^ (gy.wrapping_mul(19349663)) ^ (gz.wrapping_mul(83492791));

    if let Some(mat) = sample_mountain_cliffs(pos_m, wy, dist_center) {
        return mat;
    }
    if let Some(mat) = sample_reflecting_pool(pos_m, center_xz, rx, rz, wy, voxel_hash) {
        return mat;
    }
    if let Some(mat) = sample_courtyard_floor(rx, rz, wy, voxel_hash) {
        return mat;
    }
    if let Some(mat) = sample_mirror_structures(rx, rz, wy) {
        return mat;
    }
    if let Some(mat) = sample_colonnade_and_vaulted_arches(rx, rz, wy) {
        return mat;
    }
    if let Some(mat) = sample_spiral_staircases_and_skybridge(rx, rz, wy, voxel_hash) {
        return mat;
    }
    if let Some(mat) = sample_volcanic_magma_grotto(rx, rz, wy, voxel_hash) {
        return mat;
    }
    if let Some(mat) = sample_emissive_crystals(rx, rz, wy) {
        return mat;
    }

    MAT_AIR
}

fn sample_mountain_cliffs(pos_m: Vec3, wy: f32, dist_center: f32) -> Option<MaterialId> {
    if dist_center <= 38.0 {
        return None;
    }
    let wx = pos_m.x;
    let wz = pos_m.z;
    let cliff_t = ((dist_center - 38.0) / 22.0).clamp(0.0, 1.0);
    let cliff_height = 12.5 + cliff_t * cliff_t * 22.0;
    if wy <= cliff_height {
        let depth = cliff_height - wy;
        if depth < 0.65 {
            let n = ((wx * 0.1).sin() * (wz * 0.1).cos() * 0.5 + 0.5).clamp(0.0, 1.0);
            if n > 0.6 {
                return Some(MAT_GRASS_LUSH);
            } else if n > 0.3 {
                return Some(MAT_MOSS_VIBRANT);
            } else {
                return Some(MAT_GRASS_GOLDEN);
            }
        } else if depth < 1.5 {
            return Some(MAT_DIRT);
        } else if depth < 6.0 {
            return Some(MAT_LIMESTONE);
        } else {
            return Some(MAT_STONE_DEEP);
        }
    }
    Some(MAT_AIR)
}

fn sample_reflecting_pool(
    pos_m: Vec3,
    center_xz: Vec2,
    rx: f32,
    rz: f32,
    wy: f32,
    voxel_hash: u32,
) -> Option<MaterialId> {
    let in_pool_bounds = rx.abs() <= 13.5 && rz.abs() <= 13.5;
    if !in_pool_bounds {
        return None;
    }
    let water_level = 11.25f32;
    let pool_bed_y = 9.80f32;
    let max_coord = rx.abs().max(rz.abs());

    let r_center = (rx * rx + rz * rz).sqrt();
    if r_center <= 1.45 && wy >= pool_bed_y && wy <= 18.2 {
        let rel_h = wy - pool_bed_y;
        if rel_h <= 1.6 {
            if r_center <= 1.4 - rel_h * 0.12 {
                return Some(MAT_MARBLE_TEMPLE);
            }
        } else if rel_h <= 2.2 {
            if r_center <= 1.25 {
                return Some(MAT_GOLD_TRIM);
            }
        } else if rel_h <= 3.2 {
            if r_center <= 1.0 - (rel_h - 2.2) * 0.25 {
                return Some(MAT_SLATE_CARVED);
            }
        } else {
            let spire_t = (rel_h - 3.2) / 5.2;
            let max_r = (0.75 * (1.0 - spire_t)).max(0.0);
            let oct_dist = rx.abs().max(rz.abs()) * 0.707 + (rx.abs() + rz.abs()) * 0.5;
            if oct_dist <= max_r * 1.2 {
                return Some(if spire_t > 0.7 {
                    MAT_CRYSTAL_AMBER
                } else {
                    MAT_CRYSTAL_CYAN
                });
            }
        }
    }

    let orb_center = Vec3::new(center_xz.x, 14.8, center_xz.y);
    let dist_orb = pos_m.distance(orb_center);
    if dist_orb <= 1.15 {
        return Some(MAT_MIRROR_CHROME);
    }

    if max_coord >= 10.5 {
        let step_dist = max_coord - 10.5;
        let step_idx = (step_dist / 0.5).floor();
        let step_y = 12.5 - (5.0 - step_idx) * (3.0 * VOXEL_SIZE);
        if wy <= step_y {
            return Some(if (voxel_hash & 7) == 0 {
                MAT_MOSS_VIBRANT
            } else {
                MAT_SLATE_CARVED
            });
        }
    }

    if wy <= pool_bed_y {
        if wy > pool_bed_y - 0.25 {
            let r_mosaic = (rx * rx + rz * rz).sqrt();
            if (r_mosaic - 4.5).abs() < 0.25 || (r_mosaic - 8.5).abs() < 0.25 {
                return Some(MAT_GOLD_TRIM);
            }
            let grid_x = (rx * 0.35).floor() as i32;
            let grid_z = (rz * 0.35).floor() as i32;
            if ((grid_x * 19 + grid_z * 31) & 7) == 0 {
                let lx = (rx * 0.35).fract() - 0.5;
                let lz = (rz * 0.35).fract() - 0.5;
                if lx * lx + lz * lz < 0.04 {
                    return Some(MAT_CRYSTAL_CYAN);
                }
            }
            return Some(if (voxel_hash & 3) == 0 {
                MAT_SAND_RIVER
            } else {
                MAT_SLATE_RUIN
            });
        }
        return Some(MAT_STONE_DEEP);
    }

    if wy <= water_level {
        return Some(MAT_WATER_CRYSTAL);
    }

    if wy < 12.5 && max_coord < 10.5 {
        return Some(MAT_AIR);
    }

    None
}

fn sample_courtyard_floor(rx: f32, rz: f32, wy: f32, voxel_hash: u32) -> Option<MaterialId> {
    if wy <= 12.5 {
        if wy > 12.25 {
            let seam_x = (rx.rem_euclid(1.0) - 0.5).abs();
            let seam_z = (rz.rem_euclid(1.0) - 0.5).abs();
            if seam_x > 0.46 || seam_z > 0.46 {
                return Some(if (voxel_hash & 3) == 0 {
                    MAT_MOSS_VIBRANT
                } else {
                    MAT_SLATE_CARVED
                });
            }
            return Some(if (voxel_hash & 7) == 0 {
                MAT_LIMESTONE_WEATHERED
            } else {
                MAT_MARBLE_TEMPLE
            });
        }
        return Some(MAT_STONE_DEEP);
    }
    None
}

fn sample_mirror_structures(rx: f32, rz: f32, wy: f32) -> Option<MaterialId> {
    for &mx in &[-7.5f32, 7.5f32] {
        let mz = -7.5f32;
        let d_mirror_col = ((rx - mx).powi(2) + (rz - mz).powi(2)).sqrt();
        if d_mirror_col <= 0.85 && (12.0..=21.0).contains(&wy) {
            if wy <= 13.0 || wy >= 20.2 {
                return Some(MAT_GOLD_TRIM);
            }
            return Some(MAT_MIRROR_CHROME);
        }
    }

    if (16.0..=16.5).contains(&rx) && (-10.0..=10.0).contains(&rz) && wy <= 20.5 {
        if rx >= 16.35 || rz <= -9.5 || rz >= 9.5 || wy <= 13.0 || wy >= 20.0 {
            return Some(MAT_GOLD_TRIM);
        }
        return Some(MAT_MIRROR_CHROME);
    }

    None
}

fn sample_colonnade_and_vaulted_arches(rx: f32, rz: f32, wy: f32) -> Option<MaterialId> {
    let near_col_x = (rx.abs() - 16.0).abs() < 1.4 && rz.abs() <= 18.0;
    let near_col_z = (rz.abs() - 16.0).abs() < 1.4 && rx.abs() <= 18.0;

    if (near_col_x || near_col_z) && wy <= 25.5 {
        let px = (rx / 4.0).round() * 4.0;
        let pz = (rz / 4.0).round() * 4.0;
        let dx = rx - px;
        let dz = rz - pz;
        let d_col = (dx * dx + dz * dz).sqrt();

        let is_central_portal = px.abs() < 0.5 && (pz.abs() - 16.0).abs() < 0.5;

        if is_central_portal && wy <= 20.5 && rx.abs() <= 2.2 {
            return Some(MAT_AIR);
        }

        if !is_central_portal && d_col <= 0.68 && wy <= 21.0 {
            if wy <= 13.5 {
                let square_r = dx.abs().max(dz.abs());
                if square_r <= 0.82 {
                    return Some(if wy > 13.3 {
                        MAT_GOLD_TRIM
                    } else {
                        MAT_MARBLE_TEMPLE
                    });
                }
            } else if wy <= 20.2 {
                let angle = dz.atan2(dx);
                let flute = (angle * 8.0).sin() * 0.035;
                if d_col <= 0.55 + flute {
                    return Some(MAT_MARBLE_TEMPLE);
                }
            } else {
                let cap_r = 0.55 + (wy - 20.2) * 0.35;
                if d_col <= cap_r {
                    return Some(MAT_GOLD_TRIM);
                }
            }
        }

        if (20.8..=25.0).contains(&wy) {
            let arch_span_x = (rx.rem_euclid(4.0) - 2.0).abs();
            let arch_span_z = (rz.rem_euclid(4.0) - 2.0).abs();
            let arch_curve_x = 21.0 + (1.0 - (arch_span_x / 2.0).powi(2)).max(0.0).sqrt() * 3.5;
            let arch_curve_z = 21.0 + (1.0 - (arch_span_z / 2.0).powi(2)).max(0.0).sqrt() * 3.5;
            let arch_h = arch_curve_x.min(arch_curve_z);

            if wy >= arch_h - 0.45 && wy <= arch_h + 0.35 {
                if (arch_span_x < 0.25 || arch_span_z < 0.25) || (wy > arch_h) {
                    return Some(MAT_GOLD_TRIM);
                }
                return Some(MAT_SLATE_CARVED);
            }
        }
    }

    None
}

fn sample_spiral_staircases_and_skybridge(
    rx: f32,
    rz: f32,
    wy: f32,
    voxel_hash: u32,
) -> Option<MaterialId> {
    for &sx in &[-16.0f32] {
        for &sz in &[-16.0f32, 16.0f32] {
            let dx = rx - sx;
            let dz = rz - sz;
            let r_spiral = (dx * dx + dz * dz).sqrt();

            if r_spiral <= 2.6 && (12.5..=21.5).contains(&wy) {
                if r_spiral <= 0.45 {
                    return Some(MAT_MARBLE_TEMPLE);
                }
                let angle = dz.atan2(dx).rem_euclid(std::f32::consts::TAU);
                let turn_progress = angle / std::f32::consts::TAU;
                for turn in 0..2 {
                    let base_step_y = 12.5 + (turn as f32 + turn_progress) * 4.5;
                    if wy >= base_step_y && wy <= base_step_y + (3.0 * VOXEL_SIZE) {
                        return Some(if (voxel_hash & 7) == 0 {
                            MAT_MOSS_VIBRANT
                        } else {
                            MAT_SLATE_CARVED
                        });
                    }
                }
            }
        }
    }

    if (-17.5..=-14.5).contains(&rx) && (-16.0..=16.0).contains(&rz) {
        if (20.5..=21.0).contains(&wy) {
            return Some(MAT_MARBLE_TEMPLE);
        }
        let at_edge = (rx - -17.5).abs() < 0.25 || (rx - -14.5).abs() < 0.25;
        if at_edge && wy > 21.0 && wy <= 22.0 {
            return Some(if wy > 21.8 {
                MAT_GOLD_TRIM
            } else {
                MAT_LIMESTONE
            });
        }
    }

    None
}

fn sample_volcanic_magma_grotto(rx: f32, rz: f32, wy: f32, voxel_hash: u32) -> Option<MaterialId> {
    if (16.5..=28.0).contains(&rz) && rx.abs() <= 8.5 {
        let alcove_rel = Vec3::new(rx / 8.0, (wy - 15.0) / 6.0, (rz - 22.0) / 6.0);
        let d_alcove = alcove_rel.length();

        if d_alcove <= 1.0 {
            let chute_x = rx.abs();
            if chute_x <= 1.8 && rz >= 20.0 {
                let cascade_y = 12.0 + (rz - 20.0) * 0.85;
                if wy <= cascade_y && wy >= 11.5 {
                    return Some(MAT_LAVA_MAGMA);
                }
                if wy <= cascade_y + 0.35 && chute_x <= 2.2 {
                    return Some(MAT_EMBER_GLOW);
                }
            }

            if wy <= 12.0 && rz >= 19.0 {
                return Some(MAT_LAVA_MAGMA);
            }

            if d_alcove >= 0.82 {
                return Some(if (voxel_hash & 3) == 0 {
                    MAT_SCORCHED_ROCK
                } else {
                    MAT_STONE_DEEP
                });
            }

            return Some(MAT_AIR);
        }
    }

    None
}

fn sample_emissive_crystals(rx: f32, rz: f32, wy: f32) -> Option<MaterialId> {
    let d_mag = ((rx - 12.0).powi(2) + (rz - 0.0).powi(2)).sqrt();
    if d_mag <= 1.8 && (12.5..=16.8).contains(&wy) {
        let max_r = (1.1 - (wy - 12.5) * 0.25).max(0.0);
        if d_mag <= max_r {
            return Some(MAT_CRYSTAL_MAGENTA);
        }
    }

    let d_amb = ((rx - 0.0).powi(2) + (rz - -12.0).powi(2)).sqrt();
    if d_amb <= 1.8 && (12.5..=16.5).contains(&wy) {
        let max_r = (1.1 - (wy - 12.5) * 0.28).max(0.0);
        if d_amb <= max_r {
            return Some(MAT_CRYSTAL_AMBER);
        }
    }

    None
}

/// Represents a camera control waypoint for the deterministic benchmark flythrough path.
#[derive(Clone, Copy, Debug)]
pub struct BenchmarkCameraKeyframe {
    pub pos: Vec3,
    pub look_at: Vec3,
    pub fov_y_deg: f32,
}

/// Cinematic deterministic camera path through the benchmark sanctuary.
pub struct BenchmarkCameraPath {
    pub keyframes: Vec<BenchmarkCameraKeyframe>,
}

impl BenchmarkCameraPath {
    /// Creates the standard 9-waypoint benchmark flythrough sequence.
    pub fn new(center_xz: Vec2) -> Self {
        let cx = center_xz.x;
        let cz = center_xz.y;

        let keyframes = vec![
            // Keyframe 0 (t = 0.00): High panoramic establishing shot looking down across the
            // grand colonnade, reflecting pool, and mirror chrome columns.
            BenchmarkCameraKeyframe {
                pos: Vec3::new(cx, 26.0, cz - 34.0),
                look_at: Vec3::new(cx, 13.5, cz + 2.0),
                fov_y_deg: 65.0,
            },
            // Keyframe 1 (t = 0.125): Descending towards southern entrance, framing colonnade arch.
            BenchmarkCameraKeyframe {
                pos: Vec3::new(cx, 16.5, cz - 19.0),
                look_at: Vec3::new(cx, 13.5, cz),
                fov_y_deg: 65.0,
            },
            // Keyframe 2 (t = 0.25): Passing into courtyard, viewing West colonnade and mirror monolith.
            BenchmarkCameraKeyframe {
                pos: Vec3::new(cx - 5.5, 14.5, cz - 10.0),
                look_at: Vec3::new(cx + 6.0, 14.0, cz - 2.0),
                fov_y_deg: 65.0,
            },
            // Keyframe 3 (t = 0.375): Panning across pool toward the Grand Mirror Chrome Wall (Eastern flank).
            // Heavy real-time ray-traced reflections of water, columns, and crystal obelisk.
            BenchmarkCameraKeyframe {
                pos: Vec3::new(cx + 5.5, 13.8, cz - 5.0),
                look_at: Vec3::new(cx + 16.2, 14.5, cz),
                fov_y_deg: 65.0,
            },
            // Keyframe 4 (t = 0.50): Low sweep across reflecting basin.
            // 0.95m above water, curving around central crystal obelisk and mirror orb.
            BenchmarkCameraKeyframe {
                pos: Vec3::new(cx - 5.0, 12.2, cz - 1.0),
                look_at: Vec3::new(cx, 13.5, cz + 6.0),
                fov_y_deg: 68.0,
            },
            // Keyframe 5 (t = 0.625): Ascending smoothly from the pool toward the open northern portal.
            BenchmarkCameraKeyframe {
                pos: Vec3::new(cx - 1.5, 14.0, cz + 11.0),
                look_at: Vec3::new(cx, 14.5, cz + 22.0),
                fov_y_deg: 65.0,
            },
            // Keyframe 6 (t = 0.75): Gliding through open arch into the Volcanic Magma Grotto alcove.
            // Multi-bounce ReSTIR GI from flowing lava cascade, glowing embers, and scorched basalt.
            BenchmarkCameraKeyframe {
                pos: Vec3::new(cx, 14.5, cz + 18.0),
                look_at: Vec3::new(cx, 14.0, cz + 24.0),
                fov_y_deg: 65.0,
            },
            // Keyframe 7 (t = 0.875): Pulling back into courtyard, framing northwest spiral stair & gallery.
            BenchmarkCameraKeyframe {
                pos: Vec3::new(cx - 3.5, 16.8, cz + 10.0),
                look_at: Vec3::new(cx - 14.0, 16.0, cz + 14.0),
                fov_y_deg: 65.0,
            },
            // Keyframe 8 (t = 1.00): Rising into open courtyard sky, overlooking entire illuminated sanctuary.
            BenchmarkCameraKeyframe {
                pos: Vec3::new(cx, 21.5, cz - 6.0),
                look_at: Vec3::new(cx, 13.5, cz + 2.0),
                fov_y_deg: 65.0,
            },
        ];

        Self { keyframes }
    }

    /// Evaluates the camera state at normalized time `t in [0.0, 1.0]` using Catmull-Rom spline interpolation.
    pub fn sample(&self, t: f32) -> (Vec3, Vec3, f32) {
        let n = self.keyframes.len();
        if n == 0 {
            return (Vec3::ZERO, Vec3::Z, 65.0);
        }
        if n == 1 {
            return (
                self.keyframes[0].pos,
                self.keyframes[0].look_at,
                self.keyframes[0].fov_y_deg,
            );
        }

        let clamped_t = t.clamp(0.0, 1.0);
        let segment_count = (n - 1) as f32;
        let scaled_t = clamped_t * segment_count;
        let idx = (scaled_t.floor() as usize).min(n - 2);
        let local_t = scaled_t - idx as f32;

        let p0 = if idx > 0 {
            self.keyframes[idx - 1]
        } else {
            self.keyframes[0]
        };
        let p1 = self.keyframes[idx];
        let p2 = self.keyframes[idx + 1];
        let p3 = if idx + 2 < n {
            self.keyframes[idx + 2]
        } else {
            p2
        };

        let pos = catmull_rom_vec3(p0.pos, p1.pos, p2.pos, p3.pos, local_t);
        let look_at = catmull_rom_vec3(p0.look_at, p1.look_at, p2.look_at, p3.look_at, local_t);
        let fov = catmull_rom_f32(
            p0.fov_y_deg,
            p1.fov_y_deg,
            p2.fov_y_deg,
            p3.fov_y_deg,
            local_t,
        );

        (pos, look_at, fov)
    }
}

#[inline]
fn catmull_rom_vec3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

#[inline]
fn catmull_rom_f32(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Convenience helper to create the benchmark camera trajectory.
pub fn get_benchmark_camera_path(center_xz: Vec2) -> BenchmarkCameraPath {
    BenchmarkCameraPath::new(center_xz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_camera_path_interpolation() {
        let center = Vec2::new(80.0, 80.0);
        let path = get_benchmark_camera_path(center);
        assert_eq!(path.keyframes.len(), 9);

        for step in 0..=200 {
            let t = step as f32 / 200.0;
            let (pos, look_at, fov) = path.sample(t);
            assert!(pos.is_finite());
            assert!(look_at.is_finite());
            assert!((50.0..=80.0).contains(&fov));
            assert!(pos.distance(look_at) > 1.0);

            // INVARIANT: Camera must not clip into any solid voxel along the entire trajectory.
            let gx = (pos.x / VOXEL_SIZE).floor() as u32;
            let gy = (pos.y / VOXEL_SIZE).floor() as u32;
            let gz = (pos.z / VOXEL_SIZE).floor() as u32;
            let mat = sample_benchmark_voxel(pos, gx, gy, gz, center);
            assert!(
                mat == MAT_AIR || mat == MAT_WATER_CRYSTAL,
                "Camera collision detected at t={:.3}, pos={:?}, sampled solid material={:?}",
                t,
                pos,
                mat
            );
        }
    }

    #[test]
    fn test_generate_benchmark_world_small_grid() {
        let dims = UVec3::new(2, 2, 2);
        let world = generate_benchmark_world(dims);
        assert_eq!(world.chunks.len(), 8);
        assert!(!world.gpu_nodes.is_empty());
        assert!(!world.gpu_bricks.is_empty());
    }

    #[test]
    fn test_benchmark_pillar_carving() {
        let dims = UVec3::new(10, 6, 10);
        let mut world = generate_benchmark_world(dims);
        let world_size_m = dims.as_vec3() * (CHUNK_DIM as f32 * VOXEL_SIZE);
        let center_xz = Vec2::new(world_size_m.x * 0.5, world_size_m.z * 0.5);

        // Target one of the freestanding colonnade pillars on the North flank (rx = 4.0, rz = 16.0):
        let pillar_pos = Vec3::new(center_xz.x + 4.0, 14.5, center_xz.y + 16.0);
        println!("Carving colonnade pillar base at {:?}", pillar_pos);
        let event = world.carve_sphere_meters(pillar_pos, 0.95, false, 0);
        println!(
            "Detached islands after cut 1 (base cut): {}",
            event.detached_islands.len()
        );
        for (i, comp) in event.detached_islands.iter().enumerate() {
            println!("  Island {}: {} voxels", i, comp.len());
        }
        assert!(
            !event.detached_islands.is_empty(),
            "Colonnade pillar must detach on base cut without requiring neighbor pillar to fall"
        );

        // Target one of the standalone mirror pillars at rx = 7.5, rz = -7.5:
        let mirror_pos = Vec3::new(center_xz.x + 7.5, 14.5, center_xz.y - 7.5);
        println!("Carving standalone mirror pillar at {:?}", mirror_pos);
        let event_mirror = world.carve_sphere_meters(mirror_pos, 0.95, false, 0);
        println!(
            "Mirror pillar detached islands: {}",
            event_mirror.detached_islands.len()
        );
        assert!(
            !event_mirror.detached_islands.is_empty(),
            "Standalone mirror pillar must detach when severed mid-shaft"
        );
    }
}
