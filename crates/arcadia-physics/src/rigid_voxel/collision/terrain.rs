//! Exact compound sub-box vs. static `WorldMap` terrain and bedrock contact generation.
//! Owns CPU BVH-pruned world brick run merging, exposed-face SAT contacts, and volume-weighted water immersion.

use super::super::types::{ContactManifold, RigidVoxelBody};
use super::broadphase::Obb;
use super::compound::{FACE_NEG_X, FACE_NEG_Y, FACE_NEG_Z, FACE_POS_X, FACE_POS_Y, FACE_POS_Z};
use super::manifold::{compute_subbox_obb_manifold, orthonormal_basis, reduce_contact_samples};
use arcadia_core::{
    child_bit_index, unpack_child_index, MAT_AIR, MAT_WATER_CRYSTAL, VOXELS_PER_METER, VOXEL_SIZE,
};
use arcadia_world::WorldMap;
use glam::{Mat3, Vec3};

const BEDROCK_TOP_VOXELS: f32 = 5.0;
const BEDROCK_TOP_METERS: f32 = BEDROCK_TOP_VOXELS * VOXEL_SIZE;
const CONTACT_SKIN_METERS: f32 = 0.002;
const MAX_TERRAIN_CLUSTERS: usize = 8;
const MAX_CONTACTS_PER_TERRAIN_CLUSTER: usize = 8;

#[derive(Clone, Debug)]
struct TerrainNormalCluster {
    normal_sum: Vec3,
    unit_normal: Vec3,
    max_pen: f32,
    samples: Vec<(Vec3, f32)>,
}

#[derive(Default)]
struct NeighborBrickCache {
    nx: Option<u64>,
    px: Option<u64>,
    ny: Option<u64>,
    py: Option<u64>,
    nz: Option<u64>,
    pz: Option<u64>,
}

impl NeighborBrickCache {
    #[inline]
    fn get_nx(&mut self, world: &WorldMap, bx: i32, by: i32, bz: i32) -> u64 {
        *self
            .nx
            .get_or_insert_with(|| world.get_brick_solid_mask(bx - 1, by, bz))
    }

    #[inline]
    fn get_px(&mut self, world: &WorldMap, bx: i32, by: i32, bz: i32) -> u64 {
        *self
            .px
            .get_or_insert_with(|| world.get_brick_solid_mask(bx + 1, by, bz))
    }

    #[inline]
    fn get_ny(&mut self, world: &WorldMap, bx: i32, by: i32, bz: i32) -> u64 {
        *self.ny.get_or_insert_with(|| {
            if by <= 1 {
                // WHY: Global brick y == 0 covers vy in 0..4 (solid bedrock); y == 1 starts at vy = 4 (also bedrock at ly = 0).
                if by <= 0 {
                    u64::MAX
                } else {
                    world.get_brick_solid_mask(bx, by - 1, bz) | 0x000F_000F_000F_000Fu64
                }
            } else {
                world.get_brick_solid_mask(bx, by - 1, bz)
            }
        })
    }

    #[inline]
    fn get_py(&mut self, world: &WorldMap, bx: i32, by: i32, bz: i32) -> u64 {
        *self
            .py
            .get_or_insert_with(|| world.get_brick_solid_mask(bx, by + 1, bz))
    }

    #[inline]
    fn get_nz(&mut self, world: &WorldMap, bx: i32, by: i32, bz: i32) -> u64 {
        *self
            .nz
            .get_or_insert_with(|| world.get_brick_solid_mask(bx, by, bz - 1))
    }

    #[inline]
    fn get_pz(&mut self, world: &WorldMap, bx: i32, by: i32, bz: i32) -> u64 {
        *self
            .pz
            .get_or_insert_with(|| world.get_brick_solid_mask(bx, by, bz + 1))
    }
}

impl RigidVoxelBody {
    /// Computes the volume-weighted fraction $\in [0, 1]$ of this body submerged in `MAT_WATER_CRYSTAL` voxels.
    pub fn compute_submersion_ratio(&self, world: &WorldMap) -> f32 {
        let vpm = VOXELS_PER_METER as f32;
        if self.collider.sub_boxes.is_empty() {
            let v = (self.position * vpm).floor().as_ivec3();
            return if world.get_voxel(v.x, v.y, v.z) == MAT_WATER_CRYSTAL {
                1.0
            } else {
                0.0
            };
        }

        let r_mat = Mat3::from_quat(self.orientation);
        let mut submerged_weight = 0.0f32;
        let mut total_weight = 0.0f32;

        for sub in &self.collider.sub_boxes {
            let vol = sub.volume_voxels() as f32;
            if vol <= 0.0 {
                continue;
            }
            let dx_v = sub.max_v[0] - sub.min_v[0];
            let dy_v = sub.max_v[1] - sub.min_v[1];
            let dz_v = sub.max_v[2] - sub.min_v[2];

            // Stratify samples along the longest dimensions of large sub-boxes (up to 2x4x2 = 16 probes/box):
            let sx = if dx_v >= 4 { 2u32 } else { 1 };
            let sy = if dy_v >= 8 {
                4u32
            } else if dy_v >= 3 {
                2
            } else {
                1
            };
            let sz = if dz_v >= 4 { 2u32 } else { 1 };
            let sample_count = (sx * sy * sz) as f32;
            let weight_per_sample = vol / sample_count;

            let min_rel = sub.center_rel_com - sub.half_extents;
            let span = sub.half_extents * 2.0;

            for iz in 0..sz {
                let fz = (iz as f32 + 0.5) / (sz as f32);
                for iy in 0..sy {
                    let fy = (iy as f32 + 0.5) / (sy as f32);
                    for ix in 0..sx {
                        let fx = (ix as f32 + 0.5) / (sx as f32);
                        let p_rel = min_rel + Vec3::new(span.x * fx, span.y * fy, span.z * fz);
                        let p_world = self.position + r_mat * p_rel;
                        let v = (p_world * vpm).floor().as_ivec3();
                        if world.get_voxel(v.x, v.y, v.z) == MAT_WATER_CRYSTAL {
                            submerged_weight += weight_per_sample;
                        }
                        total_weight += weight_per_sample;
                    }
                }
            }
        }

        if total_weight > 0.0 {
            (submerged_weight / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Tests whether any bottom `VoxelSubBox` of this body rests on static terrain within support tolerance.
    pub(crate) fn check_ground_support(&self, world: &WorldMap) -> bool {
        let (aabb_min, aabb_max) = self.world_aabb_meters();
        let probe_pad = VOXEL_SIZE * 1.5;
        let query_min = Vec3::new(aabb_min.x, aabb_min.y - probe_pad, aabb_min.z);
        if !world.intersects_aabb_meters(query_min, aabb_max) {
            return false;
        }

        if self.collider.bvh_nodes.is_empty() {
            return false;
        }

        let r_mat = Mat3::from_quat(self.orientation);
        let abs_r = Mat3::from_cols(r_mat.x_axis.abs(), r_mat.y_axis.abs(), r_mat.z_axis.abs());
        let support_band_top = aabb_min.y + VOXEL_SIZE * 4.0;

        let mut stack = [0u32; 32];
        let mut sp = 1usize;

        while sp > 0 {
            sp -= 1;
            let node = &self.collider.bvh_nodes[stack[sp] as usize];
            let nc = (node.min_rel_com + node.max_rel_com) * 0.5;
            let nh = (node.max_rel_com - node.min_rel_com) * 0.5;
            let wc = self.position + r_mat * nc;
            let wh = abs_r * nh;
            let w_min = wc - wh;
            let w_max = wc + wh;

            if w_min.y > support_band_top {
                continue;
            }
            let p_min = Vec3::new(w_min.x, w_min.y - probe_pad, w_min.z);
            if !world.intersects_aabb_meters(p_min, w_max) {
                continue;
            }

            if node.count > 0 {
                let start = node.left_or_start as usize;
                let end = start + (node.count as usize);
                for sub in &self.collider.sub_boxes[start..end] {
                    let sc = self.position + r_mat * sub.center_rel_com;
                    let sh = abs_r * sub.half_extents;
                    let s_min = sc - sh;
                    let s_max = sc + sh;
                    let sp_min = Vec3::new(s_min.x, s_min.y - probe_pad, s_min.z);
                    if world.intersects_aabb_meters(sp_min, s_max) {
                        return true;
                    }
                }
            } else if sp + 2 <= stack.len() {
                stack[sp] = node.right_child;
                stack[sp + 1] = node.left_or_start;
                sp += 2;
            }
        }

        false
    }

    /// Generates exact box-vs-terrain and box-vs-bedrock contact constraints using the compound BVH.
    #[allow(dead_code)]
    pub fn generate_world_contacts(
        &self,
        world: &WorldMap,
        inv_i_world: Mat3,
        r_mat: Mat3,
    ) -> (Vec<ContactManifold>, f32) {
        let (contacts, max_impact_speed, _) =
            self.generate_world_contacts_detailed(world, inv_i_world, r_mat);
        (contacts, max_impact_speed)
    }

    /// Generates exact box-vs-terrain contacts plus a flag indicating whether any contact is wet/submerged.
    pub fn generate_world_contacts_detailed(
        &self,
        world: &WorldMap,
        inv_i_world: Mat3,
        r_mat: Mat3,
    ) -> (Vec<ContactManifold>, f32, bool) {
        if self.collider.bvh_nodes.is_empty() {
            return (Vec::new(), 0.0, false);
        }

        let abs_r = Mat3::from_cols(r_mat.x_axis.abs(), r_mat.y_axis.abs(), r_mat.z_axis.abs());
        let vpm = VOXELS_PER_METER as f32;
        let s = VOXEL_SIZE;

        let mut clusters: Vec<TerrainNormalCluster> = Vec::new();
        let mut any_wet_contact = false;

        let mut stack = [0u32; 32];
        let mut sp = 1usize;

        while sp > 0 {
            sp -= 1;
            let node = &self.collider.bvh_nodes[stack[sp] as usize];
            let nc = (node.min_rel_com + node.max_rel_com) * 0.5;
            let nh = (node.max_rel_com - node.min_rel_com) * 0.5;
            let wc = self.position + r_mat * nc;
            let wh = abs_r * nh;
            let w_min = wc - wh;
            let w_max = wc + wh;

            if w_min.y > BEDROCK_TOP_METERS && !world.intersects_aabb_meters(w_min, w_max) {
                continue;
            }

            if node.count == 0 {
                if sp + 2 <= stack.len() {
                    stack[sp] = node.right_child;
                    stack[sp + 1] = node.left_or_start;
                    sp += 2;
                }
                continue;
            }

            let start = node.left_or_start as usize;
            let end = start + (node.count as usize);
            for sub in &self.collider.sub_boxes[start..end] {
                let sb_obb = Obb::from_sub_box(sub, self.position, r_mat);
                let sb_half_w = abs_r * sub.half_extents;
                let sb_min = sb_obb.center - sb_half_w;
                let sb_max = sb_obb.center + sb_half_w;

                // 1. Analytic Bedrock Plane (y <= 5 * VOXEL_SIZE):
                if sb_min.y < BEDROCK_TOP_METERS {
                    let bed_min = Vec3::new(sb_min.x - 1.0, -2.0, sb_min.z - 1.0);
                    let bed_max = Vec3::new(sb_max.x + 1.0, BEDROCK_TOP_METERS, sb_max.z + 1.0);
                    let bed_obb = Obb::from_aabb(bed_min, bed_max);
                    if let Some(raw) = compute_subbox_obb_manifold(
                        &sb_obb,
                        sub.exposed_faces,
                        &bed_obb,
                        FACE_POS_Y,
                        -Vec3::Y,
                    ) {
                        // INVARIANT: `raw.normal` points Body -> World; negate so terrain reaction pushes Body outward.
                        insert_terrain_raw_contact(
                            &mut clusters,
                            -raw.normal,
                            &raw.points[0..raw.point_count],
                        );
                    }
                }

                // 2. Static WorldMap Voxels (vy >= 5):
                if sb_max.y <= BEDROCK_TOP_METERS || !world.intersects_aabb_meters(sb_min, sb_max) {
                    continue;
                }

                let v_min_x = (sb_min.x * vpm).floor() as i32;
                let v_max_x = ((sb_max.x - 1e-6) * vpm).floor() as i32;
                let v_min_y = ((sb_min.y * vpm).floor() as i32).max(5);
                let v_max_y = ((sb_max.y - 1e-6) * vpm).floor() as i32;
                let v_min_z = (sb_min.z * vpm).floor() as i32;
                let v_max_z = ((sb_max.z - 1e-6) * vpm).floor() as i32;

                if v_min_x > v_max_x || v_min_y > v_max_y || v_min_z > v_max_z {
                    continue;
                }

                let bx_min = v_min_x.div_euclid(4);
                let bx_max = v_max_x.div_euclid(4);
                let by_min = v_min_y.div_euclid(4);
                let by_max = v_max_y.div_euclid(4);
                let bz_min = v_min_z.div_euclid(4);
                let bz_max = v_max_z.div_euclid(4);

                for bz in bz_min..=bz_max {
                    let base_z = bz * 4;
                    let wz0 = (v_min_z - base_z).clamp(0, 3) as u32;
                    let wz1 = (v_max_z - base_z).clamp(0, 3) as u32;

                    for by in by_min..=by_max {
                        let base_y = by * 4;
                        let wy0 = (v_min_y - base_y).clamp(0, 3) as u32;
                        let wy1 = (v_max_y - base_y).clamp(0, 3) as u32;

                        for bx in bx_min..=bx_max {
                            let solid_mask = world.get_brick_solid_mask(bx, by, bz);
                            if solid_mask == 0 {
                                continue;
                            }

                            let base_x = bx * 4;
                            let wx0 = (v_min_x - base_x).clamp(0, 3) as u32;
                            let wx1 = (v_max_x - base_x).clamp(0, 3) as u32;

                            let win_mask = brick_window_mask(wx0, wx1, wy0, wy1, wz0, wz1);
                            let mut rem = solid_mask & win_mask;
                            if rem == 0 {
                                continue;
                            }

                            let mut nb_cache = NeighborBrickCache::default();

                            while rem != 0 {
                                let first_bit = rem.trailing_zeros();
                                let (lx0, ly0, lz0) = unpack_child_index(first_bit);

                                let mut lx1 = lx0 + 1;
                                while lx1 <= wx1
                                    && (rem & (1u64 << child_bit_index(lx1, ly0, lz0))) != 0
                                {
                                    lx1 += 1;
                                }
                                let x_nib = ((1u64 << (lx1 - lx0)) - 1) << lx0;

                                let mut ly1 = ly0 + 1;
                                while ly1 <= wy1 {
                                    let row_shift = (lz0 << 4) | (ly1 << 2);
                                    if ((rem >> row_shift) & x_nib) != x_nib {
                                        break;
                                    }
                                    ly1 += 1;
                                }

                                let mut xy_slice = 0u64;
                                for ly in ly0..ly1 {
                                    xy_slice |= x_nib << (ly << 2);
                                }

                                let mut lz1 = lz0 + 1;
                                while lz1 <= wz1 {
                                    if ((rem >> (lz1 << 4)) & xy_slice) != xy_slice {
                                        break;
                                    }
                                    lz1 += 1;
                                }

                                for lz in lz0..lz1 {
                                    rem &= !(xy_slice << (lz << 4));
                                }

                                let world_exposed = compute_brick_run_exposed_faces(
                                    solid_mask,
                                    world,
                                    (bx, by, bz),
                                    &mut nb_cache,
                                    (lx0, lx1),
                                    (ly0, ly1),
                                    (lz0, lz1),
                                );
                                if world_exposed == 0 {
                                    continue;
                                }

                                let run_min = Vec3::new(
                                    (base_x + lx0 as i32) as f32,
                                    (base_y + ly0 as i32) as f32,
                                    (base_z + lz0 as i32) as f32,
                                ) * s;
                                let run_max = Vec3::new(
                                    (base_x + lx1 as i32) as f32,
                                    (base_y + ly1 as i32) as f32,
                                    (base_z + lz1 as i32) as f32,
                                ) * s;
                                let run_obb = Obb::from_aabb(run_min, run_max);
                                let fallback_dir =
                                    (run_obb.center - sb_obb.center).normalize_or_zero();

                                if let Some(raw) = compute_subbox_obb_manifold(
                                    &sb_obb,
                                    sub.exposed_faces,
                                    &run_obb,
                                    world_exposed,
                                    fallback_dir,
                                ) {
                                    insert_terrain_raw_contact(
                                        &mut clusters,
                                        -raw.normal,
                                        &raw.points[0..raw.point_count],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        if clusters.is_empty() {
            return (Vec::new(), 0.0, false);
        }

        // WHY: Sort clusters descending by deepest penetration before truncation so bowl wall contacts never flicker by BVH order.
        clusters.sort_by(|a, b| b.max_pen.total_cmp(&a.max_pen));
        clusters.truncate(MAX_TERRAIN_CLUSTERS);

        let mut contacts = Vec::with_capacity(16);
        let mut max_impact_speed = 0.0f32;

        for cluster in clusters {
            let normal = cluster.unit_normal;
            let (tangent1, tangent2) = orthonormal_basis(normal);
            let reduced = reduce_contact_samples(
                &cluster.samples,
                tangent1,
                tangent2,
                MAX_CONTACTS_PER_TERRAIN_CLUSTER,
            );

            for (pos, raw_pen) in reduced {
                let r_world = pos - self.position;
                let penetration = (raw_pen - CONTACT_SKIN_METERS).max(0.0);

                if !any_wet_contact {
                    let cv = (pos * vpm).floor().as_ivec3();
                    if world.get_voxel(cv.x, cv.y, cv.z) == MAT_WATER_CRYSTAL
                        || world.get_voxel(cv.x, cv.y + 1, cv.z) == MAT_WATER_CRYSTAL
                    {
                        any_wet_contact = true;
                    }
                }

                let v_point = self.linear_velocity + self.angular_velocity.cross(r_world);
                let v_rel_n = v_point.dot(normal);
                if v_rel_n < 0.0 {
                    max_impact_speed = max_impact_speed.max(-v_rel_n);
                }

                let rn = r_world.cross(normal);
                let denom_n = self.inv_mass + rn.dot(inv_i_world * rn);
                if denom_n < 1e-8 {
                    continue;
                }
                let effective_mass_n = 1.0 / denom_n;

                let rt1 = r_world.cross(tangent1);
                let denom_t1 = self.inv_mass + rt1.dot(inv_i_world * rt1);
                let effective_mass_t1 = if denom_t1 > 1e-8 { 1.0 / denom_t1 } else { 0.0 };

                let rt2 = r_world.cross(tangent2);
                let denom_t2 = self.inv_mass + rt2.dot(inv_i_world * rt2);
                let effective_mass_t2 = if denom_t2 > 1e-8 { 1.0 / denom_t2 } else { 0.0 };

                contacts.push(ContactManifold {
                    r_world,
                    normal,
                    penetration,
                    accum_n: 0.0,
                    accum_t1: 0.0,
                    accum_t2: 0.0,
                    tangent1,
                    tangent2,
                    effective_mass_n,
                    effective_mass_t1,
                    effective_mass_t2,
                });
            }
        }

        (contacts, max_impact_speed, any_wet_contact)
    }
}

/// Computes point-vs-terrain feature contact normal and raw penetration depth for point queries.
#[allow(dead_code)]
pub fn compute_feature_contact(
    p_world: Vec3,
    vx: i32,
    vy: i32,
    vz: i32,
    world: &WorldMap,
) -> (Vec3, f32) {
    if vy <= 4 {
        // INVARIANT: Bedrock surface forms a flat horizontal plane at y = 5.0 * VOXEL_SIZE.
        let raw_pen = (BEDROCK_TOP_METERS - p_world.y).max(0.0);
        return (Vec3::Y, raw_pen);
    }

    let is_solid = |mat| mat != MAT_AIR && mat != MAT_WATER_CRYSTAL;
    if !is_solid(world.get_voxel(vx, vy, vz)) {
        return (Vec3::Y, 0.0);
    }

    let mut top_vy = vy;
    while top_vy < vy + 16 && is_solid(world.get_voxel(vx, top_vy + 1, vz)) {
        top_vy += 1;
    }
    let voxel_top_y = (top_vy as f32 + 1.0) * VOXEL_SIZE;
    let vertical_pen = (voxel_top_y - p_world.y).max(0.0);

    let min_x = vx as f32 * VOXEL_SIZE;
    let max_x = (vx + 1) as f32 * VOXEL_SIZE;
    let min_y = vy as f32 * VOXEL_SIZE;
    let min_z = vz as f32 * VOXEL_SIZE;
    let max_z = (vz + 1) as f32 * VOXEL_SIZE;

    let dist_px = (max_x - p_world.x).max(0.0);
    let dist_nx = (p_world.x - min_x).max(0.0);
    let dist_ny = (p_world.y - min_y).max(0.0);
    let dist_pz = (max_z - p_world.z).max(0.0);
    let dist_nz = (p_world.z - min_z).max(0.0);

    let air_px = !is_solid(world.get_voxel(vx + 1, vy, vz));
    let air_nx = !is_solid(world.get_voxel(vx - 1, vy, vz));
    let air_ny = vy > 5 && !is_solid(world.get_voxel(vx, vy - 1, vz));
    let air_pz = !is_solid(world.get_voxel(vx, vy, vz + 1));
    let air_nz = !is_solid(world.get_voxel(vx, vy, vz - 1));

    let candidate_x = match (air_px, air_nx) {
        (true, true) => {
            if dist_px <= dist_nx {
                Some((Vec3::X, dist_px))
            } else {
                Some((-Vec3::X, dist_nx))
            }
        }
        (true, false) => Some((Vec3::X, dist_px)),
        (false, true) => Some((-Vec3::X, dist_nx)),
        (false, false) => None,
    };

    let candidate_z = match (air_pz, air_nz) {
        (true, true) => {
            if dist_pz <= dist_nz {
                Some((Vec3::Z, dist_pz))
            } else {
                Some((-Vec3::Z, dist_nz))
            }
        }
        (true, false) => Some((Vec3::Z, dist_pz)),
        (false, true) => Some((-Vec3::Z, dist_nz)),
        (false, false) => None,
    };

    let mut best_non_top = match (candidate_x, candidate_z) {
        (Some((nx, dx)), Some((nz, dz))) => {
            if dx <= dz {
                Some((nx, dx))
            } else {
                Some((nz, dz))
            }
        }
        (Some((nx, dx)), None) => Some((nx, dx)),
        (None, Some((nz, dz))) => Some((nz, dz)),
        (None, None) => None,
    };

    if air_ny {
        best_non_top = match best_non_top {
            Some((n, d)) if d <= dist_ny => Some((n, d)),
            _ => Some((-Vec3::Y, dist_ny)),
        };
    }

    let has_exposed_top = !is_solid(world.get_voxel(vx, vy + 1, vz));
    if !has_exposed_top {
        if let Some((normal, dist)) = best_non_top {
            if dist < vertical_pen {
                return (normal, dist);
            }
        }
    }

    (Vec3::Y, vertical_pen)
}

#[inline]
fn brick_window_mask(wx0: u32, wx1: u32, wy0: u32, wy1: u32, wz0: u32, wz1: u32) -> u64 {
    let x_nib = ((1u64 << (wx1 - wx0 + 1)) - 1) << wx0;
    let mut xy_slice = 0u64;
    for y in wy0..=wy1 {
        xy_slice |= x_nib << (y << 2);
    }
    let mut mask = 0u64;
    for z in wz0..=wz1 {
        mask |= xy_slice << (z << 4);
    }
    mask
}

fn compute_brick_run_exposed_faces(
    solid_mask: u64,
    world: &WorldMap,
    (bx, by, bz): (i32, i32, i32),
    nb_cache: &mut NeighborBrickCache,
    (lx0, lx1): (u32, u32),
    (ly0, ly1): (u32, u32),
    (lz0, lz1): (u32, u32),
) -> u8 {
    let mut mask = 0u8;

    // -X:
    if lx0 > 0 {
        let px = lx0 - 1;
        'nx_in: for lz in lz0..lz1 {
            for ly in ly0..ly1 {
                if (solid_mask & (1u64 << child_bit_index(px, ly, lz))) == 0 {
                    mask |= FACE_NEG_X;
                    break 'nx_in;
                }
            }
        }
    } else {
        let nb = nb_cache.get_nx(world, bx, by, bz);
        'nx_nb: for lz in lz0..lz1 {
            for ly in ly0..ly1 {
                if (nb & (1u64 << child_bit_index(3, ly, lz))) == 0 {
                    mask |= FACE_NEG_X;
                    break 'nx_nb;
                }
            }
        }
    }

    // +X:
    if lx1 < 4 {
        'px_in: for lz in lz0..lz1 {
            for ly in ly0..ly1 {
                if (solid_mask & (1u64 << child_bit_index(lx1, ly, lz))) == 0 {
                    mask |= FACE_POS_X;
                    break 'px_in;
                }
            }
        }
    } else {
        let nb = nb_cache.get_px(world, bx, by, bz);
        'px_nb: for lz in lz0..lz1 {
            for ly in ly0..ly1 {
                if (nb & (1u64 << child_bit_index(0, ly, lz))) == 0 {
                    mask |= FACE_POS_X;
                    break 'px_nb;
                }
            }
        }
    }

    // -Y (accounting for solid bedrock at vy <= 4):
    if by * 4 + (ly0 as i32) > 5 {
        if ly0 > 0 {
            let py = ly0 - 1;
            'ny_in: for lz in lz0..lz1 {
                for lx in lx0..lx1 {
                    if (solid_mask & (1u64 << child_bit_index(lx, py, lz))) == 0 {
                        mask |= FACE_NEG_Y;
                        break 'ny_in;
                    }
                }
            }
        } else {
            let nb = nb_cache.get_ny(world, bx, by, bz);
            'ny_nb: for lz in lz0..lz1 {
                for lx in lx0..lx1 {
                    if (nb & (1u64 << child_bit_index(lx, 3, lz))) == 0 {
                        mask |= FACE_NEG_Y;
                        break 'ny_nb;
                    }
                }
            }
        }
    }

    // +Y:
    if ly1 < 4 {
        'py_in: for lz in lz0..lz1 {
            for lx in lx0..lx1 {
                if (solid_mask & (1u64 << child_bit_index(lx, ly1, lz))) == 0 {
                    mask |= FACE_POS_Y;
                    break 'py_in;
                }
            }
        }
    } else {
        let nb = nb_cache.get_py(world, bx, by, bz);
        'py_nb: for lz in lz0..lz1 {
            for lx in lx0..lx1 {
                if (nb & (1u64 << child_bit_index(lx, 0, lz))) == 0 {
                    mask |= FACE_POS_Y;
                    break 'py_nb;
                }
            }
        }
    }

    // -Z:
    if lz0 > 0 {
        let pz = lz0 - 1;
        'nz_in: for ly in ly0..ly1 {
            for lx in lx0..lx1 {
                if (solid_mask & (1u64 << child_bit_index(lx, ly, pz))) == 0 {
                    mask |= FACE_NEG_Z;
                    break 'nz_in;
                }
            }
        }
    } else {
        let nb = nb_cache.get_nz(world, bx, by, bz);
        'nz_nb: for ly in ly0..ly1 {
            for lx in lx0..lx1 {
                if (nb & (1u64 << child_bit_index(lx, ly, 3))) == 0 {
                    mask |= FACE_NEG_Z;
                    break 'nz_nb;
                }
            }
        }
    }

    // +Z:
    if lz1 < 4 {
        'pz_in: for ly in ly0..ly1 {
            for lx in lx0..lx1 {
                if (solid_mask & (1u64 << child_bit_index(lx, ly, lz1))) == 0 {
                    mask |= FACE_POS_Z;
                    break 'pz_in;
                }
            }
        }
    } else {
        let nb = nb_cache.get_pz(world, bx, by, bz);
        'pz_nb: for ly in ly0..ly1 {
            for lx in lx0..lx1 {
                if (nb & (1u64 << child_bit_index(lx, ly, 0))) == 0 {
                    mask |= FACE_POS_Z;
                    break 'pz_nb;
                }
            }
        }
    }

    mask
}

fn insert_terrain_raw_contact(
    clusters: &mut Vec<TerrainNormalCluster>,
    terrain_normal: Vec3,
    pts: &[(Vec3, f32)],
) {
    let mut best_idx = None;
    let mut best_dot = 0.85f32;
    for (idx, c) in clusters.iter().enumerate() {
        let d = c.unit_normal.dot(terrain_normal);
        if d > best_dot {
            best_dot = d;
            best_idx = Some(idx);
        }
    }

    if let Some(idx) = best_idx {
        let c = &mut clusters[idx];
        let mut w = 0.0f32;
        for &(pos, pen) in pts {
            c.max_pen = c.max_pen.max(pen);
            w += pen.max(1e-3);
            c.samples.push((pos, pen));
        }
        c.normal_sum += terrain_normal * w;
        if c.normal_sum.length_squared() > 1e-8 {
            c.unit_normal = c.normal_sum.normalize();
        }
    } else {
        let mut samples = Vec::with_capacity(8);
        let mut max_pen = 0.0f32;
        let mut w = 0.0f32;
        for &(pos, pen) in pts {
            max_pen = max_pen.max(pen);
            w += pen.max(1e-3);
            samples.push((pos, pen));
        }
        clusters.push(TerrainNormalCluster {
            normal_sum: terrain_normal * w.max(1e-3),
            unit_normal: terrain_normal,
            max_pen,
            samples,
        });
    }
}
