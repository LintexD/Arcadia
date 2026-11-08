//! Construction, discrete inertia tensor computation, and compound collider synthesis.
//! Owns CPU mass/inertia integration, exact `CompoundCollider` construction, and mass-scaled detachment impulses.

use super::collision::compound::CompoundCollider;
use super::types::RigidVoxelBody;
use arcadia_core::{ChunkTree64, MaterialId, MaterialPalette, MAT_AIR, VOXEL_SIZE};
use glam::{IVec3, Mat3, Quat, Vec3};
use std::sync::atomic::{AtomicU32, Ordering};

static NEXT_BODY_ID: AtomicU32 = AtomicU32::new(1);

impl RigidVoxelBody {
    /// Constructs a `RigidVoxelBody` from detached global voxel coordinates `(global_v, mat_id)` with exact compound collision.
    pub fn from_global_voxels(
        global_voxels: &[(IVec3, MaterialId)],
        palette: &MaterialPalette,
        blast_origin_m: Option<Vec3>,
    ) -> Option<Self> {
        if global_voxels.is_empty() {
            return None;
        }

        let mut min_v = IVec3::splat(i32::MAX);
        let mut max_v = IVec3::splat(i32::MIN);
        for &(gv, mat) in global_voxels {
            if mat != MAT_AIR {
                min_v = min_v.min(gv);
                max_v = max_v.max(gv);
            }
        }
        if min_v.x > max_v.x {
            return None;
        }

        let raw_dims = max_v - min_v + IVec3::ONE;
        let dims_v = raw_dims.min(IVec3::splat(256));

        let mut local_voxels = Vec::with_capacity(global_voxels.len());
        let mut occupied_set = std::collections::HashSet::with_capacity(global_voxels.len() * 2);

        for &(gv, mat) in global_voxels {
            if mat == MAT_AIR {
                continue;
            }
            let lv = gv - min_v;
            if lv.x >= 0
                && lv.x < dims_v.x
                && lv.y >= 0
                && lv.y < dims_v.y
                && lv.z >= 0
                && lv.z < dims_v.z
            {
                let lx = lv.x as u32;
                let ly = lv.y as u32;
                let lz = lv.z as u32;
                if occupied_set.insert((lx as i32, ly as i32, lz as i32)) {
                    local_voxels.push((lx, ly, lz, mat));
                }
            }
        }

        if local_voxels.is_empty() {
            return None;
        }

        let s = VOXEL_SIZE;
        let m0 = 2400.0 * s * s * s;
        let n = local_voxels.len() as f32;
        let total_mass = (n * m0).max(1e-3);
        let inv_mass = 1.0 / total_mass;

        let mut sum_pos = Vec3::ZERO;
        for &(lx, ly, lz, _) in &local_voxels {
            let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
            sum_pos += p_local;
        }
        let com_local_m = sum_pos / n;

        let (inertia_body, inv_inertia_body) =
            compute_inertia_tensor(&local_voxels, com_local_m, total_mass, m0, s);

        let boundary_voxels = Self::extract_boundary_voxels(&local_voxels, dims_v, com_local_m);
        let collider = CompoundCollider::build(&local_voxels, dims_v, com_local_m);
        let local_tree = ChunkTree64::from_sparse_voxels(&local_voxels, palette);

        let origin_world_m = min_v.as_vec3() * s;
        let world_com = origin_world_m + com_local_m;
        let half_extents_m = dims_v.as_vec3() * (0.5 * s);
        let emissive_light = compute_emissive_light(&local_voxels, palette, com_local_m, s);

        let (init_vel, init_omega) = if let Some(blast_pos) = blast_origin_m {
            let offset = world_com - blast_pos;
            let dist = offset.length().max(0.25);
            let dir = offset / dist;
            let horiz = Vec3::new(offset.x, 0.0, offset.z);
            let horiz_dist = horiz.length();
            // WHY: Extract pure XZ unit direction at the cut plane so near-center cuts still generate full overturning torque without yaw spin.
            let lateral_dir = if horiz_dist > 1e-3 {
                horiz / horiz_dist
            } else {
                Vec3::new(0.8, 0.0, -0.6)
            };
            // WHY: Couple CoM velocity to base-pivot rotation so floor friction does not annihilate 75% of angular momentum on stump contact.
            let ref_mass = 140.0;
            let lin_scale = (ref_mass / total_mass.max(ref_mass))
                .powf(0.40)
                .clamp(0.18, 1.0);
            let rot_scale = (ref_mass / total_mass.max(ref_mass))
                .powf(0.18)
                .clamp(0.50, 1.0);
            let raw_push_speed = (2.2 / dist.sqrt()).clamp(0.4, 4.0);
            let push_speed = raw_push_speed * lin_scale;
            let push_vel_y = (dir.y * 0.2 * push_speed).clamp(-push_speed, 0.05);

            let r_lever = blast_pos - world_com;
            let f_blast = Vec3::new(
                lateral_dir.x * raw_push_speed,
                0.0,
                lateral_dir.z * raw_push_speed,
            ) * total_mass;
            let torque = r_lever.cross(f_blast);
            let ang = (inv_inertia_body * torque) * rot_scale;
            let ang_speed = ang.length();
            let min_omega = 0.65 * rot_scale;
            let max_omega = 2.4 * rot_scale;
            let ang_clamped = if ang_speed > 1e-4 {
                ang * (ang_speed.clamp(min_omega, max_omega) / ang_speed)
            } else {
                lateral_dir.cross(Vec3::Y) * min_omega
            };

            let h_above_base = (world_com.y - origin_world_m.y).max(0.0);
            let lin = if offset.y > 0.0 && h_above_base > 0.20 {
                let pivot_vel =
                    (ang_clamped.cross(Vec3::Y * h_above_base) * 0.58).clamp_length_max(1.15);
                Vec3::new(pivot_vel.x, 0.0, pivot_vel.z)
            } else {
                let horiz_slide_scale = 0.35 + 0.65 * (horiz_dist / 0.35).min(1.0);
                Vec3::new(
                    lateral_dir.x * push_speed * horiz_slide_scale,
                    push_vel_y,
                    lateral_dir.z * push_speed * horiz_slide_scale,
                )
            };
            (lin, ang_clamped)
        } else {
            (Vec3::ZERO, Vec3::ZERO)
        };

        Some(Self {
            id: NEXT_BODY_ID.fetch_add(1, Ordering::Relaxed),
            is_baking: false,
            local_voxels,
            boundary_voxels,
            collider,
            dims_v,
            local_tree,
            gpu_root_node_idx: arcadia_world::EMPTY_CHUNK_SENTINEL,
            com_local_m,
            half_extents_m,
            mass: total_mass,
            inv_mass,
            inertia_body,
            inv_inertia_body,
            position: world_com,
            orientation: Quat::IDENTITY,
            prev_position: world_com,
            prev_orientation: Quat::IDENTITY,
            linear_velocity: init_vel,
            angular_velocity: init_omega,
            restitution: 0.24,
            friction: 0.62,
            sleep_timer: 0.0,
            sleeping: false,
            submersion_ratio: 0.0,
            age: 0.0,
            rest_time: 0.0,
            emissive_light,
        })
    }

    /// Constructs a `RigidVoxelBody` from a detached component of parent-local voxels, preserving exact orientation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_local_component(
        split_voxels: &[(u32, u32, u32, MaterialId)],
        parent_position: Vec3,
        parent_orientation: Quat,
        parent_com_local_m: Vec3,
        parent_linear_velocity: Vec3,
        parent_angular_velocity: Vec3,
        blast_origin_m: Option<Vec3>,
        palette: &MaterialPalette,
    ) -> Option<Self> {
        if split_voxels.is_empty() {
            return None;
        }

        let mut min_c = IVec3::splat(i32::MAX);
        let mut max_c = IVec3::splat(i32::MIN);
        for &(lx, ly, lz, mat) in split_voxels {
            if mat != MAT_AIR {
                let c = IVec3::new(lx as i32, ly as i32, lz as i32);
                min_c = min_c.min(c);
                max_c = max_c.max(c);
            }
        }
        if min_c.x > max_c.x {
            return None;
        }

        let raw_dims = max_c - min_c + IVec3::ONE;
        let dims_v = raw_dims.min(IVec3::splat(256));

        let mut local_voxels = Vec::with_capacity(split_voxels.len());
        for &(lx, ly, lz, mat) in split_voxels {
            if mat != MAT_AIR {
                let cx = (lx as i32 - min_c.x) as u32;
                let cy = (ly as i32 - min_c.y) as u32;
                let cz = (lz as i32 - min_c.z) as u32;
                if cx < 256 && cy < 256 && cz < 256 {
                    local_voxels.push((cx, cy, cz, mat));
                }
            }
        }

        if local_voxels.is_empty() {
            return None;
        }

        let s = VOXEL_SIZE;
        let m0 = 2400.0 * s * s * s;
        let total_mass = (local_voxels.len() as f32 * m0).max(1e-3);
        let inv_mass = 1.0 / total_mass;

        let mut sum_pos = Vec3::ZERO;
        for &(lx, ly, lz, _) in &local_voxels {
            let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
            sum_pos += p_local;
        }
        let com_local_m = sum_pos / (local_voxels.len() as f32);

        let (inertia_body, inv_inertia_body) =
            compute_inertia_tensor(&local_voxels, com_local_m, total_mass, m0, s);

        let boundary_voxels = Self::extract_boundary_voxels(&local_voxels, dims_v, com_local_m);
        let collider = CompoundCollider::build(&local_voxels, dims_v, com_local_m);
        let local_tree = ChunkTree64::from_sparse_voxels(&local_voxels, palette);

        let rot = Mat3::from_quat(parent_orientation);
        let com_in_parent_local = min_c.as_vec3() * s + com_local_m;
        let world_com = parent_position + rot * (com_in_parent_local - parent_com_local_m);
        let half_extents_m = dims_v.as_vec3() * (0.5 * s);
        let emissive_light = compute_emissive_light(&local_voxels, palette, com_local_m, s);

        let r_split = world_com - parent_position;
        let mut linear_velocity = parent_linear_velocity + parent_angular_velocity.cross(r_split);
        let mut angular_velocity = parent_angular_velocity;

        if let Some(blast_pos) = blast_origin_m {
            let offset = world_com - blast_pos;
            let dist = offset.length().max(0.25);
            let dir = offset / dist;
            // WHY: Scale split-carve push inversely by mass and eliminate minimum angular-speed floor so severed upper pillars settle naturally.
            let ref_mass = 60.0;
            let mass_scale = (ref_mass / total_mass.max(ref_mass)).powf(0.75);
            let push_speed = (1.6 / dist.sqrt()).clamp(0.15, 2.2) * mass_scale;
            let push_dir_y = (dir.y * 0.15).clamp(-1.0, 0.08);
            let push_vel_y = (push_dir_y * push_speed).clamp(-push_speed, 0.05);
            let lin_blast = Vec3::new(dir.x * push_speed, push_vel_y, dir.z * push_speed);
            linear_velocity += lin_blast;

            let r_lever = blast_pos - world_com;
            let f_blast = lin_blast * total_mass;
            let torque = r_lever.cross(f_blast);
            let inv_i_world = rot * inv_inertia_body * rot.transpose();
            let ang = inv_i_world * torque;
            let ang_speed = ang.length();
            let max_omega = 2.4 * mass_scale;
            if ang_speed > 1e-4 {
                angular_velocity += ang * (ang_speed.min(max_omega) / ang_speed);
            }
        }

        Some(Self {
            id: NEXT_BODY_ID.fetch_add(1, Ordering::Relaxed),
            is_baking: false,
            local_voxels,
            boundary_voxels,
            collider,
            dims_v,
            local_tree,
            gpu_root_node_idx: arcadia_world::EMPTY_CHUNK_SENTINEL,
            com_local_m,
            half_extents_m,
            mass: total_mass,
            inv_mass,
            inertia_body,
            inv_inertia_body,
            position: world_com,
            orientation: parent_orientation,
            prev_position: world_com,
            prev_orientation: parent_orientation,
            linear_velocity,
            angular_velocity,
            restitution: 0.24,
            friction: 0.62,
            sleep_timer: 0.0,
            sleeping: false,
            submersion_ratio: 0.0,
            age: 0.0,
            rest_time: 0.0,
            emissive_light,
        })
    }

    /// Returns the world-space point light `[x, y, z, intensity]` if this body contains emissive voxels.
    #[inline]
    pub fn world_emissive_light(&self) -> Option<[f32; 4]> {
        self.emissive_light.map(|(offset, radius)| {
            let r = Mat3::from_quat(self.orientation);
            let world_pos = self.position + r * offset;
            [world_pos.x, world_pos.y, world_pos.z, radius]
        })
    }

    /// Extracts up to 96 exterior 6-connected boundary voxels for debris splintering and legacy hull queries.
    pub fn extract_boundary_voxels(
        local_voxels: &[(u32, u32, u32, MaterialId)],
        dims_v: IVec3,
        com_local_m: Vec3,
    ) -> Vec<(Vec3, MaterialId)> {
        if local_voxels.is_empty() {
            return Vec::new();
        }

        let s = VOXEL_SIZE;
        let sy = dims_v.x as usize;
        let sz = sy * (dims_v.y as usize);
        let volume = sz * (dims_v.z as usize);

        let mut all_hull = Vec::with_capacity(local_voxels.len().min(128));
        let mut min_local_y = f32::INFINITY;

        if volume > 0 && volume <= 2_097_152 {
            let mut occupied_bits = vec![0u64; volume.div_ceil(64)];
            for &(lx, ly, lz, _) in local_voxels {
                let idx = (lx as usize) + (ly as usize) * sy + (lz as usize) * sz;
                occupied_bits[idx / 64] |= 1u64 << (idx % 64);
            }

            for &(lx, ly, lz, mat) in local_voxels {
                let idx = (lx as usize) + (ly as usize) * sy + (lz as usize) * sz;
                let is_boundary = lx == 0
                    || (occupied_bits[(idx - 1) / 64] & (1u64 << ((idx - 1) % 64))) == 0
                    || lx + 1 == dims_v.x as u32
                    || (occupied_bits[(idx + 1) / 64] & (1u64 << ((idx + 1) % 64))) == 0
                    || ly == 0
                    || (occupied_bits[(idx - sy) / 64] & (1u64 << ((idx - sy) % 64))) == 0
                    || ly + 1 == dims_v.y as u32
                    || (occupied_bits[(idx + sy) / 64] & (1u64 << ((idx + sy) % 64))) == 0
                    || lz == 0
                    || (occupied_bits[(idx - sz) / 64] & (1u64 << ((idx - sz) % 64))) == 0
                    || lz + 1 == dims_v.z as u32
                    || (occupied_bits[(idx + sz) / 64] & (1u64 << ((idx + sz) % 64))) == 0;

                if is_boundary {
                    let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
                    let r = p_local - com_local_m;
                    min_local_y = min_local_y.min(r.y);
                    all_hull.push((r, mat));
                }
            }
        } else {
            let mut occupied_set = std::collections::HashSet::with_capacity(local_voxels.len() * 2);
            for &(lx, ly, lz, _) in local_voxels {
                occupied_set.insert((lx as i32, ly as i32, lz as i32));
            }
            for &(lx, ly, lz, mat) in local_voxels {
                let ix = lx as i32;
                let iy = ly as i32;
                let iz = lz as i32;
                let is_boundary = !occupied_set.contains(&(ix - 1, iy, iz))
                    || !occupied_set.contains(&(ix + 1, iy, iz))
                    || !occupied_set.contains(&(ix, iy - 1, iz))
                    || !occupied_set.contains(&(ix, iy + 1, iz))
                    || !occupied_set.contains(&(ix, iy, iz - 1))
                    || !occupied_set.contains(&(ix, iy, iz + 1));
                if is_boundary {
                    let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
                    let r = p_local - com_local_m;
                    min_local_y = min_local_y.min(r.y);
                    all_hull.push((r, mat));
                }
            }
        }

        let max_contacts = 96usize;
        if all_hull.len() <= max_contacts {
            all_hull
        } else {
            let mut prioritized = Vec::with_capacity(max_contacts);
            let mut remainder = Vec::new();
            let base_cutoff = min_local_y + 2.0 * s;

            for (r, mat) in all_hull {
                if r.y <= base_cutoff && prioritized.len() < 32 {
                    prioritized.push((r, mat));
                } else {
                    remainder.push((r, mat));
                }
            }

            let needed = max_contacts.saturating_sub(prioritized.len());
            if !remainder.is_empty() && needed > 0 {
                let step = (remainder.len() as f32 / needed as f32).ceil() as usize;
                for (r, mat) in remainder.into_iter().step_by(step.max(1)) {
                    if prioritized.len() >= max_contacts {
                        break;
                    }
                    prioritized.push((r, mat));
                }
            }
            prioritized
        }
    }

    /// Recomputes mass, tight voxel bounds, discrete inertia tensor, and `CompoundCollider` after voxel edits.
    pub fn rebuild_after_carve(&mut self, palette: &MaterialPalette) {
        self.rebuild_after_carve_internal(palette, true);
    }

    /// Recomputes mass, tight voxel bounds, discrete inertia tensor, and `CompoundCollider` when `local_tree` was updated in-place.
    pub(crate) fn rebuild_after_carve_preserve_tree(&mut self, palette: &MaterialPalette) {
        self.rebuild_after_carve_internal(palette, false);
    }

    fn rebuild_after_carve_internal(&mut self, palette: &MaterialPalette, rebuild_tree: bool) {
        if self.local_voxels.is_empty() {
            return;
        }
        let s = VOXEL_SIZE;
        let m0 = 2400.0 * s * s * s;
        let n = self.local_voxels.len() as f32;
        self.mass = (n * m0).max(1e-3);
        self.inv_mass = 1.0 / self.mass;

        let mut min_coord = IVec3::splat(i32::MAX);
        let mut max_coord = IVec3::splat(i32::MIN);
        for &(lx, ly, lz, _) in &self.local_voxels {
            let c = IVec3::new(lx as i32, ly as i32, lz as i32);
            min_coord = min_coord.min(c);
            max_coord = max_coord.max(c);
        }

        let shift_m = if min_coord != IVec3::ZERO {
            for voxel in &mut self.local_voxels {
                voxel.0 = (voxel.0 as i32 - min_coord.x) as u32;
                voxel.1 = (voxel.1 as i32 - min_coord.y) as u32;
                voxel.2 = (voxel.2 as i32 - min_coord.z) as u32;
            }
            min_coord.as_vec3() * s
        } else {
            Vec3::ZERO
        };

        self.dims_v = max_coord - min_coord + IVec3::ONE;
        self.half_extents_m = self.dims_v.as_vec3() * (0.5 * s);

        let mut sum_pos = Vec3::ZERO;
        for &(lx, ly, lz, _) in &self.local_voxels {
            let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
            sum_pos += p_local;
        }
        let old_com = self.com_local_m;
        let new_com = sum_pos / n;
        let rot = Mat3::from_quat(self.orientation);
        // INVARIANT: Shifting `position`, `prev_position`, and `linear_velocity` by `delta_world_com` keeps every surviving voxel kinematically continuous.
        let delta_world_com = rot * ((new_com + shift_m) - old_com);
        self.position += delta_world_com;
        self.prev_position += delta_world_com;
        self.linear_velocity += self.angular_velocity.cross(delta_world_com);
        self.com_local_m = new_com;

        let (inertia_body, inv_inertia_body) =
            compute_inertia_tensor(&self.local_voxels, new_com, self.mass, m0, s);
        self.inertia_body = inertia_body;
        self.inv_inertia_body = inv_inertia_body;

        if rebuild_tree {
            self.local_tree = ChunkTree64::from_sparse_voxels(&self.local_voxels, palette);
        }
        self.boundary_voxels =
            Self::extract_boundary_voxels(&self.local_voxels, self.dims_v, self.com_local_m);
        self.collider = CompoundCollider::build(&self.local_voxels, self.dims_v, self.com_local_m);
        self.emissive_light = compute_emissive_light(&self.local_voxels, palette, new_com, s);
    }
}

fn compute_inertia_tensor(
    local_voxels: &[(u32, u32, u32, MaterialId)],
    com_local_m: Vec3,
    total_mass: f32,
    m0: f32,
    s: f32,
) -> (Mat3, Mat3) {
    let cube_corr = (s * s) / 6.0;
    let mut i_xx = 0.0f32;
    let mut i_yy = 0.0f32;
    let mut i_zz = 0.0f32;
    let mut i_xy = 0.0f32;
    let mut i_xz = 0.0f32;
    let mut i_yz = 0.0f32;

    for &(lx, ly, lz, _) in local_voxels {
        let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
        let r = p_local - com_local_m;
        i_xx += m0 * (r.y * r.y + r.z * r.z + cube_corr);
        i_yy += m0 * (r.x * r.x + r.z * r.z + cube_corr);
        i_zz += m0 * (r.x * r.x + r.y * r.y + cube_corr);
        i_xy -= m0 * (r.x * r.y);
        i_xz -= m0 * (r.x * r.z);
        i_yz -= m0 * (r.y * r.z);
    }

    let min_inertia = total_mass * cube_corr;
    i_xx = i_xx.max(min_inertia);
    i_yy = i_yy.max(min_inertia);
    i_zz = i_zz.max(min_inertia);

    let inertia_body = Mat3::from_cols(
        Vec3::new(i_xx, i_xy, i_xz),
        Vec3::new(i_xy, i_yy, i_yz),
        Vec3::new(i_xz, i_yz, i_zz),
    );
    let det = inertia_body.determinant();
    let inv_inertia_body = if det.abs() > 1e-8 {
        inertia_body.inverse()
    } else {
        Mat3::from_diagonal(Vec3::new(1.0 / i_xx, 1.0 / i_yy, 1.0 / i_zz))
    };
    (inertia_body, inv_inertia_body)
}

pub(crate) fn compute_emissive_light(
    local_voxels: &[(u32, u32, u32, MaterialId)],
    palette: &MaterialPalette,
    com_local_m: Vec3,
    s: f32,
) -> Option<(Vec3, f32)> {
    let mut sum_emissive_pos = Vec3::ZERO;
    let mut total_emission_power = 0.0f32;
    for &(lx, ly, lz, mat) in local_voxels {
        let m = palette.get(mat);
        let em = Vec3::new(m.emission_meta[0], m.emission_meta[1], m.emission_meta[2]);
        let em_strength = em.length();
        if em_strength > 0.1 {
            let p_local = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5) * s;
            sum_emissive_pos += p_local * em_strength;
            total_emission_power += em_strength;
        }
    }
    if total_emission_power > 0.1 {
        let em_centroid_local = (sum_emissive_pos / total_emission_power) - com_local_m;
        let radius = (total_emission_power * 0.18).clamp(0.8, 6.0);
        Some((em_centroid_local, radius))
    } else {
        None
    }
}
