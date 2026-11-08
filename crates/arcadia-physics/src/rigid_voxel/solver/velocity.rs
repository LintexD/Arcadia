//! Sequential impulse velocity constraints, warm starting, and shock propagation passes.

use super::sleep::{CachedTerrainContact, TerrainContactData};
use super::{MAX_PENETRATION_BIAS_VEL, PENETRATION_SLOP};
use crate::rigid_voxel::types::{BodyPairManifold, ContactManifold, RigidVoxelBody};
use glam::{Mat3, Vec3};

const WARM_START_FACTOR: f32 = 0.80;
const WARM_START_MATCH_DIST_SQ: f32 = 0.035 * 0.035;

impl RigidVoxelBody {
    /// Resolves mutual collision and dynamic contact impulses across all directional compound manifolds for `(body_a, body_b)`.
    pub fn resolve_collision_pair(body_a: &mut Self, body_b: &mut Self) {
        if !Self::check_broadphase_overlap(body_a, body_b) {
            return;
        }

        let mut manifolds = Self::compute_compound_pair_manifolds_indexed(body_a, body_b, 0, 1);
        if manifolds.is_empty() {
            return;
        }

        body_a.wake_up();
        body_b.wake_up();

        for manifold in &mut manifolds {
            Self::prepare_pair_manifold(manifold, body_a, body_b);
            Self::apply_pair_warm_start(manifold, body_a, body_b);
        }

        let solver_iterations = 6;
        let sub_dt = 0.016 / 4.0;
        for iter in 0..solver_iterations {
            for manifold in &mut manifolds {
                Self::solve_pair_manifold_iteration(manifold, body_a, body_b, sub_dt, iter);
            }
        }
    }

    /// Prepares constraint denominators and effective inverse masses for all active contact points.
    pub fn prepare_pair_manifold(manifold: &mut BodyPairManifold, body_a: &Self, body_b: &Self) {
        Self::prepare_pair_manifold_with_masses(
            manifold,
            body_a.inv_mass,
            body_a.world_inv_inertia(),
            body_b.inv_mass,
            body_b.world_inv_inertia(),
        );
    }

    /// Applies warm-started normal and tangential impulses from previous substep solutions.
    pub fn apply_pair_warm_start(
        manifold: &BodyPairManifold,
        body_a: &mut Self,
        body_b: &mut Self,
    ) {
        let inv_i_a = body_a.world_inv_inertia();
        let inv_i_b = body_b.world_inv_inertia();
        let normal = manifold.normal;
        let tangent1 = manifold.tangent1;
        let tangent2 = manifold.tangent2;

        for pt in &manifold.points[0..manifold.point_count] {
            let p = normal * pt.accum_n + tangent1 * pt.accum_t1 + tangent2 * pt.accum_t2;
            if p.length_squared() > 1e-12 {
                // INVARIANT: Normal points from A to B. Reaction force pushes A along -P and B along +P.
                body_a.linear_velocity -= p * body_a.inv_mass;
                body_a.angular_velocity -= inv_i_a * pt.r_a.cross(p);
                body_b.linear_velocity += p * body_b.inv_mass;
                body_b.angular_velocity += inv_i_b * pt.r_b.cross(p);
            }
        }
    }

    /// Solves one standalone iteration of normal contact separation (with clamped Baumgarte bias) and 2D Coulomb friction.
    pub fn solve_pair_manifold_iteration(
        manifold: &mut BodyPairManifold,
        body_a: &mut Self,
        body_b: &mut Self,
        sub_dt: f32,
        iter: usize,
    ) {
        Self::solve_pair_manifold_velocity_iteration(manifold, body_a, body_b, Some(sub_dt), iter);
    }

    #[allow(dead_code)]
    pub fn solve_terrain_contacts_iteration(
        &mut self,
        contacts: &mut [ContactManifold],
        inv_i_world: Mat3,
        sub_dt: f32,
        restitution_coeff: f32,
        iter: usize,
    ) {
        self.solve_terrain_contacts_velocity_iteration(
            contacts,
            inv_i_world,
            Some(sub_dt),
            restitution_coeff,
            iter,
            None,
        );
    }

    pub(crate) fn prepare_pair_manifold_active(
        manifold: &mut BodyPairManifold,
        body_a: &Self,
        body_b: &Self,
    ) {
        let inv_mass_a = if body_a.sleeping {
            0.0
        } else {
            body_a.inv_mass
        };
        let inv_i_a = if body_a.sleeping {
            Mat3::ZERO
        } else {
            body_a.world_inv_inertia()
        };
        let inv_mass_b = if body_b.sleeping {
            0.0
        } else {
            body_b.inv_mass
        };
        let inv_i_b = if body_b.sleeping {
            Mat3::ZERO
        } else {
            body_b.world_inv_inertia()
        };
        Self::prepare_pair_manifold_with_masses(manifold, inv_mass_a, inv_i_a, inv_mass_b, inv_i_b);
    }

    pub(crate) fn prepare_pair_manifold_with_masses(
        manifold: &mut BodyPairManifold,
        inv_mass_a: f32,
        inv_i_a: Mat3,
        inv_mass_b: f32,
        inv_i_b: Mat3,
    ) {
        let normal = manifold.normal;
        let tangent1 = manifold.tangent1;
        let tangent2 = manifold.tangent2;
        let point_count = manifold.point_count;

        for pt in &mut manifold.points[0..point_count] {
            let rn_a = pt.r_a.cross(normal);
            let rn_b = pt.r_b.cross(normal);
            let kn = inv_mass_a + inv_mass_b + rn_a.dot(inv_i_a * rn_a) + rn_b.dot(inv_i_b * rn_b);
            pt.effective_mass_n = if kn > 1e-8 { 1.0 / kn } else { 0.0 };

            let rt1_a = pt.r_a.cross(tangent1);
            let rt1_b = pt.r_b.cross(tangent1);
            let kt1 =
                inv_mass_a + inv_mass_b + rt1_a.dot(inv_i_a * rt1_a) + rt1_b.dot(inv_i_b * rt1_b);
            pt.effective_mass_t1 = if kt1 > 1e-8 { 1.0 / kt1 } else { 0.0 };

            let rt2_a = pt.r_a.cross(tangent2);
            let rt2_b = pt.r_b.cross(tangent2);
            let kt2 =
                inv_mass_a + inv_mass_b + rt2_a.dot(inv_i_a * rt2_a) + rt2_b.dot(inv_i_b * rt2_b);
            pt.effective_mass_t2 = if kt2 > 1e-8 { 1.0 / kt2 } else { 0.0 };
        }
    }

    pub(crate) fn apply_pair_warm_start_active(
        manifold: &BodyPairManifold,
        body_a: &mut Self,
        body_b: &mut Self,
    ) {
        let inv_mass_a = if body_a.sleeping || body_a.is_baking {
            0.0
        } else {
            body_a.inv_mass
        };
        let inv_i_a = if body_a.sleeping || body_a.is_baking {
            Mat3::ZERO
        } else {
            body_a.world_inv_inertia()
        };
        let inv_mass_b = if body_b.sleeping || body_b.is_baking {
            0.0
        } else {
            body_b.inv_mass
        };
        let inv_i_b = if body_b.sleeping || body_b.is_baking {
            Mat3::ZERO
        } else {
            body_b.world_inv_inertia()
        };
        let normal = manifold.normal;
        let tangent1 = manifold.tangent1;
        let tangent2 = manifold.tangent2;

        for pt in &manifold.points[0..manifold.point_count] {
            let p = normal * pt.accum_n + tangent1 * pt.accum_t1 + tangent2 * pt.accum_t2;
            if p.length_squared() > 1e-12 {
                body_a.linear_velocity -= p * inv_mass_a;
                body_a.angular_velocity -= inv_i_a * pt.r_a.cross(p);
                body_b.linear_velocity += p * inv_mass_b;
                body_b.angular_velocity += inv_i_b * pt.r_b.cross(p);
            }
        }
    }

    pub(crate) fn warm_start_terrain_contacts(
        &mut self,
        contacts: &mut [ContactManifold],
        prev_cache: &[CachedTerrainContact],
        inv_i_world: Mat3,
    ) {
        if prev_cache.is_empty() {
            return;
        }
        for c in contacts.iter_mut() {
            let vn = (self.linear_velocity + self.angular_velocity.cross(c.r_world)).dot(c.normal);
            if vn > 0.50 {
                continue;
            }
            let world_pos = self.position + c.r_world;
            let mut best: Option<&CachedTerrainContact> = None;
            let mut best_dist_sq = WARM_START_MATCH_DIST_SQ;
            for prev in prev_cache {
                if prev.normal.dot(c.normal) > 0.90 {
                    let d2 = prev.world_pos.distance_squared(world_pos);
                    if d2 < best_dist_sq {
                        best_dist_sq = d2;
                        best = Some(prev);
                    }
                }
            }
            let Some(prev) = best else {
                continue;
            };

            c.accum_n = prev.accum_n * WARM_START_FACTOR;
            // WHY: Re-project 3D world-space tangential impulse onto current tangent basis for rotational frame invariance.
            let pt_vec =
                (prev.tangent1 * prev.accum_t1 + prev.tangent2 * prev.accum_t2) * WARM_START_FACTOR;
            let max_f = self.friction * c.accum_n;
            c.accum_t1 = pt_vec.dot(c.tangent1).clamp(-max_f, max_f);
            c.accum_t2 = pt_vec.dot(c.tangent2).clamp(-max_f, max_f);

            let p = c.normal * c.accum_n + c.tangent1 * c.accum_t1 + c.tangent2 * c.accum_t2;
            if p.length_squared() > 1e-12 {
                self.linear_velocity += p * self.inv_mass;
                self.angular_velocity += inv_i_world * c.r_world.cross(p);
            }
        }
    }

    pub(crate) fn solve_pair_manifold_velocity_iteration(
        manifold: &mut BodyPairManifold,
        body_a: &mut Self,
        body_b: &mut Self,
        baumgarte_dt: Option<f32>,
        iter: usize,
    ) {
        let baumgarte_beta = 0.20;
        let inv_mass_a = if body_a.sleeping || body_a.is_baking {
            0.0
        } else {
            body_a.inv_mass
        };
        let inv_i_a = if body_a.sleeping || body_a.is_baking {
            Mat3::ZERO
        } else {
            body_a.world_inv_inertia()
        };
        let inv_mass_b = if body_b.sleeping || body_b.is_baking {
            0.0
        } else {
            body_b.inv_mass
        };
        let inv_i_b = if body_b.sleeping || body_b.is_baking {
            Mat3::ZERO
        } else {
            body_b.world_inv_inertia()
        };

        let normal = manifold.normal;
        let tangent1 = manifold.tangent1;
        let tangent2 = manifold.tangent2;
        let friction = manifold.friction;
        let restitution = manifold.restitution;
        let point_count = manifold.point_count;

        for pt in &mut manifold.points[0..point_count] {
            let va = body_a.linear_velocity + body_a.angular_velocity.cross(pt.r_a);
            let vb = body_b.linear_velocity + body_b.angular_velocity.cross(pt.r_b);
            let v_rel = vb - va;
            let vn = v_rel.dot(normal);

            // WHY: Split-impulse stepping passes baumgarte_dt = None so overlap never injects kinetic energy or inflates friction cones.
            let bias = if let Some(sub_dt) = baumgarte_dt {
                let effective_pen = (pt.penetration - PENETRATION_SLOP).max(0.0);
                ((baumgarte_beta / sub_dt) * effective_pen).min(MAX_PENETRATION_BIAS_VEL)
            } else {
                0.0
            };

            let rest = if iter == 0 && vn < -1.5 {
                restitution
            } else {
                0.0
            };

            let delta_vn = -(vn - bias) + rest * (-vn);
            let delta_jn = delta_vn * pt.effective_mass_n;

            // INVARIANT: Clamping normal impulse >= 0 prevents attractive adhesive forces.
            let new_accum_n = (pt.accum_n + delta_jn).max(0.0);
            let actual_jn = new_accum_n - pt.accum_n;
            pt.accum_n = new_accum_n;

            let p_n = normal * actual_jn;
            body_a.linear_velocity -= p_n * inv_mass_a;
            body_a.angular_velocity -= inv_i_a * pt.r_a.cross(p_n);
            body_b.linear_velocity += p_n * inv_mass_b;
            body_b.angular_velocity += inv_i_b * pt.r_b.cross(p_n);

            let va = body_a.linear_velocity + body_a.angular_velocity.cross(pt.r_a);
            let vb = body_b.linear_velocity + body_b.angular_velocity.cross(pt.r_b);
            let v_rel = vb - va;
            let vt1 = v_rel.dot(tangent1);
            let delta_jt1 = -vt1 * pt.effective_mass_t1;
            let max_friction = friction * pt.accum_n;
            // INVARIANT: Tangential friction impulses are clamped within the Coulomb friction cone [-mu*Fn, mu*Fn].
            let new_accum_t1 = (pt.accum_t1 + delta_jt1).clamp(-max_friction, max_friction);
            let actual_jt1 = new_accum_t1 - pt.accum_t1;
            pt.accum_t1 = new_accum_t1;

            let p_t1 = tangent1 * actual_jt1;
            body_a.linear_velocity -= p_t1 * inv_mass_a;
            body_a.angular_velocity -= inv_i_a * pt.r_a.cross(p_t1);
            body_b.linear_velocity += p_t1 * inv_mass_b;
            body_b.angular_velocity += inv_i_b * pt.r_b.cross(p_t1);

            let va = body_a.linear_velocity + body_a.angular_velocity.cross(pt.r_a);
            let vb = body_b.linear_velocity + body_b.angular_velocity.cross(pt.r_b);
            let v_rel = vb - va;
            let vt2 = v_rel.dot(tangent2);
            let delta_jt2 = -vt2 * pt.effective_mass_t2;
            let new_accum_t2 = (pt.accum_t2 + delta_jt2).clamp(-max_friction, max_friction);
            let actual_jt2 = new_accum_t2 - pt.accum_t2;
            pt.accum_t2 = new_accum_t2;

            let p_t2 = tangent2 * actual_jt2;
            body_a.linear_velocity -= p_t2 * inv_mass_a;
            body_a.angular_velocity -= inv_i_a * pt.r_a.cross(p_t2);
            body_b.linear_velocity += p_t2 * inv_mass_b;
            body_b.angular_velocity += inv_i_b * pt.r_b.cross(p_t2);
        }
    }

    pub(crate) fn solve_terrain_contacts_velocity_iteration(
        &mut self,
        contacts: &mut [ContactManifold],
        inv_i_world: Mat3,
        baumgarte_dt: Option<f32>,
        restitution_coeff: f32,
        iter: usize,
        bounce_targets: Option<&[f32]>,
    ) {
        let baumgarte_beta = 0.15;
        for (idx, c) in contacts.iter_mut().enumerate() {
            let v_point = self.linear_velocity + self.angular_velocity.cross(c.r_world);
            let v_n = v_point.dot(c.normal);

            let bias = if let Some(sub_dt) = baumgarte_dt {
                let effective_pen = (c.penetration - PENETRATION_SLOP).max(0.0);
                ((baumgarte_beta / sub_dt) * effective_pen).min(MAX_PENETRATION_BIAS_VEL)
            } else {
                0.0
            };
            let target_vn = match bounce_targets {
                Some(bt) => bt[idx],
                None if iter == 0 && v_n < -2.0 => -restitution_coeff * v_n,
                None => 0.0,
            };

            let delta_vn = (target_vn - v_n) + bias;
            let delta_jn = delta_vn * c.effective_mass_n;

            // INVARIANT: Clamped normal impulse >= 0 ensures contact reaction force only repels.
            let new_accum_n = (c.accum_n + delta_jn).max(0.0);
            let actual_jn = new_accum_n - c.accum_n;
            c.accum_n = new_accum_n;

            let impulse_n = c.normal * actual_jn;
            self.linear_velocity += impulse_n * self.inv_mass;
            self.angular_velocity += inv_i_world * c.r_world.cross(impulse_n);

            let v_post = self.linear_velocity + self.angular_velocity.cross(c.r_world);
            let vt1 = v_post.dot(c.tangent1);
            let delta_jt1 = -vt1 * c.effective_mass_t1;
            let max_friction = self.friction * c.accum_n;
            // INVARIANT: Tangential friction impulses are clamped within the Coulomb friction cone.
            let new_accum_t1 = (c.accum_t1 + delta_jt1).clamp(-max_friction, max_friction);
            let actual_jt1 = new_accum_t1 - c.accum_t1;
            c.accum_t1 = new_accum_t1;

            let impulse_t1 = c.tangent1 * actual_jt1;
            self.linear_velocity += impulse_t1 * self.inv_mass;
            self.angular_velocity += inv_i_world * c.r_world.cross(impulse_t1);

            let v_post2 = self.linear_velocity + self.angular_velocity.cross(c.r_world);
            let vt2 = v_post2.dot(c.tangent2);
            let delta_jt2 = -vt2 * c.effective_mass_t2;
            let new_accum_t2 = (c.accum_t2 + delta_jt2).clamp(-max_friction, max_friction);
            let actual_jt2 = new_accum_t2 - c.accum_t2;
            c.accum_t2 = new_accum_t2;

            let impulse_t2 = c.tangent2 * actual_jt2;
            self.linear_velocity += impulse_t2 * self.inv_mass;
            self.angular_velocity += inv_i_world * c.r_world.cross(impulse_t2);
        }
    }

    pub(crate) fn apply_terrain_ground_settling(
        &mut self,
        contacts: &[ContactManifold],
        is_wet: bool,
        stably_supported: bool,
        sub_dt: f32,
    ) {
        if contacts.is_empty() {
            return;
        }

        if is_wet {
            let has_floor_contact = contacts.iter().any(|c| c.normal.y > 0.3);
            // WHY: Squeeze-film hydrodynamic damping prevents residual depenetration from launching wet bodies off pool floors.
            if has_floor_contact && self.linear_velocity.y > 0.0 {
                self.linear_velocity.y = (self.linear_velocity.y * 0.35).min(0.10);
            }
            self.linear_velocity.x *= (1.0 - 8.0 * sub_dt).max(0.0);
            self.linear_velocity.z *= (1.0 - 8.0 * sub_dt).max(0.0);
            if stably_supported {
                self.angular_velocity *= (1.0 - 9.0 * sub_dt).max(0.0);
            }
        } else if self.linear_velocity.y > 0.50 {
            return;
        }

        // INVARIANT: Never damp or zero angular velocity when CoM overhangs outside the support polygon.
        if !stably_supported {
            return;
        }

        let lin_speed = self.linear_velocity.length();
        let ang_speed = self.angular_velocity.length();
        // WHY: Apply Coulomb sliding and rolling resistance only when the CoM rests inside its support polygon.
        self.linear_velocity.x *= (1.0 - 5.0 * sub_dt).max(0.0);
        self.linear_velocity.z *= (1.0 - 5.0 * sub_dt).max(0.0);
        self.angular_velocity *= (1.0 - 6.0 * sub_dt).max(0.0);

        if lin_speed < 0.28 && ang_speed < 0.45 {
            self.linear_velocity.x = 0.0;
            self.linear_velocity.z = 0.0;
            self.angular_velocity = Vec3::ZERO;
            if self.linear_velocity.y.abs() < 0.16 {
                self.linear_velocity.y = 0.0;
            }
        }
    }

    pub(crate) fn solve_sequential_impulses(
        &mut self,
        contacts: &mut [ContactManifold],
        inv_i_world: Mat3,
        sub_dt: f32,
        max_impact_speed: f32,
        is_wet: bool,
        stably_supported: bool,
    ) {
        let solver_iterations = 6;
        let restitution_coeff = if max_impact_speed > 2.0 && !is_wet {
            self.restitution
        } else {
            0.0
        };
        let bounce_targets = if restitution_coeff > 0.0 {
            compute_terrain_bounce_targets(self, contacts, restitution_coeff)
        } else {
            Vec::new()
        };
        let bt = if bounce_targets.is_empty() {
            None
        } else {
            Some(bounce_targets.as_slice())
        };

        for iter in 0..solver_iterations {
            self.solve_terrain_contacts_velocity_iteration(
                contacts,
                inv_i_world,
                None,
                restitution_coeff,
                iter,
                bt,
            );
        }

        self.apply_terrain_ground_settling(contacts, is_wet, stably_supported, sub_dt);
    }
}

pub(crate) fn apply_bottom_up_velocity_shock_pass(
    bodies: &mut [RigidVoxelBody],
    terrain_data: &mut [TerrainContactData],
    manifolds: &mut [BodyPairManifold],
    support_levels: &[u16],
    substep_stable: &[bool],
) {
    for (i, body) in bodies.iter_mut().enumerate() {
        if !body.sleeping && !terrain_data[i].contacts.is_empty() {
            let inv_i_world = terrain_data[i].inv_i_world;
            body.solve_terrain_contacts_velocity_iteration(
                &mut terrain_data[i].contacts,
                inv_i_world,
                None,
                0.0,
                1,
                None,
            );
        }
    }

    // WHY: Guendelman-Bridson-Fedkiw bottom-up shock pass propagates unyielding ground support through light bodies under heavy loads.
    for m in manifolds.iter_mut() {
        let a = m.body_a_idx;
        let b = m.body_b_idx;
        let la = support_levels[a];
        let lb = support_levels[b];

        let a_is_foundation = (!bodies[b].sleeping && !bodies[b].is_baking)
            && ((la < lb && m.normal.y > 0.20)
                || bodies[a].is_baking
                || (terrain_backs_direction(&terrain_data[a].contacts, m.normal)
                    && !terrain_backs_direction(&terrain_data[b].contacts, -m.normal)));
        let b_is_foundation = !a_is_foundation
            && (!bodies[a].sleeping && !bodies[a].is_baking)
            && ((lb < la && m.normal.y < -0.20)
                || bodies[b].is_baking
                || (terrain_backs_direction(&terrain_data[b].contacts, -m.normal)
                    && !terrain_backs_direction(&terrain_data[a].contacts, m.normal)));

        if !a_is_foundation && !b_is_foundation {
            continue;
        }

        let (left, right) = bodies.split_at_mut(b);
        let body_a = &mut left[a];
        let body_b = &mut right[0];

        let (inv_mass_a, inv_i_a, inv_mass_b, inv_i_b) = if a_is_foundation {
            // INVARIANT: Keep rotational inertia live when `!substep_stable[a]` so cantilevered seesaw rods still flip freely.
            let shock_inv_i_a = if substep_stable[a] || body_a.sleeping || body_a.is_baking {
                Mat3::ZERO
            } else {
                body_a.world_inv_inertia()
            };
            (
                0.0,
                shock_inv_i_a,
                body_b.inv_mass,
                body_b.world_inv_inertia(),
            )
        } else {
            let shock_inv_i_b = if substep_stable[b] || body_b.sleeping || body_b.is_baking {
                Mat3::ZERO
            } else {
                body_b.world_inv_inertia()
            };
            (
                body_a.inv_mass,
                body_a.world_inv_inertia(),
                0.0,
                shock_inv_i_b,
            )
        };

        let normal = m.normal;
        for pt in &mut m.points[0..m.point_count] {
            let va = body_a.linear_velocity + body_a.angular_velocity.cross(pt.r_a);
            let vb = body_b.linear_velocity + body_b.angular_velocity.cross(pt.r_b);
            let vn = (vb - va).dot(normal);
            if vn >= 0.0 {
                continue;
            }
            let rn_a = pt.r_a.cross(normal);
            let rn_b = pt.r_b.cross(normal);
            let kn = inv_mass_a + inv_mass_b + rn_a.dot(inv_i_a * rn_a) + rn_b.dot(inv_i_b * rn_b);
            if kn <= 1e-8 {
                continue;
            }
            let actual_jn = (-vn / kn).max(0.0);
            pt.accum_n += actual_jn;
            let p_n = normal * actual_jn;
            body_a.linear_velocity -= p_n * inv_mass_a;
            body_a.angular_velocity -= inv_i_a * pt.r_a.cross(p_n);
            body_b.linear_velocity += p_n * inv_mass_b;
            body_b.angular_velocity += inv_i_b * pt.r_b.cross(p_n);
        }
    }
}

pub(crate) fn compute_support_levels(
    len: usize,
    bodies: &[RigidVoxelBody],
    terrain_data: &[TerrainContactData],
    manifolds: &[BodyPairManifold],
) -> Vec<u16> {
    let mut levels = vec![u16::MAX; len];
    let mut queue = Vec::with_capacity(len);

    for i in 0..len {
        let on_terrain = bodies[i].sleeping
            || terrain_data[i]
                .contacts
                .iter()
                .any(|c| c.normal.y > 0.20 || c.penetration > 0.0);
        if on_terrain {
            levels[i] = 0;
            queue.push(i);
        }
    }

    let mut head = 0usize;
    while head < queue.len() {
        let u = queue[head];
        head += 1;
        let next_lvl = levels[u].saturating_add(1);

        for m in manifolds {
            if m.point_count == 0 {
                continue;
            }
            if m.body_a_idx == u && m.normal.y > 0.20 && next_lvl < levels[m.body_b_idx] {
                levels[m.body_b_idx] = next_lvl;
                queue.push(m.body_b_idx);
            } else if m.body_b_idx == u && m.normal.y < -0.20 && next_lvl < levels[m.body_a_idx] {
                levels[m.body_a_idx] = next_lvl;
                queue.push(m.body_a_idx);
            }
        }
    }

    levels
}

pub(crate) fn terrain_backs_direction(contacts: &[ContactManifold], push_dir: Vec3) -> bool {
    contacts.iter().any(|c| c.normal.dot(push_dir) > 0.25)
}

pub(crate) fn compute_terrain_bounce_targets(
    body: &RigidVoxelBody,
    contacts: &[ContactManifold],
    restitution_coeff: f32,
) -> Vec<f32> {
    // WHY: Pre-compute per-contact elastic rebound targets before PGS iteration 0 so multi-corner impacts bounce symmetrically.
    contacts
        .iter()
        .map(|c| {
            let vn = (body.linear_velocity + body.angular_velocity.cross(c.r_world)).dot(c.normal);
            if vn < -2.0 {
                -restitution_coeff * vn
            } else {
                0.0
            }
        })
        .collect()
}
