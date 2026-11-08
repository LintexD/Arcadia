//! Split-impulse non-linear position stabilization and penetration resolution passes.

use super::sleep::TerrainContactData;
use super::velocity::terrain_backs_direction;
use super::PENETRATION_SLOP;
use crate::rigid_voxel::types::{BodyPairManifold, ContactManifold, RigidVoxelBody};
use glam::{Mat3, Quat, Vec3};

const POSITION_SOLVER_ERP: f32 = 0.35;
const SHOCK_POSITION_ERP: f32 = 0.55;
const MAX_POSITION_STEP_CORR: f32 = 0.025;
const SANDWICHED_MASS_RATIO_CAP: f32 = 12.0;

impl RigidVoxelBody {
    pub(crate) fn solve_single_body_terrain_positions(
        &mut self,
        contacts: &[ContactManifold],
        inv_i_world: Mat3,
    ) {
        if contacts.is_empty() {
            return;
        }
        let mut delta_pos = Vec3::ZERO;
        let mut delta_ang = Vec3::ZERO;
        let mut accum_pos = vec![0.0f32; contacts.len()];

        for _ in 0..4 {
            for (idx, c) in contacts.iter().enumerate() {
                let moved = (delta_pos + delta_ang.cross(c.r_world)).dot(c.normal);
                let pen_curr = c.penetration - moved;
                let err = pen_curr - PENETRATION_SLOP;
                let step = (POSITION_SOLVER_ERP * err)
                    .clamp(-MAX_POSITION_STEP_CORR, MAX_POSITION_STEP_CORR);
                let new_accum = (accum_pos[idx] + step * c.effective_mass_n).max(0.0);
                let actual_jp = new_accum - accum_pos[idx];
                accum_pos[idx] = new_accum;

                if actual_jp.abs() > 1e-9 {
                    let p = c.normal * actual_jp;
                    delta_pos += p * self.inv_mass;
                    delta_ang += inv_i_world * c.r_world.cross(p);
                }
            }
        }

        self.apply_position_delta(delta_pos, delta_ang);
    }

    pub(crate) fn apply_position_delta(&mut self, delta_pos: Vec3, delta_ang: Vec3) {
        self.position += delta_pos;
        let ang_len = delta_ang.length();
        if ang_len > 1e-6 {
            let clamped = ang_len.min(0.25);
            let dq = Quat::from_axis_angle(delta_ang / ang_len, clamped);
            self.orientation = (dq * self.orientation).normalize();
        }
    }

    pub(crate) fn update_post_integration_terrain_penetrations(
        &self,
        contacts: &mut [ContactManifold],
        sub_dt: f32,
    ) {
        for c in contacts.iter_mut() {
            let vn = (self.linear_velocity + self.angular_velocity.cross(c.r_world)).dot(c.normal);
            c.penetration = (c.penetration - vn * sub_dt).max(0.0);
        }
    }
}

pub(crate) fn solve_split_impulse_positions(
    bodies: &mut [RigidVoxelBody],
    terrain_data: &[TerrainContactData],
    manifolds: &[BodyPairManifold],
    support_levels: &[u16],
    substep_stable: &[bool],
    contact_degree: &[u16],
) {
    let len = bodies.len();
    let mut pos_corr = vec![Vec3::ZERO; len];
    let mut ang_corr = vec![Vec3::ZERO; len];
    let mut terrain_accum: Vec<Vec<f32>> = terrain_data
        .iter()
        .map(|td| vec![0.0f32; td.contacts.len()])
        .collect();
    let mut pair_accum = vec![[0.0f32; 4]; manifolds.len()];

    let inv_i_worlds: Vec<Mat3> = bodies
        .iter()
        .map(|b| {
            if b.sleeping || b.is_baking {
                Mat3::ZERO
            } else {
                b.world_inv_inertia()
            }
        })
        .collect();

    for _ in 0..6 {
        solve_terrain_position_pass(
            bodies,
            terrain_data,
            &inv_i_worlds,
            &mut pos_corr,
            &mut ang_corr,
            &mut terrain_accum,
            POSITION_SOLVER_ERP,
        );

        for (m_idx, m) in manifolds.iter().enumerate() {
            let a = m.body_a_idx;
            let b = m.body_b_idx;
            let (mut inv_ma, mut inv_ia) = if bodies[a].sleeping || bodies[a].is_baking {
                (0.0, Mat3::ZERO)
            } else {
                (bodies[a].inv_mass, inv_i_worlds[a])
            };
            let (mut inv_mb, mut inv_ib) = if bodies[b].sleeping || bodies[b].is_baking {
                (0.0, Mat3::ZERO)
            } else {
                (bodies[b].inv_mass, inv_i_worlds[b])
            };

            // WHY: Condition geometric position compliance on sandwiched multi-contact bodies so heavy boulders never crush light blocks in bowls.
            if inv_ma > 0.0 && inv_mb > 0.0 {
                if inv_ma > inv_mb * SANDWICHED_MASS_RATIO_CAP && contact_degree[a] >= 2 {
                    let conditioned = inv_ma / SANDWICHED_MASS_RATIO_CAP;
                    let scale = conditioned / inv_mb;
                    inv_mb = conditioned;
                    inv_ib *= scale;
                } else if inv_mb > inv_ma * SANDWICHED_MASS_RATIO_CAP && contact_degree[b] >= 2 {
                    let conditioned = inv_mb / SANDWICHED_MASS_RATIO_CAP;
                    let scale = conditioned / inv_ma;
                    inv_ma = conditioned;
                    inv_ia *= scale;
                }
            }

            solve_single_manifold_position(
                m,
                (inv_ma, inv_ia),
                (inv_mb, inv_ib),
                &mut pos_corr,
                &mut ang_corr,
                &mut pair_accum[m_idx],
                POSITION_SOLVER_ERP,
            );
        }
    }

    for _ in 0..2 {
        solve_terrain_position_pass(
            bodies,
            terrain_data,
            &inv_i_worlds,
            &mut pos_corr,
            &mut ang_corr,
            &mut terrain_accum,
            SHOCK_POSITION_ERP,
        );

        for (m_idx, m) in manifolds.iter().enumerate() {
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

            let ((inv_ma, inv_ia), (inv_mb, inv_ib)) = if a_is_foundation {
                let shock_ia = if substep_stable[a] || bodies[a].sleeping || bodies[a].is_baking {
                    Mat3::ZERO
                } else {
                    inv_i_worlds[a]
                };
                ((0.0, shock_ia), (bodies[b].inv_mass, inv_i_worlds[b]))
            } else {
                let shock_ib = if substep_stable[b] || bodies[b].sleeping || bodies[b].is_baking {
                    Mat3::ZERO
                } else {
                    inv_i_worlds[b]
                };
                ((bodies[a].inv_mass, inv_i_worlds[a]), (0.0, shock_ib))
            };

            solve_single_manifold_position(
                m,
                (inv_ma, inv_ia),
                (inv_mb, inv_ib),
                &mut pos_corr,
                &mut ang_corr,
                &mut pair_accum[m_idx],
                SHOCK_POSITION_ERP,
            );
        }
    }

    for (i, body) in bodies.iter_mut().enumerate() {
        if !body.sleeping && !body.is_baking {
            body.apply_position_delta(pos_corr[i], ang_corr[i]);
        }
    }
}

fn solve_terrain_position_pass(
    bodies: &[RigidVoxelBody],
    terrain_data: &[TerrainContactData],
    inv_i_worlds: &[Mat3],
    pos_corr: &mut [Vec3],
    ang_corr: &mut [Vec3],
    terrain_accum: &mut [Vec<f32>],
    erp: f32,
) {
    for (i, body) in bodies.iter().enumerate() {
        if body.sleeping || terrain_data[i].contacts.is_empty() {
            continue;
        }
        let inv_m = body.inv_mass;
        let inv_i = inv_i_worlds[i];
        for (c_idx, c) in terrain_data[i].contacts.iter().enumerate() {
            let moved = (pos_corr[i] + ang_corr[i].cross(c.r_world)).dot(c.normal);
            let pen_curr = c.penetration - moved;
            let err = pen_curr - PENETRATION_SLOP;
            let step = (erp * err).clamp(-MAX_POSITION_STEP_CORR, MAX_POSITION_STEP_CORR);
            let new_accum = (terrain_accum[i][c_idx] + step * c.effective_mass_n).max(0.0);
            let actual_jp = new_accum - terrain_accum[i][c_idx];
            terrain_accum[i][c_idx] = new_accum;

            if actual_jp.abs() > 1e-9 {
                let p = c.normal * actual_jp;
                pos_corr[i] += p * inv_m;
                ang_corr[i] += inv_i * c.r_world.cross(p);
            }
        }
    }
}

fn solve_single_manifold_position(
    m: &BodyPairManifold,
    (inv_ma, inv_ia): (f32, Mat3),
    (inv_mb, inv_ib): (f32, Mat3),
    pos_corr: &mut [Vec3],
    ang_corr: &mut [Vec3],
    pt_accum: &mut [f32; 4],
    erp: f32,
) {
    let a = m.body_a_idx;
    let b = m.body_b_idx;
    let normal = m.normal;

    for (k, pt) in m.points[0..m.point_count].iter().enumerate() {
        let moved_a = pos_corr[a] + ang_corr[a].cross(pt.r_a);
        let moved_b = pos_corr[b] + ang_corr[b].cross(pt.r_b);
        let separated = (moved_b - moved_a).dot(normal);
        let pen_curr = pt.penetration - separated;
        let err = pen_curr - PENETRATION_SLOP;

        let rn_a = pt.r_a.cross(normal);
        let rn_b = pt.r_b.cross(normal);
        let kn = inv_ma + inv_mb + rn_a.dot(inv_ia * rn_a) + rn_b.dot(inv_ib * rn_b);
        if kn <= 1e-8 {
            continue;
        }
        let eff_mass = 1.0 / kn;
        let step = (erp * err).clamp(-MAX_POSITION_STEP_CORR, MAX_POSITION_STEP_CORR);
        let new_accum = (pt_accum[k] + step * eff_mass).max(0.0);
        let actual_jp = new_accum - pt_accum[k];
        pt_accum[k] = new_accum;

        if actual_jp.abs() > 1e-9 {
            let p = normal * actual_jp;
            pos_corr[a] -= p * inv_ma;
            ang_corr[a] -= inv_ia * pt.r_a.cross(p);
            pos_corr[b] += p * inv_mb;
            ang_corr[b] += inv_ib * pt.r_b.cross(p);
        }
    }
}

pub(crate) fn update_post_integration_pair_penetrations(
    bodies: &[RigidVoxelBody],
    manifolds: &mut [BodyPairManifold],
    sub_dt: f32,
) {
    for m in manifolds.iter_mut() {
        let body_a = &bodies[m.body_a_idx];
        let body_b = &bodies[m.body_b_idx];
        let normal = m.normal;
        for pt in &mut m.points[0..m.point_count] {
            let va = if body_a.sleeping {
                Vec3::ZERO
            } else {
                body_a.linear_velocity + body_a.angular_velocity.cross(pt.r_a)
            };
            let vb = if body_b.sleeping {
                Vec3::ZERO
            } else {
                body_b.linear_velocity + body_b.angular_velocity.cross(pt.r_b)
            };
            let vn = (vb - va).dot(normal);
            pt.penetration = (pt.penetration - vn * sub_dt).max(0.0);
        }
    }
}
