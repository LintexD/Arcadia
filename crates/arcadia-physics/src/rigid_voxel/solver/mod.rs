//! Rigid body constraint solver orchestration, time stepping, and integration.
//! Owns CPU substepping loops, broadphase pair gathering, and velocity/position solver passes.

mod position;
mod sleep;
mod velocity;

use self::position::{solve_split_impulse_positions, update_post_integration_pair_penetrations};
use self::sleep::{
    evaluate_body_support_stability, is_terrain_support_stable, should_wake_sleeping_pair,
    update_island_sleep_states, wake_unsupported_sleeping_bodies,
    warm_start_pair_manifold_from_cache, CachedTerrainContact, SupportSleepKind,
    TerrainContactData,
};
use self::velocity::{
    apply_bottom_up_velocity_shock_pass, compute_support_levels, compute_terrain_bounce_targets,
};
use super::types::{BodyPairManifold, RigidStepOutcome, RigidVoxelBody};
use arcadia_core::VOXEL_SIZE;
use arcadia_world::{DebrisParticle, WorldMap};
use glam::{Mat3, Quat, Vec3};

/// Penetration slop threshold (2 mm) beneath which no positional correction is generated.
pub const PENETRATION_SLOP: f32 = 0.002;

/// Maximum bias velocity (0.8 m/s) applied by standalone Baumgarte velocity iterations.
pub const MAX_PENETRATION_BIAS_VEL: f32 = 0.8;

impl RigidVoxelBody {
    /// Awakens this rigid body from sleep mode without wiping its accumulated rest time.
    #[inline]
    pub fn wake_up(&mut self) {
        if !self.is_baking {
            self.sleeping = false;
            self.sleep_timer = 0.0;
        }
    }

    /// Resets the accumulated resting time to zero (e.g. on direct player carve/place or high-energy blast).
    #[inline]
    pub fn reset_rest_timer(&mut self) {
        if !self.is_baking {
            self.rest_time = 0.0;
        }
    }

    /// Applies buoyancy acceleration, progressive hydrodynamic drag, and rotational viscous damping in fluid.
    pub fn apply_fluid_dynamics(&mut self, submersion_ratio: f32, sub_dt: f32) {
        if submersion_ratio <= 0.0 {
            return;
        }

        let density_ratio = 1000.0 / 2400.0;
        let a_buoyancy = Vec3::Y * (18.0 * density_ratio * submersion_ratio);
        self.linear_velocity += a_buoyancy * sub_dt;

        let speed = self.linear_velocity.length();
        if speed > 1e-4 {
            let drag_mag = (1.5 * speed + 0.8 * speed * speed) * submersion_ratio;
            let drag_force = self.linear_velocity * (drag_mag / speed);
            let new_vel = self.linear_velocity - drag_force * sub_dt;
            // INVARIANT: Fluid drag opposes motion and must never reverse velocity direction.
            self.linear_velocity = if new_vel.dot(self.linear_velocity) > 0.0 {
                new_vel.clamp_length_max(speed)
            } else {
                Vec3::ZERO
            };
        }

        self.angular_velocity *= (1.0 - 5.0 * sub_dt * submersion_ratio).max(0.0);
    }

    /// Advances the single-body 6-DOF simulation by `dt` seconds against static `WorldMap` terrain.
    pub fn step(&mut self, dt: f32, world: &WorldMap) -> RigidStepOutcome {
        let mut outcome = RigidStepOutcome::default();
        let dt = dt.clamp(0.001, 0.05);
        self.age += dt;

        // WHY: Snapshot previous transform once per rendered frame for exact temporal motion vectors.
        self.prev_position = self.position;
        self.prev_orientation = self.orientation;

        if self.sleeping {
            self.rest_time += dt;
            if !self.check_ground_support(world) {
                // WHY: Ground beneath was carved away; awaken body without resetting rest timer.
                self.sleeping = false;
                self.sleep_timer = 0.0;
            } else {
                return outcome;
            }
        }

        // WHY: Adaptive substeps keep sub_dt <= 4.2 ms across variable frame rates for consistent solver stiffness.
        let num_substeps = ((dt / 0.004).ceil() as u32).clamp(4, 12);
        let sub_dt = dt / (num_substeps as f32);
        let gravity = Vec3::new(0.0, -18.0, 0.0);

        let mut any_stably_supported = false;
        let mut any_contact = false;
        let mut prev_terrain_cache: Vec<CachedTerrainContact> = Vec::new();

        for _ in 0..num_substeps {
            self.linear_velocity += gravity * sub_dt;
            self.linear_velocity *= (1.0 - 0.05 * sub_dt).max(0.0);
            self.angular_velocity *= (1.0 - 0.08 * sub_dt).max(0.0);

            let submersion_ratio = self.compute_submersion_ratio(world);
            self.submersion_ratio = submersion_ratio;
            self.apply_fluid_dynamics(submersion_ratio, sub_dt);

            let (aabb_min, aabb_max) = self.world_aabb_meters();
            if !world.intersects_aabb_meters(aabb_min, aabb_max) && aabb_min.y > VOXEL_SIZE * 5.0 {
                prev_terrain_cache.clear();
                self.integrate_transform(sub_dt);
                continue;
            }

            let r_mat = Mat3::from_quat(self.orientation);
            let inv_i_world = self.world_inv_inertia();
            let (mut contacts, max_impact_speed, has_wet_contact) =
                self.generate_world_contacts_detailed(world, inv_i_world, r_mat);
            let is_wet = has_wet_contact || self.submersion_ratio > 0.05;
            let stably_supported = is_terrain_support_stable(&contacts);

            any_contact |= !contacts.is_empty();
            any_stably_supported |= stably_supported;

            self.warm_start_terrain_contacts(&mut contacts, &prev_terrain_cache, inv_i_world);

            self.solve_sequential_impulses(
                &mut contacts,
                inv_i_world,
                sub_dt,
                max_impact_speed,
                is_wet,
                stably_supported,
            );

            self.integrate_transform(sub_dt);
            self.update_post_integration_terrain_penetrations(&mut contacts, sub_dt);
            self.solve_single_body_terrain_positions(&contacts, inv_i_world);

            prev_terrain_cache.clear();
            for c in &contacts {
                prev_terrain_cache.push(CachedTerrainContact {
                    world_pos: self.position + c.r_world,
                    normal: c.normal,
                    tangent1: c.tangent1,
                    tangent2: c.tangent2,
                    accum_n: c.accum_n,
                    accum_t1: c.accum_t1,
                    accum_t2: c.accum_t2,
                });
            }

            if self.check_high_energy_impact(
                world,
                contacts.is_empty(),
                max_impact_speed,
                r_mat,
                &mut outcome,
            ) {
                return outcome;
            }
        }

        self.update_single_sleep_state(dt, any_stably_supported, any_contact);
        outcome
    }

    /// Advances the unified multi-body simulation with split-impulse position projection, shock propagation, and island sleeping.
    pub fn step_system(bodies: &mut [Self], dt: f32, world: &WorldMap) -> Vec<DebrisParticle> {
        let len = bodies.len();
        if len == 0 {
            return Vec::new();
        }

        let dt = dt.clamp(0.001, 0.05);
        let num_substeps = ((dt / 0.004).ceil() as u32).clamp(4, 12);
        let sub_dt = dt / (num_substeps as f32);
        let gravity = Vec3::new(0.0, -18.0, 0.0);
        let mut debris = Vec::new();

        for body in bodies.iter_mut() {
            body.prev_position = body.position;
            body.prev_orientation = body.orientation;
            body.age += dt;
        }

        wake_unsupported_sleeping_bodies(bodies, world);

        let mut frame_support_kind = vec![SupportSleepKind::None; len];
        let mut frame_contact_edges: Vec<(usize, usize)> = Vec::new();
        let mut prev_terrain_cache: Vec<Vec<CachedTerrainContact>> = vec![Vec::new(); len];
        let mut prev_pair_manifolds: Vec<BodyPairManifold> = Vec::new();

        for _ in 0..num_substeps {
            for (i, body) in bodies.iter_mut().enumerate() {
                if body.sleeping || body.is_baking {
                    prev_terrain_cache[i].clear();
                    continue;
                }
                body.linear_velocity += gravity * sub_dt;
                body.linear_velocity *= (1.0 - 0.05 * sub_dt).max(0.0);
                body.angular_velocity *= (1.0 - 0.08 * sub_dt).max(0.0);

                let submersion_ratio = body.compute_submersion_ratio(world);
                body.submersion_ratio = submersion_ratio;
                body.apply_fluid_dynamics(submersion_ratio, sub_dt);
            }

            let mut terrain_data = Vec::with_capacity(len);
            for body in bodies.iter() {
                if body.sleeping || body.is_baking {
                    terrain_data.push(TerrainContactData::default());
                    continue;
                }

                let (aabb_min, aabb_max) = body.world_aabb_meters();
                if !world.intersects_aabb_meters(aabb_min, aabb_max)
                    && aabb_min.y > VOXEL_SIZE * 5.0
                {
                    terrain_data.push(TerrainContactData::default());
                    continue;
                }

                let r_mat = Mat3::from_quat(body.orientation);
                let inv_i_world = body.world_inv_inertia();
                let (contacts, max_impact_speed, has_wet_contact) =
                    body.generate_world_contacts_detailed(world, inv_i_world, r_mat);
                let is_wet = has_wet_contact || body.submersion_ratio > 0.05;
                let restitution_coeff = if max_impact_speed > 2.0 && !is_wet {
                    body.restitution
                } else {
                    0.0
                };

                terrain_data.push(TerrainContactData {
                    contacts,
                    inv_i_world,
                    r_mat,
                    max_impact_speed,
                    restitution_coeff,
                    is_wet,
                });
            }

            let mut manifolds: Vec<BodyPairManifold> = Vec::new();
            for i in 0..len {
                for j in (i + 1)..len {
                    if (bodies[i].sleeping || bodies[i].is_baking)
                        && (bodies[j].sleeping || bodies[j].is_baking)
                    {
                        continue;
                    }
                    if !Self::check_broadphase_overlap(&bodies[i], &bodies[j]) {
                        continue;
                    }
                    let pair_manifolds =
                        Self::compute_compound_pair_manifolds_indexed(&bodies[i], &bodies[j], i, j);
                    for mut m in pair_manifolds {
                        if m.point_count == 0 {
                            continue;
                        }
                        // WHY: Only wake a sleeping neighbor when struck above gentle impact/overlap thresholds; never reset sleep_timer on awake-awake pairs.
                        if (bodies[i].sleeping || bodies[i].is_baking)
                            != (bodies[j].sleeping || bodies[j].is_baking)
                            && should_wake_sleeping_pair(&m, &bodies[i], &bodies[j])
                        {
                            bodies[i].wake_up();
                            bodies[j].wake_up();
                        }

                        Self::prepare_pair_manifold_active(&mut m, &bodies[i], &bodies[j]);
                        let pair_is_wet = bodies[i].submersion_ratio > 0.05
                            || bodies[j].submersion_ratio > 0.05
                            || terrain_data[i].is_wet
                            || terrain_data[j].is_wet;
                        if pair_is_wet {
                            m.restitution = 0.0;
                        }
                        frame_contact_edges.push((i, j));
                        manifolds.push(m);
                    }
                }
            }

            let support_levels = compute_support_levels(len, bodies, &terrain_data, &manifolds);
            manifolds.sort_by_key(|m| {
                let la = support_levels[m.body_a_idx];
                let lb = support_levels[m.body_b_idx];
                (la.min(lb), la.max(lb))
            });

            let mut substep_stable = vec![false; len];
            let mut is_cradled = vec![false; len];
            let mut contact_degree = vec![0u16; len];
            for i in 0..len {
                if !terrain_data[i].contacts.is_empty() {
                    contact_degree[i] = contact_degree[i].saturating_add(1);
                }
            }
            for m in &manifolds {
                contact_degree[m.body_a_idx] = contact_degree[m.body_a_idx].saturating_add(1);
                contact_degree[m.body_b_idx] = contact_degree[m.body_b_idx].saturating_add(1);
            }

            for i in 0..len {
                if bodies[i].sleeping {
                    substep_stable[i] = true;
                    continue;
                }
                let (stable, has_c, cradled) =
                    evaluate_body_support_stability(i, &terrain_data[i].contacts, &manifolds);
                substep_stable[i] = stable;
                is_cradled[i] = cradled;
                let kind = if stable {
                    SupportSleepKind::StablePolygon
                } else if cradled {
                    SupportSleepKind::MultiCradled
                } else if has_c {
                    SupportSleepKind::SingleContact
                } else {
                    SupportSleepKind::None
                };
                if (kind as u8) > (frame_support_kind[i] as u8) {
                    frame_support_kind[i] = kind;
                }
            }

            for i in 0..len {
                if !bodies[i].sleeping && !terrain_data[i].contacts.is_empty() {
                    let inv_i_world = terrain_data[i].inv_i_world;
                    bodies[i].warm_start_terrain_contacts(
                        &mut terrain_data[i].contacts,
                        &prev_terrain_cache[i],
                        inv_i_world,
                    );
                }
            }

            for manifold in &mut manifolds {
                warm_start_pair_manifold_from_cache(manifold, &prev_pair_manifolds);
                let i = manifold.body_a_idx;
                let j = manifold.body_b_idx;
                let (left, right) = bodies.split_at_mut(j);
                Self::apply_pair_warm_start_active(manifold, &mut left[i], &mut right[0]);
            }

            let velocity_iterations = 10;
            let terrain_bounce: Vec<Vec<f32>> = (0..len)
                .map(|i| {
                    if bodies[i].sleeping
                        || bodies[i].is_baking
                        || terrain_data[i].restitution_coeff <= 0.0
                        || contact_degree[i] > 1
                    {
                        Vec::new()
                    } else {
                        compute_terrain_bounce_targets(
                            &bodies[i],
                            &terrain_data[i].contacts,
                            terrain_data[i].restitution_coeff,
                        )
                    }
                })
                .collect();

            for iter in 0..velocity_iterations {
                for i in 0..len {
                    if !bodies[i].sleeping
                        && !bodies[i].is_baking
                        && !terrain_data[i].contacts.is_empty()
                    {
                        let inv_i_world = terrain_data[i].inv_i_world;
                        let restitution_coeff = terrain_data[i].restitution_coeff;
                        let bt = if terrain_bounce[i].is_empty() {
                            None
                        } else {
                            Some(terrain_bounce[i].as_slice())
                        };
                        bodies[i].solve_terrain_contacts_velocity_iteration(
                            &mut terrain_data[i].contacts,
                            inv_i_world,
                            None,
                            restitution_coeff,
                            iter,
                            bt,
                        );
                    }
                }

                for manifold in &mut manifolds {
                    let i = manifold.body_a_idx;
                    let j = manifold.body_b_idx;
                    let (left, right) = bodies.split_at_mut(j);
                    Self::solve_pair_manifold_velocity_iteration(
                        manifold,
                        &mut left[i],
                        &mut right[0],
                        None,
                        iter,
                    );
                }
            }

            apply_bottom_up_velocity_shock_pass(
                bodies,
                &mut terrain_data,
                &mut manifolds,
                &support_levels,
                &substep_stable,
            );

            for i in 0..len {
                if !bodies[i].sleeping && !bodies[i].is_baking {
                    bodies[i].apply_terrain_ground_settling(
                        &terrain_data[i].contacts,
                        terrain_data[i].is_wet,
                        substep_stable[i],
                        sub_dt,
                    );
                    if substep_stable[i] && contact_degree[i] > 0 {
                        // WHY: Gentle rolling/sliding decay when CoM is stably supported inside its combined support polygon.
                        bodies[i].angular_velocity *= (1.0 - 4.0 * sub_dt).max(0.0);
                    } else if is_cradled[i] {
                        let lin_sq = bodies[i].linear_velocity.length_squared();
                        let ang_sq = bodies[i].angular_velocity.length_squared();
                        if lin_sq < 0.0625 && ang_sq < 0.25 {
                            // WHY: Point-contact Coulomb friction lacks rolling/torsional torque; damp low-speed micro-jitter in multi-contact bowls/piles.
                            bodies[i].linear_velocity *= (1.0 - 7.0 * sub_dt).max(0.0);
                            bodies[i].angular_velocity *= (1.0 - 9.0 * sub_dt).max(0.0);
                        }
                    }
                }
            }

            for i in 0..len {
                if bodies[i].sleeping || bodies[i].is_baking {
                    continue;
                }
                bodies[i].integrate_transform(sub_dt);
                bodies[i].update_post_integration_terrain_penetrations(
                    &mut terrain_data[i].contacts,
                    sub_dt,
                );
            }
            update_post_integration_pair_penetrations(bodies, &mut manifolds, sub_dt);

            solve_split_impulse_positions(
                bodies,
                &terrain_data,
                &manifolds,
                &support_levels,
                &substep_stable,
                &contact_degree,
            );

            for i in 0..len {
                prev_terrain_cache[i].clear();
                if bodies[i].sleeping {
                    continue;
                }
                for c in &terrain_data[i].contacts {
                    prev_terrain_cache[i].push(CachedTerrainContact {
                        world_pos: bodies[i].position + c.r_world,
                        normal: c.normal,
                        tangent1: c.tangent1,
                        tangent2: c.tangent2,
                        accum_n: c.accum_n,
                        accum_t1: c.accum_t1,
                        accum_t2: c.accum_t2,
                    });
                }

                let mut outcome = RigidStepOutcome::default();
                let destroyed = bodies[i].check_high_energy_impact(
                    world,
                    terrain_data[i].contacts.is_empty(),
                    terrain_data[i].max_impact_speed,
                    terrain_data[i].r_mat,
                    &mut outcome,
                );
                if !outcome.spawned_debris.is_empty() {
                    debris.extend(outcome.spawned_debris);
                }
                if destroyed {
                    // INVARIANT: When shattered by high-energy impact, teleport below culling plane.
                    bodies[i].position.y = -999.0;
                    bodies[i].sleeping = true;
                }
            }

            prev_pair_manifolds = manifolds;
        }

        update_island_sleep_states(bodies, dt, &frame_support_kind, &frame_contact_edges);

        debris
    }

    fn check_high_energy_impact(
        &self,
        world: &WorldMap,
        contacts_empty: bool,
        max_impact_speed: f32,
        r_mat: Mat3,
        outcome: &mut RigidStepOutcome,
    ) -> bool {
        if !contacts_empty && max_impact_speed > 6.5 {
            let impact_ke = 0.5 * self.mass * max_impact_speed * max_impact_speed;
            let fracture_threshold = self.mass * 22.0;
            if impact_ke > fracture_threshold && outcome.spawned_debris.len() < 96 {
                let num_splinters = ((max_impact_speed * 2.5) as usize).clamp(6, 48);
                let stride = (self.boundary_voxels.len() / num_splinters).max(1);
                for (idx, &(r_local, mat_id)) in
                    self.boundary_voxels.iter().step_by(stride).enumerate()
                {
                    let p_world = self.position + r_mat * r_local;
                    let hash = (idx as u32).wrapping_mul(747796405) ^ 2891336453;
                    let rx = (((hash & 0xFF) as f32) / 127.5) - 1.0;
                    let ry = (((hash >> 8) & 0xFF) as f32) / 255.0;
                    let rz = (((hash >> 16) & 0xFF) as f32) / 127.5 - 1.0;
                    let vel = Vec3::new(rx * 4.5, 2.0 + ry * 4.0, rz * 4.5);
                    let packed_col = world.palette.get(mat_id).pack_lod_payload(mat_id);
                    outcome.spawned_debris.push(DebrisParticle {
                        pos_life: [p_world.x, p_world.y, p_world.z, 2.5 + ry * 1.5],
                        vel_color: [vel.x, vel.y, vel.z, f32::from_bits(packed_col)],
                    });
                }

                if self.local_voxels.len() <= 96 && max_impact_speed > 11.0 {
                    outcome.destroyed = true;
                    return true;
                }
            }
        }
        false
    }

    fn integrate_transform(&mut self, sub_dt: f32) {
        self.position += self.linear_velocity * sub_dt;
        let omega_len = self.angular_velocity.length();
        if omega_len > 1e-5 {
            let dq = Quat::from_axis_angle(self.angular_velocity / omega_len, omega_len * sub_dt);
            self.orientation = (dq * self.orientation).normalize();
        }
    }
}
