//! Island sleep management, support graph evaluation, and wake propagation.

use super::super::types::{BodyPairManifold, ContactManifold, RigidVoxelBody};
use arcadia_world::WorldMap;
use glam::{Mat3, Vec2, Vec3};

const WARM_START_FACTOR: f32 = 0.80;
const WARM_START_MATCH_DIST_SQ: f32 = 0.035 * 0.035;
const WAKE_CONTACT_REL_SPEED_SQ: f32 = 0.12 * 0.12;
const WAKE_CONTACT_PENETRATION: f32 = 0.008;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum SupportSleepKind {
    None = 0,
    SingleContact = 1,
    MultiCradled = 2,
    StablePolygon = 3,
}

#[derive(Default)]
pub(crate) struct TerrainContactData {
    pub(crate) contacts: Vec<ContactManifold>,
    pub(crate) inv_i_world: Mat3,
    pub(crate) r_mat: Mat3,
    pub(crate) max_impact_speed: f32,
    pub(crate) restitution_coeff: f32,
    pub(crate) is_wet: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct CachedTerrainContact {
    pub(crate) world_pos: Vec3,
    pub(crate) normal: Vec3,
    pub(crate) tangent1: Vec3,
    pub(crate) tangent2: Vec3,
    pub(crate) accum_n: f32,
    pub(crate) accum_t1: f32,
    pub(crate) accum_t2: f32,
}

impl RigidVoxelBody {
    pub(crate) fn update_single_sleep_state(
        &mut self,
        dt: f32,
        stably_supported: bool,
        has_contact: bool,
    ) {
        if self.sleeping {
            self.rest_time += dt;
            return;
        }

        let lin_speed_sq = self.linear_velocity.length_squared();
        let ang_speed_sq = self.angular_velocity.length_squared();
        let disp_speed_sq = (self.position - self.prev_position).length_squared() / (dt * dt);

        if stably_supported {
            if lin_speed_sq < 0.16 && ang_speed_sq < 0.30 {
                self.sleep_timer += dt;
                self.rest_time += dt;
                if self.sleep_timer >= 0.25 {
                    self.linear_velocity = Vec3::ZERO;
                    self.angular_velocity = Vec3::ZERO;
                    self.sleeping = true;
                }
            } else if lin_speed_sq < 0.35 && ang_speed_sq < 0.70 {
                self.sleep_timer = 0.0;
                self.rest_time += dt;
            } else {
                self.sleep_timer = 0.0;
                self.rest_time = 0.0;
            }
        } else if has_contact
            && lin_speed_sq < 0.0025
            && ang_speed_sq < 0.0064
            && disp_speed_sq < 0.0025
        {
            self.sleep_timer += dt;
            self.rest_time += dt;
            if self.sleep_timer >= 0.40 {
                self.linear_velocity = Vec3::ZERO;
                self.angular_velocity = Vec3::ZERO;
                self.sleeping = true;
            }
        } else {
            self.sleep_timer = 0.0;
            if lin_speed_sq >= 0.35 || ang_speed_sq >= 0.70 {
                self.rest_time = 0.0;
            }
        }
    }
}

pub(crate) fn wake_unsupported_sleeping_bodies(bodies: &mut [RigidVoxelBody], world: &WorldMap) {
    let len = bodies.len();
    let mut aabbs = Vec::with_capacity(len);
    let mut grounded = vec![false; len];
    let mut queue = Vec::with_capacity(len);

    for (i, body) in bodies.iter().enumerate() {
        aabbs.push(body.world_aabb_meters());
        if body.sleeping && body.check_ground_support(world) {
            grounded[i] = true;
            queue.push(i);
        }
    }

    let mut head = 0usize;
    while head < queue.len() {
        let u = queue[head];
        head += 1;
        let (min_u, max_u) = aabbs[u];

        for v in 0..len {
            if grounded[v] || !bodies[v].sleeping {
                continue;
            }
            let (min_v, max_v) = aabbs[v];
            let x_overlap = min_v.x <= max_u.x + 0.025 && max_v.x >= min_u.x - 0.025;
            let z_overlap = min_v.z <= max_u.z + 0.025 && max_v.z >= min_u.z - 0.025;
            let y_support = max_u.y >= min_v.y - 0.035 && min_u.y <= max_v.y + 0.035;
            if x_overlap && z_overlap && y_support {
                grounded[v] = true;
                queue.push(v);
            }
        }
    }

    for (i, body) in bodies.iter_mut().enumerate() {
        if body.sleeping && !grounded[i] && !body.is_baking {
            body.sleeping = false;
            body.sleep_timer = 0.0;
        }
    }
}

pub(crate) fn should_wake_sleeping_pair(
    m: &BodyPairManifold,
    body_a: &RigidVoxelBody,
    body_b: &RigidVoxelBody,
) -> bool {
    let mut max_rel_speed_sq = 0.0f32;
    let mut max_pen = 0.0f32;

    for pt in &m.points[0..m.point_count] {
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
        max_rel_speed_sq = max_rel_speed_sq.max((vb - va).length_squared());
        max_pen = max_pen.max(pt.penetration);
    }

    max_rel_speed_sq > WAKE_CONTACT_REL_SPEED_SQ || max_pen > WAKE_CONTACT_PENETRATION
}

pub(crate) fn warm_start_pair_manifold_from_cache(
    manifold: &mut BodyPairManifold,
    prev_manifolds: &[BodyPairManifold],
) {
    if prev_manifolds.is_empty() {
        return;
    }

    for pt in &mut manifold.points[0..manifold.point_count] {
        let mut best_match: Option<(&BodyPairManifold, usize)> = None;
        let mut best_dist_sq = WARM_START_MATCH_DIST_SQ;

        for prev in prev_manifolds {
            if prev.body_a_idx != manifold.body_a_idx || prev.body_b_idx != manifold.body_b_idx {
                continue;
            }
            if prev.normal.dot(manifold.normal) <= 0.90 {
                continue;
            }
            for (k, prev_pt) in prev.points[0..prev.point_count].iter().enumerate() {
                let d2 = prev_pt.position.distance_squared(pt.position);
                if d2 < best_dist_sq {
                    best_dist_sq = d2;
                    best_match = Some((prev, k));
                }
            }
        }

        let Some((prev_m, pt_idx)) = best_match else {
            continue;
        };
        let prev_pt = &prev_m.points[pt_idx];
        pt.accum_n = prev_pt.accum_n * WARM_START_FACTOR;
        let pt_vec = (prev_m.tangent1 * prev_pt.accum_t1 + prev_m.tangent2 * prev_pt.accum_t2)
            * WARM_START_FACTOR;
        let max_f = manifold.friction * pt.accum_n;
        pt.accum_t1 = pt_vec.dot(manifold.tangent1).clamp(-max_f, max_f);
        pt.accum_t2 = pt_vec.dot(manifold.tangent2).clamp(-max_f, max_f);
    }
}

pub(crate) fn update_island_sleep_states(
    bodies: &mut [RigidVoxelBody],
    dt: f32,
    frame_support_kind: &[SupportSleepKind],
    frame_contact_edges: &[(usize, usize)],
) {
    let len = bodies.len();
    let mut required_sleep_time = vec![0.25f32; len];
    let mut ready_to_sleep = vec![false; len];

    for (i, body) in bodies.iter_mut().enumerate() {
        if body.sleeping || body.is_baking {
            if !body.is_baking {
                body.rest_time += dt;
            }
            ready_to_sleep[i] = true;
            continue;
        }

        let lin_sq = body.linear_velocity.length_squared();
        let ang_sq = body.angular_velocity.length_squared();
        let disp_sq = (body.position - body.prev_position).length_squared() / (dt * dt);

        match frame_support_kind[i] {
            SupportSleepKind::StablePolygon => {
                required_sleep_time[i] = 0.25;
                if lin_sq < 0.16 && ang_sq < 0.30 {
                    body.sleep_timer += dt;
                    body.rest_time += dt;
                    ready_to_sleep[i] = body.sleep_timer >= 0.25;
                } else if lin_sq < 0.35 && ang_sq < 0.70 {
                    body.sleep_timer = 0.0;
                    body.rest_time += dt;
                } else {
                    body.sleep_timer = 0.0;
                    body.rest_time = 0.0;
                }
            }
            SupportSleepKind::MultiCradled
                if lin_sq < 0.0225 && ang_sq < 0.0625 && disp_sq < 0.0064 =>
            {
                required_sleep_time[i] = 0.28;
                body.sleep_timer += dt;
                body.rest_time += dt;
                ready_to_sleep[i] = body.sleep_timer >= 0.28;
            }
            SupportSleepKind::SingleContact
                if lin_sq < 0.0025 && ang_sq < 0.0064 && disp_sq < 0.0025 =>
            {
                required_sleep_time[i] = 0.40;
                body.sleep_timer += dt;
                body.rest_time += dt;
                ready_to_sleep[i] = body.sleep_timer >= 0.40;
            }
            _ => {
                body.sleep_timer = 0.0;
                if lin_sq >= 0.35 || ang_sq >= 0.70 {
                    body.rest_time = 0.0;
                }
            }
        }
    }

    let mut parent: Vec<usize> = (0..len).collect();
    for &(a, b) in frame_contact_edges {
        if !bodies[a].sleeping && !bodies[b].sleeping {
            let ra = uf_find(&mut parent, a);
            let rb = uf_find(&mut parent, b);
            if ra != rb {
                parent[rb] = ra;
            }
        }
    }

    let mut island_ready = vec![true; len];
    for i in 0..len {
        if !bodies[i].sleeping {
            let root = uf_find(&mut parent, i);
            if !ready_to_sleep[i] {
                island_ready[root] = false;
            }
        }
    }

    // WHY: Synchronize sleeping across connected contact islands so entire piles in a bowl freeze atomically without sleep-wake thrashing.
    for i in 0..len {
        if !bodies[i].sleeping {
            let root = uf_find(&mut parent, i);
            if island_ready[root] && bodies[i].sleep_timer >= required_sleep_time[i] {
                bodies[i].linear_velocity = Vec3::ZERO;
                bodies[i].angular_velocity = Vec3::ZERO;
                bodies[i].sleeping = true;
            }
        }
    }
}

pub(crate) fn is_terrain_support_stable(contacts: &[ContactManifold]) -> bool {
    let mut pts = [Vec2::ZERO; 32];
    let mut count = 0usize;
    let mut max_h_com = 0.0f32;

    for c in contacts {
        if c.normal.y > 0.35 && count < pts.len() {
            pts[count] = Vec2::new(c.r_world.x, c.r_world.z);
            max_h_com = max_h_com.max(-c.r_world.y);
            count += 1;
        }
    }

    is_com_stably_supported(&pts[0..count], max_h_com)
}

pub(crate) fn evaluate_body_support_stability(
    body_idx: usize,
    terrain_contacts: &[ContactManifold],
    manifolds: &[BodyPairManifold],
) -> (bool, bool, bool) {
    let mut pts = [Vec2::ZERO; 32];
    let mut count = 0usize;
    let mut max_h_com = 0.0f32;
    let mut has_any_contact = !terrain_contacts.is_empty();
    let mut distinct_dirs = [Vec3::ZERO; 16];
    let mut dir_count = 0usize;
    let mut neighbor_bodies = [usize::MAX; 16];
    let mut neighbor_count = 0usize;

    for c in terrain_contacts {
        if dir_count < distinct_dirs.len()
            && !distinct_dirs[0..dir_count]
                .iter()
                .any(|d| d.dot(c.normal) > 0.85)
        {
            distinct_dirs[dir_count] = c.normal;
            dir_count += 1;
        }
        if c.normal.y > 0.35 && count < pts.len() {
            pts[count] = Vec2::new(c.r_world.x, c.r_world.z);
            max_h_com = max_h_com.max(-c.r_world.y);
            count += 1;
        }
    }

    for m in manifolds {
        if m.point_count == 0 {
            continue;
        }
        if m.body_b_idx == body_idx {
            has_any_contact = true;
            if neighbor_count < neighbor_bodies.len()
                && !neighbor_bodies[0..neighbor_count].contains(&m.body_a_idx)
            {
                neighbor_bodies[neighbor_count] = m.body_a_idx;
                neighbor_count += 1;
            }
            if dir_count < distinct_dirs.len()
                && !distinct_dirs[0..dir_count]
                    .iter()
                    .any(|d| d.dot(m.normal) > 0.85)
            {
                distinct_dirs[dir_count] = m.normal;
                dir_count += 1;
            }
            if m.normal.y > 0.35 {
                for pt in &m.points[0..m.point_count] {
                    if count < pts.len() {
                        pts[count] = Vec2::new(pt.r_b.x, pt.r_b.z);
                        max_h_com = max_h_com.max(-pt.r_b.y);
                        count += 1;
                    }
                }
            }
        } else if m.body_a_idx == body_idx {
            has_any_contact = true;
            if neighbor_count < neighbor_bodies.len()
                && !neighbor_bodies[0..neighbor_count].contains(&m.body_b_idx)
            {
                neighbor_bodies[neighbor_count] = m.body_b_idx;
                neighbor_count += 1;
            }
            let n_on_a = -m.normal;
            if dir_count < distinct_dirs.len()
                && !distinct_dirs[0..dir_count]
                    .iter()
                    .any(|d| d.dot(n_on_a) > 0.85)
            {
                distinct_dirs[dir_count] = n_on_a;
                dir_count += 1;
            }
            if m.normal.y < -0.35 {
                for pt in &m.points[0..m.point_count] {
                    if count < pts.len() {
                        pts[count] = Vec2::new(pt.r_a.x, pt.r_a.z);
                        max_h_com = max_h_com.max(-pt.r_a.y);
                        count += 1;
                    }
                }
            }
        }
    }

    let partner_count = usize::from(!terrain_contacts.is_empty()) + neighbor_count;
    let is_cradled = partner_count >= 2 || dir_count >= 2;

    (
        is_com_stably_supported(&pts[0..count], max_h_com),
        has_any_contact,
        is_cradled,
    )
}

#[inline]
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

#[inline]
fn cross2(u: Vec2, v: Vec2) -> f32 {
    u.x * v.y - u.y * v.x
}

fn is_com_stably_supported(support_pts: &[Vec2], max_h_com: f32) -> bool {
    if support_pts.len() < 3 {
        return false;
    }

    let mut unique = [Vec2::ZERO; 32];
    let mut n = 0usize;
    for &p in support_pts {
        if n >= unique.len() {
            break;
        }
        let is_dup = unique[0..n]
            .iter()
            .any(|q| (*q - p).length_squared() < 1e-6);
        if !is_dup {
            unique[n] = p;
            n += 1;
        }
    }

    if n < 3 {
        return false;
    }

    unique[0..n].sort_by(|a, b| a.x.total_cmp(&b.x).then_with(|| a.y.total_cmp(&b.y)));

    let mut hull = [Vec2::ZERO; 64];
    let mut k = 0usize;

    for &p in &unique[0..n] {
        while k >= 2 && cross2(hull[k - 1] - hull[k - 2], p - hull[k - 2]) <= 1e-7 {
            k -= 1;
        }
        hull[k] = p;
        k += 1;
    }

    let lower_len = k;
    for &p in unique[0..(n - 1)].iter().rev() {
        while k > lower_len && cross2(hull[k - 1] - hull[k - 2], p - hull[k - 2]) <= 1e-7 {
            k -= 1;
        }
        hull[k] = p;
        k += 1;
    }
    k -= 1;

    if k < 3 {
        return false;
    }

    let mut double_area = 0.0f32;
    let mut min_inward_dist = f32::INFINITY;
    for idx in 0..k {
        let a = hull[idx];
        let b = hull[(idx + 1) % k];
        double_area += cross2(a, b);
        let edge = b - a;
        let len = edge.length();
        if len < 1e-5 {
            continue;
        }
        let inward_dist = cross2(edge, -a) / len;
        min_inward_dist = min_inward_dist.min(inward_dist);
    }

    if double_area <= 1e-5 {
        return false;
    }

    // WHY: Taller CoM requires proportionally wider support margin so tall pillars never balance upright on a 1-voxel stem.
    let required_margin = (0.006 + 0.025 * max_h_com.max(0.0)).min(0.065);
    min_inward_dist >= required_margin
}
