//! First-person kinematic character controller using direct 64-tree `u64` bitwise AABB queries.
//! Owns CPU player movement state, collision detection, and step-up kinematics.
//!
//! Features:
//! - Zero triangle collision meshes — tests player AABB (`0.6m × 1.8m × 0.6m`) directly against
//!   the world's `ChunkTree64` bitmasks.
//! - Smooth wall sliding along X and Z axes independently.
//! - Automatic **1-to-3 voxel step-up** (`6.25 cm` to `18.75 cm`) so walking up terraced grass
//!   and temple staircases feels effortless without needing to spam jump.
//! - Instant toggle (`F` key) between physical **Walk Mode** and free **Fly Mode**.

use crate::rigid_voxel::RigidVoxelBody;
use arcadia_core::{MtfCache, MAT_WATER_CRYSTAL, VOXELS_PER_METER, VOXEL_SIZE};
use arcadia_world::WorldMap;
use glam::{IVec3, Vec3};

/// Per-frame player movement intent.
#[derive(Clone, Copy, Debug, Default)]
pub struct MovementInput {
    /// Local wish direction (`x` = strafe right, `y` = fly up/down, `z` = forward).
    pub move_dir: Vec3,
    pub sprint: bool,
    pub jump: bool,
}

/// First-person player physics state and camera orientation.
pub struct CharacterController {
    /// World-space position of the bottom-center of the player's feet (in meters).
    pub foot_pos: Vec3,
    /// Current linear velocity in m/s.
    pub velocity: Vec3,
    /// Camera yaw in radians (rotation around +Y).
    pub yaw: f32,
    /// Camera pitch in radians (clamped to ±89°).
    pub pitch: f32,
    /// Half-extents in X/Z (0.3m = 0.6m width) and full height in Y (1.8m = 29 voxels).
    pub half_width: f32,
    pub height: f32,
    pub eye_height: f32,
    pub on_ground: bool,
    /// When `false`, player uses gravity + 64-tree AABB collisions; when `true`, free fly-cam.
    pub fly_mode: bool,
    /// Move-to-Front (MTF) cache of quantized voxel AABB collision results `(v_min, v_max) -> bool`
    /// across substeps and auto-step probes within a frame.
    aabb_mtf_cache: MtfCache<(IVec3, IVec3), bool, 8>,
}

impl CharacterController {
    pub fn new(spawn_foot_pos: Vec3, yaw: f32, pitch: f32) -> Self {
        Self {
            foot_pos: spawn_foot_pos,
            velocity: Vec3::ZERO,
            yaw,
            pitch,
            half_width: 0.28,
            height: 1.75,
            eye_height: 1.60,
            on_ground: false,
            fly_mode: false,
            aabb_mtf_cache: MtfCache::new((IVec3::MIN, IVec3::MIN), false),
        }
    }

    /// World-space position of the first-person camera eye.
    #[inline]
    pub fn eye_pos(&self) -> Vec3 {
        self.foot_pos + Vec3::new(0.0, self.eye_height, 0.0)
    }

    /// Unit forward look vector in world space.
    #[inline]
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, -cy * cp).normalize_or_zero()
    }

    /// Horizontal forward vector in the XZ plane.
    #[inline]
    pub fn flat_forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(sy, 0.0, -cy).normalize_or_zero()
    }

    /// Horizontal right vector in the XZ plane.
    #[inline]
    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy, 0.0, sy).normalize_or_zero()
    }

    /// Unit camera-local up vector (`right × forward`), strictly orthogonal to both `forward` and `right`.
    #[inline]
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or_zero()
    }

    /// Rotates camera look angles from raw mouse delta.
    pub fn rotate(&mut self, delta_x: f32, delta_y: f32, sensitivity: f32) {
        self.yaw += delta_x * sensitivity;
        self.pitch = (self.pitch - delta_y * sensitivity).clamp(-1.54, 1.54);
    }

    /// Advances character physics by `dt` seconds against `world` and dynamic `bodies`.
    pub fn update(
        &mut self,
        dt: f32,
        input: MovementInput,
        world: &WorldMap,
        bodies: &mut [RigidVoxelBody],
    ) {
        let dt = dt.min(0.05);

        if self.fly_mode {
            let horiz_speed = if input.sprint { 21.0 } else { 10.5 };
            let vert_speed = 8.5;

            let horiz_wish = (self.flat_forward() * input.move_dir.z
                + self.right() * input.move_dir.x)
                .clamp_length_max(1.0)
                * horiz_speed;
            let vert_wish = input.move_dir.y.clamp(-1.0, 1.0) * vert_speed;

            let horiz_accel = if horiz_wish.length_squared() > 0.01 {
                12.0
            } else {
                6.5
            };
            self.velocity.x += (horiz_wish.x - self.velocity.x) * (horiz_accel * dt).min(1.0);
            self.velocity.z += (horiz_wish.z - self.velocity.z) * (horiz_accel * dt).min(1.0);
            self.velocity.y += (vert_wish - self.velocity.y) * (16.0 * dt).min(1.0);

            self.aabb_mtf_cache.clear();
            let total_delta = self.velocity * dt;
            let substeps =
                ((total_delta.length() / (VOXEL_SIZE * 0.8)).ceil() as usize).clamp(1, 8);
            let step_delta = total_delta / (substeps as f32);

            for _ in 0..substeps {
                self.move_substep(step_delta, world, bodies);
            }

            if self.on_ground && input.move_dir.y <= 0.0 {
                self.fly_mode = false;
                self.velocity.y = 0.0;
            }

            self.foot_pos.y = self.foot_pos.y.clamp(1.0, 120.0);
            return;
        }

        // Check if player's feet/waist are submerged in crystal water (`MAT_WATER_CRYSTAL`):
        let waist_v = ((self.foot_pos + Vec3::new(0.0, 0.65, 0.0)) * (VOXELS_PER_METER as f32))
            .floor()
            .as_ivec3();
        let feet_v = ((self.foot_pos + Vec3::new(0.0, 0.15, 0.0)) * (VOXELS_PER_METER as f32))
            .floor()
            .as_ivec3();
        let in_water = world.get_voxel(waist_v.x, waist_v.y, waist_v.z) == MAT_WATER_CRYSTAL
            || world.get_voxel(feet_v.x, feet_v.y, feet_v.z) == MAT_WATER_CRYSTAL;

        // Walk / swim mode with gravity, buoyancy, wall-sliding, and 3-voxel (18.75cm) auto-step-up:
        let base_speed = if input.sprint { 8.5 } else { 5.0 };
        let walk_speed = if in_water {
            base_speed * 0.68
        } else {
            base_speed
        };
        let horiz_wish = (self.flat_forward() * input.move_dir.z + self.right() * input.move_dir.x)
            .clamp_length_max(1.0)
            * walk_speed;

        let accel = if in_water {
            8.0
        } else if self.on_ground {
            18.0
        } else {
            6.0
        };
        self.velocity.x += (horiz_wish.x - self.velocity.x) * (accel * dt).min(1.0);
        self.velocity.z += (horiz_wish.z - self.velocity.z) * (accel * dt).min(1.0);

        if in_water {
            if input.jump {
                self.velocity.y = (self.velocity.y + 18.0 * dt).min(4.2);
            } else {
                self.velocity.y = (self.velocity.y - 5.5 * dt).max(-2.5);
                self.velocity.y *= (1.0 - 3.5 * dt).max(0.0);
            }
        } else {
            if self.on_ground && input.jump {
                self.velocity.y = 6.8;
                self.on_ground = false;
            }

            self.velocity.y = (self.velocity.y - 18.5 * dt).max(-38.0);
        }

        self.aabb_mtf_cache.clear();

        let total_delta = self.velocity * dt;
        let substeps = ((total_delta.length() / (VOXEL_SIZE * 0.8)).ceil() as usize).clamp(1, 8);
        let step_delta = total_delta / (substeps as f32);

        for _ in 0..substeps {
            self.move_substep(step_delta, world, bodies);
        }

        // WHY: Eject player vertically if spawned inside terrain geometry.
        if self.collides_at(self.foot_pos, world, bodies) {
            for up in 1..=32 {
                let candidate = self.foot_pos + Vec3::new(0.0, up as f32 * VOXEL_SIZE, 0.0);
                if !self.collides_at(candidate, world, bodies) {
                    self.foot_pos = candidate;
                    self.velocity.y = 0.0;
                    break;
                }
            }
        }
    }

    fn move_substep(&mut self, delta: Vec3, world: &WorldMap, bodies: &mut [RigidVoxelBody]) {
        let max_step_height = 3.1 * VOXEL_SIZE;

        // 1. Move along X with auto-step:
        if delta.x.abs() > 1e-6 {
            let target_x = self.foot_pos + Vec3::new(delta.x, 0.0, 0.0);
            if !self.collides_at(target_x, world, bodies) {
                self.foot_pos = target_x;
            } else if self.on_ground
                && self.try_auto_step(Vec3::new(delta.x, 0.0, 0.0), max_step_height, world, bodies)
            {
                // WHY: Auto-step traversal successfully elevated player over voxel threshold.
            } else {
                Self::push_bodies_at(
                    target_x,
                    self.half_width,
                    self.height,
                    Vec3::new(delta.x.signum() * 1.5, 0.0, 0.0),
                    bodies,
                );
                self.velocity.x = 0.0;
            }
        }

        // 2. Move along Z with auto-step:
        if delta.z.abs() > 1e-6 {
            let target_z = self.foot_pos + Vec3::new(0.0, 0.0, delta.z);
            if !self.collides_at(target_z, world, bodies) {
                self.foot_pos = target_z;
            } else if self.on_ground
                && self.try_auto_step(Vec3::new(0.0, 0.0, delta.z), max_step_height, world, bodies)
            {
                // WHY: Auto-step traversal successfully elevated player over voxel threshold.
            } else {
                Self::push_bodies_at(
                    target_z,
                    self.half_width,
                    self.height,
                    Vec3::new(0.0, 0.0, delta.z.signum() * 1.5),
                    bodies,
                );
                self.velocity.z = 0.0;
            }
        }

        // 3. Move along Y (gravity / jump):
        if delta.y.abs() > 1e-6 {
            let target_y = self.foot_pos + Vec3::new(0.0, delta.y, 0.0);
            if !self.collides_at(target_y, world, bodies) {
                self.foot_pos = target_y;
                self.on_ground = false;
            } else {
                if delta.y < 0.0 {
                    self.on_ground = true;
                }
                self.velocity.y = 0.0;
            }
        }
    }

    fn push_bodies_at(
        foot: Vec3,
        half_w: f32,
        h: f32,
        impulse: Vec3,
        bodies: &mut [RigidVoxelBody],
    ) {
        let bmin = Vec3::new(foot.x - half_w, foot.y + 1e-3, foot.z - half_w);
        let bmax = Vec3::new(foot.x + half_w, foot.y + h, foot.z + half_w);
        for body in bodies.iter_mut() {
            if body.compound_intersects_aabb(bmin, bmax) {
                body.linear_velocity += impulse * (body.inv_mass.min(1.5));
                body.wake_up();
            }
        }
    }

    fn try_auto_step(
        &mut self,
        horiz_delta: Vec3,
        max_step_h: f32,
        world: &WorldMap,
        bodies: &[RigidVoxelBody],
    ) -> bool {
        for step_voxels in 1..=3 {
            let lift = (step_voxels as f32) * VOXEL_SIZE + 1e-3;
            if lift > max_step_h {
                break;
            }
            let stepped_pos = self.foot_pos + Vec3::new(0.0, lift, 0.0) + horiz_delta;
            if !self.collides_at(stepped_pos, world, bodies) {
                self.foot_pos = stepped_pos;
                self.on_ground = true;
                return true;
            }
        }
        false
    }

    #[inline]
    pub fn collides_at(&mut self, foot: Vec3, world: &WorldMap, bodies: &[RigidVoxelBody]) -> bool {
        let bmin = Vec3::new(
            foot.x - self.half_width,
            foot.y + 1e-3,
            foot.z - self.half_width,
        );
        let bmax = Vec3::new(
            foot.x + self.half_width,
            foot.y + self.height,
            foot.z + self.half_width,
        );
        let v_min = (bmin * (VOXELS_PER_METER as f32)).floor().as_ivec3();
        let v_max = ((bmax - Vec3::splat(1e-5)) * (VOXELS_PER_METER as f32))
            .floor()
            .as_ivec3();
        let hit_world = self
            .aabb_mtf_cache
            .get_or_insert_with((v_min, v_max), |_| world.intersects_aabb_meters(bmin, bmax));
        if hit_world {
            return true;
        }

        // Dynamic rigid body collision:
        let player_center = (bmin + bmax) * 0.5;
        let player_radius = (bmax - bmin).length() * 0.5;
        for body in bodies {
            if body.position.distance_squared(player_center)
                > (body.bounding_radius() + player_radius).powi(2)
            {
                continue;
            }
            if body.compound_intersects_aabb(bmin, bmax) {
                return true;
            }
        }

        false
    }
}
