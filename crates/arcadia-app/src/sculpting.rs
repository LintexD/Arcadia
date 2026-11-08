//! Real-time voxel sculpting, carving, sphere placement, and dynamic island detachment.
//! Owns CPU voxel mutation logic and synchronizes GPU world patches.

use glam::Vec3;

use arcadia_core::RayHit;
use arcadia_physics::{RigidVoxelBody, MAX_RIGID_BODIES};

use crate::app::App;

/// Target entity hit by a scene raycast query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaycastTarget {
    World,
    RigidBody(usize),
}

/// Result of a unified scene raycast against static terrain and dynamic rigid bodies.
#[derive(Clone, Debug)]
pub struct SceneRayHit {
    pub hit: RayHit,
    pub target: RaycastTarget,
    pub hit_point: Vec3,
    pub normal: Vec3,
}

impl App {
    /// Raycasts against static world geometry and all active dynamic rigid voxel bodies,
    /// returning the closest hit point, surface normal, and target entity.
    pub fn raycast_scene(&self, eye: Vec3, dir: Vec3, max_dist: f32) -> Option<SceneRayHit> {
        let mut closest_hit =
            self.world
                .raycast_meters(eye, dir, max_dist)
                .map(|hit| SceneRayHit {
                    hit_point: hit.hit_pos,
                    normal: hit.normal,
                    hit,
                    target: RaycastTarget::World,
                });

        let mut closest_dist = closest_hit.as_ref().map_or(max_dist, |h| h.hit.distance);

        for (idx, body) in self.rigid_bodies.iter().enumerate() {
            if let Some(body_hit) = body.raycast_meters(eye, dir, closest_dist) {
                if body_hit.distance < closest_dist {
                    closest_dist = body_hit.distance;
                    closest_hit = Some(SceneRayHit {
                        hit_point: body_hit.hit_pos,
                        normal: body_hit.normal,
                        hit: body_hit,
                        target: RaycastTarget::RigidBody(idx),
                    });
                }
            }
        }

        closest_hit
    }

    /// Registers a dynamic rigid body's CS64 tree on GPU and adds it to the active simulation list,
    /// evicting sleeping or oldest bodies if the capacity limit is reached.
    pub fn add_rigid_body(&mut self, mut body: RigidVoxelBody) {
        body.gpu_root_node_idx = self.world.register_dynamic_tree(&body.local_tree);
        if self.rigid_bodies.len() >= MAX_RIGID_BODIES {
            if let Some(sleep_idx) = self
                .rigid_bodies
                .iter()
                .position(|b| b.sleeping && !b.is_baking)
            {
                let evicted = self.rigid_bodies.remove(sleep_idx);
                let voxels = evicted.to_world_voxels();
                if !voxels.is_empty() {
                    self.world.set_voxels_global(&voxels);
                }
            } else {
                let evict_idx = self
                    .rigid_bodies
                    .iter()
                    .position(|b| !b.is_baking)
                    .unwrap_or(0);
                self.rigid_bodies.remove(evict_idx);
            }
        }
        self.rigid_bodies.push(body);
    }

    /// Triggers explosive sphere carving at the voxel targeted by the camera crosshair.
    pub fn carve_action(&mut self) {
        let eye = self.player.eye_pos();
        let dir = self.player.forward();
        let Some(scene_hit) = self.raycast_scene(eye, dir, 60.0) else {
            return;
        };
        let hit_pos = scene_hit.hit.hit_pos;

        let remove_r = self.brush_radius + 0.65;
        self.point_lights
            .retain(|pl| Vec3::new(pl[0], pl[1], pl[2]).distance(hit_pos) > remove_r);

        let mut spawned_debris = Vec::new();
        let mut new_split_bodies = Vec::new();

        let blast_wake_dist = self.brush_radius + 3.0;
        let mut i = 0;
        while i < self.rigid_bodies.len() {
            let dist = self.rigid_bodies[i].position.distance(hit_pos);
            let bound_r = self.rigid_bodies[i].bounding_radius();
            if dist <= self.brush_radius + bound_r {
                let outcome = self.rigid_bodies[i].carve_sphere_meters(
                    hit_pos,
                    self.brush_radius,
                    self.scorch_rim,
                    &self.world.palette,
                );
                spawned_debris.extend(outcome.spawned_debris);
                if outcome.destroyed {
                    self.rigid_bodies.swap_remove(i);
                    continue;
                }
                if outcome.modified {
                    let tree = &self.rigid_bodies[i].local_tree;
                    self.rigid_bodies[i].gpu_root_node_idx = self.world.register_dynamic_tree(tree);
                }
                new_split_bodies.extend(outcome.split_bodies);
            } else if dist <= blast_wake_dist + bound_r {
                self.rigid_bodies[i].wake_up();
            }
            i += 1;
        }

        for split in new_split_bodies {
            self.add_rigid_body(split);
        }

        let (max_island_voxels, max_surface_voxels, max_island_depth_m) =
            if self.quality.island_detection_enabled {
                (
                    self.quality.island_max_voxels as usize,
                    self.quality.island_max_surface_voxels as usize,
                    self.quality.island_max_depth_m,
                )
            } else {
                (0, 0, 0.0)
            };
        let event = self.world.carve_sphere_meters_with_limits(
            hit_pos,
            self.brush_radius,
            self.scorch_rim,
            384,
            max_island_voxels,
            max_island_depth_m,
            max_surface_voxels,
        );

        for island_voxels in &event.detached_islands {
            let palette = self.world.palette;
            let Some(body) =
                RigidVoxelBody::from_global_voxels(island_voxels, &palette, Some(hit_pos))
            else {
                continue;
            };
            self.add_rigid_body(body);
        }

        if !event.spawned_debris.is_empty() {
            spawned_debris.extend(event.spawned_debris);
        }

        if let Some(renderer) = &mut self.renderer {
            if !spawned_debris.is_empty() {
                renderer.spawn_debris(&spawned_debris);
            }
            renderer.apply_world_patches(&mut self.world);
        }
    }

    /// Places a sphere of the selected material on the surface targeted by the crosshair.
    pub fn place_action(&mut self) {
        let eye = self.player.eye_pos();
        let dir = self.player.forward();
        let Some(scene_hit) = self.raycast_scene(eye, dir, 60.0) else {
            return;
        };

        let hit = scene_hit.hit;
        let center = hit.hit_pos + hit.normal * (self.brush_radius * 0.6);

        match scene_hit.target {
            RaycastTarget::RigidBody(body_idx) => {
                if let Some(body) = self.rigid_bodies.get_mut(body_idx) {
                    let palette = self.world.palette;
                    if body.place_sphere_meters(
                        center,
                        self.brush_radius,
                        self.selected_mat,
                        &palette,
                    ) {
                        body.gpu_root_node_idx = self.world.register_dynamic_tree(&body.local_tree);
                    }
                }
            }
            RaycastTarget::World => {
                self.world
                    .place_sphere_meters(center, self.brush_radius, self.selected_mat);
            }
        }

        if let Some(renderer) = &mut self.renderer {
            renderer.apply_world_patches(&mut self.world);
        }
    }
}
