//! Integration tests for voxel sculpting, dynamic body carving, and baking routines.

use arcadia_app::sculpting::RaycastTarget;
use arcadia_app::{App, CameraView};
use arcadia_core::{MAT_AIR, MAT_CRYSTAL_CYAN, MAT_STONE_DEEP};
use arcadia_physics::{RigidVoxelBody, MAX_RIGID_BODIES};
use arcadia_world::EMPTY_CHUNK_SENTINEL;
use glam::{IVec3, Vec3};

#[test]
fn test_carve_and_split_dynamic_rigid_body() {
    let mut app = App::new_test();

    // 16x2x2 stone beam along X:
    let mut voxels = Vec::new();
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..16 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let body = RigidVoxelBody::from_global_voxels(&voxels, &app.world.palette, None)
        .expect("INVARIANT: Valid voxels must create RigidVoxelBody");
    app.add_rigid_body(body);
    assert_eq!(app.rigid_bodies.len(), 1);
    assert_ne!(app.rigid_bodies[0].gpu_root_node_idx, EMPTY_CHUNK_SENTINEL);

    // Position player to look directly at the center of the beam along -Z:
    let beam_center = app.rigid_bodies[0].position;
    app.player.foot_pos = beam_center + Vec3::new(0.0, -app.player.eye_height, 2.0);
    app.player.yaw = 0.0;
    app.player.pitch = 0.0;

    // Verify raycast_scene hits the rigid body:
    let scene_hit = app.raycast_scene(app.player.eye_pos(), app.player.forward(), 10.0);
    assert!(scene_hit.is_some());
    let hit = scene_hit.expect("INVARIANT: Hit verified");
    assert_eq!(hit.target, RaycastTarget::RigidBody(0));

    // Carve through the middle of the beam:
    app.brush_radius = 0.15;
    app.carve_action();

    // INVARIANT: Carving beam in two must produce two unique rigid bodies.
    assert_eq!(
        app.rigid_bodies.len(),
        2,
        "Carving beam in two must produce two unique rigid bodies"
    );
    let body_a = &app.rigid_bodies[0];
    let body_b = &app.rigid_bodies[1];

    assert_ne!(body_a.gpu_root_node_idx, EMPTY_CHUNK_SENTINEL);
    assert_ne!(body_b.gpu_root_node_idx, EMPTY_CHUNK_SENTINEL);
    assert_ne!(body_a.gpu_root_node_idx, body_b.gpu_root_node_idx);

    assert!(!body_a.local_voxels.is_empty());
    assert!(!body_b.local_voxels.is_empty());

    let sep = (body_a.position - body_b.position).length();
    assert!(
        sep > 0.1,
        "Split pieces must be spatially separated, got {}",
        sep
    );
}

#[test]
fn test_place_voxels_on_dynamic_rigid_body() {
    let mut app = App::new_test();

    let mut voxels = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                voxels.push((IVec3::new(x, y, z), MAT_STONE_DEEP));
            }
        }
    }
    let body = RigidVoxelBody::from_global_voxels(&voxels, &app.world.palette, None)
        .expect("INVARIANT: Valid voxels must create RigidVoxelBody");
    app.add_rigid_body(body);
    let initial_voxel_count = app.rigid_bodies[0].local_voxels.len();

    let body_pos = app.rigid_bodies[0].position;
    app.player.foot_pos = body_pos + Vec3::new(0.0, -app.player.eye_height, 2.0);
    app.player.yaw = 0.0;
    app.player.pitch = 0.0;

    app.brush_radius = 0.10;
    app.selected_mat = MAT_CRYSTAL_CYAN;
    app.place_action();

    assert_eq!(app.rigid_bodies.len(), 1);
    assert!(
        app.rigid_bodies[0].local_voxels.len() > initial_voxel_count,
        "Placing voxels on rigid body must increase voxel count"
    );
    assert_ne!(app.rigid_bodies[0].gpu_root_node_idx, EMPTY_CHUNK_SENTINEL);
    assert!(app.rigid_bodies[0].emissive_light.is_some());
}

#[test]
fn test_rigid_body_bakes_into_world_after_timeout() {
    let mut app = App::new_test();
    app.quality.object_bake_timeout_s = 5.0;

    // Ground support beneath:
    app.world.set_voxels_global(&[
        (IVec3::new(20, 19, 20), MAT_STONE_DEEP),
        (IVec3::new(21, 19, 20), MAT_STONE_DEEP),
    ]);

    let original_voxels = vec![
        (IVec3::new(20, 20, 20), MAT_STONE_DEEP),
        (IVec3::new(21, 20, 20), MAT_STONE_DEEP),
    ];
    let mut body = RigidVoxelBody::from_global_voxels(&original_voxels, &app.world.palette, None)
        .expect("INVARIANT: Valid voxels must create RigidVoxelBody");
    body.sleeping = true;
    body.linear_velocity = Vec3::ZERO;
    body.angular_velocity = Vec3::ZERO;
    body.rest_time = 4.90;
    app.add_rigid_body(body);

    assert_eq!(app.rigid_bodies.len(), 1);
    assert_eq!(app.world.get_voxel(20, 20, 20), MAT_AIR);

    let (_debris, instances) = app.step_and_collect_rigid_bodies(0.05, None);
    assert_eq!(app.rigid_bodies.len(), 1);
    assert_eq!(instances.len(), 1);
    assert_eq!(app.world.get_voxel(20, 20, 20), MAT_AIR);

    let (_debris, instances) = app.step_and_collect_rigid_bodies(0.10, None);
    assert_eq!(
        app.rigid_bodies.len(),
        1,
        "Body remains in app while baking"
    );
    assert!(
        app.rigid_bodies[0].is_baking,
        "Body starts async rasterization"
    );
    assert_eq!(
        instances.len(),
        1,
        "Body remains rendered on GPU during worker computation"
    );

    std::thread::sleep(std::time::Duration::from_millis(50));

    let (_debris, instances) = app.step_and_collect_rigid_bodies(0.01, None);
    assert_eq!(
        app.rigid_bodies.len(),
        0,
        "Baked body retired from simulation"
    );
    assert_eq!(instances.len(), 0);
    assert_eq!(app.world.get_voxel(20, 20, 20), MAT_STONE_DEEP);
    assert_eq!(app.world.get_voxel(21, 20, 20), MAT_STONE_DEEP);
}

#[test]
fn test_rigid_body_visibility_gated_baking() {
    let mut app = App::new_test();
    app.quality.object_bake_timeout_s = 5.0;

    // Ground support beneath:
    app.world.set_voxels_global(&[
        (IVec3::new(20, 19, 20), MAT_STONE_DEEP),
        (IVec3::new(21, 19, 20), MAT_STONE_DEEP),
    ]);

    let original_voxels = vec![
        (IVec3::new(20, 20, 20), MAT_STONE_DEEP),
        (IVec3::new(21, 20, 20), MAT_STONE_DEEP),
    ];
    let mut body = RigidVoxelBody::from_global_voxels(&original_voxels, &app.world.palette, None)
        .expect("INVARIANT: Valid voxels must create RigidVoxelBody");
    body.sleeping = true;
    body.linear_velocity = Vec3::ZERO;
    body.angular_velocity = Vec3::ZERO;
    body.rest_time = 4.95;
    let body_pos = body.position;
    app.add_rigid_body(body);

    let looking_at_camera = CameraView {
        eye_pos: body_pos - Vec3::new(0.0, 0.0, 3.0),
        forward: Vec3::Z,
        right: Vec3::X,
        up: Vec3::Y,
        fov_y_rad: 60.0_f32.to_radians(),
        aspect_ratio: 16.0 / 9.0,
    };

    // 1. Looking directly at resting body: does not bake at bake_timeout (4.95 + 0.10 = 5.05s >= 5.0s)
    let (_debris, instances) = app.step_and_collect_rigid_bodies(0.10, Some(looking_at_camera));
    assert_eq!(
        app.rigid_bodies.len(),
        1,
        "Visible body must not bake at bake_timeout"
    );
    assert_eq!(instances.len(), 1);
    assert_eq!(app.world.get_voxel(20, 20, 20), MAT_AIR);

    // 2. Camera looks away (body is behind camera): starts async rasterization on background worker
    let looking_away_camera = CameraView {
        eye_pos: body_pos - Vec3::new(0.0, 0.0, 3.0),
        forward: -Vec3::Z,
        right: -Vec3::X,
        up: Vec3::Y,
        fov_y_rad: 60.0_f32.to_radians(),
        aspect_ratio: 16.0 / 9.0,
    };
    let (_debris, instances) = app.step_and_collect_rigid_bodies(0.01, Some(looking_away_camera));
    assert_eq!(
        app.rigid_bodies.len(),
        1,
        "Body remains in app during async rasterization"
    );
    assert!(
        app.rigid_bodies[0].is_baking,
        "Body must begin background baking"
    );
    assert_eq!(
        instances.len(),
        1,
        "Body remains rendered on GPU to prevent visual pop-in"
    );

    // WHY: Allow background worker task to complete async rasterization.
    std::thread::sleep(std::time::Duration::from_millis(50));

    // WHY: Next frame commits bake and retires the dynamic body from simulation.
    let (_debris, instances) = app.step_and_collect_rigid_bodies(0.01, Some(looking_away_camera));
    assert_eq!(
        app.rigid_bodies.len(),
        0,
        "Baked body retired from simulation"
    );
    assert_eq!(instances.len(), 0);
    assert_eq!(app.world.get_voxel(20, 20, 20), MAT_STONE_DEEP);
    assert_eq!(app.world.get_voxel(21, 20, 20), MAT_STONE_DEEP);

    // 3. Staring directly past forced_timeout forces the bake
    app.world.set_voxels_global(&[
        (IVec3::new(20, 20, 20), MAT_AIR),
        (IVec3::new(21, 20, 20), MAT_AIR),
    ]);
    let mut forced_body =
        RigidVoxelBody::from_global_voxels(&original_voxels, &app.world.palette, None)
            .expect("INVARIANT: Valid voxels must create RigidVoxelBody");
    forced_body.sleeping = true;
    forced_body.linear_velocity = Vec3::ZERO;
    forced_body.angular_velocity = Vec3::ZERO;
    // forced_timeout = 5.0 * 3.0 = 15.0s
    forced_body.rest_time = 14.95;
    app.add_rigid_body(forced_body);

    let (_debris, instances) = app.step_and_collect_rigid_bodies(0.10, Some(looking_at_camera));
    assert_eq!(
        app.rigid_bodies.len(),
        1,
        "Body remains in app while baking"
    );
    assert!(
        app.rigid_bodies[0].is_baking,
        "Body exceeding forced timeout begins baking"
    );
    assert_eq!(
        instances.len(),
        1,
        "Body remains visible while worker computes voxels"
    );

    std::thread::sleep(std::time::Duration::from_millis(50));

    let (_debris, instances) = app.step_and_collect_rigid_bodies(0.01, Some(looking_at_camera));
    assert_eq!(
        app.rigid_bodies.len(),
        0,
        "Body retired once bake completes"
    );
    assert_eq!(instances.len(), 0);
    assert_eq!(app.world.get_voxel(20, 20, 20), MAT_STONE_DEEP);
    assert_eq!(app.world.get_voxel(21, 20, 20), MAT_STONE_DEEP);
}

#[test]
fn test_rigid_body_baking_time_slicing() {
    let mut app = App::new_test();
    app.quality.object_bake_timeout_s = 5.0;

    app.world.set_voxels_global(&[
        (IVec3::new(20, 19, 20), MAT_STONE_DEEP),
        (IVec3::new(40, 19, 20), MAT_STONE_DEEP),
    ]);

    let voxels_a = vec![(IVec3::new(20, 20, 20), MAT_STONE_DEEP)];
    let voxels_b = vec![(IVec3::new(40, 20, 20), MAT_STONE_DEEP)];

    let mut body_a = RigidVoxelBody::from_global_voxels(&voxels_a, &app.world.palette, None)
        .expect("INVARIANT: Valid voxels must create RigidVoxelBody");
    body_a.sleeping = true;
    body_a.linear_velocity = Vec3::ZERO;
    body_a.angular_velocity = Vec3::ZERO;
    body_a.rest_time = 6.0;
    app.add_rigid_body(body_a);

    let mut body_b = RigidVoxelBody::from_global_voxels(&voxels_b, &app.world.palette, None)
        .expect("INVARIANT: Valid voxels must create RigidVoxelBody");
    body_b.sleeping = true;
    body_b.linear_velocity = Vec3::ZERO;
    body_b.angular_velocity = Vec3::ZERO;
    body_b.rest_time = 6.0;
    app.add_rigid_body(body_b);

    assert_eq!(app.rigid_bodies.len(), 2);

    // Frame 1: Only 1 body dispatches for baking:
    let (_debris, instances) = app.step_and_collect_rigid_bodies(0.01, None);
    assert_eq!(
        app.rigid_bodies.len(),
        2,
        "Both bodies remain in app during async rasterization"
    );
    assert_eq!(
        instances.len(),
        2,
        "Both bodies remain rendered on GPU to prevent visual pop-in"
    );
    assert_eq!(
        app.rigid_bodies.iter().filter(|b| b.is_baking).count(),
        1,
        "Only 1 body bakes at a time"
    );

    std::thread::sleep(std::time::Duration::from_millis(50));

    // Frame 2: First body commits into world; second body begins baking:
    let (_debris, instances) = app.step_and_collect_rigid_bodies(0.01, None);
    assert_eq!(
        app.rigid_bodies.len(),
        1,
        "First body retired after completion"
    );
    assert_eq!(instances.len(), 1);
    assert_eq!(
        app.rigid_bodies.iter().filter(|b| b.is_baking).count(),
        1,
        "Second body now baking"
    );

    std::thread::sleep(std::time::Duration::from_millis(50));

    // Frame 3: Second body commits and is retired:
    let (_debris, instances) = app.step_and_collect_rigid_bodies(0.01, None);
    assert_eq!(
        app.rigid_bodies.len(),
        0,
        "Second body retired after completion"
    );
    assert_eq!(instances.len(), 0);
}

#[test]
fn test_eviction_ignores_baking_bodies() {
    let mut app = App::new_test();
    let voxels = vec![(IVec3::new(10, 10, 10), MAT_STONE_DEEP)];
    for _ in 0..MAX_RIGID_BODIES {
        let body = RigidVoxelBody::from_global_voxels(&voxels, &app.world.palette, None)
            .expect("INVARIANT: Valid voxels must create body");
        app.add_rigid_body(body);
    }
    assert_eq!(app.rigid_bodies.len(), MAX_RIGID_BODIES);

    // Mark body 0 as is_baking:
    app.rigid_bodies[0].is_baking = true;
    app.rigid_bodies[0].sleeping = true;
    let baking_id = app.rigid_bodies[0].id;

    // Mark body 1 as ordinary sleeping:
    app.rigid_bodies[1].sleeping = true;
    let normal_sleep_id = app.rigid_bodies[1].id;

    // Adding another body must evict body 1, NOT body 0:
    let extra = RigidVoxelBody::from_global_voxels(&voxels, &app.world.palette, None)
        .expect("INVARIANT: Valid voxels must create body");
    app.add_rigid_body(extra);

    assert!(
        app.rigid_bodies.iter().any(|b| b.id == baking_id),
        "Baking body must not be evicted"
    );
    assert!(
        !app.rigid_bodies.iter().any(|b| b.id == normal_sleep_id),
        "Normal sleeping body should be evicted"
    );
}
