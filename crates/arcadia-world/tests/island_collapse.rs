//! Integration tests for bitwise island detection and compressive structural support.
//! Owns CPU verification of island streaming, graph traversal, and culling.

use arcadia_core::coord::{child_bit_index, child_bit_mask};
use arcadia_core::{
    ChunkTree64, MaterialPalette, MAT_AIR, MAT_BEDROCK, MAT_STONE_DEEP, VOXEL_SIZE,
};
use arcadia_world::islands::*;
use arcadia_world::world_map::WorldMap;
use glam::{IVec3, UVec3, Vec3};

#[test]
fn test_intra_brick_bitwise_dilation() {
    let b0 = 1u64 << child_bit_index(0, 0, 0);
    let b1 = 1u64 << child_bit_index(1, 0, 0);
    let b2 = 1u64 << child_bit_index(2, 0, 0);
    let b3 = 1u64 << child_bit_index(2, 1, 0);
    let b4 = 1u64 << child_bit_index(2, 1, 1);
    let isolated = 1u64 << child_bit_index(3, 3, 3);

    let solid = b0 | b1 | b2 | b3 | b4 | isolated;
    let filled = flood_fill_brick_u64(b0, solid);
    assert_eq!(filled, b0 | b1 | b2 | b3 | b4);
    assert_eq!(filled & isolated, 0);
}

#[test]
fn test_cantilever_and_floating_arch_edge_cases() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    for z in 10..26 {
        for x in 10..26 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    for y in 5..=20 {
        voxels.push((12, y as u32, 12, MAT_STONE_DEEP));
    }

    for x in 13..=24 {
        voxels.push((x as u32, 20, 12, MAT_STONE_DEEP));
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    assert!(is_externally_anchored(&world, 12, 20, 12));
    assert!(!is_externally_anchored(&world, 18, 20, 12));
    assert!(!is_externally_anchored(&world, 24, 20, 12));
}

#[test]
fn test_anti_slicing_boundary_anchors() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    for z in 0..32 {
        for x in 0..32 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    for x in 4..=28 {
        voxels.push((x as u32, 10, 12, MAT_STONE_DEEP));
    }
    for y in 5..=9 {
        voxels.push((8, y as u32, 12, MAT_STONE_DEEP));
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let min_v = IVec3::new(8, 0, 8);
    let max_v = IVec3::new(20, 16, 20);

    let comps = detect_disconnected_island_components(&world, min_v, max_v, 262_144);
    assert_eq!(
        comps.len(),
        0,
        "Bridge extending through box boundaries must not be sliced in a straight line"
    );

    let mut severed_voxels = Vec::new();
    for z in 0..32 {
        for x in 0..32 {
            for y in 0..=4 {
                severed_voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    for y in 5..=9 {
        severed_voxels.push((8, y as u32, 12, MAT_STONE_DEEP));
    }
    for x in 9..=19 {
        severed_voxels.push((x as u32, 10, 12, MAT_STONE_DEEP));
    }
    let severed_chunk = ChunkTree64::from_sparse_voxels(&severed_voxels, &palette);
    let severed_world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![severed_chunk], palette);

    let severed_comps =
        detect_disconnected_island_components(&severed_world, min_v, max_v, 262_144);
    assert_eq!(
        severed_comps.len(),
        1,
        "Genuinely severed floating segment must detach as an island"
    );
    assert_eq!(severed_comps[0].len(), 11);
}

#[test]
fn test_world_region_snapshot_island_detection() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 20..44 {
        for x in 20..44 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    for y in 5..=24 {
        for z in 31..33 {
            for x in 31..33 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(
        UVec3::new(2, 2, 2),
        vec![
            chunk,
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
        ],
        palette,
    );
    let center_m = Vec3::new(2.0, 0.5, 2.0);
    let radius_m = 0.22;
    let (_tot, _deb, bbox) = world.carve_crater(center_m, radius_m, false, 0, 240);
    assert!(bbox.is_some(), "Bbox must be Some");
    let (min_v, max_v) =
        bbox.expect("INVARIANT: crater bounding box must exist for carved test region");
    let snapshot = WorldRegionSnapshot::capture(&world, min_v, max_v);
    let comps_direct = detect_disconnected_island_components(&world, min_v, max_v, 262_144);
    let comps_snapshot = detect_disconnected_island_components(&snapshot, min_v, max_v, 262_144);
    assert_eq!(
        comps_direct.len(),
        1,
        "Direct island detection must find 1 island"
    );
    assert_eq!(
        comps_snapshot.len(),
        1,
        "Snapshot island detection must find 1 island"
    );
}

#[test]
fn test_arch_severing_detachment() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 10..54 {
        for x in 0..140 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    for y in 5..=20 {
        for z in 30..32 {
            for x in 20..22 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    for y in 5..=20 {
        for z in 30..32 {
            for x in 120..122 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    for x in 20..=122 {
        for z in 30..32 {
            for y in 20..22 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(
        UVec3::new(2, 2, 2),
        vec![
            chunk,
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
        ],
        palette,
    );

    let pillar1_top_cut = Vec3::new(21.0 * VOXEL_SIZE, 18.0 * VOXEL_SIZE, 31.0 * VOXEL_SIZE);
    let _event1 = world.carve_sphere_meters(pillar1_top_cut, 0.18, false, 0);

    let pillar2_top_cut = Vec3::new(121.0 * VOXEL_SIZE, 18.0 * VOXEL_SIZE, 31.0 * VOXEL_SIZE);
    let event2 = world.carve_sphere_meters(pillar2_top_cut, 0.18, false, 0);

    let arch_center = Vec3::new(71.0 * VOXEL_SIZE, 21.0 * VOXEL_SIZE, 31.0 * VOXEL_SIZE);
    let event3 = world.carve_sphere_meters(arch_center, 0.18, false, 0);

    assert!(
        !event2.detached_islands.is_empty() || !event3.detached_islands.is_empty(),
        "Arch must detach either when severed from pillars or when carved!"
    );
}

#[test]
fn test_arch_bottom_severing_detachment() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();
    for z in 10..54 {
        for x in 0..140 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    for y in 5..=20 {
        for z in 30..32 {
            for x in 20..22 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    for y in 5..=20 {
        for z in 30..32 {
            for x in 120..122 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }
    for x in 20..=122 {
        for z in 30..32 {
            for y in 20..22 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(
        UVec3::new(2, 2, 2),
        vec![
            chunk,
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
            ChunkTree64::new_empty(),
        ],
        palette,
    );

    let p1_base = Vec3::new(21.0 * VOXEL_SIZE, 7.0 * VOXEL_SIZE, 31.0 * VOXEL_SIZE);
    let _event1 = world.carve_sphere_meters(p1_base, 0.18, false, 0);

    let p2_base = Vec3::new(121.0 * VOXEL_SIZE, 7.0 * VOXEL_SIZE, 31.0 * VOXEL_SIZE);
    let event2 = world.carve_sphere_meters(p2_base, 0.18, false, 0);

    assert_eq!(
        event2.detached_islands.len(),
        1,
        "Entire severed arch structure must detach when both base supports are cut"
    );
}

#[test]
fn test_downward_support_propagation_stairs() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    for z in 10..26 {
        for x in 10..26 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    for y in 5..=20 {
        voxels.push((12, y as u32, 12, MAT_STONE_DEEP));
    }

    for step in 1..=10 {
        let x = 12 + step;
        let y = 20 - step;
        voxels.push((x, y + 1, 12, MAT_STONE_DEEP));
        voxels.push((x, y, 12, MAT_STONE_DEEP));
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let min_v = IVec3::new(10, 0, 10);
    let max_v = IVec3::new(26, 24, 16);

    let comps = detect_disconnected_island_components(&world, min_v, max_v, 262_144);
    assert_eq!(
        comps.len(),
        0,
        "Downward staircase supported from pillar top must not detach"
    );
}

#[test]
fn test_perimeter_seeded_floating_island_detection() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // Bedrock floor: y in 0..=4
    for z in 10..30 {
        for x in 10..30 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    // A floating 3x3x3 block at y = 20..23, x = 18..21, z = 18..21 (27 voxels total)
    for z in 18..21 {
        for y in 20..23 {
            for x in 18..21 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    // Carve a small sphere adjacent to the floating block: center at (17.5, 21.0, 19.5), radius 1.5
    let center_v = Vec3::new(17.5, 21.0, 19.5);
    let radius_v = 1.5;
    let _ = world.carve_crater(center_v * VOXEL_SIZE, radius_v * VOXEL_SIZE, false, 0, 0);

    let islands = detect_disconnected_islands_from_carve(&world, center_v, radius_v, 262_144);
    assert_eq!(
        islands.len(),
        1,
        "Perimeter search must detect the detached floating block"
    );
    assert!(
        !islands[0].is_empty(),
        "Detected island must contain solid voxels"
    );
}

#[test]
fn test_perimeter_seeded_grounded_pillar_remains_anchored() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // Bedrock floor: y in 0..=4
    for z in 10..30 {
        for x in 10..30 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    // Grounded pillar from y = 5 up to y = 25 at x = 19..22, z = 19..22
    for y in 5..=25 {
        for z in 19..22 {
            for x in 19..22 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    // Carve a side notch into the pillar at y = 15, but leave the pillar standing
    let center_v = Vec3::new(18.5, 15.0, 20.0);
    let radius_v = 1.2;
    let _ = world.carve_crater(center_v * VOXEL_SIZE, radius_v * VOXEL_SIZE, false, 0, 0);

    let islands = detect_disconnected_islands_from_carve(&world, center_v, radius_v, 262_144);
    assert_eq!(
        islands.len(),
        0,
        "Grounded pillar must not be detected as a disconnected island"
    );
}

#[test]
fn test_perimeter_seeded_empty_seeds_in_air() {
    let palette = MaterialPalette::default();
    let chunk = ChunkTree64::new_empty();
    let world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let center_v = Vec3::new(50.0, 50.0, 50.0);
    let radius_v = 4.0;
    let islands = detect_disconnected_islands_from_carve(&world, center_v, radius_v, 262_144);
    assert!(
        islands.is_empty(),
        "Carve in empty air must return zero islands"
    );
}

#[test]
fn test_nearby_edit_does_not_collapse_bridge_or_ceiling() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    for z in 10..30 {
        for x in 10..50 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_BEDROCK));
            }
        }
    }

    for y in 5..=20 {
        voxels.push((15, y as u32, 20, MAT_STONE_DEEP));
        voxels.push((45, y as u32, 20, MAT_STONE_DEEP));
    }

    for x in 15..=45 {
        voxels.push((x as u32, 20, 20, MAT_STONE_DEEP));
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let carve_pos = Vec3::new(30.0 * VOXEL_SIZE, 5.0 * VOXEL_SIZE, 20.0 * VOXEL_SIZE);
    let event = world.carve_sphere_meters(carve_pos, 0.20, false, 0);

    assert!(
        event.detached_islands.is_empty(),
        "Bridge must not collapse when an edit occurs nearby"
    );

    for x in 15..=45 {
        assert_ne!(
            world.get_voxel(x, 20, 20),
            MAT_AIR,
            "Bridge deck voxel at x={x}, y=20, z=20 must remain solid"
        );
    }
}

#[test]
fn test_floating_mid_air_structure_detaches_without_false_boundary_anchors() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    for z in 10..30 {
        for x in 10..30 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_BEDROCK));
            }
        }
    }

    for x in 10..=30 {
        voxels.push((x as u32, 30, 20, MAT_STONE_DEEP));
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let carve_pos = Vec3::new(20.0 * VOXEL_SIZE, 30.0 * VOXEL_SIZE, 20.0 * VOXEL_SIZE);
    let event = world.carve_sphere_meters(carve_pos, 0.12, false, 0);

    assert!(
        !event.detached_islands.is_empty(),
        "Floating structure severed in mid-air must detach as islands"
    );
}

#[test]
fn test_bounded_carve_island_detection_short_circuits_on_depth() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    for z in 10..30 {
        for x in 10..30 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_BEDROCK));
            }
        }
    }

    for x in 10..=35 {
        voxels.push((x as u32, 20, 20, MAT_STONE_DEEP));
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let center_v = Vec3::new(9.5, 20.0, 20.0);
    let radius_v = 1.0;
    let _ = world.carve_crater(center_v * VOXEL_SIZE, radius_v * VOXEL_SIZE, false, 0, 0);

    // INVARIANT: max_depth_v = 5.0 must short-circuit and consider island grounded because rod reaches x = 35.
    let islands_bounded = detect_disconnected_islands_from_carve_bounded(
        &world, center_v, radius_v, 262_144, 5.0, 262_144,
    );
    assert_eq!(
        islands_bounded.len(),
        0,
        "Rod exceeding max_depth_v must short-circuit and remain static"
    );

    // INVARIANT: max_depth_v = 50.0 allows the full 26-voxel rod to detach.
    let islands_deep = detect_disconnected_islands_from_carve_bounded(
        &world, center_v, radius_v, 262_144, 50.0, 262_144,
    );
    assert_eq!(
        islands_deep.len(),
        1,
        "Rod within max_depth_v must detach as an island"
    );
    assert_eq!(islands_deep[0].len(), 26);
}

#[test]
fn test_bounded_carve_island_detection_short_circuits_on_voxel_count() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    for z in 10..30 {
        for x in 10..30 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_BEDROCK));
            }
        }
    }

    for z in 15..18 {
        for y in 20..23 {
            for x in 15..18 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let center_v = Vec3::new(14.0, 21.0, 16.5);
    let radius_v = 1.2;
    let _ = world.carve_crater(center_v * VOXEL_SIZE, radius_v * VOXEL_SIZE, false, 0, 0);

    // INVARIANT: max_island_voxels = 10 must short-circuit on 27-voxel block.
    let islands_limited =
        detect_disconnected_islands_from_carve_bounded(&world, center_v, radius_v, 10, 50.0, 10);
    assert_eq!(
        islands_limited.len(),
        0,
        "Block exceeding max_island_voxels must short-circuit and remain static"
    );

    // INVARIANT: max_island_voxels = 100 allows the 27-voxel block to detach.
    let islands_large =
        detect_disconnected_islands_from_carve_bounded(&world, center_v, radius_v, 100, 50.0, 100);
    assert_eq!(
        islands_large.len(),
        1,
        "Block within max_island_voxels must detach as an island"
    );
    assert_eq!(islands_large[0].len(), 27);
}

#[test]
fn test_bounded_carve_island_detection_short_circuits_on_surface_voxel_count() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    for z in 10..30 {
        for x in 10..30 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_BEDROCK));
            }
        }
    }

    for z in 15..18 {
        for y in 20..23 {
            for x in 15..18 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let center_v = Vec3::new(14.0, 21.0, 16.5);
    let radius_v = 1.2;
    let _ = world.carve_crater(center_v * VOXEL_SIZE, radius_v * VOXEL_SIZE, false, 0, 0);

    // INVARIANT: 3x3x3 cube has 26 surface voxels.
    // max_surface_voxels = 10 must short-circuit during surface traversal even if max_island_voxels is 100.
    let islands_limited =
        detect_disconnected_islands_from_carve_bounded(&world, center_v, radius_v, 100, 50.0, 10);
    assert_eq!(
        islands_limited.len(),
        0,
        "Surface traversal exceeding max_surface_voxels must short-circuit and remain static"
    );

    // INVARIANT: max_surface_voxels = 50 allows all 26 surface voxels to be explored and detach.
    let islands_large =
        detect_disconnected_islands_from_carve_bounded(&world, center_v, radius_v, 100, 50.0, 50);
    assert_eq!(
        islands_large.len(),
        1,
        "Surface traversal within max_surface_voxels must detach as an island"
    );
    assert_eq!(islands_large[0].len(), 27);
}

#[test]
fn test_small_islands_spawn_debris_and_clear_from_world() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    for z in 10..30 {
        for x in 10..30 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_BEDROCK));
            }
        }
    }

    for x in 20..=23 {
        voxels.push((x as u32, 20, 20, MAT_STONE_DEEP));
    }

    for x in 20..=29 {
        voxels.push((x as u32, 30, 20, MAT_STONE_DEEP));
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let center_small = Vec3::new(19.0 * VOXEL_SIZE, 20.0 * VOXEL_SIZE, 20.0 * VOXEL_SIZE);
    let event_small = world.carve_sphere_meters_with_limits(
        center_small,
        0.8 * VOXEL_SIZE,
        false,
        64,
        1000,
        20.0,
        1000,
    );

    // INVARIANT: Small island (4 voxels <= 8) must spawn debris, clear voxels, and not appear in detached_islands.
    assert!(
        event_small.detached_islands.is_empty(),
        "Small island (<= 8 voxels) must not be placed in detached_islands"
    );
    assert!(
        !event_small.spawned_debris.is_empty(),
        "Small island must spawn debris particles"
    );
    assert_eq!(world.get_voxel(20, 20, 20), MAT_AIR);
    assert_eq!(world.get_voxel(21, 20, 20), MAT_AIR);
    assert_eq!(world.get_voxel(22, 20, 20), MAT_AIR);
    assert_eq!(world.get_voxel(23, 20, 20), MAT_AIR);

    let center_large = Vec3::new(19.0 * VOXEL_SIZE, 30.0 * VOXEL_SIZE, 20.0 * VOXEL_SIZE);
    let event_large = world.carve_sphere_meters_with_limits(
        center_large,
        0.8 * VOXEL_SIZE,
        false,
        64,
        1000,
        20.0,
        1000,
    );

    // INVARIANT: Island with 10 voxels (> 8) must be placed in detached_islands.
    assert_eq!(
        event_large.detached_islands.len(),
        1,
        "Island with > 8 voxels must be placed in detached_islands"
    );
    assert_eq!(event_large.detached_islands[0].len(), 10);
    assert_eq!(world.get_voxel(20, 30, 20), MAT_AIR);
    assert_eq!(world.get_voxel(29, 30, 20), MAT_AIR);
}

#[test]
fn test_crater_scorch_rim_clears_single_voxel_floater() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // WHY: Anchor terrain bedrock floor at y <= 4 to define grounded anchor plane.
    for z in 10..30 {
        for x in 10..30 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_BEDROCK));
            }
        }
    }

    // WHY: Place voxel inside crater and floater voxel on scorch rim at d = 3.0 ≈ radius_v + 0.8.
    voxels.push((20, 20, 22, MAT_STONE_DEEP));
    voxels.push((20, 20, 23, MAT_STONE_DEEP));

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let center = Vec3::new(20.5 * VOXEL_SIZE, 20.5 * VOXEL_SIZE, 20.5 * VOXEL_SIZE);
    let radius = 2.2 * VOXEL_SIZE;

    let event = world.carve_sphere_meters_with_limits(center, radius, true, 64, 1000, 20.0, 1000);

    // INVARIANT: Single remaining voxel floater (1 voxel) on scorch rim must be cleared and spawned as debris.
    assert_eq!(world.get_voxel(20, 20, 22), MAT_AIR);
    assert_eq!(world.get_voxel(20, 20, 23), MAT_AIR);
    assert!(
        event.destroyed_voxels >= 2,
        "Both the crater voxel and the floater rim voxel must be counted as destroyed"
    );
    assert!(
        !event.spawned_debris.is_empty(),
        "Debris particle must be spawned for the destroyed floater"
    );
    assert!(
        event.detached_islands.is_empty(),
        "Small island (1 voxel <= 8) must not be placed in detached_islands"
    );
}

#[test]
fn test_carve_crater_clears_rim_crumbs_even_when_island_detachment_disabled() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // WHY: Anchor terrain bedrock floor at y <= 4 to define grounded anchor plane.
    for z in 10..30 {
        for x in 10..30 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_BEDROCK));
            }
        }
    }

    // WHY: Place voxel inside crater and floater voxel on scorch rim at d = 3.0 ≈ radius_v + 0.8.
    voxels.push((20, 20, 22, MAT_STONE_DEEP));
    voxels.push((20, 20, 23, MAT_STONE_DEEP));

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let center = Vec3::new(20.5 * VOXEL_SIZE, 20.5 * VOXEL_SIZE, 20.5 * VOXEL_SIZE);
    let radius = 2.2 * VOXEL_SIZE;

    let event = world.carve_sphere_meters_with_limits(center, radius, true, 64, 0, 0.0, 0);

    // INVARIANT: 1-8 voxel crumbs must still be swept into debris even when structural island detachment is disabled.
    assert_eq!(world.get_voxel(20, 20, 22), MAT_AIR);
    assert_eq!(world.get_voxel(20, 20, 23), MAT_AIR);
    assert!(
        !event.spawned_debris.is_empty(),
        "Crumb floater must spawn debris"
    );
    assert!(event.detached_islands.is_empty());
}

#[test]
fn test_extract_brick_surface_mask_fully_enclosed_solid() {
    let solid_mask = u64::MAX;
    let surface = extract_brick_surface_mask(
        solid_mask,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
    );
    assert_eq!(surface, 0);
}

#[test]
fn test_extract_brick_surface_mask_single_exposed_face() {
    let solid_mask = u64::MAX;
    let surface = extract_brick_surface_mask(
        solid_mask,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        0,
        u64::MAX,
        u64::MAX,
    );

    let slice_0 = bitwise::FACE_Y_MIN;
    let slice_1 = bitwise::FACE_Y_MIN << 4;
    let slice_2 = bitwise::FACE_Y_MIN << 8;
    let slice_3 = bitwise::FACE_Y_MAX;

    // INVARIANT: Underground slices (y=0, 1) are culled; top slice (y=3) and 26-neighborhood depth (y=2) are retained.
    assert_eq!(surface & slice_0, 0);
    assert_eq!(surface & slice_1, 0);
    assert_eq!(surface & slice_2, slice_2);
    assert_eq!(surface & slice_3, slice_3);
    assert_eq!(surface, slice_2 | slice_3);
}

#[test]
fn test_extract_brick_surface_mask_concave_crease() {
    let air_voxel = child_bit_mask(2, 2, 1);
    let solid_mask = !air_voxel;

    let surface = extract_brick_surface_mask(
        solid_mask,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
    );

    let v_111 = child_bit_mask(1, 1, 1);
    assert_ne!(
        surface & v_111,
        0,
        "Concave crease voxel at (1,1,1) with only diagonal air at (2,2,1) must be retained"
    );

    let v_000 = child_bit_mask(0, 0, 0);
    assert_eq!(
        surface & v_000,
        0,
        "Voxel at (0,0,0) at Chebyshev distance 2 from air must not be retained"
    );
}

#[test]
fn test_extract_brick_surface_mask_isolated_voxel() {
    let isolated = child_bit_mask(2, 2, 2);
    let surface_all_air = extract_brick_surface_mask(isolated, 0, 0, 0, 0, 0, 0);
    assert_eq!(surface_all_air, isolated);

    let surface_solid_neighbors = extract_brick_surface_mask(
        isolated,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
    );
    assert_eq!(surface_solid_neighbors, isolated);
}

#[test]
fn test_surface_shell_flood_fill_recovers_interior_rock() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // WHY: Anchor terrain bedrock floor at y <= 4 to define grounded anchor plane.
    for z in 10..30 {
        for x in 10..30 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_BEDROCK));
            }
        }
    }

    // WHY: Place a solid 3x3x3 cube (27 voxels: 26 surface, 1 interior) floating at y=20.
    for z in 15..18 {
        for y in 20..23 {
            for x in 15..18 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    let center_v = Vec3::new(14.0, 21.0, 16.5);
    let radius_v = 1.2;
    let _ = world.carve_crater(center_v * VOXEL_SIZE, radius_v * VOXEL_SIZE, false, 0, 0);

    // INVARIANT: Surface shell has 26 voxels <= 26, but interior fill adds the 27th voxel > 26.
    // Deferred 3D interior fill must short-circuit on max_island_voxels and classify component as grounded.
    let islands_limited =
        detect_disconnected_islands_from_carve_bounded(&world, center_v, radius_v, 26, 50.0, 50);
    assert_eq!(
        islands_limited.len(),
        0,
        "Solid interior exceeding max_island_voxels during deferred fill must not detach"
    );

    // INVARIANT: When max_island_voxels allows 27 voxels, all 26 surface voxels and the interior voxel detach.
    let islands_full =
        detect_disconnected_islands_from_carve_bounded(&world, center_v, radius_v, 50, 50.0, 50);
    assert_eq!(
        islands_full.len(),
        1,
        "Closed surface shell must detach and include enclosed interior solid voxels"
    );
    assert_eq!(islands_full[0].len(), 27);
}

#[test]
fn test_concave_crease_preserves_grounded_pillar_on_elevated_platform() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // WHY: Anchor terrain bedrock floor at y <= 4 across x in 5..35, z in 5..35.
    for z in 5..35 {
        for x in 5..35 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_BEDROCK));
            }
        }
    }

    // WHY: Anchor platform to bedrock at four corners from y = 5 up to y = 9.
    for y in 5..=9 {
        for z in 5..8 {
            for x in 5..8 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
            for x in 32..35 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
        for z in 32..35 {
            for x in 5..8 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
            for x in 32..35 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    // WHY: Elevated horizontal solid stone platform at y in 10..=12 extending across x in 5..35, z in 5..35.
    for y in 10..=12 {
        for z in 5..35 {
            for x in 5..35 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    // WHY: Solid vertical pillar standing in center of platform (x in 19..21, z in 19..21) rising from y=13 to y=25.
    for y in 13..=25 {
        for z in 19..21 {
            for x in 19..21 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    // WHY: Carve side notch into pillar at y = 18.
    let center_v = Vec3::new(18.5, 18.0, 20.0);
    let radius_v = 1.2;
    let _ = world.carve_crater(center_v * VOXEL_SIZE, radius_v * VOXEL_SIZE, false, 0, 0);

    // INVARIANT: Concave crease theorem guarantees separable 26-neighborhood air dilation bridges
    // the orthogonal junction between pillar and platform, recognizing grounded support down to bedrock.
    let islands = detect_disconnected_islands_from_carve(&world, center_v, radius_v, 262_144);
    assert_eq!(
        islands.len(),
        0,
        "Grounded pillar standing on elevated platform must not detach as an island"
    );
}

#[test]
fn test_hollow_floating_boulder_detaches_with_interior_cavity_intact() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // WHY: Anchor bedrock floor at y <= 4 to define grounded anchor plane.
    for z in 10..40 {
        for x in 10..40 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_BEDROCK));
            }
        }
    }

    // WHY: Floating 8x8x8 hollow cube of stone at x in 20..28, y in 20..28, z in 20..28 with 4x4x4 interior cavity.
    let mut expected_solid_count = 0usize;
    for z in 20..28 {
        for y in 20..28 {
            for x in 20..28 {
                let is_cavity =
                    (22..26).contains(&x) && (22..26).contains(&y) && (22..26).contains(&z);
                if !is_cavity {
                    voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
                    expected_solid_count += 1;
                }
            }
        }
    }
    assert_eq!(expected_solid_count, 448);

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    // WHY: Carve small crater sphere adjacent to outer surface of hollow cube at x = 19.0.
    let center_v = Vec3::new(19.0, 24.0, 24.0);
    let radius_v = 1.2;
    let _ = world.carve_crater(center_v * VOXEL_SIZE, radius_v * VOXEL_SIZE, false, 0, 0);

    let islands = detect_disconnected_islands_from_carve(&world, center_v, radius_v, 262_144);

    // INVARIANT: Exactly 1 floating island must detach.
    assert_eq!(
        islands.len(),
        1,
        "Floating hollow boulder must detach as exactly 1 island"
    );

    // INVARIANT: Island contains exact solid voxels (448) and does NOT fill the internal 64-voxel cavity with stone.
    assert_eq!(
        islands[0].len(),
        448,
        "Detached island must contain exactly the 448 solid stone voxels"
    );

    let island_voxels: std::collections::HashSet<IVec3> =
        islands[0].iter().map(|&(pos, _)| pos).collect();

    for z in 22..26 {
        for y in 22..26 {
            for x in 22..26 {
                let cavity_coord = IVec3::new(x, y, z);
                assert!(
                    !island_voxels.contains(&cavity_coord),
                    "Interior cavity coord {cavity_coord:?} must not be filled into island voxel list"
                );
            }
        }
    }
}

#[test]
fn test_solid_mountain_carve_skips_interior_volume() {
    let palette = MaterialPalette::default();
    let mut voxels = Vec::new();

    // WHY: Large solid stone mountain block anchored to bedrock at y <= 4.
    for z in 0..32 {
        for x in 0..32 {
            for y in 0..=4 {
                voxels.push((x as u32, y as u32, z as u32, MAT_BEDROCK));
            }
            for y in 5..32 {
                voxels.push((x as u32, y as u32, z as u32, MAT_STONE_DEEP));
            }
        }
    }

    let chunk = ChunkTree64::from_sparse_voxels(&voxels, &palette);
    let mut world = WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette);

    // WHY: Carve small crater near top surface at y = 28.
    let center_v = Vec3::new(16.0, 28.0, 16.0);
    let radius_v = 4.0;
    let _ = world.carve_crater(center_v * VOXEL_SIZE, radius_v * VOXEL_SIZE, false, 0, 0);

    // INVARIANT: Surface shell flood fill traverses only the 2D boundary surface to bedrock,
    // skipping the massive 3D interior volume of the mountain and returning 0 islands.
    let islands = detect_disconnected_islands_from_carve(&world, center_v, radius_v, 262_144);
    assert_eq!(
        islands.len(),
        0,
        "Solid grounded mountain must not produce disconnected islands when carved"
    );
}
