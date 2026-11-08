//! Integration tests for chunk tree voxel placement, cuboid, and sphere edits.

use arcadia_core::*;
use glam::Vec3;

#[test]
fn test_in_place_set_voxels_air_to_solid() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();
    assert_eq!(tree.solid_voxel_count, 0);

    let voxels = vec![
        (0, 0, 0, MAT_CRYSTAL_CYAN),
        (0, 0, 1, MAT_CRYSTAL_CYAN),
        (0, 0, 2, MAT_CRYSTAL_CYAN),
    ];
    let added = tree.set_voxels_local(&voxels, &palette);
    assert_eq!(added, 3);
    assert_eq!(tree.solid_voxel_count, 3);

    assert_eq!(tree.get_voxel(0, 0, 0), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(0, 0, 1), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(0, 0, 2), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(0, 0, 3), MAT_AIR);
}

#[test]
fn test_in_place_overwrite_solid_and_clear_to_air() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();

    let voxels = vec![(10, 10, 10, MAT_CRYSTAL_CYAN), (10, 10, 11, MAT_STONE_DEEP)];
    tree.set_voxels_local(&voxels, &palette);
    assert_eq!(tree.solid_voxel_count, 2);
    assert_eq!(tree.get_voxel(10, 10, 10), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(10, 10, 11), MAT_STONE_DEEP);

    // Overwrite existing solid with different solid:
    let overwrite = vec![(10, 10, 10, MAT_STONE_DEEP)];
    let added_overwrite = tree.set_voxels_local(&overwrite, &palette);
    assert_eq!(added_overwrite, 0);
    assert_eq!(tree.solid_voxel_count, 2);
    assert_eq!(tree.get_voxel(10, 10, 10), MAT_STONE_DEEP);

    // Clear solid to air:
    let clear = vec![(10, 10, 10, MAT_AIR)];
    let added_clear = tree.set_voxels_local(&clear, &palette);
    assert_eq!(added_clear, 0);
    assert_eq!(tree.solid_voxel_count, 1);
    assert_eq!(tree.get_voxel(10, 10, 10), MAT_AIR);
    assert_eq!(tree.get_voxel(10, 10, 11), MAT_STONE_DEEP);

    // Clear the second voxel to air:
    let clear_second = vec![(10, 10, 11, MAT_AIR)];
    tree.set_voxels_local(&clear_second, &palette);
    assert_eq!(tree.solid_voxel_count, 0);
    assert!(tree.is_empty());
    assert_eq!(tree.get_voxel(10, 10, 11), MAT_AIR);
}

#[test]
fn test_in_place_across_multiple_bricks() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();

    let voxels = vec![
        (0, 0, 0, MAT_CRYSTAL_CYAN),
        (4, 0, 0, MAT_STONE_DEEP),
        (100, 150, 200, MAT_CRYSTAL_CYAN),
        (255, 255, 255, MAT_STONE_DEEP),
    ];
    let added = tree.set_voxels_local(&voxels, &palette);
    assert_eq!(added, 4);
    assert_eq!(tree.solid_voxel_count, 4);

    assert_eq!(tree.get_voxel(0, 0, 0), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(4, 0, 0), MAT_STONE_DEEP);
    assert_eq!(tree.get_voxel(100, 150, 200), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(255, 255, 255), MAT_STONE_DEEP);

    // Verify solid masks for these bricks:
    assert_ne!(tree.get_brick_solid_mask(0, 0, 0), 0);
    assert_ne!(tree.get_brick_solid_mask(1, 0, 0), 0);
    assert_ne!(tree.get_brick_solid_mask(25, 37, 50), 0);
    assert_ne!(tree.get_brick_solid_mask(63, 63, 63), 0);
    assert_eq!(tree.get_brick_solid_mask(2, 0, 0), 0);
}

#[test]
fn test_setting_air_in_empty_chunk_does_not_allocate() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();
    let initial_nodes = tree.nodes.len();

    let clear = vec![(50, 50, 50, MAT_AIR), (100, 100, 100, MAT_AIR)];
    let added = tree.set_voxels_local(&clear, &palette);
    assert_eq!(added, 0);
    assert_eq!(tree.solid_voxel_count, 0);
    assert_eq!(tree.nodes.len(), initial_nodes);
    assert!(tree.is_empty());
}

#[test]
fn test_in_place_mixed_brick_transition() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();

    // Step 1: Set 3 uniform voxels in a single leaf (0, 0, 0..2)
    let uni_voxels = vec![
        (0, 0, 0, MAT_CRYSTAL_CYAN),
        (0, 0, 1, MAT_CRYSTAL_CYAN),
        (0, 0, 2, MAT_CRYSTAL_CYAN),
    ];
    tree.set_voxels_local(&uni_voxels, &palette);
    assert_eq!(tree.bricks.len(), 0); // Still uniform, no bricks allocated

    // Step 2: Add a 4th voxel with a DIFFERENT material in the same leaf
    let mixed_voxel = vec![(0, 0, 3, MAT_STONE_DEEP)];
    tree.set_voxels_local(&mixed_voxel, &palette);
    assert_eq!(tree.bricks.len(), 1); // Brick allocated
    assert_eq!(tree.solid_voxel_count, 4);

    assert_eq!(tree.get_voxel(0, 0, 0), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(0, 0, 1), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(0, 0, 2), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(0, 0, 3), MAT_STONE_DEEP);

    // Step 3: Overwrite an existing voxel in the leaf (0, 0, 0) from CYAN to STONE
    let overwrite = vec![(0, 0, 0, MAT_STONE_DEEP)];
    let added = tree.set_voxels_local(&overwrite, &palette);
    assert_eq!(added, 0);
    assert_eq!(tree.bricks.len(), 1); // No new brick allocated (mutated in-place!)
    assert_eq!(tree.solid_voxel_count, 4);
    assert_eq!(tree.get_voxel(0, 0, 0), MAT_STONE_DEEP);
}

#[test]
fn test_in_place_uncollapse_and_sculpt() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::from_generator(
        &palette,
        |_ox, _oy, _oz, _size| Some(MAT_STONE_DEEP),
        |_x, _y, _z| MAT_STONE_DEEP,
    );
    let initial_solid = tree.solid_voxel_count;
    assert!(initial_solid > 0);
    assert!(tree.nodes[0].is_leaf());

    // Mutate a single voxel in the center of the collapsed chunk:
    let added = tree.set_voxels_local(&[(128, 128, 128, MAT_CRYSTAL_CYAN)], &palette);
    assert_eq!(added, 0); // Overwrite solid with solid
    assert_eq!(tree.solid_voxel_count, initial_solid);
    assert_eq!(tree.get_voxel(128, 128, 128), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(128, 128, 129), MAT_STONE_DEEP);

    // Clear a voxel to air:
    let clear = tree.set_voxels_local(&[(128, 128, 129, MAT_AIR)], &palette);
    assert_eq!(clear, 0);
    assert_eq!(tree.solid_voxel_count, initial_solid - 1);
    assert_eq!(tree.get_voxel(128, 128, 129), MAT_AIR);
    assert_eq!(tree.get_voxel(128, 128, 130), MAT_STONE_DEEP);
}

#[test]
fn test_in_place_set_voxel_single() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();

    let became_solid = tree.set_voxel(10, 20, 30, MAT_CRYSTAL_CYAN, &palette);
    assert!(became_solid);
    assert_eq!(tree.solid_voxel_count, 1);
    assert_eq!(tree.get_voxel(10, 20, 30), MAT_CRYSTAL_CYAN);

    let became_solid_again = tree.set_voxel(10, 20, 30, MAT_STONE_DEEP, &palette);
    assert!(!became_solid_again);
    assert_eq!(tree.solid_voxel_count, 1);
    assert_eq!(tree.get_voxel(10, 20, 30), MAT_STONE_DEEP);

    let became_solid_clear = tree.set_voxel(10, 20, 30, MAT_AIR, &palette);
    assert!(!became_solid_clear);
    assert_eq!(tree.solid_voxel_count, 0);
    assert_eq!(tree.get_voxel(10, 20, 30), MAT_AIR);
    assert!(tree.is_empty());
}

#[test]
fn test_chunk_edit_report_tracking() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();

    // 1. Empty edit produces no dirty ranges:
    let empty_rep = tree.set_voxels_local_tracked(&[], &palette);
    assert!(!empty_rep.has_dirty_nodes());
    assert!(!empty_rep.has_dirty_bricks());
    assert!(!empty_rep.had_structural_growth);

    // 2. Setting voxels in empty chunk causes structural growth:
    let rep1 = tree.set_voxels_local_tracked(
        &[(0, 0, 0, MAT_CRYSTAL_CYAN), (0, 0, 1, MAT_CRYSTAL_CYAN)],
        &palette,
    );
    assert!(rep1.had_structural_growth);
    assert!(rep1.has_dirty_nodes());
    assert_eq!(rep1.min_dirty_node, 0);
    assert!(rep1.max_dirty_node >= 3);
    assert!(!rep1.has_dirty_bricks()); // Uniform leaf, 0 bricks
    assert_eq!(rep1.added_voxels, 2);

    // 3. Mutating in-place without branch growth:
    let rep2 = tree.set_voxels_local_tracked(&[(0, 0, 2, MAT_CRYSTAL_CYAN)], &palette);
    assert!(!rep2.had_structural_growth);
    assert!(rep2.has_dirty_nodes());
    assert!(!rep2.has_dirty_bricks()); // Still uniform
    assert_eq!(rep2.added_voxels, 1);

    // 4. Adding second material to same leaf allocates a brick (structural growth):
    let rep3 = tree.set_voxels_local_tracked(&[(0, 0, 3, MAT_STONE_DEEP)], &palette);
    assert!(rep3.had_structural_growth);
    assert!(rep3.has_dirty_nodes());
    assert!(rep3.has_dirty_bricks());
    assert_eq!(rep3.min_dirty_brick, 0);
    assert_eq!(rep3.max_dirty_brick, 0);
    assert_eq!(rep3.added_voxels, 1);

    // 5. Overwriting existing brick voxel in-place:
    let rep4 = tree.set_voxels_local_tracked(&[(0, 0, 3, MAT_CRYSTAL_CYAN)], &palette);
    assert!(!rep4.had_structural_growth);
    assert!(rep4.has_dirty_nodes());
    assert!(rep4.has_dirty_bricks());
    assert_eq!(rep4.min_dirty_brick, 0);
    assert_eq!(rep4.max_dirty_brick, 0);
    assert_eq!(rep4.added_voxels, 0);
}

#[test]
fn test_in_place_remove_child_no_node_growth() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();

    let voxels = vec![
        (0, 0, 0, MAT_CRYSTAL_CYAN),
        (0, 0, 1, MAT_CRYSTAL_CYAN),
        (4, 0, 0, MAT_STONE_DEEP),
    ];
    tree.set_voxels_local(&voxels, &palette);
    let nodes_len_before = tree.nodes.len();

    // WHY: Clearing one voxel within a multi-voxel leaf should not grow node storage.
    let rep1 = tree.set_voxels_local_tracked(&[(0, 0, 0, MAT_AIR)], &palette);
    assert!(!rep1.had_structural_growth);
    assert_eq!(tree.nodes.len(), nodes_len_before);
    assert_eq!(tree.get_voxel(0, 0, 0), MAT_AIR);
    assert_eq!(tree.get_voxel(0, 0, 1), MAT_CRYSTAL_CYAN);

    // WHY: Clearing the last voxel in brick (4, 0, 0) removes the leaf child without allocating new nodes.
    let rep2 = tree.set_voxels_local_tracked(&[(4, 0, 0, MAT_AIR)], &palette);
    assert!(!rep2.had_structural_growth);
    assert_eq!(tree.nodes.len(), nodes_len_before);
    assert_eq!(tree.get_voxel(4, 0, 0), MAT_AIR);
    assert_eq!(tree.get_voxel(0, 0, 1), MAT_CRYSTAL_CYAN);
}

#[test]
fn test_compact_strips_orphaned_nodes_and_preserves_voxels() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();

    let initial_voxels = vec![
        (0, 0, 0, MAT_CRYSTAL_CYAN),
        (4, 4, 4, MAT_STONE_DEEP),
        (100, 100, 100, MAT_CRYSTAL_CYAN),
        (200, 200, 200, MAT_STONE_DEEP),
    ];
    tree.set_voxels_local(&initial_voxels, &palette);

    // WHY: Growing extra branches then removing leaves leaves orphaned node ranges.
    tree.set_voxel(8, 8, 8, MAT_CRYSTAL_CYAN, &palette);
    tree.set_voxel(8, 8, 8, MAT_AIR, &palette);

    let nodes_len_before = tree.nodes.len();
    let solid_before = tree.solid_voxel_count;

    tree.compact();

    assert_eq!(tree.solid_voxel_count, solid_before);
    assert!(tree.nodes.len() <= nodes_len_before);

    assert_eq!(tree.get_voxel(0, 0, 0), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(4, 4, 4), MAT_STONE_DEEP);
    assert_eq!(tree.get_voxel(100, 100, 100), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(200, 200, 200), MAT_STONE_DEEP);
    assert_eq!(tree.get_voxel(8, 8, 8), MAT_AIR);
    assert_eq!(tree.get_voxel(50, 50, 50), MAT_AIR);

    // WHY: Verify compacting an empty tree produces a single empty root node.
    let mut empty_tree = ChunkTree64::default();
    empty_tree.compact();
    assert_eq!(empty_tree.nodes.len(), 1);
    assert!(empty_tree.bricks.is_empty());
    assert!(empty_tree.is_empty());
}

#[test]
fn test_carve_sphere_destructive() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::from_generator(
        &palette,
        |_ox, _oy, _oz, _size| Some(MAT_STONE_DEEP),
        |_x, _y, _z| MAT_STONE_DEEP,
    );

    let initial_solid = tree.solid_voxel_count;
    assert!(initial_solid > 0);

    let center = Vec3::new(64.0, 64.0, 64.0);
    let radius = 6.0;
    let mut debris_count = 0u32;
    let destroyed = tree.carve_sphere(center, radius, true, &palette, |_x, _y, _z, _mat| {
        debris_count += 1;
    });

    assert!(destroyed > 500);
    assert_eq!(tree.solid_voxel_count, initial_solid - destroyed);
    assert_eq!(tree.get_voxel(64, 64, 64), MAT_AIR);
    assert_eq!(tree.get_voxel(65, 64, 64), MAT_AIR);
    assert!(debris_count > 0);

    // WHY: Outer scorch rim can produce ember glow or scorched rock at boundaries.
    let mut found_scorch_or_ember = false;
    for dz in -8..=8 {
        for dy in -8..=8 {
            for dx in -8..=8 {
                let vx = (64 + dx) as u32;
                let vy = (64 + dy) as u32;
                let vz = (64 + dz) as u32;
                let mat = tree.get_voxel(vx, vy, vz);
                if mat == MAT_SCORCHED_ROCK || mat == MAT_EMBER_GLOW {
                    found_scorch_or_ember = true;
                    break;
                }
            }
        }
    }
    assert!(found_scorch_or_ember);

    // Out of bounds carving produces 0 destroyed:
    let out_destroyed = tree.carve_sphere(
        Vec3::new(500.0, 500.0, 500.0),
        5.0,
        false,
        &palette,
        |_x, _y, _z, _m| {},
    );
    assert_eq!(out_destroyed, 0);
}

#[test]
fn test_place_sphere_additive() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();
    assert_eq!(tree.solid_voxel_count, 0);

    let center = Vec3::new(32.0, 32.0, 32.0);
    let radius = 4.0;
    tree.place_sphere(center, radius, MAT_CRYSTAL_CYAN, &palette);

    assert!(tree.solid_voxel_count > 200);
    assert_eq!(tree.get_voxel(32, 32, 32), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(32, 33, 32), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(32, 50, 32), MAT_AIR);
}

#[test]
fn test_clear_voxels_local() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::from_generator(
        &palette,
        |_ox, _oy, _oz, _size| Some(MAT_STONE_DEEP),
        |_x, _y, _z| MAT_STONE_DEEP,
    );
    let initial_solid = tree.solid_voxel_count;

    let to_clear = vec![
        (10, 10, 10),
        (10, 10, 11),
        (10, 10, 12),
        (10, 10, 13),
        (20, 20, 20),
    ];
    let removed = tree.clear_voxels_local(&to_clear, &palette);
    assert_eq!(removed, 5);
    assert_eq!(tree.solid_voxel_count, initial_solid - 5);

    for &(x, y, z) in &to_clear {
        assert_eq!(tree.get_voxel(x, y, z), MAT_AIR);
    }

    // Clearing already cleared voxels returns 0:
    let removed_again = tree.clear_voxels_local(&to_clear, &palette);
    assert_eq!(removed_again, 0);
}

#[test]
fn test_set_voxels_local() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();
    assert_eq!(tree.solid_voxel_count, 0);

    let to_set = vec![
        (5, 5, 5, MAT_CRYSTAL_CYAN),
        (5, 5, 6, MAT_CRYSTAL_CYAN),
        (5, 5, 7, MAT_STONE_DEEP),
    ];
    let added = tree.set_voxels_local(&to_set, &palette);
    assert_eq!(added, 3);
    assert_eq!(tree.solid_voxel_count, 3);

    assert_eq!(tree.get_voxel(5, 5, 5), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(5, 5, 6), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(5, 5, 7), MAT_STONE_DEEP);

    // Overwriting existing voxel updates material without increasing solid count:
    let overwrite = vec![(5, 5, 5, MAT_STONE_DEEP)];
    let added_overwrite = tree.set_voxels_local(&overwrite, &palette);
    assert_eq!(added_overwrite, 0);
    assert_eq!(tree.solid_voxel_count, 3);
    assert_eq!(tree.get_voxel(5, 5, 5), MAT_STONE_DEEP);
}

#[test]
fn test_empty_chunk_operations() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();

    assert_eq!(
        tree.carve_sphere(
            Vec3::new(10.0, 10.0, 10.0),
            5.0,
            true,
            &palette,
            |_x, _y, _z, _m| {}
        ),
        0
    );
    assert_eq!(tree.clear_voxels_local(&[(10, 10, 10)], &palette), 0);
    assert_eq!(tree.set_voxels_local(&[], &palette), 0);
    assert_eq!(tree.clear_aabb((0, 0, 0), (10, 10, 10), &palette), 0);
    tree.place_cuboid((0, 0, 0), (10, 10, 10), MAT_AIR, &palette);
    assert_eq!(
        tree.place_cuboid_tracked((0, 0, 0), (10, 10, 10), MAT_AIR, &palette)
            .added_voxels,
        0
    );
}

#[test]
fn test_carve_sphere_tracked_zero_repack() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::from_generator(
        &palette,
        |_ox, _oy, _oz, _size| Some(MAT_STONE_DEEP),
        |_x, _y, _z| MAT_STONE_DEEP,
    );

    let center = Vec3::new(64.0, 64.0, 64.0);
    let radius = 5.0;
    let (destroyed, report) =
        tree.carve_sphere_tracked(center, radius, false, &palette, |_x, _y, _z, _m| {});

    assert!(destroyed > 300);
    // WHY: First carve on a fully-collapsed solid tree uncollapses ancestors into active nodes:
    assert!(report.had_structural_growth);
    assert!(report.has_dirty_nodes());
    assert_eq!(tree.get_voxel(64, 64, 64), MAT_AIR);

    // Subsequent carve in the already uncollapsed region mutates bricks in-place without structural growth:
    let (destroyed2, report2) =
        tree.carve_sphere_tracked(center, radius + 0.5, false, &palette, |_x, _y, _z, _m| {});
    assert!(destroyed2 > 0);
    assert!(!report2.had_structural_growth);
    assert!(report2.has_dirty_nodes());
}

#[test]
fn test_clear_aabb_and_place_cuboid() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::default();

    // Place a 4x4x4 cuboid:
    let report = tree.place_cuboid_tracked((10, 10, 10), (13, 13, 13), MAT_CRYSTAL_CYAN, &palette);
    assert_eq!(report.added_voxels, 64);
    assert_eq!(tree.solid_voxel_count, 64);
    assert_eq!(tree.get_voxel(10, 10, 10), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(13, 13, 13), MAT_CRYSTAL_CYAN);
    assert_eq!(tree.get_voxel(14, 14, 14), MAT_AIR);

    // Clear a 2x2x2 sub-region with clear_aabb_tracked:
    let (removed, clear_report) = tree.clear_aabb_tracked((10, 10, 10), (11, 11, 11), &palette);
    assert_eq!(removed, 8);
    assert_eq!(tree.solid_voxel_count, 56);
    assert!(clear_report.has_dirty_nodes());
    assert_eq!(tree.get_voxel(10, 10, 10), MAT_AIR);
    assert_eq!(tree.get_voxel(12, 12, 12), MAT_CRYSTAL_CYAN);
}

#[test]
fn test_carve_crater_clears_parent_occupancy_masks() {
    let palette = MaterialPalette::default();
    let mut tree = ChunkTree64::from_generator(
        &palette,
        |_ox, _oy, _oz, _size| Some(MAT_STONE_DEEP),
        |_x, _y, _z| MAT_STONE_DEEP,
    );
    let initial_solid = tree.solid_voxel_count;
    assert!(initial_solid > 0);

    let center = Vec3::new(32.0, 32.0, 32.0);
    let radius = 10.0;
    let (destroyed, report) =
        tree.carve_sphere_tracked(center, radius, false, &palette, |_x, _y, _z, _m| {});

    assert!(destroyed > 0);
    assert_eq!(tree.solid_voxel_count, initial_solid - destroyed);
    assert!(report.has_dirty_nodes());

    // Verify center crater voxels are MAT_AIR:
    assert_eq!(tree.get_voxel(32, 32, 32), MAT_AIR);
    assert_eq!(tree.get_voxel(30, 30, 30), MAT_AIR);
    assert_eq!(tree.get_voxel(34, 34, 34), MAT_AIR);

    // Verify outside the crater is MAT_STONE_DEEP:
    assert_eq!(tree.get_voxel(0, 0, 0), MAT_STONE_DEEP);
    assert_eq!(tree.get_voxel(32, 32, 50), MAT_STONE_DEEP);
    assert_eq!(tree.get_voxel(50, 32, 32), MAT_STONE_DEEP);

    // Check hollowed-out brick (8, 8, 8) in the center of the crater:
    // (bx=8, by=8, bz=8) corresponds to voxels [32..36, 32..36, 32..36].
    // All voxels are within distance sqrt(3.5^2 * 3) = 6.06 < 10.0 from (32.0, 32.0, 32.0).
    let bx = 8u32;
    let by = 8u32;
    let bz = 8u32;
    let b_0 = child_bit_index((bx >> 4) & 3, (by >> 4) & 3, (bz >> 4) & 3);
    assert!(tree.nodes[0].has_child(b_0));
    let l1 = tree.nodes[0].child_slot_offset(b_0) as usize;

    let b_1 = child_bit_index((bx >> 2) & 3, (by >> 2) & 3, (bz >> 2) & 3);
    assert!(tree.nodes[l1].has_child(b_1));
    let l2 = tree.nodes[l1].child_slot_offset(b_1) as usize;

    let b_2 = child_bit_index(bx & 3, by & 3, bz & 3);
    // Crucial invariant: The parent Level 2 node's bit for that brick MUST be 0!
    assert!(!tree.nodes[l2].has_child(b_2));

    // Also verify brick (7, 7, 7) covering [28..32, 28..32, 28..32]:
    let bx7 = 7u32;
    let by7 = 7u32;
    let bz7 = 7u32;
    let b_0_7 = child_bit_index((bx7 >> 4) & 3, (by7 >> 4) & 3, (bz7 >> 4) & 3);
    assert!(tree.nodes[0].has_child(b_0_7));
    let l1_7 = tree.nodes[0].child_slot_offset(b_0_7) as usize;

    let b_1_7 = child_bit_index((bx7 >> 2) & 3, (by7 >> 2) & 3, (bz7 >> 2) & 3);
    assert!(tree.nodes[l1_7].has_child(b_1_7));
    let l2_7 = tree.nodes[l1_7].child_slot_offset(b_1_7) as usize;

    let b_2_7 = child_bit_index(bx7 & 3, by7 & 3, bz7 & 3);
    assert!(!tree.nodes[l2_7].has_child(b_2_7));
}
