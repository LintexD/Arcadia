//! Intra-brick bitwise dilation, boundary face masks, and ALU flood fill.
//! Owns pure CPU register bit manipulation for 4x4x4 leaf bricks.

// Bitmasks for intra-brick 6-connected 64-bit bitwise dilation (`b = x | (y << 2) | (z << 4)`):
pub const NOT_X_MAX: u64 = 0x7777_7777_7777_7777; // x in 0..3 (excludes x == 3)
pub const NOT_X_MIN: u64 = 0xEEEE_EEEE_EEEE_EEEE; // x in 1..4 (excludes x == 0)
pub const NOT_Y_MAX: u64 = 0x0FFF_0FFF_0FFF_0FFF; // y in 0..3 (excludes y == 3)
pub const NOT_Y_MIN: u64 = 0xFFF0_FFF0_FFF0_FFF0; // y in 1..4 (excludes y == 0)
pub const NOT_Z_MAX: u64 = 0x0000_FFFF_FFFF_FFFF; // z in 0..3 (excludes z == 3)
pub const NOT_Z_MIN: u64 = 0xFFFF_FFFF_FFFF_0000; // z in 1..4 (excludes z == 0)

// Boundary face masks within a 4x4x4 brick:
pub const FACE_X_MIN: u64 = 0x1111_1111_1111_1111; // x == 0
pub const FACE_X_MAX: u64 = 0x8888_8888_8888_8888; // x == 3
pub const FACE_Y_MIN: u64 = 0x000F_000F_000F_000F; // y == 0
pub const FACE_Y_MAX: u64 = 0xF000_F000_F000_F000; // y == 3
pub const FACE_Z_MIN: u64 = 0x0000_0000_0000_FFFF; // z == 0
pub const FACE_Z_MAX: u64 = 0xFFFF_0000_0000_0000; // z == 3

/// Extracts the 64-bit surface occupancy mask for a 4x4x4 leaf brick.
///
/// A solid voxel is on the surface if it is within Chebyshev distance 1 (26-neighborhood)
/// of an air voxel (intra-brick empty space or adjacent neighbor face air).
///
/// Executes in pure ALU register bitwise operations in ~5 nanoseconds.
#[inline(always)]
pub fn extract_brick_surface_mask(
    solid_mask: u64,
    neg_x_solid: u64,
    pos_x_solid: u64,
    neg_y_solid: u64,
    pos_y_solid: u64,
    neg_z_solid: u64,
    pos_z_solid: u64,
) -> u64 {
    if solid_mask == 0 {
        return 0;
    }

    // WHY: Air exposed within the leaf brick or across the 6 adjacent neighbor faces.
    let intra_air = !solid_mask;
    let face_air = (((!neg_x_solid) & FACE_X_MAX) >> 3)
        | (((!pos_x_solid) & FACE_X_MIN) << 3)
        | (((!neg_y_solid) & FACE_Y_MAX) >> 12)
        | (((!pos_y_solid) & FACE_Y_MIN) << 12)
        | (((!neg_z_solid) & FACE_Z_MAX) >> 48)
        | (((!pos_z_solid) & FACE_Z_MIN) << 48);

    let total_air = intra_air | face_air;
    if total_air == 0 {
        return 0;
    }

    // WHY: Separable 1D dilations across X, Y, Z expand Chebyshev distance 1 (26-neighborhood).
    let air_x = total_air | ((total_air & NOT_X_MAX) << 1) | ((total_air & NOT_X_MIN) >> 1);

    let air_xy = air_x | ((air_x & NOT_Y_MAX) << 4) | ((air_x & NOT_Y_MIN) >> 4);

    let air_xyz = air_xy | ((air_xy & NOT_Z_MAX) << 16) | ((air_xy & NOT_Z_MIN) >> 16);

    solid_mask & air_xyz
}

/// Dilates `seed` by 1 step in all 6 cardinal directions inside a 4x4x4 brick.
#[inline(always)]
pub fn dilate_brick_step(seed: u64, solid_mask: u64) -> u64 {
    let expanded = seed
        | ((seed & NOT_X_MAX) << 1)
        | ((seed & NOT_X_MIN) >> 1)
        | ((seed & NOT_Y_MAX) << 4)
        | ((seed & NOT_Y_MIN) >> 4)
        | ((seed & NOT_Z_MAX) << 16)
        | ((seed & NOT_Z_MIN) >> 16);
    expanded & solid_mask
}

/// Flood-fills `seed` to convergence within a single 4x4x4 brick solid mask.
#[inline(always)]
pub fn flood_fill_brick_u64(mut seed: u64, solid_mask: u64) -> u64 {
    seed &= solid_mask;
    if seed == 0 {
        return 0;
    }
    loop {
        let next = dilate_brick_step(seed, solid_mask);
        if next == seed {
            return seed;
        }
        seed = next;
    }
}

/// Dilates `seed` in all 6 directions (+Y, -Y, +X, -X, +Z, -Z) for compressive support.
#[inline(always)]
pub fn dilate_support_brick_step(seed: u64, solid_mask: u64) -> u64 {
    let expanded = seed
        | ((seed & NOT_X_MAX) << 1)
        | ((seed & NOT_X_MIN) >> 1)
        | ((seed & NOT_Y_MAX) << 4) // +Y
        | ((seed & NOT_Y_MIN) >> 4) // -Y (downward support)
        | ((seed & NOT_Z_MAX) << 16)
        | ((seed & NOT_Z_MIN) >> 16);
    expanded & solid_mask
}

/// Flood-fills compressive support `seed` to convergence within a single 4x4x4 brick.
#[inline(always)]
pub fn flood_fill_support_brick_u64(mut seed: u64, solid_mask: u64) -> u64 {
    seed &= solid_mask;
    if seed == 0 {
        return 0;
    }
    loop {
        let next = dilate_support_brick_step(seed, solid_mask);
        if next == seed {
            return seed;
        }
        seed = next;
    }
}
