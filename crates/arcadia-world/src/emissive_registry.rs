//! CPU registry of emissive voxel clusters and the Vose alias table that drives ReSTIR DI light selection.
//! Owns CPU state only; `arcadia-render::di` uploads `sources()` / `alias()` to the GPU.

use std::collections::HashMap;

use arcadia_core::{
    unpack_child_index, MaterialId, MaterialPalette, CHUNK_DIM, MAT_AIR, MAT_WATER_CRYSTAL,
    VOXEL_SIZE,
};
use bytemuck::{Pod, Zeroable};
use glam::IVec3;

use crate::world_map::WorldMap;

/// Capacity of the GPU emissive pool and alias table (`shaders/common/di.wgsl` `DI_MAX_SOURCES`).
pub const MAX_EMISSIVE_SOURCES: usize = 16_384;
/// Edge of a fine cluster cell in voxels: one Level-2 Meso-Node (16 voxels = 1 m).
pub const EMISSIVE_CELL_VOXELS: i32 = 16;

/// Area of one voxel face in square meters.
const FACE_AREA: f64 = (1.0 / 16.0) * (1.0 / 16.0);

/// The six axis-aligned face directions of a voxel.
const FACE_DIRS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// One clustered emissive light as read by `regir_update.wgsl`, Pass 2 and Pass 3.
// GPU-SYNC: Mirrors `EmissiveSource` in `shaders/common/di.wgsl` (32 bytes, 16-byte aligned).
// `normal_packed` holds the flux-averaged exposed face normal as unorm10 x/y/z (`n * 0.5 + 0.5`, bits 0..29; magnitude encodes directional concentration eta).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct EmissiveSource {
    /// Flux-weighted centroid of the exposed emissive faces (meters).
    pub pos: [f32; 3],
    /// Bounding radius of the exposed faces around `pos` (meters); trims shadow rays.
    pub radius: f32,
    /// Radiant intensity scale `L_e * A_exposed` (RGB); irradiance is `intensity * cos / d^2`.
    pub intensity: [f32; 3],
    pub normal_packed: u32,
}

/// One Vose alias table entry (`shaders/common/di.wgsl` `AliasEntry`).
// GPU-SYNC: Mirrors `AliasEntry` in `shaders/common/di.wgsl` (8 bytes, 4-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct AliasEntry {
    /// Probability of keeping the drawn column `i` instead of jumping to `alias_idx`.
    pub prob: f32,
    pub alias_idx: u32,
}

/// Rec.709 luma shared by CPU flux weights and the GPU `luma()` helper.
pub fn emissive_luma(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

/// Returns true for materials that are tracked as ReSTIR DI lights.
// GPU-SYNC: Mirrors `di_is_emitter_material` in `shaders/common/di.wgsl`.
pub fn is_registered_emitter(palette: &MaterialPalette, mat: MaterialId) -> bool {
    palette.get(mat).is_emissive()
}

/// Additive statistics of the exposed emissive faces inside one cluster cell.
///
/// Every field is a plain sum or min/max so clusters merge exactly when the registry coarsens.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterAccum {
    pub cell: [i32; 3],
    flux_weight: f64,
    pos_sum: [f64; 3],
    normal_sum: [f64; 3],
    intensity: [f64; 3],
    bounds_min: [f64; 3],
    bounds_max: [f64; 3],
    faces: u32,
}

impl ClusterAccum {
    fn new(cell: [i32; 3]) -> Self {
        Self {
            cell,
            flux_weight: 0.0,
            pos_sum: [0.0; 3],
            normal_sum: [0.0; 3],
            intensity: [0.0; 3],
            bounds_min: [f64::INFINITY; 3],
            bounds_max: [f64::NEG_INFINITY; 3],
            faces: 0,
        }
    }

    fn add_face(&mut self, center: [f64; 3], dir: [f64; 3], emission: [f64; 3]) {
        let luma = emission[0] * 0.2126 + emission[1] * 0.7152 + emission[2] * 0.0722;
        let w = luma * FACE_AREA;
        self.flux_weight += w;
        for i in 0..3 {
            self.pos_sum[i] += w * center[i];
            self.normal_sum[i] += w * dir[i];
            self.intensity[i] += emission[i] * FACE_AREA;
            self.bounds_min[i] = self.bounds_min[i].min(center[i]);
            self.bounds_max[i] = self.bounds_max[i].max(center[i]);
        }
        self.faces += 1;
    }

    fn merge_into(&self, dst: &mut Self) {
        dst.flux_weight += self.flux_weight;
        for i in 0..3 {
            dst.pos_sum[i] += self.pos_sum[i];
            dst.normal_sum[i] += self.normal_sum[i];
            dst.intensity[i] += self.intensity[i];
            dst.bounds_min[i] = dst.bounds_min[i].min(self.bounds_min[i]);
            dst.bounds_max[i] = dst.bounds_max[i].max(self.bounds_max[i]);
        }
        dst.faces += self.faces;
    }

    /// Converts the accumulated statistics into a GPU light; `None` when no emissive face was recorded.
    pub fn finalize(&self) -> Option<EmissiveSource> {
        if self.faces == 0 || self.flux_weight <= 0.0 {
            return None;
        }
        let inv_w = 1.0 / self.flux_weight;
        let centroid = [
            self.pos_sum[0] * inv_w,
            self.pos_sum[1] * inv_w,
            self.pos_sum[2] * inv_w,
        ];

        // Average normal: magnitude encodes directional concentration eta in [0, 1].
        // Planar walls/slabs have length ~ 1.0 (directional).
        // Pillars/cubes/torches have opposing faces that cancel to ~ 0.0 (omnidirectional).
        let normal_avg = [
            (self.normal_sum[0] * inv_w).clamp(-1.0, 1.0),
            (self.normal_sum[1] * inv_w).clamp(-1.0, 1.0),
            (self.normal_sum[2] * inv_w).clamp(-1.0, 1.0),
        ];

        let mut r2 = 0.0f64;
        for i in 0..3 {
            let d = (centroid[i] - self.bounds_min[i])
                .abs()
                .max((self.bounds_max[i] - centroid[i]).abs());
            r2 += d * d;
        }
        // WHY: Half a voxel margin covers the face extent around each face center.
        let radius = r2.sqrt() + 0.5 * f64::from(VOXEL_SIZE);

        Some(EmissiveSource {
            pos: [centroid[0] as f32, centroid[1] as f32, centroid[2] as f32],
            radius: radius as f32,
            intensity: [
                self.intensity[0] as f32,
                self.intensity[1] as f32,
                self.intensity[2] as f32,
            ],
            normal_packed: pack_normal_unorm10(normal_avg),
        })
    }
}

/// Active emissive lights of the static world plus their flux-proportional alias table.
pub struct EmissiveRegistry {
    per_chunk: Vec<Vec<ClusterAccum>>,
    pending_direct: Vec<bool>,
    pending_neighbor: Vec<bool>,
    sources: Vec<EmissiveSource>,
    alias: Vec<AliasEntry>,
    total_flux: f32,
    generation: u32,
}

impl EmissiveRegistry {
    /// Creates a registry for `num_chunks` chunks with every chunk pending its initial scan.
    pub fn new(num_chunks: usize) -> Self {
        Self {
            per_chunk: vec![Vec::new(); num_chunks],
            pending_direct: vec![true; num_chunks],
            pending_neighbor: vec![false; num_chunks],
            sources: Vec::new(),
            alias: Vec::new(),
            total_flux: 0.0,
            generation: 1,
        }
    }

    /// Flags a chunk whose voxels changed; it is fully rescanned on the next refresh.
    pub fn mark_dirty(&mut self, chunk_idx: usize) {
        if let Some(flag) = self.pending_direct.get_mut(chunk_idx) {
            *flag = true;
        }
    }

    /// Flags a face-adjacent chunk of an edited chunk; rescanned only if it already holds emitters,
    /// because an edit next door can only change the exposure of existing emissive faces.
    pub fn mark_neighbor_dirty(&mut self, chunk_idx: usize) {
        if let Some(flag) = self.pending_neighbor.get_mut(chunk_idx) {
            *flag = true;
        }
    }

    /// Drains the pending flags and returns the chunk indices that need a rescan.
    pub fn take_pending(&mut self) -> Vec<usize> {
        let mut out = Vec::new();
        for idx in 0..self.per_chunk.len() {
            let direct = self.pending_direct[idx];
            let neighbor = self.pending_neighbor[idx] && !self.per_chunk[idx].is_empty();
            if direct || neighbor {
                out.push(idx);
            }
            self.pending_direct[idx] = false;
            self.pending_neighbor[idx] = false;
        }
        out
    }

    /// Stores rescanned chunk clusters; rebuilds the pool and alias table only if anything changed.
    ///
    /// Returns true when the light pool changed (the generation was bumped).
    pub fn apply_scans(&mut self, scans: Vec<(usize, Vec<ClusterAccum>)>) -> bool {
        let mut changed = false;
        for (idx, clusters) in scans {
            let Some(slot) = self.per_chunk.get_mut(idx) else {
                continue;
            };
            if *slot != clusters {
                *slot = clusters;
                changed = true;
            }
        }
        if changed {
            self.rebuild();
            self.generation = self.generation.wrapping_add(1).max(1);
        }
        changed
    }

    /// Lights in alias-table order; the index is the `light_idx` stored in GPU reservoirs.
    pub fn sources(&self) -> &[EmissiveSource] {
        &self.sources
    }

    /// Vose alias table with one entry per source.
    pub fn alias(&self) -> &[AliasEntry] {
        &self.alias
    }

    /// Sum of `emissive_luma(intensity)` over all sources (the alias table normalizer).
    pub fn total_flux(&self) -> f32 {
        self.total_flux
    }

    /// Monotonic counter (never 0) bumped whenever the pool changes; invalidates GPU-side light indices.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    fn rebuild(&mut self) {
        let mut cells: HashMap<[i32; 3], ClusterAccum> = HashMap::new();
        for clusters in &self.per_chunk {
            for c in clusters {
                let dst = cells
                    .entry(c.cell)
                    .or_insert_with(|| ClusterAccum::new(c.cell));
                c.merge_into(dst);
            }
        }

        // WHY: Coarsening merges neighboring cells (2x edge per level) instead of dropping lights,
        // so total emitted energy is conserved when the world exceeds the GPU pool capacity.
        while cells.len() > MAX_EMISSIVE_SOURCES {
            let mut coarse: HashMap<[i32; 3], ClusterAccum> = HashMap::new();
            for acc in cells.values() {
                let key = [
                    acc.cell[0].div_euclid(2),
                    acc.cell[1].div_euclid(2),
                    acc.cell[2].div_euclid(2),
                ];
                let dst = coarse.entry(key).or_insert_with(|| ClusterAccum::new(key));
                acc.merge_into(dst);
            }
            cells = coarse;
        }

        let mut keys: Vec<[i32; 3]> = cells.keys().copied().collect();
        keys.sort_unstable();
        self.sources = keys
            .iter()
            .filter_map(|key| cells.get(key).and_then(ClusterAccum::finalize))
            .collect();

        let weights: Vec<f32> = self
            .sources
            .iter()
            .map(|s| emissive_luma(s.intensity))
            .collect();
        self.total_flux = weights.iter().map(|&w| f64::from(w)).sum::<f64>() as f32;
        self.alias = build_alias_table(&weights);
    }
}

/// Scans one chunk for exposed emissive voxel faces and clusters them into 1 m cells.
///
/// Neighbor lookups go through `world` so faces on chunk borders are classified correctly.
pub fn scan_chunk_clusters(world: &WorldMap, chunk_idx: usize) -> Vec<ClusterAccum> {
    let chunk = &world.chunks[chunk_idx];
    if chunk.is_empty() {
        return Vec::new();
    }

    let mut emitter_lut = [false; 256];
    for (id, flag) in emitter_lut.iter_mut().enumerate() {
        *flag = is_registered_emitter(&world.palette, id as MaterialId);
    }
    if !emitter_lut.iter().any(|&e| e) {
        return Vec::new();
    }

    let gx = world.grid_dims.x as usize;
    let gy = world.grid_dims.y as usize;
    let chunk_coord = IVec3::new(
        (chunk_idx % gx) as i32,
        ((chunk_idx / gx) % gy) as i32,
        (chunk_idx / (gx * gy)) as i32,
    );

    let mut ctx = ScanCtx {
        world,
        chunk,
        origin: chunk_coord * CHUNK_DIM as i32,
        emitter_lut,
        clusters: HashMap::new(),
    };
    ctx.scan_node(0, IVec3::ZERO, 6);

    let mut out: Vec<ClusterAccum> = ctx.clusters.into_values().collect();
    out.sort_by_key(|c| c.cell);
    out
}

/// Builds a Vose alias table in `O(N)`; `weights` need not be normalized.
///
/// Drawing column `k` uniformly and keeping it with probability `prob[k]` (else `alias_idx[k]`)
/// selects index `i` with probability `weights[i] / sum(weights)`.
pub fn build_alias_table(weights: &[f32]) -> Vec<AliasEntry> {
    let n = weights.len();
    let mut table: Vec<AliasEntry> = (0..n)
        .map(|i| AliasEntry {
            prob: 1.0,
            alias_idx: i as u32,
        })
        .collect();
    let total: f64 = weights.iter().map(|&w| f64::from(w.max(0.0))).sum();
    if n == 0 || total <= 0.0 {
        return table;
    }

    let mut scaled: Vec<f64> = weights
        .iter()
        .map(|&w| f64::from(w.max(0.0)) * n as f64 / total)
        .collect();
    let mut small: Vec<usize> = Vec::new();
    let mut large: Vec<usize> = Vec::new();
    for (i, &s) in scaled.iter().enumerate() {
        if s < 1.0 {
            small.push(i);
        } else {
            large.push(i);
        }
    }

    while let (Some(s), Some(l)) = (small.pop(), large.pop()) {
        table[s] = AliasEntry {
            prob: scaled[s] as f32,
            alias_idx: l as u32,
        };
        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        if scaled[l] < 1.0 {
            small.push(l);
        } else {
            large.push(l);
        }
    }
    // WHY: Leftovers are 1.0 up to rounding error; they keep their own column with probability 1.
    for i in small.into_iter().chain(large) {
        table[i] = AliasEntry {
            prob: 1.0,
            alias_idx: i as u32,
        };
    }
    table
}

/// Packs a unit vector as three unorm10 channels of `n * 0.5 + 0.5`.
// GPU-SYNC: Inverse of `di_unpack_normal` in `shaders/common/di.wgsl`.
pub fn pack_normal_unorm10(n: [f64; 3]) -> u32 {
    let q = |v: f64| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 1023.0).round() as u32;
    q(n[0]) | (q(n[1]) << 10) | (q(n[2]) << 20)
}

struct ScanCtx<'a> {
    world: &'a WorldMap,
    chunk: &'a arcadia_core::ChunkTree64,
    /// Global voxel coordinate of the chunk's `(0, 0, 0)` voxel.
    origin: IVec3,
    emitter_lut: [bool; 256],
    clusters: HashMap<[i32; 3], ClusterAccum>,
}

impl ScanCtx<'_> {
    /// Walks the chunk tree; `shift` is log2 of the child cell edge in voxels (6, 4, 2, 0).
    fn scan_node(&mut self, node_idx: usize, local_origin: IVec3, shift: u32) {
        let node = self.chunk.nodes[node_idx];
        let mut mask = node.mask_u64();
        while mask != 0 {
            let b = mask.trailing_zeros();
            mask &= mask - 1;

            let (cx, cy, cz) = unpack_child_index(b);
            let child_origin =
                local_origin + IVec3::new(cx as i32, cy as i32, cz as i32) * (1i32 << shift);

            if !node.is_leaf() {
                self.scan_node(node.child_slot_offset(b) as usize, child_origin, shift - 2);
                continue;
            }

            // INVARIANT: A leaf above Level 3 is a homogeneous-collapsed cell whose material is `primary_material`.
            let mat = if shift > 0 || node.is_uniform() {
                node.primary_material()
            } else {
                self.chunk.bricks[node.base_ptr() as usize].get_material(b)
            };
            if !self.emitter_lut[mat as usize] {
                continue;
            }

            let global = self.origin + child_origin;
            if shift == 0 {
                for dir in FACE_DIRS {
                    self.try_add_face(global, dir, mat);
                }
            } else {
                self.add_cell_boundary_faces(global, 1i32 << shift, mat);
            }
        }
    }

    /// Adds the exposed faces on the outer shell of a uniform emissive cube; its interior is never exposed.
    fn add_cell_boundary_faces(&mut self, origin: IVec3, size: i32, mat: MaterialId) {
        for axis in 0..3usize {
            let a1 = (axis + 1) % 3;
            let a2 = (axis + 2) % 3;
            for positive in [false, true] {
                let mut dir = IVec3::ZERO;
                dir[axis] = if positive { 1 } else { -1 };
                let layer = if positive { size - 1 } else { 0 };
                for u in 0..size {
                    for v in 0..size {
                        let mut voxel = origin;
                        voxel[axis] += layer;
                        voxel[a1] += u;
                        voxel[a2] += v;
                        self.try_add_face(voxel, dir, mat);
                    }
                }
            }
        }
    }

    fn try_add_face(&mut self, voxel: IVec3, dir: IVec3, mat: MaterialId) {
        let n = voxel + dir;
        let neighbor = self.world.get_voxel(n.x, n.y, n.z);
        if neighbor != MAT_AIR && neighbor != MAT_WATER_CRYSTAL {
            return;
        }

        let em = self.world.palette.get(mat).emission_meta;
        let emission = [f64::from(em[0]), f64::from(em[1]), f64::from(em[2])];
        let vs = f64::from(VOXEL_SIZE);
        let center = [
            (f64::from(voxel.x) + 0.5 + 0.5 * f64::from(dir.x)) * vs,
            (f64::from(voxel.y) + 0.5 + 0.5 * f64::from(dir.y)) * vs,
            (f64::from(voxel.z) + 0.5 + 0.5 * f64::from(dir.z)) * vs,
        ];
        let cell = [
            voxel.x.div_euclid(EMISSIVE_CELL_VOXELS),
            voxel.y.div_euclid(EMISSIVE_CELL_VOXELS),
            voxel.z.div_euclid(EMISSIVE_CELL_VOXELS),
        ];
        self.clusters
            .entry(cell)
            .or_insert_with(|| ClusterAccum::new(cell))
            .add_face(
                center,
                [f64::from(dir.x), f64::from(dir.y), f64::from(dir.z)],
                emission,
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcadia_core::{ChunkTree64, MAT_CRYSTAL_CYAN, MAT_LAVA_MAGMA, MAT_STONE_DEEP};
    use glam::UVec3;

    fn lava_slab_world() -> WorldMap {
        let palette = MaterialPalette::default();
        let mut chunk = ChunkTree64::default();
        let mut voxels = Vec::new();
        for x in 100..104 {
            for z in 100..104 {
                voxels.push((x, 49, z, MAT_STONE_DEEP));
                voxels.push((x, 50, z, MAT_LAVA_MAGMA));
            }
        }
        chunk.set_voxels_local(&voxels, &palette);
        WorldMap::from_chunks(UVec3::new(1, 1, 1), vec![chunk], palette)
    }

    #[test]
    fn alias_table_reproduces_weight_distribution() {
        let weights = [5.0f32, 1.0, 0.25, 3.0, 0.75, 10.0, 2.0];
        let table = build_alias_table(&weights);
        let total: f64 = weights.iter().map(|&w| f64::from(w)).sum();
        let n = weights.len() as f64;

        let mut prob = vec![0.0f64; weights.len()];
        for (k, entry) in table.iter().enumerate() {
            assert!((0.0..=1.0).contains(&entry.prob));
            prob[k] += f64::from(entry.prob) / n;
            prob[entry.alias_idx as usize] += (1.0 - f64::from(entry.prob)) / n;
        }
        for (i, &w) in weights.iter().enumerate() {
            assert!(
                (prob[i] - f64::from(w) / total).abs() < 1e-6,
                "index {i}: sampled {} expected {}",
                prob[i],
                f64::from(w) / total
            );
        }
    }

    #[test]
    fn alias_table_handles_degenerate_inputs() {
        assert!(build_alias_table(&[]).is_empty());
        let single = build_alias_table(&[3.0]);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].alias_idx, 0);
        assert!((single[0].prob - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normal_packing_round_trips() {
        let packed = pack_normal_unorm10([0.0, 1.0, 0.0]);
        assert_eq!(packed & 1023, 512);
        assert_eq!((packed >> 10) & 1023, 1023);
        assert_eq!((packed >> 20) & 1023, 512);
    }

    #[test]
    fn exposed_lava_slab_becomes_one_source() {
        let world = lava_slab_world();
        let reg = &world.emissive;
        assert_eq!(reg.sources().len(), 1);
        assert_eq!(reg.alias().len(), 1);

        let src = reg.sources()[0];
        // 16 top faces + 16 side faces; the bottom faces touch stone and are blocked.
        let em = world.palette.get(MAT_LAVA_MAGMA).emission_meta;
        let expected = 32.0 * FACE_AREA;
        for c in 0..3 {
            assert!(((f64::from(src.intensity[c])) - f64::from(em[c]) * expected).abs() < 1e-4);
        }
        // Slab spans voxels 100..104; its centroid sits above the 4x4 footprint center.
        assert!((src.pos[0] - 102.0 / 16.0).abs() < 0.05);
        assert!((src.pos[2] - 102.0 / 16.0).abs() < 0.05);
        assert!(src.pos[1] > 50.0 / 16.0 && src.pos[1] < 51.0 / 16.0);
        assert!(src.radius > 0.5 * VOXEL_SIZE);

        let n = src.normal_packed;
        let ny = ((n >> 10) & 1023) as f32 / 1023.0 * 2.0 - 1.0;
        assert!(ny > 0.45, "dominant normal must point up, got {ny}");
        assert!((reg.total_flux() - emissive_luma(src.intensity)).abs() < 1e-5);
    }

    #[test]
    fn point_light_material_is_not_registered() {
        let palette = MaterialPalette::default();
        assert!(is_registered_emitter(&palette, MAT_LAVA_MAGMA));
        assert!(is_registered_emitter(&palette, MAT_CRYSTAL_CYAN));
        assert!(!is_registered_emitter(&palette, MAT_STONE_DEEP));
    }

    #[test]
    fn edits_bump_generation_only_when_emitters_change() {
        let mut world = lava_slab_world();
        let g0 = world.emissive.generation();
        assert!(g0 >= 1);

        world.set_voxels_global(&[(IVec3::new(10, 10, 10), MAT_STONE_DEEP)]);
        assert!(!world.refresh_emissive_registry());
        assert_eq!(world.emissive.generation(), g0);

        world.set_voxels_global(&[(IVec3::new(140, 50, 140), MAT_CRYSTAL_CYAN)]);
        assert!(world.refresh_emissive_registry());
        assert_eq!(world.emissive.sources().len(), 2);
        assert!(world.emissive.generation() != g0);

        let g1 = world.emissive.generation();
        world.clear_voxels_global(&[IVec3::new(140, 50, 140)]);
        assert!(world.refresh_emissive_registry());
        assert_eq!(world.emissive.sources().len(), 1);
        assert!(world.emissive.generation() != g1);
    }

    #[test]
    fn neighbor_chunk_edit_updates_border_exposure() {
        let palette = MaterialPalette::default();
        let mut chunk0 = ChunkTree64::default();
        chunk0.set_voxels_local(&[(255, 40, 40, MAT_LAVA_MAGMA)], &palette);
        let mut chunk1 = ChunkTree64::default();
        chunk1.set_voxels_local(&[(0, 40, 40, MAT_STONE_DEEP)], &palette);
        let mut world = WorldMap::from_chunks(UVec3::new(2, 1, 1), vec![chunk0, chunk1], palette);

        let blocked_intensity = world.emissive.sources()[0].intensity[0];
        world.clear_voxels_global(&[IVec3::new(256, 40, 40)]);
        assert!(world.refresh_emissive_registry());
        let exposed_intensity = world.emissive.sources()[0].intensity[0];
        assert!(
            (exposed_intensity / blocked_intensity - 6.0 / 5.0).abs() < 1e-3,
            "removing the blocking stone must expose the sixth face"
        );
    }

    #[test]
    fn overflow_coarsens_instead_of_dropping_energy() {
        let palette = MaterialPalette::default();
        let chunk_count = 5usize;
        let mut chunks = Vec::new();
        for _ in 0..chunk_count {
            let mut chunk = ChunkTree64::default();
            let mut voxels = Vec::new();
            for cx in 0..16u32 {
                for cy in 0..16u32 {
                    for cz in 0..16u32 {
                        voxels.push((cx * 16 + 3, cy * 16 + 3, cz * 16 + 3, MAT_LAVA_MAGMA));
                    }
                }
            }
            chunk.set_voxels_local(&voxels, &palette);
            chunks.push(chunk);
        }
        let world = WorldMap::from_chunks(UVec3::new(chunk_count as u32, 1, 1), chunks, palette);

        let sources = world.emissive.sources();
        assert!(!sources.is_empty() && sources.len() <= MAX_EMISSIVE_SOURCES);
        assert_eq!(world.emissive.alias().len(), sources.len());

        let em = world.palette.get(MAT_LAVA_MAGMA).emission_meta;
        let total_r: f64 = sources.iter().map(|s| f64::from(s.intensity[0])).sum();
        let expected = f64::from(em[0]) * 6.0 * FACE_AREA * (chunk_count * 4096) as f64;
        assert!(
            (total_r / expected - 1.0).abs() < 1e-3,
            "coarsening must conserve emitted intensity: {total_r} vs {expected}"
        );
    }
}
