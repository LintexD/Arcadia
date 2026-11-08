//! Pairwise dynamic body contact detection entry points for voxel-accurate collisions.
//! Owns CPU narrowphase pairwise contact detection without macro-box distortion.

use super::super::types::{BodyPairManifold, RigidVoxelBody};

impl RigidVoxelBody {
    /// Detects contact between two rigid voxel bodies at true voxel-box precision.
    #[inline]
    #[allow(dead_code)]
    pub fn detect_contact_manifold(body_a: &Self, body_b: &Self) -> Option<BodyPairManifold> {
        Self::compute_compound_pair_manifolds_indexed(body_a, body_b, 0, 1)
            .into_iter()
            .next()
    }
}
