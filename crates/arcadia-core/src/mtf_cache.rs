//! Self-Organizing Move-to-Front (MTF) LRU Cache (`mtf_cache.rs`).
//! Owns CPU-side register/L1-resident cache state.
//!
//! Implements a register/L1-resident N-slot array cache where:
//! - **Hit at slot 0**: Returns immediately in $O(1)$ with zero memory writes.
//! - **Hit at slot $i > 0$**: Bubbles the entry toward the front (swapping with slot $i - 1$ or
//!   rotating to slot 0) so hot lookups permanently live at slot 0.
//! - **Miss**: Shifts slots $0..N-1$ down by 1 (evicting the oldest/coldest entry at slot $N - 1$)
//!   and inserts the new key-value pair at slot 0.

/// Fixed-capacity Move-to-Front (MTF) / LRU array cache with zero heap allocation.
#[derive(Clone, Debug)]
pub struct MtfCache<K: Copy + PartialEq, V: Copy, const N: usize> {
    keys: [K; N],
    vals: [V; N],
    len: usize,
    hits: u64,
    misses: u64,
}

impl<K: Copy + PartialEq + Default, V: Copy + Default, const N: usize> Default
    for MtfCache<K, V, N>
{
    fn default() -> Self {
        Self::new(K::default(), V::default())
    }
}

impl<K: Copy + PartialEq, V: Copy, const N: usize> MtfCache<K, V, N> {
    /// Creates a new empty `MtfCache` initialized with dummy values.
    #[inline]
    pub const fn new(default_key: K, default_val: V) -> Self {
        Self {
            keys: [default_key; N],
            vals: [default_val; N],
            len: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Queries `key` in the cache.
    /// - If found at slot 0, returns immediately.
    /// - If found at slot `i > 0`, bubbles the entry to the front (slot 0) and shifts `0..i` down.
    #[inline(always)]
    pub fn get(&mut self, key: K) -> Option<V> {
        if self.len > 0 && self.keys[0] == key {
            self.hits += 1;
            return Some(self.vals[0]);
        }

        let mut i = 1;
        while i < self.len {
            if self.keys[i] == key {
                self.hits += 1;
                let k = self.keys[i];
                let v = self.vals[i];
                // Bubble to front (slot 0), shifting 0..i down by 1:
                let mut j = i;
                while j > 0 {
                    self.keys[j] = self.keys[j - 1];
                    self.vals[j] = self.vals[j - 1];
                    j -= 1;
                }
                self.keys[0] = k;
                self.vals[0] = v;
                return Some(v);
            }
            i += 1;
        }

        self.misses += 1;
        None
    }

    /// Inserts `(key, val)` at the front of the cache (slot 0), pushing older entries back
    /// and evicting the oldest entry at slot `N - 1` if full.
    #[inline(always)]
    pub fn insert(&mut self, key: K, val: V) {
        if N == 0 {
            return;
        }
        let limit = if self.len < N {
            let old = self.len;
            self.len += 1;
            old
        } else {
            N - 1
        };

        let mut j = limit;
        while j > 0 {
            self.keys[j] = self.keys[j - 1];
            self.vals[j] = self.vals[j - 1];
            j -= 1;
        }
        self.keys[0] = key;
        self.vals[0] = val;
    }

    /// Looks up `key` in the Move-to-Front cache, or computes and inserts it at slot 0 on a miss.
    #[inline(always)]
    pub fn get_or_insert_with<F: FnOnce(K) -> V>(&mut self, key: K, compute: F) -> V {
        if let Some(v) = self.get(key) {
            return v;
        }
        let v = compute(key);
        self.insert(key, v);
        v
    }

    /// Clears all cached entries.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Number of valid entries currently in the cache (`0..=N`).
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the cache is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total cache hits recorded.
    #[inline]
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// Total cache misses recorded.
    #[inline]
    pub const fn misses(&self) -> u64 {
        self.misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtf_bubble_and_eviction() {
        let mut cache: MtfCache<u32, u32, 4> = MtfCache::new(0, 0);
        assert_eq!(cache.get_or_insert_with(10, |k| k * 10), 100);
        assert_eq!(cache.get_or_insert_with(20, |k| k * 10), 200);
        assert_eq!(cache.get_or_insert_with(30, |k| k * 10), 300);
        assert_eq!(cache.get_or_insert_with(40, |k| k * 10), 400);
        // Order is now: [40, 30, 20, 10]
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.misses(), 4);

        // Touch 10 -> bubbles to front: [10, 40, 30, 20]
        assert_eq!(cache.get(10), Some(100));
        assert_eq!(cache.hits(), 1);

        // Insert 50 -> evicts 20 (which fell off the back): [50, 10, 40, 30]
        cache.insert(50, 500);
        assert_eq!(cache.get(20), None);
        assert_eq!(cache.get(10), Some(100));
        assert_eq!(cache.get(50), Some(500));
        assert_eq!(cache.get(40), Some(400));
        assert_eq!(cache.get(30), Some(300));
    }
}
