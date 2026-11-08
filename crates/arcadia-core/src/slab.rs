//! Generic free-list Slab Allocator for managing contiguous GPU/CPU pools (`Node64` and `LeafBrick64`).
//! Owns CPU-side pool memory recycling and slot indices.

/// Manages a flat `Vec<T>` pool with free-list recycling for single slots (`LeafBrick64`).
#[derive(Clone, Debug)]
pub struct SlabAllocator<T: Copy + Default> {
    data: Vec<T>,
    free_indices: Vec<u32>,
}

impl<T: Copy + Default> Default for SlabAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default> SlabAllocator<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            free_indices: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
            free_indices: Vec::new(),
        }
    }

    #[inline]
    pub fn alloc(&mut self, item: T) -> u32 {
        if let Some(idx) = self.free_indices.pop() {
            self.data[idx as usize] = item;
            idx
        } else {
            let idx = self.data.len() as u32;
            self.data.push(item);
            idx
        }
    }

    #[inline]
    pub fn free(&mut self, idx: u32) {
        self.free_indices.push(idx);
    }

    #[inline]
    pub fn get(&self, idx: u32) -> &T {
        &self.data[idx as usize]
    }

    #[inline]
    pub fn get_mut(&mut self, idx: u32) -> &mut T {
        &mut self.data[idx as usize]
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
