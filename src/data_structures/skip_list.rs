//! Lock-Free Skip List Implementation
//! Ultra-fast O(log N) L2 order book price level indexing with zero garbage collection.
//! Uses custom memory arenas to prevent OS-level malloc fragmentation.

use core::sync::atomic::{AtomicI64, AtomicPtr, AtomicUsize, Ordering};
use core::ptr;

/// Maximum height for skip list levels (log2 of max elements)
const MAX_HEIGHT: usize = 16;

/// Node capacity in arena
const ARENA_CAPACITY: usize = 4096;

/// Skip list node with atomic pointers for lock-free operations
#[repr(C)]
pub struct SkipListNode {
    /// Key (price level for order book)
    key: i64,
    /// Value associated with key
    value: i64,
    /// Forward pointers for each level
    forward: [AtomicPtr<SkipListNode>; MAX_HEIGHT],
    /// Height of this node
    height: usize,
}

impl SkipListNode {
    const fn new(key: i64, value: i64, height: usize) -> Self {
        Self {
            key,
            value,
            forward: [AtomicPtr::new(ptr::null_mut()); MAX_HEIGHT],
            height,
        }
    }
}

/// Memory arena for skip list nodes (pre-allocated to avoid malloc)
#[repr(C)]
pub struct SkipListArena {
    nodes: [SkipListNode; ARENA_CAPACITY],
    allocated: AtomicUsize,
}

impl SkipListArena {
    const fn new() -> Self {
        Self {
            nodes: unsafe { core::mem::zeroed() },
            allocated: AtomicUsize::new(0),
        }
    }

    fn allocate(&self, key: i64, value: i64, height: usize) -> Option<&mut SkipListNode> {
        let idx = self.allocated.fetch_add(1, Ordering::Relaxed);
        if idx >= ARENA_CAPACITY {
            return None;
        }

        let node = unsafe { &mut *self.nodes.get_unchecked_mut(idx) };
        node.key = key;
        node.value = value;
        node.height = height;
        
        // Reset forward pointers
        for i in 0..MAX_HEIGHT {
            node.forward[i].store(ptr::null_mut(), Ordering::Relaxed);
        }

        Some(node)
    }

    fn reset(&self) {
        self.allocated.store(0, Ordering::Relaxed);
    }
}

/// Random number generator for level determination (xorshift32)
struct XorShift32 {
    state: AtomicUsize,
}

impl XorShift32 {
    const fn new(seed: usize) -> Self {
        Self {
            state: AtomicUsize::new(seed),
        }
    }

    #[inline]
    fn next(&self) -> u32 {
        let mut state = self.state.load(Ordering::Relaxed);
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        self.state.store(state, Ordering::Relaxed);
        state as u32
    }

    /// Generate random level with geometric distribution
    #[inline]
    fn random_level(&self, probability: i32) -> usize {
        let mut level = 1;
        while (self.next() as i32 % 100) < probability && level < MAX_HEIGHT {
            level += 1;
        }
        level
    }
}

/// Lock-free skip list for price level indexing
pub struct LockFreeSkipList {
    /// Head node (sentinel)
    head: AtomicPtr<SkipListNode>,
    /// Current height of the skip list
    height: AtomicUsize,
    /// Number of elements
    count: AtomicUsize,
    /// Memory arena
    arena: SkipListArena,
    /// RNG for level generation
    rng: XorShift32,
}

unsafe impl Send for LockFreeSkipList {}
unsafe impl Sync for LockFreeSkipList {}

impl Default for LockFreeSkipList {
    fn default() -> Self {
        Self::new()
    }
}

impl LockFreeSkipList {
    /// Create a new empty skip list
    pub const fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
            height: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            arena: SkipListArena::new(),
            rng: XorShift32::new(12345),
        }
    }

    /// Insert or update a key-value pair
    pub fn insert(&self, key: i64, value: i64) -> bool {
        let mut update: [*mut SkipListNode; MAX_HEIGHT] = [ptr::null_mut(); MAX_HEIGHT];
        let mut current = self.head.load(Ordering::Acquire);

        // Find position at each level
        for i in (0..self.height.load(Ordering::Relaxed)).rev() {
            unsafe {
                while !current.is_null() {
                    let node = &*current;
                    let next = node.forward[i].load(Ordering::Acquire);
                    
                    if next.is_null() || (*next).key > key {
                        break;
                    }
                    
                    if (*next).key == key {
                        // Update existing value
                        (*next).value = value;
                        return true;
                    }
                    
                    current = next;
                }
            }
            update[i] = current;
        }

        // Generate random level for new node
        let new_height = self.rng.random_level(50);

        // Update list height if necessary
        let current_height = self.height.load(Ordering::Relaxed);
        if new_height > current_height {
            for i in current_height..new_height {
                update[i] = ptr::null_mut();
            }
            self.height.store(new_height, Ordering::Release);
        }

        // Allocate new node from arena
        let new_node = match self.arena.allocate(key, value, new_height) {
            Some(node) => node as *mut SkipListNode,
            None => return false, // Arena full
        };

        // Insert node at each level
        unsafe {
            for i in 0..new_height {
                let next = if update[i].is_null() {
                    self.head.load(Ordering::Acquire)
                } else {
                    (*update[i]).forward[i].load(Ordering::Acquire)
                };
                
                (*new_node).forward[i].store(next, Ordering::Relaxed);
                
                if update[i].is_null() {
                    // Compare-and-swap at head
                    let mut expected = next;
                    while self
                        .head
                        .compare_exchange(expected, new_node, Ordering::SeqCst, Ordering::Relaxed)
                        .is_err()
                    {
                        expected = next;
                    }
                } else {
                    // Compare-and-swap at update node
                    let mut expected = next;
                    while (*update[i])
                        .forward[i]
                        .compare_exchange(expected, new_node, Ordering::SeqCst, Ordering::Relaxed)
                        .is_err()
                    {
                        expected = next;
                    }
                }
            }
        }

        self.count.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Get value for a key
    pub fn get(&self, key: i64) -> Option<i64> {
        let mut current = self.head.load(Ordering::Acquire);

        unsafe {
            for i in (0..self.height.load(Ordering::Relaxed)).rev() {
                while !current.is_null() {
                    let node = &*current;
                    let next = node.forward[i].load(Ordering::Acquire);

                    if next.is_null() || (*next).key >= key {
                        break;
                    }
                    current = next;
                }
            }

            // Check if we found the key
            if !current.is_null() {
                let next = (*current).forward[0].load(Ordering::Acquire);
                if !next.is_null() && (*next).key == key {
                    return Some((*next).value);
                }
            }
        }

        None
    }

    /// Remove a key from the skip list
    pub fn remove(&self, key: i64) -> bool {
        let mut update: [*mut SkipListNode; MAX_HEIGHT] = [ptr::null_mut(); MAX_HEIGHT];
        let mut current = self.head.load(Ordering::Acquire);

        // Find the node to remove
        for i in (0..self.height.load(Ordering::Relaxed)).rev() {
            unsafe {
                while !current.is_null() {
                    let node = &*current;
                    let next = node.forward[i].load(Ordering::Acquire);

                    if next.is_null() || (*next).key >= key {
                        break;
                    }
                    current = next;
                }
            }
            update[i] = current;
        }

        unsafe {
            let current_height = self.height.load(Ordering::Relaxed);
            let target = if current.is_null() {
                None
            } else {
                let next = (*current).forward[0].load(Ordering::Acquire);
                if !next.is_null() && (*next).key == key {
                    Some(next)
                } else {
                    None
                }
            };

            if let Some(target) = target {
                // Remove from each level
                for i in 0..current_height {
                    let next_ptr = (*update[i]).forward[i].load(Ordering::Acquire);
                    if next_ptr == target {
                        let new_next = (*target).forward[i].load(Ordering::Acquire);
                        (*update[i]).forward[i].store(new_next, Ordering::Release);
                    }
                }

                // Update height if necessary
                while self.height.load(Ordering::Relaxed) > 1 {
                    let h = self.height.load(Ordering::Relaxed);
                    if self.head.load(Ordering::Acquire).is_null()
                        || (*self.head.load(Ordering::Acquire)).forward[h - 1].load(Ordering::Acquire).is_null()
                    {
                        self.height.store(h - 1, Ordering::Release);
                    } else {
                        break;
                    }
                }

                self.count.fetch_sub(1, Ordering::Relaxed);
                return true;
            }
        }

        false
    }

    /// Find the largest key less than or equal to given key
    pub fn floor(&self, key: i64) -> Option<(i64, i64)> {
        let mut current = self.head.load(Ordering::Acquire);
        let mut result: Option<(i64, i64)> = None;

        unsafe {
            for i in (0..self.height.load(Ordering::Relaxed)).rev() {
                while !current.is_null() {
                    let node = &*current;
                    let next = node.forward[i].load(Ordering::Acquire);

                    if next.is_null() || (*next).key > key {
                        break;
                    }
                    current = next;
                }
            }

            if !current.is_null() && (*current).key <= key {
                result = Some(((*current).key, (*current).value));
            } else if !current.is_null() {
                let next = (*current).forward[0].load(Ordering::Acquire);
                if !next.is_null() && (*next).key <= key {
                    result = Some(((*next).key, (*next).value));
                }
            }
        }

        result
    }

    /// Find the smallest key greater than or equal to given key
    pub fn ceiling(&self, key: i64) -> Option<(i64, i64)> {
        let mut current = self.head.load(Ordering::Acquire);

        unsafe {
            for i in (0..self.height.load(Ordering::Relaxed)).rev() {
                while !current.is_null() {
                    let node = &*current;
                    let next = node.forward[i].load(Ordering::Acquire);

                    if next.is_null() || (*next).key >= key {
                        break;
                    }
                    current = next;
                }
            }

            if !current.is_null() {
                let next = (*current).forward[0].load(Ordering::Acquire);
                if !next.is_null() && (*next).key >= key {
                    return Some(((*next).key, (*next).value));
                }
            }
        }

        None
    }

    /// Get the number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all elements (reset arena)
    pub fn clear(&mut self) {
        self.arena.reset();
        self.head.store(ptr::null_mut(), Ordering::Release);
        self.height.store(0, Ordering::Release);
        self.count.store(0, Ordering::Release);
    }

    /// Range query: iterate over keys in range [start, end]
    pub fn range<F>(&self, start: i64, end: i64, mut f: F)
    where
        F: FnMut(i64, i64),
    {
        let mut current = self.head.load(Ordering::Acquire);

        unsafe {
            // Find starting position
            for i in (0..self.height.load(Ordering::Relaxed)).rev() {
                while !current.is_null() {
                    let node = &*current;
                    let next = node.forward[i].load(Ordering::Acquire);

                    if next.is_null() || (*next).key >= start {
                        break;
                    }
                    current = next;
                }
            }

            // Iterate through range
            if !current.is_null() {
                let mut node = (*current).forward[0].load(Ordering::Acquire);
                while !node.is_null() {
                    let key = (*node).key;
                    if key > end {
                        break;
                    }
                    if key >= start {
                        f(key, (*node).value);
                    }
                    node = (*node).forward[0].load(Ordering::Acquire);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let list = LockFreeSkipList::new();
        
        list.insert(100, 1000);
        list.insert(200, 2000);
        list.insert(150, 1500);
        
        assert_eq!(list.get(100), Some(1000));
        assert_eq!(list.get(150), Some(1500));
        assert_eq!(list.get(200), Some(2000));
        assert_eq!(list.get(50), None);
    }

    #[test]
    fn test_floor_ceiling() {
        let list = LockFreeSkipList::new();
        
        list.insert(100, 1000);
        list.insert(200, 2000);
        list.insert(300, 3000);
        
        assert_eq!(list.floor(150), Some((100, 1000)));
        assert_eq!(list.ceiling(150), Some((200, 2000)));
        assert_eq!(list.floor(100), Some((100, 1000)));
        assert_eq!(list.ceiling(300), Some((300, 3000)));
    }

    #[test]
    fn test_remove() {
        let list = LockFreeSkipList::new();
        
        list.insert(100, 1000);
        list.insert(200, 2000);
        
        assert!(list.remove(100));
        assert_eq!(list.get(100), None);
        assert_eq!(list.get(200), Some(2000));
    }
}
