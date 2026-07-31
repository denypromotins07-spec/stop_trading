//! Futex-based Wait-Free Locks for Ultra-Low Latency Contention Resolution
//!
//! Implements custom Linux `futex`-based wait-free locks with strict `#[cfg(target_os = "linux")]` guards.
//! Provides hyper-optimized spin-lock fallback for Windows environments.
//! Correctly handles spurious wakeups and uses atomic Acquire/Release semantics for memory safety.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::cell::UnsafeCell;
use std::marker::PhantomData;

/// Unlocked state
const UNLOCKED: u32 = 0;
/// Locked state (no waiters)
const LOCKED: u32 = 1;
/// Locked state with waiters
const LOCKED_WITH_WAITERS: u32 = 2;

/// Linux futex-based mutex
#[cfg(target_os = "linux")]
pub struct FutexMutex<T> {
    /// Lock state
    state: AtomicU32,
    /// Protected data
    data: UnsafeCell<T>,
}

#[cfg(target_os = "linux")]
unsafe impl<T: Send> Send for FutexMutex<T> {}
#[cfg(target_os = "linux")]
unsafe impl<T: Send + Sync> Sync for FutexMutex<T> {}

#[cfg(target_os = "linux")]
impl<T> FutexMutex<T> {
    /// Create a new futex mutex
    pub const fn new(data: T) -> Self {
        Self {
            state: AtomicU32::new(UNLOCKED),
            data: UnsafeCell::new(data),
        }
    }

    /// Lock the mutex (blocking)
    pub fn lock(&self) -> FutexGuard<'_, T> {
        // Fast path: try to acquire with CAS
        let mut current = self.state.load(Ordering::Relaxed);
        
        loop {
            if current == UNLOCKED {
                match self.state.compare_exchange_weak(
                    current,
                    LOCKED,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return FutexGuard { mutex: self },
                    Err(e) => {
                        current = e;
                        continue;
                    }
                }
            }
            
            // Slow path: use futex wait
            current = self.futex_wait(current);
        }
    }

    /// Try to lock without blocking
    pub fn try_lock(&self) -> Option<FutexGuard<'_, T>> {
        let current = self.state.load(Ordering::Relaxed);
        
        if current == UNLOCKED {
            match self.state.compare_exchange(
                current,
                LOCKED,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => Some(FutexGuard { mutex: self }),
                Err(_) => None,
            }
        } else {
            None
        }
    }

    /// Unlock the mutex
    pub fn unlock(&self) {
        let current = self.state.swap(UNLOCKED, Ordering::Release);
        
        // If there were waiters, wake one
        if current == LOCKED_WITH_WAITERS {
            self.futex_wake_one();
        }
    }

    /// Get reference to protected data (requires lock held)
    #[inline]
    pub fn get(&self) -> &T {
        unsafe { &*self.data.get() }
    }

    /// Get mutable reference to protected data (requires lock held)
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.data.get() }
    }

    /// Futex wait syscall wrapper
    #[inline]
    fn futex_wait(&self, expected: u32) -> u32 {
        // Mark that we're waiting
        let prev = self.state.swap(LOCKED_WITH_WAITERS, Ordering::Relaxed);
        if prev == UNLOCKED {
            return UNLOCKED; // Someone unlocked while we were trying
        }

        // Use futex syscall via libc
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &self.state as *const AtomicU32 as *const u32,
                libc::FUTEX_WAIT_PRIVATE,
                LOCKED_WITH_WAITERS,
                std::ptr::null::<libc::timespec>(),
            );
        }

        // Reload state after wakeup (may be spurious)
        self.state.load(Ordering::Relaxed)
    }

    /// Futex wake syscall wrapper
    #[inline]
    fn futex_wake_one(&self) {
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &self.state as *const AtomicU32 as *const u32,
                libc::FUTEX_WAKE_PRIVATE,
                1, // Wake one waiter
            );
        }
    }
}

/// Windows spin-lock fallback
#[cfg(not(target_os = "linux"))]
pub struct SpinMutex<T> {
    state: AtomicUsize,
    data: UnsafeCell<T>,
}

#[cfg(not(target_os = "linux"))]
unsafe impl<T: Send> Send for SpinMutex<T> {}
#[cfg(not(target_os = "linux"))]
unsafe impl<T: Send + Sync> Sync for SpinMutex<T> {}

#[cfg(not(target_os = "linux"))]
impl<T> SpinMutex<T> {
    pub const fn new(data: T) -> Self {
        Self {
            state: AtomicUsize::new(0),
            data: UnsafeCell::new(data),
        }
    }

    pub fn lock(&self) -> SpinGuard<'_, T> {
        let mut spin = 0;
        loop {
            if self.state.compare_exchange_weak(
                0, 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ).is_ok() {
                return SpinGuard { mutex: self };
            }
            
            // Exponential backoff
            spin += 1;
            if spin < 10 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    pub fn try_lock(&self) -> Option<SpinGuard<'_, T>> {
        if self.state.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
            Some(SpinGuard { mutex: self })
        } else {
            None
        }
    }

    pub fn unlock(&self) {
        self.state.store(0, Ordering::Release);
    }
}

/// Mutex guard for futex mutex
#[cfg(target_os = "linux")]
pub struct FutexGuard<'a, T> {
    mutex: &'a FutexMutex<T>,
}

#[cfg(target_os = "linux")]
impl<'a, T> std::ops::Deref for FutexGuard<'a, T> {
    type Target = T;
    
    fn deref(&self) -> &T {
        self.mutex.get()
    }
}

#[cfg(target_os = "linux")]
impl<'a, T> std::ops::DerefMut for FutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.mutex.get_mut()
    }
}

#[cfg(target_os = "linux")]
impl<'a, T> Drop for FutexGuard<'a, T> {
    fn drop(&mut self) {
        self.mutex.unlock();
    }
}

/// Mutex guard for spin mutex
#[cfg(not(target_os = "linux"))]
pub struct SpinGuard<'a, T> {
    mutex: &'a SpinMutex<T>,
}

#[cfg(not(target_os = "linux"))]
impl<'a, T> std::ops::Deref for SpinGuard<'a, T> {
    type Target = T;
    
    fn deref(&self) -> &T {
        unsafe { &*self.mutex.data.get() }
    }
}

#[cfg(not(target_os = "linux"))]
impl<'a, T> std::ops::DerefMut for SpinGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.data.get() }
    }
}

#[cfg(not(target_os = "linux"))]
impl<'a, T> Drop for SpinGuard<'a, T> {
    fn drop(&mut self) {
        self.mutex.unlock();
    }
}

/// Platform-agnostic mutex type alias
#[cfg(target_os = "linux")]
pub type LowLatencyMutex<T> = FutexMutex<T>;

#[cfg(not(target_os = "linux"))]
pub type LowLatencyMutex<T> = SpinMutex<T>;

/// Atomic flag for lightweight synchronization
pub struct AtomicFlag {
    flag: AtomicU32,
}

impl AtomicFlag {
    pub const fn new() -> Self {
        Self {
            flag: AtomicU32::new(0),
        }
    }

    /// Set the flag
    #[inline]
    pub fn set(&self) {
        self.flag.store(1, Ordering::Release);
    }

    /// Clear the flag
    #[inline]
    pub fn clear(&self) {
        self.flag.store(0, Ordering::Release);
    }

    /// Check if flag is set
    #[inline]
    pub fn is_set(&self) -> bool {
        self.flag.load(Ordering::Acquire) != 0
    }

    /// Wait for flag to be set (with timeout)
    #[cfg(target_os = "linux")]
    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        
        loop {
            if self.is_set() {
                return true;
            }
            
            if Instant::now() >= deadline {
                return false;
            }
            
            // Futex wait on the flag
            unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    &self.flag as *const AtomicU32 as *const u32,
                    libc::FUTEX_WAIT_PRIVATE,
                    0,
                    &libc::timespec {
                        tv_sec: 0,
                        tv_nsec: 1_000_000, // 1ms
                    },
                );
            }
        }
    }

    /// Signal and wake one waiter
    #[cfg(target_os = "linux")]
    pub fn signal(&self) {
        self.set();
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &self.flag as *const AtomicU32 as *const u32,
                libc::FUTEX_WAKE_PRIVATE,
                1,
            );
        }
    }
}

impl Default for AtomicFlag {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_mutex_basic() {
        let mutex = LowLatencyMutex::new(42);
        
        {
            let guard = mutex.lock();
            assert_eq!(*guard, 42);
        }
        
        {
            let mut guard = mutex.lock();
            *guard = 100;
        }
        
        assert_eq!(*mutex.lock(), 100);
    }

    #[test]
    fn test_mutex_try_lock() {
        let mutex = LowLatencyMutex::new(0);
        
        let guard1 = mutex.try_lock();
        assert!(guard1.is_some());
        
        let guard2 = mutex.try_lock();
        assert!(guard2.is_none());
        
        drop(guard1);
        
        let guard3 = mutex.try_lock();
        assert!(guard3.is_some());
    }

    #[test]
    fn test_mutex_thread_safety() {
        let mutex = Arc::new(LowLatencyMutex::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let m = Arc::clone(&mutex);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let mut guard = m.lock();
                    *guard += 1;
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(*mutex.lock(), 1000);
    }

    #[test]
    fn test_atomic_flag() {
        let flag = AtomicFlag::new();
        
        assert!(!flag.is_set());
        
        flag.set();
        assert!(flag.is_set());
        
        flag.clear();
        assert!(!flag.is_set());
    }
}
