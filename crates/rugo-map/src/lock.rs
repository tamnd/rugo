//! The one lock a shard has.
//!
//! A shard is small and every operation on it is short: probe a group, compare a key, copy some bytes. Parking a thread costs a syscall each way, which is more than the whole critical section, so this spins and only gives the scheduler its turn after spinning has clearly failed.
//!
//! # Why not a reader-writer lock
//!
//! A shared lock has to be written to in order to be taken, so two readers on two cores still bounce the same cache line between them. The exclusive lock here costs a reader exactly the same line, and costs nothing to release, so with thousands of shards the sharing that a reader-writer lock would recover is sharing that was not contended anyway.
//!
//! # Why one hundred and twenty-eight bytes
//!
//! Two shards whose locks share a cache line contend for that line even when no key they hold has anything to do with the other, which is a slowdown that no amount of sharding removes. A cache line is sixty-four bytes on most x86 and one hundred and twenty-eight on Apple silicon, and x86 prefetches in pairs of lines, so the alignment is the larger of the two.

use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};

#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, Ordering};

#[cfg(not(loom))]
use core::sync::atomic::{AtomicBool, Ordering};

/// Spins before the thread gives up its turn.
///
/// Long enough to cover a critical section that is only waiting on a cache line to arrive, short enough that a thread descheduled while holding the lock does not cost a full timeslice of burnt cycles.
const SPINS: u32 = 40;

/// An exclusive lock that spins.
#[repr(align(128))]
#[derive(Debug)]
pub struct Lock<T> {
    /// Whether somebody holds it.
    held: AtomicBool,
    /// What it guards.
    value: UnsafeCell<T>,
}

// SAFETY: the lock hands out at most one `&mut T` at a time, and the acquire and release orderings below make the previous holder's writes visible to the next one. Sending the `T` between threads is the whole point, so `T: Send` is the requirement and `T: Sync` is not needed.
unsafe impl<T: Send> Sync for Lock<T> {}
// SAFETY: as above; moving the lock moves the value it owns.
unsafe impl<T: Send> Send for Lock<T> {}

impl<T> Lock<T> {
    /// A lock nobody holds.
    #[cfg(not(loom))]
    pub const fn new(value: T) -> Self {
        Self {
            held: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    /// A lock nobody holds.
    #[cfg(loom)]
    pub fn new(value: T) -> Self {
        Self {
            held: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    /// Take the lock, waiting for it if somebody else has it.
    pub fn lock(&self) -> Guard<'_, T> {
        loop {
            if let Some(guard) = self.try_lock() {
                return guard;
            }
            let mut spun = 0;
            // Read rather than exchange while waiting. A failing exchange takes the line exclusively and takes it away from the holder, which slows down the very work being waited on.
            while self.held.load(Ordering::Relaxed) {
                if spun < SPINS {
                    spun += 1;
                    #[cfg(not(loom))]
                    core::hint::spin_loop();
                    #[cfg(loom)]
                    loom::thread::yield_now();
                } else {
                    #[cfg(not(loom))]
                    std::thread::yield_now();
                    #[cfg(loom)]
                    loom::thread::yield_now();
                }
            }
        }
    }

    /// Take the lock if it is free, and do not wait if it is not.
    pub fn try_lock(&self) -> Option<Guard<'_, T>> {
        if self
            .held
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(Guard { lock: self })
        } else {
            None
        }
    }

    /// The value, without locking, which is sound because the borrow proves nobody else has a reference.
    pub fn get_mut(&mut self) -> &mut T {
        self.value.get_mut()
    }
}

/// Proof that the holder has the lock, and the way to reach what it guards.
#[derive(Debug)]
pub struct Guard<'a, T> {
    /// The lock this came from.
    lock: &'a Lock<T>,
}

impl<T> Deref for Guard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: this guard exists only while the lock is held, and the lock is held by exactly one guard, so no other reference to the value exists.
        unsafe { &*self.lock.value.get() }
    }
}

impl<T> DerefMut for Guard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as in `deref`, and `&mut self` proves this guard is not aliased either.
        unsafe { &mut *self.lock.value.get() }
    }
}

impl<T> Drop for Guard<'_, T> {
    fn drop(&mut self) {
        self.lock.held.store(false, Ordering::Release);
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn a_lock_guards() {
        let lock = Lock::new(5u32);
        assert_eq!(*lock.lock(), 5);
        *lock.lock() += 1;
        assert_eq!(*lock.lock(), 6);
    }

    #[test]
    fn a_second_take_fails_while_the_first_is_held() {
        let lock = Lock::new(());
        let held = lock.lock();
        assert!(lock.try_lock().is_none());
        drop(held);
        assert!(lock.try_lock().is_some());
    }

    #[test]
    fn nothing_is_lost_under_contention() {
        // Every increment has to survive. A lost update here is the shape of the bug that a missing acquire or release would cause in the map, where it would show up as an entry that half exists.
        let lock = Arc::new(Lock::new(0u64));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let lock = Arc::clone(&lock);
                thread::spawn(move || {
                    for _ in 0..20_000 {
                        *lock.lock() += 1;
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(*lock.lock(), 8 * 20_000);
    }

    #[test]
    fn a_lock_does_not_share_a_cache_line() {
        // Two shards whose locks land in one line contend even when nothing they hold is related, and no amount of sharding fixes it.
        assert_eq!(align_of::<Lock<u8>>(), 128);
        let locks: Vec<Lock<u8>> = (0..4).map(Lock::new).collect();
        let first = std::ptr::from_ref(&locks[0]) as usize;
        let second = std::ptr::from_ref(&locks[1]) as usize;
        assert!(
            second - first >= 128,
            "two locks are {} bytes apart",
            second - first
        );
    }
}
