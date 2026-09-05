//! What the server counts, counted where it happens.
//!
//! One block of counters a thread, each on its own cache line, each written only by the thread that owns it. `INFO` sums them, which is the one place that reads somebody else's.
//!
//! The alternative — one shared counter incremented by every thread on every command — is a single cache line moving between every core in the machine at the command rate. On a thirty-two core box that costs more than some of the commands being counted. This is the same reasoning that puts the map's byte counts in per-shard atomics.

use core::sync::atomic::{AtomicU64, Ordering};

/// One thread's counters, alone on a cache line.
///
/// Sixty-four byte aligned and padded to it, so two threads' counters are never in the same line and no thread's store invalidates another's.
#[derive(Debug, Default)]
#[repr(align(64))]
pub struct Counters {
    /// Commands run.
    commands: AtomicU64,
    /// Reads that found a value.
    hits: AtomicU64,
    /// Reads that did not.
    misses: AtomicU64,
    /// Connections this thread has accepted.
    connections: AtomicU64,
}

impl Counters {
    /// Add one, without an atomic read-modify-write.
    ///
    /// Only the owning thread ever writes these, so there is no increment to lose and no need for a locked instruction. The atomic is what makes reading them from another thread defined, not what makes the arithmetic correct.
    #[inline]
    fn bump(counter: &AtomicU64) {
        counter.store(counter.load(Ordering::Relaxed) + 1, Ordering::Relaxed);
    }

    /// A command was run.
    #[inline]
    pub fn command(&self) {
        Self::bump(&self.commands);
    }

    /// A read found what it was looking for.
    #[inline]
    pub fn hit(&self) {
        Self::bump(&self.hits);
    }

    /// A read did not.
    #[inline]
    pub fn miss(&self) {
        Self::bump(&self.misses);
    }

    /// A connection was accepted.
    #[inline]
    pub fn connection(&self) {
        Self::bump(&self.connections);
    }
}

/// Every thread's counters.
#[derive(Debug)]
pub struct Stats {
    /// One block a thread, indexed by thread number.
    per: Box<[Counters]>,
}

/// What the whole server has counted, summed across its threads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Total {
    /// Commands run.
    pub commands: u64,
    /// Reads that found a value.
    pub hits: u64,
    /// Reads that did not.
    pub misses: u64,
    /// Connections accepted.
    pub connections: u64,
}

impl Stats {
    /// Counters for `threads` threads.
    #[must_use]
    pub fn new(threads: usize) -> Self {
        Self {
            per: (0..threads).map(|_| Counters::default()).collect(),
        }
    }

    /// One thread's block.
    ///
    /// # Panics
    ///
    /// If `thread` is not one of the threads this was built for, which would be a server that started more threads than it counted.
    #[must_use]
    pub fn thread(&self, thread: usize) -> &Counters {
        &self.per[thread]
    }

    /// Everything, added up.
    ///
    /// Not a consistent snapshot: the threads go on counting while this reads them, so two of these numbers may be from different instants. `INFO` on a server under load has no other kind of answer, and pretending otherwise would cost a barrier on every command to make a report prettier.
    #[must_use]
    pub fn total(&self) -> Total {
        let mut total = Total::default();
        for counters in &self.per {
            total.commands += counters.commands.load(Ordering::Relaxed);
            total.hits += counters.hits.load(Ordering::Relaxed);
            total.misses += counters.misses.load(Ordering::Relaxed);
            total.connections += counters.connections.load(Ordering::Relaxed);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_thread_counts_into_its_own_block() {
        let stats = Stats::new(4);
        stats.thread(2).command();
        stats.thread(2).hit();
        stats.thread(3).miss();
        assert_eq!(
            stats.total(),
            Total {
                commands: 1,
                hits: 1,
                misses: 1,
                connections: 0,
            }
        );
    }

    #[test]
    fn every_thread_gets_its_own_cache_line() {
        // The whole point of the layout. Two adjacent blocks sixty-four bytes apart is what keeps one thread's counting out of another thread's cache.
        assert_eq!(align_of::<Counters>(), 64);
        assert_eq!(size_of::<Counters>(), 64);
        let stats = Stats::new(2);
        let first = std::ptr::from_ref(stats.thread(0)) as usize;
        let second = std::ptr::from_ref(stats.thread(1)) as usize;
        assert_eq!(second - first, 64, "two threads' counters shared a line");
    }

    #[test]
    fn counting_from_eight_threads_loses_nothing() {
        // Only the owning thread writes a block, which is what makes a plain load and store enough. If that ever stopped being true this would find it.
        let stats = std::sync::Arc::new(Stats::new(8));
        std::thread::scope(|scope| {
            for thread in 0..8 {
                let stats = std::sync::Arc::clone(&stats);
                scope.spawn(move || {
                    for _ in 0..10_000 {
                        stats.thread(thread).command();
                    }
                });
            }
        });
        assert_eq!(stats.total().commands, 80_000);
    }
}
