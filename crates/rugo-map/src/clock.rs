//! What time it is, without asking the operating system on every key.
//!
//! Expiry is checked on every read, and reading a clock is a call into the kernel's shared page at best and a syscall at worst. A cache doing a million reads a second does not need a million distinct answers to a question whose resolution is one second, so the answer is taken once per event loop turn and read from a word after that.
//!
//! One second is the resolution because that is the resolution the protocols have: Redis's `EXPIRE` and memcache's `exptime` are both whole seconds, and `PEXPIRE`'s milliseconds are converted by the caller. Storing seconds is what lets an expiry fit in the four bytes [`crate::entry`] gives it.

use core::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Seconds since the Unix epoch, straight from the operating system.
///
/// Saturates rather than wrapping in 2106, so a clock that has run past what four bytes hold makes every entry look expired instead of making an ancient entry look fresh. Neither is good, and only one of them serves stale data.
#[must_use]
pub fn unix_now() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| {
            u32::try_from(since.as_secs()).unwrap_or(u32::MAX)
        })
}

/// A clock read from a word, refreshed by whoever calls [`Clock::tick`].
#[derive(Debug)]
pub struct Clock {
    /// The last answer, in seconds since the Unix epoch.
    secs: AtomicU32,
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock {
    /// A clock reading the current time.
    #[must_use]
    pub fn new() -> Self {
        Self {
            secs: AtomicU32::new(unix_now()),
        }
    }

    /// A clock reading exactly `secs`, which does not move until it is told to.
    ///
    /// For tests, which need an expiry to pass without the test taking that long.
    #[must_use]
    pub const fn fixed(secs: u32) -> Self {
        Self {
            secs: AtomicU32::new(secs),
        }
    }

    /// The current second.
    ///
    /// Relaxed because a reader that sees the previous second treats an entry as live for one more read than it strictly should, and that is the same tolerance the one second resolution already has.
    #[inline]
    #[must_use]
    pub fn now(&self) -> u32 {
        self.secs.load(Ordering::Relaxed)
    }

    /// Ask the operating system again, and return the new answer.
    pub fn tick(&self) -> u32 {
        let now = unix_now();
        self.secs.store(now, Ordering::Relaxed);
        now
    }

    /// Move a fixed clock forward, for tests.
    pub fn advance(&self, secs: u32) {
        self.secs.fetch_add(secs, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clock_reads_a_plausible_year() {
        // Somewhere after 2020 and before 2106. A clock that read zero would make every expiry in the future and one that saturated would make every expiry in the past.
        let now = unix_now();
        assert!(now > 1_577_836_800, "the clock reads before 2020");
        assert!(now < u32::MAX, "the clock has saturated");
    }

    #[test]
    fn a_fixed_clock_does_not_move_by_itself() {
        let clock = Clock::fixed(100);
        assert_eq!(clock.now(), 100);
        assert_eq!(clock.now(), 100);
        clock.advance(50);
        assert_eq!(clock.now(), 150);
    }

    #[test]
    fn a_tick_catches_up_to_the_operating_system() {
        let clock = Clock::fixed(0);
        assert_eq!(clock.now(), 0);
        let ticked = clock.tick();
        assert_eq!(clock.now(), ticked);
        assert!(ticked > 1_577_836_800);
    }
}
