//! Sixteen control bytes at a time.
//!
//! The table stores one control byte per slot, separate from the slots themselves, so that a probe touches a dense run of tags instead of striding over entries. Sixteen of them fit in one vector register and in a quarter of a cache line, so a probe compares sixteen candidate slots in about three instructions and then visits only the slots that matched.
//!
//! This is the reason the table beats a distance-to-bucket probe rather than trading against it. Robin Hood hashing reads one bucket, compares, and decides whether to continue, which is a dependent load per step; here the comparison for sixteen steps happens at once and the loop body runs once per sixteen slots.
//!
//! Three implementations, chosen at compile time, and every one of them agrees on the same [`BitMask`] contract: NEON on `aarch64`, SSE2 on `x86_64`, and a word-at-a-time fallback of eight lanes everywhere else. The fallback is correct rather than fast, and it exists so the crate builds on a platform nobody has benchmarked yet.

/// An empty slot: nothing has ever been stored here, and a probe that reaches one may stop.
pub(crate) const EMPTY: u8 = 0b1000_0000;

/// A slot whose entry was removed. A probe must walk past it, because a live entry may have been placed beyond it while this one was occupied.
pub(crate) const DELETED: u8 = 0b1111_1110;

/// `match_free` finds both markers with one signed comparison against zero, which is only the same answer as two equality tests while both markers have their top bit set and no tag does. Checked here rather than asserted at run time, because it is a property of two constants and a compiler can say so.
const _: () = assert!(EMPTY & 0x80 != 0 && DELETED & 0x80 != 0);

/// Whether a control byte holds a live entry.
///
/// Both [`EMPTY`] and [`DELETED`] have the top bit set and every tag has it clear, so this is one test rather than two comparisons.
#[inline]
pub(crate) const fn is_full(ctrl: u8) -> bool {
    ctrl & 0x80 == 0
}

/// The seven bit tag a hash contributes to a control byte.
///
/// Taken from bits 32 to 38. The shard index comes off the top of the hash and the slot index comes off the bottom, so a tag drawn from the middle is evidence those two did not already provide. Drawing it from the top instead, which is the obvious thing to do, would make the tag a restatement of the shard number: every key in a shard would carry nearly the same tag, every probe would match every lane, and the table would degrade to comparing keys one at a time.
///
/// The top bit is cleared so that no tag can be mistaken for [`EMPTY`] or [`DELETED`].
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "taking one byte out of the middle is the whole operation"
)]
pub(crate) const fn tag_of(hash: u64) -> u8 {
    ((hash >> 32) as u8) & 0x7f
}

/// Which lanes of a [`Group`] matched.
///
/// The bits are spaced [`BitMask::STRIDE`] apart because NEON has no move-mask instruction and the cheapest substitute produces four bits per lane. Callers go through [`BitMask::next`] and never see the spacing.
#[derive(Debug, Clone)]
pub(crate) struct BitMask(u64);

impl BitMask {
    /// How far apart the bits of one mask are, which is a property of how each implementation produces them.
    ///
    /// SSE2 has a move-mask instruction and gives one bit per lane. NEON has none, and the cheapest substitute narrows sixteen bytes to eight, leaving four bits per lane. The word-at-a-time fallback lights the top bit of each byte, which is eight apart. Getting this wrong does not fail to compile; it silently reports lane 7 as lane 0, so each arm is spelled out rather than left to a fallback.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    const STRIDE: usize = 4;
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    const STRIDE: usize = 1;
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    const STRIDE: usize = 8;

    /// Whether some lane matched.
    #[inline]
    pub(crate) const fn any(&self) -> bool {
        self.0 != 0
    }

    /// The index of the lowest matching lane, if there is one.
    #[inline]
    pub(crate) const fn lowest(&self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(self.0.trailing_zeros() as usize / Self::STRIDE)
        }
    }
}

impl Iterator for BitMask {
    type Item = usize;

    /// Each matching lane in ascending order.
    #[inline]
    fn next(&mut self) -> Option<usize> {
        let at = self.lowest()?;
        // A lane occupies STRIDE bits and every one of them has to go, not just the lowest, or the same lane is reported again on the next call.
        let lane = if Self::STRIDE == 64 {
            u64::MAX
        } else {
            ((1u64 << Self::STRIDE) - 1) << (at * Self::STRIDE)
        };
        self.0 &= !lane;
        Some(at)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod imp {
    use super::{BitMask, EMPTY};
    use core::arch::aarch64 as neon;

    /// Control bytes compared at once.
    pub(crate) const WIDTH: usize = 16;

    /// Sixteen control bytes held in a vector register.
    #[derive(Clone, Copy)]
    pub(crate) struct Group(neon::uint8x16_t);

    impl Group {
        /// Read the sixteen control bytes starting at `at`.
        ///
        /// # Safety
        ///
        /// `at` must be the start of sixteen readable bytes. The table probes group-aligned positions only, and its capacity is a multiple of [`WIDTH`], so a group never runs past the end of the control array and no tail padding is needed.
        #[inline]
        pub(crate) unsafe fn load(at: *const u8) -> Self {
            // SAFETY: the caller guarantees sixteen readable bytes at `at`, and an unaligned load is what this intrinsic is for.
            Self(unsafe { neon::vld1q_u8(at) })
        }

        /// Turn a lane-wise all-ones or all-zeros vector into a mask with four bits per lane.
        ///
        /// `vshrn_n_u16` narrows eight 16-bit lanes to eight 8-bit ones by taking four bits from each, which over a 16-byte vector read as eight halfwords gives exactly four mask bits per original byte.
        #[inline]
        fn to_mask(cmp: neon::uint8x16_t) -> BitMask {
            // SAFETY: both intrinsics are unconditionally available under target_feature = "neon", which this module is gated on.
            let narrowed = unsafe {
                neon::vreinterpret_u64_u8(neon::vshrn_n_u16(neon::vreinterpretq_u16_u8(cmp), 4))
            };
            // SAFETY: as above.
            BitMask(unsafe { neon::vget_lane_u64(narrowed, 0) })
        }

        /// Lanes holding exactly `byte`.
        #[inline]
        pub(crate) fn match_byte(self, byte: u8) -> BitMask {
            // SAFETY: available under target_feature = "neon".
            Self::to_mask(unsafe { neon::vceqq_u8(self.0, neon::vdupq_n_u8(byte)) })
        }

        /// Lanes holding [`EMPTY`].
        #[inline]
        pub(crate) fn match_empty(self) -> BitMask {
            self.match_byte(EMPTY)
        }

        /// Lanes holding [`EMPTY`](super::EMPTY) or [`DELETED`](super::DELETED), which is to say every lane a new entry could be placed in.
        ///
        /// Both have the top bit set and no tag does, so this is a signed comparison against zero rather than two equality tests.
        #[inline]
        pub(crate) fn match_free(self) -> BitMask {
            // SAFETY: available under target_feature = "neon".
            Self::to_mask(unsafe {
                neon::vreinterpretq_u8_s8(neon::vshrq_n_s8(neon::vreinterpretq_s8_u8(self.0), 7))
            })
        }
    }

    /// Start fetching the cache line holding `at` into L1.
    ///
    /// `core::arch::aarch64::_prefetch` is not stable, so this is the instruction it would emit, written out. `pldl1keep` is a load into L1 with the line kept for reuse, which is what a slot about to be read wants.
    #[inline]
    pub(crate) fn prefetch(at: *const u8) {
        // SAFETY: `prfm` is a hint. It never faults, never traps on an unmapped or misaligned address, and has no architectural effect other than on the cache, so any value of `at` is sound. `nostack` and `readonly` say the same thing to the compiler, and `at` is only read as an address.
        unsafe {
            core::arch::asm!("prfm pldl1keep, [{at}]", at = in(reg) at, options(nostack, readonly, preserves_flags));
        }
    }

    /// Start fetching the cache line holding `at` into L2.
    ///
    /// `pldl2keep` rather than `pldl1keep`, for the reason on [`super::prefetch_far`].
    #[inline]
    pub(crate) fn prefetch_far(at: *const u8) {
        // SAFETY: as in `prefetch`. Which level of cache is named changes nothing about whether the hint is sound.
        unsafe {
            core::arch::asm!("prfm pldl2keep, [{at}]", at = in(reg) at, options(nostack, readonly, preserves_flags));
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod imp {
    use super::{BitMask, EMPTY};
    use core::arch::x86_64 as sse;

    /// Control bytes compared at once.
    pub(crate) const WIDTH: usize = 16;

    /// Sixteen control bytes held in a vector register.
    #[derive(Clone, Copy)]
    pub(crate) struct Group(sse::__m128i);

    impl Group {
        /// Read the sixteen control bytes starting at `at`.
        ///
        /// # Safety
        ///
        /// `at` must be the start of sixteen readable bytes, which the table's group-aligned probing guarantees.
        #[inline]
        pub(crate) unsafe fn load(at: *const u8) -> Self {
            // SAFETY: the caller guarantees sixteen readable bytes, and `loadu` is the unaligned form.
            Self(unsafe { sse::_mm_loadu_si128(at.cast()) })
        }

        /// Lanes holding exactly `byte`.
        #[inline]
        pub(crate) fn match_byte(self, byte: u8) -> BitMask {
            // SAFETY: both intrinsics are unconditionally available under target_feature = "sse2", which this module is gated on.
            unsafe {
                let cmp = sse::_mm_cmpeq_epi8(self.0, sse::_mm_set1_epi8(byte.cast_signed()));
                BitMask(u64::from(sse::_mm_movemask_epi8(cmp).cast_unsigned()))
            }
        }

        /// Lanes holding [`EMPTY`].
        #[inline]
        pub(crate) fn match_empty(self) -> BitMask {
            self.match_byte(EMPTY)
        }

        /// Lanes holding [`EMPTY`](super::EMPTY) or [`DELETED`](super::DELETED).
        ///
        /// The move-mask instruction takes the top bit of each lane, and the top bit is exactly the "not a tag" bit, so this needs no comparison at all.
        #[inline]
        pub(crate) fn match_free(self) -> BitMask {
            // SAFETY: available under target_feature = "sse2".
            BitMask(u64::from(
                unsafe { sse::_mm_movemask_epi8(self.0) }.cast_unsigned(),
            ))
        }
    }

    /// Start fetching the cache line holding `at` into L1.
    #[inline]
    pub(crate) fn prefetch(at: *const u8) {
        // SAFETY: `_mm_prefetch` is a hint. It reads nothing and faults on nothing, so any address is sound to pass it, and it is unconditionally available under target_feature = "sse2".
        unsafe { sse::_mm_prefetch::<{ sse::_MM_HINT_T0 }>(at.cast()) }
    }

    /// Start fetching the cache line holding `at` into L2.
    ///
    /// `_MM_HINT_T1` rather than `_MM_HINT_T0`, for the reason on [`super::prefetch_far`].
    #[inline]
    pub(crate) fn prefetch_far(at: *const u8) {
        // SAFETY: as in `prefetch`. Which level of cache is named changes nothing about whether the hint is sound.
        unsafe { sse::_mm_prefetch::<{ sse::_MM_HINT_T1 }>(at.cast()) }
    }
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "sse2")
)))]
mod imp {
    use super::{BitMask, EMPTY};

    /// Control bytes compared at once.
    pub(crate) const WIDTH: usize = 8;

    /// Eight control bytes held in a word.
    #[derive(Clone, Copy)]
    pub(crate) struct Group(u64);

    /// One in the low bit of each lane.
    const ONES: u64 = 0x0101_0101_0101_0101;
    /// One in the top bit of each lane.
    const HIGHS: u64 = 0x8080_8080_8080_8080;

    impl Group {
        /// Read the eight control bytes starting at `at`.
        ///
        /// # Safety
        ///
        /// `at` must be the start of eight readable bytes, which the table's group-aligned probing guarantees.
        #[inline]
        pub(crate) unsafe fn load(at: *const u8) -> Self {
            // SAFETY: the caller guarantees eight readable bytes, and the read is unaligned by construction.
            Self(u64::from_le(unsafe { at.cast::<u64>().read_unaligned() }))
        }

        /// Lanes holding exactly `byte`.
        ///
        /// The usual zero-byte trick: xor makes the wanted lanes zero, and `(x - ONES) & !x & HIGHS` lights the top bit of every zero lane. It can also light the top bit of a lane that borrowed from its neighbour, which is why the result is masked with the lanes that are genuinely not full further down. Tags never use the top bit, so a borrow cannot forge a match against a real tag.
        #[inline]
        pub(crate) fn match_byte(self, byte: u8) -> BitMask {
            let x = self.0 ^ (ONES.wrapping_mul(u64::from(byte)));
            BitMask(x.wrapping_sub(ONES) & !x & HIGHS)
        }

        /// Lanes holding [`EMPTY`].
        #[inline]
        pub(crate) fn match_empty(self) -> BitMask {
            self.match_byte(EMPTY)
        }

        /// Lanes holding [`EMPTY`](super::EMPTY) or [`DELETED`](super::DELETED).
        #[inline]
        pub(crate) fn match_free(self) -> BitMask {
            BitMask(self.0 & HIGHS)
        }
    }

    /// Start fetching the cache line holding `at` into L1, on a target where there is no portable way to ask.
    ///
    /// Nothing happens. The caller loads the address a few instructions later either way, so a target without a prefetch hint gets the same answer more slowly rather than a different answer.
    #[inline]
    pub(crate) fn prefetch(at: *const u8) {
        let _ = at;
    }

    /// As above, and as little.
    #[inline]
    pub(crate) fn prefetch_far(at: *const u8) {
        let _ = at;
    }
}

/// Start fetching the cache line holding `at` into a level of cache that will still hold it a while.
///
/// [`prefetch`] asks for L1, which is what a line wanted a few instructions later should ask for. This is for a line wanted much further off, and L1 is the wrong place to put that: the work in between streams a value through it and the line is gone before anything reads it, so the hint buys a fetch that has to happen twice. L2 is far enough back to survive that and near enough that reaching it is a fraction of the cost of going to memory.
pub(crate) use imp::prefetch_far;
pub(crate) use imp::{Group, WIDTH, prefetch};

#[cfg(test)]
mod tests {
    use super::*;

    /// Load a group out of a fixed array, which is padded so the load is always in bounds.
    fn group_of(bytes: &[u8]) -> Group {
        let mut buf = [EMPTY; WIDTH * 2];
        buf[..bytes.len()].copy_from_slice(bytes);
        // SAFETY: `buf` is WIDTH * 2 bytes and the load reads the first WIDTH of them.
        unsafe { Group::load(buf.as_ptr()) }
    }

    #[test]
    fn a_tag_matches_only_its_own_lane() {
        let mut bytes = [EMPTY; WIDTH];
        bytes[3] = 0x2a;
        let found: Vec<usize> = group_of(&bytes).match_byte(0x2a).collect();
        assert_eq!(found, vec![3]);
    }

    #[test]
    fn every_lane_can_match() {
        // A stride bug or a narrowing bug shows up as one particular lane never being reported.
        for lane in 0..WIDTH {
            let mut bytes = [EMPTY; WIDTH];
            bytes[lane] = 0x11;
            let found: Vec<usize> = group_of(&bytes).match_byte(0x11).collect();
            assert_eq!(found, vec![lane], "lane {lane} did not match");
        }
    }

    #[test]
    fn several_lanes_come_back_in_order() {
        let mut bytes = [0x00; WIDTH];
        bytes[1] = 0x7f;
        bytes[4] = 0x7f;
        bytes[WIDTH - 1] = 0x7f;
        let found: Vec<usize> = group_of(&bytes).match_byte(0x7f).collect();
        assert_eq!(found, vec![1, 4, WIDTH - 1]);
    }

    #[test]
    fn empty_and_deleted_are_free_and_tags_are_not() {
        let mut bytes = [0x00; WIDTH];
        bytes[0] = EMPTY;
        bytes[1] = DELETED;
        bytes[2] = 0x7f;
        let free: Vec<usize> = group_of(&bytes).match_free().collect();
        assert_eq!(free, vec![0, 1]);
    }

    #[test]
    fn only_empty_is_empty() {
        let mut bytes = [0x00; WIDTH];
        bytes[0] = EMPTY;
        bytes[1] = DELETED;
        let found: Vec<usize> = group_of(&bytes).match_empty().collect();
        assert_eq!(found, vec![0], "a tombstone was reported as never-used");
    }

    #[test]
    fn a_tag_never_collides_with_a_marker() {
        // The word-at-a-time path can light a lane that borrowed from its neighbour. Tags are seven bits and both markers have the top bit set, so this walks every tag against a group full of markers to prove no tag is ever mistaken for one.
        for tag in 0..=0x7fu8 {
            let bytes = [DELETED; WIDTH];
            assert!(
                !group_of(&bytes).match_byte(tag).any(),
                "tag {tag:#x} matched a tombstone"
            );
            let bytes = [EMPTY; WIDTH];
            assert!(
                !group_of(&bytes).match_byte(tag).any(),
                "tag {tag:#x} matched an empty slot"
            );
        }
    }

    #[test]
    fn a_full_group_of_one_tag_reports_every_lane() {
        let bytes = [0x33; WIDTH];
        let found: Vec<usize> = group_of(&bytes).match_byte(0x33).collect();
        assert_eq!(found, (0..WIDTH).collect::<Vec<_>>());
    }

    #[test]
    fn the_tag_never_has_its_top_bit_set() {
        // If it did, a live entry would read as free and the table would hand out an occupied slot.
        for i in 0..10_000u64 {
            let tag = tag_of(i.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            assert!(is_full(tag), "tag {tag:#x} would read as a marker");
        }
    }
}
