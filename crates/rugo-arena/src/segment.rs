//! Where a segment's bytes come from.
//!
//! The arena above this asks for a segment and then fills it up over time. Between the asking and the filling, the part not yet written to is the arena's reserve, and how much that reserve costs is the difference between two designs.
//!
//! A segment cut from the ordinary allocator costs all of itself immediately. `vec![0u8; n]` has to produce zeroed bytes, and producing them means writing them, so every page of a fresh segment is resident before a single entry is in it. The reserve is real memory and the growth rule has to be miserly to keep it small, which means many small segments.
//!
//! A segment that is its own anonymous mapping costs only the pages that have been written to. The kernel supplies zeroed pages on first touch and charges for them then, so the reserve is address space and nothing else. The growth rule can then be generous, which means few large mappings, which is the thing that keeps a cache of four thousand shards under the kernel's limit on how many mappings one process may have.
//!
//! Measured on a million hundred byte entries across four thousand shards, that reserve was seven per cent of everything the arena held. It is the largest single item of overhead in the cache and this module is what removes it.
//!
//! # Where it does not apply
//!
//! [`LAZY`] says which of the two is in use. Windows has no `mmap` and the equivalent is `VirtualAlloc` with a separate reserve and commit, which is a second implementation rather than a spelling change, and Miri interprets the program rather than running it and has no mapping to give. Both fall back to a boxed slice, which is what this crate did everywhere before, so they are correct and pay the reserve. Neither is a platform rugo publishes a memory number from.

use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

/// Whether a byte of a segment that has not been written to is free.
///
/// True where segments are anonymous mappings. The arena reads this to choose a growth rule and to decide whether its reserve counts as resident, so the two answers cannot drift apart.
pub(crate) const LAZY: bool = cfg!(all(unix, not(miri)));

/// The bytes of one segment.
///
/// Owns its storage and gives it out as a slice. What that storage is depends on [`LAZY`]; nothing above this module needs to know which.
pub(crate) struct Segment {
    /// The first byte.
    ptr: NonNull<u8>,
    /// How many bytes there are, after any rounding the platform imposed.
    len: usize,
}

// SAFETY: a segment owns its storage outright and shares it only through `&self` and `&mut self`, exactly as the `Box<[u8]>` it replaces did. Nothing else holds the pointer, so moving one between threads or sharing a reference to one is sound for the same reason it was sound for the box.
unsafe impl Send for Segment {}
// SAFETY: as above.
unsafe impl Sync for Segment {}

impl Deref for Segment {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` and `len` come from `new`, which produced `len` readable bytes at `ptr` and zeroed them, and the storage lives exactly as long as `self`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for Segment {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `deref`, and `&mut self` is what makes the exclusive borrow of the storage exclusive.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl core::fmt::Debug for Segment {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The bytes are entry data and printing a megabyte of it helps nobody.
        write!(f, "Segment({} bytes)", self.len)
    }
}

#[cfg(all(unix, not(miri)))]
mod platform {
    use super::{NonNull, Segment};
    use core::sync::atomic::{AtomicUsize, Ordering};

    /// The page size, asked for once.
    ///
    /// Four kilobytes nearly everywhere and sixteen on Apple silicon, which is why it is asked rather than assumed: a segment rounded to the wrong page size would have the arena counting bytes the kernel did not give it.
    pub(super) fn granule() -> usize {
        static CACHED: AtomicUsize = AtomicUsize::new(0);
        let seen = CACHED.load(Ordering::Relaxed);
        if seen != 0 {
            return seen;
        }
        // SAFETY: `sysconf` reads a numeric limit by name and touches no memory of ours. `_SC_PAGESIZE` is required to exist by POSIX.
        let asked = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        // A machine that will not say gets four kilobytes, which is the smallest page any of these platforms has, so a segment is rounded to at least a whole one.
        let size = usize::try_from(asked).unwrap_or(4096).max(4096);
        CACHED.store(size, Ordering::Relaxed);
        size
    }

    /// Map `size` bytes, rounded up to a page.
    pub(super) fn new(size: usize) -> Option<Segment> {
        let len = size.next_multiple_of(granule());
        // SAFETY: an anonymous private mapping with a null hint and a non-zero length is a call the kernel either satisfies at an address of its choosing or refuses with `MAP_FAILED`, which is checked below. No pointer of ours is passed in and the pages come zeroed.
        let ptr = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return None;
        }
        Some(Segment {
            ptr: NonNull::new(ptr.cast::<u8>())?,
            len,
        })
    }

    /// Give the mapping back.
    pub(super) fn drop(segment: &mut Segment) {
        // SAFETY: `ptr` and `len` are the address and length of a mapping this segment made in `new` and has not unmapped, and `drop` runs once.
        unsafe {
            libc::munmap(segment.ptr.as_ptr().cast::<libc::c_void>(), segment.len);
        }
    }
}

#[cfg(not(all(unix, not(miri))))]
mod platform {
    use super::{NonNull, Segment};

    /// One byte, because every byte of a boxed slice is resident whether or not anything was written to it.
    pub(super) const fn granule() -> usize {
        1
    }

    /// Take `size` zeroed bytes from the ordinary allocator.
    pub(super) fn new(size: usize) -> Option<Segment> {
        let boxed = vec![0u8; size].into_boxed_slice();
        let len = boxed.len();
        Some(Segment {
            ptr: NonNull::new(Box::into_raw(boxed).cast::<u8>())?,
            len,
        })
    }

    /// Give the box back.
    pub(super) fn drop(segment: &mut Segment) {
        // SAFETY: `ptr` came from `Box::into_raw` on a `Box<[u8]>` of `len` bytes in `new`, and `drop` runs once, so this reconstructs exactly the box that was taken apart.
        unsafe {
            core::mem::drop(Box::from_raw(core::ptr::slice_from_raw_parts_mut(
                segment.ptr.as_ptr(),
                segment.len,
            )));
        }
    }
}

impl Segment {
    /// Storage for at least `size` bytes, zeroed, or `None` if the platform would not give it.
    ///
    /// May be longer than asked for, because a mapping is a whole number of pages. The extra is usable and the caller is told about it by [`Segment::len`], so it is not lost.
    pub(super) fn new(size: usize) -> Option<Self> {
        platform::new(size)
    }
}

/// The unit an untouched byte is charged in: a page where segments are mapped, one byte where they are not.
pub(crate) fn granule() -> usize {
    platform::granule()
}

impl Drop for Segment {
    fn drop(&mut self) {
        platform::drop(self);
    }
}

#[cfg(test)]
mod tests {
    use super::{Segment, granule};

    #[test]
    fn a_segment_is_zeroed_and_writable() {
        // The arena writes an entry before reading it, but a free list link is read out of a block that was only ever a segment tail, so the zeroing is not decorative.
        let mut segment = Segment::new(64 * 1024).expect("a sixty-four kilobyte segment");
        assert!(segment.iter().all(|&b| b == 0), "a segment came back dirty");
        let len = segment.len();
        segment[len - 1] = 0xff;
        assert_eq!(segment[len - 1], 0xff);
        assert_eq!(segment[0], 0, "a write at the end reached the start");
    }

    #[test]
    fn a_segment_is_at_least_what_was_asked_for() {
        // Rounding up to a page is allowed and rounding down is not, because the arena cuts allocations out of the length this reports.
        for size in [4 * 1024, 1024 * 1024, 8 * 1024 * 1024] {
            let segment = Segment::new(size).expect("a segment");
            assert!(
                segment.len() >= size,
                "asked for {size} and got {}",
                segment.len()
            );
            assert!(
                segment.len().is_multiple_of(granule()),
                "{} is not a whole number of {} byte granules",
                segment.len(),
                granule()
            );
        }
    }

    #[test]
    fn many_segments_can_be_held_and_dropped() {
        // A cache of four thousand shards holds thousands of these at once, and dropping one must not disturb another.
        let mut held: Vec<Segment> = (0..64)
            .map(|_| Segment::new(64 * 1024).expect("a segment"))
            .collect();
        for (n, segment) in held.iter_mut().enumerate() {
            let byte = u8::try_from(n % 256).unwrap_or(0);
            segment[0] = byte;
        }
        for (n, segment) in held.iter().enumerate() {
            assert_eq!(segment[0], u8::try_from(n % 256).unwrap_or(0));
        }
    }
}
