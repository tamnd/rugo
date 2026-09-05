//! The per-shard slab allocator rugo stores entry bytes in.
//!
//! This exists for one number. A cache that calls the system allocator once per entry pays a header on every key, and on glibc and on macOS that header is eight to sixteen bytes rounded up to a sixteen byte grain. At a hundred byte entry that is a sixth of the cache spent on bookkeeping the cache cannot read. Pogocache pays it, because pogocache mallocs each entry; rugo does not, because entries come from here.
//!
//! What replaces the header is a size class. An allocation is rounded up to a multiple of eight bytes and comes off the free list for that exact size, so the waste is four bytes on average and nothing is stored beside the entry to say how big it is. The map already knows how big an entry is, because the entry says so in its own header, so a second copy in an allocator header would be the redundancy being removed.
//!
//! # Addressing
//!
//! A [`Ref`] is four bytes, which is the whole point: it is what a table slot holds, and a slot that held a pointer would be eight.
//!
//! The top bit says which of two spaces the rest indexes.
//!
//! With the bit clear, eleven bits name a segment and twenty give an eight byte unit within it. Resolving a reference is a shift, a mask and two loads.
//!
//! With the bit set, they index a table of individually allocated oversized entries. Anything longer than [`SMALL_MAX`] goes there and is allocated at exactly its own length, which means a large value pays the system allocator's header but pays no rounding, and a large value is rare enough that the header does not show up in the total. Making the slab cover every size instead would mean either coarse classes, which waste half of a large entry, or thousands of fine ones, which waste more in free list heads than they save.
//!
//! # Why segments grow
//!
//! Splitting the reference into a segment number and an offset, rather than treating it as one flat unit index, is what lets segments differ in size, and that is what keeps a shard that holds three keys from costing what a shard that holds three million does. The first segment is [`MIN_SEGMENT`] and each one after it is a sixteenth of what the arena already holds, up to [`MAX_SEGMENT`]. See `next_segment_size` for why a sixteenth and not a doubling.
//!
//! One fixed segment size has to choose between a floor and a tail. Large segments mean a cache with four thousand shards pays that size four thousand times over before it is useful; small segments mean a segment ends every time its tail is too short for the next entry. Growing segments have the small floor, and the tail is not lost either: it goes on the free list of whatever class it happens to be, which is what makes the small floor affordable.
//!
//! # What it does not do
//!
//! It does not move entries, so a [`Ref`] is stable until it is freed, and it does not return memory to the operating system: a freed block goes on a free list for the next allocation of its class, and an emptied segment stays. That is why [`Arena::live_bytes`] and [`Arena::resident_bytes`] are separate numbers and why a memory ceiling is enforced against the first of them.

use core::mem::size_of;

/// The first segment a shard allocates.
///
/// Small, because with four thousand shards this is multiplied by four thousand as soon as every shard holds one key, and a cache holding four thousand keys should not cost a quarter of a gigabyte.
pub const MIN_SEGMENT: usize = 4 * 1024;

/// The largest a segment grows to.
///
/// A cap rather than a target: once a shard holds sixteen megabytes a sixteenth of that is a megabyte, and a segment larger than that is a lot to hold in reserve for a marginal saving in list length.
pub const MAX_SEGMENT: usize = 1024 * 1024;

/// The allocation grain, and the alignment every [`Ref`] therefore has.
pub const GRAIN: usize = 8;

/// The largest allocation the slab serves. Anything above this becomes an oversized entry.
///
/// Chosen so that the benchmark's largest cell, a kilobyte value with its key and header, still lands in the slab rather than falling out of the path being measured.
pub const SMALL_MAX: usize = 2048;

/// Bits of a [`Ref`] naming a segment.
const SEG_BITS: u32 = 11;

/// Bits of a [`Ref`] naming an eight byte unit within a segment.
const OFF_BITS: u32 = 20;

/// The mask that takes the unit offset out of a [`Ref`].
const OFF_MASK: u32 = (1 << OFF_BITS) - 1;

/// Segments one shard may have.
const MAX_SEGMENTS: usize = 1 << SEG_BITS;

/// The top bit of a [`Ref`], set when it indexes the oversized table.
const LARGE: u32 = 1 << 31;

/// How many size classes the slab keeps a free list for.
const CLASSES: usize = SMALL_MAX / GRAIN;

/// The reciprocal of how much a new segment adds to what the arena already holds.
///
/// Sixteen, so a segment is a sixteenth. See `next_segment_size`.
const GROWTH: usize = 16;

/// A reference to bytes in an [`Arena`], four bytes wide.
///
/// Valid only in the arena that produced it. Handing one to another arena is a bug this type does not try to catch, in the same way that an index into the wrong slice is a bug a `usize` does not catch, because the alternative is a generation counter that would take the width back to eight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ref(u32);

impl Ref {
    /// The absence of a reference.
    ///
    /// Not a valid allocation: it has the oversized bit set, and the oversized table is capped below the index this would name.
    pub const NONE: Self = Self(u32::MAX);

    /// Whether this is [`Ref::NONE`].
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    /// The raw bits, for a caller storing this in a packed slot.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// A reference from bits that came out of [`Ref::bits`].
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }
}

/// Why an allocation could not be served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Full {
    /// The shard's slab has used all the segments a four byte reference can name.
    Slab,
    /// The shard's oversized table has as many live entries as a reference can name.
    Large,
}

impl core::fmt::Display for Full {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Slab => f.write_str("this shard's slab is full"),
            Self::Large => f.write_str("this shard's oversized table is full"),
        }
    }
}

impl core::error::Error for Full {}

/// How large the next segment should be, given the bytes the arena already holds.
///
/// One [`GROWTH`]th of what is there, rounded up to a whole [`MIN_SEGMENT`] and held between that and [`MAX_SEGMENT`].
///
/// Doubling is the obvious rule and it is the wrong one. Under doubling the newest segment is as large as every earlier one put together, so an arena holds up to twice what it has handed out and half of that is a segment nobody has written to. Measured on a million hundred byte entries across four thousand shards, doubling cost two hundred and fifty-six bytes an entry against a hundred and twenty-four of payload. A sixteenth costs a few percent, because the newest segment is a seventeenth of the total rather than half of it.
///
/// The price is more segments: about a hundred and forty to reach the [`MAX_SEGMENT`] ceiling, against eight under doubling. A segment costs one `Box<[u8]>` in a list, which is sixteen bytes, so a hundred and forty of them is two kilobytes against the sixteen megabytes they address, and [`MAX_SEGMENTS`] still leaves room for nearly two thousand more.
#[inline]
#[must_use]
const fn next_segment_size(held: usize) -> usize {
    let wanted = (held / GROWTH).next_multiple_of(MIN_SEGMENT);
    if wanted < MIN_SEGMENT {
        MIN_SEGMENT
    } else if wanted > MAX_SEGMENT {
        MAX_SEGMENT
    } else {
        wanted
    }
}

/// Narrow a count the arena has already bounded.
///
/// Three counts become `u32` here and every one of them is small: a unit count inside a segment is at most [`MAX_SEGMENT`] over [`GRAIN`], which is a hundred and thirty thousand, and a segment number is at most [`MAX_SEGMENTS`]. Writing the bound in one place rather than three is what makes it checkable.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "callers pass counts bounded by MAX_SEGMENT / GRAIN or by MAX_SEGMENTS, both far under u32::MAX"
)]
const fn narrow(count: usize) -> u32 {
    debug_assert!(count <= u32::MAX as usize);
    count as u32
}

/// A slab allocator for one shard.
///
/// Not synchronised. The shard's lock is what makes it safe to use, and putting a second lock here would be paying twice for one guarantee.
#[derive(Debug)]
pub struct Arena {
    /// Backing storage, allocated on first use and growing by a sixteenth up to [`MAX_SEGMENT`].
    segments: Vec<Box<[u8]>>,
    /// The segment new allocations are being cut from.
    seg: u32,
    /// The next unallocated unit within that segment.
    off: u32,
    /// Free list heads, one per size class in use, each holding [`Ref`] bits or `u32::MAX`.
    ///
    /// Grown to reach the largest class the shard has actually allocated, never to [`CLASSES`]. A full table is a kilobyte, and a cache is normally holding entries of a few sizes near each other, so a shard that stores hundred byte values keeps sixteen heads rather than two hundred and fifty-six. Across four thousand shards that is the difference between four megabytes of free list heads and sixty kilobytes, and the four megabytes would be there whether or not anything was stored.
    free: Vec<u32>,
    /// Individually allocated oversized entries. A hole is an empty slice.
    large: Vec<Box<[u8]>>,
    /// Indices into `large` that are holes.
    large_free: Vec<u32>,
    /// Bytes handed out and not yet freed, rounded up to the grain.
    live: usize,
    /// Bytes freed and sitting on a free list.
    dead: usize,
    /// Bytes held by live oversized blocks, kept as a running total.
    large_live: usize,
    /// Bytes held by segments, kept as a running total so that reporting memory is not a walk.
    segment_bytes: usize,
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

impl Arena {
    /// An arena holding nothing and owning no memory.
    ///
    /// A shard that is never written to costs this struct and no allocation at all, which is what keeps a four thousand shard table from having a floor measured in gigabytes.
    #[must_use]
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            seg: 0,
            off: 0,
            free: Vec::new(),
            large: Vec::new(),
            large_free: Vec::new(),
            live: 0,
            dead: 0,
            large_live: 0,
            segment_bytes: 0,
        }
    }

    /// The size class an allocation of `len` bytes falls in, or `None` if it is oversized.
    #[inline]
    const fn class_of(len: usize) -> Option<usize> {
        if len == 0 || len > SMALL_MAX {
            None
        } else {
            Some(len.div_ceil(GRAIN) - 1)
        }
    }

    /// Build a reference out of a segment number and a unit offset within it.
    #[inline]
    const fn make(seg: u32, off: u32) -> Ref {
        Ref((seg << OFF_BITS) | off)
    }

    /// Reserve `len` bytes and return a reference to them.
    ///
    /// The contents are whatever the last owner left there. Callers write the whole allocation before reading any of it, which every caller in rugo does because an entry is written in one pass.
    ///
    /// # Errors
    ///
    /// [`Full`] when this shard's address space is exhausted, which at two gigabytes in one shard of thousands means the process is out of memory rather than that the shard is unlucky.
    pub fn alloc(&mut self, len: usize) -> Result<Ref, Full> {
        let Some(class) = Self::class_of(len) else {
            return self.alloc_large(len);
        };

        if class >= self.free.len() {
            self.free.resize(class + 1, u32::MAX);
        }

        let rounded = (class + 1) * GRAIN;

        // A free block first. The block's own first four bytes hold the next one, which is why the smallest class is eight bytes wide and not four.
        let head = self.free[class];
        if head != u32::MAX {
            self.free[class] = self.next_free(Ref(head));
            self.dead -= rounded;
            self.live += rounded;
            return Ok(Ref(head));
        }

        let units = narrow(rounded / GRAIN);

        // An allocation may not straddle two segments, because resolving one is a shift and a mask and a straddling block has no single segment to resolve to.
        let room = self
            .segments
            .get(self.seg as usize)
            .map_or(0, |segment| narrow(segment.len() / GRAIN) - self.off);
        if room < units {
            // The tail is too short for this request but it is long enough for a smaller one, and the free list is exactly the place that knows about smaller ones. Abandoning it instead would be a fixed loss per segment, which is affordable only if segments are large, and large segments are the floor a four thousand shard cache cannot pay.
            if room > 0 {
                self.push_free(Self::make(self.seg, self.off), room as usize * GRAIN);
            }
            self.grow()?;
        }
        // Every segment is at least twice [`SMALL_MAX`], so a fresh one always has room for the largest request the slab serves and one growth is always enough.
        debug_assert!(
            self.off as usize + units as usize <= self.segments[self.seg as usize].len() / GRAIN
        );

        let at = Self::make(self.seg, self.off);
        self.off += units;
        self.live += rounded;
        Ok(at)
    }

    /// Add a segment and cut from it.
    fn grow(&mut self) -> Result<(), Full> {
        if self.segments.len() >= MAX_SEGMENTS {
            return Err(Full::Slab);
        }
        let size = next_segment_size(self.segment_bytes);
        self.segments.push(vec![0u8; size].into_boxed_slice());
        self.segment_bytes += size;
        self.seg = narrow(self.segments.len() - 1);
        self.off = 0;
        Ok(())
    }

    /// Reserve an oversized allocation.
    fn alloc_large(&mut self, len: usize) -> Result<Ref, Full> {
        // The slot is found before the block is made, so that a refusal leaves the counters where they were rather than charging for memory nobody got.
        let at = if let Some(hole) = self.large_free.pop() {
            hole
        } else {
            let at = u32::try_from(self.large.len()).map_err(|_| Full::Large)?;
            if at & LARGE != 0 {
                return Err(Full::Large);
            }
            self.large.push(Box::new([]));
            at
        };

        self.live += len;
        self.large_live += len;
        self.large[at as usize] = vec![0u8; len].into_boxed_slice();
        Ok(Ref(at | LARGE))
    }

    /// Give `at`, which must have been `len` bytes from [`Arena::alloc`] on this arena, back.
    ///
    /// Passing a length that is not the one the block was allocated with corrupts the free lists. The map is the only caller and it reads the length out of the entry header it is about to drop, so the two cannot disagree.
    pub fn free(&mut self, at: Ref, len: usize) {
        if at.is_none() {
            return;
        }

        let Some(class) = Self::class_of(len) else {
            let index = (at.0 & !LARGE) as usize;
            let held = self.large[index].len();
            self.live -= held;
            self.large_live -= held;
            self.large[index] = Box::new([]);
            self.large_free.push(at.0 & !LARGE);
            return;
        };

        // The rounded length, not the caller's: the block is the size of its class and the free list has to see it that way.
        let rounded = (class + 1) * GRAIN;
        self.live -= rounded;
        self.push_free(at, rounded);
    }

    /// Put a block of `len` bytes, which must be a whole number of grains and no larger than [`SMALL_MAX`], on its class's free list.
    ///
    /// Does not touch [`Arena::live_bytes`]: a freed entry was live and a segment's abandoned tail never was, and both end up here.
    #[inline]
    fn push_free(&mut self, at: Ref, len: usize) {
        debug_assert!(len > 0 && len <= SMALL_MAX && len.is_multiple_of(GRAIN));
        let class = len / GRAIN - 1;
        // A head that has never been allocated cannot be freed, and a segment tail is always shorter than the request that ended the segment, so its class is one the request has already made room for.
        debug_assert!(class < CLASSES && class < self.free.len());
        self.dead += len;

        let head = self.free[class];
        self.set_next_free(at, head);
        self.free[class] = at.0;
    }

    /// Where a small reference lands: which segment, and how many bytes into it.
    #[inline]
    const fn place(at: Ref) -> (usize, usize) {
        (
            (at.0 >> OFF_BITS) as usize,
            (at.0 & OFF_MASK) as usize * GRAIN,
        )
    }

    /// The bytes behind `at`, which must be a live reference of length `len`.
    #[inline]
    #[must_use]
    pub fn get(&self, at: Ref, len: usize) -> &[u8] {
        if at.0 & LARGE != 0 {
            return &self.large[(at.0 & !LARGE) as usize];
        }
        let (seg, off) = Self::place(at);
        &self.segments[seg][off..off + len]
    }

    /// The bytes behind `at`, to write into.
    #[inline]
    #[must_use]
    pub fn get_mut(&mut self, at: Ref, len: usize) -> &mut [u8] {
        if at.0 & LARGE != 0 {
            return &mut self.large[(at.0 & !LARGE) as usize];
        }
        let (seg, off) = Self::place(at);
        &mut self.segments[seg][off..off + len]
    }

    /// Up to `len` bytes behind `at`, however many of them are actually there.
    ///
    /// For a caller reading a header whose own contents say how long the whole thing is: it knows the header is at most `len` bytes but cannot know the allocation is that long, and asking [`Arena::get`] for more bytes than were allocated would run off the end of a segment. What comes back always covers the allocation itself, which is all a header inside it needs.
    #[inline]
    #[must_use]
    pub fn peek(&self, at: Ref, len: usize) -> &[u8] {
        if at.0 & LARGE != 0 {
            let block = &self.large[(at.0 & !LARGE) as usize];
            return &block[..len.min(block.len())];
        }
        let (seg, off) = Self::place(at);
        let segment = &self.segments[seg];
        &segment[off..off.saturating_add(len).min(segment.len())]
    }

    /// Read the free list link stored inside a free block.
    #[inline]
    fn next_free(&self, at: Ref) -> u32 {
        let (seg, off) = Self::place(at);
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.segments[seg][off..off + 4]);
        u32::from_le_bytes(buf)
    }

    /// Write the free list link into a free block.
    #[inline]
    fn set_next_free(&mut self, at: Ref, next: u32) {
        let (seg, off) = Self::place(at);
        self.segments[seg][off..off + 4].copy_from_slice(&next.to_le_bytes());
    }

    /// Bytes currently handed out, including the rounding up to the grain.
    ///
    /// This is what a memory ceiling is enforced against, rather than [`Arena::resident_bytes`], because freeing an entry makes its bytes available to the next one but does not give a segment back to the operating system. A ceiling measured against resident memory could never be met once the segments existed, and eviction would empty the cache trying.
    #[must_use]
    pub const fn live_bytes(&self) -> usize {
        self.live
    }

    /// Bytes freed and held on a free list for reuse.
    ///
    /// This is not leaked memory; it is memory a later allocation of the same class will take without touching the operating system. It is reported because a cache whose value size changes over its life accumulates it, and a number nobody publishes is a number nobody notices.
    #[must_use]
    pub const fn dead_bytes(&self) -> usize {
        self.dead
    }

    /// Every byte this arena has taken from the operating system, including its own bookkeeping.
    ///
    /// This is the number the memory gate is measured against, so it counts the segment list, the free list heads and the oversized table as well as the segments themselves. An accounting that flattered itself by omitting its own overhead would be measuring the wrong thing.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        let tables = self.segments.capacity() * size_of::<Box<[u8]>>()
            + self.large.capacity() * size_of::<Box<[u8]>>()
            + self.large_free.capacity() * size_of::<u32>()
            + self.free.capacity() * size_of::<u32>();
        self.segment_bytes + self.large_live + tables
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_arena_owns_nothing() {
        // The floor matters: four thousand shards start out empty, and even a kilobyte of free list heads each would be four megabytes before the first key.
        let arena = Arena::new();
        assert_eq!(arena.live_bytes(), 0);
        assert_eq!(arena.resident_bytes(), 0);
    }

    #[test]
    fn the_first_segment_is_small() {
        // A shard holding one key should cost about a page, not about a megabyte, because four thousand shards each hold one key before any of them holds a thousand.
        let mut arena = Arena::new();
        let _ = arena.alloc(64);
        assert!(
            arena.resident_bytes() < 2 * MIN_SEGMENT,
            "one key cost {} bytes",
            arena.resident_bytes()
        );
    }

    #[test]
    fn a_segment_is_a_sixteenth_of_what_is_already_there() {
        assert_eq!(next_segment_size(0), MIN_SEGMENT, "the first segment");
        assert_eq!(
            next_segment_size(MIN_SEGMENT),
            MIN_SEGMENT,
            "an eighth of one segment still rounds up to one segment"
        );
        assert_eq!(
            next_segment_size(GROWTH * MIN_SEGMENT),
            MIN_SEGMENT,
            "a sixteenth of sixteen segments is one, exactly at the rounding boundary"
        );
        assert_eq!(
            next_segment_size(64 * GROWTH * MIN_SEGMENT),
            64 * MIN_SEGMENT,
            "a sixteenth, once a sixteenth is worth having"
        );
        assert_eq!(
            next_segment_size(usize::MAX),
            MAX_SEGMENT,
            "the ceiling holds against anything"
        );

        // The invariants the addressing and the slab depend on, over the whole range a shard can reach.
        let mut held = 0usize;
        for step in 0..MAX_SEGMENTS {
            let size = next_segment_size(held);
            assert!((MIN_SEGMENT..=MAX_SEGMENT).contains(&size));
            assert!(
                size.is_multiple_of(GRAIN) && size / GRAIN <= OFF_MASK as usize + 1,
                "segment {step} holds more units than a reference can name"
            );
            assert!(
                size >= SMALL_MAX,
                "segment {step} cannot hold the largest allocation the slab serves, so one growth would not be enough"
            );
            held += size;
        }
        assert!(
            held > 1 << 30,
            "{held} bytes is all a shard could ever address"
        );
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a slack ratio over byte counts far under 2^52"
    )]
    fn the_growth_rule_holds_little_in_reserve() {
        // The fault this replaced: under doubling the newest segment was as large as all the earlier ones together, so an arena held nearly twice what it had handed out and the memory gate failed by a factor of two.
        let mut arena = Arena::new();
        let mut at_worst = 0.0f64;
        for len in 0..200_000usize {
            arena.alloc(64 + len % 64).unwrap();
            if arena.live_bytes() > 256 * 1024 {
                let slack = (arena.resident_bytes() - arena.live_bytes()) as f64
                    / arena.live_bytes() as f64;
                at_worst = at_worst.max(slack);
            }
        }
        assert!(
            at_worst < 0.15,
            "the arena held {:.1}% more than it had handed out",
            at_worst * 100.0
        );
    }

    #[test]
    fn an_abandoned_tail_goes_on_the_free_list() {
        // A segment's tail is too short for the request that ended it and long enough for a smaller one. Losing it would be a fixed cost per segment, which is the cost that forces segments to be large, which is the floor a four thousand shard cache cannot pay.
        let mut arena = Arena::new();
        // Allocations that do not divide the segment evenly, so a tail is left behind every time.
        for _ in 0..(MIN_SEGMENT / 24) * 4 {
            arena.alloc(24).unwrap();
        }
        assert!(
            arena.segments.len() > 1,
            "the test did not fill a single segment"
        );
        assert!(
            arena.dead_bytes() > 0,
            "no tail reached a free list, so every segment lost one"
        );
        // The tail is reusable, not merely counted. Every segment here is a [`MIN_SEGMENT`], so each tail is the same known size and an allocation of exactly that size has to come off the free list rather than out of a new segment.
        let tail = MIN_SEGMENT % 24;
        assert!(
            tail > 0 && tail.is_multiple_of(GRAIN),
            "this test's arithmetic assumed a tail of whole grains, and got {tail}"
        );
        let dead = arena.dead_bytes();
        let resident = arena.resident_bytes();
        arena.alloc(tail).unwrap();
        assert_eq!(arena.dead_bytes(), dead - tail, "the tail was not reused");
        assert_eq!(
            arena.resident_bytes(),
            resident,
            "reusing a tail should not have taken a new segment"
        );
    }

    #[test]
    fn what_goes_in_comes_back_out() {
        let mut arena = Arena::new();
        let mut refs = Vec::new();
        for len in 1..=300usize {
            let at = arena.alloc(len).unwrap();
            let payload: Vec<u8> = (0..len).map(|i| u8::try_from(i % 251).unwrap()).collect();
            arena.get_mut(at, len).copy_from_slice(&payload);
            refs.push((at, len, payload));
        }
        for (at, len, payload) in &refs {
            assert_eq!(
                arena.get(*at, *len),
                &payload[..],
                "len {len} came back wrong"
            );
        }
    }

    #[test]
    fn two_live_allocations_never_overlap() {
        // The bug this catches is an off-by-one in the bump or in the segment straddle check, which does not show up as a crash but as one key quietly overwriting another.
        let mut arena = Arena::new();
        let mut refs = Vec::new();
        for i in 0..200_000usize {
            let len = 1 + (i * 7) % 400;
            let at = arena.alloc(len).unwrap();
            arena.get_mut(at, len).fill(u8::try_from(i % 251).unwrap());
            refs.push((at, len, u8::try_from(i % 251).unwrap()));
        }
        for (at, len, byte) in refs {
            assert!(
                arena.get(at, len).iter().all(|&b| b == byte),
                "an allocation was written over"
            );
        }
    }

    #[test]
    fn a_freed_block_is_handed_out_again() {
        let mut arena = Arena::new();
        let first = arena.alloc(64).unwrap();
        arena.free(first, 64);
        let second = arena.alloc(64).unwrap();
        assert_eq!(first, second, "the free list was not consulted");
        assert_eq!(arena.dead_bytes(), 0, "the block is live again");
    }

    #[test]
    fn a_free_list_survives_being_walked() {
        // Every block of one class freed, then every one taken back. If the link written inside a free block were being clobbered this would return a duplicate or run off the end.
        let mut arena = Arena::new();
        let taken: Vec<Ref> = (0..1000).map(|_| arena.alloc(48).unwrap()).collect();
        // Measured as a difference rather than a total, because filling those segments also put their tails on other classes' free lists and those bytes are dead too.
        let tails = arena.dead_bytes();
        for at in &taken {
            arena.free(*at, 48);
        }
        assert_eq!(arena.dead_bytes() - tails, 48 * 1000);
        let mut again: Vec<Ref> = (0..1000).map(|_| arena.alloc(48).unwrap()).collect();
        again.sort_unstable();
        again.dedup();
        assert_eq!(again.len(), 1000, "the free list handed out a duplicate");
        assert_eq!(arena.dead_bytes(), tails, "the class did not empty");
    }

    #[test]
    fn a_free_list_link_survives_crossing_segments() {
        // The link is stored inside the freed block, so a free list spanning several segments is the case where a mistaken segment number would corrupt it.
        let mut arena = Arena::new();
        let taken: Vec<Ref> = (0..50_000).map(|_| arena.alloc(64).unwrap()).collect();
        let segments: std::collections::BTreeSet<usize> =
            taken.iter().map(|at| Arena::place(*at).0).collect();
        assert!(segments.len() > 4, "the test did not span enough segments");
        for at in &taken {
            arena.free(*at, 64);
        }
        let mut again: Vec<Ref> = (0..50_000).map(|_| arena.alloc(64).unwrap()).collect();
        again.sort_unstable();
        again.dedup();
        assert_eq!(again.len(), 50_000);
    }

    #[test]
    fn a_size_class_does_not_serve_a_larger_request() {
        let mut arena = Arena::new();
        let small = arena.alloc(8).unwrap();
        arena.free(small, 8);
        let large = arena.alloc(64).unwrap();
        assert_ne!(
            small, large,
            "an eight byte block served a sixty-four byte request"
        );
    }

    #[test]
    fn an_oversized_entry_is_exact_and_reusable() {
        let mut arena = Arena::new();
        let len = SMALL_MAX + 1;
        let at = arena.alloc(len).unwrap();
        assert_eq!(
            at.bits() & LARGE,
            LARGE,
            "an oversized entry stayed in the slab"
        );
        assert_eq!(arena.live_bytes(), len, "an oversized entry was rounded");
        arena.get_mut(at, len).fill(0xab);
        assert!(arena.get(at, len).iter().all(|&b| b == 0xab));

        arena.free(at, len);
        assert_eq!(arena.live_bytes(), 0);
        let again = arena.alloc(len).unwrap();
        assert_eq!(at, again, "the hole in the oversized table was not reused");
    }

    #[test]
    fn nothing_straddles_a_segment_boundary() {
        // A block that straddled would resolve into the wrong segment and read another key's bytes. The size does not divide any segment evenly, so tails are abandoned repeatedly.
        let mut arena = Arena::new();
        let len = SMALL_MAX - 8;
        for i in 0..2000usize {
            let at = arena.alloc(len).unwrap();
            let (seg, off) = Arena::place(at);
            assert!(
                off + len <= arena.segments[seg].len(),
                "allocation {i} runs past the end of segment {seg}"
            );
            arena.get_mut(at, len).fill(1);
        }
    }

    #[test]
    fn a_peek_never_runs_off_the_end() {
        // The map reads an eleven byte header prefix off entries that may be four bytes long, and the last one in a segment has nothing after it. Asking `get` for eleven there would panic.
        let mut arena = Arena::new();
        let units = MIN_SEGMENT / GRAIN;
        let mut last = Ref::NONE;
        for _ in 0..units {
            last = arena.alloc(8).unwrap();
            arena.get_mut(last, 8).fill(0xcd);
        }
        assert_eq!(
            Arena::place(last),
            (0, MIN_SEGMENT - 8),
            "the last unit was not reached"
        );
        let seen = arena.peek(last, 11);
        assert_eq!(
            seen.len(),
            8,
            "a peek should stop at the end of the segment"
        );
        assert!(seen.iter().all(|&b| b == 0xcd));
    }

    #[test]
    fn a_peek_of_an_oversized_entry_is_clamped_too() {
        let mut arena = Arena::new();
        let len = SMALL_MAX + 1;
        let at = arena.alloc(len).unwrap();
        arena.get_mut(at, len).fill(7);
        assert_eq!(arena.peek(at, 11), &[7u8; 11][..]);
        assert_eq!(arena.peek(at, usize::MAX).len(), len);
    }

    #[test]
    fn the_grain_is_the_whole_overhead() {
        // The claim the crate exists to make. A hundred byte entry costs a hundred and four bytes, not a hundred and twenty-odd, because there is no header.
        let mut arena = Arena::new();
        for _ in 0..1000 {
            let _ = arena.alloc(100);
        }
        assert_eq!(arena.live_bytes(), 104 * 1000);
    }

    #[test]
    fn the_slack_over_a_million_entries_is_small() {
        // Tails, rounding and the segment list together. Pogocache pays a malloc header of eight to sixteen bytes on each of these, which at a hundred byte entry is eight to sixteen per cent.
        let mut arena = Arena::new();
        for i in 0..1_000_000usize {
            let _ = arena.alloc(90 + i % 20);
        }
        let live = arena.live_bytes();
        let resident = arena.resident_bytes();
        assert!(
            resident * 100 < live * 103,
            "{resident} resident against {live} live is over three per cent of slack"
        );
    }

    #[test]
    fn a_reference_is_four_bytes() {
        // If this ever grows, the table slot grows with it and the memory claim goes with it.
        assert_eq!(size_of::<Ref>(), 4);
        assert_eq!(
            size_of::<Option<Ref>>(),
            8,
            "Ref::NONE exists so Option is not needed"
        );
        assert_eq!(
            1 + SEG_BITS + OFF_BITS,
            32,
            "the reference layout does not fill the word"
        );
    }
}
