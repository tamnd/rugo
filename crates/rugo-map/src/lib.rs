//! The cache: a sharded, SIMD-probed table of byte strings with expiry.
//!
//! # Shape
//!
//! Thousands of independent [`table::Table`]s, each behind its own [`lock::Lock`], picked by the top bits of the key's hash. That is the same shape pogocache has and for the same reason: a lock that thousands of keys share is a lock almost nobody waits on, and a shard small enough to fit in cache is a shard whose critical section is a few tens of nanoseconds.
//!
//! What differs is what is inside a shard. Pogocache stores a ten byte bucket entry and calls `malloc` for the entry itself; rugo stores five bytes a slot and takes entry bytes from a per-shard slab with no header at all. Both halves of that are checked by tests rather than asserted here.
//!
//! # How a hash is spent
//!
//! One hash, three disjoint jobs, so that no two of them are the same evidence twice.
//!
//! - The top bits pick the shard.
//! - Bits 32 to 38 are the control tag a probe compares sixteen at a time.
//! - The low bits pick the group within the shard.
//!
//! # Example
//!
//! ```
//! use rugo_map::Map;
//!
//! let map = Map::new(64, 0);
//! map.set(b"greeting", b"hello", None, None).unwrap();
//! assert_eq!(map.get(b"greeting", <[u8]>::to_vec), Some(b"hello".to_vec()));
//! assert!(map.remove(b"greeting"));
//! ```

pub mod clock;
mod entry;
mod group;
pub mod lock;
pub mod table;

use core::sync::atomic::{AtomicUsize, Ordering};

pub use entry::View;
pub use rugo_arena::Full;
pub use table::{Table, Wrote};

use clock::Clock;
use lock::Lock;

/// The most shards a map will make, which is also what the server defaults to.
///
/// Four thousand and ninety-six is pogocache's number, and it is a good one: at sixty-four threads it is sixty-four shards a thread, which is enough that two threads colliding on the same lock is rare, and the whole index of an empty map is still under two megabytes.
pub const MAX_SHARDS: usize = 4096;

/// One shard: a table, its lock, and the counters a reader may look at without taking it.
///
/// The counters are written only by whoever holds the lock and read by anyone, which is why they are atomics and why the write is a plain store rather than a read-modify-write. They live in the shard's own padded block, so counting the whole map touches one line per shard and no line twice.
#[derive(Debug)]
struct Shard {
    /// The table and the lock over it.
    lock: Lock<Table>,
    /// Live entries, as of the last time somebody held the lock.
    len: AtomicUsize,
    /// Bytes this shard has taken, as of the last time somebody held the lock.
    resident: AtomicUsize,
    /// Bytes this shard is charged for, as of the last time somebody held the lock.
    charged: AtomicUsize,
    /// The slot a background sweep should resume at.
    sweep_at: AtomicUsize,
}

impl Shard {
    /// Publish what the table now holds, for readers who will not take the lock.
    ///
    /// Called by whoever holds the lock, which is what makes plain stores enough: nobody else writes these, so there is no read-modify-write to lose. They live in this shard's own padded block, so the stores hit a line this core already owns.
    #[inline]
    fn publish(&self, table: &Table) {
        self.len.store(table.len(), Ordering::Relaxed);
        self.resident
            .store(table.resident_bytes(), Ordering::Relaxed);
        self.charged.store(table.charged_bytes(), Ordering::Relaxed);
    }
}

/// A sharded cache of byte strings.
#[derive(Debug)]
pub struct Map {
    /// The shards, a power of two of them.
    shards: Box<[Shard]>,
    /// How far to shift a hash to get a shard index.
    shard_shift: u32,
    /// The seed every key in this map is hashed with.
    seed: u64,
    /// The ceiling eviction defends, or zero for none.
    maxmemory: usize,
    /// One shard's share of that ceiling, or zero for none.
    ///
    /// Eviction is enforced per shard, against the lock the write already holds, rather than against a total that every write on every core would have to touch. A single shared counter read and written a million times a second is one cache line moving between every core in the machine, which costs more than the eviction it is deciding about.
    ///
    /// The price is that an unusually full shard evicts sooner than the whole cache being over would justify. With thousands of shards and a hash that spreads, the fullest shard runs within a small factor of the mean, and the shard count is what buys that.
    shard_budget: usize,
    /// The clock expiry is measured against.
    clock: Clock,
    /// Which shard the next background sweep should visit.
    sweep_shard: AtomicUsize,
}

impl Map {
    /// A map with `shards` shards, rounded up to a power of two and capped at [`MAX_SHARDS`], holding at most `maxmemory` bytes.
    ///
    /// `maxmemory` of zero means no ceiling, which is what a benchmark that is measuring memory wants: a cache that evicts is a cache whose memory number says only what it was told.
    ///
    /// The hash seed is drawn from the operating system, so two runs place keys differently and a key set built to collide against one run does not collide against the next.
    #[must_use]
    pub fn new(shards: usize, maxmemory: usize) -> Self {
        Self::with_seed(shards, maxmemory, rugo_hash::seed())
    }

    /// A map whose seed is given rather than drawn, for tests and for reproducing a report.
    #[must_use]
    pub fn with_seed(shards: usize, maxmemory: usize, seed: u64) -> Self {
        // Capped before it is rounded, because rounding `usize::MAX` up to a power of two is not a thing a `usize` can hold.
        let shards = shards.clamp(1, MAX_SHARDS).next_power_of_two();
        let shard_shift = 64 - shards.trailing_zeros();
        Self {
            shards: (0..shards)
                .map(|_| Shard {
                    lock: Lock::new(Table::new(seed)),
                    len: AtomicUsize::new(0),
                    resident: AtomicUsize::new(0),
                    charged: AtomicUsize::new(0),
                    sweep_at: AtomicUsize::new(0),
                })
                .collect(),
            shard_shift,
            seed,
            maxmemory,
            shard_budget: maxmemory / shards,
            clock: Clock::new(),
            sweep_shard: AtomicUsize::new(0),
        }
    }

    /// The clock this map measures expiry against.
    ///
    /// Public so the event loop can [`Clock::tick`] it once a turn rather than once a key.
    #[must_use]
    pub const fn clock(&self) -> &Clock {
        &self.clock
    }

    /// How many shards there are.
    #[must_use]
    pub const fn shards(&self) -> usize {
        self.shards.len()
    }

    /// The hash of `key` under this map's seed.
    ///
    /// Exposed because a caller batching commands wants to hash every key first, sort the batch by shard, and then take each lock once, which it cannot do if hashing is private.
    #[inline]
    #[must_use]
    pub fn hash(&self, key: &[u8]) -> u64 {
        rugo_hash::hash(key, self.seed)
    }

    /// Which shard a hash belongs to.
    ///
    /// The top bits, because the tag takes the middle and the group index takes the bottom. Two of those three drawn from the same bits would be one piece of evidence counted twice.
    #[inline]
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the shift leaves at most MAX_SHARDS - 1, which is twelve bits"
    )]
    pub const fn shard_of(&self, hash: u64) -> usize {
        // A one shard map would shift by sixty-four, which is not a shift.
        if self.shards.len() == 1 {
            0
        } else {
            (hash >> self.shard_shift) as usize
        }
    }

    /// Run `each` against the table `hash` belongs to, holding its lock.
    ///
    /// This is the whole of the map's concurrency: everything else here is written in terms of it, and so is anything the server needs that is not here, such as an increment that has to read and write under one lock.
    ///
    /// `each` cannot return anything borrowed from the table, because the lock is released before this returns. That is the constraint that makes every operation copy what it needs while it still holds the lock, which is what a caller should be doing anyway.
    pub fn with_key<R>(&self, hash: u64, each: impl FnOnce(&mut Table, u32) -> R) -> R {
        let now = self.clock.now();
        let shard = &self.shards[self.shard_of(hash)];
        let mut table = shard.lock.lock();
        let out = each(&mut table, now);
        shard.publish(&table);
        out
    }

    /// The value under `key`, passed to `each` while the shard is still locked.
    ///
    /// A closure rather than a returned `Vec` because the caller is a server about to copy these bytes into a write buffer, and an owned copy in between would be a copy nobody asked for.
    pub fn get<R>(&self, key: &[u8], each: impl FnOnce(&[u8]) -> R) -> Option<R> {
        let hash = self.hash(key);
        self.with_key(hash, |table, now| table.get(key, hash, now).map(each))
    }

    /// The whole entry under `key`, passed to `each` while the shard is still locked.
    pub fn view<R>(&self, key: &[u8], each: impl FnOnce(View<'_>) -> R) -> Option<R> {
        let hash = self.hash(key);
        self.with_key(hash, |table, now| table.view(key, hash, now).map(each))
    }

    /// Whether `key` is present and unexpired.
    pub fn contains(&self, key: &[u8]) -> bool {
        let hash = self.hash(key);
        self.with_key(hash, |table, now| table.contains(key, hash, now))
    }

    /// Store `value` under `key`, expiring at `expiry` seconds since the Unix epoch.
    ///
    /// # Errors
    ///
    /// [`Full`] when the shard cannot take the entry, which means the process is out of memory. Being over `maxmemory` is not this error: that is what eviction is for, and it happens after the write rather than instead of it.
    pub fn set(
        &self,
        key: &[u8],
        value: &[u8],
        expiry: Option<u32>,
        user_flags: Option<u32>,
    ) -> Result<Wrote, Full> {
        let hash = self.hash(key);
        let budget = self.shard_budget;
        self.with_key(hash, |table, _| {
            let wrote = table.set(key, value, hash, expiry, user_flags)?;
            evict_to_fit(table, budget, hash);
            Ok(wrote)
        })
    }

    /// Remove `key`, reporting whether it was there.
    pub fn remove(&self, key: &[u8]) -> bool {
        let hash = self.hash(key);
        self.with_key(hash, |table, _| table.remove(key, hash))
    }

    /// Live entries across every shard.
    ///
    /// Read without taking a lock, so under concurrent writes it is a count that was true recently rather than one that is true now. `DBSIZE` on a cache being written to has no other kind of answer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.len.load(Ordering::Relaxed))
            .sum()
    }

    /// Whether every shard is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes every shard has taken, summed exactly, one lock-free read a shard.
    ///
    /// This is the memory gate's numerator. It is the shards' own accounting rather than the process's resident set, so it excludes the connection buffers and the binary itself, and a report that wants those has to read them from the operating system.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.resident.load(Ordering::Relaxed))
            .sum()
    }

    /// Bytes the map is charged for: the index and the entries, and nothing a later write can reuse.
    ///
    /// This is what `maxmemory` is compared against, and it is smaller than [`Map::resident_bytes`] by the slack the arena is holding for reuse. Publishing both is the honest thing to do, in the same way that Redis publishes its used memory and its resident set side by side rather than choosing whichever is flattering.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.charged.load(Ordering::Relaxed))
            .sum()
    }

    /// Bytes of key and value handed out, without any index or slack.
    #[must_use]
    pub fn live_bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.lock.lock().live_bytes())
            .sum()
    }

    /// Bytes every shard's index costs, at five per slot.
    ///
    /// Separate from [`Map::live_bytes`] because the two are controlled by different things and regress for different reasons: the index by the load factor and the growth rule, the entries by the encoding and the allocation grain. A memory gate that only saw their sum could not say which had moved.
    ///
    /// Takes every shard's lock in turn, so this is for a report or a test rather than for a serving path.
    #[must_use]
    pub fn index_bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.lock.lock().index_bytes())
            .sum()
    }

    /// Drop everything.
    pub fn clear(&self) {
        for shard in &self.shards {
            let mut table = shard.lock.lock();
            table.clear();
            table.shrink();
            shard.publish(&table);
        }
    }

    /// Visit at most `budget` slots looking for expired entries, resuming where the last call stopped.
    ///
    /// Returns how many went. Meant to be called from an idle event loop turn: expiry is already checked on read, so this is only what reclaims a key nobody ever asks for again.
    pub fn sweep(&self, budget: usize) -> usize {
        let now = self.clock.now();
        let index = self.sweep_shard.fetch_add(1, Ordering::Relaxed) % self.shards.len();
        let shard = &self.shards[index];
        let from = shard.sweep_at.load(Ordering::Relaxed);

        let mut table = shard.lock.lock();
        let (next, removed) = table.sweep(from, budget, now);
        if removed > 0 {
            table.shrink();
        }
        shard.publish(&table);
        shard.sweep_at.store(next, Ordering::Relaxed);
        removed
    }

    /// The ceiling this map defends, or zero if it has none.
    #[must_use]
    pub const fn maxmemory(&self) -> usize {
        self.maxmemory
    }
}

/// Evict from `table` until it fits in `budget`, seeding the sampler from `hash`.
///
/// Bounded, because this runs on the thread that just served a `SET` and holds the shard's lock while it does. A client waiting on a reply should not pay for the whole overshoot, and a shard should not be held while somebody else's key waits. Whatever is left over is taken by the next write, and the one after that if it has to be.
///
/// The sampler is seeded from the key's own hash and stepped locally, so it needs no shared state at all. Eviction wants numbers that are spread, not numbers that are unguessable, and a shared generator would be exactly the contended cache line that per-shard accounting exists to avoid.
fn evict_to_fit(table: &mut Table, budget: usize, hash: u64) {
    if budget == 0 || table.charged_bytes() <= budget {
        return;
    }

    let mut roll = hash | 1;
    let mut went = 0;
    while table.charged_bytes() > budget && went < 64 {
        roll ^= roll << 13;
        roll ^= roll >> 7;
        roll ^= roll << 17;
        if !table.evict_one(roll) {
            break;
        }
        went += 1;
    }
    if went > 0 {
        table.shrink();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    const SEED: u64 = 0x5eed;

    fn map() -> Map {
        Map::with_seed(64, 0, SEED)
    }

    fn get(map: &Map, key: &[u8]) -> Option<Vec<u8>> {
        map.get(key, <[u8]>::to_vec)
    }

    #[test]
    fn what_goes_in_comes_out() {
        let map = map();
        for i in 0..50_000u32 {
            map.set(
                format!("k{i}").as_bytes(),
                format!("v{i}").as_bytes(),
                None,
                None,
            )
            .unwrap();
        }
        assert_eq!(map.len(), 50_000);
        for i in 0..50_000u32 {
            assert_eq!(
                get(&map, format!("k{i}").as_bytes()).as_deref(),
                Some(format!("v{i}").as_bytes())
            );
        }
    }

    #[test]
    fn the_shards_get_roughly_equal_shares() {
        // A shard index drawn from bits that the tag or the group index also uses would pile keys into a few shards, and the lock nobody waits on would become the lock everybody waits on.
        let map = map();
        for i in 0..64_000u32 {
            map.set(format!("memtier-{i}").as_bytes(), b"v", None, None)
                .unwrap();
        }
        let counts: Vec<usize> = map
            .shards
            .iter()
            .map(|shard| shard.len.load(Ordering::Relaxed))
            .collect();
        let mean = 64_000 / map.shards();
        let worst = counts.iter().copied().max().unwrap_or(0);
        let least = counts.iter().copied().min().unwrap_or(0);
        assert!(
            worst < mean * 2,
            "the fullest shard holds {worst} against a mean of {mean}"
        );
        assert!(
            least > mean / 2,
            "the emptiest shard holds {least} against a mean of {mean}"
        );
    }

    #[test]
    fn a_one_shard_map_works() {
        // The shift would be sixty-four, which is not a shift, so this is the case that would panic rather than merely misbehave.
        let map = Map::with_seed(1, 0, SEED);
        assert_eq!(map.shards(), 1);
        for i in 0..1000u32 {
            map.set(format!("k{i}").as_bytes(), b"v", None, None)
                .unwrap();
        }
        assert_eq!(map.len(), 1000);
        assert_eq!(get(&map, b"k7").as_deref(), Some(&b"v"[..]));
    }

    #[test]
    fn a_shard_count_is_rounded_and_capped() {
        assert_eq!(Map::with_seed(100, 0, SEED).shards(), 128);
        assert_eq!(Map::with_seed(0, 0, SEED).shards(), 1);
        assert_eq!(Map::with_seed(usize::MAX, 0, SEED).shards(), MAX_SHARDS);
    }

    #[test]
    fn expiry_is_measured_against_the_map_clock() {
        let map = map();
        let now = map.clock().now();
        map.set(b"soon", b"v", Some(now + 1), None).unwrap();
        map.set(b"never", b"v", None, None).unwrap();
        assert_eq!(get(&map, b"soon").as_deref(), Some(&b"v"[..]));

        map.clock().advance(2);
        assert_eq!(get(&map, b"soon"), None, "an expired key is still readable");
        assert_eq!(get(&map, b"never").as_deref(), Some(&b"v"[..]));
        assert_eq!(map.len(), 1, "the expired key was not reclaimed");
    }

    #[test]
    fn a_sweep_reclaims_what_nobody_reads() {
        let map = Map::with_seed(8, 0, SEED);
        let now = map.clock().now();
        for i in 0..4000u32 {
            map.set(format!("k{i}").as_bytes(), b"v", Some(now + 1), None)
                .unwrap();
        }
        map.clock().advance(2);
        assert_eq!(map.len(), 4000, "expiry alone should not reclaim anything");

        let mut removed = 0;
        for _ in 0..4000 {
            removed += map.sweep(64);
            if map.is_empty() {
                break;
            }
        }
        assert_eq!(removed, 4000);
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn a_ceiling_is_defended() {
        // Half a megabyte of ceiling against several megabytes of writes. The exact resting point is not the claim; not growing without bound is.
        let ceiling = 512 * 1024;
        let map = Map::with_seed(16, ceiling, SEED);
        for i in 0..100_000u32 {
            map.set(format!("k{i}").as_bytes(), &[0u8; 64], None, None)
                .unwrap();
        }
        assert!(!map.is_empty(), "eviction emptied the cache");
        assert!(
            map.resident_bytes() < ceiling * 4,
            "{} bytes resident against a ceiling of {ceiling}",
            map.resident_bytes()
        );
    }

    #[test]
    fn no_ceiling_means_no_eviction() {
        let map = map();
        for i in 0..20_000u32 {
            map.set(format!("k{i}").as_bytes(), &[0u8; 256], None, None)
                .unwrap();
        }
        assert_eq!(
            map.len(),
            20_000,
            "something was evicted from an unlimited map"
        );
    }

    #[test]
    fn clearing_empties_every_shard() {
        let map = map();
        for i in 0..10_000u32 {
            map.set(format!("k{i}").as_bytes(), b"v", None, None)
                .unwrap();
        }
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(get(&map, b"k5"), None);
        map.set(b"again", b"v", None, None).unwrap();
        assert_eq!(get(&map, b"again").as_deref(), Some(&b"v"[..]));
    }

    #[test]
    fn the_accounting_adds_up() {
        let map = map();
        for i in 0..20_000u32 {
            map.set(format!("k{i}").as_bytes(), &[0u8; 100], None, None)
                .unwrap();
        }
        let resident = map.resident_bytes();
        let live = map.live_bytes();
        assert!(
            live > 20_000 * 100,
            "the live bytes do not cover the values"
        );
        assert!(resident > live, "resident memory should include the index");
        assert!(
            resident < live * 3,
            "{resident} resident against {live} live is more slack than the design allows"
        );
    }

    #[test]
    fn eight_threads_do_not_lose_a_key() {
        // Each thread owns a disjoint key range, so every key must survive. A lock that let two writers into one shard would show up as a missing key or a corrupt value rather than as a crash.
        let map = Arc::new(map());
        let threads: Vec<_> = (0..8u32)
            .map(|t| {
                let map = Arc::clone(&map);
                thread::spawn(move || {
                    for i in 0..20_000u32 {
                        let key = format!("t{t}:k{i}");
                        map.set(key.as_bytes(), key.as_bytes(), None, None).unwrap();
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }

        assert_eq!(map.len(), 8 * 20_000);
        for t in 0..8u32 {
            for i in (0..20_000u32).step_by(97) {
                let key = format!("t{t}:k{i}");
                assert_eq!(
                    get(&map, key.as_bytes()).as_deref(),
                    Some(key.as_bytes()),
                    "{key} was lost"
                );
            }
        }
    }

    #[test]
    fn readers_and_writers_together_see_whole_values() {
        // A reader that saw a half-written entry would see a value whose bytes do not match its key. Nothing here checks timing; it checks that no torn value is ever observed.
        let map = Arc::new(map());
        for i in 0..2000u32 {
            let key = format!("k{i}");
            map.set(key.as_bytes(), key.as_bytes(), None, None).unwrap();
        }

        let threads: Vec<_> = (0..8u32)
            .map(|t| {
                let map = Arc::clone(&map);
                thread::spawn(move || {
                    for round in 0..5000u32 {
                        let i = (round * 7 + t) % 2000;
                        let key = format!("k{i}");
                        if t % 2 == 0 {
                            map.set(key.as_bytes(), key.as_bytes(), None, None).unwrap();
                        } else if let Some(value) = get(&map, key.as_bytes()) {
                            assert_eq!(value, key.as_bytes(), "a value was torn");
                        }
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    }

    #[test]
    fn an_empty_map_is_nearly_free() {
        // Four thousand shards that hold nothing should cost their structs and no allocation, or every small deployment pays for shards it never fills.
        let map = Map::with_seed(MAX_SHARDS, 0, SEED);
        assert_eq!(map.len(), 0);
        assert_eq!(
            map.resident_bytes(),
            0,
            "an untouched shard allocated something"
        );
    }
}
