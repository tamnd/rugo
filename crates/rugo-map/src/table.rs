//! One shard's table: control bytes, slots, and the arena the entries live in.
//!
//! # Why five bytes a slot
//!
//! A slot is one control byte and one four byte [`Ref`]. Pogocache's bucket entry is ten: one byte of distance-to-bucket, three of hash, and six of pointer. At the same occupancy that is half the index, and the index is the part of a cache that is pure overhead, because it is the part that holds no key and no value.
//!
//! The three bytes of stored hash are what pay for the six byte pointer there and are not needed here: the control byte already filters, and confirming a match by comparing the key is a read the entry was about to do anyway. The pointer shrinks to four bytes because entries are arena offsets rather than addresses, which is [`rugo_arena`]'s job.
//!
//! # Probing
//!
//! Groups are aligned, so a group never straddles the end of the control array and the array needs no mirrored tail. The sequence over groups is triangular, stepping by one group then two then three, which visits every group exactly once when the count is a power of two. A probe therefore either finds the key or reaches an empty slot, and it cannot run forever while the load factor is under one.

use crate::entry::{self, Head, View};
use crate::group::{DELETED, EMPTY, Group, WIDTH, is_full, prefetch, tag_of};
use rugo_arena::{Arena, Full, Ref};

/// The smallest table, in slots.
///
/// One group's worth on every implementation, including the eight-lane fallback, so that probing never has a special case for a table smaller than a group.
const MIN_CAPACITY: usize = 16;

/// The longest entry header: one flags byte and two five byte varints.
const MAX_HEADER: usize = 11;

/// Occupancy at which the table grows, as a fraction: seven eighths.
///
/// Higher wastes less memory and probes longer. Seven eighths is where a group of sixteen still finds a free lane on nearly every first probe, so the extra occupancy costs almost nothing in time and saves an eighth of the index.
const LOAD_NUM: usize = 7;
/// The denominator of [`LOAD_NUM`].
const LOAD_DEN: usize = 8;

/// What a write did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wrote {
    /// The key was not there and now is.
    Inserted,
    /// The key was there and its value was replaced.
    Replaced,
}

/// One shard's table.
#[derive(Debug)]
pub struct Table {
    /// One byte per slot: a tag, [`EMPTY`] or [`DELETED`].
    ctrl: Box<[u8]>,
    /// One [`Ref`] per slot, meaningful only where the control byte is a tag.
    slots: Box<[u32]>,
    /// Where the entry bytes live.
    arena: Arena,
    /// Live entries.
    len: usize,
    /// Slots holding [`DELETED`].
    tombstones: usize,
    /// Entries carrying an expiry.
    ///
    /// Kept so that [`Table::sweep`] can tell, without reading anything, that there is nothing here to expire. The sweep walks slot numbers and asks each full one when it expires, and the only place that answer is written is the entry itself, so asking costs a read of the arena at a random offset. On a shard where nothing was given a TTL every one of those reads returns the same nothing, and there are a couple of hundred of them per call.
    ///
    /// Counting is exact rather than a hint, so a shard that does hold a timed entry is swept exactly as before. Every site that can change the answer already has the entry's header in hand for its own reasons, so the count is maintained without reading anything extra.
    timed: usize,
    /// Inserts remaining before the table has to be rebuilt.
    growth_left: usize,
    /// The hash seed every key in this table was placed with.
    seed: u64,
}

impl Table {
    /// An empty table that owns no slots and no arena.
    ///
    /// Nothing is allocated until the first insert, because a shard nobody has written to should cost the struct and no more. With four thousand shards, allocating even a minimum table eagerly would be control bytes and slots for sixty-five thousand entries that do not exist.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            ctrl: Box::new([]),
            slots: Box::new([]),
            arena: Arena::new(),
            len: 0,
            tombstones: 0,
            timed: 0,
            growth_left: 0,
            seed,
        }
    }

    /// Live entries, counting any that have expired but not yet been noticed.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the table holds nothing.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Slots the table has room for.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Bytes the index costs: one control byte and one four byte slot each.
    #[must_use]
    pub const fn index_bytes(&self) -> usize {
        self.ctrl.len() + self.slots.len() * 4
    }

    /// Every byte this table has taken from the operating system.
    ///
    /// The index, the arena, and each of their own bookkeeping tables. This is what the memory gate divides by [`Table::len`], so it counts everything rather than only the parts that flatter it.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.index_bytes() + self.arena.resident_bytes()
    }

    /// Entry bytes currently handed out, without the index or any slack.
    #[must_use]
    pub const fn live_bytes(&self) -> usize {
        self.arena.live_bytes()
    }

    /// What a memory ceiling is enforced against: the index plus the entries, and nothing that a later write can reuse.
    ///
    /// Not [`Table::resident_bytes`], because a freed entry's bytes go back on a free list and an emptied segment stays where it is. A ceiling measured against resident memory could never be met once the segments existed, so eviction would empty the whole cache trying to meet it and then keep trying. This is the same distinction Redis draws between the memory it has used and the memory the operating system has given it, and the gap between the two is reported rather than hidden.
    #[must_use]
    pub const fn charged_bytes(&self) -> usize {
        self.index_bytes() + self.arena.live_bytes()
    }

    /// How many groups the control array holds.
    #[inline]
    pub(crate) const fn groups(&self) -> usize {
        self.slots.len() / WIDTH
    }

    /// Where the control bytes begin.
    ///
    /// Handed out so that a shard can publish it for a reader who will not take the lock and only wants to name a cache line, not read one. Nothing may dereference this: the table it belongs to can rebuild and free it at any moment after the lock is released, and the only use that survives that is a prefetch, which is a hint about an address rather than a read of it.
    #[inline]
    pub(crate) fn ctrl_ptr(&self) -> *const u8 {
        self.ctrl.as_ptr()
    }

    /// Read the group starting at slot `at`, which must be a multiple of [`WIDTH`].
    #[inline]
    fn group_at(&self, at: usize) -> Group {
        debug_assert!(at.is_multiple_of(WIDTH) && at + WIDTH <= self.ctrl.len());
        // SAFETY: `at` is group aligned and below the capacity, and the capacity is a multiple of WIDTH, so WIDTH bytes from `at` are inside `ctrl`.
        unsafe { Group::load(self.ctrl.as_ptr().add(at)) }
    }

    /// The header of the entry at `at`.
    ///
    /// The arena does not record lengths, which is how it avoids a per-entry header, so the length comes from the entry's own header. That header is the single copy of the fact, which is what makes the arena's not keeping a second one free rather than a trade.
    #[inline]
    fn head_at(&self, at: Ref) -> Head {
        entry::head(self.arena.peek(at, MAX_HEADER))
    }

    /// The header of the entry at `at`, and the bytes it describes.
    ///
    /// Every read of an entry comes through here, so that a header is decoded once per operation rather than once per question asked about it. The two arena lookups are the price of not letting a slice run past the end of the allocation: the first bounds the header, the header says how long the entry is, and the second bounds the entry.
    #[inline]
    fn entry_at(&self, at: Ref) -> (Head, &[u8]) {
        let head = self.head_at(at);
        (head, self.arena.get(at, head.size()))
    }

    /// When the entry at `at` expires, if it does.
    ///
    /// Takes a header the caller has already decoded, and stops at the flags byte when there is no expiry, which is what keeps the ordinary entry from paying a second arena lookup to be told it has none.
    #[inline]
    fn expiry_of(&self, at: Ref, head: Head) -> Option<u32> {
        if head.flags & entry::HAS_EXPIRY == 0 {
            return None;
        }
        head.expiry(self.arena.get(at, head.size()))
    }

    /// Record that an entry which did or did not carry an expiry now does or does not.
    ///
    /// One place rather than five, because a count that is wrong in one direction makes the sweep skip a shard that has work in it, and an expired key that is never swept is a memory leak that nothing else in the map would notice.
    #[inline]
    fn retime(&mut self, was: bool, now: bool) {
        if was && !now {
            self.timed -= 1;
        } else if !was && now {
            self.timed += 1;
        }
    }

    /// Find the slot holding `key`, if any.
    ///
    /// The probe stops at the first group containing an empty slot, because a key placed later in this sequence would have taken that slot instead of running past it.
    #[inline]
    fn find(&self, key: &[u8], hash: u64) -> Option<(usize, Head)> {
        if self.slots.is_empty() {
            return None;
        }
        let tag = tag_of(hash);
        let mask = self.groups() - 1;
        let mut group = group_of(hash, mask);
        let mut step = 1usize;

        loop {
            let base = group * WIDTH;
            // The two loads this group needs are in unrelated allocations and at ten million keys both of them miss, but only the first can start on its own: the lane a slot is read from comes out of the tag comparison, so the slot load cannot issue until the control bytes have arrived. Asking for the slots here rather than waiting to be told which one costs a hint and lets the second miss run underneath the first, which turns three chained misses on the way to a key into two.
            //
            // The whole group of slots is asked for at once, at `base`, because the address is known before the lane is. Sixteen slots are sixteen four byte references, so the group is a cache line, and any lane that matches is in the line the hint asked for or the one after it.
            prefetch(self.slots.as_ptr().wrapping_add(base).cast());
            let probe = self.group_at(base);
            for lane in probe.match_byte(tag) {
                let at = base + lane;
                let (head, bytes) = self.entry_at(Ref::from_bits(self.slots[at]));
                if head.key(bytes) == key {
                    return Some((at, head));
                }
            }
            if probe.match_empty().any() {
                return None;
            }
            group = (group + step) & mask;
            step += 1;
        }
    }

    /// The first slot in `hash`'s probe sequence a new entry may be placed in.
    ///
    /// Only called once the key is known to be absent, which is what makes the earliest free slot the right one: nothing further along the sequence can be this key.
    #[inline]
    fn free_slot(&self, hash: u64) -> usize {
        let mask = self.groups() - 1;
        let mut group = group_of(hash, mask);
        let mut step = 1usize;
        loop {
            let base = group * WIDTH;
            if let Some(lane) = self.group_at(base).match_free().lowest() {
                return base + lane;
            }
            group = (group + step) & mask;
            step += 1;
        }
    }

    /// Whether the entry in slot `at` has expired by `now`.
    ///
    /// For [`Table::sweep`], which walks slot numbers and has no header in hand. Everything on a lookup path already has one and uses [`Table::expiry_of`] instead.
    #[inline]
    fn is_expired(&self, at: usize, now: u32) -> bool {
        let entry = Ref::from_bits(self.slots[at]);
        let head = self.head_at(entry);
        self.expiry_of(entry, head).is_some_and(|when| when <= now)
    }

    /// The value stored under `key`, if it is there and has not expired.
    ///
    /// `now` is the clock the expiry was written against. An entry found to have expired is removed here rather than merely hidden, so a key read once after it expires does not go on costing memory until a sweep reaches it.
    pub fn get(&mut self, key: &[u8], hash: u64, now: u32) -> Option<&[u8]> {
        let (at, head) = self.find(key, hash)?;
        let entry = Ref::from_bits(self.slots[at]);
        if self.expiry_of(entry, head).is_some_and(|when| when <= now) {
            self.erase(at);
            return None;
        }
        Some(head.value(self.arena.get(entry, head.size())))
    }

    /// The whole entry under `key`, for the commands that want more than the value.
    pub fn view(&mut self, key: &[u8], hash: u64, now: u32) -> Option<View<'_>> {
        let (at, head) = self.find(key, hash)?;
        let entry = Ref::from_bits(self.slots[at]);
        if self.expiry_of(entry, head).is_some_and(|when| when <= now) {
            self.erase(at);
            return None;
        }
        Some(head.view(self.arena.get(entry, head.size())))
    }

    /// Whether `key` is present and unexpired.
    pub fn contains(&mut self, key: &[u8], hash: u64, now: u32) -> bool {
        let Some((at, head)) = self.find(key, hash) else {
            return false;
        };
        let entry = Ref::from_bits(self.slots[at]);
        if self.expiry_of(entry, head).is_some_and(|when| when <= now) {
            self.erase(at);
            return false;
        }
        true
    }

    /// Store `value` under `key`.
    ///
    /// # Errors
    ///
    /// [`Full`] when the shard's arena cannot take the entry, which at sixteen gigabytes in one shard of thousands means the process is out of memory rather than that the key was rejected.
    pub fn set(
        &mut self,
        key: &[u8],
        value: &[u8],
        hash: u64,
        expiry: Option<u32>,
        user_flags: Option<u32>,
    ) -> Result<Wrote, Full> {
        let size = entry::size_of_entry(
            key.len(),
            value.len(),
            expiry.is_some(),
            user_flags.is_some(),
        );

        if let Some((at, head)) = self.find(key, hash) {
            let old = Ref::from_bits(self.slots[at]);
            let old_size = head.size();

            let was_timed = head.flags & entry::HAS_EXPIRY != 0;

            // Setting a key to a value of the length it already had is what a benchmark does over and over, and taking it in place skips a free and an allocation entirely.
            if old_size == size {
                entry::write(
                    self.arena.get_mut(old, size),
                    key,
                    value,
                    expiry,
                    user_flags,
                );
                self.retime(was_timed, expiry.is_some());
                return Ok(Wrote::Replaced);
            }

            // The new block is taken before the old one is given back, so a refusal leaves the old value in place rather than losing both.
            let new = self.arena.alloc(size)?;
            entry::write(
                self.arena.get_mut(new, size),
                key,
                value,
                expiry,
                user_flags,
            );
            self.arena.free(old, old_size);
            self.slots[at] = new.bits();
            self.retime(was_timed, expiry.is_some());
            return Ok(Wrote::Replaced);
        }

        if self.growth_left == 0 {
            self.reserve();
        }

        let new = self.arena.alloc(size)?;
        entry::write(
            self.arena.get_mut(new, size),
            key,
            value,
            expiry,
            user_flags,
        );

        let at = self.free_slot(hash);
        // Taking a tombstone does not spend growth, because the slot was already counted as used when it became one.
        if self.ctrl[at] == DELETED {
            self.tombstones -= 1;
        } else {
            self.growth_left -= 1;
        }
        self.ctrl[at] = tag_of(hash);
        self.slots[at] = new.bits();
        self.len += 1;
        self.retime(false, expiry.is_some());
        Ok(Wrote::Inserted)
    }

    /// Remove `key`, reporting whether it was there.
    pub fn remove(&mut self, key: &[u8], hash: u64) -> bool {
        let Some((at, _)) = self.find(key, hash) else {
            return false;
        };
        self.erase(at);
        true
    }

    /// Change when `key` expires, reporting whether the key was there to change.
    ///
    /// `None` clears the expiry, which is `PERSIST`. A key found to have already expired is removed and reported absent, the same as any other read of it would do.
    ///
    /// # Errors
    ///
    /// [`Full`] when the entry has to be rewritten at a new width and the shard cannot take it. The old entry is left alone in that case, so the key keeps the expiry it had rather than losing both.
    pub fn expire(
        &mut self,
        key: &[u8],
        hash: u64,
        expiry: Option<u32>,
        now: u32,
    ) -> Result<bool, Full> {
        let Some((at, head)) = self.find(key, hash) else {
            return Ok(false);
        };
        let entry = Ref::from_bits(self.slots[at]);
        if self.expiry_of(entry, head).is_some_and(|when| when <= now) {
            self.erase(at);
            return Ok(false);
        }

        // Read before the rewrite branch below shadows `head` with the copy it decodes out of the entry it is moving.
        let was_timed = head.flags & entry::HAS_EXPIRY != 0;

        match (was_timed, expiry) {
            // The field is already there and is the same width either way, so this is four bytes written over four bytes and nothing else moves.
            (true, Some(when)) => {
                let size = head.size();
                let word = head.header + head.klen + head.vlen;
                self.arena.get_mut(entry, size)[word..word + 4]
                    .copy_from_slice(&when.to_le_bytes());
            }
            // Persisting a key that was never going to expire.
            (false, None) => {}
            // The trailer changes width, so the entry is rewritten somewhere it fits. `EXPIRE` and `PERSIST` are not on any hot path, and paying a copy here is what keeps an entry with no expiry from carrying four bytes of room to grow into.
            _ => {
                let old = head.size();
                let bytes = self.arena.get(entry, old).to_vec();
                let head = entry::head(&bytes);
                let user_flags = head.user_flags(&bytes);
                let size = entry::size_of_entry(
                    head.klen,
                    head.vlen,
                    expiry.is_some(),
                    user_flags.is_some(),
                );

                let new = self.arena.alloc(size)?;
                entry::write(
                    self.arena.get_mut(new, size),
                    head.key(&bytes),
                    head.value(&bytes),
                    expiry,
                    user_flags,
                );
                self.arena.free(entry, old);
                self.slots[at] = new.bits();
                self.retime(was_timed, expiry.is_some());
            }
        }
        Ok(true)
    }

    /// Drop the entry in slot `at`, leaving the slot reusable.
    ///
    /// A slot becomes [`EMPTY`] only when its group already holds an empty lane, because in that case no probe sequence ever ran past this slot and none will now. Otherwise it becomes a tombstone, which later probes have to walk over. Getting this backwards does not make the table slow, it makes unrelated keys unreachable, which is why it is the one place here that tests a condition rather than storing unconditionally.
    fn erase(&mut self, at: usize) {
        let entry = Ref::from_bits(self.slots[at]);
        // The header is decoded here anyway, to learn how much arena to give back, so the expiry flag comes along for nothing.
        let head = self.head_at(entry);
        let size = head.size();
        let was_timed = head.flags & entry::HAS_EXPIRY != 0;
        let base = at - (at % WIDTH);

        if self.group_at(base).match_empty().any() {
            self.ctrl[at] = EMPTY;
            self.growth_left += 1;
        } else {
            self.ctrl[at] = DELETED;
            self.tombstones += 1;
        }
        self.slots[at] = Ref::NONE.bits();
        self.arena.free(entry, size);
        self.len -= 1;
        self.retime(was_timed, false);
    }

    /// Make room for at least one more insert.
    ///
    /// A table whose free budget went mostly to tombstones is rebuilt at the same size rather than doubled, so a workload of matched inserts and deletes settles instead of growing without bound. Either way the rebuild leaves room: rebuilding in place happens only when tombstones outnumber entries, which puts the live count under half the budget, and doubling puts it under half by construction.
    fn reserve(&mut self) {
        let wanted = if self.slots.is_empty() {
            MIN_CAPACITY
        } else if self.tombstones >= self.len {
            self.slots.len()
        } else {
            self.slots.len() * 2
        };
        self.rehash(wanted);
        debug_assert!(self.growth_left > 0);
    }

    /// Give back index slots the table has stopped needing, reporting whether it did.
    ///
    /// Not done from inside `erase`, which is where it would be tidiest, because rehashing moves every entry and `sweep` walks slot numbers across successive erases. A shrink under that walk would leave it reading positions that no longer mean what it thought. So the caller asks for this once, after it has finished erasing.
    ///
    /// The target leaves the table at half full rather than at its load factor, so that the next few inserts do not immediately rebuild what this just tore down.
    pub fn shrink(&mut self) -> bool {
        if self.slots.len() <= MIN_CAPACITY {
            return false;
        }
        let wanted = (self.len * 2).next_power_of_two().max(MIN_CAPACITY);
        if wanted >= self.slots.len() {
            return false;
        }
        self.rehash(wanted);
        true
    }

    /// Rebuild the table at `capacity` slots, which must be a power of two of at least [`MIN_CAPACITY`].
    fn rehash(&mut self, capacity: usize) {
        debug_assert!(capacity.is_power_of_two() && capacity >= MIN_CAPACITY);

        let mut ctrl = vec![EMPTY; capacity].into_boxed_slice();
        let mut slots = vec![Ref::NONE.bits(); capacity].into_boxed_slice();
        let mask = capacity / WIDTH - 1;

        for at in 0..self.slots.len() {
            if !is_full(self.ctrl[at]) {
                continue;
            }
            let entry = Ref::from_bits(self.slots[at]);
            let (head, bytes) = self.entry_at(entry);
            let hash = rugo_hash::hash(head.key(bytes), self.seed);

            // The same aligned triangular walk as `free_slot`, over the arrays being built. It cannot call that method because these arrays are not installed yet, and every slot in them is either empty or freshly taken, so there are no tombstones to consider.
            let mut group = group_of(hash, mask);
            let mut step = 1usize;
            let put = loop {
                let base = group * WIDTH;
                // SAFETY: `base` is group aligned and below `capacity`, which is a multiple of WIDTH, so WIDTH bytes from it are inside `ctrl`.
                let probe = unsafe { Group::load(ctrl.as_ptr().add(base)) };
                if let Some(lane) = probe.match_empty().lowest() {
                    break base + lane;
                }
                group = (group + step) & mask;
                step += 1;
            };
            ctrl[put] = tag_of(hash);
            slots[put] = entry.bits();
        }

        self.ctrl = ctrl;
        self.slots = slots;
        self.tombstones = 0;
        self.growth_left = capacity * LOAD_NUM / LOAD_DEN - self.len;
    }

    /// Drop every entry, keeping the slots.
    pub fn clear(&mut self) {
        for at in 0..self.slots.len() {
            if is_full(self.ctrl[at]) {
                let entry = Ref::from_bits(self.slots[at]);
                let size = self.head_at(entry).size();
                self.arena.free(entry, size);
            }
            self.ctrl[at] = EMPTY;
            self.slots[at] = Ref::NONE.bits();
        }
        self.len = 0;
        self.tombstones = 0;
        self.timed = 0;
        self.growth_left = self.slots.len() * LOAD_NUM / LOAD_DEN;
    }

    /// Remove entries that expired at or before `now`, visiting at most `budget` slots from `from`.
    ///
    /// Returns where to resume and how many went. Bounded rather than exhaustive so a caller can sweep a large shard in pieces without holding its lock for the whole of it, which is the difference between a background sweep and a stall.
    pub fn sweep(&mut self, from: usize, budget: usize, now: u32) -> (usize, usize) {
        // Nothing in this shard was given a TTL, so there is nothing here that can have expired. The walk below would read the header of every full slot it passed to be told so one entry at a time, and those headers are at unrelated offsets in the arena, so the reading is what the call costs rather than the walking. Measured on `server3` with user space counters, one thread, pipeline depth twenty-five over a unix socket, five million keys of one to a thousand bytes none of which had an expiry and every lookup hitting, the walk is eighteen instructions per operation out of twelve hundred. What that is worth in cycles is inside the spread of that box and no figure is claimed for it.
        //
        // The position is returned unmoved. There was nothing to find here, so resuming where this left off loses nothing, and advancing it would mean a shard that later gains a TTL starts its first real sweep partway through.
        if self.timed == 0 {
            return (from, 0);
        }
        if self.slots.is_empty() {
            return (0, 0);
        }
        let capacity = self.slots.len();
        let mut at = from % capacity;
        let mut removed = 0;
        for _ in 0..budget.min(capacity) {
            if is_full(self.ctrl[at]) && self.is_expired(at, now) {
                self.erase(at);
                removed += 1;
            }
            at = (at + 1) % capacity;
        }
        (at, removed)
    }

    /// Drop one entry chosen by sampling, for when the cache is over its memory limit.
    ///
    /// Two candidates are drawn and the one that expires sooner goes, an entry with no expiry counting as expiring last. This is the two-random rule rather than a true recency order, because maintaining recency costs two links per entry, and two links are sixteen bytes sitting on a five byte slot.
    pub fn evict_one(&mut self, roll: u64) -> bool {
        if self.len == 0 {
            return false;
        }
        let capacity = self.slots.len();
        let once = mix(roll);
        let twice = mix(once);
        let first = self.nearest_full(spot(once, capacity));
        let second = self.nearest_full(spot(twice, capacity));
        let (Some(first), Some(second)) = (first, second) else {
            return false;
        };

        let victim = if self.deadline(first) <= self.deadline(second) {
            first
        } else {
            second
        };
        self.erase(victim);
        true
    }

    /// When the entry in slot `at` expires, with no expiry reading as never.
    #[inline]
    fn deadline(&self, at: usize) -> u32 {
        let entry = Ref::from_bits(self.slots[at]);
        let head = self.head_at(entry);
        self.expiry_of(entry, head).unwrap_or(u32::MAX)
    }

    /// The first occupied slot at or after `from`, wrapping once.
    fn nearest_full(&self, from: usize) -> Option<usize> {
        let capacity = self.slots.len();
        (0..capacity)
            .map(|step| (from + step) % capacity)
            .find(|&at| is_full(self.ctrl[at]))
    }

    /// Call `each` with every live entry, expired or not. The table cannot be changed from inside it.
    pub fn for_each(&self, mut each: impl FnMut(View<'_>)) {
        for at in 0..self.slots.len() {
            if is_full(self.ctrl[at]) {
                let (head, bytes) = self.entry_at(Ref::from_bits(self.slots[at]));
                each(head.view(bytes));
            }
        }
    }
}

/// The splitmix64 finaliser.
///
/// [`Table::evict_one`] takes one number and needs two independent positions from it. Shifting the same number twice would not give two, and a caller whose generator has weak low bits — a plain linear congruential one, say — would draw its second candidate near its first for a long stretch. Mixing costs two multiplies and removes the question.
#[inline]
const fn mix(x: u64) -> u64 {
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// A slot number in `[0, capacity)` drawn from the top half of `bits`.
///
/// The top half, because the capacity is a power of two and a mask therefore keeps only the lowest bits, which are the weakest ones in most cheap generators.
#[inline]
const fn spot(bits: u64, capacity: usize) -> usize {
    ((bits >> 32) as usize) & (capacity - 1)
}

/// The low bits of a hash as a group number, in a table of `mask + 1` groups.
///
/// Narrowing to a `usize` on a target with 32-bit pointers keeps the low half, and the mask was about to keep a good deal less than that, so the bits the cast drops are bits no caller was going to read.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the mask keeps fewer bits than the narrowest usize has"
)]
pub(crate) const fn group_of(hash: u64, mask: usize) -> usize {
    (hash as usize) & mask
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 0x5eed;

    fn h(key: &[u8]) -> u64 {
        rugo_hash::hash(key, SEED)
    }

    fn set(table: &mut Table, key: &[u8], value: &[u8]) -> Wrote {
        table.set(key, value, h(key), None, None).unwrap()
    }

    fn get(table: &mut Table, key: &[u8]) -> Option<Vec<u8>> {
        table.get(key, h(key), 0).map(<[u8]>::to_vec)
    }

    /// How many entries actually carry an expiry, counted by looking at every one of them.
    ///
    /// The slow answer to the question `timed` keeps the fast answer to. A test that compared the field against itself would pass no matter which sites forgot to maintain it.
    fn count_timed(table: &Table) -> usize {
        (0..table.slots.len())
            .filter(|&at| is_full(table.ctrl[at]))
            .filter(|&at| {
                let entry = Ref::from_bits(table.slots[at]);
                table.head_at(entry).flags & entry::HAS_EXPIRY != 0
            })
            .count()
    }

    #[test]
    fn the_count_of_timed_entries_survives_every_way_one_can_change() {
        // What the sweep now trusts instead of reading. If it is ever too low the sweep skips a shard that has expired keys in it, and those keys are then held until something happens to look one of them up, which for a key nobody asks for again is never. So this walks every route by which an entry can gain or lose an expiry and checks the count against the entries themselves rather than against itself.
        let mut table = Table::new(SEED);

        // Inserted with and without.
        for i in 0..500u32 {
            let key = format!("key:{i}");
            let expiry = if i % 3 == 0 { Some(1_000) } else { None };
            table
                .set(key.as_bytes(), b"v", h(key.as_bytes()), expiry, None)
                .unwrap();
        }
        assert_eq!(table.timed, count_timed(&table), "after inserting");

        // Replaced at the same width, which is the path that writes over the entry in place.
        for i in 0..200u32 {
            let key = format!("key:{i}");
            let expiry = if i % 2 == 0 { Some(2_000) } else { None };
            table
                .set(key.as_bytes(), b"w", h(key.as_bytes()), expiry, None)
                .unwrap();
        }
        assert_eq!(table.timed, count_timed(&table), "after replacing in place");

        // Replaced at a different width, which is the path that moves the entry.
        for i in 0..200u32 {
            let key = format!("key:{i}");
            let expiry = if i % 4 == 0 { Some(3_000) } else { None };
            let value = "x".repeat(i as usize % 40 + 1);
            table
                .set(
                    key.as_bytes(),
                    value.as_bytes(),
                    h(key.as_bytes()),
                    expiry,
                    None,
                )
                .unwrap();
        }
        assert_eq!(
            table.timed,
            count_timed(&table),
            "after replacing at a new width"
        );

        // EXPIRE and PERSIST, which is the only path that changes the trailer without changing the value.
        for i in 0..300u32 {
            let key = format!("key:{i}");
            let expiry = if i % 5 == 0 { Some(4_000) } else { None };
            table
                .expire(key.as_bytes(), h(key.as_bytes()), expiry, 0)
                .unwrap();
        }
        assert_eq!(table.timed, count_timed(&table), "after expire and persist");

        // Removed.
        for i in (0..500u32).step_by(3) {
            let key = format!("key:{i}");
            table.remove(key.as_bytes(), h(key.as_bytes()));
        }
        assert_eq!(table.timed, count_timed(&table), "after removing");

        // Rebuilt, which moves every entry to a new slot.
        table.shrink();
        assert_eq!(table.timed, count_timed(&table), "after a rebuild");

        table.clear();
        assert_eq!(table.timed, 0, "a cleared table holds no expiry");
        assert_eq!(table.timed, count_timed(&table), "after clearing");
    }

    #[test]
    fn a_shard_with_an_expiry_in_it_is_still_swept() {
        // The other half of the skip. Nothing above would fail if `timed` were wired to zero and the sweep never ran again, so this is the test that says the shortcut is a shortcut and not a removal.
        let mut table = Table::new(SEED);
        for i in 0..100u32 {
            let key = format!("key:{i}");
            table
                .set(key.as_bytes(), b"v", h(key.as_bytes()), Some(10), None)
                .unwrap();
        }
        assert_eq!(table.timed, 100);

        // Every key expired at ten and the clock reads eleven.
        let (_, removed) = table.sweep(0, table.capacity(), 11);
        assert_eq!(
            removed, 100,
            "the sweep found nothing to remove in a table of expired keys"
        );
        assert_eq!(table.len(), 0);
        assert_eq!(
            table.timed, 0,
            "removing the last timed entry left the count behind"
        );
    }

    #[test]
    fn an_empty_table_owns_no_slots() {
        // Four thousand of these exist before the first key arrives, so the empty case is the one that sets the floor.
        let table = Table::new(SEED);
        assert_eq!(table.capacity(), 0);
        assert_eq!(table.len(), 0);
        assert_eq!(table.index_bytes(), 0);
    }

    #[test]
    fn what_goes_in_comes_out() {
        let mut table = Table::new(SEED);
        for i in 0..10_000u32 {
            set(
                &mut table,
                format!("key:{i}").as_bytes(),
                format!("value:{i}").as_bytes(),
            );
        }
        assert_eq!(table.len(), 10_000);
        for i in 0..10_000u32 {
            let key = format!("key:{i}");
            assert_eq!(
                get(&mut table, key.as_bytes()).as_deref(),
                Some(format!("value:{i}").as_bytes()),
                "{key} came back wrong"
            );
        }
    }

    #[test]
    fn a_missing_key_is_missing() {
        let mut table = Table::new(SEED);
        for i in 0..1000u32 {
            set(&mut table, format!("key:{i}").as_bytes(), b"v");
        }
        for i in 1000..2000u32 {
            assert_eq!(get(&mut table, format!("key:{i}").as_bytes()), None);
        }
    }

    #[test]
    fn setting_the_same_key_replaces_it() {
        let mut table = Table::new(SEED);
        assert_eq!(set(&mut table, b"k", b"one"), Wrote::Inserted);
        assert_eq!(set(&mut table, b"k", b"two"), Wrote::Replaced);
        assert_eq!(table.len(), 1);
        assert_eq!(get(&mut table, b"k").as_deref(), Some(&b"two"[..]));
    }

    #[test]
    fn a_replacement_of_a_different_length_still_works() {
        // The in-place path only fires when the sizes match, so a growing and then a shrinking value exercise the other one in both directions.
        let mut table = Table::new(SEED);
        set(&mut table, b"k", b"short");
        set(
            &mut table,
            b"k",
            b"a considerably longer value than the one before it",
        );
        assert_eq!(
            get(&mut table, b"k").as_deref(),
            Some(&b"a considerably longer value than the one before it"[..])
        );
        set(&mut table, b"k", b"tiny");
        assert_eq!(get(&mut table, b"k").as_deref(), Some(&b"tiny"[..]));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn removal_removes() {
        let mut table = Table::new(SEED);
        for i in 0..1000u32 {
            set(&mut table, format!("key:{i}").as_bytes(), b"v");
        }
        for i in 0..1000u32 {
            let key = format!("key:{i}");
            assert!(
                table.remove(key.as_bytes(), h(key.as_bytes())),
                "{key} was not there"
            );
        }
        assert_eq!(table.len(), 0);
        for i in 0..1000u32 {
            assert_eq!(get(&mut table, format!("key:{i}").as_bytes()), None);
        }
    }

    #[test]
    fn a_tombstone_does_not_hide_a_later_key() {
        // The bug this catches is marking a slot empty on removal when a probe sequence ran past it, which does not crash: it makes an unrelated key silently unreachable. It needs enough keys that collisions are certain.
        let mut table = Table::new(SEED);
        let n = 20_000u32;
        for i in 0..n {
            set(&mut table, format!("k{i}").as_bytes(), b"v");
        }
        for i in (0..n).step_by(2) {
            let key = format!("k{i}");
            assert!(table.remove(key.as_bytes(), h(key.as_bytes())));
        }
        for i in (1..n).step_by(2) {
            let key = format!("k{i}");
            assert_eq!(
                get(&mut table, key.as_bytes()).as_deref(),
                Some(&b"v"[..]),
                "{key} was lost behind a tombstone"
            );
        }
        assert_eq!(table.len(), (n / 2) as usize);
    }

    #[test]
    fn churn_does_not_grow_the_table_without_bound() {
        // Matched inserts and deletes forever. If tombstones were never reclaimed the capacity would double every time the table filled with them, and the memory claim would be false for any long-lived cache.
        let mut table = Table::new(SEED);
        for i in 0..200u32 {
            set(&mut table, format!("k{i}").as_bytes(), b"v");
        }
        let settled = table.capacity();
        for round in 1..200u32 {
            for i in 0..200u32 {
                let old = format!("k{}", i + (round - 1) * 200);
                table.remove(old.as_bytes(), h(old.as_bytes()));
                set(&mut table, format!("k{}", i + round * 200).as_bytes(), b"v");
            }
        }
        assert_eq!(table.len(), 200);
        assert!(
            table.capacity() <= settled * 2,
            "capacity ran from {settled} to {} under churn",
            table.capacity()
        );
    }

    #[test]
    fn an_expired_key_is_gone() {
        let mut table = Table::new(SEED);
        table.set(b"k", b"v", h(b"k"), Some(100), None).unwrap();
        assert_eq!(table.get(b"k", h(b"k"), 99), Some(&b"v"[..]));
        assert_eq!(
            table.get(b"k", h(b"k"), 100),
            None,
            "an expiry is inclusive"
        );
        assert_eq!(table.len(), 0, "reading an expired key should reclaim it");
    }

    #[test]
    fn a_sweep_reclaims_expired_keys() {
        let mut table = Table::new(SEED);
        for i in 0..1000u32 {
            let key = format!("k{i}");
            let expiry = if i % 2 == 0 { Some(10) } else { None };
            table
                .set(key.as_bytes(), b"v", h(key.as_bytes()), expiry, None)
                .unwrap();
        }
        let (_, removed) = table.sweep(0, table.capacity(), 20);
        assert_eq!(removed, 500);
        assert_eq!(table.len(), 500);
        for i in (1..1000u32).step_by(2) {
            assert_eq!(
                get(&mut table, format!("k{i}").as_bytes()).as_deref(),
                Some(&b"v"[..]),
                "the sweep took an unexpiring key"
            );
        }
    }

    #[test]
    fn a_sweep_resumes_where_it_stopped() {
        let mut table = Table::new(SEED);
        for i in 0..1000u32 {
            let key = format!("k{i}");
            table
                .set(key.as_bytes(), b"v", h(key.as_bytes()), Some(10), None)
                .unwrap();
        }
        let mut at = 0;
        let mut removed = 0;
        while !table.is_empty() {
            let (next, took) = table.sweep(at, 64, 20);
            at = next;
            removed += took;
        }
        assert_eq!(removed, 1000);
    }

    #[test]
    fn eviction_removes_something_every_time() {
        let mut table = Table::new(SEED);
        for i in 0..1000u32 {
            set(&mut table, format!("k{i}").as_bytes(), b"v");
        }
        let mut roll = 0x1234_5678_9abc_def0u64;
        for _ in 0..1000 {
            roll = roll.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            assert!(table.evict_one(roll));
        }
        assert_eq!(table.len(), 0);
        assert!(
            !table.evict_one(roll),
            "an empty table has nothing to evict"
        );
    }

    #[test]
    fn eviction_prefers_the_entry_that_expires_sooner() {
        // Half the keys expire and half never do. Two-random is not exact, but over five hundred draws it should take the expiring ones far more often than chance, which would leave about two hundred and fifty.
        let mut table = Table::new(SEED);
        for i in 0..1000u32 {
            let key = format!("k{i}");
            let expiry = if i % 2 == 0 { Some(1000) } else { None };
            table
                .set(key.as_bytes(), b"v", h(key.as_bytes()), expiry, None)
                .unwrap();
        }
        let mut roll = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..500 {
            roll = roll.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            table.evict_one(roll);
        }
        let mut survived_with_expiry = 0;
        table.for_each(|view| {
            if view.expiry.is_some() {
                survived_with_expiry += 1;
            }
        });
        // Chance would leave 250 with a standard deviation near 8. Two-random against a pool that keeps shrinking cannot reach zero and simulates out at about 167, so 200 is a real signal rather than a rubber stamp: it is six standard deviations under chance and within a fifth of what the rule can do at its best.
        assert!(
            survived_with_expiry < 200,
            "{survived_with_expiry} expiring keys survived where chance would leave about 250 and two-random should leave about 167"
        );
    }

    #[test]
    fn user_flags_and_expiry_survive() {
        let mut table = Table::new(SEED);
        table
            .set(b"k", b"v", h(b"k"), Some(500), Some(0xdead_beef))
            .unwrap();
        let view = table.view(b"k", h(b"k"), 0).unwrap();
        assert_eq!(view.key, b"k");
        assert_eq!(view.value, b"v");
        assert_eq!(view.expiry, Some(500));
        assert_eq!(view.user_flags, Some(0xdead_beef));
    }

    #[test]
    fn a_value_larger_than_the_slab_works() {
        let mut table = Table::new(SEED);
        let big = vec![7u8; 200_000];
        set(&mut table, b"big", &big);
        assert_eq!(get(&mut table, b"big").as_deref(), Some(&big[..]));
        assert!(table.remove(b"big", h(b"big")));
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn an_empty_key_and_an_empty_value_are_ordinary() {
        let mut table = Table::new(SEED);
        set(&mut table, b"", b"");
        assert_eq!(get(&mut table, b"").as_deref(), Some(&b""[..]));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn clearing_empties_it_and_leaves_it_usable() {
        let mut table = Table::new(SEED);
        for i in 0..1000u32 {
            set(&mut table, format!("k{i}").as_bytes(), b"v");
        }
        table.clear();
        assert_eq!(table.len(), 0);
        for i in 0..1000u32 {
            assert_eq!(get(&mut table, format!("k{i}").as_bytes()), None);
        }
        set(&mut table, b"after", b"v");
        assert_eq!(get(&mut table, b"after").as_deref(), Some(&b"v"[..]));
    }

    #[test]
    fn a_shrunken_table_gives_its_index_back() {
        let mut table = Table::new(SEED);
        for i in 0..100_000u32 {
            set(&mut table, format!("k{i}").as_bytes(), b"v");
        }
        let grown = table.index_bytes();
        for i in 0..99_000u32 {
            let key = format!("k{i}");
            table.remove(key.as_bytes(), h(key.as_bytes()));
        }
        assert!(!table.shrink() || table.index_bytes() < grown / 8);
        assert_eq!(table.len(), 1000);
        for i in 99_000..100_000u32 {
            assert_eq!(
                get(&mut table, format!("k{i}").as_bytes()).as_deref(),
                Some(&b"v"[..]),
                "a key was lost in the shrink"
            );
        }
        assert!(!table.shrink(), "a table already at its size shrank again");
    }

    #[test]
    fn charged_bytes_fall_when_entries_go() {
        // The property eviction depends on, and the reason the ceiling is measured against charged bytes rather than resident ones: the index comes back when the table shrinks, but the arena's segments never do.
        let mut table = Table::new(SEED);
        for i in 0..20_000u32 {
            set(&mut table, format!("k{i}").as_bytes(), &[0u8; 64]);
        }
        let full = table.charged_bytes();
        let segments = table.resident_bytes() - table.index_bytes();
        for i in 0..19_000u32 {
            let key = format!("k{i}");
            table.remove(key.as_bytes(), h(key.as_bytes()));
        }
        table.shrink();
        assert!(
            table.charged_bytes() < full / 8,
            "charged bytes went from {full} to {}",
            table.charged_bytes()
        );
        assert_eq!(
            table.resident_bytes() - table.index_bytes(),
            segments,
            "the arena is expected to keep every segment it took, which is why a ceiling measured against resident bytes could never converge"
        );
        assert!(
            table.resident_bytes() > table.charged_bytes() * 4,
            "resident {} against charged {}: the gap is the freed entry bytes the arena is still holding",
            table.resident_bytes(),
            table.charged_bytes()
        );
    }

    #[test]
    fn the_index_costs_five_bytes_a_slot() {
        // The claim the crate exists to make, checked rather than asserted in prose. Pogocache's bucket entry is ten bytes for the same job.
        let mut table = Table::new(SEED);
        for i in 0..100_000u32 {
            set(&mut table, format!("k{i}").as_bytes(), b"v");
        }
        assert_eq!(table.index_bytes(), table.capacity() * 5);
    }

    #[test]
    fn the_table_stays_under_its_load_factor() {
        // A load factor that slipped would show up here before it showed up in a memory sweep.
        let mut table = Table::new(SEED);
        for i in 0..100_000u32 {
            set(&mut table, format!("k{i}").as_bytes(), b"v");
        }
        assert!(
            table.capacity() * LOAD_NUM >= table.len() * LOAD_DEN,
            "{} entries in {} slots is over the load factor",
            table.len(),
            table.capacity()
        );
    }
}
