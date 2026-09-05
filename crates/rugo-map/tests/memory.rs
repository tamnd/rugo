//! The memory claim, measured.
//!
//! rugo exists partly to hold a cache in less memory than the servers it is measured against, and a claim like that is worth exactly as much as the test that checks it. These are that test. They run in CI as `cargo test -p rugo-map --test memory`, and the numbers they print are the ones `SCOREBOARD.md` quotes.
//!
//! # Two numbers, not one
//!
//! **Total bytes per entry** is everything the map has taken from the operating system divided by the entries in it. It is dominated by the key and the value, which every cache server stores identically, so at a hundred bytes of payload no design can be twice as good as another on this number. It is reported anyway, because it is what an operator sees in `ps`.
//!
//! **Overhead bytes per entry** is the same total with the key and value bytes subtracted: the index, the entry headers, the allocation grain and the arena's reserve. This is the number a design controls and the one the 2x target is really about.
//!
//! Quoting only the first would understate the design; quoting only the second would overstate what an operator will see. Both are published, and the scoreboard says which is which.
//!
//! # Why the gate is three assertions and not one
//!
//! An overhead ceiling of, say, twenty bytes is a number with no argument behind it, and the temptation when it fails by half a byte is to write twenty-one. The three assertions below each bound one mechanism, so a failure names its own cause and a threshold cannot be moved without admitting which part of the design got worse.
//!
//! Their sum is the overhead, and the report at the end prints it.

// Every number here is a ratio of byte counts meant for a human to read to two decimal places, and a byte count large enough to lose a mantissa bit is a byte count no test in this file allocates.
#![expect(
    clippy::cast_precision_loss,
    reason = "measurements reported as ratios, over counts far under 2^52"
)]

use rugo_map::Map;

/// A key of the shape a benchmark generates: a prefix and a number.
fn key_of(n: u32) -> Vec<u8> {
    format!("memtier-{n:016}").into_bytes()
}

/// What filling a map with `count` entries of `value_len` bytes cost.
struct Cost {
    /// Key and value bytes, and nothing else.
    payload: usize,
    /// Control bytes and slots across every shard.
    index: usize,
    /// Entry bytes handed out, rounded up to the grain.
    live: usize,
    /// Everything taken from the operating system.
    resident: usize,
    /// Entries in the map.
    entries: usize,
}

impl Cost {
    /// Bytes per entry that are not key or value.
    fn overhead(&self) -> f64 {
        (self.resident.saturating_sub(self.payload)) as f64 / self.entries as f64
    }

    /// Bytes per entry, everything counted.
    fn total(&self) -> f64 {
        self.resident as f64 / self.entries as f64
    }
}

/// Fill a map and measure it.
fn fill(count: u32, value_len: usize, shards: usize) -> Cost {
    let map = Map::new(shards, 0);
    let value = vec![0xa5u8; value_len];
    let mut payload = 0usize;
    for n in 0..count {
        let key = key_of(n);
        payload += key.len() + value.len();
        assert!(
            map.set(&key, &value, None, None).is_ok(),
            "the map refused a write with no ceiling set"
        );
    }
    assert_eq!(
        map.len(),
        count as usize,
        "the map lost entries while filling"
    );
    Cost {
        payload,
        index: map.index_bytes(),
        live: map.live_bytes(),
        resident: map.resident_bytes(),
        entries: map.len(),
    }
}

/// The shapes every gate below is measured over.
///
/// Four rather than one, because an index that is cheap for a hundred thousand short keys and expensive for a million long ones has not solved anything. The counts are chosen to land at different points in the doubling cycle, so at least one of them measures a table that has just grown and is half empty.
const SHAPES: &[(u32, usize, usize)] = &[
    (100_000, 10, 64),
    (100_000, 100, 64),
    (1_000_000, 100, 64),
    (1_000_000, 100, 4096),
    (200_000, 1000, 256),
];

#[test]
fn the_index_costs_under_eleven_bytes_an_entry() {
    // Five bytes a slot is the structural claim and `table.rs` checks it exactly. This checks what it comes to per entry, which is five divided by the load factor, and the load factor is what a growth rule controls.
    //
    // A power-of-two table that grows at seven eighths is between forty-four and eighty-seven percent full, so five bytes a slot is between five and three quarters and eleven and a third per entry, and eleven is the worst case rather than the typical one. Pogocache's bucket is ten bytes under the same rule and so lands between eleven and a half and twenty-three: the ratio is two at every point in the cycle, which is the claim, and neither number is flattered by choosing where to sample.
    for &(count, value_len, shards) in SHAPES {
        let cost = fill(count, value_len, shards);
        let per_entry = cost.index as f64 / cost.entries as f64;
        println!("{count}x{value_len} over {shards} shards: {per_entry:.2} index bytes an entry");
        assert!(
            per_entry < 11.5,
            "{per_entry:.2} index bytes an entry for {count} entries over {shards} shards"
        );
    }
}

/// How many bytes a LEB128 varint takes for `value`.
///
/// The test computes this rather than importing it, so that a change to the entry encoding has to be restated here to pass, instead of moving the expectation along with the code it is meant to check.
fn varint_len(mut value: usize) -> usize {
    let mut len = 1;
    while value >= 128 {
        value >>= 7;
        len += 1;
    }
    len
}

#[test]
fn an_entry_costs_its_header_and_nothing_but_the_grain() {
    // The entry encoding and the allocation grain, and no allowance for anything else. The header is one flags byte and a varint for each of the two lengths, which is exactly predictable, so it is predicted here and subtracted. What is left can only be the rounding up to the eight byte grain, and rounding to a grain cannot cost a whole grain.
    //
    // There is no allocator header in this number, because there is no allocator header. That is the eight to sixteen bytes a cache calling `malloc` once per entry pays and this one does not.
    for &(count, value_len, shards) in SHAPES {
        let cost = fill(count, value_len, shards);
        let key_len = key_of(0).len();
        let header = 1 + varint_len(key_len) + varint_len(value_len);
        let beyond = (cost.live - cost.payload) as f64 / cost.entries as f64;
        let grain = beyond - header as f64;
        println!(
            "{count}x{value_len} over {shards} shards: {header} bytes of header, {grain:.2} of grain"
        );
        assert!(
            (0.0..8.0).contains(&grain),
            "{beyond:.2} bytes an entry beyond key and value, against a {header} byte header, leaves {grain:.2} for a grain of 8"
        );
    }
}

#[test]
fn the_arena_holds_under_a_tenth_in_reserve() {
    // The segment that has been allocated and not yet written into. A doubling rule put nearly a hundred percent here, which was the single largest fault the gate caught; growing by a sixteenth puts it in the low single digits, and a tenth is the line past which the growth rule has regressed rather than drifted.
    for &(count, value_len, shards) in SHAPES {
        let cost = fill(count, value_len, shards);
        let reserve = cost.resident - cost.index - cost.live;
        let share = reserve as f64 / cost.live as f64;
        println!(
            "{count}x{value_len} over {shards} shards: {:.1}% held in reserve",
            share * 100.0
        );
        assert!(
            share < 0.10,
            "the arena held {:.1}% more than it had handed out, for {count} entries of {value_len} over {shards} shards",
            share * 100.0
        );
    }
}

#[test]
fn report_the_numbers_the_scoreboard_quotes() {
    // Not a gate. The three above bound the mechanisms; this prints what they add up to, in the form the scoreboard carries, so that the published number and the tested one are the same number.
    println!("entries  value  shards  total B/entry  overhead B/entry");
    for &(count, value_len, shards) in SHAPES {
        let cost = fill(count, value_len, shards);
        println!(
            "{count:>7}  {value_len:>5}  {shards:>6}  {:>13.2}  {:>16.2}",
            cost.total(),
            cost.overhead()
        );
    }
}

#[test]
fn four_thousand_empty_shards_are_nearly_free() {
    // The design cost of sharding this finely. An arena that allocated eagerly would charge sixty-four kilobytes a shard here, which is a quarter of a gigabyte before a single key exists, and would make the fine sharding the throughput target depends on unaffordable.
    let map = Map::new(4096, 0);
    assert_eq!(map.shards(), 4096);
    let empty = map.resident_bytes();
    println!("4096 empty shards cost {empty} bytes");
    assert_eq!(empty, 0, "an empty shard is expected to allocate nothing");
}

#[test]
fn one_key_does_not_wake_four_thousand_shards() {
    // A shard allocates on its first write and only then, so a map holding one key should be holding one index and one segment, not four thousand of each.
    let map = Map::new(4096, 0);
    map.set(b"only", b"one", None, None).unwrap();
    let resident = map.resident_bytes();
    println!("4096 shards holding one key cost {resident} bytes");
    assert!(
        resident < 8 * 1024,
        "{resident} bytes to hold one key across 4096 shards"
    );
}

#[test]
fn sharding_does_not_cost_much_at_scale() {
    // Sixty-four shards against four thousand over the same working set. Every shard carries its own index and its own arena reserve, so more shards is strictly more slack; the claim is that at a million entries the slack is small enough that the concurrency is worth having.
    let few = fill(1_000_000, 100, 64).total();
    let many = fill(1_000_000, 100, 4096).total();
    println!("{few:.2} bytes per entry over 64 shards, {many:.2} over 4096");
    assert!(
        many < few * 1.10,
        "going from 64 shards to 4096 cost {:.1}% more memory per entry",
        (many / few - 1.0) * 100.0
    );
}
