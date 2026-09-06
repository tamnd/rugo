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

/// The seed every map in this file is built with.
///
/// A map ordinarily takes a random seed, and every test here except this one wants that. These do not. A gate that answers differently on two runs of the same code is not a gate, and the arena is where that bites hardest: which shard a key lands in decides how many segments that shard grows, the newest of those segments is the reserve, and at a few hundred shards holding kilobyte values the difference between a lucky seed and an unlucky one is a few per cent of everything the map holds. Measured on the boxed slice path, three runs of one unchanged binary came out at 20.12, 20.32 and 20.68 megabytes of slack against a bound of 20.56, so the gate passed twice and failed once without a line of code changing.
///
/// Fixing the seed does not weaken what is measured. That keys spread evenly over shards is a property of the hash and it has its own test, in `rugo-map`'s own suite, which is where a hash that piles keys into a few shards ought to be caught. What is measured here is what a spread costs, and for that the spread only has to be a real one and the same one every time.
///
/// The value is arbitrary and nothing depends on which one it is.
const SEED: u64 = 0x5265_6d65_6d62_6572;

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

/// The value length entry `n` gets under the spread the harness uses.
///
/// `memtier_benchmark` is run with `--data-size-range 1-1024`, so every length in that range occurs and the rounding up to the grain is paid at every offset within it. A fixed length either rounds or does not, which is why the shapes above can miss a change to the grain entirely and this one cannot.
fn spread_len(n: u32) -> usize {
    1 + (n as usize).wrapping_mul(2_654_435_761) % 1024
}

/// Fill a map with the harness's spread of value lengths and measure it.
fn fill_spread(count: u32, shards: usize) -> Cost {
    let map = Map::with_seed(shards, 0, SEED);
    let mut payload = 0usize;
    for n in 0..count {
        let key = key_of(n);
        let value = vec![0xa5u8; spread_len(n)];
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

/// Fill a map and measure it.
fn fill(count: u32, value_len: usize, shards: usize) -> Cost {
    let map = Map::with_seed(shards, 0, SEED);
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

/// How many bytes an entry spends describing itself, restated here for the same reason [`varint_len`] is.
///
/// One flags byte, which carries the key length itself for a key of sixty-two bytes or less, and a varint for the value. A longer key spends a varint of its own on top.
fn header_len(key_len: usize, value_len: usize) -> usize {
    let described = if key_len <= 62 {
        0
    } else {
        varint_len(key_len)
    };
    1 + described + varint_len(value_len)
}

#[test]
fn an_entry_costs_its_header_and_nothing_but_the_grain() {
    // The entry encoding and the allocation grain, and no allowance for anything else. The header is one flags byte carrying the key length, and a varint for the value, which is exactly predictable, so it is predicted here and subtracted. What is left can only be the rounding up to the grain, and rounding to a grain cannot cost a whole grain.
    //
    // The bound is the crate's own constant rather than the number it currently holds, because a gate written against a literal eight goes on passing after the grain halves and stops being a gate at all.
    //
    // There is no allocator header in this number, because there is no allocator header. That is the eight to sixteen bytes a cache calling `malloc` once per entry pays and this one does not.
    let limit = rugo_arena::GRAIN as f64;
    for &(count, value_len, shards) in SHAPES {
        let cost = fill(count, value_len, shards);
        let header = header_len(key_of(0).len(), value_len);
        let beyond = (cost.live - cost.payload) as f64 / cost.entries as f64;
        let grain = beyond - header as f64;
        println!(
            "{count}x{value_len} over {shards} shards: {header} bytes of header, {grain:.2} of grain"
        );
        assert!(
            (0.0..limit).contains(&grain),
            "{beyond:.2} bytes an entry beyond key and value, against a {header} byte header, leaves {grain:.2} for a grain of {}",
            rugo_arena::GRAIN
        );
    }
}

#[test]
fn the_grain_costs_half_itself_where_every_length_occurs() {
    // The shapes above hold one value length each, so each of them either rounds or does not and the average is whichever it happened to be. The harness draws its lengths from the whole of one to a thousand and twenty-four, where every remainder occurs about equally often and the rounding costs half a grain an entry on average.
    //
    // That average is what the grain is actually worth on the published shape, so it is the thing worth bounding: below half a grain the arithmetic would have to be wrong, and above a whole grain something other than rounding is being counted.
    let cost = fill_spread(1_000_000, 4096);
    let mut headers = 0usize;
    for n in 0..1_000_000u32 {
        headers += header_len(key_of(n).len(), spread_len(n));
    }
    let grain = (cost.live - cost.payload - headers) as f64 / cost.entries as f64;
    let half = rugo_arena::GRAIN as f64 / 2.0;
    println!(
        "a million entries over the harness's spread: {grain:.2} bytes of grain an entry, against a grain of {}",
        rugo_arena::GRAIN
    );
    assert!(
        grain > half - 1.0 && grain < rugo_arena::GRAIN as f64,
        "{grain:.2} bytes an entry of rounding, which is not the half of {} that a uniform spread of lengths should cost",
        rugo_arena::GRAIN
    );
}

#[test]
fn the_arena_holds_little_beyond_what_it_handed_out() {
    // Everything the arena is charged for that is neither the index nor an entry: the part of the newest segment nothing has been written into, and the tails abandoned where a segment ended too short for the next entry.
    //
    // Which of those dominates is a property of the platform, so the bound is too, and each half of it names the mechanism it bounds.
    //
    // Where a segment is a mapping the reserve is address space and costs nothing. What is left is a part-used last page per shard, which is a floor and not a rate: four thousand shards is sixteen megabytes on a four kilobyte page whether they hold ten entries each or ten thousand. It is charged as a floor, plus two per cent for the tails.
    //
    // Where a segment is a boxed slice the reserve is real memory. A doubling rule put nearly a hundred per cent here, which was the single largest fault this gate ever caught, and a sixteenth puts it in the low single digits at every shape but the last.
    //
    // The last is nine and nine tenths per cent against a bound of ten, which is a margin worth explaining rather than leaving to look like luck. Values there are a kilobyte and a segment starts at four, so the first segments hold three entries and abandon most of a fourth, and a sixteenth reaches a kilobyte value's working set in tens of segments rather than the handful a doubling would take. The tails of those segments, not the reserve, are most of what the bound is holding back at that shape. It is deliberate that the bound is not widened to make room: the figure is the same on every run now, so the next thing to push it over will be a change rather than a seed, and that is the report this test exists to make.
    for &(count, value_len, shards) in SHAPES {
        let cost = fill(count, value_len, shards);
        let slack = cost.resident - cost.index - cost.live;
        let allowed = if rugo_arena::LAZY_RESERVE {
            shards * rugo_arena::granule() + cost.live / 50
        } else {
            cost.live / 10
        };
        println!(
            "{count}x{value_len} over {shards} shards: {slack} bytes beyond the entries, {allowed} allowed ({:.1}% of what was handed out)",
            slack as f64 / cost.live as f64 * 100.0
        );
        assert!(
            slack <= allowed,
            "the arena held {slack} bytes beyond what it handed out against {allowed} allowed, for {count} entries of {value_len} over {shards} shards"
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
    // The published shape, and the only rows here whose value lengths are the harness's rather than one repeated number. A million entries rather than the ten million the sweep fills, because a test that runs on every push cannot hold five gigabytes.
    //
    // One row per shard count rather than one row, because the shard count is the single largest thing in this table and it does not look like a memory decision at all. Every shard owns an arena, an arena is charged in whole pages, and a part-used last page is a floor a shard pays whether it holds ten entries or ten thousand. So the cost of a shard count is that floor divided by the entries in the map, which is why the spread below is wide at a million entries and would be a tenth of it at ten million, and why it is worth reading the row for the count a server would actually pick rather than the largest one the map allows.
    for shards in [4096usize, 512, 256, 128, 64] {
        let cost = fill_spread(1_000_000, shards);
        println!(
            "{:>7}  {:>5}  {shards:>6}  {:>13.2}  {:>16.2}",
            1_000_000,
            "1-1k",
            cost.total(),
            cost.overhead()
        );
    }
}

#[test]
fn four_thousand_empty_shards_are_nearly_free() {
    // The design cost of sharding this finely. An arena that allocated eagerly would charge sixty-four kilobytes a shard here, which is a quarter of a gigabyte before a single key exists, and would make the fine sharding the throughput target depends on unaffordable.
    let map = Map::with_seed(4096, 0, SEED);
    assert_eq!(map.shards(), 4096);
    let empty = map.resident_bytes();
    println!("4096 empty shards cost {empty} bytes");
    assert_eq!(empty, 0, "an empty shard is expected to allocate nothing");
}

#[test]
fn one_key_does_not_wake_four_thousand_shards() {
    // A shard allocates on its first write and only then, so a map holding one key should be holding one index and one segment, not four thousand of each.
    let map = Map::with_seed(4096, 0, SEED);
    map.set(b"only", b"one", None, None).unwrap();
    let resident = map.resident_bytes();
    println!("4096 shards holding one key cost {resident} bytes");
    // Sixty-four kilobytes rather than eight, because a shard's arena is charged in whole pages and a page is sixteen kilobytes on Apple silicon. The number this is really distinguishing is four thousand shards each waking, which is sixteen megabytes at the smallest page there is, so anything in kilobytes says one shard woke.
    assert!(
        resident < 64 * 1024,
        "{resident} bytes to hold one key across 4096 shards"
    );
}

#[test]
fn sharding_does_not_cost_much_at_scale() {
    // Sixty-four shards against four thousand over the same working set. Every shard carries its own index and its own part-used last page, so more shards is strictly more slack; the claim is that at a million entries the slack is small enough that the concurrency is worth having.
    //
    // Where segments are mappings that slack is four thousand and thirty-two more part-used pages, which is a quantity this can state exactly rather than guess at as a percentage, so it does. Two per cent on top covers the index growing with the shard count and the extra tails.
    const ENTRIES: u32 = 1_000_000;
    let few = fill(ENTRIES, 100, 64).total();
    let many = fill(ENTRIES, 100, 4096).total();
    let allowed = if rugo_arena::LAZY_RESERVE {
        ((4096 - 64) * rugo_arena::granule()) as f64 / f64::from(ENTRIES) + few * 0.02
    } else {
        few * 0.10
    };
    println!("{few:.2} bytes per entry over 64 shards, {many:.2} over 4096, {allowed:.2} allowed");
    assert!(
        many - few <= allowed,
        "going from 64 shards to 4096 cost {:.2} bytes an entry against {allowed:.2} allowed",
        many - few
    );
}
