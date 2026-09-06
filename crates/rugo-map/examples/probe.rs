//! One lookup's worth of memory traffic, with no server around it.
//!
//! The server's profile says the get path is most of where the cycles and the misses are, and it cannot say how many of those belong to the map, because a server also reads a socket, parses a command and writes a reply, and a profile attributes a miss to the instruction that took it rather than to the thing that caused it. This fills a map the size the sweep uses and then does nothing but look keys up in it, so a counter attached to it is counting the map.
//!
//! It is built for `perf stat` rather than for a timer, so the fill and the lookups have to be counted apart. The fill is as expensive as the pass being measured and would otherwise be half of every number. So it prints `ready` when the map is full and then waits for a line on standard input, which is the point for something outside to attach, and prints how many lookups it did when it is done, which is the divisor.
//!
//! ```text
//! cargo build --release --example probe
//! ./target/release/examples/probe 5120000 8 5000000 4096
//! ```
//!
//! The arguments are entries, value size, lookups and shards, each optional and taken in that order.

use rugo_map::Map;
use std::hint::black_box;
use std::io::{BufRead, Write};

/// A key of the shape memtier generates when it is given no prefix, which is the shape the sweep measures.
fn key_of(n: u64, into: &mut Vec<u8>) {
    into.clear();
    let mut at = n;
    let start = into.len();
    loop {
        // The digits come out backwards and are turned around below, which is cheaper than formatting and is the whole of what this does.
        into.push(b'0' + u8::try_from(at % 10).unwrap_or(b'?'));
        at /= 10;
        if at == 0 {
            break;
        }
    }
    into[start..].reverse();
}

/// The next number in a cheap sequence that visits every key without repeating and without any order the prefetcher can follow.
///
/// A lookup pass that walks the keys in order measures the hardware prefetcher rather than the map, and one that draws from a real random source measures the random source. This is xorshift, which is neither.
const fn next(state: u64) -> u64 {
    let mut x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// The argument at `n`, or the default.
fn arg(n: usize, default: u64) -> u64 {
    std::env::args()
        .nth(n)
        .and_then(|a| a.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let entries = arg(1, 5_120_000);
    let value_len = arg(2, 8);
    let lookups = arg(3, 5_000_000);
    let shards = arg(4, 4096);

    let value = vec![0xa5_u8; usize::try_from(value_len).unwrap_or(8)];
    let map = Map::with_seed(
        usize::try_from(shards).unwrap_or(4096),
        0,
        0x5265_6d65_6d62_6572,
    );

    let mut key = Vec::with_capacity(24);
    for n in 0..entries {
        key_of(n, &mut key);
        if map.set(&key, &value, None, None).is_err() {
            eprintln!("the arena filled up at entry {n}");
            return;
        }
    }

    println!("ready {} entries in {} shards", map.len(), map.shards());
    let _ = std::io::stdout().flush();
    let mut go = String::new();
    let _ = std::io::stdin().lock().read_line(&mut go);

    let mut state = 0x243f_6a88_85a3_08d3_u64;
    let mut found = 0_u64;
    for _ in 0..lookups {
        state = next(state);
        key_of(state % entries, &mut key);
        if map.get(&key, <[u8]>::len).is_some() {
            found += 1;
        }
    }
    println!("{lookups} lookups, {} hit", black_box(found));
}
