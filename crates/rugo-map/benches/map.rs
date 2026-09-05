//! The per-operation cost of the map, which is the other half of M1's gate.
//!
//! `tests/memory.rs` bounds what an entry costs to hold. This bounds what one costs to touch. Both have to hold at once, because a table that is small because every lookup walks a chain has not solved anything, and one that is fast because it is mostly empty has not either.
//!
//! # What these numbers are and are not
//!
//! These are single-threaded costs with no socket, no parser and no lock contention: one call into the map, measured. The throughput target in `SCOREBOARD.md` is a whole server under memtier and is a different measurement with syscalls in it, so a number here does not convert into a number there. What it does is say which of the two is the limit. If a `GET` costs forty nanoseconds here and the server serves three hundred thousand a second a core, the map is not what is holding the server back and the poller is.
//!
//! # Why the sizes are what they are
//!
//! At a thousand entries the whole table and its arena are in L2 and the measurement is the instruction count. At a hundred thousand it is around L3. At a million every probe is a trip to memory and the measurement is really the cache hierarchy, which is exactly where a five-byte slot is supposed to beat a ten-byte one: twice as many slots to a cache line is twice as many candidates for the same miss.
//!
//! Reading only the first size flatters the design and reading only the last hides what the SIMD probe is doing. All three are reported.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rugo_map::Map;
use std::hint::black_box;

/// A key of the shape memtier generates, and the shape `tests/memory.rs` measures.
fn key_of(n: usize) -> Vec<u8> {
    format!("memtier-{n:016}").into_bytes()
}

/// The value every benchmark stores, at the size the harness's default workload uses.
const VALUE: [u8; 100] = [0xa5; 100];

/// Shards, fixed across every benchmark here.
///
/// Sixty-four rather than the server's default of four thousand, because this file measures one thread and a shard count only matters once there are threads to keep off each other. Four thousand shards under one thread would measure four thousand nearly empty tables, which is a memory question and `tests/memory.rs` already asks it.
const SHARDS: usize = 64;

/// Entry counts, chosen against the cache hierarchy rather than for round numbers.
const SIZES: [usize; 3] = [1_000, 100_000, 1_000_000];

/// The sizes to actually run.
///
/// `deep.yml` runs every benchmark once to check it still builds and still runs, and gets nothing out of filling a million-key map to look one thing up in it. `RUGO_BENCH_SMOKE` is that run; anything measuring anything leaves it unset.
fn sizes() -> &'static [usize] {
    if smoke() { &SIZES[..1] } else { &SIZES }
}

/// The sizes that build a whole map inside the timing loop, which is the most expensive thing in this file and the first thing a smoke run should drop.
fn fill_sizes() -> &'static [usize] {
    if smoke() { &SIZES[..0] } else { &SIZES[..2] }
}

fn smoke() -> bool {
    std::env::var_os("RUGO_BENCH_SMOKE").is_some()
}

/// A map holding `n` entries, with no ceiling.
fn filled(n: usize) -> Map {
    let map = Map::new(SHARDS, 0);
    for i in 0..n {
        assert!(
            map.set(&key_of(i), &VALUE, None, None).is_ok(),
            "the map refused a write with no ceiling set"
        );
    }
    map
}

/// A thousand and twenty-four keys drawn from `[0, n)` in an order that is not the order they were written in.
///
/// Stepping by a prime that does not divide `n` visits the whole range without repeating, which gives a probe sequence that defeats the prefetcher the way a real key stream does, without a random number generator inside the timing loop.
fn probes(n: usize) -> Vec<Vec<u8>> {
    (0..1024).map(|i| key_of(i * 7919 % n)).collect()
}

/// Keys that are not in a map of `n` entries.
fn misses(n: usize) -> Vec<Vec<u8>> {
    (0..1024).map(|i| key_of(n + i)).collect()
}

fn bench_get(c: &mut Criterion) {
    let mut g = c.benchmark_group("get");
    g.throughput(Throughput::Elements(1));

    for &n in sizes() {
        let map = filled(n);
        let hits = probes(n);
        let gone = misses(n);

        g.bench_with_input(BenchmarkId::new("hit", n), &n, |bench, _| {
            let mut i = 0usize;
            bench.iter(|| {
                i = (i + 1) & 1023;
                black_box(map.get(black_box(&hits[i]), <[u8]>::len))
            });
        });

        // A miss is the cheaper path and the more common one under a cache workload that is doing its job, because a miss stops at the first group with an empty slot in it and never reads an entry at all. Measured separately for that reason: a benchmark that mixed the two would report their ratio as if it were a cost.
        g.bench_with_input(BenchmarkId::new("miss", n), &n, |bench, _| {
            let mut i = 0usize;
            bench.iter(|| {
                i = (i + 1) & 1023;
                black_box(map.get(black_box(&gone[i]), <[u8]>::len))
            });
        });
    }

    g.finish();
}

fn bench_set(c: &mut Criterion) {
    let mut g = c.benchmark_group("set");
    g.throughput(Throughput::Elements(1));

    // Overwriting a key that is already there, which is the write a cache client issues most of the time and the only one that does not grow anything. The old entry is freed onto the arena's free list and the new one is taken straight back off it, so this also measures whether the size classes are doing their job.
    for &n in sizes() {
        let map = filled(n);
        let hits = probes(n);
        g.bench_with_input(BenchmarkId::new("overwrite", n), &n, |bench, _| {
            let mut i = 0usize;
            bench.iter(|| {
                i = (i + 1) & 1023;
                black_box(map.set(black_box(&hits[i]), &VALUE, None, None).is_ok())
            });
        });
    }

    // Filling an empty map, which is a different operation and has to be measured as one: it carries the table doublings and the arena's segment allocations, and averaging it together with an overwrite gives a number that describes neither. The amortised cost of a doubling is in here, which is why it is per entry rather than per fill.
    for &n in fill_sizes() {
        let keys: Vec<Vec<u8>> = (0..n).map(key_of).collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("fill", n), &n, |bench, _| {
            bench.iter_batched_ref(
                || Map::new(SHARDS, 0),
                |map| {
                    for key in &keys {
                        black_box(map.set(black_box(key), &VALUE, None, None).is_ok());
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }

    g.finish();
}

fn bench_remove(c: &mut Criterion) {
    let mut g = c.benchmark_group("remove");
    g.throughput(Throughput::Elements(1));

    // Removing every key of a filled map, batched, because removing a key that has already been removed is a miss and would quietly turn this into a second copy of `get/miss`. The tombstone the removal leaves is part of what is being measured: a table that never rehashes fills with them.
    for &n in fill_sizes() {
        let keys: Vec<Vec<u8>> = (0..n).map(key_of).collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("drain", n), &n, |bench, _| {
            bench.iter_batched_ref(
                || filled(n),
                |map| {
                    for key in &keys {
                        black_box(map.remove(black_box(key)));
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }

    g.finish();
}

fn bench_evict(c: &mut Criterion) {
    let mut g = c.benchmark_group("evict");
    g.throughput(Throughput::Elements(1));

    // Writing into a map that is already at its ceiling, so every insert of a key that is not there has to find a victim first. That is the state a cache spends its whole life in, and it is the only state in which the two-random sampler costs anything at all.
    //
    // The key stream is twice the population the ceiling allows, so about half the writes are overwrites and half are inserts that evict. A stream of pure inserts would be the worse case but would also need a fresh key per iteration, and criterion runs tens of millions of them.
    for &n in sizes() {
        // About what an entry is charged: a twenty-four byte key, a hundred bytes of value, a four byte header rounded up to the eight byte grain, and six and a half bytes of index. Half of `n` entries fit, so the other half of the key stream evicts.
        let ceiling = n / 2 * 136;
        let map = Map::new(SHARDS, ceiling);
        for i in 0..n {
            assert!(
                map.set(&key_of(i), &VALUE, None, None).is_ok(),
                "the map refused a write while evicting"
            );
        }
        assert!(
            map.len() < n,
            "the ceiling of {ceiling} bytes did not evict anything out of {n} entries, so this benchmark is measuring an ordinary set"
        );

        let keys = probes(n);
        g.bench_with_input(BenchmarkId::new("churn", n), &n, |bench, _| {
            let mut i = 0usize;
            bench.iter(|| {
                i = (i + 1) & 1023;
                black_box(map.set(black_box(&keys[i]), &VALUE, None, None).is_ok())
            });
        });
    }

    g.finish();
}

criterion_group!(benches, bench_get, bench_set, bench_remove, bench_evict);
criterion_main!(benches);
