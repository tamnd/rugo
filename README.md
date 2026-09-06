# rugo

A cache server. Keys, byte-string values, a time to live, and nothing else.

It speaks RESP, so `redis-cli` connects to it and a Redis client library drives it. It does not persist, replicate, or do anything a database does, and the places where it answers a Redis client differently are written down in [divergences.md](divergences.md) rather than left to be discovered.

Apache-2.0. The architecture is taken from [tidwall/pogocache](https://github.com/tidwall/pogocache) — thousands of shards, a lock per shard, a poller per thread, entries packed rather than allocated one at a time — and none of its code is. Where the design deliberately parts company with it, and what that trade is expected to cost, is in the second half of [divergences.md](divergences.md).

## What it is trying to do

Twice the throughput of any comparable server, and half the memory per entry.

That is a hard target and it is stated as one. Whether it is met is a measurement, and the measurement lives in [SCOREBOARD.md](SCOREBOARD.md), which is generated from measurement files committed alongside it and checked on every push so a ratio cannot be typed in by hand.

**As of the second measurement it says: memory measured, throughput not.** On overhead per entry rugo is smaller than Garnet, memcached, Redis, Valkey, pogocache and yo by more than the factor of two the gate asks for, and smaller than Dragonfly by less than it. Two memory sweeps are published rather than one, a day apart on the same host, because the second is the first with the arena's reserve moved out of resident memory and the gap between them is what that was worth. No throughput sweep has run, so half the gate has no row at all rather than a row that guesses.

The numbers come from [tamnd/cache-bench](https://github.com/tamnd/cache-bench) driving `memtier_benchmark` on named hosts, against memcached, Redis, Valkey, Dragonfly, Garnet, pogocache and yo. Nothing measured on a laptop goes in: two runs of identical code on the development machine here disagreed by 198 percent, which is not a measurement.

The memory half needs two numbers and they are different claims. Total bytes per entry is the whole resident set divided by the keys in it, which is what a machine has to have. Overhead bytes per entry is what is left after the keys and the values themselves, which is what the design is actually about. At a hundred-odd bytes of payload per key no index can halve the first of those, whatever it does to the second, and the scoreboard reports both in separate columns rather than quoting whichever one flatters.

## Running it

Linux and macOS. The poller is epoll and kqueue and the listener is a unix socket, and the Windows equivalent is IOCP, which would be a second server rather than a build of this one. The crates below the server — the hash, the arena, the map and the parser — are portable and are built and tested on Windows too, so a change that only compiles on one target is caught, but there is no Windows binary and no `rugo.exe`.

```
cargo build --release
./target/release/rugo --maxmemory 4gb
```

```
rugo, a cache server

Usage: rugo [options]

Options:
  --port <n>            TCP port to listen on (default 6379)
  --no-port             do not listen on TCP at all
  --unixsocket <path>   also listen on a unix socket
  --threads <n>         serving threads (default: one per core)
  --shards <n>          map shards, rounded up to a power of two (default: 16 a thread)
  --maxmemory <size>    byte ceiling, as a number or with kb/mb/gb (default: none)
  --uring <auto|yes|no> use io_uring where the kernel has it (default auto)
  --version             print the version and exit
  --help                print this and exit
```

`--uring` is accepted and does nothing yet.

## The commands

`GET SET SETEX PSETEX MGET MSET INCR DECR INCRBY DECRBY STRLEN DEL UNLINK EXISTS EXPIRE PEXPIRE EXPIREAT PEXPIREAT TTL PTTL PERSIST DBSIZE FLUSHALL FLUSHDB PING ECHO HELLO SELECT RESET QUIT INFO COMMAND CONFIG CLIENT SHUTDOWN`

`SET` takes `EX PX EXAT PXAT KEEPTTL NX XX GET`. RESP2 and RESP3 are both spoken and `HELLO 3` switches. Inline commands and pipelining work, so `PING\r\n` on a raw socket answers.

A command that is not in that list is an error rather than a silence, so a client finds out by being told rather than by waiting.

## The crates

The server is the binary; everything under it is a library somebody could use on its own.

| crate | what it is |
|---|---|
| `rugo-hash` | a seeded hash for short keys, no dependencies |
| `rugo-arena` | a slab allocator addressed by `u32` offset, with size classes |
| `rugo-map` | the sharded table, the entry encoding, expiry and eviction |
| `rugo-resp` | RESP2 and RESP3, parse and encode |
| `rugo-net` | epoll and kqueue, connection state |
| `rugo-server` | command dispatch, threads, configuration |
| `rugo` | the binary |

`rugo-map` is the one worth using directly: a concurrent cache with a memory ceiling and no server attached.

## Building on it

The floor is Rust 1.94 and the pinned toolchain is 1.98. Edition 2024.

`cargo test --workspace` is the whole test suite. `cargo xtask scoreboard` regenerates `SCOREBOARD.md` and `cargo xtask check` says whether the committed one is what the generator would write.

Unsafe is not banned here, because a SIMD probe and a slab arena cannot be written without it. What stands in for a ban is a safety comment on every block, denied by lint, plus Miri under both borrow models, loom on the shard lock and two fuzz targets, all of which run nightly in `deep.yml`.

[CHANGELOG.md](CHANGELOG.md) is what each release changed and what it costs you. [RELEASING.md](RELEASING.md) is what the version numbers mean. [PERFORMANCE.md](PERFORMANCE.md) is the optimisations that were written and measured and thrown away, which is the half of the work that otherwise gets done twice.
