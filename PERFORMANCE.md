# What was measured and not kept

Every optimisation that landed is in [CHANGELOG.md](CHANGELOG.md), with the machine and the profile it was measured on beside it. This file is the other half: the changes that were written, measured and thrown away.

They are here because a rejected optimisation leaves nothing behind in the code, so the next person to read the same profile has the same idea, writes the same patch and spends the same day finding out. A number that says an idea does not pay is worth about as much as one that says it does, and it is only worth anything if it is written down.

The rule from the changelog holds here too. Any number carries the machine and the profile it was measured on, or it does not go in.

The shape almost everything below was measured on: `server3`, which is an eight core EPYC with a real PMU, one server thread, pipeline depth twenty five over a unix socket, five million keys of one to a thousand and twenty four bytes, every lookup hitting, counters restricted to user space, and the two binaries run in alternation rather than one after the other because the box is shared and its load moves over an hour.

## The ring against the poller, on boxes that could not tell them apart

Not a change and not thrown away. This is a measurement that failed to say anything, written down so the next attempt begins on a quiet host instead of on these three.

The question is which of the two serving loops is faster, and the answer decides what `--uring` should default to. `cargo xtask load` was written for it: it drives a running server over a unix socket and reads the server's own processor time an operation out of `/proc`, which is the figure that survives a run where the load generator rather than the server is the thing in short supply.

`server1` is four cores with about a core and a half of somebody else's work on it. First attempt: the server pinned to two cores with two threads, the generator pinned to the other two, four connections at pipeline depth twenty five, a million operations a run, two hundred thousand keys of a hundred bytes, one write in eleven, five rounds of `--uring no` against `--uring yes` alternated. The ring read higher processor time an operation in four rounds of the five, 2.88 microseconds against 1.86 at the median, which reads like an answer until the second attempt.

Second attempt: one server thread on one core, the generator on another, eight rounds each of four binaries. The lowest processor time an operation over the eight was 0.74 for the poller and 0.99 for the ring, and the median was 1.37 against 1.51. The same direction, a fifth of the size, and in every row of both attempts the generator's own processor time an operation tracked the server's within a few per cent, which is what a run looks like when what it is measuring is the box rather than the program.

The other hosts were worse. `server2` is six cores and ran a crawler at four hundred per cent of a core through the whole window. `server3` is the quiet eight core box and is most of a day into the sweep that produces the first published throughput numbers. `gpc` restarts every few minutes under load it does not control.

Two of the four binaries in the second attempt were the pacing change in the changelog, the clock tick and the sweep moved off the turn and onto a millisecond, and they were inside the same noise. That one could not have shown there whatever the box was doing: this load sets no expiries, and a shard with nothing in it to expire is already skipped without being read, so the work the change removes is a lock and a counter rather than the walk it removes on a keyspace that does use TTLs.

The lesson is about the harness rather than about either loop. Processor time an operation is the right metric and it is not enough on its own: a run only says something if the generator's share of it stays flat while the server's moves. Both attempts are worth about half an hour on `server3` once its sweep is done, and nothing at all before then.

## What a shard count costs in memory, which is less than it looks

Not a change. This is the price list a throughput change would be spent against, measured before anybody spends it, and it came out somewhere other than where it was expected to.

A server picks sixteen shards a thread, so eight threads gets a hundred and twenty-eight. Raising that is one of the few levers left on the lock side, and the reason nobody had pulled it is that more shards was assumed to cost memory: each shard owns a table and an arena, and four thousand of them sounded like four thousand indexes and four thousand part-used segments. Half of that turns out to be wrong.

The index does not move at all. `Map::with_seed` rounds the shard count up to a power of two, and a shard's table capacity is a power of two chosen from the entries that land in it, so doubling the shard count halves what a shard holds and halves the capacity it grows to, and the two cancel exactly. At ten million entries the whole index is 83,886,080 bytes at a hundred and twenty-eight shards, and at two hundred and fifty-six, and at five hundred and twelve, and at one thousand and twenty-four, at two thousand and forty-eight and at four thousand and ninety-six. To the byte, every time: two to the twenty-fourth slots at five bytes each. That is arithmetic rather than a measurement and it holds on any machine, which is why no machine is named for it.

What does move is arena slack, and that is the whole of the price. Measured on `server1`, which is four cores with other tenants on it, a server started with `--threads 8 --shards N --no-port` over a unix socket and filled by `cargo xtask load` with ten million keys of eight bytes, reading `used_memory` out of `INFO` and `VmHWM` out of `/proc/<pid>/status` and dividing the difference by the entries. A memory high water mark is far less sensitive to a noisy box than a cycle count is, which is why this one was worth taking there and the throughput half was not.

| shards | bytes an entry beyond what the map accounts for |
| --- | --- |
| 128 | 0.56, 0.57, 0.61 |
| 256 | 0.79, 0.83, 0.84, 0.94 |
| 512 | 1.74, 1.90, 1.97, 2.03, 2.03 |
| 1024 | 1.90, 1.92, 1.97 |
| 2048 | 2.19, 2.23, 2.29 |
| 4096 | 2.78, 2.82, 2.86 |

Two of the runs in the two hundred and fifty-six row were asked for a hundred and ninety-two, and two of the runs in the five hundred and twelve row were asked for three hundred and eighty-four. They are in those rows because that is what they got: the count is rounded up to a power of two, so a hundred and ninety-two is two hundred and fifty-six and the reading says so. It is worth knowing that `--shards 192` is not a setting.

So going from the hundred and twenty-eight an eight thread server picks by default to four thousand and ninety-six costs about two and a quarter bytes an entry. Against an overhead of around twenty that is real and it is not a wall, and the memory gate would survive it.

The wall is on the other side, and it was already measured. `SHARDS_PER_THREAD` in `rugo-server`'s `config.rs` carries a reading from `server3`, one thread, five million eight byte entries: 3525 cycles a lookup at four thousand and ninety-six shards, 2477 at five hundred and twelve and 2113 at sixty-four, with the instruction count identical to the digit at all three. Nothing about the work changed, only where it landed. That is why the default follows the thread count instead of being four thousand, and it means the lever points the other way from where it was reached for.

What is still open is whether that one thread result holds at eight, where the locks are actually being shared and a coarse count has something to lose. `cargo xtask ab` at a hundred and twenty-eight against one thousand and twenty-four and four thousand and ninety-six, at eight threads, on `server3` once its sweep is done, is the run that answers it. The memory column above is already filled in, so whatever that run says, the trade is a known one in both directions.

## A batch of parsed commands rather than a batch of asked-for keys

Never opened. The idea it was competing with landed instead, and this is the version of it that did not work.

Looking several commands ahead and asking the map for their cache lines pays, and by a lot. The question this settles is how the connection should hold the commands it has looked ahead at. The obvious answer is to parse them: a pipelined read buffer holds several whole commands, so parse eight of them into eight argument lists, ask for all eight keys, then run all eight, and every command is parsed exactly once. The version that shipped instead walks the framing twice, once to find the key and again to parse the command when its turn comes, which is strictly more work.

The obvious answer was slower than doing nothing at all. Every figure below is `server3`, one thread, five million keys of one to eight bytes, every lookup hitting, three interleaved rounds, and each group of figures comes from one run so its baseline is its own.

Parsing eight commands ahead cost 3133 cycles an operation against 2588 for the unchanged server, and 23.7 cache misses against 14.9. In a second run against a quieter box it was 2531 against 2101 and 22.6 misses against 13.0. What settles where that came from is a third binary with the same restructuring and the hints compiled out, which in that same run read 3788 cycles and 23.6 misses, worse than either. The extra misses are the batch of argument lists, and the hints were paying for them rather than causing them.

Depth is where it shows. The same code holding four commands ahead instead of eight reads 1765 cycles and 14.3 misses, and holding two reads 1783 and 14.2, both about a third under the unchanged server rather than a fifth over it. So the cost is not the deferral of execution, which is the same at every depth, and it is not the hints, which are the same too. It is the eight argument lists themselves, each a small heap allocation of its own, all live at once, across sixty-four connections.

A miss attribution profile says the same thing in the other direction. On the unchanged server `Conn::execute` takes 6.5 per cent of the misses and the parser takes less than half a per cent; on the eight deep batch `execute` takes 20.1 per cent and the parser 4.6, and the map's share falls not because the map got cheaper but because the connection got dearer.

Depth four would have shipped a magic number one step away from a cliff nobody has explained, on a box that is not the only box this will run on. The look-ahead that shipped keeps one argument list at any depth, was measured at four and eight with no difference between them, and costs about three hundred instructions an operation for the second walk over the framing, which is the price of not having a cliff.

## A lookup that starts before the command in front of it is done

Closed as [#6](https://github.com/tamnd/rugo/pull/6).

A pipelined batch holds several commands and the server reads them one at a time, so every lookup starts cold. The idea was to parse one command ahead of the one being served, take its key, and start fetching the first cache line that key's probe will want, so the fetch runs underneath the serving of the command in front of it instead of stalling after it.

It was tried twice. Asking for L1 measured 6021 cycles an operation against 4994, with cache misses up from 39.9 to 41.1 rather than down. A whole command happens between the hint and the load it is for, and serving a command copies a value, which streams enough bytes through L1 to throw the hinted line out before anything reads it, so the line was fetched twice and the hint was worse than nothing.

Asking for L2 fixed that half. Over four interleaved pairs it measured 5239 cycles an operation against 4784, instructions 1287 against 1206, and cache misses 37.24 against 38.47. The miss count finally moved the right way and it did not come close to paying for itself.

What was wrong was the size of the prize rather than the mechanism. The control bytes for this working set come to about ten megabytes over thousands of shards, and this box has more last level cache than that, so the line being named is usually a hit in L3 rather than a trip to memory. The hint therefore saves tens of cycles, while the hash and the argument scan that compute the address cost eighty one instructions that were not being run before.

There is a second lesson in it about the profile that suggested it. A cache miss attribution profile put `entry_at` at 20.63 per cent of all misses, `find` at 17.49 and `Map::get` at 6.33, and that was read as the lookup being about forty five per cent of the misses in a get. It is not. `entry_at` is where the value bytes are reached, so most of what it counts is the copy rather than the probe. A one kilobyte value is sixteen lines read and sixteen written, which is most of the thirty eight misses an operation takes in this shape, and naming a control line in advance does not touch any of them.

Worth trying again only for a workload whose values are small enough that the probe is most of the work, and only if the address can be had without hashing the key twice.
