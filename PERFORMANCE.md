# What was measured and not kept

Every optimisation that landed is in [CHANGELOG.md](CHANGELOG.md), with the machine and the profile it was measured on beside it. This file is the other half: the changes that were written, measured and thrown away.

They are here because a rejected optimisation leaves nothing behind in the code, so the next person to read the same profile has the same idea, writes the same patch and spends the same day finding out. A number that says an idea does not pay is worth about as much as one that says it does, and it is only worth anything if it is written down.

The rule from the changelog holds here too. Any number carries the machine and the profile it was measured on, or it does not go in.

The shape almost everything below was measured on: `server3`, which is an eight core EPYC with a real PMU, one server thread, pipeline depth twenty five over a unix socket, five million keys of one to a thousand and twenty four bytes, every lookup hitting, counters restricted to user space, and the two binaries run in alternation rather than one after the other because the box is shared and its load moves over an hour.

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
