//! The map, against a sequence of operations nobody chose.
//!
//! The table is a Swiss table over a slab arena: a control byte and a four-byte offset per slot, entries packed head to tail with no allocator header, and a free list per size class. Every one of those is a place where an index can be computed wrongly, and none of them is checked at run time on the hot path, because checking would be most of the cost of a lookup.
//!
//! So the checking happens here instead. A model — an ordinary `HashMap` — is kept beside the real one and every operation is applied to both, which turns a wrong offset into a value that does not match rather than into a benchmark result nobody can explain. Expiry is deliberately left out of the model: the clock moves on its own, so a key with a deadline may vanish between two operations and a model that insisted otherwise would fail for a reason that is not a bug.
//!
//! Run under a sanitizer by `cargo fuzz`, so a read that runs off the end of the arena is caught even when the bytes it read happened to be the right ones.

#![no_main]

use std::collections::HashMap;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rugo_map::{Expiry, Map, When};

/// One thing to do to the map.
#[derive(Arbitrary, Debug)]
enum Op {
    /// Store a value with no expiry, which is the only kind the model can follow.
    Set { key: Vec<u8>, value: Vec<u8> },
    /// Store one with an expiry, which the model forgets about on purpose.
    SetTimed {
        key: Vec<u8>,
        value: Vec<u8>,
        seconds: u16,
    },
    /// Store one only if the key is absent or only if it is present.
    SetWhen {
        key: Vec<u8>,
        value: Vec<u8>,
        present: bool,
    },
    Get { key: Vec<u8> },
    Remove { key: Vec<u8> },
    Increment { key: Vec<u8>, by: i32 },
    /// Take a key's expiry away, which puts it back under the model's rules.
    Persist { key: Vec<u8> },
    /// Reclaim some expired slots, which is what moves entries around.
    Sweep,
    Clear,
}

/// What a fuzz case may ask for, so that the shard count and the ceiling are part of what is being explored.
#[derive(Arbitrary, Debug)]
struct Case {
    /// One shard finds crowding, many find the sharding arithmetic.
    shards: u8,
    /// Nought for no ceiling. Anything else makes eviction part of the run, and an evicted key is one the model has to stop expecting.
    tight: bool,
    ops: Vec<Op>,
}

/// Keys are kept short so that a case spends its bytes on the sequence rather than on one enormous key, and values short for the same reason. The arena's size classes are what care about the length, and they are all well under this.
const LIMIT: usize = 64;

fuzz_target!(|case: Case| {
    if case.ops.len() > 512 {
        return;
    }
    let shards = usize::from(case.shards).max(1);
    // A ceiling small enough that a few hundred short entries reach it, so that eviction runs rather than being a branch nothing takes.
    let maxmemory = if case.tight { 4096 } else { 0 };
    // A fixed seed, so a crash found here reproduces from the same input rather than from the same input and the same morning.
    let map = Map::with_seed(shards, maxmemory, 0x9e37_79b9_7f4a_7c15);
    let mut model: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();

    // Once anything can be evicted or expired, the model can only say what a key's value would be if it is still there, not whether it is. A single flag rather than a set, because the moment one key is at risk the map may evict any other to make room for it.
    let mut lossy = maxmemory != 0;

    for op in &case.ops {
        match op {
            Op::Set { key, value } => {
                if key.is_empty() || key.len() > LIMIT || value.len() > LIMIT {
                    continue;
                }
                if map.set(key, value, None, None).is_ok() {
                    model.insert(key.clone(), value.clone());
                } else {
                    // A refused write leaves the key as it was, whatever that was.
                    lossy = true;
                }
            }
            Op::SetTimed {
                key,
                value,
                seconds,
            } => {
                if key.is_empty() || key.len() > LIMIT || value.len() > LIMIT {
                    continue;
                }
                let at = map.clock().now().saturating_add(u32::from(*seconds));
                let _ = map.set(key, value, Some(at), None);
                // The clock may pass that moment at any point after this, so the model stops speaking for this key.
                model.remove(key);
                lossy = true;
            }
            Op::SetWhen {
                key,
                value,
                present,
            } => {
                if key.is_empty() || key.len() > LIMIT || value.len() > LIMIT {
                    continue;
                }
                let when = if *present { When::Present } else { When::Absent };
                let held = model.contains_key(key);
                match map.set_when(key, value, when, Expiry::Never, None) {
                    Ok(Some(_)) => {
                        model.insert(key.clone(), value.clone());
                        if !lossy {
                            assert_eq!(held, *present, "a conditional write took the wrong branch");
                        }
                    }
                    Ok(None) => {
                        if !lossy {
                            assert_eq!(held, !*present, "a conditional write refused wrongly");
                        }
                    }
                    Err(_) => lossy = true,
                }
            }
            Op::Get { key } => {
                let found = map.get(key, <[u8]>::to_vec);
                if !lossy && let Some(wanted) = model.get(key) {
                    assert_eq!(
                        found.as_ref(),
                        Some(wanted),
                        "a key read back as something other than what was written"
                    );
                }
                if found.is_some() && !lossy {
                    assert!(
                        model.contains_key(key),
                        "the map held a key that was never written"
                    );
                }
            }
            Op::Remove { key } => {
                let went = map.remove(key);
                let expected = model.remove(key).is_some();
                if !lossy {
                    assert_eq!(went, expected, "a removal disagreed about whether it removed");
                }
            }
            Op::Increment { key, by } => {
                if key.is_empty() || key.len() > LIMIT {
                    continue;
                }
                // An error is not a number, out of range, or no room, and each of those leaves the key as it was, so the model already agrees and there is nothing to do.
                if let Ok(value) = map.increment(key, i64::from(*by)) {
                    model.insert(key.clone(), value.to_string().into_bytes());
                }
            }
            Op::Persist { key } => {
                let _ = map.expire(key, None);
            }
            Op::Sweep => {
                map.sweep(64);
            }
            Op::Clear => {
                map.clear();
                model.clear();
                // Nothing is at risk in an empty map, so the model can speak again — unless the ceiling is still there.
                lossy = maxmemory != 0;
            }
        }

        // Invariants that hold whatever happened. A length that disagrees with the entries is the shape of a slot marked full that holds nothing, which is how a Swiss table probe starts reading somebody else's bytes.
        assert!(map.live_bytes() <= map.charged_bytes());
        assert!(map.charged_bytes() <= map.resident_bytes() + map.index_bytes());
        if map.is_empty() {
            assert_eq!(map.len(), 0);
        }
        if !lossy {
            assert_eq!(map.len(), model.len(), "the map and the model disagree on how many keys there are");
        }
    }

    // Everything written and still expected has to still be there at the end, which is the check that a rehash or a sweep did not quietly drop a live entry.
    if !lossy {
        for (key, value) in &model {
            assert_eq!(
                map.get(key, <[u8]>::to_vec).as_ref(),
                Some(value),
                "a key was written and then lost"
            );
        }
    }
});
