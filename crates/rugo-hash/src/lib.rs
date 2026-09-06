//! The seeded short-key hash rugo places entries with.
//!
//! One 64-bit value does three jobs, and the bit ranges do not overlap:
//!
//! - the top bits pick the shard, so two keys in the same shard already disagree in the bits the table indexes with;
//! - one byte out of the middle becomes the control tag a SIMD probe compares sixteen at a time;
//! - the low bits pick the slot inside the shard.
//!
//! Splitting one hash three ways only works if the whole 64 bits are well mixed, which is why every path here ends in a fold of a 64x64 to 128 multiply rather than in a cheaper truncation.
//!
//! The seed is drawn once at startup from the operating system. Two processes therefore place the same key differently, which is what stops a client from choosing keys that all land in one shard and turning the table into a linked list. [`hash`] is not a cryptographic hash and the seed does not make it one; it raises the cost of finding a collision set from free to impractical, and that is all it is for.
//!
//! ```
//! let seed = rugo_hash::seed();
//! let h = rugo_hash::hash(b"user:1", seed);
//! assert_eq!(h, rugo_hash::hash(b"user:1", seed));
//! ```

#![forbid(unsafe_code)]
// Every function here is a handful of instructions on the hottest path in the program, called once per command, and the whole design is that a short key costs two loads and two multiplies. A call that survived the inlining decision would cost more than the hash.
#![expect(
    clippy::inline_always,
    reason = "leaf functions of a few instructions each"
)]

use core::hash::Hasher;

/// Odd 64-bit constants with roughly half their bits set.
///
/// They are the fractional digits of pi, cube roots and the golden ratio, which is the usual way of saying that nothing was chosen to make any particular input behave.
const P0: u64 = 0x243f_6a88_85a3_08d3;
const P1: u64 = 0x1319_8a2e_0370_7344;
const P2: u64 = 0xa409_3822_299f_31d0;
const P3: u64 = 0x082e_fa98_ec4e_6c89;

/// Multiply two words to 128 bits and fold the halves together.
///
/// This is the only mixing primitive in the file. One multiply moves every input bit into roughly half the output bits, and the xor of the two halves is what stops the high bits of the product from being thrown away.
#[inline(always)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "taking each half of the product is what folding means"
)]
const fn fold(a: u64, b: u64) -> u64 {
    let wide = (a as u128).wrapping_mul(b as u128);
    (wide as u64) ^ ((wide >> 64) as u64)
}

/// The first eight bytes of `at`, little endian.
///
/// Callers are responsible for `at.len() >= 8`; the slice indexing below is what enforces it.
#[inline(always)]
fn word(at: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&at[..8]);
    u64::from_le_bytes(buf)
}

/// The first four bytes of `at`, little endian, widened.
#[inline(always)]
fn half(at: &[u8]) -> u64 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&at[..4]);
    u64::from(u32::from_le_bytes(buf))
}

/// Hash `key` under `seed`.
///
/// The length is mixed in rather than merely walked over, so that `b"ab"` and `b"ab\0"` disagree, and every branch below reads a fixed number of bytes from the front and the back of the key rather than looping. Keys of sixteen bytes or fewer, which is most of what a cache holds, take two loads and two multiplies.
#[must_use]
pub fn hash(key: &[u8], seed: u64) -> u64 {
    let len = key.len();
    // The length enters first so that the constants a short key is folded against already depend on it.
    let start = seed ^ fold(seed ^ P0, (len as u64) ^ P1);

    let (a, b) = match len {
        0 => (P2, P3),
        // Front, middle and back byte. A one byte key uses the same byte three times, which is fine: the fold, not the packing, is what separates them.
        1..=3 => {
            let packed = (u64::from(key[0]) << 48)
                | (u64::from(key[len >> 1]) << 24)
                | u64::from(key[len - 1]);
            (packed ^ P2, start ^ P3)
        }
        // Overlapping four byte reads from each end. For len 4 they are the same four bytes, which the differing multiplicands still separate.
        4..=7 => (half(key) ^ P2, half(&key[len - 4..]) ^ P3),
        // Overlapping eight byte reads from each end. This is the common case and it is two loads.
        8..=16 => (word(key) ^ P2, word(&key[len - 8..]) ^ P3),
        // Four overlapping eight byte reads: two from the front, two from the back.
        17..=32 => (
            fold(word(key) ^ P2, word(&key[8..]) ^ P3),
            fold(word(&key[len - 16..]) ^ P0, word(&key[len - 8..]) ^ P1),
        ),
        _ => long(key, start),
    };

    fold(a ^ start, b ^ P1.wrapping_add(len as u64))
}

/// The loop, for keys over thirty-two bytes.
///
/// Four lanes rather than one, because the fold is a multiply and a multiply has latency that a single accumulator would serialise on. The last thirty-two bytes are folded in unconditionally and may overlap the loop's final block, which costs one extra pass over at most thirty-two bytes and removes the tail branch entirely.
#[inline]
fn long(key: &[u8], start: u64) -> (u64, u64) {
    let mut lanes = [start ^ P0, start ^ P1, start ^ P2, start ^ P3];
    let mut rest = key;

    while rest.len() >= 32 {
        lanes[0] = fold(lanes[0] ^ word(rest), P0);
        lanes[1] = fold(lanes[1] ^ word(&rest[8..]), P1);
        lanes[2] = fold(lanes[2] ^ word(&rest[16..]), P2);
        lanes[3] = fold(lanes[3] ^ word(&rest[24..]), P3);
        rest = &rest[32..];
    }

    let tail = &key[key.len() - 32..];
    lanes[0] ^= word(tail);
    lanes[1] ^= word(&tail[8..]);
    lanes[2] ^= word(&tail[16..]);
    lanes[3] ^= word(&tail[24..]);

    (fold(lanes[0], lanes[1] ^ P2), fold(lanes[2], lanes[3] ^ P3))
}

/// A seed from the operating system.
///
/// Called once at startup. If the platform will not produce one, the fallback is the address of a heap allocation mixed with the clock, which is weaker than the real thing and still not something a remote client can guess.
#[must_use]
pub fn seed() -> u64 {
    if let Some(from_os) = os_seed() {
        return from_os;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| fold(d.as_secs(), u64::from(d.subsec_nanos())));
    let boxed = Box::new(0u8);
    let address = std::ptr::from_ref::<u8>(&*boxed) as u64;
    fold(now ^ P0, address ^ P1) | 1
}

#[cfg(unix)]
fn os_seed() -> Option<u64> {
    use std::io::Read as _;

    let mut buf = [0u8; 8];
    let mut file = std::fs::File::open("/dev/urandom").ok()?;
    file.read_exact(&mut buf).ok()?;
    Some(u64::from_le_bytes(buf))
}

#[cfg(not(unix))]
fn os_seed() -> Option<u64> {
    None
}

/// A [`Hasher`] over [`hash`], for the places the standard library asks for one.
///
/// It buffers, because [`hash`] reads a whole key at once and the [`Hasher`] contract is a stream. Nothing on rugo's hot path uses it; the map calls [`hash`] directly.
#[derive(Debug, Clone)]
pub struct RugoHasher {
    seed: u64,
    buf: Vec<u8>,
}

impl RugoHasher {
    /// A hasher seeded with `seed`.
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self {
            seed,
            buf: Vec::new(),
        }
    }
}

impl Hasher for RugoHasher {
    fn finish(&self) -> u64 {
        hash(&self.buf, self.seed)
    }

    fn write(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }
}

/// Two tests here are skipped under Miri.
///
/// Both are statistical: one hashes a million keys and counts collisions, the other hashes four hundred thousand and looks at how they land across four thousand shards. Neither survives being made smaller, because the sample size is the claim rather than an implementation detail of it, and neither finishes at full size under an interpreter that runs every instruction. This crate is `forbid(unsafe_code)` besides, so there is nothing here for Miri to find that the ordinary test run does not already say.
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Keys chosen to hit every branch in `hash`, including both sides of each length boundary.
    fn corpus() -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = Vec::new();
        for len in 0..=80usize {
            out.push((0..len).map(|i| u8::try_from(i % 251).unwrap()).collect());
        }
        for name in [
            "",
            "a",
            "ab",
            "key",
            "user:1",
            "user:2",
            "memtier-1234567890",
        ] {
            out.push(name.as_bytes().to_vec());
        }
        out
    }

    #[test]
    fn the_same_key_and_seed_give_the_same_value() {
        for key in corpus() {
            assert_eq!(hash(&key, 1), hash(&key, 1), "key of {} bytes", key.len());
        }
    }

    #[test]
    fn a_different_seed_moves_every_key() {
        // Not a probabilistic claim about one key. Across the corpus, two seeds agreeing on any key at all would be a sign the seed is being dropped on some branch.
        for key in corpus() {
            assert_ne!(
                hash(&key, 1),
                hash(&key, 2),
                "seed ignored for a key of {} bytes",
                key.len()
            );
        }
    }

    #[test]
    fn keys_that_differ_by_a_trailing_zero_differ() {
        // The failure this catches is a hash that walks the bytes and never mixes the length, which collides "ab" with "ab\0" and is a real bug in more than one published hash.
        for len in 0..40usize {
            let short: Vec<u8> = (0..len).map(|i| u8::try_from(i % 251).unwrap()).collect();
            let mut long = short.clone();
            long.push(0);
            assert_ne!(hash(&short, 7), hash(&long, 7), "length ignored at {len}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "a collision count that needs its million keys")]
    fn a_million_sequential_keys_do_not_collide() {
        // The shape memtier generates: one prefix and an ascending number. A hash that is weak on this pattern is a hash that is weak on the benchmark.
        let mut seen = HashSet::with_capacity(1 << 20);
        for i in 0..1_000_000u32 {
            let key = format!("memtier-{i}");
            seen.insert(hash(key.as_bytes(), 0x5eed));
        }
        // 2^20 draws from 2^64 expects well under one collision. Any at all means the output is not using its whole range.
        assert_eq!(seen.len(), 1_000_000, "sequential keys collided");
    }

    #[test]
    fn every_output_bit_moves() {
        // Flipping one input bit should change about half the output bits. What this actually catches is an output bit that is constant, or one that no input reaches, which would silently halve the shard count or freeze the control tag.
        let mut ever_set = 0u64;
        let mut ever_clear = 0u64;
        for i in 0..4096u32 {
            let h = hash(&i.to_le_bytes(), 0xabcd_ef01);
            ever_set |= h;
            ever_clear |= !h;
        }
        assert_eq!(ever_set, u64::MAX, "some output bit is never set");
        assert_eq!(ever_clear, u64::MAX, "some output bit is never clear");
    }

    #[test]
    fn one_flipped_input_bit_changes_about_half_the_output() {
        let base = b"user:0000000000000000";
        let seed = 99;
        let from = hash(base, seed);
        let mut total = 0u32;
        let mut trials = 0u32;
        for byte in 0..base.len() {
            for bit in 0..8 {
                let mut other = base.to_vec();
                other[byte] ^= 1 << bit;
                total += (from ^ hash(&other, seed)).count_ones();
                trials += 1;
            }
        }
        let mean = f64::from(total) / f64::from(trials);
        // Ideal is 32. The window is wide because this is a smoke test for a broken mix, not a statistical proof of avalanche.
        assert!(
            (28.0..=36.0).contains(&mean),
            "mean of {mean} flipped bits per input bit"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "a spread claim that needs its four hundred thousand")]
    fn the_high_bits_that_pick_a_shard_spread() {
        // The map takes the top twelve bits as the shard for a default 4096 shard table. A hash whose high bits barely move would put every key in a handful of shards and every measurement that followed would be a measurement of one lock.
        let shards = 4096usize;
        let mut counts = vec![0u32; shards];
        for i in 0..400_000u32 {
            let h = hash(format!("memtier-{i}").as_bytes(), 0x1234);
            counts[(h >> 52) as usize % shards] += 1;
        }
        let empty = counts.iter().filter(|&&c| c == 0).count();
        let worst = counts.iter().copied().max().unwrap_or(0);
        // Just under a hundred per shard on average.
        assert_eq!(empty, 0, "{empty} of {shards} shards got no keys at all");
        assert!(worst < 200, "one shard took {worst} keys, the mean is ~98");
    }

    #[test]
    fn the_hasher_agrees_with_the_function() {
        for key in corpus() {
            let mut hasher = RugoHasher::with_seed(3);
            hasher.write(&key);
            assert_eq!(hasher.finish(), hash(&key, 3));
        }
    }

    #[test]
    fn the_seed_is_not_a_constant() {
        // Two calls returning the same number would mean the OS read failed and the fallback is degenerate.
        assert_ne!(seed(), seed());
    }
}
