//! How one cached entry is laid out in arena bytes.
//!
//! ```text
//! [flags:u8][klen:varint][vlen:varint][key][value][expiry:u32?][user flags:u32?]
//! ```
//!
//! Everything optional is at the end, so key and value stay adjacent to their lengths and a `GET` reads one contiguous run. The two trailing fields are present only when the flags byte says so, which is what keeps an entry with no expiry from paying four bytes to say it has none. Most cache entries have no expiry, and every one of them would have paid.
//!
//! Lengths are LEB128, so a key under 128 bytes costs one byte to describe and a value under 128 bytes costs one. The whole header for a typical entry is therefore three bytes: one flags, one key length, one value length. Pogocache's is one byte plus its own varints, and then it pays the system allocator's header on top, which is the difference [`rugo_arena`] exists to remove.
//!
//! Nothing here stores the hash. Re-reading the key to confirm a match costs a comparison that the control byte has already made unlikely, and storing three bytes of hash per entry to avoid it, which is what pogocache does, is three bytes rugo would rather not spend.

/// Set when the entry carries an expiry.
pub(crate) const HAS_EXPIRY: u8 = 1 << 0;

/// Set when the entry carries a memcache-style user flags word.
pub(crate) const HAS_USER_FLAGS: u8 = 1 << 1;

/// How many bytes the LEB128 encoding of `value` takes.
#[inline]
pub(crate) const fn varint_len(value: usize) -> usize {
    match value {
        0..=0x7f => 1,
        0x80..=0x3fff => 2,
        0x4000..=0x1f_ffff => 3,
        0x20_0000..=0xfff_ffff => 4,
        _ => 5,
    }
}

/// Write `value` as LEB128 at the front of `into` and return how many bytes it took.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is masked to seven bits before it is narrowed"
)]
pub(crate) fn put_varint(into: &mut [u8], mut value: usize) -> usize {
    let mut at = 0;
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            into[at] = byte;
            return at + 1;
        }
        into[at] = byte | 0x80;
        at += 1;
    }
}

/// Read a LEB128 value from the front of `from`, returning it and how many bytes it took.
#[inline]
pub(crate) fn get_varint(from: &[u8]) -> (usize, usize) {
    let mut value = 0usize;
    let mut shift = 0u32;
    let mut at = 0usize;
    loop {
        let byte = from[at];
        value |= (byte as usize & 0x7f) << shift;
        at += 1;
        if byte & 0x80 == 0 {
            return (value, at);
        }
        shift += 7;
    }
}

/// How many bytes an entry with these parts occupies.
#[inline]
#[must_use]
pub(crate) const fn size_of_entry(
    klen: usize,
    vlen: usize,
    expiry: bool,
    user_flags: bool,
) -> usize {
    1 + varint_len(klen)
        + varint_len(vlen)
        + klen
        + vlen
        + if expiry { 4 } else { 0 }
        + if user_flags { 4 } else { 0 }
}

/// Write a whole entry into `into`, which must be exactly [`size_of_entry`] bytes.
pub(crate) fn write(
    into: &mut [u8],
    key: &[u8],
    value: &[u8],
    expiry: Option<u32>,
    user_flags: Option<u32>,
) {
    let mut flags = 0u8;
    if expiry.is_some() {
        flags |= HAS_EXPIRY;
    }
    if user_flags.is_some() {
        flags |= HAS_USER_FLAGS;
    }

    into[0] = flags;
    let mut at = 1;
    at += put_varint(&mut into[at..], key.len());
    at += put_varint(&mut into[at..], value.len());
    into[at..at + key.len()].copy_from_slice(key);
    at += key.len();
    into[at..at + value.len()].copy_from_slice(value);
    at += value.len();

    if let Some(when) = expiry {
        into[at..at + 4].copy_from_slice(&when.to_le_bytes());
        at += 4;
    }
    if let Some(word) = user_flags {
        into[at..at + 4].copy_from_slice(&word.to_le_bytes());
    }
}

/// An entry's header, decoded.
///
/// Every read of an entry begins by walking the flags byte and the two varints, and until this existed the `GET` path walked them six times over: once so the arena could be told how long the entry was, once to compare the key, and then twice more for the expiry and the value, each of those preceded by another length walk of its own. Decoding once and carrying the answer is what took a hit from three times the cost of a miss down to a little over it.
///
/// Offsets rather than slices, so the whole thing is [`Copy`] and can be returned out of a function that borrowed the table to produce it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Head {
    /// The flags byte.
    pub flags: u8,
    /// What the flags byte and the two lengths took, which is where the key starts.
    pub header: usize,
    /// The key's length.
    pub klen: usize,
    /// The value's length.
    pub vlen: usize,
}

impl Head {
    /// How many bytes the whole entry occupies.
    #[inline]
    pub(crate) const fn size(&self) -> usize {
        self.header + self.klen + self.vlen + self.trailer()
    }

    /// What the two optional trailing fields take, which is nothing at all for most entries.
    #[inline]
    const fn trailer(&self) -> usize {
        (if self.flags & HAS_EXPIRY != 0 { 4 } else { 0 })
            + (if self.flags & HAS_USER_FLAGS != 0 {
                4
            } else {
                0
            })
    }

    /// The key, out of the entry's own bytes.
    #[inline]
    pub(crate) fn key<'a>(&self, bytes: &'a [u8]) -> &'a [u8] {
        &bytes[self.header..self.header + self.klen]
    }

    /// The value, out of the entry's own bytes.
    #[inline]
    pub(crate) fn value<'a>(&self, bytes: &'a [u8]) -> &'a [u8] {
        let at = self.header + self.klen;
        &bytes[at..at + self.vlen]
    }

    /// When the entry expires, if it does.
    #[inline]
    pub(crate) fn expiry(&self, bytes: &[u8]) -> Option<u32> {
        if self.flags & HAS_EXPIRY == 0 {
            return None;
        }
        Some(word(bytes, self.header + self.klen + self.vlen))
    }

    /// The memcache-style user flags word, if it has one.
    #[inline]
    pub(crate) fn user_flags(&self, bytes: &[u8]) -> Option<u32> {
        if self.flags & HAS_USER_FLAGS == 0 {
            return None;
        }
        let after = if self.flags & HAS_EXPIRY != 0 { 4 } else { 0 };
        Some(word(bytes, self.header + self.klen + self.vlen + after))
    }

    /// The whole entry, for a caller that wants all of it.
    #[inline]
    pub(crate) fn view<'a>(&self, bytes: &'a [u8]) -> View<'a> {
        View {
            key: self.key(bytes),
            value: self.value(bytes),
            expiry: self.expiry(bytes),
            user_flags: self.user_flags(bytes),
        }
    }
}

/// Decode the header of the entry at the front of `bytes`.
///
/// `bytes` has to reach the end of the header and no further, which is what [`rugo_arena::Arena::peek`] is for: the length this returns is the length the arena needs to be told before it will hand over the rest.
#[inline]
pub(crate) fn head(bytes: &[u8]) -> Head {
    let flags = bytes[0];
    let (klen, one) = get_varint(&bytes[1..]);
    let (vlen, two) = get_varint(&bytes[1 + one..]);
    Head {
        flags,
        header: 1 + one + two,
        klen,
        vlen,
    }
}

/// A little-endian word at `at`.
#[inline]
fn word(bytes: &[u8], at: usize) -> u32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[at..at + 4]);
    u32::from_le_bytes(buf)
}

/// A borrowed view of an entry's bytes.
///
/// Borrowed rather than owned because the commands that want more than the value, `GETEX` and memcache's `get` with flags, want to write these bytes straight to a socket. Copying them into a struct first would be a copy the wire is about to make anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct View<'a> {
    /// The key.
    pub key: &'a [u8],
    /// The value.
    pub value: &'a [u8],
    /// When it expires, in the clock's seconds, if it does.
    pub expiry: Option<u32>,
    /// The memcache-style user flags word, if it has one.
    pub user_flags: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_varint_round_trips_at_every_width() {
        for value in [
            0usize,
            1,
            0x7f,
            0x80,
            0x3fff,
            0x4000,
            0x1f_ffff,
            0x20_0000,
            0xfff_ffff,
            0x1000_0000,
        ] {
            let mut buf = [0u8; 8];
            let wrote = put_varint(&mut buf, value);
            assert_eq!(wrote, varint_len(value), "{value} disagreed on width");
            let (read, used) = get_varint(&buf);
            assert_eq!((read, used), (value, wrote), "{value} did not round trip");
        }
    }

    #[test]
    fn an_entry_round_trips() {
        for (key, value, expiry, user_flags) in [
            (&b"k"[..], &b"v"[..], None, None),
            (&b"user:1"[..], &b"hello"[..], Some(1234), None),
            (&b"user:2"[..], &b""[..], None, Some(0xdead_beef)),
            (&b""[..], &b"only a value"[..], Some(7), Some(9)),
        ] {
            let size = size_of_entry(
                key.len(),
                value.len(),
                expiry.is_some(),
                user_flags.is_some(),
            );
            let mut buf = vec![0u8; size];
            write(&mut buf, key, value, expiry, user_flags);

            let head = head(&buf);
            assert_eq!(
                head.size(),
                size,
                "the header disagreed about its own length"
            );
            let view = head.view(&buf);
            assert_eq!(view.key, key);
            assert_eq!(view.value, value);
            assert_eq!(view.expiry, expiry);
            assert_eq!(view.user_flags, user_flags);
        }
    }

    #[test]
    fn a_long_value_round_trips() {
        let key = b"big";
        let value: Vec<u8> = (0..100_000u32)
            .map(|i| u8::try_from(i % 251).unwrap())
            .collect();
        let size = size_of_entry(key.len(), value.len(), false, false);
        let mut buf = vec![0u8; size];
        write(&mut buf, key, &value, None, None);
        let view = head(&buf).view(&buf);
        assert_eq!(view.key, key);
        assert_eq!(view.value, &value[..]);
    }

    #[test]
    fn the_header_of_an_ordinary_entry_is_three_bytes() {
        // The claim the layout is for. A twenty byte key and a hundred byte value with no expiry costs three bytes to describe, and there is no allocator header underneath it.
        assert_eq!(size_of_entry(20, 100, false, false), 20 + 100 + 3);
    }

    #[test]
    fn an_absent_expiry_costs_nothing() {
        let with = size_of_entry(20, 100, true, false);
        let without = size_of_entry(20, 100, false, false);
        assert_eq!(
            with - without,
            4,
            "an expiry should cost four bytes and only when present"
        );
    }
}
