//! The Redis wire protocol, parsed in place and written into a caller's buffer.
//!
//! # Nothing here allocates on the hot path
//!
//! A command comes out as a list of spans into the read buffer, not as a list of owned strings, and the list itself lives in a [`Command`] the connection keeps and reuses. A `SET` of a hundred-byte value therefore costs no allocation at all between the socket and the map, which is the only way the parse stops being the expensive half of a pipelined batch.
//!
//! Spans are `u32` pairs rather than slices because a `&[u8]` borrowed from the read buffer would pin the buffer for as long as the command lives, and the connection wants to keep the command across a read that refills it.
//!
//! # What it accepts
//!
//! An array of bulk strings, which is what every client sends, and an inline command, which is what a person with a telnet connection and a health check sends. `cache-bench` decides a server is up by writing `PING\r\n` at it and waiting for `+PONG`, with no array around it, so the inline form is a requirement here rather than a courtesy.
//!
//! # RESP2 and RESP3
//!
//! The request side is identical in both, so parsing does not care which the client picked. Only three replies differ, and [`Encoder`] carries the one bit that says which: a null, a map, and the double that no command here returns. `HELLO 3` moves that bit.

#![forbid(unsafe_code)]

/// How many arguments a command may carry.
///
/// `MSET` with a thousand pairs is a real thing a benchmark does, so the limit is high enough not to be met in practice and low enough that a client cannot make the server allocate without bound by announcing a large array and then sending nothing.
pub const MAX_ARGS: usize = 1024 * 1024;

/// The longest bulk string that will be accepted, which is Redis's own limit.
pub const MAX_BULK: usize = 512 * 1024 * 1024;

/// The longest line an inline command may be.
///
/// Inline commands are for a health check and a person at a terminal, and neither sends a long one. A client that sends megabytes without a newline is either broken or hostile, and either way this is where it stops.
pub const MAX_INLINE: usize = 64 * 1024;

/// One argument's place in the read buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    /// Where it starts.
    at: u32,
    /// How long it is.
    len: u32,
}

/// A parsed command: the spans of its arguments, in the buffer they were parsed from.
///
/// Reused across commands. [`Command::clear`] keeps the allocation, so a connection that has served a million commands has allocated one argument list.
#[derive(Debug, Default, Clone)]
pub struct Command {
    /// Where each argument is.
    args: Vec<Span>,
}

impl Command {
    /// An empty command with no arguments and no allocation yet.
    #[must_use]
    pub const fn new() -> Self {
        Self { args: Vec::new() }
    }

    /// How many arguments it has, including the command name.
    #[must_use]
    pub fn len(&self) -> usize {
        self.args.len()
    }

    /// Whether it has no arguments at all, which a well-formed request never is.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }

    /// Argument `n` out of the buffer it was parsed from.
    ///
    /// The buffer has to be the one [`parse`] was given, and unchanged since. Passing a different one gives the wrong bytes or none, which is why this takes the buffer rather than remembering it.
    #[must_use]
    pub fn arg<'a>(&self, n: usize, buf: &'a [u8]) -> Option<&'a [u8]> {
        let span = self.args.get(n)?;
        let at = span.at as usize;
        buf.get(at..at + span.len as usize)
    }

    /// Forget the arguments and keep the allocation.
    pub fn clear(&mut self) {
        self.args.clear();
    }

    /// Record an argument, which must lie inside the buffer being parsed.
    fn push(&mut self, at: usize, len: usize) -> Result<(), Bad> {
        if self.args.len() >= MAX_ARGS {
            return Err(Bad::TooManyArguments);
        }
        // A buffer this large cannot exist: the read buffer is bounded by the connection and by MAX_BULK, both far under four gigabytes.
        let (Ok(at), Ok(len)) = (u32::try_from(at), u32::try_from(len)) else {
            return Err(Bad::TooLong);
        };
        self.args.push(Span { at, len });
        Ok(())
    }
}

/// Why a request was rejected.
///
/// Each of these is a protocol error, which means the connection is closed after the message is sent: once the framing is wrong there is no way to find where the next command starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bad {
    /// A length or count was not a number.
    NotANumber,
    /// A bulk string was longer than [`MAX_BULK`].
    TooLong,
    /// An array had more elements than [`MAX_ARGS`].
    TooManyArguments,
    /// An array element was not a bulk string.
    ExpectedBulk,
    /// An inline command ran past [`MAX_INLINE`] without a newline.
    NoNewline,
}

impl Bad {
    /// The message that goes on the wire, without its `-ERR ` prefix or its terminator.
    #[must_use]
    pub const fn message(self) -> &'static str {
        match self {
            Self::NotANumber => "Protocol error: invalid multibulk length",
            Self::TooLong => "Protocol error: invalid bulk length",
            Self::TooManyArguments => "Protocol error: too many arguments",
            Self::ExpectedBulk => "Protocol error: expected '$', got something else",
            Self::NoNewline => "Protocol error: too big inline request",
        }
    }
}

impl core::fmt::Display for Bad {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.message())
    }
}

impl core::error::Error for Bad {}

/// What one parse attempt found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parsed {
    /// A whole command, occupying this many bytes from the front of the buffer.
    Done(usize),
    /// Not all of a command is here yet. Read more and try again from the same offset.
    ///
    /// Deliberately carries nothing about how much more is needed. A parser that reported that would be believed, and a client that announced a hundred megabytes and sent none would have the server sit on a buffer it grew for the purpose.
    More,
}

/// Parse one command from the front of `buf` into `into`.
///
/// `into` is cleared first, so a caller may reuse it without thinking about it.
///
/// # Errors
///
/// [`Bad`] when the framing is wrong, which is not recoverable on a stream.
pub fn parse(buf: &[u8], into: &mut Command) -> Result<Parsed, Bad> {
    into.clear();
    let Some(&first) = buf.first() else {
        return Ok(Parsed::More);
    };
    if first == b'*' {
        parse_array(buf, into)
    } else {
        parse_inline(buf, into)
    }
}

/// Parse the array-of-bulk-strings form, which is what a client library sends.
fn parse_array(buf: &[u8], into: &mut Command) -> Result<Parsed, Bad> {
    let Some((header, mut at)) = line(buf, 0) else {
        return Ok(Parsed::More);
    };
    // The caller only looked at the first byte to choose this function, so the `*` is still on the front of the line.
    let count = number(&header[1..])?;

    // A negative or zero count is a null or empty array. Redis treats both as a request to do nothing, and a connection that sends one is not in error, so this reports a complete command with no arguments and the caller ignores it.
    let Ok(count) = usize::try_from(count) else {
        return Ok(Parsed::Done(at));
    };
    if count > MAX_ARGS {
        return Err(Bad::TooManyArguments);
    }

    for _ in 0..count {
        let Some((header, next)) = line(buf, at) else {
            return Ok(Parsed::More);
        };
        let Some((&b'$', digits)) = header.split_first() else {
            return Err(Bad::ExpectedBulk);
        };
        let len = number(digits)?;
        let Ok(len) = usize::try_from(len) else {
            return Err(Bad::TooLong);
        };
        if len > MAX_BULK {
            return Err(Bad::TooLong);
        }
        // The body and its own terminator, which is present or the bulk is not here yet.
        if buf.len() < next + len + 2 {
            return Ok(Parsed::More);
        }
        into.push(next, len)?;
        at = next + len + 2;
    }

    Ok(Parsed::Done(at))
}

/// Parse the inline form: a line of arguments separated by spaces.
///
/// This is what `PING\r\n` typed at a socket is, and what every readiness check in `cache-bench` sends.
fn parse_inline(buf: &[u8], into: &mut Command) -> Result<Parsed, Bad> {
    let Some((line, at)) = line(buf, 0) else {
        if buf.len() > MAX_INLINE {
            return Err(Bad::NoNewline);
        }
        return Ok(Parsed::More);
    };

    let mut cursor = 0;
    while cursor < line.len() {
        if line[cursor].is_ascii_whitespace() {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while cursor < line.len() && !line[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        into.push(start, cursor - start)?;
    }

    Ok(Parsed::Done(at))
}

/// The line starting at `at`, and where the line after it starts.
///
/// A line ends at the first `\n`, and a `\r` before it belongs to the terminator rather than to the line. Redis is lenient about the `\r` and so is this, because the inline form is typed by hand often enough to matter.
fn line(buf: &[u8], at: usize) -> Option<(&[u8], usize)> {
    let from = &buf[at.min(buf.len())..];
    let end = newline(from)?;
    let text = if end > 0 && from[end - 1] == b'\r' {
        &from[..end - 1]
    } else {
        &from[..end]
    };
    Some((text, at + end + 1))
}

/// Where the first newline in `from` is.
///
/// A byte at a time, which is what a header of a dozen bytes wants: the vectorised scan that pays for itself over a long buffer costs more than it saves over `$100\r\n`. The value-sized scan is [`scan`], and the batch parser will use it once there is a batch parser.
#[inline]
fn newline(from: &[u8]) -> Option<usize> {
    from.iter().position(|&byte| byte == b'\n')
}

/// Where the first newline in `from` is, read a machine word at a time.
///
/// Kept beside [`newline`] and used by nothing yet on purpose: the point of a wide scan is a long buffer, and every line this parser reads is a short one. It is here because the batch parser in M5 wants it and because a scan with no test beside it is a scan that is wrong.
#[must_use]
pub fn scan(from: &[u8]) -> Option<usize> {
    /// One in every byte of a word.
    const ONES: usize = usize::MAX / 0xff;
    /// The top bit of every byte of a word.
    const HIGHS: usize = ONES << 7;
    const STEP: usize = size_of::<usize>();

    let mut at = 0;
    while at + STEP <= from.len() {
        let mut word = 0usize;
        for (n, &byte) in from[at..at + STEP].iter().enumerate() {
            word |= (byte as usize) << (n * 8);
        }
        // Subtracting one from every byte borrows out of the byte above unless that byte is zero, so a byte that was zero is the only one left with its top bit set. Exclusive-or with the newline first, and the zero bytes are the newlines.
        let hit = word ^ (ONES * b'\n' as usize);
        let found = hit.wrapping_sub(ONES) & !hit & HIGHS;
        if found != 0 {
            return Some(at + found.trailing_zeros() as usize / 8);
        }
        at += STEP;
    }
    from[at..]
        .iter()
        .position(|&byte| byte == b'\n')
        .map(|n| at + n)
}

/// Read a signed decimal, which is what every length and count on the wire is.
fn number(text: &[u8]) -> Result<i64, Bad> {
    let (negative, digits) = match text.split_first() {
        Some((&b'-', rest)) => (true, rest),
        Some((&b'+', rest)) => (false, rest),
        _ => (false, text),
    };
    if digits.is_empty() {
        return Err(Bad::NotANumber);
    }

    let mut value: i64 = 0;
    for &byte in digits {
        if !byte.is_ascii_digit() {
            return Err(Bad::NotANumber);
        }
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(i64::from(byte - b'0')))
            .ok_or(Bad::NotANumber)?;
    }
    Ok(if negative { -value } else { value })
}

/// Read a signed decimal out of a client's argument.
///
/// Separate from the internal [`number`] so that a bad `INCR` argument is a command error the connection survives, rather than a framing error that closes it.
///
/// # Errors
///
/// [`Bad::NotANumber`] when the argument is not a decimal integer that fits in an `i64`.
pub fn integer(text: &[u8]) -> Result<i64, Bad> {
    number(text)
}

/// Which dialect a connection is speaking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Dialect {
    /// What a connection speaks until it says `HELLO 3`.
    #[default]
    Resp2,
    /// What it speaks after.
    Resp3,
}

/// Writes replies into a caller's buffer.
///
/// A borrowed buffer rather than an owned one, because the buffer belongs to the connection and outlives any one reply, and because a batch of replies is one buffer and one write.
#[derive(Debug)]
pub struct Encoder<'a> {
    /// Where the bytes go.
    out: &'a mut Vec<u8>,
    /// Which dialect the client asked for.
    dialect: Dialect,
}

impl<'a> Encoder<'a> {
    /// An encoder writing into `out`.
    pub fn new(out: &'a mut Vec<u8>, dialect: Dialect) -> Self {
        Self { out, dialect }
    }

    /// The bytes written so far.
    #[must_use]
    pub fn buffer(&self) -> &[u8] {
        self.out
    }

    /// `+OK`, and the other one-line answers.
    pub fn simple(&mut self, text: &[u8]) {
        self.out.push(b'+');
        self.out.extend_from_slice(text);
        self.crlf();
    }

    /// An error, which the client will raise rather than return.
    ///
    /// The text carries its own code, because Redis's codes are not a closed set and a caller that wants `WRONGTYPE` rather than `ERR` should be able to say so without this knowing what either means.
    pub fn error(&mut self, text: &str) {
        self.out.push(b'-');
        self.out.extend_from_slice(text.as_bytes());
        self.crlf();
    }

    /// An integer.
    pub fn integer(&mut self, value: i64) {
        self.out.push(b':');
        self.number(value);
        self.crlf();
    }

    /// A bulk string, which is what a value is.
    pub fn bulk(&mut self, bytes: &[u8]) {
        self.out.push(b'$');
        self.count(bytes.len());
        self.crlf();
        self.out.extend_from_slice(bytes);
        self.crlf();
    }

    /// The absence of a value, which is the one reply the two dialects spell differently.
    pub fn null(&mut self) {
        match self.dialect {
            Dialect::Resp2 => self.out.extend_from_slice(b"$-1\r\n"),
            Dialect::Resp3 => self.out.extend_from_slice(b"_\r\n"),
        }
    }

    /// The header of an array of `len` elements, whose elements the caller writes next.
    pub fn array(&mut self, len: usize) {
        self.out.push(b'*');
        self.count(len);
        self.crlf();
    }

    /// The header of a map of `len` pairs, which RESP2 has to spell as a flat array of twice as many.
    pub fn map(&mut self, len: usize) {
        match self.dialect {
            Dialect::Resp2 => self.array(len * 2),
            Dialect::Resp3 => {
                self.out.push(b'%');
                self.count(len);
                self.crlf();
            }
        }
    }

    /// A verbatim run of bytes, for a reply that is already encoded.
    pub fn raw(&mut self, bytes: &[u8]) {
        self.out.extend_from_slice(bytes);
    }

    /// Write a length, which is a count of bytes or elements and so is never negative.
    ///
    /// Saturating rather than wrapping, because a length that did not fit would otherwise go on the wire as a negative number, and a negative bulk length is how RESP2 spells a null. A reply that large cannot be built on a machine that exists, but the failure mode if one could is a client silently reading a null instead of a value.
    fn count(&mut self, len: usize) {
        self.number(i64::try_from(len).unwrap_or(i64::MAX));
    }

    /// Write a decimal without allocating one.
    fn number(&mut self, value: i64) {
        let mut digits = [0u8; 20];
        let mut at = digits.len();
        let negative = value < 0;
        // Taken as unsigned so that the most negative integer, which has no positive counterpart, does not overflow on its way to being printed.
        let mut left = value.unsigned_abs();
        loop {
            at -= 1;
            digits[at] = b'0' + u8::try_from(left % 10).unwrap_or(0);
            left /= 10;
            if left == 0 {
                break;
            }
        }
        if negative {
            self.out.push(b'-');
        }
        self.out.extend_from_slice(&digits[at..]);
    }

    fn crlf(&mut self) {
        self.out.extend_from_slice(b"\r\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args<'a>(command: &Command, buf: &'a [u8]) -> Vec<&'a [u8]> {
        (0..command.len())
            .filter_map(|n| command.arg(n, buf))
            .collect()
    }

    #[test]
    fn an_array_of_bulk_strings_parses() {
        let buf = b"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n";
        let mut command = Command::new();
        assert_eq!(parse(buf, &mut command), Ok(Parsed::Done(buf.len())));
        assert_eq!(args(&command, buf), vec![&b"SET"[..], b"key", b"value"]);
    }

    #[test]
    fn an_inline_ping_parses() {
        // What cache-bench writes at a socket to decide the server is up, and the only reason the inline form exists here.
        let buf = b"PING\r\n";
        let mut command = Command::new();
        assert_eq!(parse(buf, &mut command), Ok(Parsed::Done(6)));
        assert_eq!(args(&command, buf), vec![&b"PING"[..]]);
    }

    #[test]
    fn an_inline_command_may_have_arguments_and_a_bare_newline() {
        let buf = b"get  foo\n";
        let mut command = Command::new();
        assert_eq!(parse(buf, &mut command), Ok(Parsed::Done(9)));
        assert_eq!(args(&command, buf), vec![&b"get"[..], b"foo"]);
    }

    #[test]
    fn a_command_split_across_reads_asks_for_more() {
        let whole = b"*2\r\n$3\r\nGET\r\n$3\r\nkey\r\n";
        let mut command = Command::new();
        for cut in 1..whole.len() {
            assert_eq!(
                parse(&whole[..cut], &mut command),
                Ok(Parsed::More),
                "{cut} bytes of a {} byte command should not have parsed",
                whole.len()
            );
        }
        assert_eq!(parse(whole, &mut command), Ok(Parsed::Done(whole.len())));
    }

    #[test]
    fn a_pipeline_parses_one_command_at_a_time() {
        // The shape the whole design is about: several commands in one read, each reported with its own length so the caller can step over it.
        let buf = b"*1\r\n$4\r\nPING\r\n*1\r\n$4\r\nPING\r\nPING\r\n";
        let mut command = Command::new();
        let mut at = 0;
        let mut seen = 0;
        while at < buf.len() {
            let Ok(Parsed::Done(used)) = parse(&buf[at..], &mut command) else {
                panic!("command {seen} did not parse");
            };
            assert_eq!(args(&command, &buf[at..]), vec![&b"PING"[..]]);
            at += used;
            seen += 1;
        }
        assert_eq!(seen, 3);
    }

    #[test]
    fn an_empty_array_is_a_command_that_does_nothing() {
        let buf = b"*0\r\n";
        let mut command = Command::new();
        assert_eq!(parse(buf, &mut command), Ok(Parsed::Done(4)));
        assert!(command.is_empty());
    }

    #[test]
    fn a_null_array_is_not_an_error() {
        let buf = b"*-1\r\n";
        let mut command = Command::new();
        assert_eq!(parse(buf, &mut command), Ok(Parsed::Done(5)));
        assert!(command.is_empty());
    }

    #[test]
    fn an_empty_bulk_string_is_a_value() {
        let buf = b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$0\r\n\r\n";
        let mut command = Command::new();
        assert_eq!(parse(buf, &mut command), Ok(Parsed::Done(buf.len())));
        assert_eq!(command.arg(2, buf), Some(&b""[..]));
    }

    #[test]
    fn a_value_carrying_a_crlf_is_read_by_its_length_and_not_by_its_bytes() {
        // The reason a bulk string has a length in front of it, and the test that this parser believes the length rather than scanning for a terminator.
        let buf = b"*2\r\n$3\r\nSET\r\n$6\r\na\r\nb\r\n\r\n";
        let mut command = Command::new();
        assert_eq!(parse(buf, &mut command), Ok(Parsed::Done(buf.len())));
        assert_eq!(command.arg(1, buf), Some(&b"a\r\nb\r\n"[..]));
    }

    #[test]
    fn a_bad_first_byte_is_an_inline_command_and_not_an_error() {
        // Redis takes anything that is not an array as an inline command, which means a client speaking nonsense gets an unknown-command error rather than a framing one, and stays connected.
        let buf = b"%hello\r\n";
        let mut command = Command::new();
        assert_eq!(parse(buf, &mut command), Ok(Parsed::Done(8)));
        assert_eq!(args(&command, buf), vec![&b"%hello"[..]]);
    }

    #[test]
    fn an_array_element_that_is_not_a_bulk_string_is_refused() {
        let buf = b"*1\r\n:1\r\n";
        let mut command = Command::new();
        assert_eq!(parse(buf, &mut command), Err(Bad::ExpectedBulk));
    }

    #[test]
    fn a_length_that_is_not_a_number_is_refused() {
        let mut command = Command::new();
        assert_eq!(parse(b"*x\r\n", &mut command), Err(Bad::NotANumber));
        assert_eq!(parse(b"*1\r\n$x\r\n", &mut command), Err(Bad::NotANumber));
    }

    #[test]
    fn a_bulk_string_longer_than_the_limit_is_refused() {
        let mut command = Command::new();
        let buf = format!("*1\r\n${}\r\n", MAX_BULK + 1).into_bytes();
        assert_eq!(parse(&buf, &mut command), Err(Bad::TooLong));
    }

    #[test]
    fn an_endless_inline_command_is_refused_rather_than_buffered() {
        let mut command = Command::new();
        let buf = vec![b'x'; MAX_INLINE + 1];
        assert_eq!(parse(&buf, &mut command), Err(Bad::NoNewline));
    }

    #[test]
    fn the_word_scan_agrees_with_the_byte_scan() {
        // Two implementations of one question, checked against each other at every alignment and every length, because a SWAR scan that is subtly wrong is a scan that reads the right answer nearly always.
        for len in 0..40usize {
            let mut buf = vec![b'a'; len];
            assert_eq!(scan(&buf), newline(&buf), "no newline at {len}");
            for at in 0..len {
                buf[at] = b'\n';
                assert_eq!(scan(&buf), newline(&buf), "newline at {at} of {len}");
                buf[at] = b'a';
            }
        }
    }

    #[test]
    fn the_replies_look_like_redis() {
        let mut out = Vec::new();
        let mut encoder = Encoder::new(&mut out, Dialect::Resp2);
        encoder.simple(b"OK");
        encoder.error("ERR unknown command 'x'");
        encoder.integer(-42);
        encoder.bulk(b"hello");
        encoder.null();
        encoder.array(2);
        assert_eq!(
            out,
            b"+OK\r\n-ERR unknown command 'x'\r\n:-42\r\n$5\r\nhello\r\n$-1\r\n*2\r\n"
        );
    }

    #[test]
    fn resp3_spells_a_null_differently_and_nothing_else_here() {
        let mut out = Vec::new();
        let mut encoder = Encoder::new(&mut out, Dialect::Resp3);
        encoder.null();
        encoder.bulk(b"hi");
        assert_eq!(out, b"_\r\n$2\r\nhi\r\n");
    }

    #[test]
    fn a_map_is_a_flat_array_in_resp2_and_a_map_in_resp3() {
        let mut two = Vec::new();
        Encoder::new(&mut two, Dialect::Resp2).map(3);
        assert_eq!(two, b"*6\r\n");

        let mut three = Vec::new();
        Encoder::new(&mut three, Dialect::Resp3).map(3);
        assert_eq!(three, b"%3\r\n");
    }

    #[test]
    fn every_integer_prints_including_the_one_with_no_positive_counterpart() {
        for value in [0i64, 1, -1, 9, 10, -10, 1_234_567_890, i64::MAX, i64::MIN] {
            let mut out = Vec::new();
            Encoder::new(&mut out, Dialect::Resp2).integer(value);
            assert_eq!(
                out,
                format!(":{value}\r\n").into_bytes(),
                "{value} printed wrong"
            );
        }
    }

    #[test]
    fn an_argument_that_is_not_a_number_is_a_command_error_and_not_a_framing_one() {
        assert_eq!(integer(b"12"), Ok(12));
        assert_eq!(integer(b"-12"), Ok(-12));
        assert_eq!(integer(b""), Err(Bad::NotANumber));
        assert_eq!(integer(b"1.5"), Err(Bad::NotANumber));
        assert_eq!(integer(b"99999999999999999999"), Err(Bad::NotANumber));
    }
}
