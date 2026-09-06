//! One client, its two buffers, and the cycle between them.
//!
//! A connection reads what is there, executes every whole command in it, and writes what that produced. Nothing here waits: a read that would block ends the turn, and a write that would block registers interest and ends the turn. That is the whole of it, and it is why one thread can hold thousands of these.

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;

use rugo_net::Interest;
use rugo_resp::{Command, Dialect, Encoder, Parsed};

use crate::dispatch::{self, Env, Reply};

/// How much room a read is given, and how large the read buffer starts.
///
/// Sixteen kibibytes takes a pipeline of a hundred ordinary commands in one syscall, which is deeper than the harness ever goes.
const CHUNK: usize = 16 * 1024;

/// How much unparsed room has to be free before a read, or the buffer is compacted first.
const SPARE: usize = 4 * 1024;

/// How many commands ahead the connection asks the map for cache lines.
///
/// This is how many misses the machine is asked to carry at once, so the right number is a property of the machine rather than of the map. A probe with no server around it swept four to thirty two on `epyc8` and every reading landed between 1444 and 1766 cycles a lookup with no ordering to it, so four is already enough there and more is not worse. Eight is in the middle of that flat stretch and is also the depth at which a pipeline of twenty five, which is what the sweep sends, divides into whole batches with a remainder that is still worth asking for.
const AHEAD: usize = 8;

/// The largest a single request may grow the read buffer.
///
/// A client may announce a bulk string of half a gigabyte, and the parser will hold what has arrived until the rest does. Without a bound, a handful of connections that announce a large value and then go quiet is the whole machine's memory. Sixteen mebibytes is far above any cache value anybody stores and far below anything that matters.
const MAX_REQUEST: usize = 16 * 1024 * 1024;

/// How much reply may pile up before the connection stops executing and drains.
///
/// The backpressure that keeps a deep pipeline of `MGET`s from turning into an unbounded write buffer. The commands already parsed are answered; the rest wait in the read buffer, which is bounded, until the socket takes what is here.
const MAX_REPLY: usize = 1024 * 1024;

/// A socket, whichever kind it is.
///
/// An enum rather than a boxed trait object because there are exactly two and there will not be a third, and because a virtual call per read on the hottest path in the program is a virtual call nobody has to pay for.
#[derive(Debug)]
pub(crate) enum Stream {
    /// A TCP connection.
    Tcp(TcpStream),
    /// A unix socket connection, which is what a benchmark on one machine should be using.
    Unix(UnixStream),
}

impl Stream {
    /// Put the socket in the only mode this server can use.
    ///
    /// Nagle off as well as non-blocking, because a reply held back waiting for a second one to coalesce with is a reply that arrives a millisecond late, and a latency percentile that reads as this server's fault.
    pub(crate) fn prepare(&self) -> io::Result<()> {
        match self {
            Self::Tcp(stream) => {
                stream.set_nonblocking(true)?;
                stream.set_nodelay(true)
            }
            Self::Unix(stream) => stream.set_nonblocking(true),
        }
    }
}

impl AsRawFd for Stream {
    fn as_raw_fd(&self) -> RawFd {
        match self {
            Self::Tcp(stream) => stream.as_raw_fd(),
            Self::Unix(stream) => stream.as_raw_fd(),
        }
    }
}

impl Read for Stream {
    fn read(&mut self, into: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Tcp(stream) => stream.read(into),
            Self::Unix(stream) => stream.read(into),
        }
    }
}

impl Write for Stream {
    fn write(&mut self, from: &[u8]) -> io::Result<usize> {
        match self {
            Self::Tcp(stream) => stream.write(from),
            Self::Unix(stream) => stream.write(from),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Tcp(stream) => stream.flush(),
            Self::Unix(stream) => stream.flush(),
        }
    }
}

/// What a turn on a connection decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Turn {
    /// Keep it, wanting to be told when it is readable.
    Read,
    /// Keep it, wanting to be told when it is writable as well, because a reply did not fit.
    Write,
    /// Drop it.
    Close,
}

/// A client connection.
#[derive(Debug)]
pub(crate) struct Conn {
    /// The socket.
    stream: Stream,
    /// The read buffer, whose length is its size rather than what is in it.
    ///
    /// Sized once and re-used, so that a connection serving a million commands does not zero sixteen kibibytes a million times to make room it already had.
    buf: Vec<u8>,
    /// How much of `buf` holds bytes that arrived.
    filled: usize,
    /// How much of `buf` has been parsed and executed.
    at: usize,
    /// Replies waiting to go out.
    out: Vec<u8>,
    /// How much of `out` the socket has taken.
    sent: usize,
    /// Which dialect this client asked for.
    dialect: Dialect,
    /// The argument list, re-used across every command this connection ever serves.
    command: Command,
    /// Set when the last reply has been written and the connection should go.
    closing: bool,
    /// What the poller was last told to watch this socket for.
    ///
    /// Held so that a turn wanting what the poller is already doing can skip the `epoll_ctl`, which in steady state is every turn: a connection that is reading commands and answering them wants to read again, which is what it was already registered for. At a pipeline depth of twenty-five that syscall was a third of the three this connection made per batch and it bought nothing.
    ///
    /// This is safe to track because both pollers are level triggered and a registration persists until it is changed or the descriptor is closed, so what was asked for last is what is still in force. It is one field written in one place, [`Conn::arm`], and the alternative to keeping it is paying a syscall a batch to avoid a bool.
    armed: Interest,
}

impl Conn {
    /// A connection over `stream`, which must already be non-blocking.
    #[must_use]
    pub(crate) fn new(stream: Stream) -> Self {
        Self {
            stream,
            buf: vec![0; CHUNK],
            filled: 0,
            at: 0,
            out: Vec::with_capacity(CHUNK),
            sent: 0,
            dialect: Dialect::default(),
            command: Command::new(),
            closing: false,
            // What `take` registers a freshly accepted connection for. A connection whose first turn wants exactly this makes no `epoll_ctl` at all.
            armed: Interest::READ,
        }
    }

    /// The socket's descriptor, for the poller.
    #[must_use]
    pub(crate) fn fd(&self) -> RawFd {
        self.stream.as_raw_fd()
    }

    /// What the poller is currently watching this socket for.
    ///
    /// Read by whoever accepts the connection to decide what to register it as, so the field and the registration cannot drift apart: there is one answer and both sites use it.
    #[must_use]
    pub(crate) fn armed(&self) -> Interest {
        self.armed
    }

    /// Record that the poller has been told to watch this socket for `want`.
    ///
    /// Called only after the `modify` it describes has succeeded, so a failed registration leaves the field saying what is actually in force rather than what was wanted.
    pub(crate) fn arm(&mut self, want: Interest) {
        self.armed = want;
    }

    /// Read what is there, answer it, and write what fits.
    ///
    /// `readable` and `writable` are what the poller reported. A turn woken only to write does not read, because a level-triggered poller that reported nothing readable has nothing readable to report.
    pub(crate) fn turn(&mut self, env: &Env<'_>, readable: bool, writable: bool) -> Turn {
        if writable && self.flush().is_err() {
            return Turn::Close;
        }
        if readable {
            match self.fill() {
                Ok(true) => {}
                // A peer that hung up, or a socket that failed. Either way there is nobody to tell.
                Ok(false) | Err(_) => return Turn::Close,
            }
            self.execute(env);
            if self.flush().is_err() {
                return Turn::Close;
            }
        }

        if self.sent < self.out.len() {
            // Something is still owed. Even a closing connection gets its last reply written before it goes, because a client that asked `QUIT` is entitled to the `+OK`.
            return Turn::Write;
        }
        if self.closing {
            Turn::Close
        } else {
            Turn::Read
        }
    }

    /// Read once, reporting whether the peer is still there.
    ///
    /// One read a readiness event rather than a loop to exhaustion. Level-triggered polling will say so again if there is more, and a connection that read until it blocked would be one connection holding a thread while the others waited.
    fn fill(&mut self) -> io::Result<bool> {
        self.make_room()?;
        loop {
            return match self.stream.read(&mut self.buf[self.filled..]) {
                Ok(0) => Ok(false),
                Ok(read) => {
                    self.filled += read;
                    Ok(true)
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                // Woken with nothing to read, which a shared listener and a level-triggered poller both make ordinary.
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(true),
                Err(error) => Err(error),
            };
        }
    }

    /// Make sure there is somewhere to read into.
    ///
    /// Compacting first and growing only if that was not enough, so a connection sending ordinary commands keeps one buffer forever and a connection sending one enormous value grows once.
    fn make_room(&mut self) -> io::Result<()> {
        if self.buf.len() - self.filled >= SPARE {
            return Ok(());
        }
        if self.at > 0 {
            self.buf.copy_within(self.at..self.filled, 0);
            self.filled -= self.at;
            self.at = 0;
        }
        if self.buf.len() - self.filled < SPARE {
            let wanted = self.buf.len() * 2;
            if wanted > MAX_REQUEST {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "request larger than this server will buffer",
                ));
            }
            self.buf.resize(wanted, 0);
        }
        Ok(())
    }

    /// Ask the map for the lines the next few commands in the buffer will read, and report how far ahead the asking got.
    ///
    /// A lookup is three loads that depend on each other, so past the last level of cache it is three misses that cannot overlap and that is most of what a get costs. They can overlap across commands though, and a pipelined client has already sent the next few, so the addresses are sitting in the read buffer waiting to be read off it.
    ///
    /// Two passes rather than one because the second address is inside the line the first pass asks for. Asking for an entry before its index has arrived would be asking at whatever the stale slot used to say, which is a wasted line rather than a wrong answer, but it is still wasted.
    ///
    /// Nothing here reads anything and nothing here can be wrong. A hint at a line that turns out not to be wanted costs one line of cache, and a hint at an address that has been rebuilt away is dropped by the hardware rather than faulting.
    ///
    /// The peek is deliberately not a parse. An earlier version of this held [`AHEAD`] parsed commands so the execute pass could reuse them, and on `epyc8` it cost a fifth more cycles a get than not looking ahead at all, with the extra misses landing in the argument lists rather than in the map. Reading the framing twice and keeping one argument list is cheaper than reading it once and keeping eight.
    fn ask_ahead(env: &Env<'_>, buf: &[u8], from: usize, filled: usize) -> usize {
        let mut keys = [(0_usize, 0_usize); AHEAD];
        let mut found = 0;
        let mut scan = from;
        for _ in 0..AHEAD {
            let Some((used, key)) = rugo_resp::peek(&buf[scan..filled]) else {
                break;
            };
            if let Some((at, len)) = key {
                keys[found] = (scan + at, len);
                found += 1;
            }
            scan += used;
        }

        // One key is the command about to run, and asking for a line a few instructions before reading it buys nothing. The overlap is the whole point, so there has to be something to overlap.
        if found > 1 {
            for &(at, len) in &keys[..found] {
                if let Some(key) = buf.get(at..at + len) {
                    env.map.warm(key);
                }
            }
            for &(at, len) in &keys[..found] {
                if let Some(key) = buf.get(at..at + len) {
                    env.map.warm_entry(key);
                }
            }
        }

        scan
    }

    /// Run every whole command in the read buffer.
    fn execute(&mut self, env: &Env<'_>) {
        // Split so the read buffer, the argument list and the reply buffer are three borrows rather than one, which is what lets a command read its arguments out of one and write its answer into another.
        let Self {
            buf,
            filled,
            at,
            out,
            dialect,
            command,
            closing,
            ..
        } = self;

        // How far into the buffer the map has been asked for lines. Commands are executed one at a time as they always were, and the asking runs ahead of them in steps of [`AHEAD`] commands.
        let mut asked = *at;

        while !*closing && out.len() < MAX_REPLY {
            if *at >= asked {
                asked = Self::ask_ahead(env, buf, *at, *filled);
            }
            let rest = &buf[*at..*filled];
            match rugo_resp::parse(rest, command) {
                Ok(Parsed::Done(used)) => {
                    if !command.is_empty() {
                        // The slice the spans were measured against, which is the one they have to be read out of.
                        let seen = &rest[..used];
                        if dispatch::run(env, command, seen, out, dialect) == Reply::Last {
                            *closing = true;
                        }
                    }
                    *at += used;
                }
                Ok(Parsed::More) => break,
                Err(bad) => {
                    // Framing errors are not recoverable on a stream: there is no way to find where the next command begins. Redis says the same thing and hangs up, and so does this.
                    Encoder::new(out, *dialect).error(&format!("ERR {bad}"));
                    *closing = true;
                }
            }
        }

        if *at == *filled {
            *at = 0;
            *filled = 0;
        }
    }

    /// Write what the socket will take.
    fn flush(&mut self) -> io::Result<()> {
        while self.sent < self.out.len() {
            match self.stream.write(&self.out[self.sent..]) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "the socket stopped taking bytes",
                    ));
                }
                Ok(wrote) => self.sent += wrote,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error),
            }
        }

        if self.sent == self.out.len() {
            self.out.clear();
            self.sent = 0;
        } else if self.sent > MAX_REPLY {
            // A partly written buffer is compacted rather than left to grow, because the alternative is a slow reader whose reply buffer records everything it has ever been sent.
            self.out.drain(..self.sent);
            self.sent = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rugo_map::Map;

    use super::*;
    use crate::config::Config;
    use crate::stats::Stats;

    /// Drive `input` through a connection over a socket pair and return what came back.
    ///
    /// A real socket rather than a fake one, because the parts of this worth testing are the ones that only happen against a socket: a short read, a partial command, a peer that hangs up.
    fn exchange(input: &[u8]) -> Vec<u8> {
        let (mine, theirs) = UnixStream::pair().expect("a socket pair");
        theirs.set_nonblocking(true).expect("non-blocking");
        let mut mine = mine;
        mine.write_all(input).expect("the client wrote");
        mine.shutdown(std::net::Shutdown::Write).expect("half shut");

        let map = Map::new(16, 0);
        let stats = Stats::new(1);
        let config = Config::default();
        let env = Env {
            map: &map,
            stats: &stats,
            thread: 0,
            started: std::time::Instant::now(),
            config: &config,
        };

        let mut conn = Conn::new(Stream::Unix(theirs));
        // Two turns: the first reads and answers, the second sees the peer's half close.
        for _ in 0..2 {
            if conn.turn(&env, true, false) == Turn::Close {
                break;
            }
        }
        drop(conn);

        let mut back = Vec::new();
        mine.set_nonblocking(true).expect("non-blocking");
        let mut chunk = [0u8; 4096];
        while let Ok(read) = mine.read(&mut chunk) {
            if read == 0 {
                break;
            }
            back.extend_from_slice(&chunk[..read]);
        }
        back
    }

    #[test]
    fn an_inline_ping_is_answered() {
        // What every readiness probe in the harness sends, and the reason inline commands are supported at all.
        assert_eq!(exchange(b"PING\r\n"), b"+PONG\r\n");
    }

    #[test]
    fn a_pipeline_is_answered_in_order() {
        let input = b"*3\r\n$3\r\nSET\r\n$1\r\na\r\n$1\r\n1\r\n*2\r\n$3\r\nGET\r\n$1\r\na\r\n*1\r\n$4\r\nPING\r\n";
        assert_eq!(exchange(input), b"+OK\r\n$1\r\n1\r\n+PONG\r\n");
    }

    #[test]
    fn half_a_command_waits_for_the_rest() {
        // The client sends a command in two pieces and hangs up. The first turn sees half, the second sees the close, and nothing is answered, which is what a partial request deserves.
        assert_eq!(exchange(b"*2\r\n$3\r\nGET\r\n$1\r\n"), b"");
    }

    #[test]
    fn a_framing_error_is_reported_and_the_connection_goes() {
        let back = exchange(b"*1\r\n+PING\r\n");
        assert!(
            back.starts_with(b"-ERR Protocol error"),
            "expected a protocol error, got {}",
            String::from_utf8_lossy(&back)
        );
    }

    #[test]
    fn quit_is_answered_before_the_connection_closes() {
        assert_eq!(exchange(b"QUIT\r\n"), b"+OK\r\n");
    }

    #[test]
    fn commands_after_quit_are_not_run() {
        // Everything already read is in the buffer, and a client that said goodbye does not get another answer out of it.
        assert_eq!(exchange(b"QUIT\r\nPING\r\n"), b"+OK\r\n");
    }
}
