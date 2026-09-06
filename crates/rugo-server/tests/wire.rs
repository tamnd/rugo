//! Every command in the M2 set, over a real socket, byte for byte.
//!
//! The unit tests in the crate check that each command computes the right answer. This checks that the answer reaches a client in the spelling a Redis client expects, which is a different claim and the one a benchmark depends on. Everything here goes over a unix socket to a server started the way the binary starts one, and every assertion is on the exact bytes on the wire rather than on a parsed value, because a reply that parses is not the same as a reply that is right.
//!
//! `redis-cli` proves the same thing once, on one machine, to whoever was watching. This proves it on every push, on three operating systems.
//!
//! Every test here is skipped under Miri. A real socket is the whole point of the file and Miri has no kernel to open one against, so the interpreter would stop the program rather than fail an assertion, which says nothing about the code.

// `clippy.toml` allows these in tests, but that allowance only reaches items inside a `#[cfg(test)]` module or a `#[test]` function, and the helpers below are neither: they are ordinary functions in a crate that happens to be a test. The reasoning is the same one `clippy.toml` gives — a test that cannot fail loudly has to invent a way to fail, and the inventions are worse.
#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a test says what went wrong by stopping, and this file is only ever a test"
)]

use std::fmt::Write as _;
use std::io::{Read as _, Write as _};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use rugo_server::{Config, Server, Uring};

/// How long a reply may take before the test decides there is not going to be one.
///
/// Generous, because a loaded CI runner is slow, and finite, because the alternative to a timeout here is a test that hangs until the job is killed with no output.
const PATIENCE: Duration = Duration::from_secs(10);

/// The byte ceiling every test server is given.
///
/// Written out rather than left at the default because `CONFIG GET maxmemory` has to answer something, and a test that asserted on the answer while the default moved would be a test about the default.
const MAXMEMORY: usize = 64 * 1024 * 1024;

/// Distinguishes one test's socket from another's, since they all run in the one process.
static NEXT: AtomicU32 = AtomicU32::new(0);

/// A client connected to a server of its own.
struct Client {
    /// The connection.
    stream: UnixStream,
    /// Where it is, so it can be taken away afterwards.
    path: PathBuf,
}

impl Drop for Client {
    fn drop(&mut self) {
        // The server thread runs forever and its `Drop` never gets to do this, so the test that made the socket is the only thing that can clear it up.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A server on a socket of its own, and a client talking to it.
///
/// One server a test rather than one shared between them, because `FLUSHALL` and `DBSIZE` are in the set and a shared map would make those two tests depend on the order the others ran in.
fn talking() -> Client {
    // The readiness loop rather than whichever loop this machine can run. Every test below is a claim about what the bytes on the wire are, which is not a claim about how they got there, and a suite that tested the ring on Linux and the poller everywhere else would be a suite whose passing depended on where it ran. The claim that the two are the same is one test, at the bottom of this file, and it is the only one here that needs a ring.
    talking_on(Uring::No)
}

/// The same, on a named loop.
fn talking_on(uring: Uring) -> Client {
    let path = std::env::temp_dir().join(format!(
        "rugo-wire-{}-{}.sock",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let config = Config {
        threads: 1,
        port: None,
        unixsocket: Some(path.display().to_string()),
        maxmemory: MAXMEMORY,
        // Enough to be a sharded map and few enough that starting one a test costs nothing.
        shards: 64,
        uring,
    };
    let server = Server::new(config).expect("the server bound");
    std::thread::spawn(move || {
        let _ = server.run();
    });

    for _ in 0..1000 {
        if let Ok(stream) = UnixStream::connect(&path) {
            stream.set_read_timeout(Some(PATIENCE)).expect("a timeout");
            let mut client = Client { stream, path };
            // Bound rather than serving is not good enough: the accepting thread has to have taken the connection before the first command can be answered.
            client.expect(b"PING\r\n", "+PONG\r\n");
            return client;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("the server never came up on {}", path.display());
}

/// One command, in the shape a client sends one.
fn cmd(words: &[&str]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", words.len()).into_bytes();
    for word in words {
        out.extend_from_slice(format!("${}\r\n{word}\r\n", word.len()).as_bytes());
    }
    out
}

impl Client {
    /// Send `request` and nothing else.
    fn send(&mut self, request: &[u8]) {
        self.stream.write_all(request).expect("wrote the request");
    }

    /// Read exactly `count` bytes, which is what makes a short reply a failure rather than a hang.
    fn take(&mut self, count: usize) -> String {
        let mut bytes = vec![0u8; count];
        self.stream
            .read_exact(&mut bytes)
            .expect("the reply arrived in full");
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Send `request` and assert the reply is exactly `expect`.
    fn expect(&mut self, request: &[u8], expect: &str) {
        self.send(request);
        let back = self.take(expect.len());
        assert_eq!(
            back.escape_debug().to_string(),
            expect.escape_debug().to_string(),
            "the reply to {} was not what was asked for",
            String::from_utf8_lossy(request).escape_debug()
        );
    }

    /// Send `words` as a command and assert the reply is exactly `expect`.
    fn ask(&mut self, words: &[&str], expect: &str) {
        self.expect(&cmd(words), expect);
    }

    /// One line of reply, terminator and all.
    ///
    /// Read a byte at a time rather than through a `BufReader`, which would swallow the start of the next reply and make every later assertion in the test wrong for a reason that had nothing to do with the server.
    fn line(&mut self) -> String {
        let mut line = Vec::new();
        loop {
            let mut byte = [0u8; 1];
            self.stream.read_exact(&mut byte).expect("a line arrived");
            line.push(byte[0]);
            if byte[0] == b'\n' {
                return String::from_utf8_lossy(&line).into_owned();
            }
        }
    }

    /// The integer a command answered with.
    fn integer(&mut self, words: &[&str]) -> i64 {
        self.send(&cmd(words));
        let line = self.line();
        let text = line
            .strip_prefix(':')
            .and_then(|rest| rest.strip_suffix("\r\n"))
            .unwrap_or_else(|| {
                panic!(
                    "{words:?} answered {} and not an integer",
                    line.escape_debug()
                )
            });
        text.parse().expect("an integer that parses")
    }

    /// The body of a bulk reply.
    fn bulk(&mut self, words: &[&str]) -> String {
        self.send(&cmd(words));
        let header = self.line();
        let count: usize = header
            .strip_prefix('$')
            .and_then(|rest| rest.strip_suffix("\r\n"))
            .and_then(|text| text.parse().ok())
            .unwrap_or_else(|| {
                panic!(
                    "{words:?} answered {} and not a bulk string",
                    header.escape_debug()
                )
            });
        let body = self.take(count + 2);
        body.strip_suffix("\r\n")
            .expect("a bulk string ends in a terminator")
            .to_owned()
    }
}

#[test]
#[cfg_attr(miri, ignore = "Miri has no kernel to open a socket against")]
fn the_string_commands_answer_the_way_redis_answers() {
    let mut client = talking();

    client.ask(&["SET", "k", "v"], "+OK\r\n");
    client.ask(&["GET", "k"], "$1\r\nv\r\n");
    client.ask(&["GET", "absent"], "$-1\r\n");
    client.ask(&["get", "k"], "$1\r\nv\r\n");

    client.ask(&["STRLEN", "k"], ":1\r\n");
    client.ask(&["STRLEN", "absent"], ":0\r\n");

    client.ask(&["MSET", "a", "1", "b", "2"], "+OK\r\n");
    client.ask(
        &["MGET", "a", "b", "absent"],
        "*3\r\n$1\r\n1\r\n$1\r\n2\r\n$-1\r\n",
    );

    client.ask(&["INCR", "n"], ":1\r\n");
    client.ask(&["INCR", "n"], ":2\r\n");
    client.ask(&["INCRBY", "n", "10"], ":12\r\n");
    client.ask(&["DECR", "n"], ":11\r\n");
    client.ask(&["DECRBY", "n", "5"], ":6\r\n");
    // The counter is a value like any other, and a client that reads one after writing it has to see what it wrote.
    client.ask(&["GET", "n"], "$1\r\n6\r\n");

    client.ask(
        &["INCR", "k"],
        "-ERR value is not an integer or out of range\r\n",
    );
    client.ask(
        &["INCRBY", "n", "lots"],
        "-ERR value is not an integer or out of range\r\n",
    );

    client.ask(
        &["GET"],
        "-ERR wrong number of arguments for 'get' command\r\n",
    );
    client.ask(
        &["MSET", "a"],
        "-ERR wrong number of arguments for 'mset' command\r\n",
    );
}

#[test]
#[cfg_attr(miri, ignore = "Miri has no kernel to open a socket against")]
fn set_takes_the_options_redis_gives_it() {
    let mut client = talking();

    client.ask(&["SET", "k", "v"], "+OK\r\n");
    // The key is there, so the condition refuses, and a refusal is a null rather than an error.
    client.ask(&["SET", "k", "other", "NX"], "$-1\r\n");
    client.ask(&["GET", "k"], "$1\r\nv\r\n");
    client.ask(&["SET", "fresh", "v", "NX"], "+OK\r\n");
    client.ask(&["SET", "fresh", "w", "XX"], "+OK\r\n");
    client.ask(&["SET", "absent", "v", "XX"], "$-1\r\n");

    client.ask(&["SET", "k", "v", "EX", "100"], "+OK\r\n");
    // A second may pass between the write and the question, and a test that insisted otherwise would fail once a day for no reason.
    assert!(
        (99..=100).contains(&client.integer(&["TTL", "k"])),
        "an expiry a hundred seconds out did not read back as one"
    );
    client.ask(&["SET", "k", "w", "KEEPTTL"], "+OK\r\n");
    assert!(
        (99..=100).contains(&client.integer(&["TTL", "k"])),
        "KEEPTTL did not keep the expiry"
    );
    client.ask(&["SET", "k", "x"], "+OK\r\n");
    client.ask(&["TTL", "k"], ":-1\r\n");

    client.ask(
        &["SET", "k", "v", "EX", "0"],
        "-ERR invalid expire time in 'set' command\r\n",
    );
    client.ask(&["SET", "k", "v", "SOMEHOW"], "-ERR syntax error\r\n");

    client.ask(&["SETEX", "s", "100", "v"], "+OK\r\n");
    assert!((99..=100).contains(&client.integer(&["TTL", "s"])));
    client.ask(&["PSETEX", "p", "100000", "v"], "+OK\r\n");
    assert!((99..=100).contains(&client.integer(&["TTL", "p"])));
    client.ask(
        &["SETEX", "s", "0", "v"],
        "-ERR invalid expire time in 'setex' command\r\n",
    );
}

#[test]
#[cfg_attr(miri, ignore = "Miri has no kernel to open a socket against")]
fn keys_can_be_counted_expired_and_taken_away() {
    let mut client = talking();

    client.ask(&["MSET", "a", "1", "b", "2", "c", "3"], "+OK\r\n");
    client.ask(&["DBSIZE"], ":3\r\n");
    client.ask(&["EXISTS", "a", "b", "absent"], ":2\r\n");
    // Counted rather than distinct, which is what Redis answers and what a client counting its own writes depends on.
    client.ask(&["EXISTS", "a", "a"], ":2\r\n");
    client.ask(&["DEL", "a", "absent"], ":1\r\n");
    client.ask(&["UNLINK", "b"], ":1\r\n");
    client.ask(&["DBSIZE"], ":1\r\n");

    client.ask(&["TTL", "absent"], ":-2\r\n");
    client.ask(&["TTL", "c"], ":-1\r\n");
    client.ask(&["PTTL", "absent"], ":-2\r\n");
    client.ask(&["PTTL", "c"], ":-1\r\n");

    client.ask(&["EXPIRE", "c", "100"], ":1\r\n");
    client.ask(&["EXPIRE", "absent", "100"], ":0\r\n");
    assert!((99..=100).contains(&client.integer(&["TTL", "c"])));
    assert!((99_000..=100_000).contains(&client.integer(&["PTTL", "c"])));
    client.ask(&["PERSIST", "c"], ":1\r\n");
    client.ask(&["TTL", "c"], ":-1\r\n");
    // Nothing changed, so nothing is reported, even though the key is right there.
    client.ask(&["PERSIST", "c"], ":0\r\n");
    client.ask(&["PERSIST", "absent"], ":0\r\n");

    client.ask(&["PEXPIRE", "c", "100000"], ":1\r\n");
    assert!((99..=100).contains(&client.integer(&["TTL", "c"])));

    // An expiry in the past is a deletion, which is the only reading that leaves the key and the clock agreeing.
    client.ask(&["EXPIRE", "c", "-1"], ":1\r\n");
    client.ask(&["EXISTS", "c"], ":0\r\n");
    client.ask(&["SET", "d", "1"], "+OK\r\n");
    client.ask(&["EXPIREAT", "d", "1"], ":1\r\n");
    client.ask(&["EXISTS", "d"], ":0\r\n");
    client.ask(&["SET", "e", "1"], "+OK\r\n");
    client.ask(&["PEXPIREAT", "e", "1000"], ":1\r\n");
    client.ask(&["EXISTS", "e"], ":0\r\n");

    client.ask(&["SET", "f", "1"], "+OK\r\n");
    client.ask(&["FLUSHALL"], "+OK\r\n");
    client.ask(&["DBSIZE"], ":0\r\n");
    client.ask(&["SET", "g", "1"], "+OK\r\n");
    // The words every client library sends and this server has nothing to do with, accepted rather than refused.
    client.ask(&["FLUSHDB", "ASYNC"], "+OK\r\n");
    client.ask(&["DBSIZE"], ":0\r\n");
}

#[test]
#[cfg_attr(miri, ignore = "Miri has no kernel to open a socket against")]
fn a_connection_can_greet_and_go() {
    let mut client = talking();

    client.ask(&["PING"], "+PONG\r\n");
    client.ask(&["PING", "hello"], "$5\r\nhello\r\n");
    client.ask(&["ECHO", "hello"], "$5\r\nhello\r\n");
    client.ask(&["SELECT", "0"], "+OK\r\n");
    client.ask(&["SELECT", "1"], "-ERR DB index is out of range\r\n");
    client.ask(&["RESET"], "+RESET\r\n");

    client.ask(&["QUIT"], "+OK\r\n");
    // The reply is written before the close, and the close is how the client knows it was the last one.
    let mut rest = Vec::new();
    client
        .stream
        .read_to_end(&mut rest)
        .expect("the server closed rather than hung");
    assert!(rest.is_empty(), "something followed the goodbye");
}

#[test]
#[cfg_attr(miri, ignore = "Miri has no kernel to open a socket against")]
fn resp3_spells_the_replies_that_differ() {
    let mut client = talking();
    let version = env!("CARGO_PKG_VERSION");

    let greeting = |proto: u8| {
        format!(
            "%7\r\n$6\r\nserver\r\n$4\r\nrugo\r\n$7\r\nversion\r\n${}\r\n{version}\r\n$5\r\nproto\r\n:{proto}\r\n$2\r\nid\r\n:0\r\n$4\r\nmode\r\n$10\r\nstandalone\r\n$4\r\nrole\r\n$6\r\nmaster\r\n$7\r\nmodules\r\n*0\r\n",
            version.len()
        )
    };

    // No version asked for, so the dialect does not change and the greeting comes back as a flat array of fourteen, which is how RESP2 spells a map of seven.
    client.send(&cmd(&["HELLO"]));
    assert_eq!(client.line(), "*14\r\n");
    let rest = greeting(2);
    let rest = rest.strip_prefix("%7\r\n").expect("a map header");
    let back = client.take(rest.len());
    assert_eq!(back, rest);

    client.expect(&cmd(&["HELLO", "3"]), &greeting(3));
    // The one reply the two dialects disagree about.
    client.ask(&["GET", "absent"], "_\r\n");
    client.ask(&["CLIENT", "GETNAME"], "_\r\n");
    client.ask(
        &["CONFIG", "GET", "maxmemory"],
        "%1\r\n$9\r\nmaxmemory\r\n$8\r\n67108864\r\n",
    );

    client.ask(
        &["HELLO", "4"],
        "-NOPROTO unsupported protocol version, this server supports 2 and 3\r\n",
    );
    // Still RESP3 after the refusal, because a version that was not accepted did not change anything.
    client.ask(&["GET", "absent"], "_\r\n");

    // `RESET` puts the connection back to where it started, which for a dialect means RESP2.
    client.ask(&["RESET"], "+RESET\r\n");
    client.ask(&["GET", "absent"], "$-1\r\n");
}

#[test]
#[cfg_attr(miri, ignore = "Miri has no kernel to open a socket against")]
fn the_admin_commands_answer_something_every_client_can_read() {
    let mut client = talking();
    client.ask(&["SET", "k", "v"], "+OK\r\n");

    let info = client.bulk(&["INFO"]);
    for wanted in [
        "# Server\r\n",
        "redis_version:7.4.0\r\n",
        &format!("rugo_version:{}\r\n", env!("CARGO_PKG_VERSION")),
        "redis_mode:standalone\r\n",
        "# Memory\r\n",
        "used_memory:",
        &format!("maxmemory:{MAXMEMORY}\r\n"),
        "# Stats\r\n",
        "total_commands_processed:",
        "# Keyspace\r\n",
        "db0:keys=1,expires=0,avg_ttl=0\r\n",
    ] {
        assert!(
            info.contains(wanted),
            "INFO had no {}",
            wanted.escape_debug()
        );
    }

    // A section asked for by name is the only one that comes back, which is what a tool scraping one depends on.
    let memory = client.bulk(&["INFO", "memory"]);
    assert!(memory.contains("# Memory\r\n"));
    assert!(!memory.contains("# Server\r\n"));

    client.ask(&["COMMAND"], "*0\r\n");
    client.ask(&["COMMAND", "COUNT"], ":0\r\n");
    client.ask(&["COMMAND", "DOCS"], "*0\r\n");

    client.ask(
        &["CONFIG", "GET", "maxmemory"],
        &format!("*2\r\n$9\r\nmaxmemory\r\n${}\r\n{MAXMEMORY}\r\n", 8),
    );
    client.ask(&["CONFIG", "GET", "nothing-like-that"], "*0\r\n");
    client.ask(&["CONFIG", "SET", "maxmemory", "1gb"], "+OK\r\n");

    client.ask(&["CLIENT", "GETNAME"], "$-1\r\n");
    client.ask(&["CLIENT", "ID"], ":0\r\n");
    client.ask(&["CLIENT", "SETNAME", "bench"], "+OK\r\n");
}

#[test]
#[cfg_attr(miri, ignore = "Miri has no kernel to open a socket against")]
fn inline_commands_and_pipelines_arrive_the_way_they_were_sent() {
    let mut client = talking();

    // What `cache-bench` probes a server with before it decides the server is up.
    client.expect(b"PING\r\n", "+PONG\r\n");
    client.expect(b"SET inline yes\r\nGET inline\r\n", "+OK\r\n$3\r\nyes\r\n");
    // Redis tolerates either terminator on an inline command, and so does this.
    client.expect(b"PING\n", "+PONG\r\n");

    let mut batch = Vec::new();
    for n in 0..8 {
        batch.extend_from_slice(&cmd(&["SET", &format!("k{n}"), &format!("v{n}")]));
    }
    for n in 0..8 {
        batch.extend_from_slice(&cmd(&["GET", &format!("k{n}")]));
    }
    let mut expect = "+OK\r\n".repeat(8);
    for n in 0..8 {
        let _ = write!(expect, "$2\r\nv{n}\r\n");
    }
    // One write, one read: the replies have to come back in the order the commands went out, and all of them have to come back.
    client.expect(&batch, &expect);
}

#[test]
#[cfg_attr(miri, ignore = "Miri has no kernel to open a socket against")]
fn a_reply_too_big_for_the_socket_arrives_whole() {
    // The path where a turn wants something different from what the poller is already doing, which is the only path that still re-registers a descriptor.
    //
    // A reply larger than the socket's send buffer cannot be written in one go, so the connection asks to be told when it is writable, gets told, drains, and asks to go back to being told when it is readable. Every one of those steps is a change of interest, and a connection that never got re-registered for writing would stop here with the client waiting for bytes that were never sent.
    //
    // Two hundred and fifty-six values of eight kilobytes is two megabytes of reply against a send buffer that is a few hundred kilobytes at most on either platform, so the write blocks partway whatever the operating system chose.
    let mut client = talking();

    let value = "v".repeat(8 * 1024);
    let count = 256;
    for n in 0..count {
        client.ask(&["SET", &format!("big{n}"), &value], "+OK\r\n");
    }

    let mut words = vec!["MGET".to_owned()];
    for n in 0..count {
        words.push(format!("big{n}"));
    }
    let borrowed: Vec<&str> = words.iter().map(String::as_str).collect();

    let mut expect = format!("*{count}\r\n");
    for _ in 0..count {
        let _ = write!(expect, "${}\r\n{value}\r\n", value.len());
    }
    assert!(
        expect.len() > 2 * 1024 * 1024,
        "{} bytes is not big enough to block a write",
        expect.len()
    );
    client.ask(&borrowed, &expect);

    // And the connection is usable afterwards, which is what says it went back to watching for reads rather than staying armed for a write nobody owes.
    client.ask(&["PING"], "+PONG\r\n");
}

#[test]
#[cfg_attr(miri, ignore = "Miri has no kernel to open a socket against")]
fn a_command_this_server_does_not_have_is_an_error_rather_than_a_silence() {
    let mut client = talking();

    client.ask(&["LPUSH", "list", "x"], "-ERR unknown command 'LPUSH'\r\n");
    // Echoed the way it was typed, because a client that mistyped wants to see what it typed.
    client.ask(&["lpush"], "-ERR unknown command 'lpush'\r\n");
    // Longer than any name this server has, so it is refused without being upper-cased into a buffer it would not fit.
    let long = "X".repeat(64);
    client.ask(&[&long], &format!("-ERR unknown command '{long}'\r\n"));
    // And the connection is still good afterwards, which is the part that matters.
    client.ask(&["PING"], "+PONG\r\n");
}

// The claim the second loop has to earn: the same bytes, in the same order, for the same requests.
//
// Everything above tests the readiness loop, because that is the one every machine can run. This is the only test that needs a ring, and it makes the comparison rather than restating the expectations, so it says the loops agree rather than that both agree with something written twice.
//
// It skips where there is no ring. A container's seccomp profile has the last word on whether the syscall is allowed, and a test that fails inside one is a test that gets deleted.
#[cfg(target_os = "linux")]
#[test]
#[cfg_attr(miri, ignore = "Miri has no kernel to open a socket against")]
fn the_ring_answers_what_the_poller_answers() {
    if !rugo_net::uring::Ring::available() {
        return;
    }

    // Larger than a socket buffer, so the reply takes more than one write and the loop has to come back for the rest of it.
    let big = "v".repeat(256 * 1024);
    let bulk = |value: &str| format!("${}\r\n{value}\r\n", value.len()).len();

    // The request, and how long its reply is. The length is computed rather than the reply, because what is being compared here is the two servers against each other.
    let script: Vec<(Vec<u8>, usize)> = vec![
        (cmd(&["PING"]), "+PONG\r\n".len()),
        (cmd(&["SET", "one", "1"]), "+OK\r\n".len()),
        (cmd(&["INCRBY", "one", "41"]), ":42\r\n".len()),
        (cmd(&["GET", "one"]), bulk("42")),
        (cmd(&["GET", "missing"]), "$-1\r\n".len()),
        (cmd(&["SET", "big", &big]), "+OK\r\n".len()),
        (cmd(&["STRLEN", "big"]), format!(":{}\r\n", big.len()).len()),
        (cmd(&["GET", "big"]), bulk(&big)),
        // Three commands in one write, which is the shape the sweep sends and the one a completion loop answers with a single send.
        (
            [
                cmd(&["GET", "one"]),
                cmd(&["PING"]),
                cmd(&["EXISTS", "big"]),
            ]
            .concat(),
            bulk("42") + "+PONG\r\n".len() + ":1\r\n".len(),
        ),
        // An inline command, which parses down a different path from everything above it.
        (b"PING\r\n".to_vec(), "+PONG\r\n".len()),
    ];

    let run = |uring: Uring| -> String {
        let mut client = talking_on(uring);
        let mut back = String::new();
        for (request, len) in &script {
            client.send(request);
            back.push_str(&client.take(*len));
        }
        back
    };

    let poller = run(Uring::No);
    let ring = run(Uring::Yes);

    // Compared by hand rather than with `assert_eq`, whose failure message would be six megabytes of v.
    let at = poller.bytes().zip(ring.bytes()).position(|(a, b)| a != b);
    assert!(
        at.is_none() && poller.len() == ring.len(),
        "the loops disagree at byte {at:?}, over {} bytes from the poller and {} from the ring",
        poller.len(),
        ring.len()
    );
}
