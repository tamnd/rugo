//! `cargo xtask load`, which drives a running server hard enough to compare it with itself.
//!
//! # Why this exists next to a benchmark harness that already works
//!
//! The harness of record is [cache-bench](https://github.com/tamnd/cache-bench) driving `memtier_benchmark`, and every number that goes in a document here comes from it. That harness measures eight servers against each other, which means every box it runs on has to have all eight built on it, and building `memtier_benchmark` alone wants a compiler toolchain, an event library, a regular expression library and a package manager to install them with.
//!
//! This asks a much smaller question: is this build of rugo faster than that build of rugo, on a box that is quiet. It needs nothing but the server binary and the toolchain that built it, so it runs on any machine the repository already builds on, and it answers in a minute rather than in a provisioning session.
//!
//! # What its numbers mean and do not mean
//!
//! Only rugo against rugo, on one box, in one sitting. A throughput figure from here does not belong next to a figure for Redis or for pogocache, and it is not what `SCOREBOARD.md` is generated from.
//!
//! Read the server's processor time an operation rather than the operations a second. This client is one thread a connection doing blocking reads, so on a small box it can be the thing that runs out of room first, and when it is, both sides of an A and B read the same throughput and the comparison says nothing. Processor time an operation is measured on the server's own threads and is what it costs the server to answer, whether or not the client could ask any faster. It also needs the server's process id, which is the `--pid` flag, and the file it reads is Linux's `/proc`, so elsewhere that half is simply left out.
//!
//! # The shape of the load
//!
//! Keys are written before the clock starts, so a run measures a cache that has what is being asked for rather than a cache that is mostly missing. After that every connection sends a pipeline of commands, waits for all of their replies, and sends the next one, which is what a benchmark client does and is where a server that batches its reads and writes is supposed to show it.

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::TcpStream;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Instant;

/// What `cargo xtask load --help` prints.
pub(crate) const USAGE: &str = "\
cargo xtask load [options]

Options:
  --socket <path>     unix socket to drive
  --port <n>          TCP port to drive instead (default 6379 if no socket)
  --connections <n>   connections, one thread each (default 16)
  --pipeline <n>      commands in flight on a connection (default 25)
  --ops <n>           operations in total, shared between connections (default 1000000)
  --keys <n>          how many distinct keys (default 250000)
  --value <n>         value bytes (default 100)
  --ratio <s:g>       sets to gets (default 1:10)
  --pid <n>           the server's process id, to report what it spent
  --seed <n>          the key sequence, for a run that repeats (default 1)
";

/// Where the server is listening.
#[derive(Debug, Clone)]
enum Where {
    /// A unix socket, which is what a benchmark on one box should be using.
    #[cfg(unix)]
    Socket(PathBuf),
    /// A TCP port on the loopback.
    Port(u16),
}

/// What to send and how much of it.
#[derive(Debug, Clone)]
struct Load {
    /// Where the server is.
    at: Where,
    /// Connections, and therefore client threads.
    connections: usize,
    /// Commands a connection has outstanding at once.
    pipeline: usize,
    /// Operations in total.
    ops: usize,
    /// How many distinct keys the load touches.
    keys: u64,
    /// Value bytes on every write.
    value: usize,
    /// Sets per set plus gets, as a fraction of a thousand, so a ratio of one to ten is ninety.
    sets_per_mille: u64,
    /// The server's process id, when it was given.
    pid: Option<u32>,
    /// What the key sequence is drawn from.
    seed: u64,
}

impl Default for Load {
    fn default() -> Self {
        Self {
            at: Where::Port(6379),
            connections: 16,
            pipeline: 25,
            ops: 1_000_000,
            keys: 250_000,
            value: 100,
            sets_per_mille: 90,
            pid: None,
            seed: 1,
        }
    }
}

/// What one connection did.
#[derive(Debug, Default)]
struct Did {
    /// Operations it completed.
    ops: usize,
    /// The first error reply it was sent, if it was sent one.
    error: Option<String>,
}

/// Run the load described by `args`.
pub(crate) fn run(args: &[String]) -> Result<(), String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print!("{USAGE}");
        return Ok(());
    }
    let load = parse(args)?;
    let before = cpu_of(load.pid);
    let client_before = cpu_of(Some(std::process::id()));

    fill(&load)?;

    let clock = Instant::now();
    let did = drive(&load)?;
    let elapsed = clock.elapsed().as_secs_f64();

    let after = cpu_of(load.pid);
    let client_after = cpu_of(Some(std::process::id()));

    let ops: usize = did.iter().map(|one| one.ops).sum();
    println!("{ops} operations in {elapsed:.3} s");
    println!("{:.0} operations a second", rate(ops, elapsed));
    report("server", before, after, ops);
    report("client", client_before, client_after, ops);

    if let Some(error) = did.iter().find_map(|one| one.error.clone()) {
        return Err(format!("the server answered with an error: {error}"));
    }
    Ok(())
}

/// Read the flags, leaving everything not named at its default.
fn parse(args: &[String]) -> Result<Load, String> {
    let mut load = Load::default();
    let mut socket: Option<PathBuf> = None;
    let mut port: Option<u16> = None;
    let mut rest = args.iter();

    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--socket" => socket = Some(PathBuf::from(after(flag, &mut rest)?)),
            "--port" => port = Some(number(after(flag, &mut rest)?)?),
            "--connections" => load.connections = number(after(flag, &mut rest)?)?,
            "--pipeline" => load.pipeline = number(after(flag, &mut rest)?)?,
            "--ops" => load.ops = number(after(flag, &mut rest)?)?,
            "--keys" => load.keys = number(after(flag, &mut rest)?)?,
            "--value" => load.value = number(after(flag, &mut rest)?)?,
            "--ratio" => load.sets_per_mille = ratio(after(flag, &mut rest)?)?,
            "--pid" => load.pid = Some(number(after(flag, &mut rest)?)?),
            "--seed" => load.seed = number(after(flag, &mut rest)?)?,
            other => return Err(format!("no flag called {other}")),
        }
    }

    // A socket wins over a port, because naming one is the only reason to name it.
    load.at = match (socket, port) {
        #[cfg(unix)]
        (Some(path), _) => Where::Socket(path),
        #[cfg(not(unix))]
        (Some(_), _) => return Err("this machine has no unix sockets, so use --port".to_owned()),
        (None, Some(port)) => Where::Port(port),
        (None, None) => load.at,
    };

    if load.connections == 0 || load.pipeline == 0 || load.keys == 0 {
        return Err("connections, pipeline and keys all have to be more than nought".to_owned());
    }
    Ok(load)
}

/// Whatever came after `flag`, which every flag here wants.
fn after<'a>(flag: &str, rest: &mut std::slice::Iter<'a, String>) -> Result<&'a str, String> {
    rest.next()
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} wants a value after it"))
}

/// One flag's value as a number.
fn number<T: std::str::FromStr>(text: &str) -> Result<T, String> {
    text.parse()
        .map_err(|_| format!("{text} is not a number this flag can take"))
}

/// A `sets:gets` ratio as sets per thousand operations.
fn ratio(text: &str) -> Result<u64, String> {
    let (sets, gets) = text
        .split_once(':')
        .ok_or_else(|| format!("{text} is not a ratio, which is written as sets:gets"))?;
    let sets: u64 = number(sets)?;
    let gets: u64 = number(gets)?;
    let total = sets + gets;
    if total == 0 {
        return Err("a ratio of nought to nought asks for nothing".to_owned());
    }
    Ok(sets * 1000 / total)
}

/// Write every key once, so that the gets in the timed run find something.
///
/// Untimed and single connection, because it is setup rather than measurement and a minute of it either way changes nothing.
fn fill(load: &Load) -> Result<(), String> {
    let mut wire = connect(&load.at)?;
    let mut out = Vec::with_capacity(load.pipeline * (load.value + 64));
    let value = vec![b'v'; load.value];
    let mut key = String::new();
    let mut sent = 0;

    for at in 0..load.keys {
        key.clear();
        let _ = write!(key, "key:{at}");
        set(&mut out, key.as_bytes(), &value);
        sent += 1;
        if sent == load.pipeline {
            exchange(wire.as_mut(), &out, sent)?;
            out.clear();
            sent = 0;
        }
    }
    if sent > 0 {
        exchange(wire.as_mut(), &out, sent)?;
    }
    Ok(())
}

/// Run every connection until between them they have done the whole load.
fn drive(load: &Load) -> Result<Vec<Did>, String> {
    let each = load.ops.div_ceil(load.connections);
    std::thread::scope(|scope| {
        let mut threads = Vec::with_capacity(load.connections);
        for at in 0..load.connections {
            threads.push(scope.spawn(move || one(load, at, each)));
        }
        threads
            .into_iter()
            .map(|thread| {
                thread
                    .join()
                    .map_err(|_| "a load thread went down".to_owned())?
            })
            .collect()
    })
}

/// One connection's whole run.
fn one(load: &Load, at: usize, ops: usize) -> Result<Did, String> {
    let mut wire = connect(&load.at)?;
    let mut out = Vec::with_capacity(load.pipeline * (load.value + 64));
    let value = vec![b'v'; load.value];
    let mut key = String::new();
    let mut did = Did::default();

    // Seeded from the connection number as well as the run's seed, so two connections do not walk the same keys in step.
    let step = u64::try_from(at)
        .unwrap_or(0)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut roll = load.seed ^ (step | 1);
    let mut done = 0;

    while done < ops {
        let batch = load.pipeline.min(ops - done);
        out.clear();
        for _ in 0..batch {
            roll = mix(roll);
            key.clear();
            let _ = write!(key, "key:{}", roll % load.keys);
            // The coin comes off the top bits and the key off the bottom, because drawing both from the same end would tie which key is written to which key it is.
            if (roll >> 40) % 1000 < load.sets_per_mille {
                set(&mut out, key.as_bytes(), &value);
            } else {
                get(&mut out, key.as_bytes());
            }
        }
        if let Some(error) = exchange(wire.as_mut(), &out, batch)? {
            did.error.get_or_insert(error);
        }
        done += batch;
        did.ops += batch;
    }
    Ok(did)
}

/// Send `out` and wait for `want` whole replies, reporting the first error reply among them.
fn exchange(wire: &mut dyn ReadWrite, out: &[u8], want: usize) -> Result<Option<String>, String> {
    wire.write_all(out)
        .map_err(|why| format!("the server would not take the commands: {why}"))?;

    let mut held: Vec<u8> = Vec::new();
    // On the heap rather than the stack, because sixty-four kilobytes of stack a thread is a lot to ask of the smallest default there is.
    let mut chunk = vec![0u8; 64 * 1024];
    let mut seen = 0;
    let mut error = None;

    while seen < want {
        let read = wire
            .read(&mut chunk)
            .map_err(|why| format!("the server stopped answering: {why}"))?;
        if read == 0 {
            return Err("the server closed the connection".to_owned());
        }
        held.extend_from_slice(chunk.get(..read).unwrap_or_default());
        let (whole, took, first) = replies(&held);
        seen += whole;
        if error.is_none() {
            error = first;
        }
        held.drain(..took);
    }
    Ok(error)
}

/// How many whole replies are at the front of `bytes`, how many bytes they take, and the first error among them.
fn replies(bytes: &[u8]) -> (usize, usize, Option<String>) {
    let mut at = 0;
    let mut count = 0;
    let mut error = None;

    while at < bytes.len() {
        let Some(end) = crlf(bytes, at) else { break };
        let Some(&kind) = bytes.get(at) else { break };
        let next = if kind == b'$' {
            let Some(len) = integer(bytes.get(at + 1..end).unwrap_or_default()) else {
                break;
            };
            match usize::try_from(len) {
                // The bytes of the value, and the pair that ends them.
                Ok(len) => end + 2 + len + 2,
                // A missing key, which is the whole reply.
                Err(_) => end + 2,
            }
        } else {
            if kind == b'-' && error.is_none() {
                error = Some(
                    String::from_utf8_lossy(bytes.get(at + 1..end).unwrap_or_default())
                        .into_owned(),
                );
            }
            end + 2
        };
        if next > bytes.len() {
            break;
        }
        at = next;
        count += 1;
    }
    (count, at, error)
}

/// Where the next `\r\n` starts at or after `from`.
fn crlf(bytes: &[u8], from: usize) -> Option<usize> {
    bytes
        .get(from..)?
        .windows(2)
        .position(|pair| pair == b"\r\n")
        .map(|at| from + at)
}

/// A decimal number, which may be negative because a missing key is a length of minus one.
fn integer(bytes: &[u8]) -> Option<i64> {
    std::str::from_utf8(bytes).ok()?.parse().ok()
}

/// Put `SET key value` on the end of `out`.
fn set(out: &mut Vec<u8>, key: &[u8], value: &[u8]) {
    out.extend_from_slice(b"*3\r\n$3\r\nSET\r\n");
    bulk(out, key);
    bulk(out, value);
}

/// Put `GET key` on the end of `out`.
fn get(out: &mut Vec<u8>, key: &[u8]) {
    out.extend_from_slice(b"*2\r\n$3\r\nGET\r\n");
    bulk(out, key);
}

/// Put one bulk string on the end of `out`.
fn bulk(out: &mut Vec<u8>, bytes: &[u8]) {
    let mut head = String::new();
    let _ = write!(head, "${}\r\n", bytes.len());
    out.extend_from_slice(head.as_bytes());
    out.extend_from_slice(bytes);
    out.extend_from_slice(b"\r\n");
}

/// The splitmix64 finaliser, so that consecutive keys are not consecutive numbers.
const fn mix(x: u64) -> u64 {
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Anything a connection can be, so that a unix socket and a TCP socket are one type here.
trait ReadWrite: Read + Write + Send {}

impl<T: Read + Write + Send> ReadWrite for T {}

/// Open one connection.
fn connect(at: &Where) -> Result<Box<dyn ReadWrite>, String> {
    match at {
        #[cfg(unix)]
        Where::Socket(path) => {
            let wire = UnixStream::connect(path)
                .map_err(|why| format!("no server on {}: {why}", path.display()))?;
            Ok(Box::new(wire))
        }
        Where::Port(port) => {
            let wire = TcpStream::connect(("127.0.0.1", *port))
                .map_err(|why| format!("no server on port {port}: {why}"))?;
            // Every request here is a whole pipeline, so waiting to see whether more is coming can only add latency.
            let _ = wire.set_nodelay(true);
            Ok(Box::new(wire))
        }
    }
}

/// Processor time a process has spent, in seconds, or nothing where that cannot be read.
///
/// Linux only, and read from `/proc/<pid>/stat`, whose fourteenth and fifteenth fields are the user and system time in clock ticks. The tick is a hundred a second on every Linux this is likely to run on, which is a constant the kernel does not export to a reader of that file.
fn cpu_of(pid: Option<u32>) -> Option<f64> {
    let text = std::fs::read_to_string(format!("/proc/{}/stat", pid?)).ok()?;
    // The second field is the command name in brackets and may hold spaces, so counting starts after it rather than from the front.
    let tail = text.rsplit_once(')')?.1;
    let fields: Vec<&str> = tail.split_whitespace().collect();
    let user: u64 = fields.get(11)?.parse().ok()?;
    let system: u64 = fields.get(12)?.parse().ok()?;
    Some(ticks(user + system))
}

/// Clock ticks as seconds.
#[expect(
    clippy::cast_precision_loss,
    reason = "a process would have to have run for millions of years for this to lose a tick"
)]
fn ticks(count: u64) -> f64 {
    count as f64 / 100.0
}

/// Operations a second.
#[expect(
    clippy::cast_precision_loss,
    reason = "an operation count large enough to lose a digit here is more than a run can do"
)]
fn rate(ops: usize, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        return 0.0;
    }
    ops as f64 / seconds
}

/// Print what one side spent, when it could be measured.
#[expect(
    clippy::cast_precision_loss,
    reason = "the same operation count as above, divided rather than multiplied"
)]
fn report(who: &str, before: Option<f64>, after: Option<f64>, ops: usize) {
    let (Some(before), Some(after)) = (before, after) else {
        return;
    };
    let spent = after - before;
    if ops == 0 {
        return;
    }
    let each = spent * 1e6 / ops as f64;
    println!("{who} processor time {spent:.3} s, {each:.3} microseconds an operation");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_replies_are_counted_and_a_torn_one_is_not() {
        let (count, took, error) = replies(b"+OK\r\n$3\r\nabc\r\n$-1\r\n:7\r\n");
        assert_eq!(count, 4);
        assert_eq!(took, b"+OK\r\n$3\r\nabc\r\n$-1\r\n:7\r\n".len());
        assert_eq!(error, None);

        // A bulk reply whose value has not all arrived yet, which is the case that decides whether the client waits or runs ahead of the server.
        let (count, took, _) = replies(b"+OK\r\n$5\r\nab");
        assert_eq!(count, 1);
        assert_eq!(took, 5);
    }

    #[test]
    fn an_error_reply_is_reported_rather_than_counted_as_an_answer() {
        let (count, _, error) = replies(b"+OK\r\n-ERR no room\r\n");
        assert_eq!(count, 2);
        assert_eq!(error.as_deref(), Some("ERR no room"));
    }

    #[test]
    fn a_ratio_is_read_as_writes_in_a_thousand() {
        assert_eq!(ratio("1:10"), Ok(90));
        assert_eq!(ratio("1:0"), Ok(1000));
        assert_eq!(ratio("0:1"), Ok(0));
        assert!(ratio("half").is_err());
        assert!(ratio("0:0").is_err());
    }
}
