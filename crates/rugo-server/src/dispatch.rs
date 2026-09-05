//! What each command does.
//!
//! One function a family — strings, keyspace, connection, admin — tried in turn, each saying whether the name was one of its own. Nothing here touches a socket: a command reads its arguments out of the read buffer and writes its answer into the reply buffer, and the connection decides when either of those meets the network.
//!
//! The set is what a cache needs and what the benchmark and `redis-cli` ask for, and no more. There is no `LPUSH` here and there is not going to be one.

use std::fmt::{self, Write as _};
use std::time::Instant;

use rugo_map::{Expiry, Map, Uncounted, When};
use rugo_resp::{Command, Dialect, Encoder};

use crate::config::Config;
use crate::stats::{Counters, Stats};

/// The longest command name this server has, rounded up.
///
/// A name longer than this cannot be one of ours, which is what lets the name be upper-cased into a fixed buffer with no allocation and no bound to check twice.
const MAX_NAME: usize = 24;

/// What the server claims to be when a client asks for a Redis version.
///
/// This is a compatibility claim and not an identity: clients gate features on this number, and one that read `0.1.0` would have every library decide the server cannot do `RESP3` or `EXAT`. What rugo actually is, is on the next line of `INFO` as `rugo_version`.
const REDIS_VERSION: &str = "7.4.0";

/// Whether the connection should go after this reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reply {
    /// Carry on.
    More,
    /// The client said goodbye, so write this and close.
    Last,
}

/// Everything a command may look at that is not its own arguments.
#[derive(Debug)]
pub(crate) struct Env<'a> {
    /// The cache.
    pub map: &'a Map,
    /// Every thread's counters.
    pub stats: &'a Stats,
    /// Which thread is running this, and so which block of counters is this thread's own.
    pub thread: usize,
    /// When the server started, for `INFO`'s uptime.
    pub started: Instant,
    /// How the server was configured, for the rest of `INFO`.
    pub config: &'a Config,
}

impl Env<'_> {
    /// This thread's counters.
    #[inline]
    fn counters(&self) -> &Counters {
        self.stats.thread(self.thread)
    }
}

/// Run one command, writing its reply.
pub(crate) fn run(
    env: &Env<'_>,
    cmd: &Command,
    buf: &[u8],
    out: &mut Vec<u8>,
    dialect: &mut Dialect,
) -> Reply {
    env.counters().command();
    let mut reply = Encoder::new(out, *dialect);

    let Some(raw) = cmd.arg(0, buf) else {
        return Reply::More;
    };
    if raw.len() > MAX_NAME {
        unknown(&mut reply, raw);
        return Reply::More;
    }
    let mut upper = [0u8; MAX_NAME];
    for (to, from) in upper.iter_mut().zip(raw) {
        *to = from.to_ascii_uppercase();
    }
    let name = &upper[..raw.len()];

    let mut last = Reply::More;
    let known = strings(env, name, cmd, buf, &mut reply)
        || keyspace(env, name, cmd, buf, &mut reply)
        || connection(name, cmd, buf, &mut reply, dialect, &mut last)
        || admin(env, name, cmd, buf, &mut reply);
    if !known {
        unknown(&mut reply, raw);
    }
    last
}

/// `GET`, `SET` and the counters: everything that reads or writes one value.
fn strings(env: &Env<'_>, name: &[u8], cmd: &Command, buf: &[u8], reply: &mut Encoder<'_>) -> bool {
    match name {
        b"GET" => {
            if !arity(cmd, 2, "get", reply) {
                return true;
            }
            let key = arg(cmd, buf, 1);
            if env.map.get(key, |value| reply.bulk(value)).is_none() {
                env.counters().miss();
                reply.null();
            } else {
                env.counters().hit();
            }
        }
        b"SET" => set(env, cmd, buf, reply),
        b"SETEX" | b"PSETEX" => timed_set(env, name == b"PSETEX", cmd, buf, reply),
        b"MGET" => {
            if cmd.len() < 2 {
                arity(cmd, 2, "mget", reply);
                return true;
            }
            reply.array(cmd.len() - 1);
            for n in 1..cmd.len() {
                let key = arg(cmd, buf, n);
                if env.map.get(key, |value| reply.bulk(value)).is_none() {
                    env.counters().miss();
                    reply.null();
                } else {
                    env.counters().hit();
                }
            }
        }
        b"MSET" => {
            if cmd.len() < 3 || cmd.len().is_multiple_of(2) {
                arity(cmd, 3, "mset", reply);
                return true;
            }
            for pair in (1..cmd.len()).step_by(2) {
                if env
                    .map
                    .set(arg(cmd, buf, pair), arg(cmd, buf, pair + 1), None, None)
                    .is_err()
                {
                    oom(reply);
                    return true;
                }
            }
            reply.simple(b"OK");
        }
        b"INCR" | b"DECR" => {
            let word = if name == b"INCR" { "incr" } else { "decr" };
            if !arity(cmd, 2, word, reply) {
                return true;
            }
            let by = if name == b"INCR" { 1 } else { -1 };
            count(env, arg(cmd, buf, 1), by, reply);
        }
        b"INCRBY" | b"DECRBY" => {
            let word = if name == b"INCRBY" {
                "incrby"
            } else {
                "decrby"
            };
            if !arity(cmd, 3, word, reply) {
                return true;
            }
            let Some(by) = integer(arg(cmd, buf, 2), reply) else {
                return true;
            };
            // Negating the most negative integer is the one value this cannot do, and Redis refuses it for the same reason.
            let Some(by) = (if name == b"INCRBY" {
                Some(by)
            } else {
                by.checked_neg()
            }) else {
                reply.error("ERR decrement would overflow");
                return true;
            };
            count(env, arg(cmd, buf, 1), by, reply);
        }
        b"STRLEN" => {
            if !arity(cmd, 2, "strlen", reply) {
                return true;
            }
            let len = env.map.get(arg(cmd, buf, 1), <[u8]>::len).unwrap_or(0);
            reply.integer(i64::try_from(len).unwrap_or(i64::MAX));
        }
        _ => return false,
    }
    true
}

/// The commands that are about keys rather than values: existence, expiry, and how many there are.
fn keyspace(
    env: &Env<'_>,
    name: &[u8],
    cmd: &Command,
    buf: &[u8],
    reply: &mut Encoder<'_>,
) -> bool {
    match name {
        b"DEL" | b"UNLINK" => {
            if cmd.len() < 2 {
                arity(cmd, 2, "del", reply);
                return true;
            }
            let went = (1..cmd.len())
                .filter(|n| env.map.remove(arg(cmd, buf, *n)))
                .count();
            reply.integer(i64::try_from(went).unwrap_or(i64::MAX));
        }
        b"EXISTS" => {
            if cmd.len() < 2 {
                arity(cmd, 2, "exists", reply);
                return true;
            }
            // Counted rather than distinct, so `EXISTS k k` on one key answers two, which is what Redis answers.
            let there = (1..cmd.len())
                .filter(|n| env.map.contains(arg(cmd, buf, *n)))
                .count();
            reply.integer(i64::try_from(there).unwrap_or(i64::MAX));
        }
        b"EXPIRE" | b"PEXPIRE" | b"EXPIREAT" | b"PEXPIREAT" => {
            let word = match name {
                b"EXPIRE" => "expire",
                b"PEXPIRE" => "pexpire",
                b"EXPIREAT" => "expireat",
                _ => "pexpireat",
            };
            if !arity(cmd, 3, word, reply) {
                return true;
            }
            let Some(value) = integer(arg(cmd, buf, 2), reply) else {
                return true;
            };
            let millis = name.starts_with(b"P");
            let absolute = name.ends_with(b"AT");
            expire(env, arg(cmd, buf, 1), value, millis, absolute, reply);
        }
        b"TTL" | b"PTTL" => {
            let word = if name == b"TTL" { "ttl" } else { "pttl" };
            if !arity(cmd, 2, word, reply) {
                return true;
            }
            let now = env.map.clock().now();
            reply.integer(match env.map.deadline(arg(cmd, buf, 1)) {
                // Two different absences, and a client that could not tell them apart could not tell a key with no expiry from a key that is gone.
                None => -2,
                Some(None) => -1,
                Some(Some(when)) => {
                    let left = i64::from(when.saturating_sub(now));
                    if name == b"TTL" { left } else { left * 1000 }
                }
            });
        }
        b"PERSIST" => {
            if !arity(cmd, 2, "persist", reply) {
                return true;
            }
            // A key that never had an expiry answers nought, so the reply says whether anything changed rather than whether the key is there.
            let had = env
                .map
                .deadline(arg(cmd, buf, 1))
                .is_some_and(|when| when.is_some());
            match env.map.expire(arg(cmd, buf, 1), None) {
                Ok(_) => reply.integer(i64::from(had)),
                Err(_) => oom(reply),
            }
        }
        b"DBSIZE" => reply.integer(i64::try_from(env.map.len()).unwrap_or(i64::MAX)),
        b"FLUSHALL" | b"FLUSHDB" => {
            // `ASYNC` and `SYNC` are accepted and ignored: there is no background thread to hand the work to, and refusing a word every client library sends would be worse than doing it now.
            env.map.clear();
            reply.simple(b"OK");
        }
        _ => return false,
    }
    true
}

/// What a client says before and after it says anything else.
fn connection(
    name: &[u8],
    cmd: &Command,
    buf: &[u8],
    reply: &mut Encoder<'_>,
    dialect: &mut Dialect,
    last: &mut Reply,
) -> bool {
    match name {
        b"PING" => match cmd.len() {
            1 => reply.simple(b"PONG"),
            2 => reply.bulk(arg(cmd, buf, 1)),
            _ => {
                arity(cmd, 2, "ping", reply);
            }
        },
        b"ECHO" => {
            if arity(cmd, 2, "echo", reply) {
                reply.bulk(arg(cmd, buf, 1));
            }
        }
        b"HELLO" => {
            if cmd.len() > 1 {
                match integer_of(arg(cmd, buf, 1)) {
                    Some(2) => *dialect = Dialect::Resp2,
                    Some(3) => *dialect = Dialect::Resp3,
                    _ => {
                        reply.error(
                            "NOPROTO unsupported protocol version, this server supports 2 and 3",
                        );
                        return true;
                    }
                }
            }
            // The encoder in hand still has the old dialect, and the greeting has to be written in the new one or a client that asked for RESP3 gets a RESP2 map and stops.
            hello(*dialect, reply);
        }
        b"SELECT" => {
            if !arity(cmd, 2, "select", reply) {
                return true;
            }
            // One keyspace, and a client that asked for another would be writing where it did not think it was.
            match integer_of(arg(cmd, buf, 1)) {
                Some(0) => reply.simple(b"OK"),
                _ => reply.error("ERR DB index is out of range"),
            }
        }
        b"RESET" => {
            *dialect = Dialect::Resp2;
            reply.simple(b"RESET");
        }
        b"QUIT" => {
            reply.simple(b"OK");
            *last = Reply::Last;
        }
        _ => return false,
    }
    true
}

/// The commands that are about the server rather than the data.
fn admin(env: &Env<'_>, name: &[u8], cmd: &Command, buf: &[u8], reply: &mut Encoder<'_>) -> bool {
    match name {
        b"INFO" => {
            let section = if cmd.len() > 1 {
                String::from_utf8_lossy(arg(cmd, buf, 1)).to_lowercase()
            } else {
                "all".to_owned()
            };
            reply.bulk(info(env, &section).as_bytes());
        }
        b"COMMAND" => {
            // Introspection this server does not have. An empty answer is what a client that asked gets, and every client library treats that as "no help available" rather than as an error.
            match cmd.len() {
                1 => reply.array(0),
                _ if arg(cmd, buf, 1).eq_ignore_ascii_case(b"COUNT") => reply.integer(0),
                _ if arg(cmd, buf, 1).eq_ignore_ascii_case(b"DOCS") => reply.map(0),
                _ => reply.array(0),
            }
        }
        b"CONFIG" => {
            if cmd.len() > 1 && arg(cmd, buf, 1).eq_ignore_ascii_case(b"GET") {
                config(env, cmd, buf, reply);
            } else {
                // `SET` and `RESETSTAT` are accepted and do nothing. There is no configuration to change at runtime, and a client that could not set one should not be stopped from starting.
                reply.simple(b"OK");
            }
        }
        b"CLIENT" => {
            if cmd.len() > 1 && arg(cmd, buf, 1).eq_ignore_ascii_case(b"GETNAME") {
                reply.null();
            } else if cmd.len() > 1 && arg(cmd, buf, 1).eq_ignore_ascii_case(b"ID") {
                reply.integer(0);
            } else {
                reply.simple(b"OK");
            }
        }
        b"SHUTDOWN" => {
            // No reply, because Redis sends none: the connection closing is the answer. Nothing is written first because there is nothing to save.
            std::process::exit(0);
        }
        _ => return false,
    }
    true
}

/// `SET`, with the options it may carry.
fn set(env: &Env<'_>, cmd: &Command, buf: &[u8], reply: &mut Encoder<'_>) {
    if cmd.len() < 3 {
        arity(cmd, 3, "set", reply);
        return;
    }
    let (key, value) = (arg(cmd, buf, 1), arg(cmd, buf, 2));

    // The overwhelmingly common shape, and the one the benchmark sends. It takes the unconditional path, which does not pay the lookup that a condition or a kept expiry needs.
    if cmd.len() == 3 {
        match env.map.set(key, value, None, None) {
            Ok(_) => reply.simple(b"OK"),
            Err(_) => oom(reply),
        }
        return;
    }

    let mut when = When::Always;
    let mut expiry = Expiry::Never;
    let now = env.map.clock().now();
    let mut n = 3;
    while n < cmd.len() {
        let option = arg(cmd, buf, n);
        let mut step = 1;

        let understood = if option.eq_ignore_ascii_case(b"NX") {
            when = When::Absent;
            true
        } else if option.eq_ignore_ascii_case(b"XX") {
            when = When::Present;
            true
        } else if option.eq_ignore_ascii_case(b"KEEPTTL") {
            expiry = Expiry::Keep;
            true
        } else if let Some((scale, absolute)) = unit(option) {
            step = 2;
            match cmd.arg(n + 1, buf).and_then(integer_of) {
                Some(value) if absolute || value > 0 => {
                    expiry = Expiry::At(stamp(moment(now, value, scale, absolute)));
                    true
                }
                // A relative expiry of nought or less is refused rather than obeyed, because a client that meant to keep a key and typed a negative number should hear about it. Redis refuses the same thing.
                Some(_) => {
                    reply.error("ERR invalid expire time in 'set' command");
                    return;
                }
                None => false,
            }
        } else {
            false
        };

        if !understood {
            reply.error("ERR syntax error");
            return;
        }
        n += step;
    }

    match env.map.set_when(key, value, when, expiry, None) {
        // The condition refused, which is a null rather than an error: the client asked for a write that might not happen and is being told it did not.
        Ok(None) => reply.null(),
        Ok(Some(_)) => reply.simple(b"OK"),
        Err(_) => oom(reply),
    }
}

/// `SETEX` and `PSETEX`, which are `SET` with the expiry in the middle.
fn timed_set(env: &Env<'_>, millis: bool, cmd: &Command, buf: &[u8], reply: &mut Encoder<'_>) {
    let word = if millis { "psetex" } else { "setex" };
    if !arity(cmd, 4, word, reply) {
        return;
    }
    let Some(value) = integer(arg(cmd, buf, 2), reply) else {
        return;
    };
    if value <= 0 {
        reply.error(&format!("ERR invalid expire time in '{word}' command"));
        return;
    }

    let scale = if millis { 1000 } else { 1 };
    let expiry = Expiry::At(stamp(moment(env.map.clock().now(), value, scale, false)));

    match env.map.set_when(
        arg(cmd, buf, 1),
        arg(cmd, buf, 3),
        When::Always,
        expiry,
        None,
    ) {
        Ok(_) => reply.simple(b"OK"),
        Err(_) => oom(reply),
    }
}

/// The four expiry commands, which differ only in unit and in whether the number is a moment or a duration.
fn expire(
    env: &Env<'_>,
    key: &[u8],
    value: i64,
    millis: bool,
    absolute: bool,
    reply: &mut Encoder<'_>,
) {
    let now = env.map.clock().now();
    let at = moment(now, value, if millis { 1000 } else { 1 }, absolute);

    // An expiry already in the past is a deletion, which is what Redis does with one and is the only reading that leaves the key and the clock agreeing.
    if at <= i64::from(now) {
        reply.integer(i64::from(env.map.remove(key)));
        return;
    }
    match env.map.expire(key, Some(stamp(at))) {
        Ok(changed) => reply.integer(i64::from(changed)),
        Err(_) => oom(reply),
    }
}

/// Which unit an expiry option is in, and whether it names a moment rather than a duration.
fn unit(option: &[u8]) -> Option<(i64, bool)> {
    if option.eq_ignore_ascii_case(b"EX") {
        Some((1, false))
    } else if option.eq_ignore_ascii_case(b"PX") {
        Some((1000, false))
    } else if option.eq_ignore_ascii_case(b"EXAT") {
        Some((1, true))
    } else if option.eq_ignore_ascii_case(b"PXAT") {
        Some((1000, true))
    } else {
        None
    }
}

/// The second an expiry lands on, from a number in some unit.
///
/// Milliseconds round up, because a key given nine hundred of them should outlive the second it was set in rather than vanish inside it. A clock in whole seconds cannot do better than that, and rounding the other way would make `PX 900` a deletion.
fn moment(now: u32, value: i64, scale: i64, absolute: bool) -> i64 {
    let secs = value.div_euclid(scale) + i64::from(value.rem_euclid(scale) != 0);
    if absolute {
        secs
    } else {
        i64::from(now) + secs
    }
}

/// That moment as the map spells one.
///
/// Saturating at both ends: a moment before the epoch is nought, which every clock reads as long past, and one after the end of the map's clock is the end of it.
fn stamp(at: i64) -> u32 {
    u32::try_from(at.max(0)).unwrap_or(u32::MAX)
}

/// `INCR` and its neighbours, which differ only in what they add.
fn count(env: &Env<'_>, key: &[u8], by: i64, reply: &mut Encoder<'_>) {
    match env.map.increment(key, by) {
        Ok(value) => reply.integer(value),
        Err(Uncounted::NotANumber) => reply.error("ERR value is not an integer or out of range"),
        Err(Uncounted::OutOfRange) => reply.error("ERR increment or decrement would overflow"),
        Err(Uncounted::Full) => oom(reply),
    }
}

/// The greeting, which is a map in both dialects and differs only in how a map is spelt.
fn hello(dialect: Dialect, reply: &mut Encoder<'_>) {
    let mut out = Vec::new();
    let mut greeting = Encoder::new(&mut out, dialect);
    greeting.map(7);
    greeting.bulk(b"server");
    greeting.bulk(b"rugo");
    greeting.bulk(b"version");
    greeting.bulk(env!("CARGO_PKG_VERSION").as_bytes());
    greeting.bulk(b"proto");
    greeting.integer(if dialect == Dialect::Resp3 { 3 } else { 2 });
    greeting.bulk(b"id");
    greeting.integer(0);
    greeting.bulk(b"mode");
    greeting.bulk(b"standalone");
    greeting.bulk(b"role");
    greeting.bulk(b"master");
    greeting.bulk(b"modules");
    greeting.array(0);
    reply.raw(&out);
}

/// `CONFIG GET`, which answers only about the settings this server actually has.
fn config(env: &Env<'_>, cmd: &Command, buf: &[u8], reply: &mut Encoder<'_>) {
    let known: [(&str, String); 4] = [
        ("maxmemory", env.map.maxmemory().to_string()),
        ("maxmemory-policy", "allkeys-random".to_owned()),
        ("save", String::new()),
        ("appendonly", "no".to_owned()),
    ];

    // Globs are not supported, and `*` is the only one anybody sends. A name that is not here is left out of the answer, which is how Redis reports a setting it does not have.
    let mut found: Vec<&(&str, String)> = Vec::new();
    for n in 2..cmd.len() {
        let wanted = arg(cmd, buf, n);
        for pair in &known {
            if wanted == b"*" || wanted.eq_ignore_ascii_case(pair.0.as_bytes()) {
                found.push(pair);
            }
        }
    }

    reply.map(found.len());
    for (name, value) in found {
        reply.bulk(name.as_bytes());
        reply.bulk(value.as_bytes());
    }
}

/// One `name:value` line of an `INFO` section.
///
/// `INFO` is a CRLF format, and `writeln!` writes the wrong terminator for it, so the line ending is put on separately rather than left to a macro that means something else by a newline.
fn line(out: &mut String, name: &str, value: impl fmt::Display) {
    let _ = write!(out, "{name}:{value}");
    out.push_str("\r\n");
}

/// `INFO`, in the shape every tool that reads it expects.
fn info(env: &Env<'_>, section: &str) -> String {
    let all = section == "all" || section == "default" || section == "everything";
    let want = |name: &str| all || section == name;
    let total = env.stats.total();
    let mut out = String::with_capacity(1024);

    if want("server") {
        out.push_str("# Server\r\n");
        line(&mut out, "redis_version", REDIS_VERSION);
        line(&mut out, "rugo_version", env!("CARGO_PKG_VERSION"));
        line(&mut out, "redis_mode", "standalone");
        line(&mut out, "os", std::env::consts::OS);
        line(&mut out, "arch_bits", usize::BITS);
        line(&mut out, "process_id", std::process::id());
        line(&mut out, "tcp_port", env.config.port.unwrap_or_default());
        line(
            &mut out,
            "uptime_in_seconds",
            env.started.elapsed().as_secs(),
        );
        line(&mut out, "io_threads_active", env.config.threads);
    }

    if want("clients") {
        out.push_str("\r\n# Clients\r\n");
        line(&mut out, "total_connections_received", total.connections);
    }

    if want("memory") {
        // What the map is charged for and what it holds, in the shape Redis publishes them. There is no `used_memory_rss` here on purpose: that is the process's resident set, which this crate does not measure and will not guess at. Whatever reports memory reads it from the operating system.
        out.push_str("\r\n# Memory\r\n");
        line(&mut out, "used_memory", env.map.charged_bytes());
        line(&mut out, "used_memory_slab", env.map.resident_bytes());
        line(&mut out, "used_memory_dataset", env.map.live_bytes());
        line(&mut out, "used_memory_index", env.map.index_bytes());
        line(&mut out, "maxmemory", env.map.maxmemory());
        line(&mut out, "maxmemory_policy", "allkeys-random");
    }

    if want("stats") {
        out.push_str("\r\n# Stats\r\n");
        line(&mut out, "total_commands_processed", total.commands);
        line(&mut out, "keyspace_hits", total.hits);
        line(&mut out, "keyspace_misses", total.misses);
    }

    if want("keyspace") {
        out.push_str("\r\n# Keyspace\r\n");
        let len = env.map.len();
        if len > 0 {
            line(
                &mut out,
                "db0",
                format_args!("keys={len},expires=0,avg_ttl=0"),
            );
        }
    }

    out
}

/// Argument `n`, which the caller has already checked is there.
///
/// Empty for one that is not, rather than a panic, because a command that got its own arity check wrong should answer nonsense rather than take the server down.
#[inline]
fn arg<'a>(cmd: &Command, buf: &'a [u8], n: usize) -> &'a [u8] {
    cmd.arg(n, buf).unwrap_or_default()
}

/// Check the argument count, writing the error if it is wrong.
fn arity(cmd: &Command, want: usize, name: &str, reply: &mut Encoder<'_>) -> bool {
    if cmd.len() == want {
        return true;
    }
    reply.error(&format!(
        "ERR wrong number of arguments for '{name}' command"
    ));
    false
}

/// Read an integer argument, writing the error if it is not one.
fn integer(text: &[u8], reply: &mut Encoder<'_>) -> Option<i64> {
    let value = integer_of(text);
    if value.is_none() {
        reply.error("ERR value is not an integer or out of range");
    }
    value
}

/// Read an integer argument.
fn integer_of(text: &[u8]) -> Option<i64> {
    std::str::from_utf8(text).ok()?.parse().ok()
}

/// The one reply that is the server's fault rather than the client's.
fn oom(reply: &mut Encoder<'_>) {
    reply.error("OOM command not allowed when used memory > 'maxmemory'");
}

/// A command this server does not have.
fn unknown(reply: &mut Encoder<'_>, name: &[u8]) {
    // The name is echoed the way it was sent, quoted and escaped, because a client that mistyped wants to see what it typed and because a name is arbitrary bytes.
    reply.error(&format!(
        "ERR unknown command '{}'",
        String::from_utf8_lossy(name).escape_debug()
    ));
}
