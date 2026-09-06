//! `cargo xtask ab`, which runs one build against another and says whether the box could tell them apart.
//!
//! # Why this is a task rather than a shell script
//!
//! Comparing two builds is starting a server, driving it, killing it, starting the other one, driving that, and doing the whole thing again several times because one round of anything on a shared box is a coin toss. Written as a shell loop each time it is needed, that is a new set of mistakes each time: a server left running, a kill by pattern that matches the wrong process, a fill charged to the timed part, or a pair of numbers read as a result when they are a round apart in a box whose load moved.
//!
//! # What it is careful about
//!
//! The two servers are alternated rather than run in blocks, so a box that gets busier over the sitting spends that on both of them. Every server it starts is one it spawned itself and every server it stops is stopped by that handle, so nothing here can reach a process it did not create.
//!
//! Most of all it reports the generator's own processor time an operation beside the server's. A difference in the server's cost means something only when the generator's cost stayed where it was; when the two move together, what moved was the machine, and the verdict says so rather than reporting the difference as a result.
//!
//! # What it does not do
//!
//! rugo against rugo, one box, one sitting, exactly as `cargo xtask load` says. Nothing from here goes in `SCOREBOARD.md` or into a sentence about another server.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::load::{self, Run};

/// What `cargo xtask ab --help` prints.
pub(crate) const USAGE: &str = "\
cargo xtask ab --a <command> --b <command> [options]

Options:
  --a <command>       how to start the first server, as one quoted command line
  --b <command>       how to start the second, the same way
  --rounds <n>        how many times to run each, alternating (default 6)
  --socket <path>     the unix socket both are told to listen on (default /tmp/rugo-ab.sock)
  --settle <ms>       how long to leave the box alone between runs (default 1000)

Every other flag is passed to `cargo xtask load`, except --socket and --pid, which this
sets itself. A command may have a prefix on it, so `--a 'taskset -c 0-1 ./target/release/rugo
--uring no'` pins the server, and running the whole task under taskset pins the generator.
";

/// How long to wait for a server to open its socket before giving up on it.
const PATIENCE: Duration = Duration::from_secs(10);

/// How often to look for the socket while waiting for it.
const GLANCE: Duration = Duration::from_millis(10);

/// The comparison to run.
struct Ab {
    /// The first server's command line, split into a program and its arguments.
    a: Vec<String>,
    /// The second's.
    b: Vec<String>,
    /// How many times each is run.
    rounds: usize,
    /// The socket both of them listen on, one at a time.
    socket: PathBuf,
    /// How long to leave the box alone between runs.
    settle: Duration,
    /// The flags to hand to the load generator, before the socket and the process id are added.
    load: Vec<String>,
}

/// Everything one side of the comparison measured.
struct Side {
    /// Which side it is, for the report.
    name: String,
    /// Server processor time an operation, in microseconds, one a round.
    server: Vec<f64>,
    /// The generator's own, on the same terms.
    client: Vec<f64>,
    /// Operations a second, one a round.
    opsec: Vec<f64>,
}

/// A server this task started, which it stops.
///
/// Stopping is in `Drop` rather than at the end of the run so that a load that fails part way through still leaves nothing behind. The only process it can ever signal is the child in the handle it holds.
struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Run the comparison described by `args`.
pub(crate) fn run(args: &[String]) -> Result<(), String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print!("{USAGE}");
        return Ok(());
    }
    let ab = parse(args)?;

    let mut a = Side::new("a");
    let mut b = Side::new("b");
    for round in 1..=ab.rounds {
        a.took(once(&ab, &ab.a, round, "a")?);
        b.took(once(&ab, &ab.b, round, "b")?);
    }

    println!();
    println!("{}", a.line());
    println!("{}", b.line());
    println!();
    println!("{}", verdict(&a, &b));
    Ok(())
}

/// Read the flags this task knows, and keep the rest for the load generator.
fn parse(args: &[String]) -> Result<Ab, String> {
    let mut a: Option<String> = None;
    let mut b: Option<String> = None;
    let mut rounds = 6_usize;
    let mut socket = PathBuf::from("/tmp/rugo-ab.sock");
    let mut settle = 1000_u64;
    let mut load = Vec::new();

    let mut rest = args.iter();
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--a" => a = Some(after(flag, &mut rest)?),
            "--b" => b = Some(after(flag, &mut rest)?),
            "--rounds" => rounds = number(&after(flag, &mut rest)?)?,
            "--socket" => socket = PathBuf::from(after(flag, &mut rest)?),
            "--settle" => settle = number(&after(flag, &mut rest)?)?,
            // Not a flag of this task, so it is one of the load generator's, and it goes on to be checked there rather than being guessed at here.
            other => load.push(other.to_owned()),
        }
    }

    let (Some(a), Some(b)) = (a, b) else {
        return Err(
            "both --a and --b are needed, because this compares two of something".to_owned(),
        );
    };
    if rounds == 0 {
        return Err("a comparison of nought rounds compares nothing".to_owned());
    }

    Ok(Ab {
        a: words(&a),
        b: words(&b),
        rounds,
        socket,
        settle: Duration::from_millis(settle),
        load,
    })
}

/// Whatever came after `flag`.
fn after(flag: &str, rest: &mut std::slice::Iter<'_, String>) -> Result<String, String> {
    rest.next()
        .cloned()
        .ok_or_else(|| format!("{flag} wants a value after it"))
}

/// One flag's value as a number.
fn number<T: std::str::FromStr>(text: &str) -> Result<T, String> {
    text.parse()
        .map_err(|_| format!("{text} is not a number this flag can take"))
}

/// A command line split into a program and its arguments.
///
/// Whitespace and nothing else. A path with a space in it is not supported, and a server binary living behind one is a smaller problem than a quoting language nobody asked for.
fn words(line: &str) -> Vec<String> {
    line.split_whitespace().map(str::to_owned).collect()
}

/// Start one server, drive it once, and stop it.
fn once(ab: &Ab, argv: &[String], round: usize, which: &str) -> Result<Run, String> {
    let server = start(ab, argv)?;
    // The process id of what was spawned. A `taskset` or a `nice` in front of the server execs into it rather than forking, so this is the server's own id even when the command line has a prefix on it.
    let pid = server.0.id();

    let mut flags = ab.load.clone();
    flags.push("--socket".to_owned());
    flags.push(ab.socket.display().to_string());
    flags.push("--pid".to_owned());
    flags.push(pid.to_string());
    let load = load::parse(&flags)?;

    let run = load::once(&load);
    drop(server);
    sleep(ab.settle);
    let run = run?;

    println!(
        "round {round} {which}: {:.0} operations a second, {} server, {} client",
        rate(run.ops, run.seconds),
        micros(run.server),
        micros(run.client),
    );
    Ok(run)
}

/// Start a server and wait for it to be listening.
fn start(ab: &Ab, argv: &[String]) -> Result<Server, String> {
    let Some((program, flags)) = argv.split_first() else {
        return Err("a server command line with nothing in it starts nothing".to_owned());
    };

    // The socket a killed server left behind. The server removes one of its own before binding, and this is here so that waiting for the socket to appear is not answered by the last run's.
    let _ = std::fs::remove_file(&ab.socket);

    let mut child = Command::new(program)
        .args(flags)
        .arg("--no-port")
        .arg("--unixsocket")
        .arg(&ab.socket)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("{program} did not start: {error}"))?;

    let waited = Instant::now();
    while waited.elapsed() < PATIENCE {
        if ab.socket.exists() {
            return Ok(Server(child));
        }
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!(
                "{program} stopped before it listened, with {status}"
            ));
        }
        sleep(GLANCE);
    }

    let _ = child.kill();
    let _ = child.wait();
    Err(format!(
        "{program} did not open {} within {} seconds",
        ab.socket.display(),
        PATIENCE.as_secs()
    ))
}

/// Operations a second, or nought where no time passed.
#[expect(
    clippy::cast_precision_loss,
    reason = "an operation count large enough to lose a digit is more than a round can do"
)]
fn rate(ops: usize, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        return 0.0;
    }
    ops as f64 / seconds
}

/// A processor reading, or a dash where there was none to take.
fn micros(each: Option<f64>) -> String {
    each.map_or_else(|| "-".to_owned(), |each| format!("{each:.3}us"))
}

impl Side {
    /// A side with nothing measured yet.
    fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            server: Vec::new(),
            client: Vec::new(),
            opsec: Vec::new(),
        }
    }

    /// Keep what one round measured.
    fn took(&mut self, run: Run) {
        if let Some(server) = run.server {
            self.server.push(server);
        }
        if let Some(client) = run.client {
            self.client.push(client);
        }
        self.opsec.push(rate(run.ops, run.seconds));
    }

    /// One line of the report.
    fn line(&self) -> String {
        format!(
            "{}: server {} least {}, client {}, {:.0} operations a second",
            self.name,
            micros(median(&self.server)),
            micros(least(&self.server)),
            micros(median(&self.client)),
            median(&self.opsec).unwrap_or(0.0),
        )
    }
}

/// The middle of a set of readings, or nothing if there are none.
fn median(of: &[f64]) -> Option<f64> {
    at(of, 0.5)
}

/// The smallest of a set of readings, which is the round where the box interfered least.
fn least(of: &[f64]) -> Option<f64> {
    of.iter().copied().fold(None, |best: Option<f64>, one| {
        Some(best.map_or(one, |best| best.min(one)))
    })
}

/// The reading `part` of the way through the sorted set, taking the lower of the two when it falls between.
fn at(of: &[f64], part: f64) -> Option<f64> {
    if of.is_empty() {
        return None;
    }
    let mut sorted = of.to_vec();
    sorted.sort_by(f64::total_cmp);
    // The index is a position in a list this function itself sorted, so it is in range by construction and the count is far under what an `f64` counts exactly.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a position in a list of at most a few hundred readings"
    )]
    let index = ((sorted.len() - 1) as f64 * part) as usize;
    sorted.get(index).copied()
}

/// How much a set of readings moves about, as a fraction of its middle.
///
/// The middle half of them rather than all of them, so that one round where something else on the box woke up does not become the whole answer.
fn spread(of: &[f64]) -> Option<f64> {
    let low = at(of, 0.25)?;
    let high = at(of, 0.75)?;
    let middle = median(of)?;
    if middle <= 0.0 {
        return None;
    }
    Some((high - low) / middle)
}

/// How much bigger the larger of two readings is, as a fraction of the smaller.
fn gap(a: f64, b: f64) -> Option<f64> {
    let low = a.min(b);
    let high = a.max(b);
    if low <= 0.0 {
        return None;
    }
    Some((high - low) / low)
}

/// What the two sides say, if they say anything.
///
/// Three ways a comparison can end. It can have no processor readings at all, which is every machine without a `/proc`, and then the only thing to report is throughput and the fact that throughput on its own cannot tell a faster server from a quieter box. It can have a difference smaller than the spread within either side, which is a result of nothing. Or it can have a difference that survives that, in which case the last question is whether the generator moved too: the generator is the same program in both halves, so its cost an operation moving with the server's is the box changing under both of them rather than either build being cheaper.
fn verdict(a: &Side, b: &Side) -> String {
    let (Some(one), Some(two)) = (median(&a.server), median(&b.server)) else {
        return format!(
            "no processor readings, so this is throughput alone: {:.0} against {:.0} operations a second, which says as much about how busy the box was as about either build",
            median(&a.opsec).unwrap_or(0.0),
            median(&b.opsec).unwrap_or(0.0),
        );
    };

    let Some(difference) = gap(one, two) else {
        return "a reading of nought processor time is not a reading".to_owned();
    };
    let noise = spread(&a.server)
        .unwrap_or(0.0)
        .max(spread(&b.server).unwrap_or(0.0));

    if difference <= noise {
        return format!(
            "inside the noise: {:.1} per cent between them, against {:.1} per cent of spread within a side, so this box cannot tell these two builds apart",
            difference * 100.0,
            noise * 100.0,
        );
    }

    let drift = match (median(&a.client), median(&b.client)) {
        (Some(one), Some(two)) => gap(one, two).unwrap_or(0.0),
        _ => 0.0,
    };
    if drift >= difference / 2.0 {
        return format!(
            "the generator moved with the server, by {:.1} per cent against the server's {:.1}, so what this measured is the box rather than either build",
            drift * 100.0,
            difference * 100.0,
        );
    }

    let (faster, slower) = if one < two {
        (&a.name, &b.name)
    } else {
        (&b.name, &a.name)
    };
    format!(
        "{faster} costs {:.1} per cent less processor time an operation than {slower}, against {:.1} per cent of spread within a side and {:.1} per cent of drift in the generator",
        difference * 100.0,
        noise * 100.0,
        drift * 100.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn side(name: &str, server: &[f64], client: &[f64]) -> Side {
        Side {
            name: name.to_owned(),
            server: server.to_vec(),
            client: client.to_vec(),
            opsec: vec![100.0; server.len()],
        }
    }

    #[test]
    fn a_command_line_is_split_on_spaces_so_a_prefix_can_be_put_in_front_of_it() {
        assert_eq!(
            words("taskset -c 0-1 ./target/release/rugo --uring no"),
            vec![
                "taskset",
                "-c",
                "0-1",
                "./target/release/rugo",
                "--uring",
                "no"
            ]
        );
    }

    #[test]
    fn a_flag_this_task_does_not_know_is_kept_for_the_load_generator() {
        let args: Vec<String> = ["--a", "x", "--b", "y", "--pipeline", "25"]
            .iter()
            .map(|arg| (*arg).to_owned())
            .collect();
        let ab = parse(&args).unwrap();
        assert_eq!(ab.load, vec!["--pipeline".to_owned(), "25".to_owned()]);
    }

    #[test]
    fn one_side_alone_is_not_a_comparison() {
        let args: Vec<String> = ["--a", "x"].iter().map(|arg| (*arg).to_owned()).collect();
        assert!(parse(&args).is_err());
    }

    #[test]
    fn the_middle_of_an_even_set_is_the_lower_of_the_two_it_falls_between() {
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.0));
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[]), None);
    }

    #[test]
    fn a_difference_smaller_than_the_spread_within_a_side_is_not_a_difference() {
        let a = side("a", &[1.0, 1.4, 1.8, 2.2], &[1.0, 1.0, 1.0, 1.0]);
        let b = side("b", &[1.1, 1.5, 1.9, 2.3], &[1.0, 1.0, 1.0, 1.0]);
        assert!(
            verdict(&a, &b).starts_with("inside the noise"),
            "{}",
            verdict(&a, &b)
        );
    }

    #[test]
    fn a_difference_the_generator_shared_is_reported_as_the_box() {
        let a = side("a", &[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0, 1.0]);
        let b = side("b", &[2.0, 2.0, 2.0, 2.0], &[2.0, 2.0, 2.0, 2.0]);
        assert!(
            verdict(&a, &b).starts_with("the generator moved with the server"),
            "{}",
            verdict(&a, &b)
        );
    }

    #[test]
    fn a_difference_the_generator_did_not_share_names_the_cheaper_side() {
        let a = side("a", &[1.0, 1.0, 1.02, 1.0], &[1.0, 1.0, 1.0, 1.0]);
        let b = side("b", &[2.0, 2.0, 2.02, 2.0], &[1.0, 1.0, 1.0, 1.0]);
        assert!(
            verdict(&a, &b).starts_with("a costs "),
            "{}",
            verdict(&a, &b)
        );
    }

    #[test]
    fn a_machine_with_no_proc_gets_throughput_and_a_warning_rather_than_a_verdict() {
        let a = side("a", &[], &[]);
        let b = side("b", &[], &[]);
        assert!(
            verdict(&a, &b).starts_with("no processor readings"),
            "{}",
            verdict(&a, &b)
        );
    }
}
