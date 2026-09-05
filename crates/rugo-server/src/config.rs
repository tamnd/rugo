//! What the binary was asked to do.
//!
//! Long flags only, in the shape `--flag value`, which is what `cache-bench` writes for every server that takes them. There is no configuration file and there will not be one: a cache server whose behaviour depends on a file that is not in the benchmark's record is a server whose numbers cannot be reproduced.

use std::fmt;

/// The default port, which is Redis's, because every client and every benchmark already knows it.
pub const DEFAULT_PORT: u16 = 6379;

/// How many shards a map gets when nobody says.
pub const DEFAULT_SHARDS: usize = 4096;

/// What the arguments asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Asked {
    /// Serve, with this configuration.
    Serve(Box<Config>),
    /// Print the version and stop.
    ///
    /// `cache-bench` runs `--version` with a five second grace before it gives up, so this has to answer without touching a socket or allocating a map.
    Version,
    /// Print the usage and stop.
    Usage,
}

/// Whether to use `io_uring`, when there is one to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Uring {
    /// Use it if the kernel has it, which is the only setting worth defaulting to.
    #[default]
    Auto,
    /// Use it, and fail to start if it is not there.
    Yes,
    /// Do not, even where it is available. What a comparison between the two backends is run under.
    No,
}

/// How the server was configured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// How many threads serve connections.
    pub threads: usize,
    /// The TCP port, or none if `--no-port` was given.
    pub port: Option<u16>,
    /// The unix socket path, if there is one.
    pub unixsocket: Option<String>,
    /// The byte ceiling, or zero for none.
    pub maxmemory: usize,
    /// How many shards the map has.
    pub shards: usize,
    /// Whether to use `io_uring`.
    pub uring: Uring,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // The machines that produce published numbers have eight and thirty-two cores, and a thread per core is what every server in the comparison defaults to.
            threads: std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
            port: Some(DEFAULT_PORT),
            unixsocket: None,
            maxmemory: 0,
            shards: DEFAULT_SHARDS,
            uring: Uring::Auto,
        }
    }
}

/// Why the arguments were refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bad(String);

impl fmt::Display for Bad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for Bad {}

/// The usage message, which is also the list of what this server understands.
pub const USAGE: &str = "\
rugo, a cache server

Usage: rugo [options]

Options:
  --port <n>            TCP port to listen on (default 6379)
  --no-port             do not listen on TCP at all
  --unixsocket <path>   also listen on a unix socket
  --threads <n>         serving threads (default: one per core)
  --shards <n>          map shards, rounded up to a power of two (default 4096)
  --maxmemory <size>    byte ceiling, as a number or with kb/mb/gb (default: none)
  --uring <auto|yes|no> use io_uring where the kernel has it (default auto)
  --version             print the version and exit
  --help                print this and exit
";

impl Config {
    /// Read the arguments, which must not include the program name.
    ///
    /// # Errors
    ///
    /// [`Bad`] with a message meant for a person, when a flag is unknown or its value is not what the flag takes.
    pub fn parse<I, S>(args: I) -> Result<Asked, Bad>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut config = Self::default();
        let mut args = args.into_iter();
        // Set by `--no-port` so that the order of `--port` and `--no-port` on one command line does not change the answer.
        let mut no_port = false;

        while let Some(arg) = args.next() {
            let arg = arg.as_ref();
            let mut value = || {
                args.next()
                    .map(|v| v.as_ref().to_owned())
                    .ok_or_else(|| Bad(format!("{arg} needs a value")))
            };
            match arg {
                "--version" | "-v" => return Ok(Asked::Version),
                "--help" | "-h" => return Ok(Asked::Usage),
                "--no-port" => no_port = true,
                "--port" | "-p" => config.port = Some(number(&value()?, arg)?),
                "--threads" => config.threads = number::<usize>(&value()?, arg)?.max(1),
                "--shards" => config.shards = number::<usize>(&value()?, arg)?.max(1),
                "--unixsocket" => config.unixsocket = Some(value()?),
                "--maxmemory" => config.maxmemory = bytes(&value()?)?,
                "--uring" => {
                    config.uring = match value()?.as_str() {
                        "auto" => Uring::Auto,
                        "yes" | "true" | "1" => Uring::Yes,
                        "no" | "false" | "0" => Uring::No,
                        other => {
                            return Err(Bad(format!("--uring takes auto, yes or no, not {other}")));
                        }
                    };
                }
                other => return Err(Bad(format!("unknown option {other}"))),
            }
        }

        if no_port {
            config.port = None;
        }
        if config.port.is_none() && config.unixsocket.is_none() {
            return Err(Bad(
                "--no-port with no --unixsocket leaves nothing to listen on".to_owned(),
            ));
        }
        Ok(Asked::Serve(Box::new(config)))
    }
}

/// Read a flag's value as a number.
fn number<T: std::str::FromStr>(text: &str, flag: &str) -> Result<T, Bad> {
    text.parse()
        .map_err(|_| Bad(format!("{flag} takes a number, not {text}")))
}

/// Read a size, which may carry a unit.
///
/// Redis spells these `kb`, `mb`, `gb` and means powers of two by them, and `cache-bench` writes `32gb` at every server it starts, so that is the spelling understood here. The powers of ten Redis also accepts, `k`, `m` and `g`, are understood as well, because a benchmark that meant one and got the other would be comparing two different working sets.
fn bytes(text: &str) -> Result<usize, Bad> {
    let lower = text.trim().to_ascii_lowercase();
    let (digits, scale) = if let Some(rest) = lower.strip_suffix("kb") {
        (rest, 1024)
    } else if let Some(rest) = lower.strip_suffix("mb") {
        (rest, 1024 * 1024)
    } else if let Some(rest) = lower.strip_suffix("gb") {
        (rest, 1024 * 1024 * 1024)
    } else if let Some(rest) = lower.strip_suffix('k') {
        (rest, 1000)
    } else if let Some(rest) = lower.strip_suffix('m') {
        (rest, 1_000_000)
    } else if let Some(rest) = lower.strip_suffix('g') {
        (rest, 1_000_000_000)
    } else if let Some(rest) = lower.strip_suffix('b') {
        (rest, 1)
    } else {
        (lower.as_str(), 1)
    };

    let count: usize = digits
        .trim()
        .parse()
        .map_err(|_| Bad(format!("--maxmemory takes a size, not {text}")))?;
    count.checked_mul(scale).ok_or_else(|| {
        Bad(format!(
            "--maxmemory of {text} does not fit in a machine word"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serve(args: &[&str]) -> Config {
        match Config::parse(args) {
            Ok(Asked::Serve(config)) => *config,
            other => panic!("{args:?} did not ask to serve: {other:?}"),
        }
    }

    #[test]
    fn no_arguments_is_a_server_on_the_usual_port() {
        let config = serve(&[]);
        assert_eq!(config.port, Some(DEFAULT_PORT));
        assert_eq!(config.shards, DEFAULT_SHARDS);
        assert_eq!(config.maxmemory, 0);
        assert!(config.threads >= 1);
    }

    #[test]
    fn the_flags_cache_bench_writes_are_understood() {
        // Exactly what the harness puts on the command line for a RESP server that takes long flags, which is what this list is for.
        let config = serve(&[
            "--port",
            "7777",
            "--threads",
            "8",
            "--maxmemory",
            "32gb",
            "--unixsocket",
            "/tmp/rugo.sock",
        ]);
        assert_eq!(config.port, Some(7777));
        assert_eq!(config.threads, 8);
        assert_eq!(config.maxmemory, 32 * 1024 * 1024 * 1024);
        assert_eq!(config.unixsocket.as_deref(), Some("/tmp/rugo.sock"));
    }

    #[test]
    fn version_and_help_stop_before_anything_else_is_read() {
        // The harness runs `--version` with a five second grace and nothing else on the line, and a server that tried to bind a port first would fail it.
        assert_eq!(Config::parse(["--version"]), Ok(Asked::Version));
        assert_eq!(
            Config::parse(["--version", "--nonsense"]),
            Ok(Asked::Version)
        );
        assert_eq!(Config::parse(["--help"]), Ok(Asked::Usage));
    }

    #[test]
    fn a_unix_socket_alone_is_a_server() {
        let config = serve(&["--no-port", "--unixsocket", "/tmp/rugo.sock"]);
        assert_eq!(config.port, None);
        assert_eq!(config.unixsocket.as_deref(), Some("/tmp/rugo.sock"));
    }

    #[test]
    fn the_order_of_port_and_no_port_does_not_matter() {
        for args in [
            ["--port", "7777", "--no-port", "--unixsocket", "/tmp/s"],
            ["--no-port", "--port", "7777", "--unixsocket", "/tmp/s"],
        ] {
            assert_eq!(serve(&args).port, None, "{args:?} left a port open");
        }
    }

    #[test]
    fn no_port_and_no_socket_is_refused_rather_than_started() {
        // A server that comes up listening on nothing looks healthy and answers nothing, which is the worst way for this to fail.
        assert!(Config::parse(["--no-port"]).is_err());
    }

    #[test]
    fn every_size_spelling_means_what_redis_means_by_it() {
        assert_eq!(bytes("1024"), Ok(1024));
        assert_eq!(bytes("1kb"), Ok(1024));
        assert_eq!(bytes("1KB"), Ok(1024));
        assert_eq!(bytes("32gb"), Ok(32 * 1024 * 1024 * 1024));
        assert_eq!(bytes("1k"), Ok(1000));
        assert_eq!(bytes("2m"), Ok(2_000_000));
        assert_eq!(bytes("100b"), Ok(100));
        assert!(bytes("").is_err());
        assert!(bytes("lots").is_err());
        assert!(bytes("-1").is_err());
    }

    #[test]
    fn an_unknown_flag_is_refused_and_says_which() {
        let Err(bad) = Config::parse(["--appendonly", "yes"]) else {
            panic!("an unknown flag was accepted");
        };
        assert!(
            bad.to_string().contains("--appendonly"),
            "the message did not name the flag: {bad}"
        );
    }

    #[test]
    fn a_flag_with_no_value_is_refused() {
        assert!(Config::parse(["--port"]).is_err());
        assert!(Config::parse(["--maxmemory"]).is_err());
    }

    #[test]
    fn the_uring_switch_takes_three_answers_and_no_others() {
        assert_eq!(serve(&["--uring", "auto"]).uring, Uring::Auto);
        assert_eq!(serve(&["--uring", "yes"]).uring, Uring::Yes);
        assert_eq!(serve(&["--uring", "no"]).uring, Uring::No);
        assert!(Config::parse(["--uring", "maybe"]).is_err());
    }

    #[test]
    fn a_thread_count_of_zero_is_read_as_one() {
        // Nought threads is a server that accepts and never answers. Refusing would be defensible too; reading it as one is what every other server in the comparison does.
        assert_eq!(serve(&["--threads", "0"]).threads, 1);
    }
}
