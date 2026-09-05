//! The `rugo` binary.
//!
//! Reads the arguments, binds what they ask for, and serves. Everything else is in [`rugo_server`].

use std::process::ExitCode;

use rugo_server::{Asked, Config, Server, USAGE};

/// What `--version` prints, which is the name and the version and nothing else.
///
/// `cache-bench` runs this with a five second grace before it decides the binary is broken, so nothing above this line may bind a socket, allocate a map or read a file.
fn version() {
    println!("rugo {}", env!("CARGO_PKG_VERSION"));
}

fn main() -> ExitCode {
    match Config::parse(std::env::args().skip(1)) {
        Ok(Asked::Version) => {
            version();
            ExitCode::SUCCESS
        }
        Ok(Asked::Usage) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Asked::Serve(config)) => serve(*config),
        Err(bad) => {
            eprintln!("rugo: {bad}");
            eprintln!("Try 'rugo --help' for the list of options.");
            ExitCode::FAILURE
        }
    }
}

/// Bind and serve, reporting where.
///
/// The line on standard output is what a person watching a terminal reads and what a script waiting for a server may grep for, so it names every socket that was bound and is written before the first connection can arrive.
fn serve(config: Config) -> ExitCode {
    let threads = config.threads;
    let port = config.port;
    let socket = config.unixsocket.clone();

    let server = match Server::new(config) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("rugo: {error}");
            return ExitCode::FAILURE;
        }
    };

    let mut listening = Vec::new();
    if port.is_some() {
        // Asked rather than assumed, because a configured port of nought is a request for whichever one is free and the number that matters is the one the kernel chose.
        if let Ok(Some(port)) = server.port() {
            listening.push(format!("port {port}"));
        }
    }
    if let Some(path) = &socket {
        listening.push(format!("socket {path}"));
    }
    println!(
        "rugo {} serving on {} with {threads} thread{}",
        env!("CARGO_PKG_VERSION"),
        listening.join(" and "),
        if threads == 1 { "" } else { "s" }
    );

    if let Err(error) = server.run() {
        eprintln!("rugo: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
