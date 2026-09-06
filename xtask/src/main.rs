//! Repository chores.
//!
//! `cargo xtask scoreboard` writes `SCOREBOARD.md` from the sweep data in `bench/`, and `cargo xtask check` says whether the committed one is what would be written. CI runs the second, so a scoreboard edited by hand fails the build instead of quietly disagreeing with the numbers beside it.
//!
//! Why a generator rather than a document. The claim this project makes is a ratio between measurements, and a ratio typed into a markdown file by whoever ran the sweep is a claim with no evidence attached. Generating it from the committed `output.json` means the number and the data that produced it move together or not at all.

mod ab;
mod load;
mod scoreboard;

use std::process::ExitCode;

/// What the chores are, which is also what `--help` prints.
const USAGE: &str = "\
cargo xtask <task>

Tasks:
  scoreboard    write SCOREBOARD.md from the sweeps in bench/
  check         fail if any generated file is not what the generator would write
  load          drive a running server, to compare a build with another build
  ab            start two builds in turn, drive both, and say whether they differ
";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let task = args.next();

    let result = match task.as_deref() {
        Some("scoreboard") => scoreboard::write(),
        Some("check") => scoreboard::check(),
        Some("load") => load::run(&args.collect::<Vec<_>>()),
        Some("ab") => ab::run(&args.collect::<Vec<_>>()),
        Some("--help" | "-h" | "help") | None => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Some(other) => {
            eprintln!("xtask: no task called {other:?}");
            print!("{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("xtask: {why}");
            ExitCode::FAILURE
        }
    }
}
