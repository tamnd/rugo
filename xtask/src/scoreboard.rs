//! `SCOREBOARD.md`, from the sweeps in `bench/`.
//!
//! # What a sweep directory holds
//!
//! ```text
//! bench/<id>/sweep.json    which host, which profile, which day
//! bench/<id>/output.json   cache-bench's combined result file
//! bench/<id>/memory.json   what cb-mem measured, when it has run
//! ```
//!
//! `output.json` comes from another repository on another release cycle, so it is read as values rather than into structs: a field added to `cache-bench` should not stop the scoreboard here from rendering.
//!
//! # Which runs count
//!
//! Only the average of the trimmed set, and only the half of the matrix with no counter attached. The best run of five is the run where the machine happened to be quietest and is not what anybody gets; attaching `perf` costs throughput, so the throughput half of the matrix is the half where it was not attached. Both of those are the harness's own conventions, and the scoreboard follows them rather than inventing a third.
//!
//! # What the ratios mean
//!
//! Throughput is rugo over the rival, so above one is faster and two is the gate. Memory is the rival over rugo, so above one is smaller and two is the gate. They are written the same way round on purpose: in both columns, bigger is better for rugo, and two is the line.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// Where the sweeps live.
const BENCH: &str = "bench";

/// The document this writes.
const OUT: &str = "SCOREBOARD.md";

/// The ratio a gate is set at, on both halves.
const GATE: f64 = 2.0;

/// The pipeline depths a row reports, which are the four the harness sweeps.
const DEPTHS: [u64; 4] = [1, 10, 25, 50];

/// One measured run, reduced to the four things a scoreboard row needs.
#[derive(Debug)]
struct Point {
    /// Which server.
    cache: String,
    /// The memtier pipeline depth.
    pipeline: u64,
    /// Sets and gets added together, which is what the harness reports as the run's throughput.
    opsec: f64,
}

/// What one server cost in memory for one working set.
#[derive(Debug)]
struct Memory {
    /// Which server.
    cache: String,
    /// How many keys were live when the high-water mark was read.
    entries: f64,
    /// The peak resident set of the server's whole process group, in bytes.
    peak_rss: f64,
    /// Keys and values added up, with nothing the server added.
    payload: f64,
}

impl Memory {
    /// Resident bytes for every key held, which is the number a person renting a machine pays for.
    fn total_per_entry(&self) -> f64 {
        self.peak_rss / self.entries
    }

    /// Resident bytes a key costs beyond the key and the value themselves.
    ///
    /// The number the design is actually about. It can be negative in principle, if a server compresses, and none of the eight do.
    fn overhead_per_entry(&self) -> f64 {
        (self.peak_rss - self.payload) / self.entries
    }
}

/// One published sweep.
#[derive(Debug)]
struct Sweep {
    /// The directory name, which is also how a row refers to it.
    id: String,
    /// Which machine.
    host: String,
    /// Which `profiles.toml` profile it was run under.
    profile: String,
    /// The day it finished.
    date: String,
    /// Anything a reader needs in order not to be misled by the numbers.
    note: String,
    /// Every counted run.
    points: Vec<Point>,
    /// What `cb-mem` found, if it has run here.
    memory: Vec<Memory>,
}

/// Write `SCOREBOARD.md`.
///
/// # Errors
///
/// If a sweep directory cannot be read or holds something that is not a sweep.
pub(crate) fn write() -> Result<(), String> {
    let text = render(&read_sweeps(Path::new(BENCH))?);
    std::fs::write(OUT, &text).map_err(|why| format!("could not write {OUT}: {why}"))?;
    println!("wrote {OUT} ({} bytes)", text.len());
    Ok(())
}

/// Fail if the committed `SCOREBOARD.md` is not what [`write()`] would produce.
///
/// # Errors
///
/// If it differs, or if it is missing, or if the sweeps cannot be read.
pub(crate) fn check() -> Result<(), String> {
    let wanted = render(&read_sweeps(Path::new(BENCH))?);
    let found = std::fs::read_to_string(OUT)
        .map_err(|why| format!("could not read {OUT}: {why}. Run `cargo xtask scoreboard`."))?;
    if found == wanted {
        println!("{OUT} is what the generator would write");
        return Ok(());
    }
    // The first line that differs, because a whole diff of a generated document is mostly noise and the first disagreement is where somebody typed.
    let at = found
        .lines()
        .zip(wanted.lines())
        .position(|(a, b)| a != b)
        .map_or_else(|| "the end".to_owned(), |n| format!("line {}", n + 1));
    Err(format!(
        "{OUT} is not what the generator would write; they first differ at {at}. It is generated from bench/, so edit the data or the generator, not the document. Run `cargo xtask scoreboard`."
    ))
}

/// Every sweep under `dir`, in directory-name order.
fn read_sweeps(dir: &Path) -> Result<Vec<Sweep>, String> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    // Sorted, because a directory listing is not ordered and a generated document has to be a function of its inputs and nothing else.
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|why| format!("could not read {}: {why}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();

    dirs.iter().map(|path| read_sweep(path)).collect()
}

/// One sweep directory.
fn read_sweep(dir: &Path) -> Result<Sweep, String> {
    let id = dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{} has a name that is not text", dir.display()))?
        .to_owned();

    let meta = read_json(&dir.join("sweep.json"))?;
    let text = |field: &str| {
        meta.get(field)
            .and_then(Value::as_str)
            .unwrap_or("unrecorded")
            .to_owned()
    };

    Ok(Sweep {
        id,
        host: text("host"),
        profile: text("profile"),
        date: text("date"),
        note: text("note"),
        points: read_points(&dir.join("output.json"))?,
        memory: read_memory(&dir.join("memory.json"))?,
    })
}

/// A JSON file that has to be there.
fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|why| format!("could not read {}: {why}", path.display()))?;
    serde_json::from_str(&text).map_err(|why| format!("{} is not JSON: {why}", path.display()))
}

/// The counted runs out of a `cache-bench` `output.json`.
fn read_points(path: &Path) -> Result<Vec<Point>, String> {
    let file = read_json(path)?;
    points_of(&file).ok_or_else(|| format!("{} is not an array of runs", path.display()))
}

/// The counted runs out of a parsed `output.json`, or `None` if it is not one.
///
/// `output.json` is a bare array: `Output` is `#[serde(transparent)]` over its entries, so there is no wrapping object to reach through.
fn points_of(file: &Value) -> Option<Vec<Point>> {
    let entries = file.as_array()?;

    let mut points = Vec::new();
    for entry in entries {
        let name = entry
            .get("file")
            .and_then(Value::as_str)
            .unwrap_or_default();
        // The average of the trimmed set, with no counter attached. Everything else in the file is either a single run or a run that was paying for perf.
        if !name.contains("-perf_no-") || !name.ends_with("-run_average.json") {
            continue;
        }
        let Some(info) = entry.pointer("/data/info") else {
            continue;
        };
        let (Some(cache), Some(pipeline)) = (
            info.get("cache").and_then(Value::as_str),
            info.get("pipeline").and_then(Value::as_u64),
        ) else {
            continue;
        };
        let ops = |half: &str| {
            entry
                .pointer(&format!("/data/{half}/opsec"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
        };
        points.push(Point {
            cache: cache.to_owned(),
            pipeline,
            // Sets and gets together, which is what the run did rather than what half of it did.
            opsec: ops("sets") + ops("gets"),
        });
    }
    Some(points)
}

/// What `cb-mem` measured, if it has run in this sweep.
fn read_memory(path: &Path) -> Result<Vec<Memory>, String> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let file = read_json(path)?;
    let rows = file
        .as_array()
        .ok_or_else(|| format!("{} is not an array of measurements", path.display()))?;

    let mut out = Vec::new();
    for row in rows {
        let number = |field: &str| row.get(field).and_then(Value::as_f64).unwrap_or(0.0);
        let Some(cache) = row.get("cache").and_then(Value::as_str) else {
            continue;
        };
        let entries = number("entries");
        if entries <= 0.0 {
            return Err(format!(
                "{} says {cache} held no keys, so a bytes-per-entry number cannot be had from it",
                path.display()
            ));
        }
        out.push(Memory {
            cache: cache.to_owned(),
            entries,
            peak_rss: number("peak_rss"),
            payload: number("payload_bytes"),
        });
    }
    Ok(out)
}

/// The best throughput each server reached, and its best at each depth.
fn peaks(points: &[Point]) -> BTreeMap<&str, (f64, BTreeMap<u64, f64>)> {
    let mut by: BTreeMap<&str, (f64, BTreeMap<u64, f64>)> = BTreeMap::new();
    for point in points {
        let (peak, depths) = by.entry(point.cache.as_str()).or_default();
        // The best thread count at this depth, because every server is given the thread count it does best with rather than one chosen for it.
        *peak = peak.max(point.opsec);
        let at = depths.entry(point.pipeline).or_default();
        *at = at.max(point.opsec);
    }
    by
}

/// A ratio, or a dash where there is nothing to divide.
fn ratio(ours: Option<f64>, theirs: Option<f64>) -> String {
    match (ours, theirs) {
        (Some(ours), Some(theirs)) if theirs > 0.0 && ours > 0.0 => {
            format!("{:.2}x", ours / theirs)
        }
        _ => "—".to_owned(),
    }
}

/// Whether a ratio cleared the gate, said in a word rather than a mark, because a mark in a table is a thing people read past.
fn verdict(ours: Option<f64>, theirs: Option<f64>) -> &'static str {
    match (ours, theirs) {
        (Some(ours), Some(theirs)) if theirs > 0.0 => {
            if ours / theirs >= GATE {
                "pass"
            } else {
                "not yet"
            }
        }
        _ => "—",
    }
}

/// The whole document.
fn render(sweeps: &[Sweep]) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str(HEAD);

    if sweeps.is_empty() {
        out.push_str(NOTHING_YET);
        out.push_str(FOOT);
        return out;
    }

    for sweep in sweeps {
        let _ = writeln!(out, "## {}\n", sweep.id);
        let _ = writeln!(
            out,
            "Host `{}`, profile `{}`, finished {}.\n",
            sweep.host, sweep.profile, sweep.date
        );
        if sweep.note != "unrecorded" {
            let _ = writeln!(out, "{}\n", sweep.note);
        }
        throughput(&mut out, sweep);
        memory(&mut out, sweep);
    }

    out.push_str(FOOT);
    out
}

/// The throughput table for one sweep.
fn throughput(out: &mut String, sweep: &Sweep) {
    let by = peaks(&sweep.points);
    let Some((ours, our_depths)) = by.get("rugo") else {
        out.push_str(
            "No rugo runs in this sweep, so there is nothing to take a ratio against.\n\n",
        );
        return;
    };

    out.push_str("### Throughput\n\n");
    out.push_str("Ratios are rugo over the rival, so above one is faster and the gate is two.\n\n");
    out.push_str("| rival | peak ops/sec | rugo/rival at peak |");
    for depth in DEPTHS {
        let _ = write!(out, " pipeline {depth} |");
    }
    out.push_str(" gate |\n|---|---:|---:|");
    for _ in DEPTHS {
        out.push_str("---:|");
    }
    out.push_str("---|\n");

    for (cache, (peak, depths)) in &by {
        if *cache == "rugo" {
            continue;
        }
        let _ = write!(
            out,
            "| {cache} | {peak:.0} | {} |",
            ratio(Some(*ours), Some(*peak))
        );
        for depth in DEPTHS {
            let _ = write!(
                out,
                " {} |",
                ratio(our_depths.get(&depth).copied(), depths.get(&depth).copied())
            );
        }
        let _ = writeln!(out, " {} |", verdict(Some(*ours), Some(*peak)));
    }
    let _ = writeln!(out, "\nrugo's own peak was {ours:.0} ops/sec.\n");
}

/// The memory table for one sweep.
fn memory(out: &mut String, sweep: &Sweep) {
    if sweep.memory.is_empty() {
        out.push_str("### Memory\n\nNot measured in this sweep. `cache-bench mem` had not landed when it ran, so the memory half of the gate has no row here rather than a row that guesses.\n\n");
        return;
    }
    let Some(ours) = sweep.memory.iter().find(|row| row.cache == "rugo") else {
        out.push_str("### Memory\n\nNo rugo measurement in this sweep.\n\n");
        return;
    };

    out.push_str("### Memory\n\n");
    out.push_str("Two different claims, kept in two different columns. Total is the whole resident set divided by the keys in it, which is what a machine has to have. Overhead is what is left after the keys and values themselves, which is what the design is about. At a hundred-odd bytes of payload a key, no index can halve the first number, and the second is where the difference actually lives.\n\n");
    out.push_str("Ratios are the rival over rugo, so above one means rugo is smaller and the gate is two.\n\n");
    out.push_str("| rival | total B/entry | rival/rugo total | overhead B/entry | rival/rugo overhead | gate on overhead |\n|---|---:|---:|---:|---:|---|\n");

    let mut rows: Vec<&Memory> = sweep.memory.iter().collect();
    rows.sort_by(|a, b| a.cache.cmp(&b.cache));
    for row in rows {
        if row.cache == "rugo" {
            continue;
        }
        let _ = writeln!(
            out,
            "| {} | {:.1} | {} | {:.1} | {} | {} |",
            row.cache,
            row.total_per_entry(),
            ratio(Some(row.total_per_entry()), Some(ours.total_per_entry())),
            row.overhead_per_entry(),
            ratio(
                Some(row.overhead_per_entry()),
                Some(ours.overhead_per_entry())
            ),
            verdict(
                Some(row.overhead_per_entry()),
                Some(ours.overhead_per_entry())
            ),
        );
    }
    let _ = writeln!(
        out,
        "\nrugo held {:.0} keys in {:.1} total bytes each, of which {:.1} was overhead.\n",
        ours.entries,
        ours.total_per_entry(),
        ours.overhead_per_entry()
    );
}

/// The preamble, which says what the numbers are before anybody reads one.
const HEAD: &str = "\
<!-- Generated by `cargo xtask scoreboard`. Do not edit; edit bench/ or xtask/src/scoreboard.rs. -->

# Scoreboard

rugo against every server `cache-bench` measures, on the gate this project set itself: twice the throughput of any rival, and half the memory per entry.

Every number here is generated from a committed `output.json` that ships beside it in `bench/`, so any row can be recomputed from the data rather than taken on trust. Nothing here was measured on a laptop; each sweep names the host and the profile it ran under, and two numbers from two machines are not comparable.

The pogocache row is expected to read `not yet` for a long time, and it is published anyway. A gate that is only published once it passes is not a gate.

";

/// What the document says before the first sweep lands.
const NOTHING_YET: &str = "\
## No sweep yet

`bench/` is empty, so there is nothing to report. rugo builds, passes its tests and serves, and that is a different claim from being faster than anything.

rugo is `cache-bench`'s eighth subject and its memory metric exists, so what is left before the first sweep is provisioning the two bench hosts and running it. Until then this document has no numbers in it, which is the honest state rather than an empty table that looks like a result.

";

/// The closing note, which is about what the numbers do not say.
const FOOT: &str = "\
## What these numbers are not

They are not a claim about any workload but this one. `cache-bench` drives `memtier_benchmark` at a fixed key and value size through a fixed pipeline, and a cache server tuned for that is tuned for that.

They are not comparable between sweeps on different hosts. Core count, memory bandwidth and whether a PMU was attached all move the numbers more than most of the optimisations do.

The memory rows report rather than judge. Garnet preallocates its index and Dragonfly preallocates per proactor, so a resident-set number for either says as much about what it reserved as about what it used.
";

#[cfg(test)]
mod tests {
    use super::*;

    fn point(cache: &str, pipeline: u64, opsec: f64) -> Point {
        Point {
            cache: cache.to_owned(),
            pipeline,
            opsec,
        }
    }

    fn sweep(points: Vec<Point>, memory: Vec<Memory>) -> Sweep {
        Sweep {
            id: "2026-01-01-epyc8-server3".to_owned(),
            host: "server3".to_owned(),
            profile: "epyc8".to_owned(),
            date: "2026-01-01".to_owned(),
            note: "unrecorded".to_owned(),
            points,
            memory,
        }
    }

    #[test]
    fn an_empty_bench_directory_says_so_rather_than_drawing_an_empty_table() {
        let text = render(&[]);
        assert!(text.contains("No sweep yet"));
        assert!(!text.contains("| rival |"));
    }

    #[test]
    fn a_ratio_is_ours_over_theirs_and_the_gate_is_two() {
        assert_eq!(ratio(Some(200.0), Some(100.0)), "2.00x");
        assert_eq!(verdict(Some(200.0), Some(100.0)), "pass");
        assert_eq!(verdict(Some(199.0), Some(100.0)), "not yet");
        // Nothing to divide is a dash and not a nought, because a nought in a ratio column reads as a measurement.
        assert_eq!(ratio(Some(1.0), None), "—");
        assert_eq!(ratio(Some(1.0), Some(0.0)), "—");
    }

    #[test]
    fn a_server_is_taken_at_its_best_thread_count() {
        // Four runs of one server at one depth, differing only in threads, which the file does not distinguish here. The best is what counts, because every server is given the shape it does best with.
        let points = [
            point("rugo", 10, 100.0),
            point("rugo", 10, 400.0),
            point("rugo", 1, 50.0),
        ];
        let by = peaks(&points);
        let (peak, depths) = &by["rugo"];
        assert!((*peak - 400.0).abs() < f64::EPSILON);
        assert!((depths[&10] - 400.0).abs() < f64::EPSILON);
        assert!((depths[&1] - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn the_throughput_table_names_every_rival_and_no_rugo_row() {
        let text = render(&[sweep(
            vec![
                point("rugo", 1, 200.0),
                point("rugo", 10, 2000.0),
                point("pogocache", 1, 150.0),
                point("pogocache", 10, 1800.0),
                point("redis", 1, 50.0),
            ],
            Vec::new(),
        )]);
        assert!(text.contains("| pogocache |"));
        assert!(text.contains("| redis |"));
        assert!(!text.contains("| rugo |"), "rugo was compared with itself");
        // 2000 over 1800 is short of the gate and 200 over 50 clears it, and both have to be visible.
        assert!(text.contains("1.11x"));
        assert!(text.contains("4.00x"));
        assert!(text.contains("not yet"));
        assert!(text.contains("pass"));
    }

    #[test]
    fn a_sweep_with_no_memory_file_says_so_rather_than_reporting_nought() {
        let text = render(&[sweep(vec![point("rugo", 1, 1.0)], Vec::new())]);
        assert!(text.contains("Not measured in this sweep"));
        assert!(!text.contains("0.0 |"));
    }

    #[test]
    fn total_and_overhead_are_two_different_claims() {
        // A hundred bytes of payload a key, rugo paying twenty bytes over it and the rival paying sixty. The overhead ratio is three; the total ratio is a long way under two, which is the whole reason both columns exist.
        let ours = Memory {
            cache: "rugo".to_owned(),
            entries: 1000.0,
            peak_rss: 120_000.0,
            payload: 100_000.0,
        };
        let theirs = Memory {
            cache: "redis".to_owned(),
            entries: 1000.0,
            peak_rss: 160_000.0,
            payload: 100_000.0,
        };
        assert!((ours.overhead_per_entry() - 20.0).abs() < f64::EPSILON);
        assert!((theirs.overhead_per_entry() - 60.0).abs() < f64::EPSILON);

        let text = render(&[sweep(vec![point("rugo", 1, 1.0)], vec![ours, theirs])]);
        assert!(text.contains("3.00x"), "the overhead ratio is missing");
        assert!(text.contains("1.33x"), "the total ratio is missing");
        // The gate column is about overhead, so a rival that is three times heavier per key passes it even though the total is nowhere near two.
        assert!(text.contains("pass"));
    }

    #[test]
    fn only_the_average_of_the_runs_without_a_counter_is_counted() {
        // The shape cache-bench writes: a bare array, since `Output` is transparent over its entries, and a filename that says which of the five slots and whether perf was attached.
        let run = |file: &str, cache: &str, pipeline: u32, sets: f64, gets: f64| {
            serde_json::json!({
                "file": file,
                "data": {
                    "info": { "cache": cache, "pipeline": pipeline },
                    "sets": { "opsec": sets },
                    "gets": { "opsec": gets },
                }
            })
        };
        let file = serde_json::json!([
            run(
                "bench_rugo-threads_8-pipeline_10-perf_no-run_average.json",
                "rugo",
                10,
                1.0,
                2.0
            ),
            run(
                "bench_rugo-threads_8-pipeline_10-perf_no-run_best.json",
                "rugo",
                10,
                900.0,
                900.0
            ),
            run(
                "bench_rugo-threads_8-pipeline_10-perf_yes-run_average.json",
                "rugo",
                10,
                900.0,
                900.0
            ),
        ]);
        let points = points_of(&file).expect("an array of runs");
        assert_eq!(points.len(), 1, "a run that should not count was counted");
        // Sets and gets together, because that is what the run did rather than what half of it did.
        assert!((points[0].opsec - 3.0).abs() < f64::EPSILON);
        assert_eq!(points[0].pipeline, 10);
        assert_eq!(points[0].cache, "rugo");
    }

    #[test]
    fn rendering_twice_produces_the_same_bytes() {
        // The property the check command depends on. A document that is not a function of its inputs cannot be checked at all.
        let once = render(&[sweep(
            vec![point("rugo", 1, 2.0), point("valkey", 1, 1.0)],
            Vec::new(),
        )]);
        let twice = render(&[sweep(
            vec![point("valkey", 1, 1.0), point("rugo", 1, 2.0)],
            Vec::new(),
        )]);
        assert_eq!(once, twice, "the order of the runs changed the document");
    }
}
