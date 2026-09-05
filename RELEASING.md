# Releasing

`rugo` releases early and often, from `0.1.0`, well before anything here is worth depending on. The reason is not ceremony. This project's whole output is a ratio between measurements, and a measurement that cannot be pinned to a version is not a measurement of anything. A tag is what lets a scoreboard row say which code produced it.

## What the numbers mean

While the major is 0, the guarantees are these and only these.

**Minor** goes up when a milestone lands. `M0` through `M2` together are `0.1.0`, and every milestone after that takes the next minor. A minor may break anything: the Rust API of the `rugo-*` crates, the command set, the flags, the entry encoding. That is what a zero major is for, and pretending otherwise would be worse than saying it plainly.

**Patch** goes up for anything that lands between milestones. Fixes, optimisations, a new sweep in `bench/` and the scoreboard it moves, documentation, a dependency bump, CI. It is cut after a handful of pull requests rather than after each one.

`1.0.0` means the wire protocol and the flags are stable and the gate in `SCOREBOARD.md` has been met on a qualified host. Not before. A `1.0.0` cut while the pogocache row still reads `not yet` would be a version number claiming something the data next to it does not.

## Cutting one

1. The PR is green on `ci`, and for a minor it is also green on `deep`. `deep` does not run on a pull request unless somebody asks for it, so put the `deep` label on the PR and wait for Miri under both borrow models, loom, the fuzzers, the release matrix and the benchmark smoke run. A patch can go in on `ci` alone as long as the nightly `deep` run before it was green.
2. Cross-lint. The probe has three implementations — NEON, SSE2 and word-at-a-time — and the machine you are on compiles exactly one of them. Two SSE2 defects in this repository were found only by linting for `x86_64-unknown-linux-gnu` from an ARM Mac and would otherwise have been found by a bench host. `cargo clippy --workspace --lib --bins --target x86_64-unknown-linux-gnu` is the invocation; `--all-targets` additionally needs a Linux C toolchain, which Criterion's `alloca` pulls in, so run that part with `--exclude rugo-map` or leave it to CI.
3. Bump `workspace.package.version` in the root `Cargo.toml`, and the `version` on every path dependency in `[workspace.dependencies]` with it, then run `cargo update --workspace` so `Cargo.lock` agrees. The release workflow refuses a tag whose version does not match the manifest, and separately refuses one where a `[workspace.dependencies]` version was left behind, so a half-done bump fails before anything is published rather than after.
4. Add the release's section to `CHANGELOG.md` and replace `unreleased` in its heading with the date. Write it for somebody who was not in the room: what changed, why, and what it costs them.
5. If a sweep landed in this release, `cargo xtask scoreboard` and commit `SCOREBOARD.md` with it. The `generated` job fails the build if the committed document is not what the generator would write, and `release.yml` checks it again against the tag.
6. Merge the PR.
7. Tag the merge commit `vX.Y.Z` and push the tag. The `release` workflow re-runs the gate jobs against the tag, builds the four binaries, publishes the GitHub release with the changelog section as its body, and publishes the crates last.

A tag is never moved and never deleted. If a release is wrong, the fix is the next patch.

## crates.io

The whole workspace goes up, not just `rugo`. crates.io has no way to publish a crate without publishing what it depends on, so the six `rugo-*` members are part of every release whether or not anybody is meant to depend on them directly. `rugo-map` is the one that is genuinely worth using on its own: it is a concurrent cache with a memory ceiling and no server attached. Only `xtask` stays out, because it is `publish = false`.

The credential is `CARGO_REGISTRY_TOKEN`, scoped to publish-new and publish-update.

Binaries go first and crates.io goes last, because a GitHub release can be deleted and a version on crates.io cannot.

crates.io rate limits brand new crate names much harder than a new version of a name that already exists: a burst of five, then one every ten minutes. The first release of this workspace is seven names that have never existed, so it will sit and wait for two of them, and there is nothing to do about that but wait. The publish step asks the registry which crates are already up at this version and skips them, so a run that dies on the fifth upload is fixed by re-running the job rather than by cutting another version. Seven uploads are not a transaction and pretending otherwise is how a half-published release becomes a permanent one.

A tag push runs the copy of `release.yml` that is in the tag, so a fix landed on `main` afterwards does not apply to it and the tag cannot be moved to pick it up. The recovery is to dispatch `release.yml` from `main` with `tag:` set, which gives you main's workflow against the tag's code, because every checkout in the file takes `ref: ${{ inputs.tag || github.ref }}`.

## What a release note has to contain

Every section carries whichever of these apply, in this order. Headings that have nothing under them are left out rather than filled with "none".

- **Milestone.** Which one, and a sentence on what it was for.
- **Added**, **Changed**, **Fixed.** Ordinary changes.
- **Performance.** Numbers, each with the host, the profile and the date, and whether it is a published sweep or a development measurement. A development measurement says so in the same sentence as the number, not in a footnote. A number from a laptop does not go in at all: two runs of identical code on the development machine here disagreed by 198 percent, which is not a measurement, it is weather.
- **Memory.** Total bytes per entry and overhead bytes per entry, kept apart. They are two different claims and the difference between them is most of what this project is about; see `SCOREBOARD.md` for why one of them can be halved and the other cannot.
- **Known gaps.** What the milestone was gated on and did not reach, and what is deliberately unimplemented so far. This is the section that keeps the rest honest.

## Style

Prose is not hard wrapped. One line per paragraph and one line per bullet, however long it runs, in every markdown file, pull request body, issue body and commit message in this repository. A sentence broken across two lines makes a one word edit show up as a two line diff, and it makes a paragraph impossible to grep for.
