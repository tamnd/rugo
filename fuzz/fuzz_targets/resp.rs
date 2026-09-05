//! The parser, against bytes nobody chose.
//!
//! Every byte of a request comes from the network, so the parser is the one piece of this server that an unauthenticated stranger drives directly. What is checked here is not that it accepts the right things — the unit tests do that — but that there is no input at all for which it panics, loops, or hands back a span that is not inside the buffer it was given.
//!
//! That last one is the reason this exists. A span is an offset and a length into the read buffer, and the connection turns it into a slice without checking, because checking on every argument of every command is a cost the hot path should not pay. The parser is what makes that sound, so the parser is what has to be fuzzed.

#![no_main]

use libfuzzer_sys::fuzz_target;
use rugo_resp::{Command, Parsed, parse, scan};

fuzz_target!(|data: &[u8]| {
    let mut cmd = Command::default();

    // Parsed repeatedly, the way a connection parses a buffer that holds several commands, so that a parse which reports the wrong length is caught by the next one starting in the wrong place rather than being invisible.
    let mut at = 0;
    for _ in 0..64 {
        let rest = &data[at..];
        match parse(rest, &mut cmd) {
            Ok(Parsed::Done(len)) => {
                // A command that occupied no bytes would make the loop above a spin.
                assert!(len > 0, "a whole command took no bytes");
                assert!(len <= rest.len(), "a command ran past the buffer it was in");

                for n in 0..cmd.len() {
                    // The claim the connection relies on: every argument is inside the bytes the parser said it had read.
                    let arg = cmd.arg(n, rest).expect("an argument the command counted");
                    let start = arg.as_ptr() as usize - rest.as_ptr() as usize;
                    assert!(
                        start + arg.len() <= len,
                        "argument {n} runs past the command that contains it"
                    );
                }
                at += len;
            }
            // Nothing more can come of the same bytes, and a connection would go back to the kernel here.
            Ok(Parsed::More) | Err(_) => break,
        }
    }

    // The word-at-a-time newline scan against the definition it is meant to be faster than, at every offset the input allows.
    for from in 0..data.len().min(64) {
        let tail = &data[from..];
        assert_eq!(
            scan(tail),
            tail.iter().position(|&b| b == b'\n'),
            "the scan disagreed with a byte at a time from offset {from}"
        );
    }
});
