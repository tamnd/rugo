//! The completion loop, which is the readiness loop with the waiting moved into the kernel.
//!
//! # What is different
//!
//! Everything a command does is the same. A connection holds the same two buffers, parses the same bytes and writes the same replies, and [`Conn::step`] is the only thing here that the readiness loop does not have.
//!
//! What changes is who does the reading. Under epoll the loop is told the socket is readable and then reads it, which is two syscalls for one read and three for a batch that also writes. Here the read is submitted before the bytes exist and the completion is the bytes having arrived, so a connection serving a pipeline makes no syscall of its own at all: one `io_uring_enter` a turn carries every connection's read and every connection's write together.
//!
//! # One operation a connection
//!
//! A connection has at most one submission outstanding, and it submits the next one when that one completes. That is what makes the addresses handed to the kernel safe: between a submission and its completion nothing touches either buffer, so neither can be grown, compacted or freed while the kernel is writing into it.
//!
//! It is also why a connection's slot cannot be reused early. A slot is emptied while its own completion is being handled, at which point there is nothing of that connection left in the ring, and the descriptor is closed by dropping the socket exactly as the other loop does it.
//!
//! # The clock
//!
//! A ring has no timeout argument to wait on, so waiting for a while is an operation like any other. One timeout is kept armed and re-armed when it fires, which is what gives an idle thread the same hundred millisecond tick the poller's timeout gives it. Everything else in the loop is the same afterwards: the clock every expiry is measured against, and a few slots of the background sweep.

use std::io;
use std::net::TcpStream;
use std::os::fd::{AsRawFd as _, FromRawFd as _, RawFd};
use std::os::unix::net::UnixStream;

use rugo_net::uring::{Done, Ring, Timespec};

use super::{IDLE, Listener, SWEEP, Worker};
use crate::conn::{Conn, Step, Stream};
use crate::dispatch::Env;

/// How many submissions a thread's ring holds before it has to be flushed.
///
/// One in flight a connection, plus one accept a listener, plus the tick, so this is a connection count and what matters is how many one thread carries. The sweep opens sixteen a load generator thread, which is two hundred and fifty-six in total across every server thread. A ring that does fill is flushed rather than failed, so this is where the flushing starts rather than a ceiling on the work.
const ENTRIES: u32 = 2048;

/// What is said when the ring is still full after being handed to the kernel, which is not a state that has an explanation.
const STUCK: &str = "the submission queue stayed full after a flush";

/// Which top bits of a submission's user data say what the completion is for.
const SHIFT: u32 = 62;
/// A connection arriving on a listener, indexed by which listener.
const ACCEPT: u64 = 0 << SHIFT;
/// Bytes arriving on a connection, indexed by its slot.
const RECV: u64 = 1 << SHIFT;
/// Bytes leaving a connection, indexed by its slot.
const SEND: u64 = 2 << SHIFT;
/// The clock, indexed by nothing.
const TICK: u64 = 3 << SHIFT;
/// The rest of the user data, which is the index.
const INDEX: u64 = (1 << SHIFT) - 1;

/// The user data for an operation of `kind` on `index`.
fn token(kind: u64, index: usize) -> u64 {
    // An index this large is more connections than a machine has descriptors, so the saturation is unreachable rather than a case whose behaviour is worth defining.
    kind | (u64::try_from(index).unwrap_or(INDEX) & INDEX)
}

/// Put one read or write in the ring, flushing what is already there if it is full.
///
/// The kernel drains the submission queue on every `io_uring_enter`, so full is a state that lasts until the next one rather than a limit on how much work a turn may hold.
fn submit(ring: &mut Ring, fd: RawFd, step: Step, index: usize) -> io::Result<()> {
    for _ in 0..2 {
        // SAFETY: both addresses come from the connection at `index`, whose buffers nothing touches between here and the completion carrying this token, because that connection has no other operation in flight.
        let put = unsafe {
            match step {
                Step::Recv { at, len } => ring.recv(fd, at, len, token(RECV, index)),
                Step::Send { at, len } => ring.send(fd, at, len, token(SEND, index)),
                // Not an operation, and not reachable: a connection that is going is dropped by the caller rather than submitted for.
                Step::Close => return Ok(()),
            }
        };
        if put {
            return Ok(());
        }
        ring.submit_and_wait(0)?;
    }
    Err(io::Error::other(STUCK))
}

/// Arm one accept on `fd`, flushing the ring if it is full.
fn arm_accept(ring: &mut Ring, fd: RawFd, user_data: u64) -> io::Result<()> {
    for _ in 0..2 {
        if ring.accept(fd, user_data) {
            return Ok(());
        }
        ring.submit_and_wait(0)?;
    }
    Err(io::Error::other(STUCK))
}

/// Arm the tick, flushing the ring if it is full.
///
/// # Safety
///
/// `after` must outlive every operation in `ring`, because the kernel reads it when the timeout is armed rather than when it is submitted.
unsafe fn arm_tick(ring: &mut Ring, after: *const Timespec) -> io::Result<()> {
    for _ in 0..2 {
        // SAFETY: the caller's promise, passed on unchanged.
        if unsafe { ring.timeout(after, TICK) } {
            return Ok(());
        }
        ring.submit_and_wait(0)?;
    }
    Err(io::Error::other(STUCK))
}

/// Forget the connection in `at`, which closes its socket, and put the slot back.
///
/// Only ever called while handling that connection's own completion, so there is nothing of it left in the ring to complete against a descriptor that has gone.
fn discard(conns: &mut [Option<Conn>], free: &mut Vec<usize>, at: usize) {
    if let Some(slot) = conns.get_mut(at)
        && slot.take().is_some()
    {
        free.push(at);
    }
}

impl Worker<'_> {
    /// Serve on a ring until something fails.
    pub(super) fn serve_uring(&self, env: &Env<'_>) -> io::Result<()> {
        // Declared before the ring so that they outlive it. Closing a ring cancels every operation still in it and waits for the cancellation, which is what makes it safe to free a buffer the kernel holds the address of, and that has to happen before the buffers go.
        let mut conns: Vec<Option<Conn>> = Vec::new();
        let mut free: Vec<usize> = Vec::new();
        let mut done: Vec<Done> = Vec::new();
        let after = Timespec {
            sec: 0,
            // The poller's timeout, as a whole number of nanoseconds. The two loops should give the clock the same tick, and there is no reason for them to differ on how long an idle thread sleeps.
            nsec: i64::try_from(IDLE.as_nanos()).unwrap_or(i64::MAX),
        };

        let mut ring = Ring::new(ENTRIES)?;
        for (at, listener) in self.listeners.iter().enumerate() {
            arm_accept(&mut ring, listener.as_raw_fd(), token(ACCEPT, at))?;
        }
        // SAFETY: `after` is a local of this function, which does not return while the ring holds anything, and the ring is dropped before it.
        unsafe { arm_tick(&mut ring, &raw const after)? };

        loop {
            // One completion at least, which the tick guarantees arrives whether or not a client says anything.
            ring.submit_and_wait(1)?;
            done.clear();
            ring.reap(&mut done);

            for finished in &done {
                let at = usize::try_from(finished.user_data & INDEX).unwrap_or(0);
                match finished.user_data & !INDEX {
                    ACCEPT => {
                        self.took(&mut ring, env, at, finished.res, &mut conns, &mut free)?;
                    }
                    RECV | SEND => {
                        Self::carried(&mut ring, env, at, finished.res, &mut conns, &mut free)?;
                    }
                    // The tick, which reports `ETIME` and means only that a hundred milliseconds passed.
                    // SAFETY: as above, and there is only ever one of these in the ring at a time.
                    _ => unsafe { arm_tick(&mut ring, &raw const after)? },
                }
            }

            // Whether or not anything happened, for the reason the readiness loop gives: a thread that only ticked the clock when it was busy would leave an idle server's keys immortal.
            self.map.clock().tick();
            self.map.sweep(SWEEP);
        }
    }

    /// Take the connection an accept produced, and ask for the next one.
    ///
    /// One accept a listener a turn, which is the rule the readiness loop arrived at the hard way: a connection lives on the thread that accepted it, so a thread that takes a whole burst is a server running on one core no matter how many it was given. Re-arming here rather than arming a multishot accept is what keeps a burst divided.
    fn took(
        &self,
        ring: &mut Ring,
        env: &Env<'_>,
        listener: usize,
        res: i32,
        conns: &mut Vec<Option<Conn>>,
        free: &mut Vec<usize>,
    ) -> io::Result<()> {
        let Some(which) = self.listeners.get(listener) else {
            return Ok(());
        };

        if res < 0 {
            let error = io::Error::from_raw_os_error(res.saturating_neg());
            return match error.kind() {
                // A client that connected and went away again before this got to it, and a syscall a signal interrupted. Both are ordinary and neither says anything about the listener.
                io::ErrorKind::ConnectionAborted
                | io::ErrorKind::Interrupted
                | io::ErrorKind::WouldBlock => {
                    arm_accept(ring, which.as_raw_fd(), token(ACCEPT, listener))
                }
                // Out of descriptors, or a listener that has gone. Re-arming would spin on the same failure, and the readiness loop stops the thread on the same errors.
                _ => Err(error),
            };
        }

        // SAFETY: the descriptor came from this accept, nothing else holds it, and its kind is the kind of the listener it arrived on.
        let stream = unsafe {
            match which {
                Listener::Tcp(_) => Stream::Tcp(TcpStream::from_raw_fd(res)),
                Listener::Unix(_) => Stream::Unix(UnixStream::from_raw_fd(res)),
            }
        };
        arm_accept(ring, which.as_raw_fd(), token(ACCEPT, listener))?;

        if stream.prepare_uring().is_err() {
            return Ok(());
        }
        let mut conn = Conn::new(stream);
        let fd = conn.fd();
        // A connection that has said nothing yet has nothing owed to it and nothing parsed, so this is where its first read is asked for.
        let step = conn.step(env);

        let at = if let Some(at) = free.pop() {
            conns[at] = Some(conn);
            at
        } else {
            conns.push(Some(conn));
            conns.len() - 1
        };

        if let Err(error) = submit(ring, fd, step, at) {
            discard(conns, free, at);
            return Err(error);
        }
        self.stats.thread(self.thread).connection();
        Ok(())
    }

    /// Take what a read or a write completed and submit whatever the connection wants next.
    fn carried(
        ring: &mut Ring,
        env: &Env<'_>,
        at: usize,
        res: i32,
        conns: &mut [Option<Conn>],
        free: &mut Vec<usize>,
    ) -> io::Result<()> {
        // Nought is a peer that hung up and a negative is the socket having failed. Either way there is nobody left to tell, which is what the readiness loop does with the same two cases.
        let Some(bytes) = usize::try_from(res).ok().filter(|&bytes| bytes > 0) else {
            discard(conns, free, at);
            return Ok(());
        };
        let Some(Some(conn)) = conns.get_mut(at) else {
            return Ok(());
        };

        // Which of the two completed is not read off the token. A connection submits a send only while it owes bytes and a read only when it owes none, so its own state says which, and a token that said it as well would be a second copy of the same fact with a way to disagree.
        if conn.owes() {
            conn.written(bytes);
        } else {
            conn.arrived(bytes);
        }

        let fd = conn.fd();
        let step = conn.step(env);
        if matches!(step, Step::Close) {
            discard(conns, free, at);
            return Ok(());
        }
        if let Err(error) = submit(ring, fd, step, at) {
            discard(conns, free, at);
            return Err(error);
        }
        Ok(())
    }
}
