//! `io_uring`, the completion backend.
//!
//! # Why this is not behind [`Poller`](crate::Poller)
//!
//! A readiness interface answers the question "may I read now", and the caller then reads. A completion interface is told "here are the bytes", and the read already happened. Hiding the second behind the first means asking the kernel to tell you when a read would not block and then doing the read yourself, which is the epoll cost with an extra ring in the way. The saving is the syscalls, and the syscalls are only saved if the caller gives up asking.
//!
//! So this is its own interface and the server has its own loop for it. The two loops answer the same commands out of the same [`Conn`](../../rugo_server/conn) and differ only in how bytes get in and out.
//!
//! # What is used and what is not
//!
//! A recv and a send per connection turn, submitted from the completion of the last one, so a connection that is reading commands and answering them makes no syscall of its own at all: the ring carries them and one `io_uring_enter` a loop turn carries the ring.
//!
//! Both kinds of accept are here and the server takes the single shot one. Multishot saves a submission per connection, and a submission per connection is nothing next to what it costs: every thread has an accept armed on the same listener, and a multishot accept lets whichever ring the kernel reaches first take the whole backlog. A connection lives on the thread that accepted it, so that is the whole server's load decided by a race. Re-arming after each completion gives one connection a listener a turn, which is the rule the readiness loop already had and the one that took the busiest thread on `gpc` from 98 per cent of a core to 48.
//!
//! Registered buffers and a provided buffer ring are not used. Both are worth having and both change what a connection owns, and the thing being measured first is whether the completion model pays for itself at all.
//!
//! `IORING_SETUP_SQPOLL` is not used and is not going to be. It removes the last syscall by giving the kernel a polling thread, and a cache server that already runs a thread a core would then be asking for two.
//!
//! # The ABI
//!
//! The structures below are `linux/io_uring.h`'s, written out here because `libc` carries the syscall numbers and not the types. They are kernel user space ABI, so they are frozen: a field may be added to the tail of a union that is already there, and the offsets that exist cannot move without breaking every program built against them.
//!
//! Each one is checked for size in the tests, which is the cheap half of the check that they match. The dear half is that the kernel accepts a ring made out of them, which is what every test in this module does before it asserts anything.

use std::io;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicU32, Ordering};

/// Where in the ring's address space the submission ring is mapped from.
const OFF_SQ_RING: i64 = 0;
/// Where the completion ring is mapped from, on a kernel that does not put both in one mapping.
const OFF_CQ_RING: i64 = 0x0800_0000;
/// Where the submission entries themselves are mapped from.
const OFF_SQES: i64 = 0x1000_0000;

/// The kernel put the submission and completion rings in one mapping.
const FEAT_SINGLE_MMAP: u32 = 1 << 0;

/// Clamp the ring sizes to what the kernel allows rather than refusing a size that is too large.
const SETUP_CLAMP: u32 = 1 << 4;
/// Do not send an interrupt to wake a task that is already running, which is every completion this server reaps.
const SETUP_COOP_TASKRUN: u32 = 1 << 8;
/// Promise that one thread and only one thread submits to this ring, which is true because there is a ring a thread.
const SETUP_SINGLE_ISSUER: u32 = 1 << 12;
/// Run completion work when the ring is entered rather than at the end of whatever syscall happened to be running.
///
/// Wants [`SETUP_SINGLE_ISSUER`] with it, and wants the reaper to enter the ring rather than to spin on the completion queue, which is what [`Ring::submit_and_wait`] does.
const SETUP_DEFER_TASKRUN: u32 = 1 << 13;

/// Ask `io_uring_enter` to reap completions as well as submit.
const ENTER_GETEVENTS: u32 = 1 << 0;

/// Fire a completion after a while, and nothing else.
const OP_TIMEOUT: u8 = 11;
/// Take a connection off a listener.
const OP_ACCEPT: u8 = 13;
/// Send from a buffer.
const OP_SEND: u8 = 26;
/// Receive into a buffer.
const OP_RECV: u8 = 27;

/// The `ioprio` bit that makes an accept stay armed after it has produced a connection.
const ACCEPT_MULTISHOT: u16 = 1 << 0;

/// Set on a completion whose submission is still armed and will produce more.
pub const CQE_MORE: u32 = 1 << 1;

/// Where each part of the submission ring sits inside its mapping.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct SqOffsets {
    /// Byte offset of the head, which the kernel owns.
    head: u32,
    /// Byte offset of the tail, which this owns.
    tail: u32,
    /// Byte offset of the mask that turns a counter into an index.
    ring_mask: u32,
    /// Byte offset of the entry count.
    ring_entries: u32,
    /// Byte offset of the kernel's flags word.
    flags: u32,
    /// Byte offset of the count of submissions the kernel dropped.
    dropped: u32,
    /// Byte offset of the indirection array.
    array: u32,
    /// Reserved.
    resv1: u32,
    /// Reserved, and a user address on a kernel that takes one.
    user_addr: u64,
}

/// Where each part of the completion ring sits inside its mapping.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct CqOffsets {
    /// Byte offset of the head, which this owns.
    head: u32,
    /// Byte offset of the tail, which the kernel owns.
    tail: u32,
    /// Byte offset of the mask that turns a counter into an index.
    ring_mask: u32,
    /// Byte offset of the entry count.
    ring_entries: u32,
    /// Byte offset of the count of completions that did not fit.
    overflow: u32,
    /// Byte offset of the completion entries.
    cqes: u32,
    /// Byte offset of the kernel's flags word.
    flags: u32,
    /// Reserved.
    resv1: u32,
    /// Reserved, and a user address on a kernel that takes one.
    user_addr: u64,
}

/// What `io_uring_setup` is asked for and what it answers.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Params {
    /// How many submission entries the ring got.
    sq_entries: u32,
    /// How many completion entries the ring got.
    cq_entries: u32,
    /// The setup flags asked for.
    flags: u32,
    /// Which core a polling thread would run on, unused here.
    sq_thread_cpu: u32,
    /// How long a polling thread would idle for, unused here.
    sq_thread_idle: u32,
    /// What this kernel supports.
    features: u32,
    /// A ring to share a work queue with, unused here.
    wq_fd: u32,
    /// Reserved.
    resv: [u32; 3],
    /// Where the submission ring's parts are.
    sq_off: SqOffsets,
    /// Where the completion ring's parts are.
    cq_off: CqOffsets,
}

/// One submission.
///
/// The kernel's version is a nest of unions and this is one naming of it. Every field this server sets is at the same offset in both, which is what `#[repr(C)]` and the size check in the tests are for.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Sqe {
    /// Which operation.
    opcode: u8,
    /// Per submission flags, none of which are used here.
    flags: u8,
    /// Per operation flags. The multishot bit on an accept lives here.
    ioprio: u16,
    /// The socket.
    fd: i32,
    /// An offset for a file, and where an accept puts the address length pointer.
    off: u64,
    /// The buffer, and where an accept puts the address pointer.
    addr: u64,
    /// How many bytes.
    len: u32,
    /// Operation flags again, in the union the kernel spells `msg_flags` for a send.
    rw_flags: u32,
    /// Whatever the caller wants back on the completion.
    user_data: u64,
    /// A registered buffer or buffer group, unused here.
    buf_index: u16,
    /// A credential set, unused here.
    personality: u16,
    /// A descriptor for splice, and the file index slot for an accept.
    splice_fd_in: i32,
    /// Reserved.
    addr3: u64,
    /// Reserved.
    pad2: u64,
}

/// One completion.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Cqe {
    /// Whatever the submission carried.
    user_data: u64,
    /// The result, which is a byte count or a negative errno.
    res: i32,
    /// Flags, of which this reads only [`CQE_MORE`].
    flags: u32,
}

/// One finished operation, as the server reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Done {
    /// Whatever the submission carried.
    pub user_data: u64,
    /// A byte count or a descriptor when it is not negative, and a negative errno when it is.
    pub res: i32,
    /// The kernel's flags for this completion.
    pub flags: u32,
}

impl Done {
    /// Whether the submission that produced this is still armed.
    #[must_use]
    pub const fn more(&self) -> bool {
        self.flags & CQE_MORE != 0
    }
}

/// How long a timeout waits.
///
/// The kernel's `__kernel_timespec` rather than the C library's `timespec`, which are the same on a 64 bit machine and are not on a 32 bit one: the kernel's fields are always 64 bits wide so that a ring built by a 32 bit program can be read by a 64 bit kernel without a translation layer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Timespec {
    /// Whole seconds.
    pub sec: i64,
    /// Nanoseconds on top of them.
    pub nsec: i64,
}

/// A mapping this owns and has to give back.
#[derive(Debug)]
struct Mapping {
    /// Where it starts.
    at: *mut libc::c_void,
    /// How long it is.
    len: usize,
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the address and the length are the ones `mmap` returned and nothing else unmaps them.
        unsafe { libc::munmap(self.at, self.len) };
    }
}

/// An `io_uring`, and the two rings it is made of.
///
/// One a thread. The ring is not `Sync` and does not want to be: asking the kernel for a single issuer is a promise to the kernel that one thread submits, and the whole point of the shape is that a connection belongs to a thread.
#[derive(Debug)]
pub struct Ring {
    /// The ring's descriptor.
    fd: RawFd,
    /// The submission ring's mapping, and the completion ring's too on a kernel that shares one.
    _sq_map: Mapping,
    /// The completion ring's mapping, when it is its own.
    _cq_map: Option<Mapping>,
    /// The submission entries' mapping.
    _sqe_map: Mapping,

    /// The kernel's read position in the submission ring.
    sq_head: *const AtomicU32,
    /// This thread's write position in the submission ring.
    sq_tail: *const AtomicU32,
    /// The submission ring's index mask.
    sq_mask: u32,
    /// The submission entries.
    sqes: *mut Sqe,
    /// How many submissions the ring holds.
    entries: u32,
    /// Where the next submission goes, kept here rather than read back from the shared word every time.
    tail: u32,
    /// How many submissions have been written and not yet handed to the kernel.
    pending: u32,

    /// This thread's read position in the completion ring.
    cq_head: *const AtomicU32,
    /// The kernel's write position in the completion ring.
    cq_tail: *const AtomicU32,
    /// The completion ring's index mask.
    cq_mask: u32,
    /// The completion entries.
    cqes: *const Cqe,
}

impl Ring {
    /// Make a ring with room for `entries` submissions.
    ///
    /// The flags are asked for in three steps and each step falls back to the one below it, because a kernel that does not know a setup flag refuses the whole call rather than ignoring the flag. The order is the newest and best first: deferred completion work with a single issuer, then the cooperative task run on its own, then nothing beyond the clamp.
    ///
    /// # Errors
    ///
    /// Whatever the last attempt at `io_uring_setup` reported, which on a kernel with `io_uring` turned off is `ENOSYS`, and in a container that forbids it is `EPERM`.
    pub fn new(entries: u32) -> io::Result<Self> {
        let wanted = [
            SETUP_CLAMP | SETUP_COOP_TASKRUN | SETUP_SINGLE_ISSUER | SETUP_DEFER_TASKRUN,
            SETUP_CLAMP | SETUP_COOP_TASKRUN,
            SETUP_CLAMP,
        ];

        let mut last = io::Error::from_raw_os_error(libc::EINVAL);
        for flags in wanted {
            match Self::setup(entries, flags) {
                Ok(ring) => return Ok(ring),
                Err(error) => last = error,
            }
        }
        Err(last)
    }

    /// One attempt at a ring with exactly `flags`.
    ///
    /// The alignment the lint is worried about is the kernel's promise. Every offset used here came back from `io_uring_setup` for this mapping, the mapping starts on a page, and a ring whose head was not on a four byte boundary would be a ring the kernel could not use either.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "offsets into a page aligned mapping, given by the kernel for that mapping"
    )]
    fn setup(entries: u32, flags: u32) -> io::Result<Self> {
        let mut params = Params {
            flags,
            ..Params::default()
        };

        // SAFETY: `params` is a live local for the call and the kernel writes only into it.
        let fd = unsafe {
            libc::syscall(
                libc::SYS_io_uring_setup,
                libc::c_long::from(entries),
                &raw mut params,
            )
        };
        let fd = i32::try_from(fd).unwrap_or(-1);
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        // From here on the descriptor is ours, so every early return has to close it. `Held` does that, and is dropped without closing once the ring is built.
        let held = Held(fd);

        let sq_len = params.sq_off.array as usize + params.sq_entries as usize * size_of::<u32>();
        let cq_len = params.cq_off.cqes as usize + params.cq_entries as usize * size_of::<Cqe>();
        let single = params.features & FEAT_SINGLE_MMAP != 0;

        let sq_map = map(
            fd,
            if single { sq_len.max(cq_len) } else { sq_len },
            OFF_SQ_RING,
        )?;
        let cq_map = if single {
            None
        } else {
            Some(map(fd, cq_len, OFF_CQ_RING)?)
        };
        let entries_map = map(fd, params.sq_entries as usize * size_of::<Sqe>(), OFF_SQES)?;

        let sq_base = sq_map.at.cast::<u8>();
        let cq_base = cq_map.as_ref().map_or(sq_base, |map| map.at.cast::<u8>());

        // SAFETY: every offset here came from the kernel for this mapping, and the mapping was made at least as long as the largest of them plus the array it names.
        let ring = unsafe {
            Self {
                fd,
                sq_head: sq_base.add(params.sq_off.head as usize).cast(),
                sq_tail: sq_base.add(params.sq_off.tail as usize).cast(),
                sq_mask: *sq_base.add(params.sq_off.ring_mask as usize).cast::<u32>(),
                sqes: entries_map.at.cast(),
                entries: params.sq_entries,
                tail: 0,
                pending: 0,
                cq_head: cq_base.add(params.cq_off.head as usize).cast(),
                cq_tail: cq_base.add(params.cq_off.tail as usize).cast(),
                cq_mask: *cq_base.add(params.cq_off.ring_mask as usize).cast::<u32>(),
                cqes: cq_base.add(params.cq_off.cqes as usize).cast(),
                _sq_map: sq_map,
                _cq_map: cq_map,
                _sqe_map: entries_map,
            }
        };

        // The indirection array exists so that submissions can be written out of order. Nothing here does that, so every slot names itself once and is never written again.
        // SAFETY: the array is `sq_entries` words at the offset the kernel gave, inside the mapping made for it.
        unsafe {
            let array = sq_base.add(params.sq_off.array as usize).cast::<u32>();
            for index in 0..params.sq_entries {
                array.add(index as usize).write(index);
            }
        }

        std::mem::forget(held);
        Ok(ring)
    }

    /// Whether a ring can be made on this machine at all.
    ///
    /// What `--uring auto` asks. A kernel without `io_uring`, a container that forbids the syscall and a sysctl that turns it off all answer the same way, which is the answer this needs.
    #[must_use]
    pub fn available() -> bool {
        Self::new(8).is_ok()
    }

    /// Somewhere to write the next submission, or `None` when the ring is full.
    fn next(&mut self) -> Option<&mut Sqe> {
        // SAFETY: the head is a live word in a mapping this owns, and the kernel writes it with a release.
        let head = unsafe { (*self.sq_head).load(Ordering::Acquire) };
        if self.tail.wrapping_sub(head) >= self.entries {
            return None;
        }
        let index = (self.tail & self.sq_mask) as usize;
        self.tail = self.tail.wrapping_add(1);
        self.pending += 1;
        // SAFETY: the index is masked into the entry array, which is `sq_entries` long and mapped for exactly that.
        let sqe = unsafe { &mut *self.sqes.add(index) };
        *sqe = Sqe::default();
        Some(sqe)
    }

    /// Arm one accept on `fd`, good for one connection.
    ///
    /// The completion carries `user_data` and a descriptor in its result, and nothing is armed afterwards, so a caller that wants another connection asks for another accept. That is the point of it rather than a shortcoming: see the note at the top of this module on what a multishot accept does to how a burst of connections is divided between threads.
    ///
    /// Returns whether there was room in the ring.
    pub fn accept(&mut self, fd: RawFd, user_data: u64) -> bool {
        self.arm_accept(fd, user_data, 0)
    }

    /// Arm a multishot accept on `fd`, which stays armed until it fails.
    ///
    /// Every connection that arrives on that listener produces a completion carrying `user_data` and a descriptor, and only the last of them has [`Done::more`] clear. A completion without it is the accept having ended, and re-arming is the caller's business because only the caller knows whether it still wants the listener.
    ///
    /// Returns whether there was room in the ring.
    pub fn accept_multi(&mut self, fd: RawFd, user_data: u64) -> bool {
        self.arm_accept(fd, user_data, ACCEPT_MULTISHOT)
    }

    /// What both accepts are, differing in one `ioprio` bit.
    fn arm_accept(&mut self, fd: RawFd, user_data: u64, ioprio: u16) -> bool {
        let Some(sqe) = self.next() else {
            return false;
        };
        sqe.opcode = OP_ACCEPT;
        sqe.ioprio = ioprio;
        sqe.fd = fd;
        // A null address and a null length is an accept that does not want to know who connected, which this does not: a cache server logs no peer and authorises nobody by address.
        sqe.addr = 0;
        sqe.off = 0;
        sqe.user_data = user_data;
        true
    }

    /// Fire a completion carrying `user_data` after `after` has passed, and do nothing else.
    ///
    /// What a loop that has nothing to do is woken by. Every other backend gets this from the timeout argument to its wait call, and a ring has no such argument, so waiting for a while is an operation like any other. The completion reports `ETIME`, which is the timeout having expired rather than anything having gone wrong.
    ///
    /// Returns whether there was room in the ring.
    ///
    /// # Safety
    ///
    /// `after` must point to a `Timespec` that stays where it is and stays unchanged until the completion carrying `user_data` arrives, for the same reason [`Ring::recv`] gives.
    pub unsafe fn timeout(&mut self, after: *const Timespec, user_data: u64) -> bool {
        let Some(sqe) = self.next() else {
            return false;
        };
        sqe.opcode = OP_TIMEOUT;
        // Not a descriptor. A timeout is not an operation on a file, and minus one is what the kernel expects to be handed where a file would go.
        sqe.fd = -1;
        // As in [`Ring::recv`]: the kernel reads through this address.
        sqe.addr = u64::try_from(after.expose_provenance()).unwrap_or(0);
        // One timespec, and no count of completions to wait for as well as the clock.
        sqe.len = 1;
        sqe.off = 0;
        sqe.user_data = user_data;
        true
    }

    /// Receive into `at` for `len` bytes.
    ///
    /// Returns whether there was room in the ring.
    ///
    /// # Safety
    ///
    /// `at` must point to `len` writable bytes that stay where they are and stay untouched by anything else until the completion carrying `user_data` arrives. The kernel writes into that memory at a time of its choosing, so a buffer that is freed, moved or reused before then is a use after free that no borrow checker can see.
    pub unsafe fn recv(&mut self, fd: RawFd, at: *mut u8, len: usize, user_data: u64) -> bool {
        let Some(sqe) = self.next() else {
            return false;
        };
        sqe.opcode = OP_RECV;
        sqe.fd = fd;
        // The kernel dereferences this address, so the provenance of the pointer has to be exposed rather than merely its bits taken.
        sqe.addr = u64::try_from(at.expose_provenance()).unwrap_or(0);
        sqe.len = u32::try_from(len).unwrap_or(u32::MAX);
        sqe.user_data = user_data;
        true
    }

    /// Send `len` bytes from `at`.
    ///
    /// Returns whether there was room in the ring.
    ///
    /// # Safety
    ///
    /// `at` must point to `len` readable bytes that stay where they are and stay unchanged until the completion carrying `user_data` arrives, for the same reason [`Ring::recv`] gives.
    pub unsafe fn send(&mut self, fd: RawFd, at: *const u8, len: usize, user_data: u64) -> bool {
        let Some(sqe) = self.next() else {
            return false;
        };
        sqe.opcode = OP_SEND;
        sqe.fd = fd;
        // As in [`Ring::recv`]: the kernel reads through this address.
        sqe.addr = u64::try_from(at.expose_provenance()).unwrap_or(0);
        sqe.len = u32::try_from(len).unwrap_or(u32::MAX);
        // `MSG_NOSIGNAL`, because a send to a peer that has gone should be an error on this call rather than a signal that kills the process.
        sqe.rw_flags = libc::MSG_NOSIGNAL.cast_unsigned();
        sqe.user_data = user_data;
        true
    }

    /// Hand everything written to the kernel and wait for at least `want` completions.
    ///
    /// `want` of nought returns as soon as the submission is done, which is what a turn that already has work in hand asks for.
    ///
    /// # Errors
    ///
    /// Whatever `io_uring_enter` reported, except an interruption by a signal and a timeout, both of which come back as nothing having happened.
    pub fn submit_and_wait(&mut self, want: u32) -> io::Result<()> {
        // The kernel reads the entries this tail publishes, so everything written into them has to be visible first.
        // SAFETY: the tail is a live word in a mapping this owns.
        unsafe { (*self.sq_tail).store(self.tail, Ordering::Release) };

        if self.pending == 0 && want == 0 {
            return Ok(());
        }

        // SAFETY: no pointer is passed. The last two arguments are the signal mask and its size, and a null mask of size nought is no mask.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_io_uring_enter,
                libc::c_long::from(self.fd),
                libc::c_long::from(self.pending),
                libc::c_long::from(want),
                libc::c_long::from(ENTER_GETEVENTS),
                0,
                0,
            )
        };

        if rc < 0 {
            let error = io::Error::last_os_error();
            return match error.kind() {
                io::ErrorKind::Interrupted | io::ErrorKind::TimedOut => Ok(()),
                _ => Err(error),
            };
        }

        // What the kernel took. It takes everything in the ordinary case, and takes less only when it could not keep up, in which case the rest are still in the ring and go with the next call.
        let taken = u32::try_from(rc).unwrap_or(self.pending);
        self.pending = self.pending.saturating_sub(taken);
        Ok(())
    }

    /// Take every completion the kernel has written, into `out`.
    ///
    /// The head is published only after all of them have been copied out, so a completion is never handed back twice and the ring never reuses a slot this is still reading.
    pub fn reap(&mut self, out: &mut Vec<Done>) {
        // SAFETY: both words are live in a mapping this owns. The tail is the kernel's and is written with a release, so acquiring it is what makes the entries it names visible.
        let (head, tail) = unsafe {
            (
                (*self.cq_head).load(Ordering::Relaxed),
                (*self.cq_tail).load(Ordering::Acquire),
            )
        };

        let mut at = head;
        while at != tail {
            let index = (at & self.cq_mask) as usize;
            // SAFETY: the index is masked into the completion array, which is mapped for the entry count the kernel reported.
            let cqe = unsafe { *self.cqes.add(index) };
            out.push(Done {
                user_data: cqe.user_data,
                res: cqe.res,
                flags: cqe.flags,
            });
            at = at.wrapping_add(1);
        }

        // SAFETY: the head is this side's word to write, and releasing it is what tells the kernel the entries before it may be reused.
        unsafe { (*self.cq_head).store(tail, Ordering::Release) };
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        // SAFETY: the descriptor is ours, was made by io_uring_setup, and is not closed anywhere else. The mappings go with the fields that hold them, after this.
        unsafe { libc::close(self.fd) };
    }
}

/// A descriptor that is closed unless somebody takes it.
///
/// The setup path makes a descriptor and then does four things that can fail, and a ring that failed at the fourth would otherwise leak the first.
struct Held(
    /// The descriptor to close.
    RawFd,
);

impl Drop for Held {
    fn drop(&mut self) {
        // SAFETY: the descriptor is ours and is only reached here when the ring was not built out of it.
        unsafe { libc::close(self.0) };
    }
}

/// Map `len` bytes of the ring at `offset`.
fn map(fd: RawFd, len: usize, offset: i64) -> io::Result<Mapping> {
    // SAFETY: no pointer is passed in, and the kernel decides where the mapping lands.
    let at = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_POPULATE,
            fd,
            offset,
        )
    };
    if std::ptr::eq(at, libc::MAP_FAILED) {
        return Err(io::Error::last_os_error());
    }
    Ok(Mapping { at, len })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read as _, Write as _};
    use std::net::{Ipv4Addr, TcpListener, TcpStream};
    use std::os::fd::AsRawFd as _;

    /// A ring, or a message saying why this machine cannot have one.
    ///
    /// Every test here skips rather than fails when the kernel has no `io_uring`, because these run in containers whose seccomp profile forbids the syscall and a test that fails there is a test that gets deleted.
    ///
    /// That is not enough for Miri, which is why every caller of this carries a `cfg_attr(miri, ignore)` as well. A kernel that refuses the syscall returns an error and this returns `None`; an interpreter that has never heard of the syscall stops the program instead, and a stopped program is not a skip. The test above this one, which only checks that the ABI structures are the sizes the kernel says they are, needs no ring and is interpreted.
    fn ring() -> Option<Ring> {
        Ring::new(64).ok()
    }

    #[test]
    fn the_abi_structures_are_the_sizes_the_kernel_says_they_are() {
        assert_eq!(size_of::<Sqe>(), 64, "a submission is sixty four bytes");
        assert_eq!(size_of::<Cqe>(), 16, "a completion is sixteen bytes");
        assert_eq!(size_of::<SqOffsets>(), 40);
        assert_eq!(size_of::<CqOffsets>(), 40);
        assert_eq!(size_of::<Params>(), 120);
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri has no kernel to set a ring up against")]
    fn a_ring_carries_bytes_from_one_socket_to_another() {
        let Some(mut ring) = ring() else {
            return;
        };

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();

        client.write_all(b"hello").unwrap();

        let mut into = [0u8; 16];
        // SAFETY: the buffer is a live local that outlives the reap below, and nothing else writes to it while the kernel holds it.
        assert!(unsafe { ring.recv(server.as_raw_fd(), into.as_mut_ptr(), into.len(), 7) });
        ring.submit_and_wait(1).unwrap();

        let mut done = Vec::new();
        ring.reap(&mut done);
        assert_eq!(
            done.len(),
            1,
            "one recv was submitted and one should finish"
        );
        assert_eq!(done[0].user_data, 7, "the user data came back changed");
        assert_eq!(done[0].res, 5, "five bytes were sent");
        assert_eq!(&into[..5], b"hello");
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri has no kernel to set a ring up against")]
    fn a_send_reaches_the_peer() {
        let Some(mut ring) = ring() else {
            return;
        };

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();

        let out = b"+PONG\r\n";
        // SAFETY: the buffer is a static that outlives everything here.
        assert!(unsafe { ring.send(server.as_raw_fd(), out.as_ptr(), out.len(), 9) });
        ring.submit_and_wait(1).unwrap();

        let mut done = Vec::new();
        ring.reap(&mut done);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].res, 7, "seven bytes should have gone out");

        let mut back = [0u8; 7];
        client.read_exact(&mut back).unwrap();
        assert_eq!(&back, out);
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri has no kernel to set a ring up against")]
    fn one_multishot_accept_takes_more_than_one_connection() {
        let Some(mut ring) = ring() else {
            return;
        };

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();

        assert!(ring.accept_multi(listener.as_raw_fd(), 1));
        ring.submit_and_wait(0).unwrap();

        // Held, so that the descriptors the kernel hands back stay distinct.
        let _clients: Vec<TcpStream> = (0..3).map(|_| TcpStream::connect(addr).unwrap()).collect();

        let mut done = Vec::new();
        for _ in 0..3 {
            ring.submit_and_wait(1).unwrap();
            ring.reap(&mut done);
            if done.len() >= 3 {
                break;
            }
        }

        assert!(
            done.len() >= 3,
            "three connections arrived and {} were accepted",
            done.len()
        );
        for accepted in &done {
            assert_eq!(accepted.user_data, 1);
            assert!(accepted.res >= 0, "an accept failed with {}", accepted.res);
            assert!(
                accepted.more(),
                "a multishot accept should still be armed after each connection"
            );
            // SAFETY: the descriptor came from this accept and is not held anywhere else.
            unsafe { libc::close(accepted.res) };
        }
    }

    // What the server relies on to keep a burst of connections from landing on one thread. A single shot accept takes one and then stops, so a thread that wants a second connection asks for it and gives the other threads a turn in between.
    #[test]
    #[cfg_attr(miri, ignore = "Miri has no kernel to set a ring up against")]
    fn one_ordinary_accept_takes_one_connection_and_stops() {
        let Some(mut ring) = ring() else {
            return;
        };

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();

        assert!(ring.accept(listener.as_raw_fd(), 1));
        ring.submit_and_wait(0).unwrap();

        let _clients: Vec<TcpStream> = (0..3).map(|_| TcpStream::connect(addr).unwrap()).collect();

        let mut done = Vec::new();
        ring.submit_and_wait(1).unwrap();
        ring.reap(&mut done);

        assert_eq!(
            done.len(),
            1,
            "three connections arrived and one accept should have taken one of them"
        );
        assert!(done[0].res >= 0, "the accept failed with {}", done[0].res);
        assert!(
            !done[0].more(),
            "an ordinary accept should not still be armed"
        );
        // SAFETY: the descriptor came from this accept and is not held anywhere else.
        unsafe { libc::close(done[0].res) };

        // And nothing else arrives on its own, though two connections are still waiting.
        done.clear();
        ring.reap(&mut done);
        assert!(done.is_empty(), "the accept produced a second completion");
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri has no kernel to set a ring up against")]
    fn a_timeout_wakes_a_ring_that_has_nothing_else_to_do() {
        let Some(mut ring) = ring() else {
            return;
        };

        let after = Timespec {
            sec: 0,
            nsec: 20_000_000,
        };
        let at = std::time::Instant::now();
        // SAFETY: the timespec is a live local that outlives the wait below.
        assert!(unsafe { ring.timeout(&raw const after, 9) });
        ring.submit_and_wait(1).unwrap();

        let mut done = Vec::new();
        ring.reap(&mut done);
        assert_eq!(done.len(), 1, "the timeout did not complete");
        assert_eq!(done[0].user_data, 9);
        assert_eq!(
            done[0].res,
            -libc::ETIME,
            "a timeout that expires reports ETIME rather than an error"
        );
        // Loosely, because how late a wakeup is belongs to the scheduler. That it waited at all is the claim.
        assert!(
            at.elapsed() >= std::time::Duration::from_millis(15),
            "the ring returned in {:?}, which is not a wait",
            at.elapsed()
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri has no kernel to set a ring up against")]
    fn a_ring_that_is_full_says_so_rather_than_overwriting_itself() {
        let Some(mut ring) = ring() else {
            return;
        };

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        // One more than the ring holds, and the extra one has to be refused rather than written over a submission the kernel has not read.
        let mut refused = false;
        for _ in 0..100 {
            if !ring.accept(listener.as_raw_fd(), 1) {
                refused = true;
                break;
            }
        }
        assert!(refused, "a ring of sixty four took a hundred submissions");
    }
}
