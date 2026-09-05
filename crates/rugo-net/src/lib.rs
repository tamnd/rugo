//! Readiness notification, and the sockets a cache server listens on.
//!
//! # What is here and what is deliberately not
//!
//! A [`Poller`] over `epoll` on Linux and `kqueue` everywhere else, and the two listener kinds a cache server needs. Accepting, reading and writing are `std`'s, because `std` already wraps them and wrapping them again would be unsafe code written twice to reach the same syscall. What `std` has no answer for is waiting on a thousand sockets at once, and that is the whole of what this crate adds.
//!
//! The unsafe here is therefore three calls on Linux and two on everything else, each with the pointer and length it is given checked against a buffer this crate owns.
//!
//! # Level triggered
//!
//! Both backends are used in their default level-triggered mode. Edge triggering saves a syscall on a connection that has more to read than one buffer holds, and costs correctness on every path that forgets to drain: a connection whose reader stops one byte early under edge triggering hangs until the client sends something else. A cache server's reads are one buffer nearly always, so the saving is nearly never and the hazard is always.
//!
//! `io_uring` is the interesting backend and it is not here yet. It replaces this interface rather than sitting under it, because a completion model does not pretend to be a readiness one without giving back what makes it worth having.

use std::io;
use std::os::fd::RawFd;

/// What a socket is being watched for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interest {
    /// Wake when there is something to read, or when the peer has gone.
    pub read: bool,
    /// Wake when a write would not block, which matters only while a reply is half sent.
    pub write: bool,
}

impl Interest {
    /// Reading only, which is what a connection wants nearly all of the time.
    pub const READ: Self = Self {
        read: true,
        write: false,
    };
    /// Reading and writing, which is what a connection wants while a reply is stuck.
    pub const BOTH: Self = Self {
        read: true,
        write: true,
    };
}

/// One socket that is ready, and what it is ready for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ready {
    /// Whatever number the caller registered the socket under.
    pub token: u64,
    /// There is something to read, or there will never be anything again.
    pub read: bool,
    /// A write would make progress.
    pub write: bool,
    /// The connection is gone or broken. Not an instruction to close: there may still be bytes to read first, and a cache client that sends a command and shuts down its writing half expects the answer.
    pub gone: bool,
}

/// How many readiness events one [`Poller::wait`] may return.
///
/// One wait per event loop turn, and a turn processes everything it is handed before waiting again, so a larger number is a longer turn rather than more throughput. A thousand is more than a single thread will have ready at once at any offered load this server survives.
const CAPACITY: usize = 1024;

/// The last error the operating system reported, as an [`io::Error`].
fn last() -> io::Error {
    io::Error::last_os_error()
}

/// Whether a syscall's return means it failed.
const fn failed(rc: i32) -> bool {
    rc < 0
}

/// How many events the kernel wrote, as a length.
///
/// Only ever called on a return that [`failed`] has already said is not negative, so the fallback is unreachable. It is a fallback rather than an unwrap because an unwrap here would be a panic in an event loop, and a length of zero is the answer that does the least.
fn many(count: i32) -> usize {
    usize::try_from(count).unwrap_or(0)
}

#[cfg(target_os = "linux")]
mod imp {
    use super::{CAPACITY, Interest, RawFd, Ready, failed, io, last, many};
    use std::time::Duration;

    /// An `epoll` instance.
    #[derive(Debug)]
    pub struct Poller {
        /// The epoll file descriptor.
        fd: RawFd,
        /// Where the kernel writes the ready list, owned so its length is never in question.
        events: Vec<libc::epoll_event>,
        /// What the last wait found, translated.
        ready: Vec<Ready>,
    }

    impl Poller {
        /// Make one.
        ///
        /// # Errors
        ///
        /// Whatever `epoll_create1` reported, which in practice is the process's file descriptor limit.
        pub fn new() -> io::Result<Self> {
            // SAFETY: no pointers, and the only flag is the one that closes the descriptor across an exec.
            let fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
            if failed(fd) {
                return Err(last());
            }
            Ok(Self {
                fd,
                events: vec![libc::epoll_event { events: 0, u64: 0 }; CAPACITY],
                ready: Vec::with_capacity(CAPACITY),
            })
        }

        /// Watch `fd` under `token`.
        ///
        /// # Errors
        ///
        /// Whatever `epoll_ctl` reported.
        pub fn add(&self, fd: RawFd, token: u64, interest: Interest) -> io::Result<()> {
            self.ctl(libc::EPOLL_CTL_ADD, fd, token, interest)
        }

        /// Change what `fd` is watched for.
        ///
        /// # Errors
        ///
        /// Whatever `epoll_ctl` reported.
        pub fn modify(&self, fd: RawFd, token: u64, interest: Interest) -> io::Result<()> {
            self.ctl(libc::EPOLL_CTL_MOD, fd, token, interest)
        }

        /// Stop watching `fd`.
        ///
        /// # Errors
        ///
        /// Whatever `epoll_ctl` reported. A descriptor that has already been closed is not an error worth reporting, because closing removes it anyway.
        pub fn remove(&self, fd: RawFd) -> io::Result<()> {
            // SAFETY: the event pointer is unused by EPOLL_CTL_DEL on every kernel since 2.6.9, but is passed as a valid pointer anyway because kernels before that dereferenced it.
            let rc = unsafe {
                let mut event = libc::epoll_event { events: 0, u64: 0 };
                libc::epoll_ctl(self.fd, libc::EPOLL_CTL_DEL, fd, &raw mut event)
            };
            if failed(rc) { Err(last()) } else { Ok(()) }
        }

        fn ctl(&self, op: i32, fd: RawFd, token: u64, interest: Interest) -> io::Result<()> {
            let mut event = libc::epoll_event {
                events: mask(interest),
                u64: token,
            };
            // SAFETY: `event` is a live local for the duration of the call, and epoll_ctl copies what it needs out of it before returning.
            let rc = unsafe { libc::epoll_ctl(self.fd, op, fd, &raw mut event) };
            if failed(rc) { Err(last()) } else { Ok(()) }
        }

        /// Wait for something to happen, for at most `timeout`.
        ///
        /// # Errors
        ///
        /// Whatever `epoll_wait` reported, except an interruption by a signal, which comes back as no events rather than as an error.
        pub fn wait(&mut self, timeout: Option<Duration>) -> io::Result<&[Ready]> {
            let millis = timeout.map_or(-1, |d| {
                i32::try_from(d.as_millis()).unwrap_or(i32::MAX).max(0)
            });
            // SAFETY: the pointer and the count are this vector's own, and the vector is not touched until the call returns.
            let count = unsafe {
                libc::epoll_wait(
                    self.fd,
                    self.events.as_mut_ptr(),
                    i32::try_from(self.events.len()).unwrap_or(i32::MAX),
                    millis,
                )
            };
            if failed(count) {
                let err = last();
                if err.kind() == io::ErrorKind::Interrupted {
                    self.ready.clear();
                    return Ok(&self.ready);
                }
                return Err(err);
            }

            self.ready.clear();
            for event in &self.events[..many(count)] {
                self.ready.push(Ready {
                    token: event.u64,
                    read: event.events & (libc::EPOLLIN as u32) != 0,
                    write: event.events & (libc::EPOLLOUT as u32) != 0,
                    gone: event.events & ((libc::EPOLLHUP | libc::EPOLLERR) as u32) != 0,
                });
            }
            Ok(&self.ready)
        }
    }

    /// The event mask an interest asks for.
    ///
    /// `EPOLLRDHUP` is in the read set rather than being its own thing, because a peer that closed its writing half is a peer whose remaining bytes have to be read before the connection goes.
    fn mask(interest: Interest) -> u32 {
        let mut mask = 0u32;
        if interest.read {
            mask |= (libc::EPOLLIN | libc::EPOLLRDHUP) as u32;
        }
        if interest.write {
            mask |= libc::EPOLLOUT as u32;
        }
        mask
    }

    impl Drop for Poller {
        fn drop(&mut self) {
            // SAFETY: the descriptor is ours, was made by epoll_create1, and is not closed anywhere else.
            unsafe { libc::close(self.fd) };
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::{CAPACITY, Interest, RawFd, Ready, failed, io, last, many};
    use std::time::Duration;

    /// A `kqueue` instance.
    #[derive(Debug)]
    pub struct Poller {
        /// The kqueue file descriptor.
        fd: RawFd,
        /// Where the kernel writes the ready list.
        events: Vec<libc::kevent>,
        /// What the last wait found, translated.
        ready: Vec<Ready>,
    }

    impl Poller {
        /// Make one.
        ///
        /// # Errors
        ///
        /// Whatever `kqueue` reported, which in practice is the process's file descriptor limit.
        pub fn new() -> io::Result<Self> {
            // SAFETY: no arguments and no pointers.
            let fd = unsafe { libc::kqueue() };
            if failed(fd) {
                return Err(last());
            }
            // SAFETY: the descriptor is ours and was made a moment ago; this only asks that it close across an exec.
            unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
            Ok(Self {
                fd,
                events: vec![blank(); CAPACITY],
                ready: Vec::with_capacity(CAPACITY),
            })
        }

        /// Watch `fd` under `token`.
        ///
        /// # Errors
        ///
        /// Whatever `kevent` reported.
        pub fn add(&self, fd: RawFd, token: u64, interest: Interest) -> io::Result<()> {
            self.apply(fd, token, interest)
        }

        /// Change what `fd` is watched for.
        ///
        /// The same call as [`Poller::add`], because `EV_ADD` on a filter that is already there updates it rather than complaining, which is the one place kqueue is kinder than epoll.
        ///
        /// # Errors
        ///
        /// Whatever `kevent` reported.
        pub fn modify(&self, fd: RawFd, token: u64, interest: Interest) -> io::Result<()> {
            self.apply(fd, token, interest)
        }

        /// Stop watching `fd`.
        ///
        /// # Errors
        ///
        /// Nothing. A filter that was never added reports `ENOENT`, which is the state being asked for, so both filters are deleted and the answer is ignored.
        pub fn remove(&self, fd: RawFd) -> io::Result<()> {
            let changes = [
                change(fd, libc::EVFILT_READ, libc::EV_DELETE, 0),
                change(fd, libc::EVFILT_WRITE, libc::EV_DELETE, 0),
            ];
            // SAFETY: the change list is a live local array of the length being passed, and no event list is asked for.
            unsafe {
                libc::kevent(
                    self.fd,
                    changes.as_ptr(),
                    2,
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null(),
                );
            }
            Ok(())
        }

        /// Install both filters, enabling the ones asked for and disabling the others.
        ///
        /// Disabling rather than deleting, because a connection alternates between wanting a write and not wanting one on every reply that does not fit in a socket buffer, and a filter that is disabled and re-enabled costs the kernel nothing to keep.
        fn apply(&self, fd: RawFd, token: u64, interest: Interest) -> io::Result<()> {
            let changes = [
                change(
                    fd,
                    libc::EVFILT_READ,
                    libc::EV_ADD | flag(interest.read),
                    token,
                ),
                change(
                    fd,
                    libc::EVFILT_WRITE,
                    libc::EV_ADD | flag(interest.write),
                    token,
                ),
            ];
            // SAFETY: the change list is a live local array of the length being passed, and no event list is asked for.
            let rc = unsafe {
                libc::kevent(
                    self.fd,
                    changes.as_ptr(),
                    2,
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null(),
                )
            };
            if failed(rc) { Err(last()) } else { Ok(()) }
        }

        /// Wait for something to happen, for at most `timeout`.
        ///
        /// # Errors
        ///
        /// Whatever `kevent` reported, except an interruption by a signal, which comes back as no events rather than as an error.
        pub fn wait(&mut self, timeout: Option<Duration>) -> io::Result<&[Ready]> {
            let spec = timeout.map(|d| libc::timespec {
                tv_sec: libc::time_t::try_from(d.as_secs()).unwrap_or(libc::time_t::MAX),
                // Infallible because `c_long` is sixty-four bits on every platform that takes this branch. A thirty-two bit BSD would need a conversion here, and would not compile until it got one, which is the right way round.
                tv_nsec: libc::c_long::from(d.subsec_nanos()),
            });
            let deadline = spec.as_ref().map_or(std::ptr::null(), std::ptr::from_ref);

            // SAFETY: the event list is this vector's own pointer and length, the timeout is a live local or null, and nothing here is touched until the call returns.
            let count = unsafe {
                libc::kevent(
                    self.fd,
                    std::ptr::null(),
                    0,
                    self.events.as_mut_ptr(),
                    i32::try_from(self.events.len()).unwrap_or(i32::MAX),
                    deadline,
                )
            };
            if failed(count) {
                let err = last();
                if err.kind() == io::ErrorKind::Interrupted {
                    self.ready.clear();
                    return Ok(&self.ready);
                }
                return Err(err);
            }

            // Two filters on one socket are two events, so the same token can appear twice in one wait. They are reported as they came rather than merged, because a caller that reads and then writes does the right thing either way, and merging would cost a lookup per event to save one.
            self.ready.clear();
            for event in &self.events[..many(count)] {
                self.ready.push(Ready {
                    token: event.udata as u64,
                    read: event.filter == libc::EVFILT_READ,
                    write: event.filter == libc::EVFILT_WRITE,
                    gone: event.flags & (libc::EV_EOF | libc::EV_ERROR) != 0,
                });
            }
            Ok(&self.ready)
        }
    }

    /// An all-zero event, for filling the list the kernel writes into.
    fn blank() -> libc::kevent {
        change(0, 0, 0, 0)
    }

    /// One change list entry.
    fn change(fd: RawFd, filter: i16, flags: u16, token: u64) -> libc::kevent {
        libc::kevent {
            // A descriptor is never negative, so reading its bits as unsigned is the identity rather than a reinterpretation.
            ident: fd.cast_unsigned() as libc::uintptr_t,
            filter,
            flags,
            fflags: 0,
            data: 0,
            udata: token as *mut libc::c_void,
        }
    }

    /// Whether a filter is enabled or merely present.
    const fn flag(wanted: bool) -> u16 {
        if wanted {
            libc::EV_ENABLE
        } else {
            libc::EV_DISABLE
        }
    }

    impl Drop for Poller {
        fn drop(&mut self) {
            // SAFETY: the descriptor is ours, was made by kqueue, and is not closed anywhere else.
            unsafe { libc::close(self.fd) };
        }
    }
}

pub use imp::Poller;

/// Ask the kernel to let several sockets bind the same address, so every thread can have its own accept queue.
///
/// Linux spreads new connections across the queues itself, which removes the accept thread and the handoff that goes with it. Everywhere else this is either absent or means something different, so there it is not asked for and one listener is shared.
///
/// # Errors
///
/// Whatever `setsockopt` reported.
#[cfg(target_os = "linux")]
pub fn share_port(fd: RawFd) -> io::Result<()> {
    let on: libc::c_int = 1;
    // A `c_int` is four bytes everywhere this builds, so the conversion cannot fail. It is written as a fallback rather than an unwrap because an unwrap in a startup path is still a panic, and the lints here deny one.
    let len = libc::socklen_t::try_from(size_of_val(&on)).unwrap_or(4);
    // SAFETY: the option value is a live local of exactly the length being passed, which is what SO_REUSEPORT expects.
    let rc = unsafe {
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_REUSEPORT,
            (&raw const on).cast(),
            len,
        )
    };
    if failed(rc) { Err(last()) } else { Ok(()) }
}

/// Nothing, on a platform where one accept queue is all there is.
///
/// # Errors
///
/// Never.
#[cfg(not(target_os = "linux"))]
pub fn share_port(_fd: RawFd) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::os::fd::AsRawFd;
    use std::time::Duration;

    #[test]
    fn a_poller_reports_a_socket_that_has_something_to_read() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        server.set_nonblocking(true).unwrap();

        let mut poller = Poller::new().unwrap();
        poller.add(server.as_raw_fd(), 7, Interest::READ).unwrap();

        // Nothing has been sent, so a wait with a deadline comes back with nothing rather than hanging.
        assert!(
            poller
                .wait(Some(Duration::from_millis(50)))
                .unwrap()
                .is_empty()
        );

        client.write_all(b"hello").unwrap();
        let ready = poller.wait(Some(Duration::from_secs(5))).unwrap();
        assert_eq!(ready.len(), 1, "one socket was written to");
        assert_eq!(ready[0].token, 7, "the token came back changed");
        assert!(
            ready[0].read,
            "the socket has bytes and was not reported readable"
        );

        let mut buf = [0u8; 5];
        server.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"hello");
    }

    #[test]
    fn a_closed_peer_is_reported_as_gone() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        server.set_nonblocking(true).unwrap();

        let mut poller = Poller::new().unwrap();
        poller.add(server.as_raw_fd(), 1, Interest::READ).unwrap();
        drop(client);

        let ready = poller.wait(Some(Duration::from_secs(5))).unwrap();
        assert!(!ready.is_empty(), "a closed peer should wake the poller");
        // Readable rather than gone is a correct answer too: a read of zero bytes is how a stream ends, and the caller has to handle that either way.
        assert!(
            ready[0].gone || ready[0].read,
            "a closed peer was reported as neither gone nor readable"
        );
    }

    #[test]
    fn a_removed_socket_stops_waking_it() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        server.set_nonblocking(true).unwrap();

        let mut poller = Poller::new().unwrap();
        poller.add(server.as_raw_fd(), 1, Interest::READ).unwrap();
        poller.remove(server.as_raw_fd()).unwrap();

        client.write_all(b"hello").unwrap();
        assert!(
            poller
                .wait(Some(Duration::from_millis(100)))
                .unwrap()
                .is_empty(),
            "a socket that was removed woke the poller anyway"
        );
    }

    #[test]
    fn a_listener_is_watched_the_same_way_a_connection_is() {
        // The accept path, which is the one that has to work before anything else does.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();

        let mut poller = Poller::new().unwrap();
        poller
            .add(listener.as_raw_fd(), u64::MAX, Interest::READ)
            .unwrap();

        let _client = TcpStream::connect(addr).unwrap();
        let ready = poller.wait(Some(Duration::from_secs(5))).unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(
            ready[0].token,
            u64::MAX,
            "a token with every bit set came back wrong"
        );
        assert!(listener.accept().is_ok());
    }

    #[test]
    fn asking_for_a_write_reports_one_immediately() {
        // An empty socket buffer is always writable, so this is the check that the write filter is installed and enabled rather than merely present.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let _client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        server.set_nonblocking(true).unwrap();

        let mut poller = Poller::new().unwrap();
        poller.add(server.as_raw_fd(), 3, Interest::BOTH).unwrap();

        let ready = poller.wait(Some(Duration::from_secs(5))).unwrap();
        assert!(
            ready.iter().any(|event| event.write),
            "an idle socket should be writable"
        );
    }
}
