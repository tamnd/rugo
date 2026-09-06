//! The server: listeners, threads, and the loop between them.
//!
//! # Shape
//!
//! One thread a core, each with its own poller and its own connections, all sharing one [`rugo_map::Map`] and one set of listeners. A connection belongs to whichever thread accepted it for as long as it lives, so nothing about a connection is shared and nothing about it needs a lock. The only thing threads contend for is a shard of the map, and there are thousands of those.
//!
//! # The listener
//!
//! One listening socket, registered in every thread's poller. When a connection arrives every thread wakes and one of them wins the accept; the rest get `WouldBlock` and go back to sleep. That is a thundering herd, and it is the right trade here: it costs a wakeup a connection, and a benchmark opens its connections once and then sends a hundred million commands down them.
//!
//! `SO_REUSEPORT` with a listener a thread would remove the herd and is what [`rugo_net::share_port`] is there for. It is not done yet because it would buy nothing measurable and would cost a real complication: the kernel's balancing is by hash rather than by load, and a bad split is worse than a wakeup.
//!
//! # The loop
//!
//! Level-triggered, one read a readiness event, no timers except a poll timeout that gives the clock a tick and the map a sweep. Everything a connection does is in `conn`, and everything a command does is in `dispatch`.
//!
//! On Linux there is a second loop, in `uring`, which answers the same commands out of the same connections and differs only in how bytes get in and out. Which one runs is decided once for the whole process, by `--uring`, so a thread cannot be serving on a ring while its neighbour serves on a poller.

pub mod config;
mod conn;
mod dispatch;
mod stats;
#[cfg(target_os = "linux")]
mod uring;

use std::io;
use std::net::{Ipv4Addr, TcpListener};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::UnixListener;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rugo_map::Map;
use rugo_net::{Interest, Poller, Ready};

pub use config::{Asked, Config, USAGE, Uring};
pub use stats::{Stats, Total};

use conn::{Conn, Stream, Turn};
use dispatch::Env;

/// How long a poll waits when nothing is happening.
///
/// Short enough that the clock a hundred milliseconds of expiry is measured against is never far wrong, and long enough that an idle server with thirty-two threads wakes three hundred times a second in total rather than thirty thousand.
const IDLE: Duration = Duration::from_millis(100);

/// How many slots a background sweep looks at each idle turn.
///
/// Expiry is already checked on read, so this is only what reclaims a key nobody asks for again. Small, because it takes a shard's lock to do it.
const SWEEP: usize = 256;

/// A listening socket, whichever kind it is.
#[derive(Debug)]
enum Listener {
    /// A TCP listener.
    Tcp(TcpListener),
    /// A unix socket listener.
    Unix(UnixListener),
}

impl Listener {
    /// Take a connection, if one is waiting.
    fn accept(&self) -> io::Result<Option<Stream>> {
        let taken = match self {
            Self::Tcp(listener) => listener.accept().map(|(stream, _)| Stream::Tcp(stream)),
            Self::Unix(listener) => listener.accept().map(|(stream, _)| Stream::Unix(stream)),
        };
        match taken {
            Ok(stream) => Ok(Some(stream)),
            // Another thread won the race for this connection, which is ordinary and not worth reporting.
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(error) => Err(error),
        }
    }
}

impl AsRawFd for Listener {
    fn as_raw_fd(&self) -> RawFd {
        match self {
            Self::Tcp(listener) => listener.as_raw_fd(),
            Self::Unix(listener) => listener.as_raw_fd(),
        }
    }
}

/// A running server.
#[derive(Debug)]
pub struct Server {
    /// How it was configured.
    config: Config,
    /// The cache.
    map: Arc<Map>,
    /// What it counts.
    stats: Arc<Stats>,
    /// The sockets it listens on.
    listeners: Arc<Vec<Listener>>,
    /// When it started, for `INFO`.
    started: Instant,
}

impl Server {
    /// Bind everything `config` asks for, without serving any of it yet.
    ///
    /// Binding here rather than inside the threads is what makes a port already in use an error the caller can report and exit on, rather than a thread that dies quietly behind a process that looks healthy.
    ///
    /// # Errors
    ///
    /// Whatever binding failed with, which is almost always an address in use or a socket path that cannot be written.
    pub fn new(config: Config) -> io::Result<Self> {
        let mut listeners = Vec::new();

        if let Some(port) = config.port {
            // Loopback and every interface, in one socket: `Ipv6Addr::UNSPECIFIED` would need the dual-stack option set and would still fail where IPv6 is off, and a benchmark that could not connect over IPv4 is a benchmark that does not run.
            let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, port))?;
            listener.set_nonblocking(true)?;
            listeners.push(Listener::Tcp(listener));
        }

        if let Some(path) = &config.unixsocket {
            // A socket left behind by a process that was killed rather than stopped, which is what a benchmark harness does to every server it runs. Removing it is the only way to bind the same path twice, and refusing to would make the second run of a sweep fail.
            let _ = std::fs::remove_file(path);
            let listener = UnixListener::bind(path)?;
            listener.set_nonblocking(true)?;
            listeners.push(Listener::Unix(listener));
        }

        Ok(Self {
            map: Arc::new(Map::new(config.shards, config.maxmemory)),
            stats: Arc::new(Stats::new(config.threads)),
            listeners: Arc::new(listeners),
            started: Instant::now(),
            config,
        })
    }

    /// The address the TCP listener actually got.
    ///
    /// Not the configured port, because a port of nought is a request for whichever one is free, and a test that wants to connect has to be told which that was.
    ///
    /// # Errors
    ///
    /// Whatever asking the socket failed with.
    pub fn port(&self) -> io::Result<Option<u16>> {
        for listener in self.listeners.iter() {
            if let Listener::Tcp(listener) = listener {
                return listener.local_addr().map(|at| Some(at.port()));
            }
        }
        Ok(None)
    }

    /// What the server has counted so far.
    #[must_use]
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Serve until the process is stopped.
    ///
    /// Starts every thread but the first, then serves on the calling thread, so a server with one thread makes no threads at all and a `SIGTERM` arrives at a process that is doing the work rather than waiting on a join.
    ///
    /// # Errors
    ///
    /// Whatever the calling thread's loop failed with, which is a poller or a ring that could not be created or a syscall that failed in a way this cannot continue past, and `--uring yes` on a machine that has no `io_uring` to give.
    pub fn run(&self) -> io::Result<()> {
        // Asked once rather than a thread. Every thread would get the same answer, `--uring yes` on a kernel without one should be one refusal at startup rather than as many as there are cores, and a process half of whose threads are on a ring is a process whose numbers mean nothing.
        let uring = self.uring()?;

        std::thread::scope(|scope| {
            for thread in 1..self.config.threads {
                let worker = Worker {
                    thread,
                    map: &self.map,
                    stats: &self.stats,
                    listeners: &self.listeners,
                    config: &self.config,
                    started: self.started,
                };
                std::thread::Builder::new()
                    .name(format!("rugo-{thread}"))
                    .spawn_scoped(scope, move || {
                        if let Err(error) = worker.serve(uring) {
                            // One thread failing is not the others failing, and a server that went on serving on the rest without saying so would be a server quietly running at a fraction of its threads.
                            eprintln!("rugo: thread {thread} stopped: {error}");
                        }
                    })?;
            }

            Worker {
                thread: 0,
                map: &self.map,
                stats: &self.stats,
                listeners: &self.listeners,
                config: &self.config,
                started: self.started,
            }
            .serve(uring)
        })
    }

    /// Whether this run serves on a ring.
    ///
    /// `auto` asks the kernel by building a ring and throwing it away, which is the only honest way to ask: a version number says what was compiled, and what matters is whether the syscall is allowed, which a container's seccomp profile has the last word on.
    #[cfg(target_os = "linux")]
    fn uring(&self) -> io::Result<bool> {
        match self.config.uring {
            Uring::No => Ok(false),
            Uring::Auto => Ok(rugo_net::uring::Ring::available()),
            Uring::Yes => match rugo_net::uring::Ring::new(8) {
                Ok(_) => Ok(true),
                Err(error) => Err(io::Error::new(
                    error.kind(),
                    format!(
                        "--uring yes was asked for and this kernel would not give one: {error}"
                    ),
                )),
            },
        }
    }

    /// Whether this run serves on a ring, where there are no rings.
    #[cfg(not(target_os = "linux"))]
    fn uring(&self) -> io::Result<bool> {
        match self.config.uring {
            Uring::No | Uring::Auto => Ok(false),
            Uring::Yes => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "--uring yes was asked for and io_uring is Linux only",
            )),
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        // The socket file outlives the process that made it, and a stale one is what the next run has to clear before it can bind. Removing it here means only a killed process leaves one behind.
        if let Some(path) = &self.config.unixsocket {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// One serving thread.
#[derive(Debug)]
struct Worker<'a> {
    /// Which thread this is, and so which counters are its own.
    thread: usize,
    /// The shared cache.
    map: &'a Map,
    /// Every thread's counters.
    stats: &'a Stats,
    /// The shared listeners.
    listeners: &'a [Listener],
    /// How the server was configured.
    config: &'a Config,
    /// When the server started.
    started: Instant,
}

impl Worker<'_> {
    /// Serve until something fails, on whichever loop the process chose.
    fn serve(&self, uring: bool) -> io::Result<()> {
        let env = Env {
            map: self.map,
            stats: self.stats,
            thread: self.thread,
            started: self.started,
            config: self.config,
        };

        #[cfg(target_os = "linux")]
        if uring {
            return self.serve_uring(&env);
        }
        #[cfg(not(target_os = "linux"))]
        debug_assert!(!uring, "there are no rings on this target");

        self.serve_poller(&env)
    }

    /// Serve on a poller until something fails.
    fn serve_poller(&self, env: &Env<'_>) -> io::Result<()> {
        let mut poller = Poller::new()?;
        for (at, listener) in self.listeners.iter().enumerate() {
            poller.add(
                listener.as_raw_fd(),
                u64::try_from(at).unwrap_or(u64::MAX),
                Interest::READ,
            )?;
        }

        // Indexed by the token a connection was registered under, minus the listeners' share of the numbering. A vector with holes rather than a map, because a token is already an index and looking one up should not be a hash.
        let mut conns: Vec<Option<Conn>> = Vec::new();
        let mut free: Vec<usize> = Vec::new();
        // The events, copied out of the poller so that handling one may register another.
        let mut ready: Vec<Ready> = Vec::new();

        loop {
            ready.clear();
            ready.extend_from_slice(poller.wait(Some(IDLE))?);

            for event in &ready {
                // The tokens are this loop's own doing: a listener's index, then a connection's slot. Nothing on the wire chooses one, so a token that does not fit a machine word is a bug here rather than something a client did.
                let Ok(token) = usize::try_from(event.token) else {
                    continue;
                };
                if token < self.listeners.len() {
                    self.take(&self.listeners[token], &poller, &mut conns, &mut free)?;
                    continue;
                }

                let at = token - self.listeners.len();
                let Some(Some(conn)) = conns.get_mut(at) else {
                    continue;
                };

                let turn = if event.gone {
                    Turn::Close
                } else {
                    conn.turn(env, event.read, event.write)
                };
                match turn {
                    Turn::Read | Turn::Write => {
                        let want = Interest {
                            read: true,
                            write: turn == Turn::Write,
                        };
                        // Registered only when it changed. Both pollers are level triggered, so a registration stays in force until something changes it, and a connection reading commands and answering them wants next what it wanted last. Re-registering anyway cost a syscall per batch to restate a fact the kernel already held.
                        if conn.armed() != want {
                            poller.modify(conn.fd(), event.token, want)?;
                            conn.arm(want);
                        }
                    }
                    Turn::Close => {
                        let _ = poller.remove(conn.fd());
                        conns[at] = None;
                        free.push(at);
                    }
                }
            }

            // Whether or not anything happened. The clock is what every expiry is measured against, and a thread that only ticked it when it was busy would leave an idle server's keys immortal.
            self.map.clock().tick();
            self.map.sweep(SWEEP);
        }
    }

    /// Take one connection from `listener`, if there is one.
    ///
    /// One rather than every. Accepting until the listener blocks looks like the thrifty choice — every thread was woken for this and the ones that get nothing paid for the wakeup anyway — but it is what decides how the whole server is loaded, because a connection stays on the thread that accepted it for its whole life. A benchmark opens its connections in a burst, so the listener is readable with a backlog behind it, and the first thread to wake takes the lot. The rest get a handful each and then idle, and the server runs at a fraction of the cores it was given no matter how many threads it was asked for.
    ///
    /// Taking one apiece spreads the burst across whoever is in the poller to receive it, which is what "whoever is free takes the connection" was supposed to mean. The listener is level triggered, so a backlog wakes everybody again immediately and nothing is left waiting. The extra wakeups are paid once, while connections are being established, and the balance they buy is paid back on every command afterwards.
    fn take(
        &self,
        listener: &Listener,
        poller: &Poller,
        conns: &mut Vec<Option<Conn>>,
        free: &mut Vec<usize>,
    ) -> io::Result<()> {
        // A `while let` here is what took the whole backlog onto one thread. The loop remains only so that a socket which cannot be prepared does not cost this thread its turn.
        while let Some(stream) = listener.accept()? {
            if stream.prepare().is_err() {
                continue;
            }
            let conn = Conn::new(stream);
            let fd = conn.fd();
            // What the connection already believes it is registered for, rather than a second copy of that constant here. A connection whose first turn wants this makes no `epoll_ctl` at all, and the two cannot drift into disagreeing.
            let interest = conn.armed();

            let at = if let Some(at) = free.pop() {
                conns[at] = Some(conn);
                at
            } else {
                conns.push(Some(conn));
                conns.len() - 1
            };

            let token = u64::try_from(at + self.listeners.len()).unwrap_or(u64::MAX);
            if let Err(error) = poller.add(fd, token, interest) {
                conns[at] = None;
                free.push(at);
                return Err(error);
            }
            self.stats.thread(self.thread).connection();
            return Ok(());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;
    use std::os::unix::net::UnixStream;

    use super::*;

    /// A server on a port the operating system chose, serving on the poller.
    ///
    /// The loop is named rather than left at the default, because the default is whichever loop the machine can run and a test that answers a different question on Linux than it does on macOS is a test whose passing says less than it looks like it says.
    fn serving(threads: usize) -> u16 {
        serving_with(threads, Uring::No)
    }

    /// A server on a port the operating system chose, and the port it chose.
    ///
    /// Left running for the rest of the process: there is no stop, because nothing in the serving path checks for one, and a test that wants a fresh server asks for another port rather than reclaiming this one.
    fn serving_with(threads: usize, uring: Uring) -> u16 {
        let config = Config {
            threads,
            // Nought is a request for whichever port is free, which is what lets these tests run beside each other and beside anything else on the machine.
            port: Some(0),
            unixsocket: None,
            uring,
            ..Config::default()
        };
        let server = Server::new(config).expect("the server bound");
        let port = server.port().expect("a local address").expect("a port");
        std::thread::spawn(move || {
            let _ = server.run();
        });

        // Wait for it to be answering rather than merely bound, which is what the harness's readiness probe does and for the same reason.
        for _ in 0..200 {
            if let Ok(mut client) = TcpStream::connect((Ipv4Addr::LOCALHOST, port))
                && client.write_all(b"PING\r\n").is_ok()
            {
                let mut line = String::new();
                if BufReader::new(&mut client).read_line(&mut line).is_ok() && line == "+PONG\r\n" {
                    return port;
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("the server never answered on port {port}");
    }

    /// Send `input` to a server and read `lines` lines back.
    fn talk(port: u16, input: &[u8], lines: usize) -> Vec<String> {
        let mut client = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connected");
        client.set_nodelay(true).expect("nodelay");
        client.write_all(input).expect("wrote");
        let mut reader = BufReader::new(client);
        (0..lines)
            .map(|_| {
                let mut line = String::new();
                reader.read_line(&mut line).expect("read a line");
                line
            })
            .collect()
    }

    // What decides whether the server uses the cores it was given.
    //
    // A connection lives on the thread that accepted it, so how a burst of them is divided up is how the load is divided up for as long as they last. Accepting until the listener blocked gave the whole burst to whichever thread woke first, and measured on `gpc` at sixteen threads that left four threads carrying the work, five of them at three per cent of a core, and the process using four and a half of its sixteen. The contract that prevents it is this one, and it is worth stating where it cannot drift: one call, one connection.
    #[test]
    fn a_turn_at_the_listener_takes_one_connection_and_leaves_the_rest() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bound");
        listener.set_nonblocking(true).expect("non-blocking");
        let port = listener.local_addr().expect("an address").port();
        let listeners = vec![Listener::Tcp(listener)];

        let config = Config {
            threads: 1,
            port: None,
            unixsocket: None,
            ..Config::default()
        };
        let map = Map::new(config.shards, config.maxmemory);
        let stats = Stats::new(1);
        let worker = Worker {
            thread: 0,
            map: &map,
            stats: &stats,
            listeners: &listeners,
            config: &config,
            started: Instant::now(),
        };

        // Opened in a burst and held, which is what a benchmark does and what the old code divided up badly.
        let held: Vec<TcpStream> = (0..3)
            .map(|_| TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connected"))
            .collect();

        let poller = Poller::new().expect("a poller");
        let mut conns: Vec<Option<Conn>> = Vec::new();
        let mut free: Vec<usize> = Vec::new();
        let taken = |conns: &Vec<Option<Conn>>| conns.iter().filter(|slot| slot.is_some()).count();

        // Called until all three have been taken rather than a fixed number of times, because how soon a connect reaches the accept queue is the kernel's business and not this test's claim. The claim is on the step size: no single call may take more than one, whenever they arrive.
        for _ in 0..500 {
            let before = taken(&conns);
            worker
                .take(&listeners[0], &poller, &mut conns, &mut free)
                .expect("accepted");
            let after = taken(&conns);
            assert!(
                after - before <= 1,
                "one call took {} connections, which is the burst landing on one thread",
                after - before
            );
            if after == held.len() {
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("three connections never arrived, so nothing was proved either way");
    }

    #[test]
    fn a_client_can_set_and_get_over_tcp() {
        let port = serving(1);
        let back = talk(
            port,
            b"*3\r\n$3\r\nSET\r\n$5\r\nhello\r\n$5\r\nworld\r\n*2\r\n$3\r\nGET\r\n$5\r\nhello\r\n",
            3,
        );
        assert_eq!(back, ["+OK\r\n", "$5\r\n", "world\r\n"]);
    }

    // The completion loop doing what the readiness loop does, on the shape that exercises the most of it: several threads, a burst of connections between them, and a key written through one and read through another.
    //
    // It skips rather than fails where there is no ring, because a container's seccomp profile has the last word on whether the syscall is allowed and a test that fails in one is a test somebody deletes.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_ring_serves_what_a_poller_serves() {
        if !rugo_net::uring::Ring::available() {
            return;
        }
        let port = serving_with(4, Uring::Yes);
        for n in 0..32 {
            let set = format!("*3\r\n$3\r\nSET\r\n$4\r\nk{n:03}\r\n$4\r\nv{n:03}\r\n");
            assert_eq!(talk(port, set.as_bytes(), 1), ["+OK\r\n"]);
        }
        for n in 0..32 {
            let get = format!("*2\r\n$3\r\nGET\r\n$4\r\nk{n:03}\r\n");
            assert_eq!(
                talk(port, get.as_bytes(), 2),
                ["$4\r\n".to_owned(), format!("v{n:03}\r\n")],
                "key k{n:03} did not come back from the ring"
            );
        }

        // A pipeline in one write, which is what the sweep sends and what the loop answers with one send.
        assert_eq!(
            talk(
                port,
                b"*1\r\n$4\r\nPING\r\n*2\r\n$3\r\nGET\r\n$4\r\nk000\r\n*1\r\n$4\r\nPING\r\n",
                4
            ),
            ["+PONG\r\n", "$4\r\n", "v000\r\n", "+PONG\r\n"]
        );
    }

    #[test]
    fn several_threads_share_one_map() {
        // The claim that makes the whole shape worth having: a key written on whichever thread took one connection is readable on whichever took another.
        let port = serving(4);
        for n in 0..32 {
            let set = format!("*3\r\n$3\r\nSET\r\n$4\r\nk{n:03}\r\n$4\r\nv{n:03}\r\n");
            assert_eq!(talk(port, set.as_bytes(), 1), ["+OK\r\n"]);
        }
        for n in 0..32 {
            let get = format!("*2\r\n$3\r\nGET\r\n$4\r\nk{n:03}\r\n");
            assert_eq!(
                talk(port, get.as_bytes(), 2),
                ["$4\r\n".to_owned(), format!("v{n:03}\r\n")],
                "key k{n:03} was written on one thread and lost on another"
            );
        }
    }

    #[test]
    fn a_unix_socket_serves_the_same_thing() {
        let path = std::env::temp_dir().join(format!("rugo-test-{}.sock", std::process::id()));
        let config = Config {
            threads: 1,
            port: None,
            unixsocket: Some(path.display().to_string()),
            ..Config::default()
        };
        let server = Server::new(config).expect("the server bound");
        std::thread::spawn(move || {
            let _ = server.run();
        });

        let mut client = None;
        for _ in 0..200 {
            if let Ok(stream) = UnixStream::connect(&path) {
                client = Some(stream);
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let mut client = client.expect("the socket never appeared");
        client.write_all(b"PING\r\n").expect("wrote");
        let mut line = String::new();
        BufReader::new(&mut client)
            .read_line(&mut line)
            .expect("read");
        assert_eq!(line, "+PONG\r\n");
    }

    #[test]
    fn a_port_already_taken_is_an_error_rather_than_a_panic() {
        let port = serving(1);
        let config = Config {
            port: Some(port),
            ..Config::default()
        };
        assert!(
            Server::new(config).is_err(),
            "two servers bound the same port"
        );
    }
}
