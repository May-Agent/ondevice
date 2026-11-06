//! The store: a cache directory of installed pins.
//!
//! Layout under the root, for a pin whose digest is `<sha256>`:
//!
//! - `<sha256>`: the installed file or directory. It appears through one
//!   rename of fully written, verified contents, so its presence alone means
//!   the pin is installed.
//! - `<sha256>.lock`: the lock concurrent installs of the pin serialize on,
//!   across threads and processes.
//! - `.<sha256>.*`: the pin's download and staging entries. A killed install
//!   leaves them behind; the next install of the pin removes them while it
//!   holds the lock, so it never touches another pin's entries in flight.

use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_lite::{AsyncWriteExt as _, StreamExt as _, future};
use sha2::Digest as _;
use std::pin::pin;

use tempfile::NamedTempFile;

use crate::{Pin, Response, Sha256, Transport};

/// Deadline for the response head: connecting, redirects and the status
/// line, before any body byte.
const RESPONSE_DEADLINE: Duration = Duration::from_secs(30);

/// Deadline between received body chunks. A stalled transfer fails; a slow
/// one keeps running however long the whole download takes.
const IDLE_DEADLINE: Duration = Duration::from_secs(60);

/// Download progress of a pin, reported after each received chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// Bytes received so far.
    pub received: u64,
    /// The pin's size.
    pub total: u64,
}

/// What a pin installs as: [`File`](crate::File) or, with the `archive`
/// feature, [`Archive`](crate::Archive). Sealed; the store owns the install
/// protocol.
pub trait Content: sealed::Install + Sync {}

impl Content for crate::File {}

#[cfg(feature = "archive")]
impl Content for crate::Archive {}

mod sealed {
    use std::io;
    use std::path::Path;

    use tempfile::NamedTempFile;

    pub trait Install {
        /// Installs the verified bytes in `staged` at `dest`, staging any
        /// intermediate entry in `root` under `prefix`. Runs on the blocking
        /// pool with the pin's lock held, so `dest` is absent.
        fn install(
            staged: NamedTempFile,
            root: &Path,
            prefix: &str,
            dest: &Path,
        ) -> Result<(), Failed>;
    }

    /// An install failure, before it becomes an [`Error`](super::Error).
    #[derive(Debug)]
    pub enum Failed {
        /// See [`Error::Archive`](super::Error::Archive).
        #[cfg(feature = "archive")]
        Archive {
            entry: std::path::PathBuf,
            reason: &'static str,
        },
        Io(io::Error),
    }

    impl From<io::Error> for Failed {
        fn from(error: io::Error) -> Self {
            Self::Io(error)
        }
    }
}

use sealed::Failed;

impl<E> From<Failed> for Error<E> {
    fn from(failed: Failed) -> Self {
        match failed {
            #[cfg(feature = "archive")]
            Failed::Archive { entry, reason } => Self::Archive { entry, reason },
            Failed::Io(error) => Self::Io(error),
        }
    }
}

impl sealed::Install for crate::File {
    fn install(
        staged: NamedTempFile,
        _root: &Path,
        _prefix: &str,
        dest: &Path,
    ) -> Result<(), Failed> {
        staged.as_file().sync_all()?;
        staged.persist(dest).map_err(|error| error.error)?;
        Ok(())
    }
}

#[cfg(feature = "archive")]
impl sealed::Install for crate::Archive {
    fn install(
        staged: NamedTempFile,
        root: &Path,
        prefix: &str,
        dest: &Path,
    ) -> Result<(), Failed> {
        let staging = tempfile::Builder::new().prefix(prefix).tempdir_in(root)?;
        unpack(staged.as_file(), staging.path())?;
        std::fs::rename(staging.path(), dest)?;
        // The directory now lives at `dest`; dropping the guard would delete
        // whatever later takes the staging name.
        let _ = staging.keep();
        Ok(())
    }
}

/// A cache directory of installed pins. One store serves any number of
/// pins; each installs to its own entry.
#[derive(Debug, Clone)]
pub struct Store<T> {
    root: PathBuf,
    transport: T,
}

impl<T: Transport> Store<T> {
    /// A store rooted at `root`, created on the first install, downloading
    /// through `transport`.
    pub fn new(root: impl Into<PathBuf>, transport: T) -> Self {
        Self {
            root: root.into(),
            transport,
        }
    }

    /// The cache directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where `pin` is installed: `<root>/<sha256>`.
    #[must_use]
    pub fn path<K>(&self, pin: &Pin<'_, K>) -> PathBuf {
        self.root.join(pin.sha256.to_string())
    }

    /// The installed `pin`, downloading it on first use.
    ///
    /// A present entry is returned at once. Otherwise the caller takes the
    /// pin's lock; whoever holds it re-checks the entry, removes what killed
    /// installs left, downloads, verifies the length and the sha256, and
    /// installs. Callers that waited on the lock then find the entry.
    /// `progress` is called after each received chunk.
    ///
    /// Dropping the future abandons the install: its download entries are
    /// removed or, at worst, left for the next install to remove.
    ///
    /// # Errors
    ///
    /// [`Error`]: the transfer failed, the bytes do not match the pin, the
    /// archive is malformed, or the cache directory is not writable.
    pub async fn ensure<K: Content>(
        &self,
        pin: &Pin<'_, K>,
        mut progress: impl FnMut(Progress) + Send,
    ) -> Result<PathBuf, Error<T::Error>> {
        let dest = self.path(pin);
        let probe = dest.clone();
        if blocking::unblock(move || probe.try_exists()).await? {
            return Ok(dest);
        }
        let prefix = format!(".{}.", pin.sha256);
        let Some(lock) = self.lock(pin.sha256, &prefix, &dest).await? else {
            return Ok(dest);
        };
        let staged = download(&self.transport, pin, &self.root, &prefix, &mut progress).await?;
        let root = self.root.clone();
        let target = dest.clone();
        blocking::unblock(move || {
            let installed = K::install(staged, &root, &prefix, &target);
            drop(lock);
            installed
        })
        .await?;
        tracing::info!(path = %dest.display(), "pin installed");
        Ok(dest)
    }

    /// Takes the pin's lock on the blocking pool. `None` when `dest`
    /// appeared while waiting; otherwise the pin's leftovers are gone and
    /// the returned guard holds the lock until dropped.
    async fn lock(
        &self,
        sha256: Sha256,
        prefix: &str,
        dest: &Path,
    ) -> io::Result<Option<std::fs::File>> {
        let root = self.root.clone();
        let prefix = prefix.to_owned();
        let dest = dest.to_path_buf();
        blocking::unblock(move || {
            std::fs::create_dir_all(&root)?;
            let lock = std::fs::File::create(root.join(format!("{sha256}.lock")))?;
            lock.lock()?;
            if dest.try_exists()? {
                return Ok(None);
            }
            sweep(&root, &prefix)?;
            Ok(Some(lock))
        })
        .await
    }
}

/// Streams `pin` into a temp file under `root` named with `prefix`,
/// returning it once its length and sha256 match the pin. On failure the
/// temp file is removed on the blocking pool.
async fn download<T: Transport, K>(
    transport: &T,
    pin: &Pin<'_, K>,
    root: &Path,
    prefix: &str,
    progress: &mut (impl FnMut(Progress) + Send),
) -> Result<NamedTempFile, Error<T::Error>> {
    let url = pin.url;
    tracing::info!(url, size = pin.size, "downloading pin");
    let Response { length, body } = within(RESPONSE_DEADLINE, transport.get(url))
        .await
        .ok_or_else(|| Error::Timeout {
            url: url.to_owned(),
            after: RESPONSE_DEADLINE,
        })?
        .map_err(|source| Error::Request {
            url: url.to_owned(),
            source,
        })?;
    if let Some(declared) = length
        && declared != pin.size
    {
        return Err(Error::Size {
            url: url.to_owned(),
            expected: pin.size,
            actual: declared,
        });
    }
    let mut body = pin!(body);
    let root = root.to_path_buf();
    let prefix = prefix.to_owned();
    let (temp, writer) = blocking::unblock(move || {
        let temp = tempfile::Builder::new()
            .prefix(&prefix)
            .tempfile_in(&root)?;
        let writer = temp.reopen()?;
        io::Result::Ok((temp, writer))
    })
    .await?;
    let mut writer = async_fs::File::from(writer);
    let mut hasher = sha2::Sha256::new();
    let mut received = 0u64;
    let streamed: Result<(), Error<T::Error>> = async {
        loop {
            let chunk = match within(IDLE_DEADLINE, body.next()).await {
                None => {
                    return Err(Error::Timeout {
                        url: url.to_owned(),
                        after: IDLE_DEADLINE,
                    });
                }
                Some(None) => break,
                Some(Some(chunk)) => chunk.map_err(|source| Error::Body {
                    url: url.to_owned(),
                    source,
                })?,
            };
            let chunk = chunk.as_ref();
            received += chunk.len() as u64;
            if received > pin.size {
                return Err(Error::Size {
                    url: url.to_owned(),
                    expected: pin.size,
                    actual: received,
                });
            }
            hasher.update(chunk);
            writer.write_all(chunk).await?;
            progress(Progress {
                received,
                total: pin.size,
            });
        }
        writer.flush().await?;
        if received != pin.size {
            return Err(Error::Size {
                url: url.to_owned(),
                expected: pin.size,
                actual: received,
            });
        }
        Ok(())
    }
    .await;
    drop(writer);
    if let Err(error) = streamed {
        discard(temp).await;
        return Err(error);
    }
    let actual = Sha256::from_bytes(hasher.finalize().into());
    if actual != pin.sha256 {
        discard(temp).await;
        return Err(Error::Digest {
            url: url.to_owned(),
            expected: pin.sha256,
            actual,
        });
    }
    Ok(temp)
}

/// `future`'s output, or `None` once `after` passes first.
async fn within<T>(after: Duration, future: impl Future<Output = T>) -> Option<T> {
    future::or(async { Some(future.await) }, async {
        async_io::Timer::after(after).await;
        None
    })
    .await
}

/// Removes a temp file on the blocking pool.
async fn discard(temp: NamedTempFile) {
    blocking::unblock(move || drop(temp)).await;
}

/// Removes the entries under `root` whose name starts with `prefix`: what
/// killed installs of one pin left. The caller holds that pin's lock, so
/// every such entry is stale.
fn sweep(root: &Path, prefix: &str) -> io::Result<()> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if !entry
            .file_name()
            .as_encoded_bytes()
            .starts_with(prefix.as_bytes())
        {
            continue;
        }
        if entry.file_type()?.is_dir() {
            std::fs::remove_dir_all(entry.path())?;
        } else {
            std::fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

/// Unpacks the tar.zst `archive` into `staging`, syncing every extracted
/// file and, where the platform has directory sync, `staging` itself, so
/// the rename that follows exposes durable contents. Anything but regular
/// files and directories is refused.
#[cfg(feature = "archive")]
fn unpack(archive: &std::fs::File, staging: &Path) -> Result<(), Failed> {
    let decoder = zstd::stream::read::Decoder::new(archive)?;
    let mut tar = tar::Archive::new(decoder);
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let kind = entry.header().entry_type();
        if !matches!(kind, tar::EntryType::Regular | tar::EntryType::Directory) {
            return Err(Failed::Archive {
                entry: path,
                reason: "not a regular file or a directory",
            });
        }
        if !entry.unpack_in(staging)? {
            return Err(Failed::Archive {
                entry: path,
                reason: "escapes the archive root",
            });
        }
        if kind == tar::EntryType::Regular {
            std::fs::File::open(staging.join(&path))?.sync_all()?;
        }
    }
    // Windows opens no directory as a `File` and has no directory fsync;
    // NTFS journals the rename itself.
    #[cfg(unix)]
    std::fs::File::open(staging)?.sync_all()?;
    Ok(())
}

/// A [`Store::ensure`] failure; `E` is the [`Transport`]'s error.
#[derive(Debug, thiserror::Error)]
pub enum Error<E> {
    /// The request failed: it could not be sent, the response head could not
    /// be read, or the server answered with an error status.
    #[error("GET {url}: {source}")]
    Request {
        /// The pin's URL.
        url: String,
        /// The transport's failure.
        source: E,
    },
    /// The response head, or the next body chunk, did not arrive in time.
    #[error("GET {url}: nothing arrived for {after:?}")]
    Timeout {
        /// The pin's URL.
        url: String,
        /// The deadline that passed.
        after: Duration,
    },
    /// The response body broke off.
    #[error("GET {url}: reading the body failed: {source}")]
    Body {
        /// The pin's URL.
        url: String,
        /// The transport's failure.
        source: E,
    },
    /// The declared or received length differs from the pin's size.
    #[error("{url} is {actual} bytes, the pin says {expected}")]
    Size {
        /// The pin's URL.
        url: String,
        /// The pin's size.
        expected: u64,
        /// The declared length, or the bytes received.
        actual: u64,
    },
    /// The received bytes hash to another digest than the pin's.
    #[error("{url} has sha256 {actual}, the pin says {expected}")]
    Digest {
        /// The pin's URL.
        url: String,
        /// The pin's digest.
        expected: Sha256,
        /// The digest of the received bytes.
        actual: Sha256,
    },
    /// The archive holds an entry the store does not install.
    #[cfg(feature = "archive")]
    #[error("archive entry {}: {reason}", entry.display())]
    Archive {
        /// The entry's path inside the archive.
        entry: PathBuf,
        /// What is wrong with it.
        reason: &'static str,
    },
    /// The cache directory or the archive could not be read or written.
    #[error("model cache: {0}")]
    Io(#[from] io::Error),
}
