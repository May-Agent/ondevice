//! Pinned, verified first-use downloads of on-device model files.
//!
//! A [`Pin`] names a file by an immutable URL, the sha256 of its bytes and
//! their length. A [`Store`] downloads it once into a cache directory,
//! verifies length and digest before anything is installed, and from then on
//! answers from the cache. What a pin installs is its kind: a single
//! [`File`], or an [`Archive`] unpacked into a directory.
//!
//! The pin types carry no dependencies, so a crate that only declares pins
//! builds without the default features; the [`Store`] is the `fetch`
//! feature, [`Archive`] installs the `archive` feature. The store is
//! runtime-agnostic — blocking work runs on the `blocking` pool, timers on
//! `async-io` — and downloads through any [`Transport`], so a consumer
//! reuses the HTTP client it already has; the `zenwave` feature provides
//! one.

mod digest;
mod pin;
#[cfg(feature = "fetch")]
mod store;
#[cfg(feature = "fetch")]
mod transport;

pub use digest::{Hex, Sha256};
pub use pin::{Archive, File, Pin};
#[cfg(feature = "fetch")]
pub use store::{Content, Error, Progress, Store};
#[cfg(feature = "zenwave")]
pub use transport::Zenwave;
#[cfg(feature = "fetch")]
pub use transport::{Response, Transport};
