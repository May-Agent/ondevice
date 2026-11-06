//! What a pin names and what it installs as.

use crate::Sha256;

/// A downloadable file fixed by its contents: an immutable URL, the sha256
/// of its bytes and their length. `K` is what the bytes install as — a
/// [`File`] or an [`Archive`].
///
/// Pins are published data, usually `const`:
///
/// ```
/// use ondevice::{Archive, Pin, Sha256};
///
/// const MODEL: Pin<'static, Archive> = Pin {
///     url: "https://example.com/model.tar.zst",
///     sha256: Sha256::from_hex(
///         "57287aaa3fea181447dd2bd378d064aeef77684da0c722302e893dd95e43ab2c",
///     ),
///     size: 417_260_546,
///     kind: Archive,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pin<'a, K> {
    /// Download URL, at an immutable upstream revision.
    pub url: &'a str,
    /// The sha256 of the downloaded bytes.
    pub sha256: Sha256,
    /// The length of the downloaded bytes.
    pub size: u64,
    /// What the bytes install as.
    pub kind: K,
}

/// A pin that installs as one file, `<root>/<sha256>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct File;

/// A pin that installs as a directory, `<root>/<sha256>/`: a tar archive
/// compressed with zstd, holding only regular files and directories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Archive;
