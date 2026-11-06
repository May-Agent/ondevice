//! The sha256 digest a pin verifies against.

use std::fmt;

/// A sha256 digest. Its lowercase hex form names the cache entry a pin
/// installs to.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sha256([u8; 32]);

impl Sha256 {
    /// Parses 64 hex digits.
    ///
    /// # Panics
    ///
    /// When `hex` is not 64 hex digits. A digest is published data, so a
    /// malformed one is a bug in the pin; in a `const` it fails the build.
    #[must_use]
    pub const fn from_hex(hex: &str) -> Self {
        match const_hex::const_decode_to_array(hex.as_bytes()) {
            Ok(bytes) => Self(bytes),
            Err(_) => panic!("a sha256 digest is 64 hex digits"),
        }
    }

    /// The digest with these bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The digest's bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// The sha256 of `bytes`.
    #[cfg(feature = "fetch")]
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        use sha2::Digest as _;
        Self(sha2::Sha256::digest(bytes).into())
    }

    /// The lowercase hex form, computed without allocating; `const`, so a
    /// pin's digest can also name things as a `&'static str`:
    ///
    /// ```
    /// use ondevice::{Hex, Sha256};
    ///
    /// const DIGEST: Sha256 = Sha256::from_hex(
    ///     "57287aaa3fea181447dd2bd378d064aeef77684da0c722302e893dd95e43ab2c",
    /// );
    /// const HEX: &Hex = &DIGEST.hex();
    /// const ID: &str = HEX.as_str();
    /// assert_eq!(ID, DIGEST.to_string());
    /// ```
    #[must_use]
    pub const fn hex(&self) -> Hex {
        Hex(const_hex::Buffer::new().const_format(&self.0))
    }
}

impl fmt::Display for Sha256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.hex().as_str())
    }
}

/// A [`Sha256`]'s lowercase hex form: 64 digits.
#[derive(Clone)]
pub struct Hex(const_hex::Buffer<32>);

impl Hex {
    /// The digits.
    #[must_use]
    pub const fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for Hex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for Hex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hex({self})")
    }
}

impl fmt::Debug for Sha256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sha256({self})")
    }
}

#[cfg(test)]
mod tests {
    use super::Sha256;

    const HEX: &str = "57287aaa3fea181447dd2bd378d064aeef77684da0c722302e893dd95e43ab2c";

    #[test]
    fn hex_round_trips() {
        assert_eq!(Sha256::from_hex(HEX).to_string(), HEX);
    }

    #[test]
    #[should_panic(expected = "64 hex digits")]
    fn a_short_digest_panics() {
        let _ = Sha256::from_hex("5728");
    }

    #[cfg(feature = "fetch")]
    #[test]
    fn of_hashes_the_bytes() {
        assert_eq!(
            Sha256::of(b"").to_string(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
