//! `Store::ensure` through the `Zenwave` transport against a local one-shot
//! server.

mod support {
    pub mod http;
}

use std::path::Path;
use std::sync::Mutex;

use futures_lite::future::{block_on, zip};
use ondevice::{Archive, Error, File, Pin, Progress, Sha256, Store, Zenwave};
use support::http::{archive, serve_once, symlink_archive};

fn pin<'a, K>(url: &'a str, body: &[u8], kind: K) -> Pin<'a, K> {
    Pin {
        url,
        sha256: Sha256::of(body),
        size: body.len() as u64,
        kind,
    }
}

/// The lock files are the only entries a failed or finished install may
/// leave beside the installed ones.
fn assert_no_leftovers(root: &Path) {
    let names: Vec<String> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    assert!(
        names.iter().all(|name| !name.starts_with('.')),
        "leftover entries: {names:?}"
    );
}

#[test]
fn an_archive_downloads_verifies_and_unpacks() {
    let body = archive(&[
        ("manifest.json", b"{\"functions\": {}}"),
        ("model.mlmodelc/weights.bin", b"weights"),
    ]);
    let (url, server) = serve_once("200 OK", Some(body.len() as u64), body.clone());
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path().join("models"), Zenwave);
    let pin = pin(&url, &body, Archive);
    let last = Mutex::new(None);

    let dir =
        block_on(store.ensure(&pin, |progress| *last.lock().unwrap() = Some(progress))).unwrap();

    assert_eq!(dir, store.path(&pin));
    assert_eq!(dir, root.path().join("models").join(pin.sha256.to_string()));
    assert_eq!(
        last.into_inner().unwrap(),
        Some(Progress {
            received: body.len() as u64,
            total: body.len() as u64,
        })
    );
    assert_eq!(
        std::fs::read(dir.join("manifest.json")).unwrap(),
        b"{\"functions\": {}}"
    );
    assert_eq!(
        std::fs::read(dir.join("model.mlmodelc/weights.bin")).unwrap(),
        b"weights"
    );
    assert_no_leftovers(store.root());
    server.join().unwrap();
}

#[test]
fn a_corrupt_download_is_rejected_and_leaves_nothing() {
    let good = archive(&[("manifest.json", b"{}")]);
    let corrupt = archive(&[("manifest.json", b"[]")]);
    assert_eq!(good.len(), corrupt.len());
    let (url, server) = serve_once("200 OK", Some(corrupt.len() as u64), corrupt.clone());
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path(), Zenwave);
    let pin = pin(&url, &good, Archive);

    let error = block_on(store.ensure(&pin, |_| {})).unwrap_err();

    match error {
        Error::Digest {
            expected, actual, ..
        } => {
            assert_eq!(expected, Sha256::of(&good));
            assert_eq!(actual, Sha256::of(&corrupt));
        }
        error => panic!("expected Digest, got {error}"),
    }
    assert!(!store.path(&pin).exists());
    assert_no_leftovers(store.root());
    server.join().unwrap();
}

#[test]
fn an_installed_pin_skips_the_download() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path(), Zenwave);
    let pin = Pin {
        url: "http://127.0.0.1:1/unreachable",
        sha256: Sha256::from_bytes([7; 32]),
        size: 1,
        kind: Archive,
    };
    let dir = store.path(&pin);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("marker"), b"sentinel").unwrap();

    assert_eq!(block_on(store.ensure(&pin, |_| {})).unwrap(), dir);
    assert_eq!(std::fs::read(dir.join("marker")).unwrap(), b"sentinel");
}

#[test]
fn an_interrupted_download_leaves_nothing() {
    let body = archive(&[("big.bin", &[7u8; 8192])]);
    let sent = body.len() / 2;
    let (url, server) = serve_once("200 OK", Some(body.len() as u64), body[..sent].to_vec());
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path(), Zenwave);
    let pin = pin(&url, &body, Archive);

    let error = block_on(store.ensure(&pin, |_| {})).unwrap_err();

    assert!(
        matches!(error, Error::Body { .. } | Error::Size { .. }),
        "unexpected error: {error}"
    );
    assert!(!store.path(&pin).exists());
    assert_no_leftovers(store.root());
    server.join().unwrap();
}

#[test]
fn a_declared_length_other_than_the_pin_is_refused() {
    let body = archive(&[("manifest.json", b"{}")]);
    let (url, server) = serve_once("200 OK", Some(body.len() as u64 + 1), body.clone());
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path(), Zenwave);
    let pin = pin(&url, &body, Archive);

    let error = block_on(store.ensure(&pin, |_| {})).unwrap_err();

    match error {
        Error::Size {
            expected, actual, ..
        } => {
            assert_eq!(expected, body.len() as u64);
            assert_eq!(actual, body.len() as u64 + 1);
        }
        error => panic!("expected Size, got {error}"),
    }
    assert!(!store.path(&pin).exists());
    server.join().unwrap();
}

#[test]
fn a_body_longer_than_the_pin_is_cut_off() {
    let body = vec![1u8; 4096];
    let (url, server) = serve_once("200 OK", None, body.clone());
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path(), Zenwave);
    let pin = Pin {
        url: &url,
        sha256: Sha256::of(&body[..100]),
        size: 100,
        kind: File,
    };

    let error = block_on(store.ensure(&pin, |_| {})).unwrap_err();

    match error {
        Error::Size {
            expected, actual, ..
        } => {
            assert_eq!(expected, 100);
            assert!(actual > 100, "received {actual}");
        }
        error => panic!("expected Size, got {error}"),
    }
    assert!(!store.path(&pin).exists());
    assert_no_leftovers(store.root());
    server.join().unwrap();
}

#[test]
fn an_error_status_is_reported() {
    let (url, server) = serve_once("503 Service Unavailable", Some(0), Vec::new());
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path(), Zenwave);
    let pin = pin(&url, b"", File);

    let error = block_on(store.ensure(&pin, |_| {})).unwrap_err();

    match error {
        Error::Request {
            source: zenwave::Error::Http { status, .. },
            ..
        } => assert_eq!(status.as_u16(), 503),
        error => panic!("expected an HTTP status error, got {error}"),
    }
    server.join().unwrap();
}

#[test]
fn symlink_entries_are_refused() {
    let body = symlink_archive();
    let (url, server) = serve_once("200 OK", Some(body.len() as u64), body.clone());
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path(), Zenwave);
    let pin = pin(&url, &body, Archive);

    let error = block_on(store.ensure(&pin, |_| {})).unwrap_err();

    match error {
        Error::Archive { entry, .. } => assert_eq!(entry, Path::new("evil")),
        error => panic!("expected Archive, got {error}"),
    }
    assert!(!store.path(&pin).exists());
    assert_no_leftovers(store.root());
    server.join().unwrap();
}

#[test]
fn concurrent_installs_download_once() {
    let body = archive(&[("manifest.json", b"{}")]);
    // The server answers one request: a second download would fail.
    let (url, server) = serve_once("200 OK", Some(body.len() as u64), body.clone());
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path(), Zenwave);
    let pin = pin(&url, &body, Archive);

    let (first, second) = block_on(zip(store.ensure(&pin, |_| {}), store.ensure(&pin, |_| {})));

    assert_eq!(first.unwrap(), store.path(&pin));
    assert_eq!(second.unwrap(), store.path(&pin));
    assert_eq!(
        std::fs::read(store.path(&pin).join("manifest.json")).unwrap(),
        b"{}"
    );
    server.join().unwrap();
}

#[test]
fn an_install_removes_its_own_leftovers_only() {
    let body = b"{\"tokenizer\": true}".to_vec();
    let (url, server) = serve_once("200 OK", Some(body.len() as u64), body.clone());
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path(), Zenwave);
    let pin = pin(&url, &body, File);
    let own_file = root.path().join(format!(".{}.killed", pin.sha256));
    let own_dir = root.path().join(format!(".{}.staging", pin.sha256));
    let other = root
        .path()
        .join(format!(".{}.in-flight", Sha256::of(b"other")));
    std::fs::write(&own_file, b"partial").unwrap();
    std::fs::create_dir_all(own_dir.join("nested")).unwrap();
    std::fs::write(&other, b"another pin's download").unwrap();

    block_on(store.ensure(&pin, |_| {})).unwrap();

    assert!(!own_file.exists());
    assert!(!own_dir.exists());
    assert!(other.exists(), "another pin's entry was removed");
    server.join().unwrap();
}

#[test]
fn a_file_downloads_verifies_and_caches() {
    let body = b"{\"tokenizer\": true}".to_vec();
    let (url, server) = serve_once("200 OK", Some(body.len() as u64), body.clone());
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path().join("models"), Zenwave);
    let pin = pin(&url, &body, File);

    let path = block_on(store.ensure(&pin, |_| {})).unwrap();

    assert_eq!(
        path,
        root.path()
            .join("models")
            .join(Sha256::of(&body).to_string())
    );
    assert_eq!(std::fs::read(&path).unwrap(), body);
    assert_no_leftovers(store.root());
    server.join().unwrap();
    // The one-shot server is spent; a second ensure must be a hit.
    assert_eq!(block_on(store.ensure(&pin, |_| {})).unwrap(), path);
}

#[test]
fn a_file_mismatch_is_rejected_and_leaves_nothing() {
    let corrupt = b"corrupted".to_vec();
    let (url, server) = serve_once("200 OK", Some(corrupt.len() as u64), corrupt.clone());
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path(), Zenwave);
    let pin = Pin {
        url: &url,
        sha256: Sha256::of(b"expected!"),
        size: corrupt.len() as u64,
        kind: File,
    };

    let Err(Error::Digest { .. }) = block_on(store.ensure(&pin, |_| {})) else {
        panic!("a corrupt file is rejected");
    };
    assert!(!store.path(&pin).exists());
    assert_no_leftovers(store.root());
    server.join().unwrap();
}
