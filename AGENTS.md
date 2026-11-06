# AGENTS.md

Two library crates for running models on the user's device:

- `ondevice` (root, `src/`): pins (URL, sha256, size) and the `Store` that
  downloads, verifies and installs them into a cache, on macOS, Linux and
  Windows. Runtime-agnostic: blocking work goes through `blocking`, timers
  through `async-io`; no Tokio in the library or its tests. HTTP is the
  consumer's `Transport`, with `Zenwave` behind the `zenwave` feature; the
  store, not the transport, owns deadlines, verification and install.
- `ondevice-coreml` (`coreml/`, Apple only): a compiled multifunction Core ML
  model with one function per sequence-length bucket, typed inputs and
  outputs, and the GPU headroom monitor (`headroom` feature). It knows
  nothing of tokenizers or what the outputs mean.

## Checks

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

The model-dependent Core ML tests are `#[ignore]`d and need a model
directory:

```sh
ONDEVICE_COREML_MODEL_DIR=<dir> cargo test -p ondevice-coreml --test model -- --ignored
```

## Rules

- Every `unsafe` block is Objective-C FFI or a write into a multi-array's
  storage, and carries a `SAFETY` comment that stays accurate.
- Objective-C blocks never panic: compute before the block, copy inside it.
