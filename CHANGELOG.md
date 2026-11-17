# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.3](https://github.com/May-Agent/ondevice/compare/v0.0.2...v0.0.3) - 2026-11-17

### Other

- Merge pull request #19 from May-Agent/ci/dispatch-publish
- dispatch publish for each tag release-plz creates
- approve the release PR's held runs instead of dispatching uncounted ones

## [0.0.2](https://github.com/May-Agent/ondevice/compare/v0.0.1...v0.0.2) - 2026-11-17

### Added

- *(coreml)* [**breaking**] separate model functions, native tensors and stateful sessions

### Fixed

- *(store)* document Windows read-only attribute cleanup
- *(store)* retain writable archive handles through synchronization

### Other

- let release-plz run on demand
- report the required check on every pull request
- release v0.0.1

## [0.0.1](https://github.com/May-Agent/ondevice/releases/tag/v0.0.1) - 2026-11-06

### Added

- pinned model store and bucketed Core ML inference
