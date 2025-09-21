default: fmt

fmt:
    cargo fmt --all

clippy:
    cargo clippy --workspace --all-targets -- -D warnings

test:
    cargo test --workspace

build:
    cargo build --workspace --release

ci:
    just fmt
    just clippy
    just test

package:
    cargo build --release -p novelctl
