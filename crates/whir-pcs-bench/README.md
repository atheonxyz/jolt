# whir-pcs-bench

`whir-pcs-bench` is the WHIR-ZK side of the Jolt PCS benchmark. It reads the
integer-form dump produced by `jolt-pcs-bench`, encodes the data into a chosen
field, and times the WHIR commitment plus the family pushforward checks.

## What It Measures

The binary reports:

- `commit_ms`: WHIR-ZK commitment time for 40 dense `ra_dense` vectors, 3
  eq-weighted family pushforwards, and 2 dense increment columns.
- `claim_eval_ms`: time spent evaluating the per-family dense lookup columns at
  the sampled points used by the pushforward check.
- `gkr_ms`: time spent proving the eq-weighted family pushforwards are
  consistent with the dense lookup-index vectors.

The `gkr` block is omitted when `--no-gkr` is passed. In that mode, the 3
eq-weighted pushforwards are not built or committed, so the commit timing only
covers the 40 dense lookup vectors plus `RdInc` and `RamInc`.

## Workspace Notes

This crate is a Jolt workspace member at `crates/whir-pcs-bench`.

The root workspace pins `blake3 = "=1.8.3"` because WHIR expects digest-0.10
traits for `blake3::Hasher`. Without that pin, Cargo may resolve a newer
`blake3` that does not satisfy WHIR's hasher bounds.

## How To Run

From the repository root, first produce a dump with the Jolt-side binary:

```bash
cargo run -p jolt-pcs-bench --release -- \
    --dump /tmp/jolt-pcs-bench/polys.bin --no-dory
```

Then run WHIR for one field:

```bash
cargo run -p whir-pcs-bench --release -- \
    --field goldilocks-fp3 \
    --warmup 1 --runs 5 \
    --dump /tmp/jolt-pcs-bench/polys.bin \
    --json /tmp/jolt-pcs-bench/whir-goldilocks.json
```

Available fields:

- `bn254`: `Identity<Field256>`, 32 bytes per element.
- `goldilocks-fp3`: `Identity<Field64_3>`, 24 bytes per element.

Useful flags:

- `--no-gkr`: skip the family pushforward checks and commit only the dense
  lookup and increment vectors.
- `--warmup N` / `--runs N`: control warmup and measured iterations.
- `--dump PATH`: input dump path.
- `--json PATH`: write a structured report.
- `--trace-chrome PATH`: write a tracing-chrome profile.
- `--profile-alloc --dhat-output PATH`: write a dhat heap profile. Build with
  `cargo build -p whir-pcs-bench --features profile-alloc` for allocation
  recording.

## Files

| File | Purpose |
|---|---|
| [src/main.rs](src/main.rs) | CLI, dump loading, integer-to-field encoding, WHIR commit timing, JSON output. |
| [src/gkr.rs](src/gkr.rs) | Fractional-GKR prover used for the family pushforward checks. |
| [src/gkr_bn254.rs](src/gkr_bn254.rs) | BN254-specific helpers for the GKR path. |
| [Cargo.toml](Cargo.toml) | WHIR dependency and bench-local lint/feature configuration. |

## Output

Stdout includes a summary for the selected field. With `--json`, the output
contains:

- `scheme`
- `field`
- `field_bytes_per_elem`
- `total_field_elements`
- `encode_seconds`
- `runs_ms`
- `min_ms`, `median_ms`, `max_ms`
- `per_class_ms`
- `size_classes`
- optional `gkr` timing details

The `gkr` details include combined pushforward-check timings plus nested
`claim_eval_ms`, `gkr_only_ms`, `per_family_ms`, and `per_layer_ms` fields.

