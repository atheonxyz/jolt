# jolt-pcs-bench

`jolt-pcs-bench` is the Jolt-side driver for comparing the current Dory
commitment workload against a WHIR-ZK commitment workload on the
`p256-ecdsa-verify` example.

The crate builds the ECDSA trace, reconstructs the polynomials committed by
Jolt, times Dory commitments, and writes a field-agnostic polynomial dump for
`whir-pcs-bench`.

## What It Measures

The benchmark reports:

- `dory`: Dory commitment time for 40 one-hot chunks plus dense `RdInc` and
  `RamInc` columns.
- `whir commit`: measured by `whir-pcs-bench` after reading this crate's dump.
  The WHIR side commits 40 dense `ra_dense` vectors, 3 eq-weighted family
  pushforwards, and the same 2 dense increment columns.
- `claim_eval_ms`: WHIR-side evaluation work needed before the family
  pushforward check.
- `gkr_ms`: WHIR-side fractional GKR work proving the family pushforwards are
  consistent with the dense lookup indices.

Not measured:

- PCS opening proofs.
- The rest of the Jolt prover pipeline.
- Verifier work.
- Dory over non-BN254 fields.

## Workspace Layout

Both benchmark crates are workspace members:

```text
crates/jolt-pcs-bench/
    Builds the ECDSA workload, times Dory, and writes /tmp/jolt-pcs-bench/polys.bin.

crates/whir-pcs-bench/
    Reads the dump and times WHIR-ZK over BN254.
```

The root workspace pins `blake3 = "=1.8.3"` because WHIR expects digest-0.10
traits for `blake3::Hasher`. Without that pin, Cargo may resolve a newer
`blake3` that does not satisfy WHIR's hasher bounds.

## How To Run

From the repository root:

```bash
# One-shot: builds both binaries, runs Dory and WHIR, and prints a combined table.
crates/jolt-pcs-bench/run-bench.sh --runs 5 --warmup 1
```

Step by step:

```bash
cargo build -p jolt-pcs-bench --release
target/release/jolt-pcs-bench \
    --warmup 1 --runs 5 \
    --dump /tmp/jolt-pcs-bench/polys.bin \
    --json /tmp/jolt-pcs-bench/dory.json

cargo build -p whir-pcs-bench --release
target/release/whir-pcs-bench \
    --warmup 1 --runs 5 \
    --dump /tmp/jolt-pcs-bench/polys.bin \
    --json /tmp/jolt-pcs-bench/whir-bn254.json
```

Useful flags:

- `--verify-only`: run transformation invariants without timing Dory.
- `--no-dory`: write the WHIR dump without running Dory.
- `--no-dump`: run Jolt-side work without writing a dump.
- `--json PATH`: write a structured Jolt-side report.
- `--trace-chrome PATH`: write a tracing-chrome profile.
- `--profile-alloc --dhat-output PATH`: write a dhat heap profile. Build with
  `cargo build -p jolt-pcs-bench --features profile-alloc` for allocation
  recording.

## Profiling

The profiling orchestrator uses the same workspace-root layout as the benchmark
script:

```bash
crates/jolt-pcs-bench/profile.sh all --all
```

Artifacts are written under `/tmp/jolt-pcs-bench/traces/`.

Modes:

- `--samply`: CPU sampling profile.
- `--chrome`: tracing-chrome output for Perfetto.
- `--dhat`: heap allocation profile.
- `--all`: all profile types.

## Output Shape

The one-shot script prints a combined table and writes:

- `/tmp/jolt-pcs-bench/dory.json`
- `/tmp/jolt-pcs-bench/whir-bn254.json`
- `/tmp/jolt-pcs-bench/combined.json`

The combined report includes workload metadata, Dory timings, WHIR timings,
field-element counts, and WHIR/Dory ratios.

## Implementation Notes

- The workload is `examples/p256-ecdsa-verify` padded to `T = 2^19` cycles.
- The Jolt side uses BN254 Fr; WHIR also runs over BN254 (`Field256`).
- `src/workload.rs` drives the guest and trace extraction.
- `src/sources.rs` derives the per-cycle committed-polynomial columns natively
  (the same formulas as `CommittedPolynomial::generate_witness`).
- `src/jolt_polys.rs` reconstructs the committed polynomial set.
- `src/logup_star.rs` converts one-hot chunks into dense lookup-index vectors.
- `src/dump.rs` writes the field-agnostic dump consumed by `whir-pcs-bench`.
- `src/dory_bench.rs` times Dory commitments.
- `src/verify.rs` checks transformation invariants in debug builds and in
  release builds when `--verify-only` is supplied.

The bench commits one-hot oracles through `jolt_dory::DoryScheme` (the PCS the
in-development jolt-prover uses) with a column/address-major one-hot layout
(`flat = index * T + cycle`). `jolt-dory` derives a per-oracle square matrix
shape, which differs from jolt-core's embedded production layout, so the Dory
timing is a representative proxy rather than a byte-exact reproduction of the
production prover's commitment path.
