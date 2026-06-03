# jolt-prover-goldilocks

A handwritten **Goldilocks + WHIR** Jolt prover/verifier (non-ZK). It re-implements the Jolt IOP over
the cubic extension field `GoldilocksFp3` with a **hash-based WHIR** polynomial commitment instead of the
BN254/Dory stack used by `jolt-core`. The whole crate is gated behind the `goldilocks` Cargo feature; the
default build is empty, so the BN254 graph never pulls WHIR.

Design notes live in [`GOLDILOCKS_WHIR_IMPLEMENTATION.md`](./GOLDILOCKS_WHIR_IMPLEMENTATION.md).

## Prerequisites

- **External git dependencies (cloned on first build).** The `goldilocks` graph pulls two pushed forks
  over git — no local checkouts required:
  - WHIR PCS: `atheonxyz/whir` @ `us/jolt-whir-pcs-bench`
  - Arkworks small-Fp fork: `x-senpai-x/arkworks-algebra` @ `dev/twist-shout-smallfp`
  The first build clones both; subsequent builds use the cargo git cache.
- **For the e2e test only:** the e2e compiles the `muldiv` RISC-V guest via `jolt-core`'s host, so source
  the dev environment first to put the MLIR/LLVM + RISC-V toolchain on PATH:
  ```bash
  source .bolt-dev-env
  ```
  The crate's own unit tests do not compile a guest and do not need this.

## What it proves today

On the real `muldiv` trace, `prove_e2e` / `verify_e2e` round-trip the following stages end to end:

1. **Spartan** — limbed RV64 R1CS (32-bit limbs + carries + MUL schoolbook), satisfaction via the
   binary driver.
2. **Memory** — register and RAM read/write checking + Hamming booleanity.
3. **Bytecode read-raf** — 5-stage `Val_s` (incl. the registers val-eval + lookup-table membership
   stage).
4. **Instruction-lookup read-raf** — prefix/suffix lookup argument at production word size
   (`XLEN = 64`, `LOG_K = 128`).
5. **One-hot → dense pushforwards** — LogUp\*-GKR discharge of the committed one-hot `Ra` columns.
6. **Stage-8 WHIR opening** — a single batched opening of the committed columns (R1CS aux, `Inc`, dense
   `Ra`, pushforward) under the shared duplex-sponge transcript.

### Known gaps (in progress)

- **RAM RA virtualization (#4)** — the `RamRaVirtualization` sumcheck exists and is unit-tested, but is
  not yet wired into the e2e (a dense-vs-committed non-access-cycle reconciliation is pending). Real
  initial-RAM state and program-output/I/O binding are also pending.
- **Uni-skip Spartan (Fork-2 binding)** — the self-seeded per-stage openings are not yet bound to one
  Spartan execution.

These are tracked in the design doc; until they land, treat the e2e as a functional (not yet
soundness-complete) round-trip.

## Commands

Run from the workspace root.

### Unit tests (the crate's own suite)

```bash
cargo nextest run -p jolt-prover-goldilocks --features goldilocks
```

### Full prove → verify e2e on the real `muldiv` trace

The full `--features goldilocks` test set pulls the whole WHIR graph and is disk-heavy, so build **only**
the e2e target:

```bash
source .bolt-dev-env
cargo test -p jolt-equivalence --features goldilocks --test goldilocks_e2e \
    goldilocks_real_trace_e2e_with_read_raf -- --nocapture
```

Other e2e targets in the same test file:

| test | what it checks |
|---|---|
| `goldilocks_real_trace_r1cs_is_satisfied` | limbed R1CS is satisfied on the real trace |
| `goldilocks_real_trace_binary_driver_round_trip` | Spartan → memory → booleanity round-trips |
| `goldilocks_real_trace_e2e_with_read_raf` | **full** prove/verify (all stages above) |
| `goldilocks_e2e_geometry_matches_core_muldiv` | geometry parity vs `jolt-core` |
| `goldilocks_instruction_lookup_dispatch_matches_core` | lookup-table dispatch parity vs `jolt-core` |

Run any single one by replacing the test name; drop the name to run all e2e targets.

### Lint & format

```bash
cargo clippy -p jolt-prover-goldilocks --features goldilocks --all-targets -- -D warnings
cargo fmt -p jolt-prover-goldilocks
```

## Field generality

The protocol is field-generic; Goldilocks (`Fp3`, 32-bit limbs, WHIR rate 1/2, folding 4, Blake3, ~128-bit
security) is one instantiation. A 31-bit instantiation (e.g. BabyBear/Mersenne-31) restates the limb width
and extension degree.
