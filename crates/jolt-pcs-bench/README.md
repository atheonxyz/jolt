# jolt-pcs-bench — WHIR-ZK + GKR vs Dory PCS comparison on the ECDSA workload

A feasibility benchmark that answers a single question:

> For the actual polynomial commitment workload of a Jolt ECDSA proof at 2^19
> cycles, how does WHIR-ZK plus the LogUp\* trick (commit + GKR pushforward
> proving) compare to Jolt's Dory commit in wall-clock time and committed
> data volume?

What's measured (three timing phases plus the Dory baseline):

1. **Dory commit** of Jolt's current one-hot polynomial set (32 + 4 + 4 = 40
   sparse one-hot chunks + 2 dense `RdInc`/`RamInc`).
2. **WHIR-ZK commit** of the LogUp\* §4.1-batched set: 40 `ra_dense ∈ F^T`
   + **3 eq-weighted pushforwards** `P^F ∈ F^{2^15}` (one per Jolt family —
   InstructionRa, BytecodeRa, RamRa — per LogUp\* §4.1) + the same 2 dense
   oracles. Reported as `commit_ms`.
3. **`claim_eval_ms`**: per-family `d` MLE evaluations
   `M̃^(i)(r_row, r_col)` — the inputs to PAZK §4.5.2's claim reduction.
   In a real protocol these come from upstream sumcheck binding; the bench
   reports them separately so reviewers can decide how to attribute them.
4. **`gkr_ms`** — LogUp\* §4 / Figure-1 prover per family: §4.5.2 claim
   reduction + eq-weighted `P^F` materialization + fan-in-2 fractional GKR
   sumcheck proving the well-formedness identity
   `Σ_j eq(bits(j), r_M_row)/(α − M^(*)[j]) == Σ_k P^F[k]/(α − k)`.

What's **not** measured: opening proofs; sumcheck/Spartan/BlindFold; anything
inside the prover crate. The bench is read-only with respect to
`crates/jolt-prover`.

**Field menu** (WHIR-side, see `whir-pcs-bench --field …`):

- `bn254` (default) — `Identity<Field256>`, apples-to-apples with Dory.
- `goldilocks-fp3` — `Identity<Field64_3>`, cubic extension of Goldilocks
  (192-bit, the soundness-correct setup at 128-bit security).

The Jolt side / Dory always uses BN254 Fr (that's what Jolt's prover actually
runs). Only the WHIR side is field-agnostic.

BabyBear / Mersenne31 / KoalaBear are deferred follow-ups (would require adding
a new `MontConfig` plus an extension field).

## Design docs

Both papers are at the workspace root:

- [`twist_shout.pdf`](../../TwistShout.pdf) — original Twist/Shout protocol.
- [`twist_shout_logup_star.pdf`](../../twist_shout_logup_star.pdf) — LogUp\*
  reformulation. Key sections referenced by the bench: §4.1 (batching),
  §5.1 (Shout), §5.2 (Twist), Figure 1 (GKR pushforward), eq. 6
  (Inc-evaluation, out of scope for the bench).

Two design docs map the bench code to the paper sections and to the Jolt
codebase:

- [DESIGN_DORY.md](DESIGN_DORY.md) — the Dory side: which polynomials Jolt
  commits (Shout: `InstructionRa`, `BytecodeRa`; Twist: `RamRa`,
  `RamInc`; plus `RdInc`), how the bench builds them, the Dory API path,
  and divergences from jolt-core's production prover.
- [DESIGN_WHIR.md](DESIGN_WHIR.md) — the WHIR + LogUp\* side: the
  argmax transformation (§5.1 / §5.2), the field-agnostic dump format,
  the size-class commit with 3 per-family eq-weighted pushforwards
  (§4.1), PAZK §4.5.2 claim reduction (specialized to shared
  `(r_row, r_col)`), and the Figure-1 eq-weighted fan-in-2 fractional
  GKR pushforward prover.

---

## What it actually does

1. Drives the existing `examples/p256-ecdsa-verify` guest end-to-end through
   the public `jolt-trace` pipeline (no instrumentation of the prover — every
   API needed is already `pub`). Produces a `CommitmentTraceSources` struct that
   is byte-identical to what Jolt's prover would build internally.
2. Materializes the polynomials Jolt currently commits via Dory:
   - `InstructionRa` (32 chunks)
   - `BytecodeRa` (4 chunks)
   - `RamRa` (4 chunks)
   - `RdInc`, `RamInc` (dense T-vectors)
3. Times Dory's `commit` on every one of those (40 sparse one-hot + 2 dense calls).
4. Applies the §5.1 / §5.2 dense-encoding step: each one-hot chunk becomes
   a dense `ra_dense ∈ F^T` (the argmax index per cycle — Jolt already
   stores it that way as `Vec<Option<u8>>`). No pushforward is produced
   here — the eq-weighted pushforward depends on Fiat-Shamir randomness
   that lives in the WHIR-side transcript, so it's built at runtime later.
5. Asserts argmax / size invariants (`verify.rs`). The dump still emits a
   legacy per-chunk u32 histogram for backward compatibility, but the
   WHIR side discards it on load.
6. Serializes ra_dense (u8) + the legacy histograms (u32) + dense
   (`RdInc`, `RamInc` as i128) to disk as raw integers — field-agnostic,
   ~41 MB total.
7. A sibling binary `whir-pcs-bench` (separate workspace, see below) reads
   the dump, encodes each integer into the chosen WHIR target field
   (`bn254` or `goldilocks-fp3`).
8. **Per family** (3 families: InstructionRa with d=32, BytecodeRa d=4,
   RamRa d=4) the sibling binary builds one eq-weighted pushforward
   `P^F = M^(*)_dense_∗ eq_{r_M_row}` (per LogUp\* §4 eq. 4) and commits
   it via WHIR — so the WHIR commit phase sees 40 `ra_dense` + 2 dense +
   **3** eq-weighted `P^F`.
9. The sibling binary then runs the **Figure-1 LogUp\* prover** per
   family: PAZK §4.5.2 claim reduction (specialized: free in our case),
   the eq-weighted P^F materialization, and a fan-in-2 fractional-add
   GKR proving `Σ_j eq(bits(j), r_M_row) / (α − M^(*)[j]) == Σ_k P^F[k] /
   (α − k)`. Time reported across `claim_eval_ms` (d MLE evals per
   family) and `gkr_ms` (everything else).
10. The orchestrator script combines all JSON reports into a side-by-side
   table showing Dory commit vs each WHIR field choice's commit /
   claim_eval / gkr / combined cost.

---

## Architecture: why two binaries?

The bench is split into two crates that communicate via an on-disk polynomial dump.

```
crates/jolt-pcs-bench/                  (Jolt workspace member)
    Drives the ECDSA guest, builds polys, times Dory, dumps polys to disk.

../../../whir-pcs-bench/                (sibling crate, own workspace)
    Reads the dump and times WHIR-ZK.
```

**Why two crates?** A dependency-version conflict. Both jolt-core's
`digest = "0.11"` (transitively, via `blake2` / `sha2` / `sha3` 0.11) and WHIR's
`digest = "0.10.7"` would land in the same dep graph, and `blake3 1.8.5`
(latest in the Jolt workspace) implements digest 0.11 traits while WHIR's
`hash::DigestEngine` requires digest 0.10 traits on `blake3::Hasher`. The
trait-bound mismatch fails compilation when WHIR is added as a path dep
inside the Jolt workspace.

The cleanest workaround is to keep WHIR in its own workspace and pin
`blake3 = "=1.8.3"` (the last version that exposes digest 0.10 traits).
Both binaries exchange polynomials via a tiny binary file format (`dump.rs` /
`whir-pcs-bench`'s `read_dump`).

---

## Layout

```
crates/jolt-pcs-bench/
├── Cargo.toml
├── README.md                ← this file
├── DESIGN_DORY.md           ← Dory bench ↔ Twist/Shout paper ↔ Jolt codebase
├── DESIGN_WHIR.md           ← WHIR + LogUp* + GKR bench ↔ paper ↔ Jolt codebase
├── PROFILING.md             ← per-config wall-clock + dhat breakdown
├── run-bench.sh             ← orchestrator: builds both, runs both, combines JSON
├── profile.sh               ← captures samply + chrome + dhat traces
└── src/
    ├── main.rs              ← CLI, JSON output, glue
    ├── workload.rs          ← ECDSA guest compile → trace → CommitmentTraceSources
    ├── jolt_polys.rs        ← Builds Jolt's committed polynomial set + a local
    │                          copy of `AddressMajorOneHotPolynomial` (mirrors
    │                          `crates/jolt-prover/src/stages/commitment.rs`)
    ├── logup_star.rs        ← §5.1 / §5.2 LogUp* transformation
    ├── verify.rs            ← argmax / size invariants on ra_dense. (The
    │                           legacy per-chunk histogram invariant is now
    │                           moot — the WHIR side rebuilds P^F at runtime.)
    ├── dory_bench.rs        ← DoryScheme::commit timing
    └── dump.rs              ← Polynomial dump format consumed by whir-pcs-bench

../../../whir-pcs-bench/      ← sibling crate, separate workspace
├── Cargo.toml               ← pins blake3 = "=1.8.3" to keep digest 0.10
├── README.md                ← WHIR-side overview + how to run
└── src/
    ├── main.rs              ← read dump, encode to F, commit + GKR timing
    └── gkr.rs               ← fan-in-2 fractional GKR pushforward prover
```

---

## The LogUp\* protocol (one-paragraph summary)

Each one-hot chunk `ra(k, j) ∈ {0,1}^{K_chunk × T}` becomes a dense
`ra_dense ∈ F^T` (the argmax index per cycle — Jolt already stores this).
The d chunks per family are row-concatenated into `M^(*) ∈ F^{T·d}`
(§4.1). Per family, the prover commits **one** eq-weighted pushforward
`P^F[k] = Σ_{j : M^(*)[j] = k} eq(bits(j), r_M_row)` (LogUp\* §4 eq. 4) —
not a histogram. After the upstream sumcheck provides d input claims
`M̃^(i)(r_row, r_col)`, PAZK §4.5.2 reduces them to one combined claim,
and Figure 1's eq-weighted fan-in-2 fractional GKR proves well-formedness
of P^F. Dense oracles `RdInc`/`RamInc` are unchanged. Pushforwards are
padded to 2^15 zeros (whir_zk blinding minimum). Across the 3 Jolt
families, **3 pushforwards are committed total** (not 40). See
[DESIGN_WHIR.md](DESIGN_WHIR.md) for the full mapping to the paper.

---

## Assumptions and modifications

These are the calls and constants this bench takes that are worth being explicit
about. None of them require modifying anything in the Jolt prover crate.

### 1. ECDSA workload at T = 2^19 cycles

The benchmark fixes `T = 524288 = 2^19`, which is exactly `max_trace_length`
declared in the guest at [examples/p256-ecdsa-verify/guest/src/lib.rs:5](../../examples/p256-ecdsa-verify/guest/src/lib.rs#L5).
The actual ECDSA trace is shorter; the rest is NoOp padding. Both the actual
cycle count and the padded trace length are reported to stdout on every run.


### 2. Test vectors

The exact `(z, r, s, q)` tuple from
[examples/p256-ecdsa-verify/src/main.rs:24-54](../../examples/p256-ecdsa-verify/src/main.rs#L24-L54).
The message hash z is SHA-256("sample"), with the signature derived from an
RFC 6979 private key. Postcard-encoded just like the macro generates.

### 3. Polynomial extraction strategy: in-process rebuild (no prover instrumentation)

This was deliberate. The plan considered an `env-var dump from prover hot path`
alternative; we rejected it because everything needed
(`extract_trace`, `commitment_trace_sources`, `one_hot_chunk_indices`,
`dense_i128_column_to_field`, `OneHotParams::new`) is already `pub` in
`jolt-trace` / `jolt-witness` / `jolt-core`. Touching `commit_oracle` in
`crates/jolt-prover/src/stages/commitment.rs` (which is hot) for a one-shot
research benchmark would have been a worse engineering tradeoff.

### 4. `AddressMajorOneHotPolynomial` duplicated, not imported

The one piece in `crates/jolt-prover/src/stages/commitment.rs:214-318` we
needed is `pub(crate)`. Rather than make it public upstream (perturbing the
prover's public surface for a research bench), `src/jolt_polys.rs` duplicates
its `MultilinearPoly<Fr>` implementation. ~60 lines, byte-identical semantics.

### 5. Dory parameters

- `DoryScheme::setup_prover(num_vars=22)` — `log_T + log_k_chunk = 18 + 4 = 22`.
- One-hot oracles: `DoryScheme::commit(&AddressMajorOneHotPolynomial, &setup)` (sparse path).
- Dense oracles: `DoryScheme::commit_evaluations_with_row_len(data, row_len, &setup)`.
- Same `row_len = 2^ceil(num_vars/2)` heuristic as `commit_with_layout` in `commitment.rs:1590`.

### 6. WHIR-ZK parameters

These match the WHIR `src/bin/benchmark.rs` defaults except `hash = Blake3`:

- `security_level = 128`
- `pow_bits = 20`
- `initial_folding_factor = 4`
- `folding_factor = 4`
- `starting_log_inv_rate = 1`  (rate 2^-1 = 1/2)
- `unique_decoding = false`  (list decoding)
- `hash = Blake3`

ZK is via `whir::protocols::whir_zk::Config<Field256>` (NOT default plain WHIR —
the WHIR repo's default is non-ZK). The `rs_in_order` feature is required for
`whir_zk` and is enabled in `Cargo.toml`.

### 7. `WHIR_MIN_NUM_VARS = 15` for pushforward padding

`whir_zk::Config::new` asserts `num_blinding_variables < num_witness_variables`.
At 128-bit security with rate 1/2 and folding_factor=4, the blinding-variable
upper bound works out to ~14, so any vector with fewer than 15 variables fails
the assertion. Pushforward vectors are length `K_chunk = 16` (4 variables),
which fails — we pad each up to 2^15 zeros to satisfy the inequality.

This is the unavoidable cost of ZK at 128-bit security. The plan's risks
section flagged this; the empirical minimum (`14` fails, `15` passes) is in a
comment at `logup_star.rs:18-26`.

### 8. WHIR vector groups (size classes)

`whir_zk::Config::commit` requires uniform-size vectors. We bucket polynomials
into two size classes:

- `num_vars = 15` (2^15 = 32768): **3** eq-weighted pushforward vectors
  (one per family — InstructionRa, BytecodeRa, RamRa — per LogUp\* §4.1).
- `num_vars = 19` (2^19 = 524288): all 40 `ra_dense` + `RdInc` + `RamInc`.

Each class is committed in one batched `commit` call. The reported WHIR
wall-clock is the sum of both calls. The drop from 40 to 3 pushforwards
in the small class accounts for most of the `commit_ms` reduction vs the
prior milestone.

### 9. Field-agnostic dump format (integer-form, version 2)

The dump stores **raw integer values** rather than pre-encoded Fr field elements:

- `ra_dense` chunks: `Vec<u8>` (1 byte per cycle) — argmax index ∈ [0, 16).
- `pushforward::<family>_<chunk>` vectors: `Vec<u32>` (4 bytes per bucket).
  **Legacy field**: these are the unweighted per-chunk histograms from the
  pre-§4.1 design. The WHIR side ignores them on load and rebuilds the
  per-family eq-weighted `P^F` at runtime. Kept in the dump format
  (version 2) to avoid touching the Jolt-side dump writer for a research
  bench.
- `RdInc`, `RamInc`: `Vec<i128>` (16 bytes per cycle) — signed RAM/register increments.

Each WHIR target field independently encodes these integers at load time via
`F::from_u64` and a signed-`Neg` helper for `i128`. Total dump size shrinks from
~750 MB (Fr-encoded) to ~41 MB (integer-encoded) — useful for keeping dumps in
`/tmp` between bench iterations.

**Why this is lossless for both target fields.** The dump never contains a
254-bit field element; only `u8` / `u32` / `i128` integers whose maximum
magnitudes are far below either target field's modulus:

| Polynomial    | Stored as | Max \|v\|        | Fits BN254 Fr (~2^254)? | Fits Goldilocks Fp3 (~2^192)? |
| ------------- | --------- | ---------------- | ----------------------- | ----------------------------- |
| `ra_dense`    | `u8`      | < 16             | ✓                       | ✓                             |
| `pushforward` | `u32`     | < T = 2^19       | ✓                       | ✓                             |
| `RdInc/RamInc`| `i128`    | < 2·2^64 ≈ 2^65  | ✓ (188 bits of headroom)| ✓ (126 bits of headroom)      |

The integer→Fr conversion produces **bit-identical Fr elements** to the older
Fr-encoded dump path. This was validated by a transient `--verify-equivalence`
test that wrote 9 representative values through both paths (signs, zero, max
u64, negative max u64, mid-range) and asserted byte-equal canonical
serializations across the two `ark-ff` versions (Jolt's patched
`dev/twist-shout` and `whir-pcs-bench`'s stock 0.5). The test has since been
deleted; the trace of its presence is the equivalence argument above.

**Reporting**: WHIR-side load+encode time is reported separately from
`commit_ms` so the published commit metric is apples-to-apples with the
Fr-encoded prior milestone.

### 10. Force-linking `jolt-inlines-p256`

The guest binary uses opcode `0x0B`, funct7 `0x07` (p256 inlines). These
register at link time via `inventory::submit!`. To make sure cargo links the
inline crate, `main.rs` has `use jolt_inlines_p256 as _;`.

### 11. Workspace member, but pedantic lints relaxed

The crate is registered in `jolt/Cargo.toml` `workspace.members`. The
workspace's clippy policy is pedantic (denies print_stdout, unwrap_used, etc.)
which makes sense for a prover but is wrong for an interactive measurement
tool. `Cargo.toml` overrides these lints at the crate level only.

---

## How to run

```bash
# One-shot: builds both binaries, runs Dory + WHIR on the chosen field(s),
# prints combined comparison table. --field accepts {bn254, goldilocks-fp3, both}.
crates/jolt-pcs-bench/run-bench.sh --runs 5 --warmup 1 --field both

# Or step by step:
cargo build -p jolt-pcs-bench --release
target/release/jolt-pcs-bench \
    --warmup 1 --runs 5 \
    --dump /tmp/jolt-pcs-bench/polys.bin \
    --json /tmp/jolt-pcs-bench/dory.json

cd ../whir-pcs-bench
cargo build --release
# BN254
target/release/whir-pcs-bench \
    --field bn254 \
    --warmup 1 --runs 5 \
    --dump /tmp/jolt-pcs-bench/polys.bin \
    --json /tmp/jolt-pcs-bench/whir-bn254.json
# Goldilocks Fp3
target/release/whir-pcs-bench \
    --field goldilocks-fp3 \
    --warmup 1 --runs 5 \
    --dump /tmp/jolt-pcs-bench/polys.bin \
    --json /tmp/jolt-pcs-bench/whir-goldilocks.json
```

Other useful flags:

- `--verify-only` (jolt-pcs-bench): runs only the transformation invariants, no timing.
- `--no-dory`: skip the Dory bench (e.g. to re-dump without re-running Dory).
- `--no-dump`: skip writing the polynomial dump file.
- `--no-gkr` (whir-pcs-bench): skip the LogUp\* GKR phase entirely. With
  this flag the per-family eq-weighted pushforwards are NOT built and
  NOT committed, so `commit_ms` only covers `ra_dense` + 2 dense
  oracles — useful for isolating the pure commit-phase cost.
- `--field {bn254 | goldilocks-fp3}` (whir-pcs-bench): pick the scalar field.

## Profiling

Both binaries support three profile types:

- **`--trace-chrome <path>`** — writes a tracing-chrome span tree (view at
  https://ui.perfetto.dev/). Captures bench-side spans + all WHIR internal
  `#[instrument]` spans (irs_commit, sumcheck, matrix_commit, etc.).
- **`--profile-alloc --dhat-output <path>`** — captures a dhat heap profile
  (view at https://nnethercote.github.io/dh_view/dh_view.html). Requires
  `cargo build --features profile-alloc`.
- **samply** — CPU sampling profile (run `samply record -- target/samply/<bin> [args]`).
  Both crates have a `[profile.samply]` block: `cargo build --profile samply`.

One-shot orchestrator captures all three for all three configs:

```bash
crates/jolt-pcs-bench/profile.sh all --all
# Artifacts under /tmp/jolt-pcs-bench/traces/{dory,whir-bn254,whir-fp3}/
```

See [PROFILING.md](PROFILING.md) for the per-config wall-clock and allocation
breakdown tables, the validated bench-side wins, the GKR per-layer
breakdown, and the documented WHIR-internal hot paths (out-of-scope for the
bench-side work but prioritized for any future upstream work).

---

## Output format

```
=== ECDSA PCS Commitment Benchmark ===
Workload:     p256-ecdsa-verify
Trace length: 2^19 = 524288 cycles
log_k_chunk:  4
bytecode_k:   65536 (log 16)
ram_k:        16384 (log 14)
d-factors:    instruction_d=32, bytecode_d=4, ram_d=4

Polynomial inventory:
  Dory  (one-hot, BN254):            336.6M field elements (32 B/elem)
  WHIR  (LogUp*+dense, BN254):       22.1M field elements (32 B/elem)
  WHIR  (LogUp*+dense, Fp3-Gold):    22.1M field elements (24 B/elem)

  Ratio Dory/WHIR (by element):     15.22x  (WHIR is 6.6% of Dory)

Timing:
  Dory       (BN254)     min= 5259.3ms  median= 5287.1ms  max= 5470.7ms  (336.6M elems)
  WHIR-ZK BN254 commit   min= 5281.8ms  median= 5296.6ms  max= 5370.4ms  (22.1M elems  32B/elem)  encode=0.42s
    ├─ claim_eval         min=   88.7ms  median=   90.4ms  max=   92.6ms  (d MLE evals per family)
    ├─ gkr (eq-weighted)  min= 4353.4ms  median= 4478.0ms  max= 4651.8ms  (3 families, max A-depth=24)
    └─ commit + LogUp*   min= 9723.9ms  median= 9864.1ms  max=10113.2ms
  WHIR-ZK Goldilocks Fp3 commit min= 3201.4ms  median= 3201.7ms  max= 3249.4ms  (22.1M elems  24B/elem)  encode=0.09s
    ├─ claim_eval         min=   72.5ms  median=   73.4ms  max=   74.9ms  (d MLE evals per family)
    ├─ gkr (eq-weighted)  min= 2813.4ms  median= 2817.1ms  max= 2874.6ms  (3 families, max A-depth=24)
    └─ commit + LogUp*   min= 6088.2ms  median= 6092.3ms  max= 6198.1ms

Wall-clock ratios (WHIR / Dory):
  WHIR-ZK BN254          commit-only:     1.00x
  WHIR-ZK BN254          commit + LogUp*: 1.87x
  WHIR-ZK Goldilocks Fp3 commit-only:     0.61x
  WHIR-ZK Goldilocks Fp3 commit + LogUp*: 1.15x
```

Both binaries emit JSON reports (`dory.json`, `whir-bn254.json`,
`whir-goldilocks.json`); the orchestrator merges them into `combined.json`.
The WHIR-side report includes a `gkr` block with `min_ms`, `median_ms`,
`max_ms`, plus nested `claim_eval_ms` and `gkr_only_ms` sub-blocks, plus
`per_family_ms` (3 entries: per-family A-side+B-side GKR cost) and
`per_layer_ms` (max-A-depth entries, default 24: per-layer sumcheck cost
summed across the 3 families that reach each layer).

---

## What the numbers mean

Reference measurements from this machine (Apple M-series; 1 warmup × 5
runs; `p256-ecdsa-verify` at T = 2^19; `run-bench.sh --field both`). The
bench reports three timing phases plus the combined total.

### Per-phase wall-clock (medians over 5 measured runs)

| Scheme                       | commit ms | claim_eval ms | gkr ms | total LogUp\* ms | commit + LogUp\* |
| ---------------------------- | --------: | ------------: | -----: | ---------------: | ---------------: |
| **Dory (BN254)** — baseline  | 5287.1    | —             | —      | —                | 5287.1           |
| **WHIR-ZK BN254**            | 5296.6    | 90.4          | 4478.0 | 4568.4           | 9864.1           |
| **WHIR-ZK Goldilocks Fp3**   | 3201.7    | 73.4          | 2817.1 | 2890.5           | 6092.3           |

### Min / median / max across 5 runs

| Scheme                       | min ms | median ms | max ms |
| ---------------------------- | -----: | --------: | -----: |
| **Dory (BN254)**             | 5259.3 | 5287.1    | 5470.7 |
| WHIR-ZK BN254 — commit only  | 5281.8 | 5296.6    | 5370.4 |
| WHIR-ZK BN254 — claim_eval   |   88.7 |   90.4    |   92.6 |
| WHIR-ZK BN254 — gkr          | 4353.4 | 4478.0    | 4651.8 |
| WHIR-ZK BN254 — commit + LogUp\* | 9723.9 | 9864.1 | 10113.2 |
| WHIR-ZK Fp3 — commit only    | 3201.4 | 3201.7    | 3249.4 |
| WHIR-ZK Fp3 — claim_eval     |   72.5 |   73.4    |   74.9 |
| WHIR-ZK Fp3 — gkr            | 2813.4 | 2817.1    | 2874.6 |
| WHIR-ZK Fp3 — commit + LogUp\* | 6088.2 | 6092.3 | 6198.1 |

### Four scenarios vs Dory baseline (wall-clock ratio)

| Scenario | median ms | vs Dory (5287 ms) |
| --- | --: | --: |
| WHIR-ZK BN254, commit only | 5296.6 | **1.00x** (parity) |
| WHIR-ZK BN254, commit + LogUp\* | 9864.1 | **1.87x slower** |
| WHIR-ZK Goldilocks Fp3, commit only | 3201.7 | **0.61x** (≈ 40 % faster) |
| WHIR-ZK Goldilocks Fp3, commit + LogUp\* | 6092.3 | **1.15x slower** |

Numbers above are paper-faithful: §4.1 batching (3 family-level
eq-weighted pushforwards, not 40 per-chunk unweighted histograms),
PAZK §4.5.2 claim reduction, and Figure-1 eq-weighted fan-in-2
fractional GKR. The prior milestone's wrong-shape unweighted-histogram
implementation has been retired. See [DESIGN_WHIR.md](DESIGN_WHIR.md)
for the full mapping to the paper.

### Polynomial inventory at this workload

| Side | What's committed | Total field elements |
|---|---|---:|
| Dory | 40 sparse one-hot chunks + 2 dense (RdInc, RamInc) | 336.6 M |
| WHIR (both fields) | 40 ra_dense + **3 eq-weighted P^F** + 2 dense | **22.1 M** |

Dory/WHIR element ratio: **15.22x** (WHIR commits 6.6 % of Dory's element count).

### Six observations

1. **Field-element reduction**: WHIR's small size class is **3 P^F**
   (one per family — InstructionRa, BytecodeRa, RamRa) instead of 40
   per-chunk. Total committed-element count goes from 23.3 M (pre-§4.1
   milestone) to 22.1 M (~5 % reduction); `commit_ms` benefits more
   than the element count alone suggests because fewer per-class
   commit calls reduces WHIR-side `Config` setup overhead.

2. **`claim_eval_ms` is in the noise** (~70–95 ms). These are the d MLE
   evaluations `M̃^(i)(r_row, r_col)` per family — §4.5.2's inputs. In a
   real protocol they come from upstream sumcheck binding, but the bench
   reports them separately so reviewers can attribute as they see fit.
   Even if you count them in full, they shift the ratios by ≤ 1.5 %.

3. **`gkr_ms` dominates the LogUp\* overhead** at both fields (~98 % of
   `total LogUp*`). InstructionRa's batched A-circuit (depth 24, 32
   chunks concatenated) accounts for ~78 % of `gkr_ms`; the deepest
   sumcheck round (layer 23) is ~32 % of `gkr_ms` on its own.

4. **§4.5.2 prover cost is essentially free** in our setting. The PAZK
   canonical-curve construction for d generic points would cost O(d ·
   log W · 2^{log W}) ≈ 12 G field ops for InstructionRa — but in Jolt the
   d input points share `(r_row, r_col)` and §4.5.2 collapses to a single
   eq-weighted linear combination over the chunk dimension. Documented in
   [DESIGN_WHIR.md §2.1](DESIGN_WHIR.md#21-§452-collapsed-to-a-linear-combination).

5. **Commit-only vs Dory**: WHIR-BN254 commit is at **parity** with
   Dory (1.00x). WHIR-Fp3 commit is **~40 % faster** than Dory commit
   (0.61x), thanks to the 24 B/elem footprint vs Dory's 32 B/elem and
   the 15.22x element-count reduction together overcoming WHIR's higher
   per-element cost.

6. **Combined commit + LogUp\* vs Dory**: BN254 lands at **1.87x slower**;
   **Fp3 at 1.15x slower** — much closer than the previous (incorrect,
   unweighted-histogram) milestone suggested. The interesting future
   question is whether the leaf-claim PCS openings (out of scope for
   this bench) can be done with a single batched WHIR opening at
   `log_t + log_d` variables,
   at which point WHIR's smaller commitment footprint may outweigh the
   GKR overhead.

---

## Bench faithfulness: jolt-prover (Bolt) vs jolt-core

The bench mirrors the **Bolt-codegen commit path** in
`crates/jolt-prover/src/stages/commitment.rs`, not jolt-core's production
prover in `jolt-core/src/zkvm/prover.rs`. Both crates commit the same logical
polynomials and produce the same final commitments, but they differ in
*how* the commit is performed. The bench reports the Bolt path's wall-clock,
which is the slower of the two equivalent paths.

The five divergences (each spelled out with line refs in
[DESIGN_DORY.md §6](DESIGN_DORY.md#6-divergences-from-jolt-cores-production-prover)):

| # | Divergence              | jolt-core | Bolt / bench | Affects ECDSA numbers? |
| - | ----------------------- | --------- | ------------ | ---------------------- |
| 1 | Dory layout / streaming | CycleMajor + streaming | AddressMajor + non-streaming | Yes — bench ~15-25% slower than jolt-core |
| 2 | Dense Fr materialization | Lazy `i128 → Fr` | Eager `Vec<Fr>` | Yes — small (only RdInc/RamInc) |
| 3 | Sparse fast path        | CycleMajor specialization | Generic path | Yes — consequence of #1 |
| 4 | Commit-plan construction | Dynamic (runtime) | Static (codegen per workload) | No — bench bypasses Bolt's static plan |
| 5 | `ram_K` bytecode clamp  | Clamped | **Now clamped (was missing)** | No for ECDSA; previously yes for sparse-RAM workloads |

Bottom line: the bench's Dory wall-clock measures the Bolt AddressMajor +
non-streaming + eager-Fr path. Against jolt-core's production CycleMajor +
streaming + lazy-i128 path, expect ~15-30% headroom in Dory's favor. The
"WHIR-Fp3 commit is ~15% faster than Dory" claim is therefore *true vs the
Bolt path the bench measures, possibly false vs jolt-core's optimized
path*. Adding a `--prover-path {bolt, jolt-core}` flag is a ~2-3 hour
follow-up if you want production-vs-WHIR numbers. The GKR-vs-Dory
combined ratio (~1.32x for Fp3) shifts further in Dory's favor under that
adjustment.

---

## What is NOT in this benchmark

(Explicit so the scope of the conclusion above is unambiguous.)

- **PCS opening proofs.** The bench stops once the WHIR commit returns
  and the GKR leaf-claim transmissions are sent. A real LogUp\* prover
  would continue with batched PCS openings on `ra_dense` and `P` at the
  GKR's final randomness; that cost is not measured.
- **Sumcheck / Spartan / BlindFold prover overhead.** This bench is
  about the polynomial-commitment phase, not the rest of the Jolt
  proving pipeline.
- **Anything that touches the Jolt prover crate.** Read-only with
  respect to `crates/jolt-prover`.
- **Dory on a non-BN254 field.** Dory is the baseline; it must match the
  field Jolt's prover actually uses.
- **Verifier work.** The bench measures the prover only. Internal
  soundness is enforced via the GKR root cross-multiplication assert
  (`N_A·D_B == N_B·D_A` per pair).
- **BabyBear / KoalaBear / Mersenne31.** Adding any of these requires a
  new `MontConfig` plus an extension field (Fp4 or Fp5) for 128-bit
  soundness, plus reduction handling for the `i128` `RdInc`/`RamInc`
  values which don't fit in a 31-bit prime. Deferred follow-up.

---

## Critical files referenced

Papers (workspace root):

- [twist_shout.pdf](../../TwistShout.pdf) — original Twist/Shout (Shout: §5.1,
  Twist: §5.2 of the LogUp\* paper's numbering; the original paper uses
  different section numbers).
- [twist_shout_logup_star.pdf](../../twist_shout_logup_star.pdf) — LogUp\*
  reformulation. Sections used by the bench: §4.1 (batching across pairs),
  §5.1 (Shout via ra_dense), §5.2 (Twist via wa_dense + virtual wv/Val),
  Figure 1 (GKR pushforward circuit), eq. 6 (Inc-evaluation, out of scope).

Bench design docs:

- [DESIGN_DORY.md](DESIGN_DORY.md) — Dory bench ↔ Twist/Shout ↔ Jolt code.
- [DESIGN_WHIR.md](DESIGN_WHIR.md) — WHIR + LogUp\* + GKR bench ↔ paper ↔ Jolt code.
- [PROFILING.md](PROFILING.md) — span tree breakdown, dhat allocation, GKR per-layer table.

Jolt internals (referenced by both DESIGN docs):

- ECDSA guest: [examples/p256-ecdsa-verify/guest/src/lib.rs](../../examples/p256-ecdsa-verify/guest/src/lib.rs)
- Trace extraction: [crates/jolt-trace/src/extract.rs](../jolt-trace/src/extract.rs)
- Witness builders: [crates/jolt-witness/src/lib.rs](../jolt-witness/src/lib.rs)
- One-hot config: [jolt-core/src/zkvm/config.rs](../../jolt-core/src/zkvm/config.rs)
- Reference prover commit path (mirrored, not modified):
  [crates/jolt-prover/src/stages/commitment.rs](../jolt-prover/src/stages/commitment.rs)
- Dory commit: [crates/jolt-dory/src/scheme.rs](../jolt-dory/src/scheme.rs)
- WHIR-ZK commit: `../../../whir/src/protocols/whir_zk/`
- GKR prover (bench-local): `../../../whir-pcs-bench/src/gkr.rs`
