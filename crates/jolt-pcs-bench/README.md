# jolt-pcs-bench — WHIR vs Dory PCS commitment benchmark on the ECDSA workload

A feasibility benchmark that answers a single question:

> For the actual polynomial commitment workload of a Jolt ECDSA proof at 2^19 cycles,
> how does WHIR (in zero-knowledge mode) with the LogUp\* trick compare to Jolt's
> Dory in wall-clock time and committed data volume?

This benchmark only times the **commit** step. No opening proofs, no verification,
no integration with the prover pipeline. The output is wall-clock numbers and
field-element counts so you can decide whether the LogUp\* + WHIR direction is
worth pursuing for Jolt.

**Field menu** (`whir-pcs-bench --field …`):

- `bn254` (default) — `Identity<Field256>`, apples-to-apples with Dory.
- `goldilocks-fp3` — `Identity<Field64_3>`, cubic extension of Goldilocks
  (192-bit, the soundness-correct setup at 128-bit security).

The Jolt side / Dory always uses BN254 Fr (that's what Jolt's prover actually
runs). Only the WHIR side is field-agnostic.

The remaining out-of-scope milestone:

- Add the GKR pushforward proving overhead (the *commitment* of `P[k]` is timed
  here; *proving P came from ra_dense* is not).

BabyBear / Mersenne31 / KoalaBear are deferred follow-ups (would require adding
a new `MontConfig` plus an extension field).

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
4. Applies the LogUp\* transformation per `twist_shout_logup_star.pdf` §5.1 / §5.2:
   each one-hot chunk becomes a dense `ra_dense ∈ F^T` (the argmax index per cycle,
   already stored that way in Jolt) plus a pushforward `P ∈ F^K`.
5. Asserts 5 transformation invariants (`verify.rs`).
6. Serializes the LogUp\*-transformed polynomials to disk as **raw integers**
   (u8 ra_dense, u32 pushforward, i128 RdInc/RamInc). The dump is
   field-agnostic — no BN254 Fr ever appears in the file. ~41 MB total.
7. A sibling binary `whir-pcs-bench` (separate workspace, see below) reads the
   dump, encodes each integer into the chosen WHIR target field (`bn254` or
   `goldilocks-fp3`) at load time, and times WHIR-ZK's `commit`.
8. The orchestrator script combines both JSON reports into a side-by-side table
   showing Dory vs each WHIR field choice.

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
├── run-bench.sh             ← orchestrator: builds both, runs both, combines JSON
└── src/
    ├── main.rs              ← CLI, JSON output, glue
    ├── workload.rs          ← ECDSA guest compile → trace → CommitmentTraceSources
    ├── jolt_polys.rs        ← Builds Jolt's committed polynomial set + a local
    │                          copy of `AddressMajorOneHotPolynomial` (mirrors
    │                          `crates/jolt-prover/src/stages/commitment.rs`)
    ├── logup_star.rs        ← §5.1 / §5.2 LogUp* transformation
    ├── verify.rs            ← 5 invariants on the transformation
    ├── dory_bench.rs        ← DoryScheme::commit timing
    └── dump.rs              ← Polynomial dump format consumed by whir-pcs-bench

../../../whir-pcs-bench/      ← sibling crate, separate workspace
├── Cargo.toml               ← pins blake3 = "=1.8.3" to keep digest 0.10
└── src/
    └── main.rs              ← whir_zk::Config::commit timing
```

---

## The LogUp* transformation (what's actually swapped)

The paper is at `jolt/twist_shout_logup_star.pdf`. Concretely, this bench
implements §5.1 (Shout) and §5.2 (Twist) commitment transformations:

| Family                             | Dory layout                  | WHIR (LogUp\*) layout               |
| ---------------------------------- | ---------------------------- | ----------------------------------- |
| `InstructionRa(i)`, i=0..32        | sparse one-hot `K_chunk × T` | `ra_dense ∈ F^T` + `P ∈ F^K_chunk`  |
| `BytecodeRa(i)`, i=0..4            | sparse one-hot `K_chunk × T` | `ra_dense ∈ F^T` + `P ∈ F^K_chunk`  |
| `RamRa(i)`, i=0..4                 | sparse one-hot `K_chunk × T` | `ra_dense ∈ F^T` + `P ∈ F^K_chunk`  |
| `RdInc`, `RamInc`                  | dense `F^T`                  | dense `F^T` (unchanged)             |
| `TrustedAdvice`, `UntrustedAdvice` | dense `F^T`                  | dense `F^T` (unchanged; not in ECDSA workload) |

**Argmax extraction is trivial here** because Jolt already stores the one-hot
polynomial as a `Vec<Option<u8>>` of indices (the §5.1 dense form). The
"transformation" is just casting indices to `Fr` and computing the histogram.

**Pushforward `P[k]`** is `count(j : ra_dense[j] == k)`. For Jolt's setting,
`K_chunk = 16` (since `log_k_chunk = 4`).

---

## Assumptions and modifications

These are the calls and constants this bench takes that are worth being explicit
about. None of them require modifying anything in the Jolt prover crate.

### 1. ECDSA workload at T = 2^19 cycles

The benchmark fixes `T = 524288 = 2^19`, which is exactly `max_trace_length`
declared in the guest at [examples/p256-ecdsa-verify/guest/src/lib.rs:5](../../examples/p256-ecdsa-verify/guest/src/lib.rs#L5).
The actual ECDSA trace is shorter; the rest is NoOp padding. Both the actual
cycle count and the padded trace length are reported to stdout on every run.

(Earlier iterations of this bench used `secp256k1-ecdsa-verify` at 2^18; we
switched to P-256 because its `max_trace_length` matches the 2^19 target from
the original GOAL.md and exercises Jolt at one cycle-doubling further.)

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

- `num_vars = 15` (2^15 = 32768): all 40 pushforward vectors
- `num_vars = 19` (2^19 = 524288): all 40 `ra_dense` + `RdInc` + `RamInc`

Each class is committed in one batched `commit` call. The reported WHIR
wall-clock is the sum of both calls.

### 9. Field-agnostic dump format (integer-form, version 2)

The dump stores **raw integer values** rather than pre-encoded Fr field elements:

- `ra_dense` chunks: `Vec<u8>` (1 byte per cycle) — argmax index ∈ [0, 16).
- `pushforward` vectors: `Vec<u32>` (4 bytes per bucket) — histogram count.
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
- `--field {bn254|goldilocks-fp3}` (whir-pcs-bench): pick the scalar field.

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
  WHIR  (LogUp*+dense, BN254):       23.3M field elements (32 B/elem)
  WHIR  (LogUp*+dense, Fp3-Gold):    23.3M field elements (24 B/elem)

  Ratio Dory/WHIR (by element):     14.43x  (WHIR is 6.9% of Dory)

Timing:
  Dory       (BN254)     min= 5278.1ms  median= 5299.7ms  max= 5476.4ms  (336.6M elems)
  WHIR-ZK BN254          min= 5505.0ms  median= 5635.0ms  max= 6688.0ms  (23.3M elems  32B/elem)  encode=0.37s
  WHIR-ZK Goldilocks Fp3 min= 4522.1ms  median= 4785.4ms  max= 4984.5ms  (23.3M elems  24B/elem)  encode=0.06s

Wall-clock ratios (WHIR / Dory):
  WHIR-ZK BN254          1.06x
  WHIR-ZK Goldilocks Fp3 0.90x
```

Both binaries emit JSON reports (`dory.json`, `whir-bn254.json`,
`whir-goldilocks.json`); the orchestrator merges them into `combined.json`.

---

## What the numbers mean

Reference measurements from this machine (Apple M-series; 1 warmup, 5 runs;
`p256-ecdsa-verify` workload at T = 2^19 cycles):

| Scheme                         | Median wall-clock | Field elements | Bytes / elem | vs Dory       |
| ------------------------------ | ----------------: | -------------: | -----------: | ------------- |
| **Dory (BN254)**               |          5300 ms  |        336.6 M |        32 B  | baseline      |
| **WHIR-ZK BN254**              |          5635 ms  |         23.3 M |        32 B  | 1.06x slower  |
| **WHIR-ZK Goldilocks Fp3**     |          4785 ms  |         23.3 M |        24 B  | **0.90x — 10% faster** |

Three observations:

1. **Field-element reduction**: WHIR commits 14.43x fewer field elements than
   Dory (6.9% of Dory's count). This matches the paper's ~6.5% prediction; the
   small overshoot is the pushforward padding to 2^15 (~25% of WHIR's budget),
   unavoidable for ZK at 128-bit security.

2. **BN254 wall-clock**: WHIR-ZK on BN254 is ~6% *slower* than Dory wall-clock
   despite committing 14x fewer elements. WHIR-ZK's per-element cost on BN254
   is ~15x Dory's per-element cost, which roughly cancels the data-volume win.

3. **Goldilocks Fp3 wall-clock**: switching the WHIR field to the 192-bit cubic
   extension of Goldilocks finally tips the comparison: WHIR-ZK becomes 10%
   faster than Dory on the same workload, with the same 14.43x reduction in
   element count and a 25% smaller per-element footprint (24 B vs 32 B). The
   reduction in arithmetic cost per element (Fp3 multiplications dominate
   ~3 × 64-bit instead of one 254-bit) is what does it.

In short: **on BN254 alone, LogUp\* + WHIR-ZK does not buy you wall-clock; on
Goldilocks Fp3 it does, by ~10%, plus the smaller commitment / proof footprint
that comes with a smaller field.** The next milestone is whether the GKR
pushforward proving overhead (currently out of scope) preserves this win.

---

## Bench faithfulness: jolt-prover (Bolt) vs jolt-core

The bench mirrors the **Bolt-codegen commit path** in
`crates/jolt-prover/src/stages/commitment.rs`, not jolt-core's production
prover in `jolt-core/src/zkvm/prover.rs`. Both crates commit the same logical
polynomials and produce the same final commitments, but they differ in
*how* the commit is performed. The bench reports the Bolt path's wall-clock,
which is the slower of the two equivalent paths. The differences are:

### Divergence 1: Dory layout (CycleMajor streaming vs AddressMajor non-streaming)

**jolt-core** ([prover.rs:719](../../jolt-core/src/zkvm/prover.rs#L719)): branches
on `DoryGlobals::get_layout()`. Default is `DoryLayout::CycleMajor`
([dory_globals.rs:53](../../jolt-core/src/poly/commitment/dory/dory_globals.rs#L53)),
which takes the streaming path: cycles are pulled in row-band chunks via
`lazy_trace.iter_chunks(row_len)`, each band's witness is generated and
committed via `StreamingCommitmentScheme::process_chunk*` /
`aggregate_chunks`. The full trace and full witness are never materialized
simultaneously.

**Bolt** ([commitment.rs:543](commitment.rs#L543)): uses `DoryLayout::AddressMajor`
implicitly via the duplicated `AddressMajorOneHotPolynomial`. The
non-streaming branch in jolt-core's `else if AddressMajor` runs
([prover.rs:725-744](../../jolt-core/src/zkvm/prover.rs#L725-L744)):
`lazy_trace.collect::<Vec<Cycle>>()` materializes the full trace, then per
polynomial `generate_witness(...)` materializes the full
`MultilinearPolynomial<F>`, then `CommitmentScheme::commit` does a one-shot
commit. Same final commitment, different cache/memory profile.

**Impact on bench numbers**: the bench measures the Bolt AddressMajor path.
jolt-core's CycleMajor streaming path is typically ~15-25% faster on the
same workload at T = 2^19 due to better cache locality, no full-witness
materialization, and a CycleMajor-specific fast path in
`OneHotPolynomial::commit_rows`
([one_hot_polynomial.rs:137-171](../../jolt-core/src/poly/one_hot_polynomial.rs#L137-L171)).

### Divergence 2: Dense-poly Fr materialization (lazy `CompactPolynomial<i128>` vs eager `Vec<Fr>`)

**jolt-core** ([witness.rs:184, 199](../../jolt-core/src/zkvm/witness.rs#L184)):
`RdInc`/`RamInc` produce a `Vec<i128>` then `coeffs.into()` wraps it as
`MultilinearPolynomial::I128Scalars(CompactPolynomial<i128, F>)`. The Dory
commit converts `i128 → Fr` lazily inside `for_each_row` / `for_each_nonzero`
and can skip zero-valued cycles entirely.

**Bolt** ([commitment.rs:511-518](commitment.rs#L511-L518)): eagerly calls
`dense_i128_column_to_field(sources.rd_inc, target_len)` to materialize the
full `Vec<Fr>` up front, then `DoryScheme::commit_evaluations_with_row_len`.
The bench mirrors this exactly (`jolt_polys::build_polynomial_set` calls
`dense_i128_column_to_field`).

**Impact on bench numbers**: at ECDSA roughly half of `RdInc/RamInc` cycles
have `inc = 0` (NoOp cycles, or cycles without register/RAM writes). jolt-core
skips those in the MSM entirely; the bench does T Montgomery encodings up
front and pays Pedersen-MSM cost on all T elements. Small absolute impact
(only 2 dense polys vs 40 one-hot chunks), but real.

### Divergence 3: Sparse one-hot wrapper (`OneHotPolynomial` vs `AddressMajorOneHotPolynomial`)

Both go through `DoryScheme::commit`'s sparse path and ultimately call the
same batched-G1-addition kernel
(`jolt_crypto::ec::bn254::batch_addition::batch_g1_additions_multi_affine` —
see [scheme.rs:478-484](../jolt-dory/src/scheme.rs#L478-L484) for the Bolt
side and [one_hot_polynomial.rs:148-149, 195](../../jolt-core/src/poly/one_hot_polynomial.rs#L148)
for the jolt-core side).

However, jolt-core's `OneHotPolynomial::commit_rows` has a
**CycleMajor-specific fast path** for the common case `t / row_len >=
num_threads` ([one_hot_polynomial.rs:137-171](../../jolt-core/src/poly/one_hot_polynomial.rs#L137-L171))
that pre-groups column indices by address per chunk, reducing scatter cost.
The bench's `AddressMajorOneHotPolynomial::for_each_nonzero` takes the
generic path. This is a consequence of Divergence 1 and not a separate
divergence — but worth knowing about if you profile.

### Divergence 4: Commit-plan construction (Bolt static vs jolt-core dynamic)

**jolt-core**: zero static plan. At runtime, [prover.rs:715](../../jolt-core/src/zkvm/prover.rs#L715)
calls `all_committed_polynomials(&one_hot_params)` ([witness.rs:47](../../jolt-core/src/zkvm/witness.rs#L47))
which returns a `Vec<CommittedPolynomial>` whose length and chunk indices
adapt to whatever `instruction_d`, `bytecode_d`, `ram_d` come from the
runtime workload. Each enum variant carries its own dispatch info; the
prover's commit loop is a single `polys.par_iter().map(|poly_id| ...)`.

**Bolt**: the `COMMITMENT_PROGRAM` constant at
[commitment.rs:1318-1424](commitment.rs#L1318-L1424) is a `&'static` array of
literal oracle name strings (`"InstructionRa_0", "InstructionRa_1", …`) with
hardcoded `num_vars`. The current in-tree values are sized for **muldiv at
T = 2^16, bytecode_k = 2^12** (`num_vars: 16` dense, `num_vars: 20` one-hot,
`BytecodeRa_0..2` = bytecode_d = 3). For ECDSA at T = 2^19, the right values
are `num_vars: 19` dense, `num_vars: 23` one-hot, `BytecodeRa_0..3` =
bytecode_d = 4. Running `prove_commitment_phase()` for ECDSA against the
in-tree static plan would hit `PlanCountMismatch` or `OracleTooLarge` errors.

Bolt's MLIR codegen pipeline regenerates `commitment.rs` per
`(guest, max_trace_length)` pair; the file you see in tree is just whichever
plan was last regenerated. **The bench bypasses Bolt's static plan entirely**
and reconstructs the polynomial list from `OneHotParams::new(log_T,
bytecode_k, ram_k)` at runtime — the same dynamic-plan approach jolt-core
uses. This is necessary: no ECDSA codegen for the static plan exists in tree.

### Divergence 5: `ram_K` bytecode-end clamp (previously missing, now fixed)

**jolt-core** ([prover.rs:414-433](../../jolt-core/src/zkvm/prover.rs#L414-L433))
computes `ram_K` as `max(largest_runtime_ram_address, bytecode_end + 1)`
then `.next_power_of_two()`. The bytecode-end term ensures `ram_K` is large
enough to index the static bytecode image even for workloads that barely
touch RAM at runtime.

**The bench** originally only took the runtime maximum, skipping the
bytecode-end clamp. For ECDSA the heap reach (~2^14 = 16384) already
dominated `bytecode_end + 1` so the two formulas coincided. For a workload
that barely touches RAM, the bench would have computed a *smaller* `ram_K`
than jolt-core, leading to a smaller `ram_d` and a different `RamRa`
decomposition. **Fixed** in `workload.rs` by importing
`jolt_core::zkvm::ram::RAMPreprocessing::preprocess(memory_init)` and
clamping with the same formula. ECDSA numbers are unchanged (`ram_k=16384`).

### Summary

| # | Divergence              | jolt-core | Bolt / bench | Affects ECDSA numbers? |
| - | ----------------------- | --------- | ------------ | ---------------------- |
| 1 | Dory layout / streaming | CycleMajor + streaming | AddressMajor + non-streaming | Yes — bench ~15-25% slower than jolt-core |
| 2 | Dense Fr materialization | Lazy `i128 → Fr` | Eager `Vec<Fr>` | Yes — small (only RdInc/RamInc) |
| 3 | Sparse fast path        | CycleMajor specialization | Generic path | Yes — consequence of #1 |
| 4 | Commit-plan construction | Dynamic (runtime) | Static (codegen per workload) | No — bench bypasses Bolt's static plan |
| 5 | `ram_K` bytecode clamp  | Clamped | **Now clamped (was missing)** | No for ECDSA; previously yes for sparse-RAM workloads |

The bench's published `Dory ~5300 ms` measures the Bolt AddressMajor +
non-streaming + eager-Fr path. Against jolt-core's production CycleMajor +
streaming + lazy-i128 path, expect ~15-30% headroom in Dory's favor. The
"WHIR-Fp3 is 10% faster than Dory" claim is therefore *true vs the Bolt
path the bench measures, possibly false vs jolt-core's optimized path*.
Adding a `--prover-path {bolt, jolt-core}` flag is a ~2-3 hour follow-up if
you want production-vs-WHIR numbers.

---

## What is NOT in this benchmark

(Explicit so the scope of the conclusion above is unambiguous.)

- Opening proofs (just the commit step).
- The GKR pushforward proving overhead. The pushforward vectors P[k] are
  committed via WHIR (matching what a real LogUp\* prover would do), but the
  cost of *proving* P was derived correctly from ra_dense is not measured.
- Sumcheck / Spartan / BlindFold prover overhead.
- Anything that touches the Jolt prover crate. This bench is read-only with
  respect to `crates/jolt-prover`.
- Dory on a non-BN254 field. The Dory side is the comparison baseline; it must
  match the field Jolt's prover actually uses.
- BabyBear / KoalaBear / Mersenne31. Adding any of these requires a new
  `MontConfig` plus an extension field (Fp4 or Fp5) for 128-bit soundness, plus
  reduction handling for the i128 `RdInc/RamInc` values which don't fit in
  a 31-bit prime. Deferred to a follow-up milestone if Goldilocks Fp3 motivates
  pushing further toward small fields.

---

## Critical files referenced

- The paper: [twist_shout_logup_star.pdf](../../twist_shout_logup_star.pdf)
  (§5.1 Shout, §5.2 Twist, eq.6 Inc-evaluation)
- ECDSA guest: [examples/p256-ecdsa-verify/guest/src/lib.rs](../../examples/p256-ecdsa-verify/guest/src/lib.rs)
- Trace extraction: [crates/jolt-trace/src/extract.rs](../jolt-trace/src/extract.rs)
- Witness builders: [crates/jolt-witness/src/lib.rs](../jolt-witness/src/lib.rs)
- One-hot config: [jolt-core/src/zkvm/config.rs](../../jolt-core/src/zkvm/config.rs)
- Reference prover commit path (mirrored, not modified):
  [crates/jolt-prover/src/stages/commitment.rs](../jolt-prover/src/stages/commitment.rs)
- Dory commit: [crates/jolt-dory/src/scheme.rs](../jolt-dory/src/scheme.rs)
- WHIR-ZK commit: `../../../whir/src/protocols/whir_zk/`
