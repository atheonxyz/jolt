# Phase 1 — Goldilocks field + base-field-limb witness + WHIR base-commit

**Status: complete and verified.** This document records exactly what was built in
Phase 1 of the Jolt commitment-stack migration (BN254 + Dory → Goldilocks + WHIR),
how the Goldilocks path works today, what it does *not* yet do (Phase 2), how to run
and verify it, and the full list of files to commit.

It is the implementation/status companion to the higher-level `JOLT_GOLDILOCKS_DESIGN.md`.

---

## 1. Goal and scope

The migration replaces Jolt's BN254 + Dory commitment with a **hash-based PCS (WHIR)
over a small field (Goldilocks)**. Phase 1 stands up the *front* of the new pipeline
and proves it on a live trace:

- the **Goldilocks field** (base `Fp` + cubic extension `Fp3`),
- the **limb decomposition** of witness values into base-field columns,
- the **base-field witness representation** (`ra_dense` index columns + `Inc` limbs),
- the **WHIR base-commit** of those columns,
- a **single-point open/verify** sanity round-trip, and
- a **live fibonacci end-to-end** that commits a real trace and compares the result,
  on the *same trace*, against the **actual Jolt BN254 + Dory protocol**.

Everything new is **feature-gated behind `goldilocks`**. The existing
BN254 / Dory / BlindFold path and all sumcheck/IOP stages are **untouched** — Phase 1
adds code, it does not modify the proving system.

### What is explicitly NOT in Phase 1 (deferred to Phase 2)

- Sumcheck / IOP changes; the `JoltField::Challenge` associated type and challenge ops.
- LogUp\* pushforward GKR and the `P^F` one-hot lift.
- Range-check sumchecks and the carry / `2⁻³²` recomposition **constraints** over the
  limbs (Phase 1 produces the limb *columns*; it does not constrain them).
- Batched opening proofs, the `CommitmentScheme` trait impl for WHIR, the shared
  spongefish transcript.
- **Hiding** (`whir_zk` over `Basefield`). Phase 1 commits in the clear (sound,
  non-hiding).
- A Goldilocks-specialized accumulator fast path (Phase 1 aliases the existing
  `NaiveAccumulator` / `NaiveScalarAccumulator`; no sumcheck inner loop runs yet).

---

## 2. Architecture / data flow

```
fibonacci trace ──(jolt-trace front-end, mirrors jolt-pcs-bench/src/workload.rs)──▶ CommitmentTraceSources
                                                                                          │ (jolt-witness)
crates/jolt-field   (feature `goldilocks`, whir-free)                                     ▼
  goldilocks/{base, ext3, decompose}     ──used by──▶   crates/jolt-witness/src/goldilocks.rs
   Goldilocks, GoldilocksFp3, value↔limb                  (i128 Inc → base-limb cols;
        │                                                  one-hot indices → ra_dense index cols)
        │                                                                       │ GoldilocksWitnessColumns
        ▼                                                                       ▼
crates/jolt-whir    (feature `goldilocks`; depends on whir + jolt-field + jolt-witness)
  convert.rs (Goldilocks→Field64, Fp3→Field64_3) │ params.rs │ commit.rs commit_witness() │ sanity.rs (open+verify)
        │                                                                       │
        └──── crates/jolt-pcs-bench/src/fib_goldilocks.rs (#[ignore] e2e): live trace →────┘
              base-limb columns → WHIR commit → validate → compare vs BN254/Dory (build_polynomial_set + bench_dory)
```

`jolt-poly`'s polynomial types and `jolt-witness`'s helpers are already field-agnostic
(`<F: Field>`), so the new field slots in with **no enum surgery**. The single point of
contact with WHIR's arkworks types is `crates/jolt-whir/src/convert.rs`; `jolt-field`
itself stays free of any `whir`/arkworks dependency.

---

## 3. Component A — Goldilocks field (`crates/jolt-field/src/goldilocks/`)

Feature `goldilocks` (no runtime deps). `num-bigint` is a **dev-dependency only** (the
correctness oracle). Both field types implement the lean `Field` trait
(`crates/jolt-field/src/field.rs`) — serde-based, **no** arkworks requirement, **no**
`Challenge` type yet.

### `base.rs` — `Goldilocks` (389 lines)

- Field `p = 2⁶⁴ − 2³² + 1 = 0xFFFF_FFFF_0000_0001`. **Montgomery-free.**
- Constant `EPSILON = 2³² − 1 = 0xFFFF_FFFF`, which is `2⁶⁴ mod p`.
- **Non-canonical representation** in `[0, 2⁶⁴)`: every stored value is
  `< 2⁶⁴ = p + EPSILON < 2p`, so canonicalizing is at most **one conditional
  subtract**. Eq / `Hash` / serialize use the canonical form (`to_canonical_u64`).
- `reduce128` uses `2⁶⁴ ≡ 2³² − 1` and `2⁹⁶ ≡ −1 (mod p)` to fold a 128-bit product
  into the field; `mul` is `u64×u64→u128` then `reduce128`. `add`/`sub` apply the
  `EPSILON` correction.
- `inverse` via square-and-multiply `aᵖ⁻²`; `random` rejection-samples `x < p`.
- `NUM_BYTES = 8`; `to_bytes` writes the canonical 8 LE bytes into the low bytes of a
  `[u8; 32]` (rest zero).

### `ext3.rs` — `GoldilocksFp3` (302 lines)

- Cubic extension `Fp[x]/(x³ − 2)` (nonresidue **2**), stored as `[Goldilocks; 3]`.
  This is the Phase-2 challenge/eval field; it is built and tested now.
- `mul` = schoolbook with the `x³ → 2` reduction (9 base muls):
  `c0 = a0·b0 + 2(a1·b2 + a2·b1)`, `c1 = a0·b1 + a1·b0 + 2·a2·b2`,
  `c2 = a0·b2 + a1·b1 + a2·b0`.
- **`mul_by_base(b)` = 3 base muls** (the Phase-2 sumcheck hot path).
- `inverse` via the field norm / cofactor (adjugate) formula.
- `NUM_BYTES = 24`. Basis convention matches WHIR's `Field64_3` exactly (see §5).

### `decompose.rs` — value ↔ base-field limbs (68 lines)

- `LIMB_BITS = 32`.
- `u64_to_limbs(v) -> [Goldilocks; 2]` / `limbs_to_u64` — a 64-bit value as two 32-bit
  limbs.
- `i128_to_sign_limbs(v) -> (Goldilocks /*sign*/, [Goldilocks; 2] /*lo, hi*/)` and
  `sign_limbs_to_i128` — the signed **i65 `Inc`** transition (`RdInc`/`RamInc`) as a
  sign flag plus two unsigned 32-bit limbs (`debug_assert |v| < 2⁶⁴`).

### `tests.rs` — the correctness de-risk (237 lines)

A **pure `num-bigint` reference** (no crypto deps): every base and `Fp3` op is checked
against an independent big-integer implementation of the same field (`p`; `Fp3` mod
`x³ − 2`) over random inputs and edges — `0`, `1`, `p−1`, the `[p, 2⁶⁴)` aliasing band,
and i65 extremes — plus limb decompose→recompose round-trips.

> The oracle deliberately lives in `jolt-field` as a **`num-bigint`** check, *not* an
> arkworks/`whir` check: adding `whir`/`Field64` as a `jolt-field` dev-dependency would
> drag `digest 0.10` into the workspace's unified `digest 0.11` resolution. The
> arkworks cross-check instead lives in `jolt-whir` (§5), where `whir` is already a dep.

---

## 4. Component B — base-field-limb witness (`crates/jolt-witness/src/goldilocks.rs`, 142 lines)

Feature `goldilocks` forwards `jolt-field/goldilocks`. This module turns the
field-agnostic `CommitmentTraceSources` (the dense per-cycle index vectors the prover's
commitment phase consumes) into base-Goldilocks **committed columns**, all padded to
`2^log_t`:

- **`ra_dense`** — one **dense index column per (family, chunk)**. Reuses
  `one_hot_chunk_indices(...)` (MSB-first `Vec<Option<u8>>`) and maps `Some(k) →
  Goldilocks::from_u64(k)`, `None → 0`. There is **no** one-hot lift / `P^F` (Phase 2);
  the column stores the chunk *index* directly. Families: `InstructionRa`, `BytecodeRa`
  (padding `Some(0)`), `RamRa` (padding `None`).
- **`Inc` limbs** — `RdInc` and `RamInc` (`Vec<i128>`) each decompose into three columns
  via `decompose.rs`: `.sign`, `.lo`, `.hi`.

Types: `GoldilocksColumn { label, values }`, `FamilyLayout { label, num_chunks,
chunk_bits, padding }`, `GoldilocksLayout { trace_len, instruction, bytecode, ram }`,
and `GoldilocksWitnessColumns { log_t, columns }` with
`build(sources, layout) -> Self` and `total_elements()`.

For the fibonacci geometry (`log_k_chunk = 4`, `instruction_d = 32`, `bytecode_d = 4`,
`ram_d = 4`) this yields **46 columns** = `(32 + 4 + 4)` `ra_dense` + `2 Inc × 3 limbs`.

The recompose / range-check **constraints** over these limbs are Phase 2; Phase 1 only
produces the columns. (Phase 1 *does* verify recomposition as a test assertion — see §7.)

---

## 5. Component C — WHIR base-commit (`crates/jolt-whir/`, new workspace member)

Feature `goldilocks` pulls `whir` (path dep, like `whir-pcs-bench`) and `ark-ff`. Deps:
`jolt-field`, `jolt-poly`, `jolt-witness`, `whir`, `ark-ff`. The crate is empty without
the feature (`#![cfg(feature = "goldilocks")]`).

- **`convert.rs`** — the single WHIR seam. `Goldilocks → whir::algebra::fields::Field64`
  (`to_canonical_u64` → `Field64::from`), `GoldilocksFp3 → Field64_3`,
  `column_to_field64`. WHIR's `Field64` is arkworks Montgomery `Fp64`; ours is
  Montgomery-free; both represent the same field, so conversion is a canonical-`u64`
  round-trip.
- **`params.rs`** — `whir_params()`: security 128, `pow_bits` 20, folding factor 4,
  rate 1/2 (`starting_log_inv_rate = 1`), list decoding, Blake3 Merkle. Matches the
  `whir-pcs-bench` configuration.
- **`commit.rs`** — `commit_witness(cols) -> CommitReport`. Builds
  `Config::<Basefield<Field64_3>>::new(size, &params)` and commits every base-Goldilocks
  column through one Fiat-Shamir transcript. **The `Basefield<Field64_3>` embedding is
  the key choice**: the committed alphabet is base `Field64` (**8 B/elem**), while folds
  and challenges live in the `Fp3` extension. Witnesses are dropped after each commit so
  peak memory stays at one codeword. `CommitReport` records `log_t`, `num_columns`,
  `total_base_elements`, `committed_base_bytes`, `commit_ms`.
- **`sanity.rs`** — `sanity_roundtrip(values) -> bool`. Full commit → open at a
  pseudo-random multilinear point → verify → `FinalClaim::verify` on a single column.
  Uses a functional-correctness config (`security_level 32`, `pow_bits 0`) so it is fast;
  an honest proof verifies at any security level.

### `crates/jolt-whir/tests/`

- **`commit.rs`** (2 tests) — synthetic trace → `GoldilocksWitnessColumns::build` →
  `commit_witness` + `sanity_roundtrip` + limb recompose. Validates the full
  field/limb/commit path on non-degenerate synthetic data.
- **`crosscheck.rs`** (4 tests) — the **arkworks oracle from the commit side**: the
  hand-coded `Goldilocks` / `GoldilocksFp3` arithmetic must agree, op-for-op, with
  WHIR's `Field64` / `Field64_3` (`add`/`sub`/`mul`/`neg`/`square`/`inverse`/
  `mul_by_base`/embed) over 2000+ random samples plus edges. This is sound because
  WHIR's `Field64_3` uses **`NONRESIDUE = 2`** (`whir/src/algebra/fields.rs`), the same
  `x³ = 2` convention as `ext3.rs`.

---

## 6. Component D — live fibonacci e2e + BN254/Dory comparison

`crates/jolt-pcs-bench/src/fib_goldilocks.rs` (371 lines), gated `#[cfg(feature =
"goldilocks")]`, `#[ignore]`'d because it compiles a RISC-V guest. It is hosted in
`jolt-pcs-bench` (which already deps `jolt-core` / `jolt-trace` / `jolt-witness` and the
Dory machinery) so `jolt-whir` stays lean.

It mirrors `workload.rs` for the **fibonacci** guest: `Program::new("fibonacci-guest")
.set_func("fib")` → `decode` → `BytecodePreprocessing::preprocess` → `trace(fib(n))` →
`extract_trace::<_, Fr>` → `commitment_trace_sources`, with `OneHotParams::new(log_T,
bytecode_k, ram_k)` for the geometry. It then:

1. maps `OneHotParams → GoldilocksLayout`, builds `GoldilocksWitnessColumns`;
2. asserts the `Inc` limbs **recompose** to the original `rd_inc` / `ram_inc`;
3. `commit_witness` over base Goldilocks, then `sanity_roundtrip` on a dynamically
   chosen **non-zero** column (real witnesses contain all-zero columns — see §8);
4. builds the **actual Jolt protocol** polynomial set on the *same trace*
   (`jolt_polys::build_polynomial_set`: sparse one-hot RA families + dense `Inc` polys)
   and commits it via **BN254 + Dory** (`dory_bench::bench_dory`, the production path);
5. prints a side-by-side report.

### Measured result (fib(1000), single run; wall-clock is hardware-dependent)

```
guest                 : fibonacci-guest::fib(1000)  (11222 cycles)
committed length      : 2^16 = 65536   (log_k_chunk=4, instruction_d=32, bytecode_d=4, ram_d=4)
bytecode_k / ram_k    : 8192 / 8192

[Goldilocks base → WHIR]   (this implementation; transparent, no trusted setup)
  representation      : dense base-field columns (RA index/cycle + Inc sign/lo/hi limbs)
  committed columns   : 46
  committed elements  : 3,014,656  (dense, 8 B each)
  committed volume    : 23.00 MiB
  commit time         : 178.29 ms

[BN254 → Dory]             (actual Jolt protocol)
  representation      : sparse one-hot RA polys + dense Inc polys
  one-hot chunks      : 40 (layout 2^20 each; sparse: 2,360,080 nonzeros total)
  dense Inc elements  : 131,072
  logical field elems : 42,074,112  (32 B each; one-hot is committed sparsely)
  logical volume      : 1284.00 MiB dense-equivalent
  SRS setup (one-time): 651.14 ms  (num_vars=20)
  commit time         : 2646.80 ms

[Comparison — same fibonacci trace]
  field-element width : 32 B (BN254 Fr) → 8 B (Goldilocks) = 4× narrower
  commit wall-clock   : Dory 2646.80 ms vs WHIR 178.29 ms = 14.85× faster with WHIR
  trusted setup       : Dory needs a 2^20 SRS (651.14 ms); WHIR is transparent (none)
```

### How to read the comparison (honest framing)

The two schemes commit **fundamentally different objects**, so the headline numbers are
chosen to be directly comparable and *not* misleading:

- **Goldilocks/WHIR** commits a *dense small-field index column* per RA chunk
  (`trace_len` integers in `[0, k_chunk)`, 8 B each) plus the `Inc` limbs.
- **BN254/Dory** commits a *sparse one-hot polynomial* per RA chunk over a `k_chunk`×
  larger domain (`2^(log_t + log_k_chunk)` Booleans, committed via Dory's sparse MSM)
  plus dense `Inc` polys, with the BN254 scalar at 32 B.

Therefore the report leads with the two **exact, directly comparable** facts —
**field-element width (4×)** and **measured commit wall-clock (14.85×)** — plus the fact
that WHIR is **transparent** (no trusted setup) while Dory needs a `2²⁰` SRS. The BN254
"logical volume" (1284 MiB) is shown only as clearly-labeled context
("dense-equivalent; one-hot committed sparsely") and is **not** turned into a raw byte
ratio, because that would conflate sparse-vs-dense storage.

> The Dory run uses `warmup = 0, runs = 1` (this is an e2e validation/report, not a
> rigorous multi-run benchmark), and its SRS-setup time is reported separately.

---

## 7. How to run / verification status

All commands assume `source .bolt-dev-env` first (MLIR/LLVM paths for guest builds).

```bash
# Field correctness (num-bigint oracle for Goldilocks + Fp3 + limbs):
cargo nextest run -p jolt-field --features goldilocks            # 163 passed

# WHIR commit path + arkworks cross-check:
cargo nextest run -p jolt-whir  --features goldilocks            # 6 passed (2 commit + 4 crosscheck)

# Live fibonacci e2e + BN254/Dory comparison (compiles a RISC-V guest):
cargo nextest run -p jolt-pcs-bench --features goldilocks \
    fibonacci_goldilocks_e2e --run-ignored all --no-capture      # passes, prints the report above

# Lint + format (all goldilocks crates, both must be clean):
cargo clippy -p jolt-field -p jolt-witness -p jolt-whir -p jolt-pcs-bench \
    --features goldilocks --all-targets -q -- -D warnings        # clean
cargo fmt   -p jolt-field -p jolt-witness -p jolt-whir -p jolt-pcs-bench

# Regression — the existing BN254/Dory/BlindFold path is untouched:
cargo nextest run -p jolt-core muldiv --features host            # muldiv_e2e_dory passed
cargo nextest run -p jolt-core muldiv --features host,zk         # muldiv_e2e_dory passed
```

**Current status: all green.** `jolt-field` (163), `jolt-whir` (6), the live e2e, clippy
`-D warnings`, fmt, the default (non-goldilocks) `jolt-pcs-bench` build, and the
`muldiv` regression in both modes all pass.

---

## 8. Known Phase-2 concerns surfaced by Phase 1

- **All-zero committed columns.** Real witnesses contain all-zero columns (e.g. the high
  instruction `ra_dense` chunks `fib` never reaches, or an `Inc.hi` limb that stays
  zero). Committing them is fine, but WHIR's **open** path divides by the polynomial's
  evaluation (`the_sum / poly_eval`), which is `0/0` for the zero polynomial. The
  Phase-1 `sanity_roundtrip` therefore opens a dynamically chosen non-zero column; the
  general opening of arbitrary witness columns is a Phase-2 concern.
- **No limb constraints yet.** The `sign`/`lo`/`hi` columns are produced but not
  constrained to recompose or to be in range; that is the Phase-2 range-check /
  `2⁻³²`-carry sumcheck work. Phase 1 verifies recomposition only as a test assertion.
- **`mul_pow_2` is a perf opportunity, not a blocker.** The `Field` trait default is
  already correct for Goldilocks; a `reduce128`-based specialization is optional Phase-2
  perf (guarded by the oracle test).

---

## 9. Files to commit

### New files (Phase 1)

| File | Lines | Purpose |
|---|---:|---|
| `crates/jolt-field/src/goldilocks/mod.rs` | 19 | module wiring + re-exports |
| `crates/jolt-field/src/goldilocks/base.rs` | 389 | `Goldilocks` base field (Montgomery-free) |
| `crates/jolt-field/src/goldilocks/ext3.rs` | 302 | `GoldilocksFp3` cubic extension (`x³=2`) |
| `crates/jolt-field/src/goldilocks/decompose.rs` | 68 | value ↔ base-field limbs |
| `crates/jolt-field/src/goldilocks/tests.rs` | 237 | num-bigint correctness oracle |
| `crates/jolt-witness/src/goldilocks.rs` | 142 | base-field-limb committed columns |
| `crates/jolt-whir/Cargo.toml` | — | new workspace member manifest |
| `crates/jolt-whir/src/lib.rs` | 20 | crate root (feature-gated) |
| `crates/jolt-whir/src/convert.rs` | 32 | the WHIR seam (Goldilocks→Field64) |
| `crates/jolt-whir/src/params.rs` | 20 | WHIR protocol parameters |
| `crates/jolt-whir/src/commit.rs` | 72 | `commit_witness` over `Basefield<Field64_3>` |
| `crates/jolt-whir/src/sanity.rs` | 105 | single-point open/verify round-trip |
| `crates/jolt-whir/tests/commit.rs` | 122 | synthetic-trace commit path test |
| `crates/jolt-whir/tests/crosscheck.rs` | 164 | arkworks `Field64`/`Field64_3` oracle |
| `crates/jolt-pcs-bench/src/fib_goldilocks.rs` | 371 | live fibonacci e2e + BN254/Dory comparison |
| `PHASE1_GOLDILOCKS_STATUS.md` | — | this document |

### Modified files (Phase-1 changes only)

| File | Phase-1 change |
|---|---|
| `crates/jolt-field/Cargo.toml` | add `goldilocks = []` feature; `num-bigint` dev-dep |
| `crates/jolt-field/src/lib.rs` | gate `pub mod goldilocks` + re-export `Goldilocks`, `GoldilocksFp3` |
| `crates/jolt-witness/Cargo.toml` | add `goldilocks = ["jolt-field/goldilocks"]` feature |
| `crates/jolt-witness/src/lib.rs` | gate `pub mod goldilocks` |
| `crates/jolt-pcs-bench/Cargo.toml` | optional `jolt-whir` dep + `goldilocks` feature |
| `crates/jolt-pcs-bench/src/main.rs` | gate `mod fib_goldilocks` |
| `Cargo.toml` (root) | add `"crates/jolt-whir"` workspace member (see note) |
| `Cargo.lock` | resolve new deps (`jolt-whir`, `num-bigint`, `whir`) |

> **Root `Cargo.toml` note.** The only Phase-1-introduced change is the
> `"crates/jolt-whir"` member line. The rest of the working-tree diff in root
> `Cargo.toml` (the `crates/whir-pcs-bench` member, the local-path arkworks `[patch]`
> against `../algebra`, the blake3/digest pinning, the `[profile.samply]` block) is
> **pre-existing uncommitted plumbing** that Phase 1 *builds on* but did not introduce.
> Phase 1 needs that plumbing to compile (see below).

### Pre-existing plumbing Phase 1 depends on (must be present on the branch)

These are **not** Phase-1 deliverables, but the `goldilocks` build requires them:

- `crates/whir-pcs-bench/` — the existing WHIR bench crate (the user's uncommitted work).
- The sibling `../whir` and `../algebra` (local-path arkworks fork) checkouts that the
  root `Cargo.toml` `[patch]` / path deps point at, plus the blake3/`digest 0.11` pinning
  that lets `whir` coexist with the Jolt dependency graph.

### NOT part of Phase 1 (unrelated working-tree changes — do not commit with Phase 1)

- `crates/jolt-equivalence/*` — separate Bolt parity-oracle / checkpoint work.
- `jolt-core/Cargo.toml`, `jolt-core/src/field/mod.rs`, `jolt-core/src/field/small_fields/`
  — earlier small-field scaffolding in the **legacy** `jolt-core` crate; **superseded** by
  the `crates/jolt-field/goldilocks` implementation (kept only as a Phase-2 reference).
- Assorted top-level `*.md` / `*.pdf` design notes and references.
