# Phase 6 — Instruction lookups (prefix/suffix) for the Goldilocks/WHIR prover

> **⚠️ STALE PATHS — read this first (2026-06-03).** This prompt's file references to
> `jolt-core/src/zkvm/lookup_table/prefixes/*.rs`, `poly/prefix_suffix.rs`, and the IL-2 "dynamic
> `prefix_mle` / ~40 prefix bodies" port are **obsolete on branch `refactor/crates`**: jolt-core was
> split into per-stage crates and the entire field-generic instruction read-raf now lives in
> **`jolt-kernels/src/stage5.rs`** (the read-only port oracle). There is **no `prefix_mle`** — the prover
> uses static `Prefixes::evaluate` + `LookupTableKind::combine` (already in `jolt-lookup-tables`). The
> transcript seam (jolt-kernels' `jolt_transcript::Transcript` bound vs WHIR's spongefish) means we
> **port the math into a goldilocks `framework::SumcheckInstance`**, not call jolt-kernels. The corrected
> staged plan (F1/P1/P2/P3) is in `GOLDILOCKS_WHIR_IMPLEMENTATION.md §12` and the
> `goldilocks-instruction-lookups-plan` memory. The staging/operating-constraints below still apply.

## Role & mission

You are a senior ZK systems engineer continuing the **Jolt zkVM Goldilocks + WHIR (non-ZK) prover
migration** in `crates/jolt-prover-goldilocks`. `jolt-core/` is the **read-only parity oracle**
(BN254/Dory) — port math *from* it, gate *against* it, **never modify it**.

**The full bytecode-first e2e is DONE** (Spartan + memory + booleanity + bytecode read-raf + M7
pushforward + stage-8 WHIR opens, verifying on a real muldiv trace, gated vs jolt-core geometry).
**Mission this phase: wire instruction lookups** — the prefix/suffix decomposition that lets the
instruction read-raf run at production `LOG_K=128` (`instruction_d=32` chunks × `log_k_chunk=4`,
`K_total=2¹²⁸`) without a dense `Val` table. This is the single largest remaining functional piece;
once it lands the prover proves instruction-execution semantics (the lookup argument).

**First skim the as-built reference, then implement the IL arc as staged green sub-commits.** Surface
any genuine fork; commit each tested piece; checkpoint cleanly at a green boundary if you run low on
context (never leave a half-ported prefix dispatch — IL-2 is all-or-nothing to compile).

## Load first (context)

- **`crates/jolt-prover-goldilocks/GOLDILOCKS_WHIR_IMPLEMENTATION.md`** — the authoritative as-built
  reference (architecture, every stage, all resolved design choices, the interim soundness gaps).
- **Memory `goldilocks-instruction-lookups-plan`** (auto-loaded) — the IL-1..IL-5 arc with exact
  reuse vs port boundaries, the prover-prefix-API gap, the binding-order gotcha. Also
  `goldilocks-migration-plan`, `goldilocks-real-program-not-mock`, `m7-readraf-shared-point-gap`.
- The full instruction-lookup MAP is in this session's workflow output (the report behind the plan);
  the plan memory distills it.
- **Verify the baseline:** `source .bolt-dev-env 2>/dev/null; cargo nextest run -p jolt-prover-goldilocks
  --features goldilocks` → **131 passing** (incl. IL-1 `operand_poly`).

## Done — committed, do NOT redo

- The full e2e: `61f10492d`..`d985e3cce` (M0–M4) — see GOLDILOCKS_WHIR_IMPLEMENTATION.md §14.
- **IL-1 `6d660931d`** — `zkvm::instruction_lookups::operand_poly::{OperandPolynomial, OperandSide}`:
  the verifier operand-extraction MLE (`Right`=offset0=`uninterleave_bits().0`, `Left`=offset1=`.1`).
  `IdentityPolynomial` is reused from `jolt_poly`.

## The decisive reuse (do NOT port these — they're field-generic already)

- **`jolt-lookup-tables`** (already a dep; deps jolt-field/jolt-trace, NOT jolt-core; also used by
  jolt-verifier/kernels/equivalence → extending it is additive/safe): `LookupTableKind<64>` (40 tables:
  `all()`/`index()`/`materialize_entry`/`evaluate_mle::<F,F>`/`suffixes()`/`combine::<F>`),
  `Prefixes`(47)/`Suffixes`(43) + `PrefixEval<F>` + static `SparseDensePrefix::{default_checkpoint,
  evaluate(checkpoints,b,suffix_len)}` + `SparseDenseSuffix::suffix_mle`, `LookupBits`,
  `interleave_bits`/`uninterleave_bits`, and the trace bridge `InstructionLookupTable<64>::lookup_table()`
  + `LookupQuery<64>::{to_lookup_index,to_lookup_operands,to_lookup_output}` on `jolt_trace::instructions::*`.
- **`jolt-poly`**: `ExpandingTable<F>`, `IdentityPolynomial`.

## Staged plan (each a green sub-commit; jolt-core read-only; C=F, no F::Challenge, no ZK)

- **IL-2 (next):** add the PROVER dynamic prefix API to `jolt-lookup-tables`' `SparseDensePrefix` trait +
  `Prefixes` enum dispatch: `prefix_mle(checkpoints, r_x: Option<F>, c: u32, b: LookupBits, j) -> F` and
  `update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)`, porting the ~40 per-prefix bodies from
  `jolt-core/src/zkvm/lookup_table/prefixes/*.rs`. Reconcile `PrefixCheckpoint = PrefixEval<Option<F>>` vs
  `PrefixEval<F>`. The static `evaluate` stays. **All-or-nothing compile** (dispatch matches all variants).
  Unit-test: `prefix_mle` at an even-round/`r_x=None`/phase-aligned point == static `evaluate`. THE one
  real math port.
- **IL-3:** port `jolt-core/src/poly/prefix_suffix.rs` `PrefixSuffixDecomposition<F,ORDER>` + `PrefixRegistry`
  + `CachedPolynomial` + `init_Q_raf` (the RAF Q-aggregator, ORDER=2 over Left/Right/Identity) over
  `jolt_field::Field`; add the IL-1 polys' `PrefixSuffixPolynomial<F,2>`/`SuffixPolynomial`/`prefix_polynomial`
  impls + `Shift{Half}SuffixPolynomial`. Drop allocative/ZK/unreduced-limb-accum (→ plain F). Use the
  framework `MultilinearPolynomial` for the dense P/Q. Unit-test the `(g0,g2)` sumcheck_evals + bind.
- **IL-4:** port the driver `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`
  (`InstructionReadRafSumcheckProver/Verifier`) onto `framework::sumcheck::SumcheckInstance`: address-phase
  prefix/suffix loop (`init_suffix_polys`; `prover_msg_read_checking` via `LookupTableKind::combine(prefixes_c,
  suffixes)`; `init_log_t_rounds` `combined_val_poly`; `ingest_challenge` 2-round `update_checkpoints`)
  **replacing the dense `ReadRafStage::val_addr`** (`shout_read_raf.rs:272/313`); verifier
  `expected_output_claim` via `table.evaluate_mle(r_addr)` + `OperandPolynomial` + per-table flag claims.
  Plumbing: add `VirtualPolynomial::LookupTableFlag(usize)` (`framework/accumulator.rs`); instruction
  phase/`log_m` config (phases×log_m = 128, log_m = 4 ⇒ 32 chunks); fix `INSTRUCTION_D` 5→32;
  `cycle → Option<LookupTableKind<64>>` bridge (CycleRow lacks `lookup_table()`; route via the `Instruction`
  like `bytecode_val_polys` does, using `jolt-lookup-tables InstructionLookupTable`). **GOTCHA — binding
  order:** prefix/suffix is HighToLow (prefix vars MSB-first); the goldilocks sparse `OneHotReadRaf` address
  phase binds LowToHigh — reconcile (bind prefix/suffix HighToLow within the address phase, OR a sibling
  `InstructionReadRaf` instance). The framework `sumcheck_evals_array` is points `0,1,2,..` (not jolt-core's
  `0,2,3`), but the decomposition binds dense P/Q so the operand polys' bespoke sumcheck_evals aren't needed.
- **IL-5:** `prove_instruction_read_raf`/`verify` (mirror `prove_bytecode_read_raf`: 3 stages rv/left-op/
  right-op sharing `r_cycle=r_reduction`; seed rv/left/right from `InstructionClaimReduction`/
  `SpartanProductVirtualization` per `instruction_config()` shout_read_raf.rs); wire into `prove_e2e`/
  `verify_e2e` (instruction family) + the EXISTING M7 `prove_read_raf_pushforward` over `InstructionRa`
  (unchanged — consumes the cached `InstructionRa(i)@InstructionReadRaf` openings); extend the M4 parity gate.

## Operating constraints

- `F = GoldilocksFp3`; non-ZK; `#[expect(...)]` not `#[allow(...)]` (`allow_attributes="deny"`,
  `clippy::panic="deny"`, `unused_results`, `too_many_arguments` cap 7); `.unwrap()/.expect()` only in
  `#[cfg(test)]`. Always `cargo nextest` (never `cargo test`) for the goldilocks crate. Green on
  `cargo nextest run -p jolt-prover-goldilocks --features goldilocks` + `cargo clippy … --features
  goldilocks --all-targets -- -D warnings` + `cargo fmt -p jolt-prover-goldilocks` before each commit.
- `jolt-equivalence` goldilocks e2e: `cargo test -p jolt-equivalence --features goldilocks --test
  goldilocks_e2e` (NOT nextest — it builds ALL goldilocks test binaries → fills the disk, errno 28). Free
  `target/*/incremental` if low on disk (`rm -rf target/debug/incremental`); the machine's disk runs near-full.
- `jolt-field` `F::zero()/one()` don't resolve on concrete `GoldilocksFp3` in tests — use `from_u64`.
- Commit each tested piece locally; NO co-author trailer; do NOT push; do NOT commit `Cargo.lock`.

## Suggested first moves

1. Skim GOLDILOCKS_WHIR_IMPLEMENTATION.md + the `goldilocks-instruction-lookups-plan` memory; run the baseline (131 green).
2. Start IL-2: read `jolt-core/src/zkvm/lookup_table/prefixes/mod.rs` + 3-4 per-prefix bodies (e.g. and.rs,
   lt.rs) + `jolt-lookup-tables/src/tables/prefixes/mod.rs`; add the two trait methods + dispatch + port all
   ~40 bodies; unit-test against the static `evaluate`. Commit when green.
3. Proceed IL-3 → IL-4 → IL-5, committing each; checkpoint at a green boundary if context runs low.
