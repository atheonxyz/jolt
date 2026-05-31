# Phase 2 — Goldilocks + WHIR Prover: Progress & Handoff

**Branch:** `refactor/crates` (local integration; diverged `origin/main` +65 / local +282).
**Crate:** `crates/jolt-prover-goldilocks` (feature-gated `goldilocks`, `#![cfg(feature = "goldilocks")]`).
**Status:** Field/PCS/transcript wiring + limbed R1CS + prover framework foundation + first
subprotocol port all **done, committed, tested**. Remaining: the checking-subprotocol + Spartan
ports, range checks, integrated LogUp\*-GKR, and the stage driver / e2e.

This doc is the single source of truth for resuming. Read it top-to-bottom, then the cited code.

---

## 0. Goal (the whole migration)

Migrate the Jolt zkVM (RISC-V RV64IMAC) from **BN254 + Dory PCS** to:
- **Base field:** Goldilocks `p = 2⁶⁴ − 2³² + 1` (committed witness lives here, 8-byte commits).
- **Prover scalar / challenges:** `GoldilocksFp3 = Goldilocks[X]/(X³−2)` (~192-bit; matches WHIR's `Field64_3`).
- **PCS:** WHIR (hash-based, transparent), via `WhirScheme`.
- **Transcript:** one shared **spongefish** sponge (Jolt + WHIR speak the same sponge).
- **Non-ZK for now** (whir-zk / ZK-sumchecks / BlindFold are Phase 3).

End-state acceptance (M8): a Goldilocks+WHIR prover proves `muldiv` and `fibonacci`, its verifier
accepts, and `jolt-equivalence` cross-checks claim-level values against the **jolt-core BN254
oracle** (transcripts can't be byte-compared across fields).

**Why Goldilocks+limbs:** the design (`JOLT_GOLDILOCKS_DESIGN.md`) shows limbs win on commit volume,
proof size, AND the sumcheck inner loop (`base × ext` = 3 muls vs `ext × ext` = 9 muls, ~2.3×).

---

## 1. Repository architecture you MUST understand first

- **`jolt-core/`** (repo root, NOT under `crates/`) = **legacy** hand-written prover/verifier over its
  own `JoltField` trait (BN254-specialized; the user confirmed "JoltField is not entirely generic").
  It is the **parity / equivalence oracle ONLY**. Do **not** modify it. Do **not** call into its
  prover internals from the new crate.
- **`crates/*`** = the **new modular Jolt** = shared libraries + a typed verifier. The new-Jolt
  **prover/verifier are Bolt-GENERATED** ("generated roles" — `crates/jolt-prover` is declarative
  `StageN*Plan` data executed by `crates/jolt-kernels`, hard-specialized to BN254/Dory/Blake2b).
- **Critical discovery (verified across `main`, `refactor/audit-prep`, PRs #1455/#1521/#1512–#1515/#1523):**
  there is **NO reusable hand-written prover-side sumcheck framework** in `crates/*`. `jolt-sumcheck`
  is **verifier-side only** (`SumcheckClaim`/`SumcheckProof`/`SumcheckVerifier`/`RoundProof`).
  `jolt-poly` has a primitive dense `Polynomial<F>` (no `MultilinearPolynomial` enum, no
  `bind_parallel`/`sumcheck_evals_array`). `jolt-openings`/`jolt-claims` are stateless claim types
  (no `ProverOpeningAccumulator`/`SumcheckId`/`CommittedPolynomial` state machine). The prover-side
  framework exists **only in `jolt-core`** (hand-written) or `jolt-kernels` (generated).
- **The canonical design for a hand-written modular prover is `specs/jolt-prover-model-crate.md`**
  (on `refactor/audit-prep`, stack row 10, draft): a field-generic
  `prove<F: jolt_field::Field, PCS: CommitmentScheme<Field=F>, T: Transcript<Challenge=F>, …>` on the
  modular crates, with jolt-core as parity oracle only. **`jolt-prover-goldilocks` IS that crate**,
  instantiated at `F = GoldilocksFp3`, `PCS = WhirScheme`.

### THE ARCHITECTURE DECISION (user-confirmed: **Option 1**)

> Build the prover by **vendoring the prover-side sumcheck framework from jolt-core** (the
> `SumcheckInstance*` traits, the opening accumulator, the `MultilinearPolynomial` enum) **into
> `jolt-prover-goldilocks`**, retargeted to the lean `jolt_field::Field`, **reusing the workspace
> primitives** (`jolt-poly`, `jolt-sumcheck` verifier, `jolt-openings`, `jolt-r1cs`,
> `jolt-transcript`, `jolt-lookup-tables`, `jolt-whir`) where their APIs suffice. jolt-core is the
> math/parity oracle. The framework lives **inside the crate** (`src/framework/`), extractable to a
> shared crate later (no second consumer today since the BN254 prover is Bolt-generated).

Options 2 (re-architect onto the lean crates) and 3 (impl jolt-core's `JoltField` for Goldilocks)
were rejected as too much work / too coupled.

### Retarget rules (apply to every jolt-core port)

- `crate::field::JoltField` → `jolt_field::Field` (the lean trait; **no `Challenge` associated type**).
- **`F::Challenge` → `F`** everywhere (the `C = F = Fp3` convention; `associated_type_defaults` is
  unstable on the pinned stable 1.94). The "challenge × field" ops already exist as
  `jolt_lookup_tables::{ChallengeOps<F>, FieldOps<C>}` with `C = F` — use those only where the
  lookup-table API already carries a `C` generic.
- Unreduced-accumulator ladder (`WideAccumulator`, `MedAccumS`, `BarrettReduce`, `FMAdd`) → the M0
  accumulators `jolt_field::goldilocks::{GoldilocksAccumulator, GoldilocksScalarAccumulator,
  Fp3Accumulator, Fp3ScalarAccumulator}` (+ `Fp3Accumulator::fmadd_base` for the `base × ext` hot path).
- Drop all `#[cfg(feature = "zk")]` blocks (BlindFold) — non-ZK this phase.
- Pin the jolt-core source path in each module's doc-comment; jolt-core stays the parity oracle.

---

## 2. What is DONE (committed, tested)

All commits on `refactor/crates`, in order. **No co-author trailer, not pushed** (user handles git).

| Commit | Milestone | Summary |
|---|---|---|
| `6d263f5d1` | M0 | `jolt-field` Goldilocks/Fp3 deferred-reduction accumulators + `From<u128>` + Fp3 `mul_pow_2` test |
| `08794dc74` | M1 | `jolt-whir` shared spongefish transcript (one sponge for Jolt + WHIR), `challenge_fp3`, `from_field64_3` |
| `60bf2696c` | M2 | `WhirScheme` commit/open/verify over the shared transcript (inherent API) |
| `cc5af5c32` | M3 | `WhirScheme` batch open via WHIR geometric RLC, cross-size-class `Config` |
| `d9c7e8a99` | M4 | `jolt-field` base-field limb primitives (`decompose.rs`) + signed 2-limb `Inc` witness (`jolt-witness`) |
| `ce2440668` | M5 | `jolt-prover-goldilocks` crate skeleton + field/PCS/transcript namespace wiring |
| `7e97db47b` | M5 | Pinned limbed RV64 R1CS design doc (`LIMBED_R1CS.md`) |
| `5c571d81c` | M5 | Limbed MUL 4-limb schoolbook R1CS rows (`r1cs/mul.rs`) |
| `90d5926a0` | M5 | Signed-value derivation for dual-use MUL operands (`r1cs/signed_value.rs`) |
| `a46e62f8b` | M5 | **Full limbed RV64 R1CS matrices, limb-wise** (`r1cs/rv64_limbed.rs`) |
| `5ed82eae9` | M5 | **Framework: dense `MultilinearPolynomial`** (`framework/poly.rs`) |
| `8bf3c0143` | M5 | **Framework: `SumcheckInstance` trait + batched driver** (`framework/sumcheck.rs`) |
| `54b0540fd` | M5 | **Framework: opening accumulator** (`framework/accumulator.rs`) + threaded through the trait |
| `f320f6c41` | M5 | **Port: Inc claim-reduction sumcheck** (`zkvm/claim_reductions/increments.rs`) |

**Current `jolt-prover-goldilocks/src/` tree:**
```
lib.rs                              #![cfg(feature="goldilocks")]; pub mod field/framework/r1cs/zkvm
field.rs                            type F = GoldilocksFp3; type Base = Goldilocks; WhirScheme re-exports
framework/mod.rs
framework/poly.rs                   MultilinearPolynomial<F> (Dense only): bind_parallel, sumcheck_evals_array::<D>, final_sumcheck_claim
framework/sumcheck.rs               SumcheckInstance<F> trait + prove()/verify() driver (bridges to jolt_sumcheck::SumcheckVerifier)
framework/accumulator.rs            OpeningPoint<E,F>, SumcheckId, CommittedPolynomial, VirtualPolynomial, OpeningAccumulator trait, Openings<F> store
r1cs/mod.rs
r1cs/mul.rs                         4-limb MUL schoolbook (push_mul_constraints, MulVars, NUM_MUL_ROWS=10)
r1cs/signed_value.rs                degree-2 signed-value derivation (RIGHT_VAL) — reserved for negative-Right linear use
r1cs/rv64_limbed.rs                 full 22-constraint limbed RV64 matrices (layout(), rv64_limbed_constraints(), NUM_LIMBED_ROWS=53, 70 vars)
zkvm/mod.rs
zkvm/claim_reductions/mod.rs
zkvm/claim_reductions/increments.rs IncClaimReduction (first ported subprotocol)
LIMBED_R1CS.md                      pinned limbed-R1CS design (read this for the R1CS)
PHASE2_HANDOFF.md                   this file
```
**21 crate tests pass; `cargo clippy --features goldilocks --all-targets -D warnings` clean; `cargo fmt` clean.**

### 2.1 The prover framework (the foundation everything sits on)

**`framework/poly.rs` — `MultilinearPolynomial<F>`** (dense only, vendored from jolt-core
`poly/multilinear_polynomial.rs`):
- `bind_parallel(r: F, order: BindingOrder)` = `lo + r·(hi − lo)` (rayon).
- `sumcheck_evals_array::<DEGREE>(index, order) -> [F; DEGREE]` = linear extrapolation through the
  bound pair: `evals[k] = e0 + k·(e1 − e0)`.
- `final_sumcheck_claim()`, `len`, `num_vars`. Reuses `jolt_poly::BindingOrder`.
- **Convention:** challenges are plain `F`. The enum shape is kept so the **compact base-field
  variants** (`base × ext` hot path via `Fp3Accumulator::fmadd_base`) and **OneHot/RLC** variants slot
  in later without touching call sites.

**`framework/sumcheck.rs` — `SumcheckInstance<F>` trait + driver:**
```rust
pub trait SumcheckInstance<F: Field> {
    fn num_rounds(&self) -> usize;
    fn degree(&self) -> usize;
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F;
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UnivariatePoly<F>; // degree ≤ self.degree(), s(0)+s(1)=previous_claim
    fn bind(&mut self, r: F, round: usize);
    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]);
    fn expected_output_claim(&self, accumulator: &dyn OpeningAccumulator<F>, challenges: &[F]) -> F;
    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> { /* LITTLE_ENDIAN(challenges).match_endianness() */ }
}
pub fn prove<F, I, T>(instance: &mut I, accumulator: &mut Openings<F>, transcript: &mut T) -> (SumcheckProof<F>, Vec<F>)
pub fn verify<F, T>(claim: &SumcheckClaim<F>, proof: &SumcheckProof<F>, transcript: &mut T) -> Result<EvaluationClaim<F>, SumcheckError<F>>
```
- The driver builds each round poly via `instance.compute_message` → `UnivariatePoly` → absorbs via
  `<UnivariatePoly<F> as RoundProof<F>>::append_to_transcript` (the **same path the verifier
  replays**) → `transcript.challenge()` → `bind`. Emits `jolt_sumcheck::SumcheckProof`
  (`Vec<UnivariatePoly<F>>`) that **`jolt_sumcheck::SumcheckVerifier::verify` accepts unchanged**.
  This is the single most important de-risking result: the hand-written prover and the extracted
  workspace verifier interoperate.
- Round polys are **uncompressed** `UnivariatePoly` (full coeffs); compression
  (`CompressedLabeledRoundPoly`) is a later proof-size optimization.
- **Bind order = `LowToHigh`** (LSB first), matching jolt-core. `normalize_opening_point` reverses
  the challenge order to the BIG_ENDIAN opening point used to key cached openings.

**`framework/accumulator.rs` — opening accumulator (claim store):**
- `OpeningPoint<const E: Endianness, F>` (Vec<F>; `split_at`, `match_endianness`, `Index`).
- `SumcheckId` (23 variants, vendored verbatim), `CommittedPolynomial` (RdInc/RamInc/InstructionRa(usize)/
  BytecodeRa(usize)/RamRa(usize)/TrustedAdvice/UntrustedAdvice), `VirtualPolynomial` (subset — the
  flag-carrying `OpFlags`/`InstructionFlags`/`LookupTableFlag` variants are **not yet added**; add
  them when the Spartan/bytecode ports need them, importing the RISC-V flag enums).
- `OpeningAccumulator<F>` trait (`get_committed_polynomial_opening`, `get_virtual_polynomial_opening`).
- `Openings<F>` = `HashMap<(PolynomialId, SumcheckId), (OpeningPoint<BIG_ENDIAN,F>, F)>` used by
  **both** prover and verifier (prover fills claims it computed; verifier fills from the proof).
  `append_dense`/`append_virtual`. **Deferred:** dedup/aliases, ZK pending-claims, and the **stage-8
  batched PCS opening** (`DoryOpeningState` analog) — those land with the stage driver + WHIR opening.

### 2.2 The limbed RV64 R1CS (soundness-critical; see `LIMBED_R1CS.md`)

The BN254 `crates/jolt-r1cs/src/constraints/rv64.rs` (22 constraints / 38 vars) can't be reused: over
Goldilocks **every u64 R1CS value aliases mod p**, and the field recompose `lo + 2³²·hi` of a
multi-limb value equals the value **mod p** (because `2⁶⁴ ≡ 2³²−1`, `2⁹⁶ ≡ −1`). **Key soundness
finding (corrects the original "linear recompose" plan):** multi-limb **equality** and **arithmetic**
must be done **limb-by-limb with `2⁻³²` carries** (the lambda_vm pattern), never a single recompose.

`r1cs/rv64_limbed.rs` (`rv64_limbed_constraints<F>()`, 70 vars, 53 rows) implements all 22 constraints:
- **Per-limb equality** (constraints 1–6, 9–12, 14): `guard·(a_lo−b_lo)=0` AND `guard·(a_hi−b_hi)=0`.
- **Full-u64 lookup-operand add/sub** (7 ADD, 8 SUB): limb-wise with `{0,1}` carries. Grounded in
  `add.rs`/`sub.rs`/`mul.rs::to_lookup_operands`: `LeftLookupOperand = 0` always;
  `RightLookupOperand = Left + (Right as u64)` (ADD) / `Left + 2⁶⁴ − (Right as u64)` (SUB, encoded as
  `RLO + Right = Left + 2⁶⁴` to avoid a `2⁶⁴` field constant). For the 64-bit ADD/SUB/MUL arm
  `Right = rs2 ≥ 0`, so magnitude limbs suffice and carries stay Boolean.
- **`RamAddress = Rs1 + Imm`** (constraint 0): limb-wise, limb1 exact.
- **MUL** (constraint 19): the 4-limb schoolbook (`mul.rs`), `Left.sign` pinned to 0. Constraint 9
  (`RightLookupOperand = Product`) is per-limb equality.
- **Small-value single-element recompose** (13/15/16/17/18): PCs are `< 2³²`, results `< p`, so safe.
- **Boolean products** (20/21): `ShouldBranch = recompose(LookupOutput)·Branch`, `ShouldJump =
  Jump·(1−NextIsNoop)`.
- **Validated** by hand-built honest witnesses (no-op/ADD/SUB/MUL/load, edges + 2000 random) + tamper
  rejection. **Soundness additionally requires the M6 range checks** (every limb `< 2³²`, carries
  Boolean, signs Boolean) — without them a prover equivocates on a value's limbs.
- `signed_value.rs` (RIGHT_VAL derivation) is **built + validated but not yet wired** — reserved for
  the negative-`Right` linear-use case (signed immediates as the multiplicative right operand).

### 2.3 First subprotocol port — Inc claim-reduction (`zkvm/claim_reductions/increments.rs`)

Reduces the four `RamInc`/`RdInc` openings to a single shared opening point. Single-phase form
(jolt-core's prefix/suffix two-phase materialization is a perf opt deferred). Takes **pre-materialized
recomposed `Fp3` Inc columns** (decoupled from the trace → signed-limb materialization, which is M8).
Round-trip tested prover → `jolt_sumcheck::SumcheckVerifier`; the **endianness convention** is
established here: eq polys built via `EqPolynomial::evals(point.r)`, bound `LowToHigh`; verifier uses
`EqPolynomial::mle(normalize(challenges).r, point.r)`; cached `RamInc(ρ)` = direct MLE at
`reverse(challenges)`. **This is the template for every surviving claim/checking port.**

---

## 3. What is LEFT (the rest of Phase 2)

### Port order is RESHAPED by the LogUp\*-GKR design (M7) — read carefully

The integrated LogUp\*-GKR (M7, `JOLT_GOLDILOCKS_DESIGN.md` §3) replaces the one-hot RA machinery, so
several jolt-core subprotocols are **eliminated/changed**, NOT ported:
- **SKIP:** stage-6 RA booleanity (`subprotocols/booleanity.rs`), Hamming booleanity, Hamming-weight
  reduction (`claim_reductions/hamming_weight.rs`). The one-hot `ra` is never committed.
- **SKIP as standalone sumchecks:** the RA-family claim reductions (`ram_ra`, instruction-lookups RA,
  registers RA) — GKR territory.
- **KEEP (dense):** `claim_reductions/increments.rs` (done), and `advice` if needed (dense committed).

### M5 remaining — port the surviving checking subprotocols + Spartan (in this order)

Each: copy from jolt-core, retarget per §1, decouple from the trace (take materialized polys) so it's
standalone-testable, impl the framework `SumcheckInstance`, round-trip test prover→verifier. Sizes are
jolt-core line counts.

1. **`registers/val_evaluation.rs`** (357) — **degree-3** `Val = Σ inc·wa·LT`. The degree-sensitive
   case (M6); needs the `LT` (less-than) + `wa` (write-address) poly pieces added to the framework.
2. **`ram/output_check.rs`** (446, degree-2).
3. **`ram/val_check.rs`** (550) — RAM `Val`/`ValInit`/`ValFinal`.
4. **`registers/read_write_checking.rs`** (1064, degree-3, two-phase).
5. **`ram/read_write_checking.rs`** (large, degree-3) + **`ram/raf_evaluation.rs`**.
6. **`bytecode/read_raf_checking.rs`** (degree-3).
7. **`instruction_lookups/read_raf_checking.rs`** (Shout; the range checks fold in here, §4.2).
8. **Spartan:** `spartan/outer.rs` + `spartan/product.rs` (univariate-skip — uni-skip eval points
   must be Fp3), `spartan/shift.rs` (degree-3), `spartan/instruction_input.rs` (degree-2). Depends on
   the limbed R1CS (done) + `UniformSpartanKey` width wiring (field-agnostic; widen `num_vars`/`num_cons`
   for the 70-var layout).

### M6 — range checks + degree-sensitive ports

- (A) Boolean carry/sign bits → residual `x²−x` **booleanity** (retarget `subprotocols/booleanity.rs`
  from RA selectors to the limbed R1CS carry/sign columns; degree-3 shape unchanged).
- (B) Wide 32-bit limbs (MUL product, lookup outputs) → **reuse the existing Shout
  `RangeCheck`/`LowerHalfWord`/`UpperWord` tables inside stage-5 `instruction_read_raf`** (no new
  sumcheck instance, no round-count change; `jolt-lookup-tables/src/tables/`).
- Lock `val_evaluation` / `val_check` at degree-3 (the signed 2-limb `Inc` keeps `Val = Σ inc·wa·LT`
  degree-3, per M4).

### M7 — integrated LogUp\* pushforward-GKR (`crates/jolt-whir/src/logup/` + the prover)

- The eq-weighted pushforward `P^F[k] = Σ_{j: M*[j]=k} eq(bits(j), r)` per family (3 total via §4.1
  row-concatenation), §4.5.2 d-claim→1 reduction, fan-in-2 fractional-add GKR (Gruen, degree-3),
  expressed as framework `SumcheckInstance`s on the one shared transcript, consuming upstream
  `EvaluationClaim`s. Leaf claims (on `ra_dense`, `P^F`) feed the accumulator → `WhirScheme::open_batch`.
- **Removes** stage-6 RA booleanity, Hamming booleanity, stage-7 Hamming weight. One-hot RA never committed.
- Algorithm reference: `crates/whir-pcs-bench/src/gkr.rs` (the 3 prototype asserts → fallible checks:
  main identity, root histogram `N_A·D_B==N_B·D_A`, per-layer Gruen consistency).

### M8 — stage driver, e2e, equivalence

- Port `prover.rs`/`verifier.rs` stage driver (stages 1–8): all soundness challenges in Fp3;
  witness reads become limb-reads + recomposition; trace → signed-limb `Inc` materialization (the
  piece increments deferred); stage-8 = the accumulator's batched **`WhirScheme` opening** + GKR opening.
- Add the **stage-8 PCS batching layer** to the accumulator (dedup + RLC + WHIR open).
- `muldiv_e2e_goldilocks` + `fibonacci_e2e_goldilocks` (copy `muldiv_e2e_dory` from
  `jolt-core/src/zkvm/prover.rs`, swap prover/verifier alias, drop `DoryGlobals::reset()`).
- `jolt-equivalence` claim-level cross-check vs jolt-core (`core_oracle.rs`/`commitment_oracle.rs`/
  `core_conversion.rs`): same trace → recomposed limb values/openings equal jolt-core's reduced mod p.
- **Gate:** Goldilocks+WHIR prove→verify green on muldiv & fibonacci; **BN254 muldiv still green**
  (`cargo nextest run -p jolt-core muldiv --features host` and `--features host,zk`); clippy/fmt clean.

---

## 4. Key design choices made (+ rationale)

| Decision | Choice | Why |
|---|---|---|
| Prover framework source | **Vendor from jolt-core** (Option 1), in-crate (`src/framework/`) | No reusable hand-written prover in `crates/*`; jolt-core = oracle ⇒ faithful ports; lowest soundness risk |
| Challenge type | `C = F = GoldilocksFp3` (no `Field::Challenge` assoc type) | `associated_type_defaults` unstable on stable 1.94; `ChallengeOps<F>` already exists with C=F |
| Transcript | spongefish in `jolt-whir`, used **concretely** (not via a `jolt_transcript::Transcript` impl) | spongefish `ProverState`/`VerifierState` can't satisfy `Transcript: Default+Clone+'static`; WHIR needs its own state. New prover is monomorphized; framework tests use `Blake2bTranscript<F>` as a stand-in |
| `WhirScheme` | **inherent** API, not the `CommitmentScheme` trait | the trait pins `Transcript<Challenge=Self::Field>`; we thread the concrete spongefish state |
| `Inc` representation | **signed 2-limb** (`lo + hi·2³²`, hi signed) | linear recompose keeps `Val = Σ inc·wa·LT` **degree-3** (sign+limbs would be degree-4) |
| Limbed R1CS | **limb-wise + `2⁻³²` carries**, NOT field recompose | field recompose aliases mod p ⇒ unsound equivocation `a = b+p`; matches lambda_vm |
| Mixed limb repr | u64→2 unsigned limbs; signed-linear (Imm)→signed 2-limb; MUL operand/product→sign+magnitude limbs; small (PC<2³², flags)→single element | clean schoolbook on unsigned magnitudes; degree-2 outer sumcheck preserved |
| Round polys | uncompressed `UnivariatePoly` | correctness first; compression is a later proof-size opt |
| First-port style | **decoupled from trace** (take materialized polys) | standalone-testable; trace→limb materialization is M8 witness-gen |

---

## 5. Potential design choices for LATER (open / not yet decided)

- **Stage-8 PCS batching in the accumulator:** how to batch multiple committed-limb openings into one
  `WhirScheme::open_batch` — per-limb columns (lo, hi) opened separately + verifier recomposes, vs a
  combined recomposed virtual poly. Coupled to M3's per-size-class `Config` and the limb commitment
  layout. **Decide when porting M8.**
- **Compact base-field `MultilinearPolynomial` variants:** the `base × ext` hot path
  (`Fp3Accumulator::fmadd_base`, the 2.3× sumcheck win) needs `MultilinearPolynomial` to carry
  base-Goldilocks-coeff variants that promote to Fp3 on bind. Add when porting a hot subprotocol
  (RAM/registers read-write checking) — correctness-first dense works until then.
- **Two-phase (prefix/suffix) sumcheck materialization:** jolt-core's perf optimization for the
  claim-reductions/checking sumchecks. Port after the single-phase version is correct + equivalence-green.
- **`VirtualPolynomial` flag variants** (`OpFlags(CircuitFlags)`/`InstructionFlags`/`LookupTableFlag`):
  needed by the Spartan/bytecode ports; require the RISC-V flag enums (from `common`/`jolt-riscv`).
- **Whether `advice` claim-reduction is needed** for muldiv/fibonacci (probably not — advice e2e is a
  separate test); port only if the e2e trace exercises advice.
- **Negative-`Right` lookup operand** (signed immediate as multiplicative right operand): wire
  `signed_value.rs`'s RIGHT_VAL + a two's-complement limb derivation in `rv64_limbed.rs` constraint 10.
  Not exercised by muldiv; add when fibonacci/other traces need it.
- **Extracting `src/framework/` to a shared `jolt-sumcheck-prover` crate** if a hand-written BN254
  modular prover ever materializes (none today — BN254 prover is Bolt-generated).
- **Branch strategy:** `refactor/crates` is behind `origin/main` (which has richer `jolt-poly`,
  `jolt-claims`, the typed `jolt-verifier`, and the `jolt-prover` spec). Decide whether to rebase the
  Goldilocks work onto main's modular stack before/after Phase 2 — the framework is structured to make
  that migration mechanical.

---

## 6. Build / test / lint conventions (CRITICAL — from CLAUDE.md)

```bash
# external SSD: re-establish cwd if it drops (transient fault seen this session;
# uncommitted working-tree files can be LOST on a remount — commit promptly)
cd /Volumes/SenpaisSSD/zkVms/Jolt/jolt

# the goldilocks crate is feature-gated; ALWAYS pass --features goldilocks
cargo nextest run -p jolt-prover-goldilocks --features goldilocks --cargo-quiet
cargo clippy -p jolt-prover-goldilocks --features goldilocks --all-targets -q -- -D warnings
cargo fmt -p jolt-prover-goldilocks -q

# BN254 regression (must stay green; jolt-core untouched)
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```
- **Always `cargo nextest`, never `cargo test`.** The `algebra` `digest::generic_array` deprecation
  warnings are pre-existing noise — filter them, don't fix.
- Workspace lints: `allow_attributes = "deny"` (use `#[expect(...)]`, not `#[allow]`),
  `clippy::panic`/`expect_used`/`unwrap_used` denied in non-test code (annotate with `#[expect(...)]`
  + `reason`), `unused_results` denied (bind `HashMap::insert` etc. with `let _ =`). `#[expect]` on a
  test module must actually be *fulfilled* (an unfulfilled expectation is itself an error).
- Commit **per small piece, distinct commits, NO co-author trailer, do NOT push** (user does git).
  Leave Cargo.toml/Cargo.lock workspace-plumbing changes out of milestone commits where possible; the
  crate's `cargo-machete` `ignored` list is trimmed as deps become used.

---

## 7. Reference docs in the repo (read these)

- `crates/jolt-prover-goldilocks/LIMBED_R1CS.md` — the pinned limbed-R1CS design (limb-wise rules,
  `to_lookup_operands` grounding, MUL schoolbook, degree analysis).
- `JOLT_GOLDILOCKS_DESIGN.md` — the master design (field choice, limbs §2, LogUp\*-GKR §3, range
  checks §4, accumulators §5.3, transcript §6, WhirScheme adapter §7, stages §10).
- `JOLT_SMALLFIELD_WHIR_MIGRATION.md` — the migration plan (spongefish splice, WHIR integration).
- `PHASE1_GOLDILOCKS_STATUS.md` — Phase-1 (the front of the pipeline) status.
- `specs/jolt-prover-model-crate.md` (on `refactor/audit-prep`) — the canonical hand-written modular
  prover architecture this crate implements.
- `~/.claude/.../memory/{goldilocks-prover-framework-source,goldilocks-port-order,jolt-v2-structure-and-field-migration}.md`
  — the recorded architecture/port decisions.

**jolt-core parity-oracle source paths (the math to mirror):** `jolt-core/src/zkvm/` (spartan/, ram/,
registers/, instruction_lookups/, claim_reductions/, bytecode/, prover.rs), `subprotocols/sumcheck_prover.rs`
+ `sumcheck_verifier.rs`, `poly/{multilinear_polynomial,opening_proof,unipoly,eq_poly}.rs`,
`zkvm/witness.rs` (CommittedPolynomial/VirtualPolynomial), `zkvm/r1cs/inputs.rs`,
`crates/jolt-r1cs/src/constraints/rv64.rs` (the 22 BN254 constraints).
