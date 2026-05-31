# Goldilocks + WHIR Migration — Phase 1 & 2 Review Guide

**Audience:** you, reviewing the AI-generated code. **Goal of this doc:** explain *everything*
built so far across Phase 1 and Phase 2, *why* each thing is the way it is, what is deliberately
**deferred** (and why that's sound), and what is **left** — so you can review the code quickly and
with full context.

This is the companion to (and supersedes for review purposes) `PHASE2_HANDOFF.md`,
`PHASE1_GOLDILOCKS_STATUS.md`, `LIMBED_R1CS.md`, and `JOLT_GOLDILOCKS_DESIGN.md`. Read those for
the original design rationale; read *this* to navigate and review the code.

---

## 0. The 60-second mental model

We are migrating Jolt's commitment stack from **BN254 + Dory** to **Goldilocks base field + Fp3
extension challenges + WHIR (hash-based) PCS**, non-ZK.

- **Two fields.** Witness data lives in **Goldilocks** `Fp` (`p = 2⁶⁴−2³²+1`, 8-byte elements).
  Every Fiat-Shamir challenge, sumcheck round polynomial, and WHIR fold lives in **`Fp3 = Fp[X]/(X³−2)`**
  (~192-bit), for soundness. Convention in *all* prover code: **`C = F = GoldilocksFp3`** — there is
  no separate `Challenge` associated type (the lean `jolt_field::Field`); base-field witnesses enter
  via limbs / `from_u64`.
- **`jolt-core/` is the legacy oracle — never modified.** It's BN254-specialized. The new prover is a
  **separate crate, `crates/jolt-prover-goldilocks`**, that *vendors* (faithfully re-implements) the
  prover-side sumcheck framework from `jolt-core`, retargeted to the lean field, and reuses the
  workspace primitive crates (`jolt-poly`, `jolt-sumcheck` verifier, `jolt-transcript`, `jolt-whir`,
  `jolt-field`). jolt-core is the math/parity oracle only.
- **Phase 1** stood up the *front* of the pipeline (field, limbs, WHIR commit) in the existing crates,
  and proved it on a live fibonacci trace vs the real BN254/Dory path. **Complete & verified.**
- **Phase 2** builds the *prover proper*: the sumcheck framework + every sumcheck subprotocol, in the
  new crate. **In progress** — all leaf checking subprotocols + most of Spartan + booleanity are done;
  the LogUp\*-GKR (M7) and the stage driver / e2e (M8) remain.

### The single most important review concept: "decoupled / correctness-first"

Every ported sumcheck is **decoupled from the trace**: instead of taking a `&[Cycle]` execution trace
and the full witness-generation machinery, each instance takes **pre-materialized polynomial columns**
(`Vec<F>`) as input and proves the sumcheck identity over them. This is deliberate:

- It makes every subprotocol **standalone unit-testable** (prover → verifier round-trip + tamper test)
  *without* a working end-to-end prover.
- The trace → column materialization (witness-gen) is concentrated in **M8** (the stage driver), not
  scattered across the ports.
- jolt-core remains the parity oracle: each module's doc-comment pins the exact jolt-core source file
  it mirrors, so you can diff the math.

When you see "decoupled" or "deferred to M8" in a module, that's this convention. It is **not** a
correctness shortcut — the sumcheck *identity* is faithful; only the *source of the columns* and
certain *performance optimizations* are deferred.

### How to run / verify what exists

```bash
cd /Volumes/SenpaisSSD/zkVms/Jolt/jolt

# Phase 2 prover crate (this is the bulk of the new code): 45 tests, all green.
cargo nextest run -p jolt-prover-goldilocks --features goldilocks --cargo-quiet
cargo clippy   -p jolt-prover-goldilocks --features goldilocks --all-targets -q -- -D warnings
cargo fmt      -p jolt-prover-goldilocks -q

# Phase 1 (field / witness / WHIR commit), all green:
cargo nextest run -p jolt-field   --features goldilocks   # field correctness vs num-bigint oracle
cargo nextest run -p jolt-whir    --features goldilocks   # WHIR commit + arkworks cross-check
# (live fibonacci e2e is #[ignore]'d; see PHASE1_GOLDILOCKS_STATUS.md §7)

# Regression — legacy BN254/Dory path is untouched:
cargo nextest run -p jolt-core muldiv --features host
cargo nextest run -p jolt-core muldiv --features host,zk
```

---

## 1. Phase 1 — field, limbs, WHIR commit (COMPLETE, in the existing crates)

Phase 1 is **not** in `jolt-prover-goldilocks`; it lives in the shared crates, feature-gated
`goldilocks`. Full detail is in `PHASE1_GOLDILOCKS_STATUS.md`; the review summary:

| Component | File(s) | What & why |
|---|---|---|
| **Goldilocks base field** | `crates/jolt-field/src/goldilocks/base.rs` | `p = 2⁶⁴−2³²+1`, **Montgomery-free** (`reduce128` via `2⁶⁴≡2³²−1`, `2⁹⁶≡−1`). Non-canonical `[0,2⁶⁴)` rep, canonicalize only at boundaries. Why: Goldilocks has cheap structured reduction; arkworks' generic Montgomery is ~2× slower. |
| **Fp3 extension** | `crates/jolt-field/src/goldilocks/ext3.rs` | `Fp[X]/(X³−2)`, nonresidue 2 (matches WHIR's `Field64_3`). `mul` = 9 base muls; **`mul_by_base` = 3 base muls** (the sumcheck hot path). Why nonresidue 2: matches lambda_vm + WHIR so cross-checks are valid. |
| **Limb decomposition** | `crates/jolt-field/src/goldilocks/decompose.rs` | `u64 → 2×u32` limbs; **signed i65 `Inc` → (sign, lo, hi)**. Why: over Goldilocks a raw u64 aliases mod p (`[p,2⁶⁴)` band), so no u64 is ever a single field element. |
| **Field correctness oracle** | `crates/jolt-field/src/goldilocks/tests.rs` | Every op checked against an independent **num-bigint** reference over random + edge inputs. Why num-bigint not arkworks: avoids dragging `digest 0.10` into the workspace. |
| **Base-limb witness columns** | `crates/jolt-witness/src/goldilocks.rs` | Trace's dense index vectors → base-Goldilocks committed columns: `ra_dense` (one index column per family/chunk) + `Inc` as `(sign, lo, hi)`. **No one-hot lift / `P^F`** (that's M7). |
| **WHIR base-commit** | `crates/jolt-whir/src/{convert,params,commit,sanity}.rs` | Commit the base columns via `Config<Basefield<Field64_3>>` (commit alphabet = base `Field64`, 8 B; folds in Fp3). `convert.rs` is the single arkworks seam. `sanity.rs` does a single-point open/verify round-trip. |
| **Live e2e + comparison** | `crates/jolt-pcs-bench/src/fib_goldilocks.rs` | Real fibonacci trace → base columns → WHIR commit, compared on the *same trace* vs BN254/Dory. Measured: 4× narrower elements, ~14.85× faster commit, transparent (no SRS). |

**Phase 1 deferred to Phase 2 (by design):** all sumcheck/IOP changes, LogUp\*-GKR, the limb
recompose/range-check *constraints* (Phase 1 produces the limb columns; it does not constrain them),
batched opening proofs, hiding (`whir_zk`), and the deferred-reduction accumulators.

---

## 2. Phase 2 foundation (M0–M4): field accumulators, transcript, WhirScheme, limbed R1CS

These land in the shared crates (M0–M4) and the new crate's `field`/`r1cs` modules. They are the
substrate the prover sits on.

### M0 — deferred-reduction accumulators (`crates/jolt-field/src/goldilocks/accumulator.rs`)
`GoldilocksAccumulator`, `Fp3Accumulator` (+ `Fp3Accumulator::fmadd_base` for the base×ext hot path),
implementing the `FieldAccumulator` trait. **Why:** sumcheck inner loops do `acc += a*b` hundreds of
times per output; deferring the modular reduction across the whole loop and reducing once amortizes the
expensive `reduce128`. This is the analog of jolt-core's BN254 `WideAccumulator`. *(Used pervasively in
OPT-A — see §6.)*

### M1–M3 — `jolt-whir`: shared transcript + `WhirScheme`
- **M1:** one shared **spongefish** sponge (Jolt and WHIR speak the same sponge), `challenge_fp3`.
- **M2:** `WhirScheme` commit/open/verify over that transcript (inherent API, not the `CommitmentScheme`
  trait — the trait pins `Transcript<Challenge=Self::Field>`, which doesn't fit; we thread the concrete
  spongefish state).
- **M3:** `WhirScheme` batch-open via WHIR's native geometric RLC, cross-size-class config.

### M4 — limb primitives + signed 2-limb `Inc`
`decompose.rs` base-limb primitives + the signed 2-limb `Inc` witness in `jolt-witness`. **Why signed
2-limb (`lo + hi·2³²`, `hi` signed) and not sign+limbs:** it keeps `Val = Σ inc·wa·LT` **degree-3**
(a sign-bit factor would push it to degree-4).

### The limbed RV64 R1CS (`src/r1cs/`, ~1400 LOC) — soundness-critical, read `LIMBED_R1CS.md`

The BN254 `crates/jolt-r1cs/.../rv64.rs` (22 constraints) **cannot be reused** over Goldilocks: every
u64 R1CS value aliases mod p, and a field recompose `lo + 2³²·hi` equals the value **mod p**, not the
integer (`2⁶⁴≡2³²−1`, `2⁹⁶≡−1`). **The key soundness finding:** multi-limb equality and arithmetic must
be done **limb-by-limb with `2⁻³²` carries** (the lambda_vm pattern), never a single recompose — else a
prover equivocates with `a = b + p`.

| File | What | Why |
|---|---|---|
| `r1cs/rv64_limbed.rs` (951) | All 22 RV64 constraints, limb-wise, 70 vars / 53 rows. Per-limb equality, full-u64 add/sub with `{0,1}` carries, `RamAddress=Rs1+Imm`, MUL via the schoolbook, small-value single-element recompose, boolean products. | Sound limbed arithmetization; degree-2 outer sumcheck preserved. |
| `r1cs/mul.rs` (288) | 4-limb MUL schoolbook with `2⁻³²` virtual carries; `Left.sign` pinned 0. | MUL product is 128-bit; the single A·B=C row expands to limb-products + carry rows. Validated vs an i128 reference product. |
| `r1cs/signed_value.rs` (166) | Degree-2 signed-value derivation `RIGHT_VAL = (1−2·sign)·magnitude` for negative-`Right` linear use. | Built + validated but **not yet wired** — reserved for signed-immediate multiplicative operands (not exercised by muldiv). |

**Validation:** hand-built honest witnesses (no-op/ADD/SUB/MUL/load + 2000 random) + tamper rejection.
**Soundness additionally requires the M6 range checks** (every limb `< 2³²`, carries/signs boolean) —
the limbed R1CS lands *coupled to* the booleanity check (§5), not standalone.

---

## 3. The prover framework (`src/framework/`) — the engine every port runs on

This is **vendored from jolt-core** (`subprotocols/sumcheck_prover.rs`, `poly/multilinear_polynomial.rs`,
`poly/opening_proof.rs`, `zkvm/witness.rs`) and retargeted to the lean field. **Review this first** —
every subprotocol implements its trait and relies on its conventions.

### `framework/poly.rs` (241) — `MultilinearPolynomial<F>`
Dense-only multilinear poly: `bind_parallel(r, order)` = `lo + r·(hi−lo)`; `sumcheck_evals_array::<D>`
= linear extrapolation through the bound pair (`evals[k] = e0 + k·(e1−e0)`); `final_sumcheck_claim`.
**Why dense-only:** correctness-first; the compact base-field variants (the `base×ext` 2.3× win) and
`OneHot`/`RLC` variants are deferred (need real base-field witnesses = M8). The enum shape is kept so
they slot in later without touching call sites.

### `framework/sumcheck.rs` (257) — `SumcheckInstance<F>` trait + driver
The contract every port implements:
```rust
trait SumcheckInstance<F> {
    fn num_rounds(&self) -> usize;
    fn degree(&self) -> usize;
    fn input_claim(&self, acc) -> F;                       // Σ_x g(x), from prior openings
    fn compute_message(&mut self, round, prev_claim) -> UnivariatePoly<F>;  // s(0)+s(1)=prev
    fn bind(&mut self, r, round);
    fn cache_openings(&self, acc, challenges);             // store this instance's output openings
    fn expected_output_claim(&self, acc, challenges) -> F; // what the reduced claim must equal
    fn normalize_opening_point(&self, challenges) -> OpeningPoint<BIG_ENDIAN,F>;  // default: reverse
}
pub fn prove<F,I,T>(instance, acc, transcript) -> (SumcheckProof<F>, Vec<F>)
pub fn verify<F,T>(claim, proof, transcript) -> Result<EvaluationClaim<F>, …>
```
**The critical de-risking result:** the driver's `prove` emits a `jolt_sumcheck::SumcheckProof` that
the **workspace verifier `jolt_sumcheck::SumcheckVerifier::verify` accepts unchanged**. So the
hand-written prover and the extracted workspace verifier interoperate — this is what every round-trip
test exercises. The driver absorbs each round poly via the *same* `RoundProof::append_to_transcript`
path the verifier replays, then squeezes the challenge, then binds.

### `framework/accumulator.rs` (349) — the opening accumulator (claim store)
A `HashMap<(PolynomialId, SumcheckId) → (OpeningPoint<BIG_ENDIAN,F>, F)>` used by **both** prover (fills
claims it computed) and verifier (fills from the proof). `CommittedPolynomial` / `VirtualPolynomial` /
`SumcheckId` are vendored verbatim from jolt-core so openings are keyed identically. **This is how
subprotocols compose:** instance A's `cache_openings` writes a claim that instance B's `input_claim`
reads. Deferred: dedup/aliases, and the **stage-8 batched PCS opening** (lands with M8 + the WhirScheme
opening).

---

## 4. The ported sumchecks — the heart of the review

Every sumcheck below follows the **same template** (established by `increments.rs`): take materialized
columns, prove the identity, cache output openings, verify round-trip. **Conventions to check in each
(see §7):** bind `LowToHigh`; opening points big-endian via `normalize_opening_point`; round polys
uncompressed `UnivariatePoly::from_evals`; `expected_output_claim` recomputes eq/structural factors and
reads committed/virtual openings from the accumulator.

| Port (file) | jolt-core source | Deg | Identity proved | Cached openings |
|---|---|---|---|---|
| **Inc claim-reduction** `claim_reductions/increments.rs` | `zkvm/claim_reductions/increments.rs` | 2 | `Σ_j RamInc·eq + γ²·RdInc·eq` → reduce 4 Inc openings to one point ρ | `RamInc`,`RdInc` @IncClaimReduction |
| **registers val-eval** `registers/val_evaluation.rs` | `zkvm/registers/val_evaluation.rs` | 3 | `Val = Σ_j inc·wa·LT(j,r_cycle)` | `RdInc`,`RdWa` @RegistersValEvaluation |
| **RAM output-check** `ram/output_check.rs` | `zkvm/ram/output_check.rs` | 3 | `0 = Σ_k eq·io_mask·(Val_final−Val_io)` (zero-check) | `RamValFinal` @RamOutputCheck |
| **RAM val-check** `ram/val_check.rs` | `zkvm/ram/val_check.rs` | 3 | `Σ_j inc·wa·(LT+γ)`; input `(val_rw−init)+γ(val_final−init)` | `RamRa`,`RamInc` @RamValCheck |
| **registers RW** `registers/read_write_checking.rs` | `zkvm/registers/read_write_checking.rs` | 3 | `Σ_{k,j} eq·[ra_merged·Val + wa·(Val+inc)]`, `ra_merged=γ·ra1+γ²·ra2` | `RegistersVal`,`Rs1Ra`,`Rs2Ra`,`RdWa`,`RdInc` @RegistersReadWriteChecking |
| **RAM RW** `ram/read_write_checking.rs` | `zkvm/ram/read_write_checking.rs` | 3 | `Σ_{k,j} eq·ra·(Val+γ(inc+Val)) = rv+γ·wv` | `RamVal`,`RamRa`,`RamInc` @RamReadWriteChecking |
| **RAM RAF** `ram/raf_evaluation.rs` | `zkvm/ram/raf_evaluation.rs` | 2 | `Σ_k ra(k)·unmap(k) = raf` | `RamRa` @RamRafEvaluation |
| **read-raf (shared)** `shout_read_raf.rs` (+ thin `bytecode/`, `instruction_lookups/`) | `zkvm/{bytecode,instruction_lookups}/read_raf_checking.rs` | 3 | `Σ_{j,k} ra(k,j)·Σ_s γ^s·eq_s(j)·Val_s(k) = Σ_s γ^s·rv_s`, `ra=∏_i ra_i` (d=2) | `BytecodeRa(i)`/`InstructionRa(i)` @{Bytecode,Instruction}ReadRaf |
| **Spartan outer** `spartan/outer.rs` | `zkvm/spartan/outer.rs` | 3 | `0 = Σ_x eq(τ,x)·(Az·Bz − Cz)` (zero-check) | `SpartanAz`,`Bz`,`Cz` @SpartanOuter |
| **Spartan shift** `spartan/shift.rs` | `zkvm/spartan/shift.rs` | 2 | `Σ_j [eq+1(r_outer,j)·(s0+γs1+γ²s2+γ³s3) + γ⁴·eq+1(r_product,j)·(1−s4)] = batched Next*` | 5 shift columns @SpartanShift |
| **Spartan instruction-input** `spartan/instruction_input.rs` | `zkvm/spartan/instruction_input.rs` | 3 | `Σ_j eq·(RightInput + γ·LeftInput) = right+γ·left`, inputs = flag·value sums | 8 flag/value openings @InstructionInputVirtualization |
| **booleanity (M6)** `zkvm/booleanity.rs` | `subprotocols/booleanity.rs` | 3 | `0 = Σ_x eq(r,x)·Σ_i γ^{2i}·(b_i²−b_i)` (carry/sign columns boolean) | `R1csAux(i)` @Booleanity |

### Per-port notes worth knowing while reviewing

- **`increments.rs`** is the canonical template — read it first. It establishes the endianness contract:
  eq tables built via `EqPolynomial::evals(point)`, bound `LowToHigh`; verifier uses
  `EqPolynomial::mle(normalize(challenges), point)`; cached opening = MLE at `reverse(challenges)`.
- **The `LT` (less-than) factor** in `val_evaluation`/`val_check` reuses the workspace
  `jolt_poly::LtPolynomial::evaluations` to materialize the dense `LT(j, r_cycle)` table (jolt-core's
  `LT(r_cycle, j)` is the same function with `j` as the first/varying argument — confirmed against the
  verifier's hand-rolled loop). The split-LT (√N memory) optimization is deferred (HighToLow conflict).
- **`output_check`** is the *only* port using the full **Gruen split-eq + unreduced-accumulator**
  optimization right now (OPT-A, §6) — read it as the optimized-pattern reference. Its `compute_message`
  uses `GruenSplitEqPolynomial::fold_out_in` to compute the quadratic coefficients of
  `q = io_mask·(Val_final−Val_io)`, then `gruen_poly_deg_3`. The verifier recomputes `io_mask`/`Val_io`
  MLEs from the public columns (they're public program I/O).
- **registers/RAM RW** use a **single-phase, fully-broadcast** form: the address×cycle matrix `ra/wa/val`
  is full dense `K·T` (index `k·T+j`), and the cycle-only `eq`/`inc` are broadcast across the K address
  blocks so binding is uniform `LowToHigh`. jolt-core uses a sparse `ReadWriteMatrix` two-phase
  (cycle-major→address-major) materialization — that's the deferred OPT-C. registers RW reads `rs1_ra`
  and `rs2_ra` **directly** from separate materialized read columns (jolt-core derives `rs2_ra` and
  back-solves `rs1_ra` to avoid materializing both — a perf trick, deferred).
- **`shout_read_raf.rs`** is **shared** by bytecode and instruction-lookups read-raf: both prove the same
  batched one-hot read identity. It's parameterized by the committed RA family (`fn(usize)->CommittedPolynomial`)
  and `SumcheckId`. **`ra = ∏_i ra_i`** is the d-chunk one-hot product (fixed `d=2` → degree-3, the
  handoff's stated bytecode degree). The cached `ra_i` leaf claims are **exactly the §4.5.2 inputs the
  M7 LogUp\*-GKR pushforward consumes** — see §8 / the `m7-logupstar-readraf-relationship` memory.
- **Spartan outer** is the decoupled R1CS-satisfaction zero-check on materialized `Az/Bz/Cz`. The
  reduction of `Az(r)/Bz(r)/Cz(r)` to the committed `z`-input openings via the R1CS matrices (the *inner*
  Spartan reduction = `R1CSEval`) is deferred to OPT-E. The honest test uses `Cz = Az∘Bz`.
- **Spartan `product`** is intentionally **not yet ported**: its essence *is* the univariate-skip over
  the 5 product polynomials, so it lands with the uni-skip machinery (OPT-E), not as a plain port.
- **Flag-keyed openings:** several Spartan ports cache openings jolt-core keys with
  `OpFlags(CircuitFlags)`/`InstructionFlags(...)` variants. Those flag enums aren't in our
  `VirtualPolynomial` yet, so the decoupled ports map them to **distinct existing variants** (documented
  in each file's `KEYS`/`SHIFT_KEYS` const). This is a harmless decoupled-test artifact; M8 will add the
  real flag variants when the witness is wired. **Review note:** the keys only need to be *distinct and
  consistent* between prover and verifier — they don't affect soundness of the decoupled round-trip.

### Test shape (every port has it)
`#[cfg(test)]`: a `round_trip` (prover → `SumcheckVerifier::verify`, several sizes, Fp3) asserting
(a) the verifier point equals the prover challenges, (b) the reduced claim equals
`expected_output_claim`, and (c) cached openings equal direct MLEs at `reverse(challenges)`; plus a
`tampered_proof_rejected` (corrupt round-poly 0 → verify errors). `Blake2bTranscript<F>` is the
transcript stand-in (the real prover threads the concrete spongefish state; see M1).

---

## 5. M6 — range checks

- **Booleanity (done):** `zkvm/booleanity.rs` proves the limbed-R1CS **carry/sign columns** are boolean
  (`x²−x=0`), batched with `γ^{2i}`, degree-3, via Gruen + accumulators. This is the *only* booleanity
  surviving the LogUp\*-GKR design — jolt-core's one-hot RA booleanity is subsumed by M7 (the one-hot
  `ra` is never committed). Cached under `CommittedPolynomial::R1csAux(i)` (a new key for the R1CS aux
  columns).
- **Wide-limb range checks (deferred to M8 wiring):** the 32-bit limb `<2³²` checks (MUL product, lookup
  outputs) fold into the **stage-5 Shout `RangeCheck`/`LowerHalfWord`/`UpperWord` tables** — *no new
  sumcheck instance*, just additional `Val_s` stages on the existing read-raf path. Can't be done
  standalone because stage 5 doesn't exist until the M8 driver.
- **Degree locks:** `val_evaluation`/`val_check` are already degree-3 (the signed-2-limb `Inc` keeps
  `Val = Σ inc·wa·LT` degree-3, per M4). Done.

---

## 6. Optimizations — what's done (OPT-A) and the pattern

**OPT-A (done):**
- **Gruen + Dao-Thaler split-eq** in `output_check`: the `eq` factor is handled by
  `GruenSplitEqPolynomial::gruen_poly_deg_3` (which reconstructs the degree-3 round poly from the
  quadratic's constant + X² coefficient using the eq factorization), instead of carrying `eq` as an
  explicit multilinear factor. This is the canonical jolt-core round-poly optimization.
- **Unreduced accumulators** (`F::Accumulator` = the M0 `Fp3Accumulator`) in **every** ported
  round-poly loop: `acc[k].fmadd(a, b)` accumulates `a·b` without per-add reduction, reducing once per
  evaluation point. Mechanical, workspace-ready, correctness-preserving (tests unchanged).

**Why only `output_check` got Gruen:** Gruen split-eq applies cleanly when `eq` is a *separate* factor
over its own variables. In the RW ports, `eq` is over cycle but the sum is over (address, cycle), so
Gruen there requires the **two-phase** structure (bind cycle with Gruen, then address) — that's OPT-C
(sparse `ReadWriteMatrix`), not a drop-in. In `val_*`, `eq` is folded into `LT`, so there's no separate
`eq` factor.

**The big deferred perf win — `base×ext` (2.3×):** the compact base-field `MultilinearPolynomial`
variants (`Fp3Accumulator::fmadd_base`) only help when the columns are *real base-field witnesses*
(`ra_dense` is u8, `Inc` is small). The decoupled tests use random Fp3, so there's nothing base-field to
exercise — this activates in M8 with the real witness. **This is the principled reason it's deferred,
not laziness:** it needs a pivot to real witness types.

---

## 7. Conventions & invariants — the checklist for reviewing any port

When reviewing a sumcheck port, verify these (they're the same everywhere):

1. **Retarget:** `JoltField → jolt_field::Field`; `F::Challenge → F`; no `#[cfg(feature="zk")]`
   (BlindFold) blocks. Doc-comment pins the jolt-core source.
2. **Bind order `LowToHigh`** everywhere; the framework's `sumcheck_evals_array` pairs `(2i, 2i+1)`.
3. **Opening points big-endian.** `normalize_opening_point(challenges)` = `reverse(challenges)` (the
   default). Cached openings are at this point. Verifier eq/structural recomputes use it.
4. **Endianness self-consistency** (the subtle one): a dense table built by `EqPolynomial::evals(r)` /
   `LtPolynomial::evaluations(r)` / `EqPlusOnePolynomial::evals(r)` and bound `LowToHigh` converges to
   that poly's MLE at **`reverse(challenges)`**. The verifier independently computes `…::mle(r, reverse(challenges))`.
   The round-trip tests assert these agree (the `dot(col, eq(reverse(challenges)))` checks).
5. **Degree & eval points:** degree-`d` round poly ⇒ `d+1` evaluation points ⇒
   `UnivariatePoly::from_evals(&[d+1 evals])`. Degree-2 → 3 points, degree-3 → 4 points.
6. **input_claim / expected_output_claim must be the two sides of the same identity.** `input_claim`
   reads upstream openings; `expected_output_claim` reads this instance's cached openings + recomputes
   structural factors (eq, LT, unmap, …). A mismatch here is the #1 place a bug hides — the round-trip
   test's `assert_eq!(value, expected)` catches it.
7. **Public vs committed columns:** structural/public columns (io_mask, val_io, unmap, Val_s, eq) are
   recomputed by the verifier (held in the instance or re-derived); only genuinely-committed columns
   (Inc, ra, Az/Bz/Cz, R1csAux) are *opened* and read from the accumulator.
8. **Lint policy:** `#[expect(...)]` not `#[allow(...)]`; `.unwrap()/.expect()` only in tests; bind
   `HashMap::insert` results with `let _ =`. Unused imports kept out of the lib target (test-only
   imports go in the test module).

---

## 8. What's DEFERRED — and why each is sound to defer

| Deferred item | Where it lands | Why it's safe to defer (not a correctness gap) |
|---|---|---|
| Trace → column materialization (witness-gen) | M8 | The sumcheck *identity* is proven faithfully over given columns; only the column *source* moves. Decoupling is what makes ports unit-testable. |
| **Compact base-field MLE variants** (`base×ext` 2.3× win) | M8 | Needs *real base-field witnesses*; decoupled Fp3 tests have no base data to exercise it. Requires the real witness types (a pivot), so it's a principled deferral, not laziness. |
| **Phase/gap-round interleaving** (output_check, val_check, RW, RAF) | M8 | The dummy "gap rounds" exist *only* because instances share a **batched** binding schedule — that schedule is the M8 stage driver. Meaningless standalone. |
| Gruen split-eq for RW ports | OPT-C | Requires the two-phase (address-then-cycle) structure = the sparse `ReadWriteMatrix`. |
| split-LT (√N memory) | OPT-B | `jolt_poly::LtPolynomial` binds HighToLow; framework is LowToHigh — needs a binding-order change or a new LT variant (a small pivot). Memory-only opt. |
| Full-d one-hot (d=4/32) | OPT-D | The naive product is degree `d+1` (huge for d=32); jolt-core keeps it tractable via prefix/suffix — so full-d *requires* the prefix/suffix decomposition. |
| Sparse `ReadWriteMatrix` two-phase | OPT-C | ~vendor; perf only. The single-phase broadcast form is correctness-equivalent. |
| prefix/suffix decomposition | OPT-D | ~vendor (774 LOC); perf + enables full-d. |
| univariate-skip + Multiquadratic + Lagrange + streaming + R1CSEval | OPT-E | ~vendor (~3200 LOC). Uni-skip is a round-*compression* opt; the underlying R1CS-satisfaction identity (Spartan outer) is already proven plainly. Enables the optimized Spartan + the `product` sumcheck. |
| BlindFold / ZK | Phase 3 | Out of scope for this phase (non-ZK); WHIR-zk replaces BlindFold later. |

**The mental rule:** anything deferred is either (a) a *performance* optimization that doesn't change
what's proven, or (b) genuinely blocked on machinery that doesn't exist yet (the stage driver, real
witness types) — i.e., it would need a design-choice pivot to do now. The *soundness-relevant* content
(the sumcheck identities, the limbed R1CS, the booleanity, the endianness contract) is all present.

---

## 9. What's LEFT (the remaining phases)

| Phase | Scope | Size / risk |
|---|---|---|
| **M7 — LogUp\*-GKR** | Port `crates/whir-pcs-bench/src/gkr.rs` (962 LOC) into `crates/jolt-whir/src/logup/` as framework `SumcheckInstance`s over Fp3: eq-weighted pushforward `P^F`, §4.5.2 d-claim→1 reduction, fan-in-2 fractional-add GKR (Gruen, degree-3), the 3 prototype asserts → fallible checks, WHIR leaf openings on `ra_dense`/`P^F`. Removes the subsumed one-hot booleanity/Hamming stages. | Large; intricate multi-layer GKR + WHIR commit integration. The read-raf ports already produce its inputs (the `ra_i` leaf claims). |
| **M8 — stage driver + e2e + equivalence (the gate)** | Port `prover.rs`/`verifier.rs` stages 1–8 (challenges Fp3, limb reads + recomposition, trace → signed-limb `Inc`), the stage-8 batched `WhirScheme` opening + GKR leaves, `muldiv_e2e_goldilocks` + `fibonacci_e2e_goldilocks`, `jolt-equivalence` claim-level cross-check vs the BN254 oracle. | Largest; wires *everything* with the real trace/witness. This is where the deferred opts (compact base MLE, phase/gap, two-phase, prefix/suffix, R1CSEval) and `product`/uni-skip all naturally come together. |
| **OPT-B…E** | The deferred perf vendoring (see §8). | OPT-E (uni-skip etc.) unblocks the optimized Spartan + `product`. |
| **Spartan `product`** | The univariate-skip product virtualization. | Lands with OPT-E (uni-skip-intrinsic). |

**Definition of done (M8 gate):** Goldilocks+WHIR prove→verify green on `muldiv` & `fibonacci`;
`jolt-equivalence` claim-level match vs jolt-core; **BN254 `muldiv` still green** (`host` + `host,zk`);
clippy + fmt clean.

---

## 10. Commit map (review entry points, newest first)

Each commit is a single self-contained piece with a round-trip + tamper test. Reviewing commit-by-commit
is the fastest path.

```
95e99b376  booleanity (M6 range checks)                         zkvm/booleanity.rs, R1csAux key
7678e5a70  Spartan outer (R1CS Az·Bz−Cz zero-check)             zkvm/spartan/outer.rs, SpartanAz/Bz/Cz keys
b74b0e6c3  Spartan instruction-input virtualization             zkvm/spartan/instruction_input.rs
59d5721da  Spartan shift (eq+1 PC identity)                      zkvm/spartan/shift.rs
959cf0d9d  OPT-A: unreduced accumulators everywhere             (all compute_message loops)
12f90c8da  OPT-A: Gruen split-eq in RAM output-check            zkvm/ram/output_check.rs
50f170d80  instruction-lookups read-raf + shared OneHotReadRaf  zkvm/shout_read_raf.rs (+ thin modules)
6c0bbc39e  bytecode read-raf checking                           (now thin over shout_read_raf)
0305b1cc5  RAM raf-evaluation                                   zkvm/ram/raf_evaluation.rs
1bc1aa3ef  RAM read-write-checking                              zkvm/ram/read_write_checking.rs
1f42740dc  register read-write-checking                         zkvm/registers/read_write_checking.rs
26130df02  RAM batched val-check                                zkvm/ram/val_check.rs
3751b7d3b  RAM output-check                                     zkvm/ram/output_check.rs
96427dc5c  registers val-evaluation                             zkvm/registers/val_evaluation.rs
d84b7de2e  PHASE2_HANDOFF.md
f320f6c41  Inc claim-reduction (the template)                   zkvm/claim_reductions/increments.rs
54b0540fd  opening accumulator                                  framework/accumulator.rs
8bf3c0143  SumcheckInstance trait + driver                      framework/sumcheck.rs
5ed82eae9  dense MultilinearPolynomial                          framework/poly.rs
a46e62f8b  full limbed RV64 R1CS                                r1cs/rv64_limbed.rs
90d5926a0  signed-value derivation                              r1cs/signed_value.rs
5c571d81c  limbed MUL 4-limb schoolbook                         r1cs/mul.rs
7e97db47b  LIMBED_R1CS.md
ce2440668  crate skeleton + field/PCS/transcript wiring         field.rs, lib.rs
d9c7e8a99  M4 base-field limb primitives + signed 2-limb Inc    (jolt-field, jolt-witness)
cc5af5c32  M3 WhirScheme batch open                             (jolt-whir)
60bf2696c  M2 WhirScheme commit/open/verify                     (jolt-whir)
08794dc74  M1 shared spongefish transcript                      (jolt-whir)
6d263f5d1  M0 Goldilocks/Fp3 deferred-reduction accumulators    (jolt-field)
4e30d3e88  Phase 1 complete (field/limbs/WHIR commit)           (jolt-field, jolt-witness, jolt-whir, jolt-pcs-bench)
```

**Suggested review order:** (1) `framework/` (poly → sumcheck → accumulator) to learn the engine + the
conventions in §7; (2) `claim_reductions/increments.rs` as the canonical template; (3) the leaf ports in
the table in §4 (they're all variations on the template); (4) `shout_read_raf.rs` (the d-chunk one-hot,
the M7 hand-off point); (5) `r1cs/` + `LIMBED_R1CS.md` (the soundness-critical arithmetization);
(6) `output_check.rs` for the optimized (Gruen + accumulator) pattern. The 45-test suite is the
executable spec — every identity is checked prover→verifier with a negative case.
