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

## 10. Commit-by-commit review reference

Every commit is one self-contained, individually-tested piece (no co-author trailer, not pushed).
This section is detailed enough to **review each commit by reading its entry** — for the sumcheck
ports it states the exact identity, degree, inputs, what is cached (keys/points/claims), what
`input_claim` reads, what `expected_output_claim` recomputes, the test assertions, and the
review-focus (where a bug would hide) + what is decoupled/deferred. Entries are in **build order
(oldest first)** — the order you should review in. The condensed hash list is at the end (§10.7).

Notation reminders (full detail in §7): `T = 2^{log_t}` cycles, `K = 2^{log_k}` addresses;
`F = GoldilocksFp3`; bind `LowToHigh`; opening points big-endian = `reverse(challenges)`; a dense
table from `…::evals(r)` bound `LowToHigh` converges to that poly's MLE at `reverse(challenges)`,
which the verifier independently computes via `…::mle(r, reverse(challenges))`.

---

### 10.1 Phase 1 + foundation (shared crates: jolt-field / jolt-witness / jolt-whir)

**`4e30d3e88` — Phase 1 complete (field / limbs / WHIR commit).**
Files: `jolt-field/src/goldilocks/{base,ext3,decompose,tests}.rs`, `jolt-witness/src/goldilocks.rs`,
`jolt-whir/src/{convert,params,commit,sanity}.rs`, `jolt-pcs-bench/src/fib_goldilocks.rs`.
*What/why:* stands up Goldilocks `Fp` (Montgomery-free), `Fp3` (`X³−2`, nonresidue 2), value↔limb
decompose, base-limb committed columns, WHIR base-commit, and a live fibonacci e2e vs BN254/Dory.
*Review focus:* `base.rs` `reduce128` (uses `2⁶⁴≡2³²−1`, `2⁹⁶≡−1`; non-canonical `[0,2⁶⁴)` rep,
canonicalize only at boundaries); `ext3.rs` `mul` (9 base muls) and `mul_by_base` (3); `tests.rs`
is a **num-bigint** oracle (independent reference) — this is your correctness anchor for the field.
`jolt-whir/tests/crosscheck.rs` cross-checks the hand-coded field op-for-op vs WHIR's `Field64`.
*Deferred:* all sumcheck/IOP, LogUp\*-GKR, limb *constraints* (Phase 1 makes columns, doesn't
constrain them), hiding. Full detail in `PHASE1_GOLDILOCKS_STATUS.md`.

**`6d263f5d1` — M0: deferred-reduction accumulators.** File: `jolt-field/src/goldilocks/accumulator.rs`.
*What/why:* `GoldilocksAccumulator`, `Fp3Accumulator` (+ `fmadd_base` for base×ext), impl
`FieldAccumulator`. Defers modular reduction across a sumcheck inner loop, reduces once. Analog of
BN254 `WideAccumulator`. *Review focus:* `fmadd`/`merge`/`reduce` must be field-equivalent to
`acc += a*b` / `+` / identity; the `EPSILON² mod p` overflow correction in the 192-bit lane. This is
what OPT-A (`959cf0d9d`) wires into every round-poly loop.

**`08794dc74` / `60bf2696c` / `cc5af5c32` — M1/M2/M3: shared transcript + `WhirScheme`** (`jolt-whir`).
*What/why:* one shared **spongefish** sponge for Jolt+WHIR (`challenge_fp3`); `WhirScheme`
commit/open/verify; batch-open via WHIR's geometric RLC. *Review focus:* `WhirScheme` is an
**inherent** API (not the `CommitmentScheme` trait — that trait pins `Transcript<Challenge=Self::Field>`
which can't wrap spongefish's state). These are used by M8's stage-8 opening, not by the decoupled
ports, so they're lower-priority for reviewing the sumcheck math.

**`d9c7e8a99` — M4: base-field limb primitives + signed 2-limb `Inc`** (`jolt-field`, `jolt-witness`).
*What/why:* `decompose.rs` limb helpers; the signed 2-limb `Inc` (`lo + hi·2³²`, `hi` signed).
*Review focus:* the **signed-2-limb choice is degree-load-bearing** — it keeps `Val = Σ inc·wa·LT`
degree-3 (a separate sign-bit factor would make it degree-4). `debug_assert |v| < 2⁶⁴`.

---

### 10.2 Crate skeleton + limbed RV64 R1CS (`crates/jolt-prover-goldilocks/src/{field,r1cs}`)

**`ce2440668` — crate skeleton + field/PCS/transcript wiring.** Files: `lib.rs` (20),
`field.rs` (21). *What/why:* `#![cfg(feature="goldilocks")]`; `type F = GoldilocksFp3`, `type Base =
Goldilocks`, `WhirScheme` re-exports; module tree. *Review focus:* trivial wiring; confirm the feature
gate + type aliases.

**`7e97db47b` — `LIMBED_R1CS.md`** (pinned design). Read this *before* the R1CS code; it's the
soundness argument (why field-recompose aliases mod p ⇒ limb-wise + `2⁻³²` carries).

**`5c571d81c` — limbed MUL 4-limb schoolbook.** File: `r1cs/mul.rs` (288). *What/why:* the 128-bit
MUL product `(Llo,Lhi)×(Rlo,Rhi)=(P0..P3)` via `2⁻³²` virtual carries; `Left.sign` pinned 0; sign
relation `Product.sign = Left.sign ⊕ Right.sign`. *Review focus:* the carry chain
(`t_i = Σ limb·limb + c_{i−1}`, `P_i = t_i mod 2³²`, `c_i = t_i ÷ 2³²`) and that partial products stay
degree-2 (outer sumcheck degree preserved). Test: honest products vs an **i128 reference**, plus tamper.

**`90d5926a0` — signed-value derivation.** File: `r1cs/signed_value.rs` (166). *What/why:* degree-2
`RIGHT_VAL = (1−2·sign)·magnitude` (two product rows + one linear). *Review focus:* **built + validated
but NOT wired** — reserved for signed-immediate multiplicative operands (not exercised by muldiv). It's
dead code until a trace needs it; review for correctness, not integration.

**`a46e62f8b` — full limbed RV64 R1CS.** File: `r1cs/rv64_limbed.rs` (951). *What/why:* all 22 RV64
constraints, limb-wise, 70 vars / 53 rows: per-limb equality (`guard·(a_lo−b_lo)=0` AND hi); full-u64
lookup-operand add/sub with `{0,1}` carries (`RLO = Left+Right` for ADD; `RLO+Right=Left+2⁶⁴` for SUB,
the `+2⁶⁴` becoming `+1` on the high carry to avoid a `2⁶⁴` field constant); `RamAddress=Rs1+Imm`
(limb1 exact); MUL via `mul.rs`; small-value single-element recompose (PCs `<2³²`, results `<p`);
boolean products (`ShouldBranch`/`ShouldJump`). *Review focus:* this is **soundness-critical**. Check
each constraint against `LIMBED_R1CS.md` §"Constraint transformation" and against jolt-core
`crates/jolt-r1cs/src/constraints/rv64.rs`. Validated by hand-built honest witnesses
(no-op/ADD/SUB/MUL/load + edges + 2000 random) and tamper rejection (`check_witness` tests).
**Soundness depends on the M6 booleanity (`95e99b376`)** for the carry/sign columns — review them
together.

---

### 10.3 The prover framework (`src/framework/`) — review FIRST

**`5ed82eae9` — dense `MultilinearPolynomial`.** File: `framework/poly.rs` (241). *What/why:* vendored
from jolt-core `poly/multilinear_polynomial.rs`. `bind_parallel(r, order) = lo + r·(hi−lo)`;
`sumcheck_evals_array::<D>(i, order)` = linear extrapolation through the bound pair
(`evals[k]=e0+k·(e1−e0)`); `final_sumcheck_claim`. *Review focus:* the `LowToHigh` pairing is
`(2i, 2i+1)`; `HighToLow` is `(i, i+half)`. Dense-only is intentional (compact base-field variants
deferred to M8). Tests: `bind` matches the recurrence; mini-sumcheck consistency (`s(0)+s(1)=claim`).

**`8bf3c0143` — `SumcheckInstance` trait + driver.** File: `framework/sumcheck.rs` (257).
*What/why:* the contract every port implements (see §3) + `prove`/`verify`. **The load-bearing
result:** `prove` emits a `jolt_sumcheck::SumcheckProof` that the workspace
`jolt_sumcheck::SumcheckVerifier::verify` accepts unchanged — prover and extracted verifier
interoperate. *Review focus:* the driver absorbs each round poly via the **same** `RoundProof`
path the verifier replays, then squeezes the challenge, then binds (so transcripts stay in lockstep);
`compute_message` is called *before* `bind` each round (pre-bind state). The built-in `ProductInstance`
test (`Σ A·B`) exercises the driver↔verifier bridge in isolation.

**`54b0540fd` — opening accumulator.** File: `framework/accumulator.rs` (349). *What/why:* the claim
store `HashMap<(PolynomialId, SumcheckId) → (OpeningPoint<BIG_ENDIAN,F>, F)>` shared by prover (writes
claims) and verifier (reads). `SumcheckId` (23 variants), `CommittedPolynomial`, `VirtualPolynomial`
vendored verbatim from jolt-core so keys match. *Review focus:* `OpeningPoint::match_endianness`
(reverses on a tag change); `append_dense`/`append_virtual` overwrite (HashMap insert); a missing
opening `panic!`s (a wiring bug, not recoverable input). **This is how subprotocols compose** — A's
`cache_openings` feeds B's `input_claim`. *(Later commits add keys: `SpartanAz/Bz/Cz` in `7678e5a70`,
`R1csAux(usize)` in `95e99b376`.)*

---

### 10.4 The ported sumchecks

Review pattern for each: confirm (a) `input_claim` and `expected_output_claim` are the two sides of
the **same** identity; (b) `cache_openings` stores exactly what `expected_output_claim` later reads;
(c) the round-trip test asserts `value == expected` and cached openings == direct MLE at
`reverse(challenges)`; (d) the tamper test rejects.

**`f320f6c41` — Inc claim-reduction (THE TEMPLATE — read first).** File:
`zkvm/claim_reductions/increments.rs` (388). jolt-core: `zkvm/claim_reductions/increments.rs`.
*Identity (degree-2, log_t rounds):* `Σ_j RamInc(j)·[eq(r2,j)+γ·eq(r4,j)] + γ²·Σ_j RdInc(j)·[eq(s4,j)+γ·eq(s5,j)]`,
reducing the four `RamInc`/`RdInc` openings (from RAM/register RW-checking + val-eval at distinct
points) to a single shared point ρ. *input_claim:* `v1 + γ·v2 + γ²·w1 + γ³·w2` (the four cached
openings). *Cached:* `RamInc(ρ)`, `RdInc(ρ)` @`IncClaimReduction`. *expected_output_claim:*
`RamInc(ρ)·(eq(r2,ρ)+γ·eq(r4,ρ)) + γ²·RdInc(ρ)·(eq(s4,ρ)+γ·eq(s5,ρ))` (eq via `EqPolynomial::mle`).
*Review focus:* this establishes the **endianness contract** every other port copies — verify the
`reverse(challenges)` ↔ `mle` correspondence in the test. Takes pre-materialized recomposed `Fp3`
Inc columns (trace→signed-limb materialization is M8). γ drawn from the transcript in `params::new`.

**`d84b7de2e` — `PHASE2_HANDOFF.md`** (docs). The prior-session handoff; superseded for review by
this guide.

**`96427dc5c` — registers val-evaluation.** File: `zkvm/registers/val_evaluation.rs` (359). jolt-core:
`zkvm/registers/val_evaluation.rs`. *Identity (degree-3, log_t rounds):*
`Val(r_address,r_cycle) = Σ_j inc(j)·wa(j)·LT(j, r_cycle)`. *Inputs:* materialized `inc` (RdInc) +
`wa` (write-address) columns; `LT(·,r_cycle)` table via `jolt_poly::LtPolynomial::evaluations`.
*input_claim:* the `RegistersVal` opening from `RegistersReadWriteChecking`, split at `log_k`.
*Cached:* `RdInc(ρ)` @`RegistersValEvaluation` (at r_cycle), `RdWa` @`RegistersValEvaluation`
(at `r_address‖r_cycle`). *expected_output_claim:* `inc_claim·wa_claim·LT(ρ,r_cycle)` where
`LT` = `LtPolynomial::evaluate(reverse(challenges), r_cycle)`. *Review focus:* the LT first-arg /
second-arg orientation (jolt-core's `LT(r_cycle,j)` ≡ our `LtPolynomial::evaluate(j, r_cycle)` —
confirmed against jolt-core's verifier loop). *Deferred:* split-LT, two-phase.

**`3751b7d3b` — RAM output-check** *(later optimized by `12f90c8da`)*. File: `zkvm/ram/output_check.rs`
(340). jolt-core: `zkvm/ram/output_check.rs`. *Identity (degree-3 zero-check, log_k rounds):*
`0 = Σ_k eq(r_address,k)·io_mask(k)·(Val_final(k)−Val_io(k))`. *Inputs:* materialized
`val_final`/`val_io`/`io_mask` columns (public program-I/O). *input_claim:* `0`. *Cached:*
`RamValFinal(ρ)` @`RamOutputCheck`. *expected_output_claim:* `eq(r_address,ρ)·io_mask(ρ)·(val_final_claim − val_io(ρ))`
— the verifier recomputes `io_mask`/`val_io` MLEs from the **public** columns (dot with `eq(ρ)`),
reads `val_final` from cache. *Review focus:* honest test builds `val_io = val_final` on the I/O region
and `io_mask` its indicator ⇒ the summand vanishes everywhere ⇒ input_claim 0. *Decoupled:* no
`JoltDevice`/`RangeMaskPolynomial`. *Deferred:* phase/gap-round interleaving (M8 batched driver).

**`26130df02` — RAM batched val-check.** File: `zkvm/ram/val_check.rs` (425). jolt-core:
`zkvm/ram/val_check.rs`. *Identity (degree-3, log_t rounds):* `Σ_j inc(j)·wa(j)·(LT(j,r_cycle)+γ)`,
batching the two RAM value identities `(1) Val−Val_init = Σ inc·wa·LT` and `(2) Val_final−Val_init = Σ inc·wa`
via γ. *input_claim:* `(val_rw − init_eval) + γ·(val_final − init_eval)` where `val_rw` =
`RamVal`@`RamReadWriteChecking`, `val_final` = `RamValFinal`@`RamOutputCheck`, `init_eval =
Val_init(r_address)` (MLE of the materialized initial-RAM column). *Cached:* `RamRa` @`RamValCheck`
(at `r_address‖r_cycle′`), `RamInc(ρ)` @`RamValCheck`. *expected_output_claim:*
`inc_claim·wa_claim·(LT(ρ,r_cycle)+γ)`. *Review focus:* the γ-batch folds the `+γ` into the LT factor;
the ZK `init_eval_public`/advice decomposition is **dropped** (non-ZK). The test draws γ from the
transcript and seeds val_rw so input_claim equals the real sum.

**`1f42740dc` — register read-write-checking.** File: `zkvm/registers/read_write_checking.rs` (508).
jolt-core: `zkvm/registers/read_write_checking.rs`. *Identity (degree-3, log_k+log_t rounds):*
`Σ_{k,j} eq(r_cycle,j)·[ra_merged(k,j)·Val(k,j) + wa(k,j)·(Val(k,j)+inc(j))]`, `ra_merged = γ·ra1 + γ²·ra2`.
*input_claim:* `rd_wv + γ·rs1_rv + γ²·rs2_rv` (the three `RegistersClaimReduction` openings).
*Cached:* `RegistersVal`,`Rs1Ra`,`Rs2Ra`,`RdWa` (virtual, at the full `r_address‖r_cycle` point) +
`RdInc` (committed, at r_cycle) @`RegistersReadWriteChecking`. *expected_output_claim:*
`eq(r_cycle,ρ_cycle)·(rd_wa·(inc+val) + γ·rs1_ra·val + γ²·rs2_ra·val)`. *Review focus:* the **single-phase
fully-broadcast** form — `ra1/ra2/wa/val` are full `K·T` dense (index `k·T+j`); cycle-only `eq`/`inc`
broadcast across the K address blocks so binding is uniform `LowToHigh`. `rs1_ra`/`rs2_ra` read
**directly** from the two materialized read columns (jolt-core derives `rs2_ra` and back-solves
`rs1_ra` — a perf trick, deferred). *Deferred:* sparse `ReadWriteMatrix` two-phase + Gruen (OPT-C).

**`1bc1aa3ef` — RAM read-write-checking.** File: `zkvm/ram/read_write_checking.rs` (404). jolt-core:
`zkvm/ram/read_write_checking.rs`. *Identity (degree-3, log_k+log_t rounds):*
`Σ_{k,j} eq(r_cycle,j)·ra(k,j)·(Val(k,j) + γ·(inc(j)+Val(k,j))) = rv + γ·wv`. *input_claim:* `rv + γ·wv`
(`RamReadValue`/`RamWriteValue` @`SpartanOuter`). *Cached:* `RamVal`,`RamRa` (virtual, full point) +
`RamInc` (committed, r_cycle) @`RamReadWriteChecking`. *expected_output_claim:*
`eq·ra·(val + γ·(val+inc))`. *Review focus:* a simpler single-`ra` analog of the register RW above
(same broadcast convention). *Deferred:* same as registers RW (OPT-C).

**`0305b1cc5` — RAM raf-evaluation.** File: `zkvm/ram/raf_evaluation.rs` (301). jolt-core:
`zkvm/ram/raf_evaluation.rs`. *Identity (degree-2, log_k rounds):* `Σ_k ra(k)·unmap(k) = raf_claim`,
where `ra(k) = Σ_j eq(r_cycle,j)·1[addr(j)=k]` (per-address access counts) and `unmap(k)` maps the
remapped index back to the original RAM address. *input_claim:* the `RamAddress` opening from
`SpartanOuter`. *Cached:* `RamRa` @`RamRafEvaluation` (at `r_address‖r_cycle`).
*expected_output_claim:* `unmap(ρ)·ra_claim`, `unmap(ρ)` = MLE of the **public** unmap column (dot with
`eq(ρ)`). *Review focus:* `unmap` is modeled as a materialized affine column `start_address + k`
(decoupled); the real `UnmapRamAddressPolynomial` + split-eq `ra` materialization + phase/gap scaling
are deferred. The `mul_pow_2` gap-scaling of jolt-core is dropped (single-phase, no gap).

**`6c0bbc39e` + `50f170d80` — bytecode & instruction-lookups read-raf (shared `OneHotReadRaf`).**
Files: `zkvm/shout_read_raf.rs` (537) + thin `zkvm/bytecode/read_raf_checking.rs` (31) and
`zkvm/instruction_lookups/read_raf_checking.rs` (42). jolt-core:
`zkvm/{bytecode,instruction_lookups}/read_raf_checking.rs`. *Identity (degree-3 = d+1 with d=2,
log_k+log_t rounds):* `Σ_{j,k} ra(k,j)·Σ_s γ^s·eq_s(j)·Val_s(k) = Σ_s γ^s·rv_s`, with the one-hot read
indicator `ra(k,j) = ∏_{i=0}^{d-1} ra_i(k_i,j)` (d-chunk product). *Generic* over the committed RA
family (`fn(usize)→CommittedPolynomial`) + `SumcheckId`, so both ports share it: **bytecode** uses
per-stage cycle points; **instruction-lookups** is the special case where all stages share
`r_cycle = r_reduction` (single eq), stages `{lookup-output, left-operand, right-operand}`.
*input_claim:* `Σ_s γ^s·rv_s` (per-stage upstream openings via the `rv_key` list). *Cached:* the d
per-chunk `ra_i` leaf claims `ra_i(r_addr_chunk_i, r_cycle)` under `(ra_family(i), sumcheck_id)`.
*expected_output_claim:* `ra_0·ra_1·Σ_s γ^s·eq_s(ρ_cycle)·Val_s(ρ_addr)` (eq via `mle`, Val via public
column dot). *Review focus:* **`6c0bbc39e` first introduced this as `BytecodeReadRaf`; `50f170d80`
generalized it into `OneHotReadRaf` and made bytecode a thin wrapper** — so review the final
`shout_read_raf.rs`, not the intermediate. The chunk columns are **broadcast** to the full hypercube
(index `(k0·K1+k1)·T+j`) for uniform binding. **These cached `ra_i` leaf claims are exactly the §4.5.2
inputs the M7 LogUp\*-GKR consumes** — the sumcheck is unchanged by M7; only the leaf commitment changes
(see the `m7-logupstar-readraf-relationship` memory). *Deferred:* prefix/suffix + two-phase, entry-point
constraint, flag/table-specific Val construction, full-d (>2), the d-chunk one-hot *commitment* (M7).

**`12f90c8da` + `959cf0d9d` — OPT-A: Gruen split-eq + unreduced accumulators.**
*`12f90c8da`* rewrites `output_check.compute_message` to use `GruenSplitEqPolynomial::fold_out_in`
(compute the constant + X² coeff of `q = io_mask·(val_final−val_io)`) → `gruen_poly_deg_3` (the eq
factor handled by the eq-factorization, not as an explicit multilinear). *`959cf0d9d`* converts every
other port's round-poly accumulation to `F::Accumulator` (`Fp3Accumulator`) — `acc[k].fmadd(a,b)` then
reduce once per eval point. *Review focus:* both are **correctness-preserving** (all tests unchanged
and green) — verify the Gruen `q_constant`/`q_quadratic` derivation matches jolt-core's output_check,
and that `fmadd(a,b)` is used as `a·b` (the two-factor grouping is correct, e.g.
`acc.fmadd(eq_e[k]*ra_e[k], val + γ(val+inc))`). The base×ext `fmadd_base` win is **not** active yet
(needs real base witnesses, M8).

**`59d5721da` — Spartan shift (PC) sumcheck.** File: `zkvm/spartan/shift.rs` (419). jolt-core:
`zkvm/spartan/shift.rs`. *Identity (degree-2, log_t rounds):*
`Σ_j [EqPlusOne(r_outer,j)·(s0+γs1+γ²s2+γ³s3)(j) + γ⁴·EqPlusOne(r_product,j)·(1−s4(j))]`. *Inputs:* five
`f(j+1)`-aligned shift columns `s0..s4`; two `EqPlusOne` tables via `EqPlusOnePolynomial::evals`.
*input_claim:* `NextUnexpandedPC + γ·NextPC + γ²·NextIsVirtual + γ³·NextIsFirstInSequence + γ⁴·(1−NextIsNoop)`
(the first four from `SpartanOuter`, NextIsNoop from `SpartanProductVirtualization`). *Cached:* the 5
shift openings @`SpartanShift` (keyed by the `SHIFT_KEYS` const — distinct existing variants; jolt-core
uses `OpFlags`/`InstructionFlags`). *expected_output_claim:*
`eqp_outer·(s0+γs1+γ²s2+γ³s3) + γ⁴·eqp_product·(1−s4)`, `eqp_* = EqPlusOnePolynomial::new(r).evaluate(ρ)`.
*Review focus:* the **two** eq+1 points (`r_outer` for terms 0–3, `r_product` for term 4); the test
seeds the five Next* claims (γ-independent) so input_claim equals the real shift sum for any γ.
*Deferred:* prefix-suffix two-phase EqPlusOne (OPT-E).

**`b74b0e6c3` — Spartan instruction-input virtualization.** File: `zkvm/spartan/instruction_input.rs`
(335). jolt-core: `zkvm/spartan/instruction_input.rs`. *Identity (degree-3, log_t rounds):*
`Σ_j eq(r_cycle,j)·(RightInput + γ·LeftInput)`, `LeftInput = left_is_rs1·rs1 + left_is_pc·upc`,
`RightInput = right_is_rs2·rs2 + right_is_imm·imm`. *input_claim:* `right + γ·left`
(`Left`/`RightInstructionInput` @`SpartanProductVirtualization`). *Cached:* the 8 flag/value openings
@`InstructionInputVirtualization` (`KEYS` const, order
`[left_is_rs1, rs1, left_is_pc, upc, right_is_rs2, rs2, right_is_imm, imm]`). *expected_output_claim:*
`eq(r_cycle,ρ)·((right_is_rs2·rs2 + right_is_imm·imm) + γ·(left_is_rs1·rs1 + left_is_pc·upc))` from the
8 cached openings. *Review focus:* degree-3 comes from `eq·(flag·value)`; the verifier reconstructs
Left/RightInput from the individual openings (not from a single combined column).

**`7678e5a70` — Spartan outer (R1CS satisfaction).** File: `zkvm/spartan/outer.rs` (279). jolt-core:
`zkvm/spartan/outer.rs`. *Identity (degree-3 zero-check, |τ| rounds):*
`0 = Σ_x eq(τ,x)·(Az(x)·Bz(x) − Cz(x))`. *Inputs:* materialized `Az`/`Bz`/`Cz` columns. *input_claim:*
`0`. *Cached:* `SpartanAz`/`SpartanBz`/`SpartanCz` (new `VirtualPolynomial` variants, added here) @
`SpartanOuter`. *expected_output_claim:* `eq(τ,ρ)·(Az·Bz − Cz)`. *Review focus:* honest test uses
`Cz = Az∘Bz` so the zero-check holds. **The matrix→z reduction is deferred** — jolt-core caches the
individual `z`-input openings (`ALL_R1CS_INPUTS`) and reconstructs `Az/Bz/Cz` via the R1CS matrices
(the inner Spartan reduction = `R1CSEval`); the decoupled port abstracts that to direct `Az/Bz/Cz`
columns. *Deferred (OPT-E):* univariate-skip first round, streaming, `R1CSEval`. The Spartan **product**
sumcheck is uni-skip-intrinsic and ports with OPT-E.

**`95e99b376` — booleanity (M6 range checks).** File: `zkvm/booleanity.rs` (347). jolt-core:
`subprotocols/booleanity.rs`. *Identity (degree-3 zero-check, log_k rounds):*
`0 = Σ_x eq(r,x)·Σ_i γ^{2i}·(b_i(x)² − b_i(x))`. *Inputs:* the boolean carry/sign columns. *input_claim:*
`0`. *Cached:* `R1csAux(i)` (new `CommittedPolynomial` variant) @`Booleanity`. *expected_output_claim:*
`eq(r,ρ)·Σ_i γ^{2i}·(b_i(ρ)²−b_i(ρ))` from the cached openings. *Review focus:* uses Gruen + accumulators
(the per-pair `q` has `q_constant = b0²−b0`, X² coeff `= (b1−b0)²`). **The negative test
(`non_boolean_column_mismatch`) is subtle and worth understanding:** `gruen_poly_deg_3` *forces*
`s(0)+s(1)=claim`, so `verify()` *accepts* a non-boolean column — the failure surfaces only at the
`expected_output_claim` discharge (`value != expected`). That's how booleanity actually rejects, and it
mirrors how the M8 driver will catch it. This is the **only** booleanity surviving the LogUp\*-GKR
design; **the limbed R1CS soundness depends on it** (review with `a46e62f8b`).

---

### 10.5 Cross-cutting things to verify once (apply to all ports)

- **Endianness round-trip** (the subtle invariant): for every port, the test's
  `assert_eq!(cached_claim, dot(column, EqPolynomial::evals(reverse(challenges))))` confirms the dense
  table ↔ MLE-at-`reverse(challenges)` correspondence. If any port got this wrong, that assert fails.
- **input/output identity symmetry:** for every port, `input_claim` (LHS, from upstream openings) and
  `expected_output_claim` (RHS, from this port's cached openings + recomputed structural factors) are
  the two sides of the stated identity. The test's `assert_eq!(value, expected)` is the catch-all.
- **Degree ↔ eval points:** degree-2 → 3 points, degree-3 → 4 points (`UnivariatePoly::from_evals`).
- **Decoupled-key artifact:** Spartan/shift/instruction-input cache under *remapped* existing
  `VirtualPolynomial` variants (the `KEYS`/`SHIFT_KEYS` consts) because the real `OpFlags`/
  `InstructionFlags` variants aren't wired yet — harmless for the round-trip (keys only need to be
  distinct + consistent), to be replaced in M8.

### 10.6 What is NOT yet committed (so you don't go looking for it)

Spartan **product**; the M7 LogUp\*-GKR (`crates/jolt-whir/src/logup/`); the M8 stage driver, e2e tests,
and `jolt-equivalence` cross-check; the deferred opts OPT-B…E (split-LT, sparse `ReadWriteMatrix`,
prefix/suffix, univariate-skip + Multiquadratic + Lagrange + streaming + `R1CSEval`); the compact
base-field MLE variants; the flag-carrying `VirtualPolynomial` variants. See §8–§9.

### 10.7 Condensed hash list (newest first)

```
95e99b376  booleanity (M6)                       7678e5a70  Spartan outer
b74b0e6c3  Spartan instruction-input             59d5721da  Spartan shift
959cf0d9d  OPT-A accumulators everywhere         12f90c8da  OPT-A Gruen (output_check)
50f170d80  instr read-raf + shared OneHotReadRaf 6c0bbc39e  bytecode read-raf
0305b1cc5  RAM raf-evaluation                    1bc1aa3ef  RAM read-write-checking
1f42740dc  registers read-write-checking         26130df02  RAM val-check
3751b7d3b  RAM output-check                      96427dc5c  registers val-evaluation
d84b7de2e  PHASE2_HANDOFF.md                     f320f6c41  Inc claim-reduction (template)
54b0540fd  opening accumulator                   8bf3c0143  SumcheckInstance + driver
5ed82eae9  dense MultilinearPolynomial           a46e62f8b  full limbed RV64 R1CS
90d5926a0  signed-value derivation               5c571d81c  limbed MUL schoolbook
7e97db47b  LIMBED_R1CS.md                        ce2440668  crate skeleton
d9c7e8a99  M4 limb primitives + signed Inc       cc5af5c32  M3 WhirScheme batch open
60bf2696c  M2 WhirScheme commit/open/verify      08794dc74  M1 shared spongefish transcript
6d263f5d1  M0 deferred-reduction accumulators    4e30d3e88  Phase 1 (field/limbs/WHIR commit)
```

**Suggested review order:** (1) `framework/` (poly → sumcheck → accumulator), §10.3, to learn the
engine + the §7 conventions; (2) `increments.rs` (`f320f6c41`), the canonical template; (3) the leaf
ports §10.4 top-to-bottom (variations on the template); (4) `shout_read_raf.rs` (`50f170d80`), the
d-chunk one-hot + M7 hand-off; (5) `r1cs/` + `LIMBED_R1CS.md` (`a46e62f8b`/`5c571d81c`), the
soundness-critical arithmetization, together with booleanity (`95e99b376`); (6) `output_check.rs`
(`12f90c8da`) for the optimized Gruen + accumulator pattern. The 45-test suite is the executable
spec — every identity is checked prover→verifier with a negative case.
