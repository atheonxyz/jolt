# Goldilocks + WHIR Migration — Phase 3 Review Guide (M7 + M8-in-progress)

**Audience:** you, reviewing the AI-generated code. **Goal:** explain *everything* built **after**
`PHASE1_PHASE2_REVIEW_GUIDE.md` — the **M7 LogUp\*-GKR** and the **M8 stage-driver build-out so
far** — with *why* each design choice was made, what is deliberately **deferred** (and why that's
sound), and exactly what is **left for the e2e gate**.

Read `PHASE1_PHASE2_REVIEW_GUIDE.md` first; it covers the field/limbs/WHIR-commit (Phase 1), the
prover framework, the limbed RV64 R1CS, and the leaf/Spartan/booleanity sumcheck ports (M0–M6). That
guide's §10.6 explicitly listed *"M7 LogUp\*-GKR; the M8 stage driver, e2e tests, and
`jolt-equivalence` cross-check"* as **NOT yet committed**. **This doc is exactly that work** — the
boundary is commit `95e99b376` (booleanity, the last code commit in the Phase 1/2 hash list).

---

## 0. The 60-second mental model (what Phase 3 adds)

Phase 1/2 left us with: every **leaf sumcheck identity** proven over decoupled (pre-materialized)
columns, the limbed R1CS, booleanity, and the framework — all unit-tested, but **no way to run a
real trace end-to-end** and no LogUp\*-GKR.

Phase 3 adds, in dependency order:

1. **M7 — LogUp\*-GKR** (`src/zkvm/logup/`, ~2100 LOC): replaces jolt-core's committed one-hot `ra`
   with a **dense `ra_dense` index column** + an **eq-weighted pushforward `P^F`**, proven equal via
   a **fan-in-2 fractional-add GKR** + the **§4.5.2 d-claim→1 reduction**. The read-raf ports from
   Phase 2 produce its inputs (the `ra_i` leaf claims). Includes the **Option C** resolution of the
   read-raf↔§4.5.2 point-mismatch.
2. **M8 framework foundations** (`src/framework/{lagrange,multiquadratic,univariate_skip}.rs` +
   batched sumcheck): the OPT-E machinery (uni-skip + batched front-loaded sumcheck). **Built &
   tested, but the uni-skip *Spartan* that uses them is deferred** — see §5.
3. **M8 witness materialization** (`src/zkvm/{witness,r1cs_witness}.rs`, `registers/witness.rs`):
   trace → the committed `ra_dense`/`Inc` columns, the limbed cycle-major `z`+`Az/Bz/Cz`, and the
   register-file `K·T` matrices. This is the **"glue" that was decoupled-away in Phase 2**, now being
   filled in.
4. **M8 binary Spartan stage + multi-stage driver** (`src/zkvm/spartan/{inner,stage}.rs`,
   `src/zkvm/driver.rs`): the outer zero-check + an **inner reduction over `jolt_r1cs::R1csKey`**, and
   `prove_binary`/`verify_binary` wiring stages onto one shared transcript+accumulator. **This is the
   spine the remaining stages attach to.**
5. **M8/P5 — the 5 missing claim-reductions** (`src/zkvm/claim_reductions/`): 4 ported
   (`registers`, `ram_ra`, `instruction_lookups`, `hamming_weight`); `advice` deferred.

**The single biggest design choice in Phase 3:** the M8 Spartan is **binary** (a plain
zero-check + an `R1csKey`-based inner reduction), **NOT** jolt-core's univariate-skip Spartan. This
is the *reuse-the-workspace, correctness-first, reach-the-equivalence-gate-soonest* path; uni-skip is
deferred. See §5 — it's the load-bearing decision and the one most worth reviewing.

### Status snapshot

| | |
|---|---|
| Tests (this crate, `--features goldilocks`) | **102 passing, 0 skipped** (was 45 at end of Phase 2) |
| clippy `--features goldilocks --all-targets -D warnings` | exit 0 |
| `cargo fmt` | clean |
| BN254/Dory regression (`jolt-core muldiv`) | untouched (not run this phase; no jolt-core edits) |
| **P3 + P4 (RAM + registers stages, fully composed)** | ✅ **DONE** — `zkvm/memory.rs` round-trips on a real trace |
| **e2e (`muldiv`/`fibonacci` Goldilocks)** | **NOT yet — the remaining P6–P11 work, §9–§10** |

```bash
cargo nextest run -p jolt-prover-goldilocks --features goldilocks --cargo-quiet     # 102 green
cargo clippy   -p jolt-prover-goldilocks --features goldilocks --all-targets -q -- -D warnings
```

### The Phase-3 commit series (oldest first; `git log 95e99b376..HEAD`)

```
bc0331a02  M7 LogUp*-GKR pushforward prep + §4.5.2 reduction (piece a)
76f379d7b  M7 fan-in-2 GKR circuit + per-layer SumcheckInstance (piece b)
756a59ca5  M7 per-family pushforward GKR prover+verifier (pieces c+d)
1b92c5b6e  M7 per-family driver consuming shout_read_raf outputs (piece e)
c43b4038a  M7 pin the A-circuit leaf structural-check formula
b17e6bf1a  M8/P5 port registers claim-reduction (single-phase)
2bacf6b5e  M8/P5 port ram_ra claim-reduction (single-phase)
52cdd4484  M8/P5 port instruction_lookups claim-reduction
458d25fe3  M8/P5 port hamming_weight reduction; note advice deferral
e2b4f4978  M8 framework uni-skip + batched-sumcheck foundations
bd25a4588  M7 Option C per-chunk pushforward driver
3448f67a4  M8 committed-witness materialization (CommittedWitness)
9eddcbd9f  M8 limbed RV64 R1CS witness materialization
bae099fa7  M8 binary Spartan stage (outer zero-check + inner reduction)
980c96f36  M8 binary-Spartan multi-stage prove/verify driver
cc535697d  M8/P4 register-file witness materialization
6156a73ad  M8/P3 RAM witness materialization
812eb801e  M8/P4 registers stage pipeline (claim-reduction → RW → val-eval)
756c28aa8  M8/P3 RAM stage pipeline (batched-aligned RW+RAF+output, then val-check)
82d975986  M8/P3+P4 combined memory stage (RAM + registers + Inc/RamRa reductions)
```
(All local, no co-author trailer, **not pushed**. The two `docs(...)` commits between `95e99b376`
and `bc0331a02` are the Phase 1/2 guide itself. The last 4 commits — `6156a73ad`..`82d975986` —
**complete P3 + P4**.)

---

## 1. M7 — LogUp\*-GKR (`src/zkvm/logup/`, ~2100 LOC) — DONE & tested

### Why LogUp\*-GKR exists (the design pivot)

jolt-core commits the **one-hot `ra` matrix** (`K·T`, mostly zeros) and proves read consistency via
booleanity + Hamming-weight + read-checking sumchecks. Over Goldilocks+WHIR that's wasteful. The
LogUp\* design instead commits a **dense `ra_dense` index column** (`T` entries, each the active
address-chunk index `k`), and proves — via a GKR argument — that `ra_dense` is consistent with an
**eq-weighted pushforward** `P^F(k) = Σ_{j: ra_dense[j]=k} eq(r_cycle, j)`. This **subsumes the
one-hot booleanity + Hamming-weight stages entirely** (the one-hot `ra` is never committed).

### Module map

| File (LOC) | Role | Key types / API |
|---|---|---|
| `logup/mod.rs` (170) | namespace + LSB-first eq/MLE utilities | `GkrError`, `lsb_eq_table`, `mle_eval_lsb`, `idx_mle_lsb` |
| `logup/pushforward.rs` (362) | claim eval + §4.5.2 reduction → `PushforwardData` builder | `Family<F>{log_t,log_d,log_m,r_row,r_col,indices}`, `PushforwardData`, `VerifierView`, `GkrError::MainIdentity` |
| `logup/circuit.rs` (145) | fan-in-2 fractional-add GKR pyramid | `Circuit<F>{levels}`, `root()`, `level(k)` |
| `logup/layer.rs` (322) | one GKR layer as a framework `SumcheckInstance` (degree-3, Gruen split-eq) | `GkrLayer<F>`, `leaf_values()` |
| `logup/gkr.rs` (477) | per-family GKR prover & verifier (two circuits A+B → 3 openings) | `prove_family_gkr`/`verify_family_gkr`, `GkrProof{a,b}` |
| `logup/driver.rs` (625) | M7 entry: read-raf hand-off → per-family / **per-chunk (Option C)** dispatch | `prove_family_per_chunk`/`verify_family_per_chunk`, `ChunkPushforward`, `ChunkVerifierInput` |

### The math, in three layers

1. **§4.5.2 reduction (`pushforward.rs`):** read-raf hands M7 `d` claims `M̃^(i)(r_cycle, r_k_i)`
   (one per address chunk). The reduction absorbs the `d` claims into the transcript → squeezes a
   chunk-combiner `r_chunk` → forms a single combined claim → absorbs it → squeezes `α`. It checks
   **eq. 5**: `combined == P̃^F(r_col)` (the pushforward MLE at the column point). `GkrError::MainIdentity`
   fires if `P^F` is inconsistent with `ra_dense`.
2. **Fan-in-2 fractional-add GKR (`circuit.rs` + `layer.rs`):** two circuits.
   - **Circuit A** over the `ra_dense` leaves: numerator `eq(r_M_row, j)`, denominator `α − M*[j]`.
   - **Circuit B** over the `P^F` leaves: numerator `P^F[k]`, denominator `α − k`.
   Each level fuses two children `(n,d),(n',d')` → `(n·d' + n'·d, d·d')`. The **root histogram** must
   satisfy `N_A·D_B == N_B·D_A` (the two fractional sums agree). Each layer is a degree-3
   `SumcheckInstance` (eq × fractional-combine) using **Gruen split-eq** (the OPT-A pattern).
3. **Per-family prover/verifier (`gkr.rs`):** `prove_family_gkr` outputs **3 openings** — `M̃*(r*_A)`
   (A-circuit leaf point), `P̃^F(r*_B)` (B-circuit leaf point), `P̃^F(r_col)` (the §4.5.2 reduced
   point). `verify_family_gkr` reconstructs via root-histogram + per-layer consistency + leaf-structural
   checks (`GkrError::{RootHistogram, LayerConsistency, LeafStructural}`).

### DESIGN CHOICE — **Option C** (the read-raf ↔ §4.5.2 reconciliation) — `driver.rs`

**The gap (memory `m7-readraf-shared-point-gap`):** the read-raf ports cache each chunk's `ra_i`
opening at a **distinct** chunk point `r_k_i` (the `d` chunks bind to different randomness), but the
§4.5.2 design as written assumed a **shared** `(r_row, r_col)` across the `d` claims.

**Resolution — Option C (per-chunk pushforward at `log_d = 0`):** instead of one family-level
pushforward over all `d` chunks at a shared point, run **`d` separate pushforward-GKRs**, one per
chunk, each at its **own** `r_col = r_k_i`. At `log_d = 0` the §4.5.2 reduction degenerates to the
**base identity** `M̃^(i)(r_cycle, r_k_i) = P̃^F_i(r_k_i)` — i.e. each chunk's read-raf claim *is*
its pushforward claim, with **zero new soundness math** (it reuses the exact per-family machinery at
`d = 1`). `prove_family_per_chunk` builds one `Family{log_d: 0, r_row: rev(r_cycle), r_col:
rev(r_k_i), indices: [chunk_i]}` per chunk and appends 3 openings keyed by `base_index + i`. This is
the **M8 production path**; `prove_family` (shared `r_col`, `log_d > 0`) is retained for a future
OPT-D. *(User-approved: "go with Option C".)*

### CONVENTION — bit-ordering (LSB-first vs MSB-first)

**All internal LogUp\* math is LSB-first** (the paper's order: `lsb_eq_table`, `mle_eval_lsb`,
`idx_mle_lsb`, the `GkrLayer` split-eq). **Read-raf caches MSB-first** (`EqPolynomial`,
`BIG_ENDIAN`). The bridge is local to the driver: `prove_family_per_chunk`/`verify_family_per_chunk`
**reverse `r_cycle` and `r_col`** before building the `Family`, so the hand-off is transparent to the
GKR. This is the one place the two conventions meet — review it in `driver.rs::rev`.

### Tests (all green)

`readraf_per_chunk_option_c_round_trip` (real `d=2` read-raf → 2 chunks at distinct `r_col` → 2×3=6
accumulator openings round-trip, asserts the chunk points are distinct via `assert_ne!`),
`readraf_handoff_round_trip`, `pushforward_main_identity`, `corrupted_chunk_claim_trips_main_identity`,
`corrupted_claim_trips_main_identity`, `layer_round_trip`, `root_is_fractional_sum`,
`structural_eq_recompute_is_order_consistent`.

---

## 2. M8 framework foundations (`src/framework/`) — DONE & tested, but Spartan-use DEFERRED

These are the **OPT-E machinery**. They are built and unit-tested, but the **uni-skip Spartan** that
would consume them is **deferred** (§5) — so right now they are *foundations waiting for their
consumer*, intentionally (user: "give me a working e2e first however clearly mention it somewhere
that this is to be done later").

| File (LOC) | What | Why / design notes |
|---|---|---|
| `framework/lagrange.rs` (725) | `lagrange_kernel::<N>(x,y)` on a **symmetric integer grid** `{start..start+N-1}`, `start = -⌊(N-1)/2⌋`; `LagrangeHelper` const-fn interpolation; `check_sum_evals` | Field-agnostic (the uni-skip domain is an integer window, identical over any field) — so it ports with **zero field drift** (memory `m8-opt-e-faithful-port`). |
| `framework/multiquadratic.rs` (456) | `MultiquadraticPolynomial` on the `{0,1,∞}^n` base-3 grid; `expand_linear_grid_to_multiquadratic` `(e0,e1)→(e0,e1,e1−e0)`; `bind_first_variable(r)` via `r·(r−1)` | The uni-skip round-message compression representation. |
| `framework/univariate_skip.rs` (416) | `build_uniskip_first_round_poly::<DOMAIN_SIZE,DEGREE,EXTENDED_SIZE,NUM_COEFFS>` (`NUM_COEFFS=3·DEGREE+1`, `EXTENDED_SIZE=2·DEGREE+1`, `DOMAIN_SIZE=DEGREE+1`); `prove/verify_uniskip_round`; `UniSkipError` | The uni-skip first round: `s1 = L̃(τ_high,·)·t1(·)`. Collapses the ≤6 limbed constraint rounds → 2. |
| `framework/sumcheck.rs` (Δ) | **`prove_batched`/`verify_batched`** (front-loaded, `α^j` powers, gap-round dummies `prev/2`, `2^(max−n)` pre-scaling) + `round_offset`/`finalize` trait **defaults** | **CRITICAL:** the prover uses **`α^j` powers**, *not* jolt-core's `challenge_vector`, to match the workspace `BatchedSumcheckVerifier`. The trait gained `round_offset`/`finalize` as *defaults* (no existing port changed). |

**Drift verdict (memory `m8-opt-e-faithful-port`):** the OPT-E port is faithful — no field/domain/PCS
drift; the uni-skip domain is a field-agnostic integer window. Validated by porting + testing the
foundations before committing to the deferral.

---

## 3. M8 witness materialization — DONE & tested (the Phase-2 "decoupling", now filled in)

Phase 2 ports took columns as input; M8 produces those columns from a trace. Three materializers
exist, each **validated against the real consuming stage**, not just self-checked.

### `src/zkvm/witness.rs` (289) — `CommittedWitness` (the committed base-field columns)
`CommittedWitness::build(sources: &CommitmentTraceSources, layout: &GoldilocksLayout)` produces the
`ra_dense` index columns per `(family, chunk)` — **keyed by a global chunk index** (instruction
chunks, then bytecode, then ram) which is *exactly* the `CommittedPolynomial::RaDense(i)` /
`Pushforward(i)` accumulator key **and** the Option C `base_index` — plus the recomposed `RdInc`/`RamInc`
MLEs. **Decoupled:** takes `jolt_witness::CommitmentTraceSources` (field-agnostic dense + one-hot
sources), so the trace→sources extraction (`extract_trace`) is the e2e path, not this module.
**Deferred (documented):** the per-limb `Inc` commit layout (lo/hi vs recomposed) is a stage-8
decision (§7 fork 3).

### `src/zkvm/r1cs_witness.rs` (624) — limbed RV64 `z` + `Az/Bz/Cz`
`cycle_to_z::<C,F>(trace, t, pcs)` / `build_limbed_z` map each trace cycle into the **limbed 70-var
`Vars` layout** (signed 2-limb `Inc`, MUL schoolbook + add/sub carries, RAM address `Rs1+Imm`), and
`R1csWitness::materialize` builds the **cycle-major** `z` + `Az/Bz/Cz` over the `(cycle, constraint)`
hypercube, matching `jolt_r1cs::R1csKey`'s uniform factorization. `boolean_aux_columns()` extracts the
8 carry/sign columns for the M6 booleanity residual.

> **KEY soundness finding (review focus):** this does **NOT** reuse `jolt_trace::extract_trace`'s flat
> R1CS witness — that uses the workspace **BN254 non-limbed** `jolt_r1cs::constraints::rv64`
> (`NUM_VARS_PER_CYCLE`), which is **unsound over Goldilocks** (every u64 aliases mod p). The
> goldilocks crate deliberately builds its **own limbed `z`**. At e2e, use `extract_trace` *only* for
> the field-agnostic `CycleInput[]` (feeding `CommittedWitness`), and `build_limbed_z` for the R1CS
> witness. This reconciliation is the answer to the plan's open question #1.

### `src/zkvm/registers/witness.rs` (295) — register-file `K·T` matrices (M8/P4)
`register_witness::<C,F>(trace, register_count)` **simulates the register file**: it tracks register
state and derives `val`/`rs1_value`/`rs2_value`/`inc` from that state (read-before-write), while the
trace supplies only the read/write **addresses** + the `rd` post-value. So the materialized `K·T`
matrices (`ra1`/`ra2`/`wa`/`val`) + cycle columns satisfy the read-write-checking relation
`Σ_{k,j} eq·[(γ·ra1+γ²·ra2)·Val + wa·(Val+inc)] = rd_wv + γ·rs1 + γ²·rs2` **by construction**.
**Validated by feeding the real matrices into `RegistersReadWriteChecking` (round-trips)** + a
read-before-write snapshot test — not synthetic random columns. `register_count` is a parameter
(`REGISTER_COUNT = 128` in the real flow; tests use 8).

---

## 4. M8/P5 — the 5 missing claim-reductions (`src/zkvm/claim_reductions/`) — 4 DONE, advice deferred

Phase 2 had only `increments`. M8/P5 ports the rest, each a **faithful single-phase** port (jolt-core's
prefix/suffix two-phase materialization is a deferred perf opt), each with synthetic-opening
round-trip + tamper tests mirroring `increments.rs`.

| Port (file, LOC) | jolt-core source | Deg | Identity (single-phase) | Output openings |
|---|---|---|---|---|
| `registers.rs` (418) | `claim_reductions/registers.rs` | 2 | `Σ_j eq(r_spartan,j)·(Rd + γ·Rs1 + γ²·Rs2)` | `RdWriteValue`,`Rs1Value`,`Rs2Value` @RegistersClaimReduction |
| `ram_ra.rs` (397) | `claim_reductions/ram_ra.rs` | 2 | `Σ_c (eq_raf + γ·eq_rw + γ²·eq_val)·ra(r_addr,c)` → `RamRa(r_addr‖ρ)` | `RamRa` @RamRaClaimReduction |
| `instruction_lookups.rs` (347) | `claim_reductions/instruction_lookups.rs` | 2 | `Σ_j eq(r_spartan,j)·Σ_{i<5} γⁱ·valᵢ(j)` | 5 lookup openings @InstructionClaimReduction |
| `hamming_weight.rs` (477) | `claim_reductions/hamming_weight.rs` | 2 | fused HW + Booleanity/Virtualization reduction over `Gᵢ(k)`, `log_k_chunk` **address** rounds | `RaDense`/RA openings @HammingWeightClaimReduction |

**Design notes:**
- `ram_ra` and `hamming_weight` **override `normalize_opening_point`** — their output point is
  `r_address ‖ r_cycle` (concatenated), not just the reversed challenges.
- `hamming_weight` runs **`log_k_chunk` address rounds** (not `log_T`); `H_i` (the hamming weight) is
  `1` for instruction/bytecode (always one access/cycle) and the `RamHammingWeight` opening for RAM
  (RAM chunks share one access mask → one shared HW). `cache_openings` uses `append_dense` (the
  framework has no `append_sparse`).
- **DEFERRED — `advice`** (`claim_reductions/advice.rs`, documented in `claim_reductions/mod.rs`): the
  multi-phase (cycle + address `ReductionPhase`) advice reduction. Only exercised when advice
  polynomials are present; the e2e gate programs (`muldiv`, `fibonacci`) use **no advice**, so it is
  deferred until the advice e2e path is wired. **Must land before any advice-using guest is proved.**

After P5, **every `SumcheckInstance` the binary driver needs is ported and green** — the remaining
work is wiring + witness-gen, not new crypto.

---

## 5. M8 binary Spartan stage + driver — DONE & tested — **the load-bearing design choice**

### THE DECISION: binary Spartan, NOT univariate-skip Spartan (uni-skip DEFERRED to task #6)

jolt-core's Spartan uses a **univariate-skip** outer sumcheck (collapses the constraint rounds) +
`R1CSEval` (matrix→`z` reduction). The M8 e2e instead uses a **binary** Spartan:

- **Outer** (`spartan/outer.rs`, Phase 2): the plain degree-3 zero-check `0 = Σ_x eq(τ,x)·(Az·Bz−Cz)`.
- **Inner** (`spartan/inner.rs`, 346, **new**): reduces the outer's `Az/Bz/Cz(r_x)` to a **single
  witness opening `z(r_y)`** via the workspace **`jolt_r1cs::R1csKey`** (`combined_row(r_x,ρ)∘z`,
  degree-2 product sumcheck, verified by `evaluate_matrix_mles`). This is the **binary R1CSEval analog**.

**Rationale (memory `m8-opt-e-faithful-port`, user-approved):** the workspace `jolt_r1cs::R1csKey` +
its verifier are **binary** — so binary Spartan is the **reuse-the-workspace, correctness-first** path
that reaches the **equivalence gate** soonest. The gate is *witness-level* (matching opening claims),
and it passes with binary Spartan. The **faithful uni-skip Spartan (OPT-E, the real proving-time win)
is deferred to task #6** — its foundations (§2) are built + tested and waiting. The user explicitly
chose this ordering: *"to target equivalence the soonest lets defer it for later and give me a working
e2e implementation first."* The deferral is prominently documented in `spartan/mod.rs`.

### `src/zkvm/spartan/stage.rs` (253) — the Spartan stage
`prove_spartan`/`verify_spartan`: draw `τ` → outer zero-check → `Az/Bz/Cz(r_x)`; draw `ρ` → inner
reduction → `z(r_y)`. `SpartanProof{outer, az_rx, bz_rx, cz_rx, inner, z_ry}`. `z(r_y)` is the single
committed-witness opening the stage-8 WHIR open will discharge. New accumulator variants:
`SumcheckId::SpartanInner`, `VirtualPolynomial::SpartanWitnessZ`.

### `src/zkvm/driver.rs` (194) — the multi-stage spine
`prove_binary`/`verify_binary` wires the **Spartan stage + the booleanity stage** onto **one shared
transcript + opening accumulator** — the template every remaining stage follows (construct instances →
`framework::sumcheck` prove/verify → thread the accumulator). Round-trips on a real `MockCycle` trace;
a tampered `R1csAux` opening is rejected. This is the proven 2-stage spine; §9 lists what attaches to it.

---

## 6. Conventions added/changed since Phase 1/2

The §7 conventions of the Phase 1/2 guide still hold. Phase 3 adds:

1. **LSB-first logup math vs MSB-first everything-else** — the §1 bit-ordering bridge, local to
   `logup/driver.rs`. Internal GKR is LSB-first; the accumulator/read-raf is `BIG_ENDIAN`/MSB-first.
2. **`normalize_opening_point` overrides** — `ram_ra` and `hamming_weight` reductions prepend the
   address point (`r_address ‖ r_cycle`), unlike the default (`reverse(challenges)`).
3. **Batched-sumcheck `α^j` powers** — `prove_batched` uses running `α^j`, *not* jolt-core's
   `challenge_vector`, to match the workspace `BatchedSumcheckVerifier`. (Reductions/stages so far use
   single `prove`/`verify`; `prove_batched` is for the M8 stage batching.)
4. **The interim binary-Spartan seed pattern** — stages that read input openings binary Spartan
   doesn't emit are seeded directly from the witness (see §7 fork 2).

---

## 7. Design choices & OPEN FORKS — the soundness-relevant decisions

Three forks are **genuine design decisions** flagged by the design map. Two are *resolved/deferred by
explicit user choice*; one (#1) **needs a decision before stage-8 (P9)** and is the one to think about.

### Fork 1 — **Transcript type (P2) — STILL OPEN, needed before P9** ⚠️
Every framework sumcheck, the logup driver, every stage, and the driver are generic over
`jolt_transcript::Transcript<Challenge=F>` (run with `Blake2bTranscript`). But **stage-8 WHIR**
(`jolt-whir/src/scheme.rs`) requires the **spongefish `whir::ProverTranscript`**, which **by design
cannot** implement `jolt_transcript::Transcript` (non-`Clone`, non-`'static` duplex). The two
Fiat-Shamir streams are incompatible. Two viable designs:
- **(A)** retarget the whole driver+framework to drive the spongefish `ProverTranscript` for *all*
  Fiat-Shamir (one stream; matches the orchestration map; broad refactor).
- **(B)** keep Blake2b for stages 1–7 and spongefish only for stage 8, explicitly **absorbing the
  stage-7 transcript digest into the sponge** as a cross-binding (two streams; the cross-bind point +
  domain separation is itself a soundness decision).

This choice changes the `prove()`/`verify()` signature and **must be made before P9 (stage-8 WHIR
open).** It does **not** block P3–P8.

### Fork 2 — **Binary-Spartan seed — RESOLVED (interim seeding), full binding deferred to uni-skip**
Several stages get their `input_claim` seed from an opening **binary Spartan does not emit**
(`RamRafEvaluation` reads `RamAddress`@SpartanOuter; register RW reads `Rd/Rs1/Rs2Value`@SpartanOuter
via `RegistersClaimReduction`). In jolt-core these come from the uni-skip outer (which binds the
`z`-inputs at `r_spartan`); our binary outer only emits `Az/Bz/Cz(r_x)`, and the inner emits `z(r_y)`
at a *different* point. **Interim resolution:** the driver supplies these seeds **directly, recomputed
from the materialized witness** (their MLE at `r_spartan`), each documented as recomputable-from-public.
**This is not fully sound until bound** — the real binding arrives with the **uni-skip Spartan (task
#6)**. For the *equivalence gate* (witness-level claim matching) the interim seeding produces the
correct claims. *(User chose this ordering.)*

### Fork 3 — **Stage-8 `Inc` commit layout — DEFERRED to P9**
The committed object for `RdInc`/`RamInc` is **two signed base limbs** `(lo, hi)`, but the witness
materializer recomposes them into a single `F` MLE (to keep `Val = Σ inc·wa·LT` degree-3). Whether
stage 8 commits/opens the **two per-limb columns** (with an `eq(r_addr,0)` Lagrange embedding, as
jolt-core does) or the **recomposed virtual column** is an unresolved layout decision flagged in
`witness.rs`, deferred to P9. It affects what columns enter the WHIR batch and the eval claims.

---

## 8. What's DONE (Phase 3 scorecard)

| Area | Status | Tested |
|---|---|---|
| M7 LogUp\*-GKR core (pushforward, circuit, layer, gkr) | ✅ done | ✅ round-trips + corruption |
| M7 Option C per-chunk pushforward (driver) | ✅ done | ✅ real read-raf, distinct chunk pts |
| M8 framework uni-skip + batched sumcheck | ✅ built | ✅ (consumer deferred to task #6) |
| M8 `CommittedWitness` (ra_dense + Inc) | ✅ done | ✅ synthetic layout |
| M8 limbed `R1csWitness` (z + Az/Bz/Cz) | ✅ done | ✅ vs `R1csKey` factorization |
| M8 register witness-gen (`K·T` matrices) | ✅ done | ✅ vs `RegistersReadWriteChecking` |
| M8 binary Spartan stage (outer + inner) | ✅ done | ✅ real trace round-trip |
| M8 binary multi-stage driver (Spartan + booleanity) | ✅ done | ✅ real trace round-trip |
| M8/P5 reductions: registers, ram_ra, instruction_lookups, hamming_weight | ✅ done | ✅ round-trip + tamper |
| M8/P5 `advice` reduction | ⛔ deferred | — (no advice in gate programs) |
| **M8/P3 `ram_witness` (`K·T` matrices + `val_final`)** | ✅ done | ✅ vs `RamReadWriteChecking` |
| **M8/P4 registers stage pipeline** (`registers/stage.rs`) | ✅ done | ✅ real trace round-trip + tamper |
| **M8/P3 RAM stage pipeline** (`ram/stage.rs`, batched-aligned) | ✅ done | ✅ real trace round-trip + tamper |
| **M8/P3+P4 combined memory stage** (`zkvm/memory.rs`) | ✅ **done** | ✅ real trace (RAM+regs) round-trip + tamper |

**P3 + P4 are COMPLETE.** `zkvm/memory.rs::{prove_memory, verify_memory}` composes RAM + registers
+ `RamRaClaimReduction` + `IncClaimReduction` on one shared transcript/accumulator. Key insight
realized: the **batched-aligned RAM schedule** (RW+RAF+OutputCheck via `prove_batched`, address
rounds in lockstep) gives the shared `r_address` that `RamValCheck` *and* `RamRaClaimReduction` need;
running RAM+registers on one accumulator gives `IncClaimReduction` its four `Inc` openings. Framework
change: `verify_batched` now returns the `α^j` coeffs (for the combined output-claim check).

---

## 9. What's LEFT — the ordered P-piece plan to the e2e gate

This is the plan from the design map (Understand→Design workflow). **P3, P4, P5 are DONE.** Remaining,
in dependency order:

| Piece | Scope | Depends on | Status |
|---|---|---|---|
| ~~**P3 — RAM stages**~~ | `ram_witness` + batched-aligned `RamReadWriteChecking + RamRafEvaluation + RamOutputCheck` then `RamValCheck` | — | ✅ **DONE** (`ram/witness.rs`, `ram/stage.rs`) |
| ~~**P4 — registers stages**~~ | `register_witness` + `RegistersClaimReduction → RW → ValEvaluation`; + the combined `memory.rs` wiring `RamRaClaimReduction` + `IncClaimReduction` | — | ✅ **DONE** (`registers/{witness,stage}.rs`, `zkvm/memory.rs`) |
| **P6 — read-raf as `SumcheckInstance`** | the `instruction_lookups`/`bytecode` `read_raf_checking` files are currently **param-only** (`OneHotReadRafParams`); build the params from materialized `ra_chunks` and run the shared `OneHotReadRaf` in the driver, caching the `ra_i(r_chunk_i, r_cycle)` openings | one-hot `ra` columns from `CommittedWitness` | pending |
| **P7 — slot M7 pushforward into the driver** | bridge P6's cached read-raf openings into `prove_family_per_chunk`/`verify_family_per_chunk` between read-raf and stage 8 (per family: `r_cycle`, `ChunkPushforward` from `ra_dense` + `r_col` slices + claims, `base_index = family range start`) | P6, `CommittedWitness` | pending |
| **P8 — stage-8 opening inventory** | pure data-shaping: walk the final accumulator, dedup by `(committed poly, point)` → canonical opening, group by committed length, build the form-major `M×N` evals matrix `WhirScheme::open_batch` consumes (transcript-free, unit-testable) | P7 | pending |
| **P9 — stage-8 WHIR commit + batched open** | commit every base-field column (`WhirScheme::commit`), `open_batch` once per size class; verifier `verify_batch`. **Requires the fork-1 transcript decision.** Materialize the per-limb `Inc` columns here (fork 3). | P8, **fork 1** | pending |
| **P10 — full `prove()`/`verify()` e2e** | replace the narrow `prove_binary` signature with the full driver taking a `CycleRow` trace (or pre-extracted inputs): P1 extraction → all wired stages → stage-8 open → one proof. `muldiv_e2e_goldilocks` + `fibonacci_e2e_goldilocks`. | P9 | pending |
| **P11 — `jolt-equivalence` cross-check** | drive the same `muldiv` fixture through both the goldilocks accumulator and the jolt-core BN254 oracle; assert shared `(committed-poly, sumcheck-id)` opening claims match under the field/challenge collapse (jolt-core stays read-only) | P10 | pending |
| **task #6 — uni-skip Spartan (P, the perf win)** | the faithful jolt-core univariate-skip Spartan (outer + R1CSEval grouping) replacing binary Spartan; **also closes fork 2's interim seeding** | foundations (§2) ✅ | deferred (post-gate) |
| **`advice` reduction** | the deferred multi-phase advice claim-reduction (§4) | — | deferred (advice guests only) |
| **OPT-B…E** (Phase 1/2 §8) | split-LT, sparse `ReadWriteMatrix`, prefix/suffix, compact base-field MLE (`base×ext` 2.3×), full-d one-hot | various | deferred (perf) |

---

## 10. What's left for E2E integration specifically (the critical path)

To get `muldiv`/`fibonacci` prove→verify green (the gate), the *integration* work beyond the
per-stage math is:

1. **The trace bridge (P10):** `jolt_trace::extract_trace::<C,F>` for the field-agnostic
   `CycleInput[]` → `CommittedWitness`, **plus** the crate's own `build_limbed_z` for `R1csWitness`
   (NOT extract_trace's BN254 flat witness — §3 KEY finding). Both derive `log_t` from the actual
   trace length (they agree).
   - **Open question:** is a guest-compiled `muldiv`/`fibonacci` ELF reachable from this crate's test
     deps (the way `jolt-equivalence` does via `core_*_commitment_fixture`), or does P10 need a
     CLI/host build step (`cargo install --path .`)? The crate depends on `jolt-trace`/`jolt-riscv`,
     but it's unclear if the guest programs are pre-built fixtures or need compilation. **This gates
     how P10's e2e test acquires its trace.**
2. **The remaining `K·T` witness-gen:** `register_witness` ✅; still need **`ram_witness`** (RAM
   address remap via `MemoryLayout` + value/inc simulation) and the **one-hot `ra` chunk columns**
   (instruction/bytecode/ram) for read-raf (P6) + their `ra_dense` (already in `CommittedWitness`).
   - **Open question:** are the full-dense `K·T` matrices tractable at `muldiv`'s ~2¹⁶ cycles, or does
     P3/P4 need the deferred sparse two-phase path (OPT-C)? The current modules use the dense (M5)
     convention; this may need the sparse path for the real trace size.
3. **`cycle_to_z` op coverage:** currently covers no-op / ADD / SUB / MUL / loads / default. The real
   `muldiv` trace exercises **advice / virtual-sequence** ops — `cycle_to_z` must cover those (or the
   e2e will fail witness satisfaction). Validated only against the real trace at P10.
4. **The transcript decision (fork 1)** — must be resolved before P9 wires stage 8.
5. **The interim binary-Spartan seeds (fork 2)** — must be supplied + documented for each stage that
   needs them (P3/P4), and revisited when uni-skip Spartan (task #6) lands.
6. **Stage-8 `Inc` layout (fork 3)** — decided at P9.

**Definition of done (unchanged from Phase 1/2 §9):** Goldilocks+WHIR prove→verify green on `muldiv` &
`fibonacci`; `jolt-equivalence` claim-level match vs jolt-core; **BN254 `muldiv` still green**
(`host` + `host,zk`); clippy + fmt clean.

---

## 11. Commit-by-commit review reference (Phase 3)

Build order (oldest first = review order). Each is one self-contained, individually-tested,
individually-**compiling** commit (verified). Notation as in Phase 1/2 §10.

**`bc0331a02` — M7 pushforward prep + §4.5.2 reduction (piece a).** `logup/{mod,pushforward}.rs`.
LSB-first eq/MLE utilities; `Family`/`PushforwardData`/`VerifierView`; the §4.5.2 absorb→squeeze→combine
reduction + eq.5 check (`GkrError::MainIdentity`). *Review:* `prepare_family` vs `prepare_family_verifier`
symmetry; the `MainIdentity` check is the consistency anchor between `P^F` and `ra_dense`.

**`76f379d7b` — M7 fan-in-2 GKR circuit + per-layer SumcheckInstance (piece b).** `logup/{circuit,layer}.rs`.
The fractional-add pyramid `(n,d),(n',d')→(n·d'+n'·d, d·d')`; `GkrLayer` degree-3 with Gruen split-eq
(LSB-first point reversed for MSB-first binding). *Review:* `root_is_fractional_sum`,
`structural_eq_recompute_is_order_consistent`.

**`756a59ca5` — M7 per-family GKR prover+verifier (pieces c+d).** `logup/gkr.rs`. Two circuits A/B → 3
openings; verifier root-histogram + per-layer + leaf-structural checks. *Review:* the A-leaf numerator =
`eq(r_M_row, r*_A)` and B-leaf denominator = `α − idx_mle(r*_B)` structural checks (`GkrError::LeafStructural`).

**`1b92c5b6e` / `c43b4038a` — M7 per-family driver + A-leaf structural-check pin.** `logup/driver.rs`.
`prove_family`/`verify_family` consuming the read-raf hand-off; the structural-check formula test.

**`b17e6bf1a` / `2bacf6b5e` / `52cdd4484` / `458d25fe3` — M8/P5 the 4 reductions.** §4. Each: synthetic
upstream openings → reduce → round-trip + tamper. *Review:* `input_claim` (LHS from upstream) vs
`expected_output_claim` (RHS from this reduction's cached openings + recomputed eq) — the same identity.
For `ram_ra`/`hamming_weight` check the `normalize_opening_point` override (`r_address ‖ r_cycle`).

**`e2b4f4978` — M8 framework uni-skip + batched foundations.** §2. *Review:* `lagrange.rs` integer-window
kernel (field-agnostic); `prove_batched` uses `α^j` (NOT `challenge_vector`). Consumer (uni-skip Spartan)
deferred.

**`bd25a4588` — M7 Option C per-chunk pushforward.** §1 Option C. *Review:* `prove_family_per_chunk` builds
one `Family{log_d:0}` per chunk at its own `rev(r_col)`; `readraf_per_chunk_option_c_round_trip` asserts
distinct chunk points. The `rev` bit-ordering bridge.

**`3448f67a4` — M8 `CommittedWitness`.** §3. *Review:* global chunk indexing = the `RaDense`/`Pushforward`
key + Option C `base_index`; `ra_dense` family ranges; Inc recompose + zero-pad.

**`9eddcbd9f` — M8 limbed R1CS witness.** §3 KEY finding. *Review:* `cycle_to_z` limbing vs the BN254
`extract_trace` (must NOT reuse the latter); `materialize` cycle-major layout vs `R1csKey::evaluate_sparse_matvec`
(the `honest_multicycle_witness_satisfies_and_matches_r1cskey` test). Adds `jolt-trace`/`jolt-riscv` deps + the
`MockCycle` test helper.

**`bae099fa7` — M8 binary Spartan stage.** §5. *Review:* `SpartanInner` reduction via `R1csKey`
(`combined_row(r_x)∘z` → `M(r_x,r_y)·z(r_y)` via `evaluate_matrix_mles`); the binary-vs-uniskip decision
(uni-skip deferred). `tampered_az_rejected`.

**`980c96f36` — M8 binary multi-stage driver.** §5. *Review:* `prove_binary`/`verify_binary` thread one
shared transcript+accumulator across Spartan + booleanity; the template for the remaining stages.
`tampered_aux_rejected`.

**`cc535697d` — M8/P4 register witness-gen.** §3. *Review:* the read-before-write register-file simulation
(`val`/read-values/`inc` from tracked state); validated by feeding the real `K·T` matrices into
`RegistersReadWriteChecking` (`register_witness_satisfies_read_write_checking`).

**`6156a73ad` — M8/P3 RAM witness-gen.** §3 analog. *Review:* RAM simulation (`ra`/`val`/`inc` +
`val_final`); validated vs `RamReadWriteChecking`. Adds `MockCycle::with_ram`.

**`812eb801e` — M8/P4 registers stage pipeline.** `registers/stage.rs`. *Review:* interim Spartan-outer
seeding (fork 2, seeds carried in proof); mid-proof `wa(r_address,·)` materialization after RW yields
`r_address`; `ClaimReduction → RW → ValEvaluation` round-trips on a real trace.

**`756c28aa8` — M8/P3 RAM stage pipeline.** `ram/stage.rs`. *Review (the alignment crux):* RW+RAF+OutputCheck
via `prove_batched` bind address rounds in lockstep (`offset 0` vs `log_t`) → shared `r_address =
reverse(challenges[log_t..])`; `RamValCheck` then consumes `RamVal`(RW) + `RamValFinal`(OC) at it.
Framework change: `verify_batched` returns the `α^j` coeffs.

**`82d975986` — M8/P3+P4 combined memory stage.** `zkvm/memory.rs`. *Review:* composes RAM + registers +
`RamRaClaimReduction` (3 `RamRa` → 1, precondition = the RAM-stage alignment) + `IncClaimReduction`
(4 `Inc` → 1, precondition = RAM+regs on one accumulator) → `memory_stage_round_trip` on a real
RAM+register trace. **This is where P3+P4 land.**

---

## 12. The one-paragraph "where are we" for a new reviewer

All sumcheck **math** (leaf checks, Spartan outer/inner/shift/instruction-input, booleanity, all 5
needed claim-reductions, the full M7 LogUp\*-GKR with Option C) is **ported and unit-tested** — 102
green. **The entire memory-checking subsystem (P3 + P4) is now built and composed:** `zkvm/memory.rs`
runs RAM (batched-aligned RW+RAF+OutputCheck + ValCheck) + registers (ClaimReduction+RW+ValEval) +
the `RamRaClaimReduction`/`IncClaimReduction` cross-cutting reductions on one shared
transcript/accumulator, round-tripping on a real trace. The remaining pipeline pieces are **P6**
(give instruction/bytecode read-raf a `SumcheckInstance` from the one-hot `ra` columns), **P7** (slot
the M7 per-chunk pushforward in), and **P8/P9** (the stage-8 WHIR batched open) — then **P10** wires
everything (Spartan stage from `driver.rs` + the memory stage + read-raf + M7 + stage-8) into the full
`prove()`/`verify()` on a guest-compiled trace, and **P11** is the `jolt-equivalence` cross-check.
The two things that are *not* just "more wiring": the **transcript fork** (fork 1, needs a decision
before stage 8 / P9) and the **uni-skip Spartan** (deferred perf, task #6, which also makes fork 2's
interim seeding fully sound). Remaining witness-gen for the e2e: the one-hot `ra` chunk columns (P6)
and `cycle_to_z` coverage of advice/virtual-sequence ops (§10). The gate is reached when
`muldiv`/`fibonacci` prove→verify green and `jolt-equivalence` matches the BN254 oracle, with BN254
`muldiv` still green.

## 13. Next-session starting point

P3+P4 done. The next dependency-ordered piece is **P6 — read-raf as a `SumcheckInstance`**: the
`instruction_lookups`/`bytecode` `read_raf_checking` modules are currently param-only
(`OneHotReadRafParams`, no `SumcheckInstance` impl); build the params from the materialized one-hot
`ra` chunk columns (a new witness-gen analogous to `register_witness`/`ram_witness`, producing the
`K_chunk×T` one-hot `ra_i` columns from `CommittedWitness.ra_dense` indices) and run the shared
`OneHotReadRaf` in a `read_raf` stage, caching the `ra_i(r_chunk_i, r_cycle)` openings — which are
exactly the M7 per-chunk pushforward inputs (P7). The composable-stage pattern (`*/stage.rs` →
`prove_*`/`verify_*` on a shared transcript/accumulator, openings carried in the proof) is now
well-established by `spartan/stage.rs`, `registers/stage.rs`, `ram/stage.rs`, and `memory.rs` — P6/P7
follow it. **Open before P9:** resolve fork 1 (transcript: single spongefish vs dual + cross-bind).
