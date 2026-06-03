# Jolt × Goldilocks × WHIR — As-Built Implementation Reference

**Status: 2026-06-03.** This is the single authoritative description of the hand-written
**Goldilocks + WHIR, non-ZK** Jolt prover/verifier in `crates/jolt-prover-goldilocks`. It
**supersedes and consolidates** the historical phase docs (`PHASE1_PHASE2_REVIEW_GUIDE.md`,
`PHASE2_HANDOFF.md`, `PHASE3_REVIEW_GUIDE.md`, `LIMBED_R1CS.md`, and the `PHASE{3,4,5}_NEXT_SESSION_PROMPT.md`
handoffs). The intended-design rationale still lives in the root `JOLT_GOLDILOCKS_DESIGN.md` /
`JOLT_SMALLFIELD_WHIR_MIGRATION.md` (each now carries an "as-built corrections" header pointing here);
this doc is the truth about **what exists and runs**. The forward plan for the one large remaining
piece (instruction lookups) is `PHASE6_NEXT_SESSION_PROMPT.md`.

---

## 1. Goal, and where we are

**Goal:** a from-scratch Jolt prover/verifier over the **Goldilocks** base field (`p = 2⁶⁴−2³²+1`)
with **Fp3** sumcheck challenges, a **hash-based PCS (WHIR)** instead of BN254/Dory, a single
**spongefish** Fiat-Shamir transcript, and **base-field-limb** committed columns. `jolt-core` (BN254 +
Dory + BlindFold ZK) is the **read-only parity oracle** — we port math *from* it and gate *against* it,
never modifying it. This crate is **NON-ZK** (no BlindFold, no WHIR-zk).

**Where we are:** a full **bytecode-first** `prove()`/`verify()` runs end-to-end on a **real muldiv
RISC-V trace** and is gated against jolt-core's geometry. It proves: R1CS satisfaction (Spartan),
memory consistency (RAM + registers), booleanity of the limb carry/sign columns, bytecode consistency
(read-raf + LogUp\*-GKR pushforward), and discharges the committed columns via WHIR opens (`R1csAux`,
`Inc` limbs, bytecode `RaDense`, `Pushforward`). **Instruction lookups are not yet wired** (the one
large remaining functional piece — see §12), and several soundness bindings are interim (§11).

### Scorecard

| Area | State |
|---|---|
| Field / transcript / WHIR PCS adapter | ✅ done (`jolt_field::goldilocks`, `jolt_whir`) |
| Framework (transcript trait, sumcheck, accumulator, poly, stage-8) | ✅ done |
| Limbed RV64 R1CS + witness (`z`/`Az`/`Bz`/`Cz`) | ✅ done |
| Real-trace witness acquisition (RAM remap, pcs, public columns) | ✅ done (M0) |
| Binary Spartan (outer + inner) | ✅ done (interim Fork-2 seeding) |
| Memory stage (RAM + registers + RamRa/Inc reductions) | ✅ done (zero-init RAM, interim) |
| Booleanity | ✅ done |
| Sparse read-raf (address-first two-phase) | ✅ done |
| Bytecode read-raf (real `Val_s`, 4 stages) | ✅ done (M3b) |
| M7 LogUp\*-GKR pushforward (Option C, per-chunk) | ✅ done |
| Stage-8 WHIR open (R1csAux / Inc / RaDense / Pushforward) | ✅ done (M2/M3a/M3b-3) |
| Range-check (R-core membership) | ✅ standalone; ⏸ not integrated into e2e |
| e2e on real muldiv + jolt-core geometry parity gate | ✅ done (M0–M4) |
| **Instruction lookups (prefix/suffix)** | ✅ done (P3b: read-raf at LOG_K=128 + M7 pushforward + stage-8 open, wired into e2e; interim Fork-2 `r_reduction`) |
| Cycle→table dispatch (jolt-core-free) | ✅ done (P3b-0: `jolt_lookup_tables::instruction_lookup_table_index`, parity-gated vs jolt-core) |
| Uni-skip Spartan (sound stage binding) | ⏳ deferred (Fork 2) |
| RAM real initial-state + dense-`RamRa` PCS discharge | ⏳ deferred |
| Stage-5 lookup `Val_s` (registers val-eval + lookup membership) | ⏳ deferred |
| ZK (BlindFold / WHIR-zk) | ❌ out of scope (non-ZK crate) |

---

## 2. Architecture

`F = jolt_field::goldilocks::GoldilocksFp3` (cubic extension, ~192-bit), `Base = Goldilocks`. The crate
is `#![cfg(feature = "goldilocks")]` (the default build is empty so the WHIR graph isn't pulled into
non-goldilocks workspace builds).

**Dependencies (all field-generic, workspace crates — NOT jolt-core):**
- `jolt-field` — the `Field` trait + `GoldilocksFp3`/`Goldilocks` + `decompose` (`i128_to_signed_limbs`,
  `signed_limbs_recompose`).
- `jolt-whir` — `WhirScheme` (commit/open/verify), `ProverTranscript`/`VerifierTranscript` (spongefish),
  `WhirConfig`/`WhirCommitment`/`WhirHint`/`WhirError`. The hash-based PCS over base-Goldilocks.
- `jolt-poly` — field-generic poly machinery: `MultilinearPolynomial` (note: the crate ALSO has its own
  `framework::poly::MultilinearPolynomial`), `EqPolynomial`, `GruenSplitEqPolynomial`, `BindingOrder`,
  `UnivariatePoly`, `ExpandingTable`, `IdentityPolynomial`, `one_hot`, `lagrange`.
- `jolt-r1cs` — `R1csKey`, `ConstraintMatrices`, `SparseRow` (binary Spartan inner-reduction toolkit).
- `jolt-trace` — `Cycle`/`Instruction`, `CycleRow` trait, `BytecodePreprocessing`, `extract_trace`,
  `instruction_{circuit,instruction}_flags`, the jolt-witness `CommitmentTraceSources`.
- `jolt-witness` — `commitment_trace_sources`, `goldilocks::{GoldilocksLayout, FamilyLayout}`,
  `one_hot_chunk_indices`, `CommitmentTraceSources`.
- `jolt-lookup-tables` — field-generic lookup-table layer (`LookupTableKind`, `combine`, `evaluate_mle`,
  `Prefixes`/`Suffixes`, `LookupBits`, `interleave`/`uninterleave`, the trace bridge). Used by IL-1+;
  the linchpin reuse for instruction lookups.
- `jolt-riscv` — `CircuitFlagSet`/`InstructionFlagSet`/`CircuitFlags`/`InstructionFlags`.

**Module map (`crates/jolt-prover-goldilocks/src/`):**
- `field.rs` — `F`/`Base` aliases + WHIR/transcript re-exports.
- `r1cs/` — the limbed RV64 R1CS (`rv64_limbed.rs`, `mul.rs`, `signed_value.rs`); `rv64_limbed_constraints`.
- `framework/` — `transcript.rs` (the FS trait seam), `sumcheck.rs` (`prove`/`verify`/`prove_batched` +
  `SumcheckInstance`), `accumulator.rs` (`Openings`, key enums), `poly.rs` (`MultilinearPolynomial`),
  `lagrange.rs`/`multiquadratic.rs`/`univariate_skip.rs` (staged for uni-skip Spartan — built, unused by
  the binary path), `stage8.rs` (opening inventory), `stage8_open.rs` (WHIR open).
- `zkvm/` — `driver.rs`, `e2e.rs`, `real_trace.rs`, `witness.rs`, `r1cs_witness.rs`, `stage8_columns.rs`,
  `spartan/`, `memory.rs` + `ram/` + `registers/`, `booleanity.rs`, `shout_read_raf.rs`,
  `bytecode/read_raf_checking.rs`, `instruction_lookups/`, `logup/` (the GKR), `claim_reductions/`,
  `range_check.rs`.

---

## 3. Field, transcript, WHIR PCS

- **Field:** `GoldilocksFp3` from the arkworks fork (`../algebra`). `Base = Goldilocks`. In generic
  `F: Field` code `F::zero()`/`F::one()` work; on the **concrete** `GoldilocksFp3` in tests they do NOT
  resolve (they come from `num_traits`, not `Field`) — use `F::from_u64(0/1)`. The hot-path field
  optimizations described in the design docs (Montgomery-free, deferred-reduction accumulator,
  `mul_by_base`) live in `jolt-field`; correctness-first code uses plain `Field` ops.
- **Transcript:** a single **spongefish** NARG transcript shared across the whole prover + WHIR (per the
  resolved Fork 1 — `a16z/jolt#1455`). The crate is generic over a trait seam in `framework/transcript.rs`:
  `Challenge<F>` (squeeze), `ProverFs<F>: Challenge` (`observe` → writes NARG), `VerifierFs<F>: Challenge`
  (`read_coeffs` → reads NARG). Concrete impls are `jolt_whir::{ProverTranscript, VerifierTranscript}` at
  `F = GoldilocksFp3`. There is NO separate `SumcheckProof` carrier — round polynomials are written into
  the NARG via `sumcheck::write_round_poly`; proof structs carry only the **reduced opening claims** the
  stage-8 PCS discharges. Dory is untouched.
- **WHIR (`jolt_whir::WhirScheme`, stateless):** `config(size)` (power-of-two column length, transparent
  setup), `commit(transcript, config, &[Base]) -> WhirHint` (absorbs Merkle root + OOD), `open(transcript,
  config, column, hint, point: &[F], eval: F)`, `receive_commitment`, `verify(transcript, config,
  commitment, point, eval)`. Also `evaluate(config, column, point) -> F` (the base-column MLE at an Fp3
  point; used to fabricate claims in tests — the real prover gets claims from the accumulator).
  - **GOTCHA — RS interleaving minimum:** WHIR panics if the committed column length is below its
    interleaving floor; `2³` is too small. Use `log_t ≥ 4` (real traces are `2⁹⁺`).
  - **GOTCHA — zero columns:** WHIR's `open` inverts the claimed evaluation, so it **cannot open an
    identically-zero polynomial** (eval 0 → division by zero). The stage-8 open handles this (§10.7).

---

## 4. The framework layer

- **`SumcheckInstance<F>` (`framework/sumcheck.rs`):** the trait every leaf sumcheck implements —
  `num_rounds`, `degree`, `input_claim(&dyn OpeningAccumulator)`, `compute_message`, `bind`,
  `cache_openings(&mut Openings, &[F])`, `expected_output_claim(&dyn OpeningAccumulator, &[F])`,
  `normalize_opening_point` (default → BIG_ENDIAN `reverse(challenges)`). Drivers:
  `prove(instance, acc, transcript) -> Vec<F>` (loops compute_message → write_round_poly → challenge →
  bind, then cache_openings) and `verify(claim, transcript) -> EvaluationClaim`. `prove_batched`/
  `verify_batched` run several instances front-loaded so their address rounds bind in lockstep (used by
  the RAM stage). Binding is **LowToHigh**; opening points are **BIG_ENDIAN** (reversed challenges).
- **`Openings<F>` (`framework/accumulator.rs`):** a single shared `HashMap<(PolynomialId, SumcheckId),
  (OpeningPoint<BIG_ENDIAN,F>, F)>` + `log_t`. The same type is used by prover and verifier — the prover
  stores claims it computed, the verifier stores claims it read from the proof. `append_dense`/
  `append_virtual` insert; `get_committed_polynomial_opening`/`get_virtual_polynomial_opening` read (and
  **panic** if missing — verifiers MUST append an opening before any `expected_output_claim` reads it).
  Key enums: `CommittedPolynomial` (`RaDense`, `R1csAux`, `RamInc`, `RdInc`, `RamRa`, `InstructionRa`,
  `BytecodeRa`, `Pushforward`), `VirtualPolynomial` (Spartan Az/Bz/Cz, Ram*/Rd*/Rs* values, LookupOutput,
  LeftLookupOperand, …), `SumcheckId` (SpartanOuter/Inner, Booleanity, Ram*, IncClaimReduction,
  PushforwardGkr/PushforwardReduction, BytecodeReadRaf, InstructionReadRaf, …).
- **`MultilinearPolynomial<F>` (`framework/poly.rs`):** the dense MLE the stages bind. `sumcheck_evals_array
  ::<DEGREE>(index, order)` returns evals at points `0,1,…,DEGREE-1` (NOTE: NOT jolt-core's `0,2,3,…`
  convention — the line through `(0,e0),(1,e1)`). `bind_parallel(r, order)`, `final_sumcheck_claim`,
  `From<Vec<F>>`.
- **Uni-skip foundations (`lagrange.rs`, `multiquadratic.rs`, `univariate_skip.rs`):** built + tested but
  **unused** by the binary Spartan path — staged for the deferred uni-skip Spartan (Fork 2 closure).

---

## 5. The limbed RV64 R1CS (`r1cs/`)

The BN254 `jolt-r1cs` constraint set can't be reused under Goldilocks: every u64-valued variable aliases
mod `p` (`from_u64(v) ≡ from_u64(v−p)` for `v ≥ p`), so a single small-field element is unsound. A *new*
limbed set lives in `r1cs/rv64_limbed.rs` (`rv64_limbed_constraints::<F>() -> ConstraintMatrices<F>`),
adapted from `rv64.rs`. Three limb conventions (`F` embeds via `from_u64`/`from_i128`):
1. **Unsigned u64 → 2 unsigned 32-bit limbs** `(lo,hi)`, value `lo + 2³²·hi`. Linear recompose; range-check `< 2³²`.
2. **Signed-used-linearly → signed 2-limb** `(lo,hi)` (`i128_to_signed_limbs`; `hi` carries sign). Linear recompose.
3. **MUL operand/product → sign bit + unsigned magnitude limbs** (operand: sign+2; product: sign+4). Schoolbook on clean unsigned limbs; sign applied to the product.

**Locked correction:** multi-limb **equality/arithmetic** must be **limb-by-limb** (with `2⁻³²` carries),
NOT a single linear field recompose — because `2⁶⁴ ≡ 2³²−1` mod `p`, two u64s differing by `p` share a
recompose, so `recompose(a)=recompose(b)` does NOT force `a=b`. Only genuinely-small values (`< p`:
PC family, flags/booleans) use plain recompose. `LEFT_INSTRUCTION_INPUT` is always unsigned (2-limb);
`RIGHT_INSTRUCTION_INPUT` is sign+magnitude for the MUL schoolbook plus a derived signed `RIGHT_VAL` for
the linear eq-constraints (one degree-2 derivation). The MUL product is the 4-limb schoolbook (`mul.rs`)
with `2⁻³²` virtual carries; outer sumcheck stays **degree 2**, val-evaluation stays **degree 3**.
Soundness depends on the limbs being range-checked (the R / booleanity machinery, §10.6).

**Actual muldiv geometry (`R1csWitness`):** `num_vars` (unpadded) → `num_vars_padded = 128`,
`num_constraints` → `num_cons_padded = 64`. The carry/sign aux columns (`boolean_aux_columns()`, 8
columns) are the booleanity inputs + committed `R1csAux`.

---

## 6. Witness layer

- **`R1csWitness<F>` (`zkvm/r1cs_witness.rs`):** `build_limbed_z::<C: CycleRow, F>(trace, pcs) -> Vec<Vec<F>>`
  (one limbed `z` per cycle via `cycle_to_z`, using the per-cycle expanded PC `pcs[t]`) →
  `R1csWitness::materialize(&per_cycle)` (applies `rv64_limbed_constraints` row-wise to fill `Az/Bz/Cz`;
  pads cycles/vars/cons to powers of two). `is_satisfied()` checks `Cz == Az∘Bz`. `boolean_aux_columns()`
  extracts the 8 carry/sign columns. `cycle_to_z` handles no-op/ADD/SUB/MUL/loads/stores/advice; real
  MUL/virtual-sequence coverage is validated by the real-trace e2e (M0 asserts `is_satisfied`).
- **`CommittedWitness<F>` (`zkvm/witness.rs`):** `build(sources: &CommitmentTraceSources, layout:
  &GoldilocksLayout) -> Self`. Fields: `log_t`, `ra_dense: Vec<RaDenseColumn>` (global-index order:
  instruction, then bytecode, then ram), `instruction_range`/`bytecode_range`/`ram_range`,
  `rd_inc`/`ram_inc` (recomposed signed Inc as one `F` MLE). `RaDenseColumn{family, global_index, log_m,
  indices: Vec<u32>}` — the dense chunk-index column (chunk 0 = MSB; `one_hot_chunk_indices` decomposition,
  None→0, padded to `2^log_t`). `one_hot_ra_column(indices, log_m)` lifts to the address-major one-hot.
- **Memory witnesses:** `ram_witness::<C,F>(trace, ram_k)` (`ram/witness.rs`) and `register_witness::<C,F>
  (trace, register_count)` (`registers/witness.rs`) simulate RAM/registers from the trace: `ra`/`val`
  address-major (`k·T+j`), `inc`/value columns length `T`. **Both now also expose `inc_i128: Vec<i128>`**
  (the exact signed increments the stage's `IncClaimReduction` claims — fed to the stage-8 Inc limb open).
  **RAM is zero-initialised** (tracked state starts at 0; the real initial-memory state is NOT loaded —
  interim, §11).
- **Real-trace assembly (`zkvm/real_trace.rs`):** `assemble_real_witness::<F>(trace, bytecode, ram_lowest,
  ram_k, register_count) -> RealWitness<F>` builds all binary-driver inputs from one `&[tracer::Cycle]`:
  `pcs[t] = bytecode.get_cycle_pc(&trace[t])`; `RemappedCycle` (a `CycleRow` adapter that rewrites ONLY
  `ram_access_address` → `(addr − ram_lowest)/8`, mirroring jolt-core `remap_address`, so `ram_witness`
  indexes the dense remapped space while the R1CS witness uses the raw `rs1+imm` address); `RamPublicColumns`
  with the affine unmap inverse `unmap[k]=ram_lowest+8·k` and **empty I/O** (`val_io=io_mask=0`, which makes
  the output-check a trivial-but-honest zero-check — real program-output binding is deferred). `register_count
  = REGISTER_COUNT = 128`.

---

## 7. Proving stages (pipeline order)

### 7.1 Spartan (`zkvm/spartan/`, binary) — `prove_spartan`/`verify_spartan`
Binary Spartan (NOT uni-skip): the **outer** zero-check (`SpartanOuter`, degree-3, over `Az/Bz/Cz` at
`tau`) then the **inner** reduction (`SpartanInner`, via `jolt_r1cs::R1csKey`) reducing to a single
witness `z(r_y)` opening. Caches `SpartanAz/Bz/Cz @ SpartanOuter`, `SpartanWitnessZ @ SpartanInner`.
`SpartanProof{az_rx,bz_rx,cz_rx,z_ry}`.

### 7.2 Memory (`zkvm/memory.rs` + `ram/` + `registers/`) — `prove_memory`/`verify_memory`
RAM stage = front-loaded batched (`RamReadWriteChecking` + `RamRafEvaluation` + `RamOutputCheck`, sharing
an aligned `r_address`) then sequential `RamValCheck`; registers stage analogous. Then the `RamRaClaimReduction`
(reduces `RamRa` to one committed opening) and `IncClaimReduction` (reduces `RamInc`/`RdInc`). **Fork-2
interim seeding:** the RAM/register stages seed their own `SpartanOuter` virtual openings
(`Ram{Read,Write}Value`/`RamAddress`, `Rd{Write}Value`/`Rs1/Rs2Value`) from the witness MLE at a fresh
`r_spartan`, carried in the proof as `spartan_seeds:[F;3]` and re-seeded identically by the verifier — they
are the *correct* MLEs but not yet *bound* to a sound Spartan output (binding arrives with uni-skip Spartan).

### 7.3 Booleanity (`zkvm/booleanity.rs`)
Degree-3 zero-check `Σ γ^{2i}(b_i² − b_i) = 0` over the 8 boolean aux columns; caches `R1csAux(i) @ Booleanity`
at `ρ` (discharged by the stage-8 open).

### 7.4 Sparse read-raf (`zkvm/shout_read_raf.rs`) — the address-first two-phase machine
`OneHotReadRaf<F, const D, const NE=D+2>`: an `O(K_total + D·T)` sparse read-raf (NOT the dense `K·T`
broadcast). **Address phase** (`round < log_k`) binds per-stage `F_s` (an `O(T)` eq-scatter into a
length-`K_total` table) + `Val_s` (degree 2). **Hand-off** at `round == log_k−1` lifts each sparse
`ra_i(r_k_i, ·)` (length `T`) and captures `Val_s(r_addr)`. **Cycle phase** binds `ra_i` + plain length-`T`
`eq_s` (degree `D+1`; Gruen split-eq deferred). `cache_openings` splits the BIG_ENDIAN point as
`[r_cycle ‖ r_addr]` and caches each chunk `ra_i` at `[r_k_i ‖ r_cycle]` under `ra_family(i)`. **The dense
length-`K_total` `Val_s`/`F_s` tables are why instruction lookups (`K_total=2¹²⁸`) need prefix/suffix (§12).**

### 7.5 Bytecode read-raf (`zkvm/bytecode/read_raf_checking.rs`)
`bytecode_val_polys(&[Instruction], stage_gammas, eq_r_register) -> [Vec<F>;4]` builds the 4 dense per-row
`Val_s` columns (Spartan-outer/product-virtualization/shift/registers; stages 1–4 of jolt-core's
`compute_val_polys`, via the field-generic `jolt_trace::instruction_*_flags` bridge; stage 5 lookup-table
membership deferred). `prove_bytecode_read_raf::<F,T,D,NE>` (real muldiv `D=4`, `NE=6`): draws per-stage
γ-powers + the stage-4 register point + per-stage `r_cycle` in transcript lockstep, pads `Val_s` to
`K_total = 2^(D·log_k_chunk) = 2¹⁶`, runs `prove_read_raf`. **Fork-2 pattern:** the proof carries the 4
`rv_s` seeds (`BytecodeReadRafProof{rv_seeds:[F;4], read_raf}`) because the verifier lacks the witness chunk
indices; the read-raf sumcheck binds them. The 4 interim `rv_key`s are distinct free accumulator slots.

### 7.6 M7 LogUp\*-GKR pushforward (`zkvm/logup/`) — Option C, per-chunk
Discharges the read-raf's cached `ra_i` openings into committed `RaDense` + `Pushforward` openings. Each
family's `d` chunks are discharged by their **own** `log_d=0` pushforward GKR at the chunk's distinct
column point `r_k_i` (Option C — the read-raf opens chunks at distinct points, so no §4.1 row-concatenation
/ shared-column reduction; resolved per the `m7-readraf-shared-point-gap` memory). `prove_read_raf_pushforward`
extracts `(r_col, r_cycle, claim)` per chunk and runs `prove_family_per_chunk`. The eq-weighted pushforward
`P^F` (length `2^log_m`) is now **surfaced** out of `prove_family` (it was dropped before IL-1's prerequisite
work) so the stage-8 open can commit it. **GKR convention fix:** the GKR caches its leaf points (`RaDense`/
`Pushforward`) reversed to **BIG_ENDIAN** (the circuit sumcheck is LSB-first; the framework + WHIR are
BIG_ENDIAN; the claim is unchanged, the structural leaf checks still use the raw LSB-first point).
`GkrProof<F>` is an empty marker (all data in the shared NARG). Keys: `RaDense(base+i)`/`Pushforward(base+i)`
@ `PushforwardGkr`, `Pushforward(base+i)` @ `PushforwardReduction`.

### 7.7 Range-check (`zkvm/range_check.rs`) — R-core, standalone
`prove_range_check`/`verify_range_check`: a limb-membership `{col[j]} ⊆ [0,2^log_m)` proved by reusing the
M7 `prove_family` GKR (the root histogram at the FS α forces the subset). **Built + tested standalone; NOT
yet integrated into the e2e** (the R-integration — committing the z-resident wide limbs as `R1csRangeHalf`
columns + tying them to `z(r_y)` via SpartanInner — is deferred; until then the limbed-R1CS soundness has a
gap, §11).

### 7.8 Stage-8 WHIR open (`framework/stage8.rs` + `stage8_open.rs`)
The final PCS step. `Stage8Inventory::from_accumulator(acc, &canonical_requests(geom))` collects the
committed openings (dedup/alias by `(poly,point)`, size-class grouping, `zero_selector` for address-var
stripping). `prove_stage8`/`verify_stage8` commit every inventory column on the shared transcript (size
class ascending) then open **each at its own point (M=1, per-column)** — NOT `open_batch`, because the
verifier (no columns) can't reproduce the cross-evals; true intra-class RLC batching is a deferred opt.
- **Inc** is opened separately (`prove_inc_open`/`verify_inc_open` + `IncLimbColumns`): `RdInc`/`RamInc`
  committed as 2 signed base limbs each, checked `lo + 2³²·hi == claim` (the `IncClaimReduction` claim).
  Limbs built from the memory-stage `inc_i128` (NOT `sources` — they must equal the polynomial the stage
  claims, under the zero-init RAM model). `Stage8IncProof{evals:[F;4], present:[bool;4]}`.
- **Pushforward** opened separately (`prove_pushforward_open`/`verify_pushforward_open` + `Fp3LimbColumns::
  from_fp3`): each chunk's `P^F` Fp3 column → 3 base-coefficient limbs, checked `P^F(r)=c0+β·c1+β²·c2`
  against the `PushforwardReduction` claim. `Stage8PushforwardProof{evals:Vec<[F;3]>, present:Vec<[bool;3]>}`.
- **Bytecode `RaDense`** is merged into the inventory (requests `RaDense(base+i)@PushforwardGkr`, columns
  lifted `u32→Base`).
- **Zero-column handling (the WHIR limitation):** WHIR can't open an identically-zero column. Handled two
  ways, both lockstep-safe (prover & verifier agree from the shared accumulator claim): (a) **zero-claim
  skip** for inventory entries (`R1csAux`, `RaDense`) — a zero-poly has claim 0; a non-zero poly's MLE at
  the FS-random point is non-zero w.h.p.; the prover can't forge a zero claim (booleanity/GKR bind it);
  (b) **per-limb `present` flags** for Inc/Pushforward limbs (a zero-index chunk's `P^F=[1,0,…]` ⇒ `c1`/`c2`
  all-zero; small increments ⇒ Inc `hi` limbs all-zero) — skipped limbs contribute eval 0 and the recompose/
  reconstruct check binds them. Interim: skipped columns are bound by their surrounding sumcheck + Spartan's
  `z`-open, not a dedicated PCS check.

---

## 8. Driver & e2e orchestration

- **Binary driver (`zkvm/driver.rs`):** `prove_binary`/`verify_binary` are thin wrappers over
  `prove_binary_into`/`verify_binary_into(…, accumulator, transcript)` (the `_into` variants expose the
  accumulator so the e2e can run stage-8 on the same one). Order: Spartan → memory → booleanity.
  `BinaryProof{spartan, memory, aux_evals}`.
- **Full e2e (`zkvm/e2e.rs`):** `prove_e2e::<const D, const NE>(real, bc: &BytecodeProverInputs<D>,
  transcript)` and `verify_e2e::<D,NE>(proof, params: &VerifierParams, bc: &BytecodeVerifierInputs<D>,
  transcript)`. Pipeline on ONE transcript + accumulator: **binary driver → bytecode read-raf → M7
  pushforward → stage-8 (R1csAux + bytecode RaDense inventory open, Inc limb open, Pushforward limb open)**.
  `E2eProof{binary, bytecode_read_raf, bytecode_pushforward_gkr, inc, pushforward}`. `VerifierParams` carries
  the geometry (`num_row_vars`, `log_num_cycles`, `ram_log_k`, `reg_log_k`, `n_aux`) + `R1csKey` + the public
  RAM columns; `from_witness` derives it for the gate. `D=4`/`NE=6` pinned for muldiv (the machinery is
  const-generic; a program with a different `bytecode_d` instantiates a different `D`).

---

## 9. The e2e gate (`crates/jolt-equivalence`, `goldilocks` feature)

Two test binaries (run individually — `--features goldilocks` pulls the WHIR graph and the full test set
fills the disk; build only the named `--test`):
- **`tests/goldilocks_witness_gate.rs`** — witness-integer + geometry parity: builds the goldilocks
  `CommittedWitness` from the SAME real muldiv `commitment_trace_sources` jolt-core produces and asserts
  family geometry (`instruction_d=32`, `bytecode_d=4`, `ram_d=4`, `log_k_chunk=4`, `log_t=9`), ra_dense
  index validity, bytecode chunk recomposition, and Inc limb recomposition vs jolt-core's `JoltProtocolParams`.
- **`tests/goldilocks_e2e.rs`** — the full prove/verify (M0–M4): M0 asserts the real-trace limbed R1CS is
  satisfied; M1 the binary driver round-trips; M2/M3b the full `prove_e2e`/`verify_e2e` (incl. bytecode
  read-raf + pushforward + all stage-8 opens) round-trips; M4 asserts the e2e geometry matches jolt-core's
  `core_muldiv_commitment_fixture` params (`ram_d` ≤ jolt-core's, since the goldilocks RAM is zero-init).

---

## 10. Design choices & resolved forks (the "why")

1. **Single spongefish transcript** (Fork 1) — one FS stream across prover + WHIR; Dory untouched; a trait
   seam keeps the stages generic over `Challenge`/`ProverFs`/`VerifierFs`.
2. **NON-ZK** — no BlindFold, no WHIR-zk. (The design docs' ZK sections — `ZkOpeningScheme`, WHIR-zk,
   BlindFold mapping — are NOT implemented and out of scope for this crate.)
3. **Binary Spartan, not uni-skip** — the workspace `jolt-r1cs` binary inner-reduction reaches a working
   e2e soonest; uni-skip is a deferred perf + soundness-binding pass (its `framework/{lagrange,multiquadratic,
   univariate_skip}` foundations are built + unused).
4. **Fork-2 interim seeding** — stages self-seed their `SpartanOuter`/`rv_s` openings from the witness MLE
   (carried in the proof), not bound to a sound Spartan output. Correctness-first; the binding is uni-skip Spartan.
5. **Sparse address-first read-raf** — `O(K_total + D·T)`, not dense `K·T`. Mandatory for any real program.
6. **Option C per-chunk pushforward** — each read-raf chunk discharged by its own `log_d=0` GKR at its
   distinct column point (the read-raf produces distinct points; the §4.1 shared-point assumption in the
   design doc was wrong for the integrated read-raf — resolved via per-chunk, see `m7-readraf-shared-point-gap`).
7. **Reuse over port** — `jolt-poly` (`ExpandingTable`, `IdentityPolynomial`, `GruenSplitEqPolynomial`,
   `EqPolynomial`) and `jolt-lookup-tables` (the entire field-generic table-value layer + trace bridge) are
   reused; only genuinely jolt-core-only math is ported. (Per the `goldilocks-real-program-not-mock` feedback:
   check for a field-generic twin before porting.)
8. **Inc per-limb commit** (Fork 3) — `Inc` committed as 2 signed base limbs + a linear reconstruct, not the
   recomposed value (`CommittedPolynomial` has no limb-split variants).
9. **Per-column WHIR open (M=1)** — the verifier can't reproduce cross-evals for `open_batch`; true
   intra-class RLC batching is gated on a shared-point reduction (deferred).
10. **Zero-init RAM** — the memory stage models RAM from zero (internally consistent + verifies), not the
    real initial-memory state. The `ram_k` is sized to the max accessed remapped index.
11. **WHIR zero-column handling** — zero-claim skip (inventory) + per-limb present flags (Inc/Pushforward),
    both lockstep-safe via the shared accumulator claim.
12. **GKR caches BIG_ENDIAN points** — to match the framework + WHIR convention (the GKR is internally LSB-first).
13. **Real geometry** — muldiv is `instruction_d=32`, `bytecode_d=4`, `ram_d=4`, `log_k_chunk=4`, `log_t=9`.
    (The `INSTRUCTION_D=5`/`BYTECODE_D=2` consts in the source are small-K test placeholders; bytecode e2e
    pins `D=4`.)

---

## 11. Interim soundness gaps (honest)

The e2e round-trips and is geometry-gated, but it is **not yet a complete sound zkVM**:
- **Uni-skip Spartan (Fork 2)** — stages are self-seeded, not bound to one sound Spartan execution. **Main gap.**
- **R-integration** — the limb range-checks (R-core) are standalone, not committed as `R1csRangeHalf` z-leaves
  tied to `z(r_y)`; without them a prover could equivocate on a value's limbs.
- **Zero-init RAM** — no real initial-memory state; program-output binding (`val_io`/`io_mask`) is empty.
- **Dense `RamRa` PCS discharge** — the memory stage opens a dense one-hot `RamRa`, but the committed witness
  is per-chunk `RaDense`; reconciling needs RAM-via-read-raf (not done).
- **Zero-column skips** are bound by booleanity/recompose + Spartan `z`-open, not a dedicated PCS check.
- **Instruction read-raf `r_reduction` self-seeded (Fork 2)** — the instruction read-raf is wired and PCS-
  discharged (P3b), but its reduction point is squeezed fresh from the transcript rather than bound to the
  upstream `InstructionClaimReduction` (which itself awaits the uni-skip Spartan closure). So
  instruction-lookup *structure* (read+raf at a random point, table membership) is proven, but the binding of
  that random point to the Spartan-derived lookup-output claim is interim — same posture as the bytecode
  read-raf and the memory-stage seeds.

---

## 12. What's left for full e2e parity with jolt-core

### Instruction lookups (prefix/suffix) — ✅ DONE (P3b)

The dominant remaining piece is now landed and wired into `prove_e2e`/`verify_e2e`. As built:
- **P3b-0** — jolt-core-free `Cycle→Option<usize>` table dispatch: `with_isa_struct!` is exported from
  jolt-trace (`$crate`-relative) and reused by `jolt_lookup_tables::instruction_lookup_table_index::<XLEN>`
  (`LookupTableKind::index()` = the `enum_index` analog). Gated by a jolt-equivalence parity test: matches
  jolt-core on every cycle of the real muldiv trace (483 cycles, 21 distinct tables).
- **P3b-1** — `instruction_lookup_columns::<XLEN>` (goldilocks `instruction_lookups/trace.rs`) mirrors
  `stage5_lookup_trace`: per-cycle `lookup_index` / table dispatch / `is_interleaved` over the padded length.
- **P3b-2** — `prove_e2e`/`verify_e2e` gain `Instruction{Prover,Verifier}Inputs`, three `E2eProof` fields,
  a fresh transcript-squeezed `r_reduction` (interim Fork-2), pinned `XLEN=64, D=32, NE=34`, and
  `instruction_pushforward_family` (drop-in over the family-generic M7 pushforward).
- **P3b-3** — full PCS discharge: instruction `RaDense(0..32)` joins the stage-8 inventory open and the
  per-chunk `Pushforward` `P^F` limbs are opened (base `instruction_range.start`, distinct from bytecode).
- **P3b-4** — `goldilocks_e2e` round-trips the full e2e with the instruction family and asserts the
  `instruction_range` geometry (5 tests green; the read-raf at production `LOG_K=128` proves in ~5s on muldiv).

The prefix/suffix address-phase math (P1) + the `InstructionReadRaf` sibling instance (P2) + the composable
`prove/verify_instruction_read_raf` stage (P3a) were landed earlier in the arc (see §14). The historical
plan for that math is preserved below for reference.

<details><summary>Historical IL plan (P1–P3a, as-built)</summary>

> **PORT-SOURCE CORRECTION (2026-06-03, branch `refactor/crates`).** The forward plan below supersedes
> the IL section of `PHASE6_NEXT_SESSION_PROMPT.md`, which assumed a `jolt-core/src/zkvm/lookup_table/`
> that **no longer exists** — the `refactor/crates` split deleted jolt-core. The actual read-only port
> oracle is now **`jolt-kernels/src/stage5.rs`** (the hand-written field-generic "coarse-kernel ABI"),
> where the *entire* instruction read-raf already lives (`InstructionReadRafStage5State` et al.). There is
> **no dynamic `prefix_mle`** anywhere — the prover uses static `Prefixes::evaluate` + `LookupTableKind::
> combine` (already in `jolt-lookup-tables`) plus tiny `operand_prefix_poly`/`identity_prefix_poly` helpers.
> jolt-kernels stage5 is bound to `jolt_transcript::Transcript: Clone+Default+'static`, which the WHIR
> spongefish (`ProverFs`/`VerifierFs`, borrowing, non-`Clone`) cannot satisfy and the single-spongefish
> rule forbids working around — so we **port the math into a goldilocks `framework::SumcheckInstance`**, we
> do **not** call jolt-kernels (treat it as oracle, like jolt-core was). See the `goldilocks-instruction-
> lookups-plan` memory for the full reuse map.

At `LOG_K=128` the dense `Val_s`/`RafVal_s` are infeasible; the prefix/suffix decomposition is
`Val(k)=Σ_i prefix_i(r_high)·suffix_i(r_low)`. **Most of it is REUSE:** `jolt-lookup-tables` supplies the
40 tables (`LookupTableKind`, `combine`, `evaluate_mle`, `suffixes`), the `Prefixes`/`Suffixes` enums +
static `evaluate`, `LookupBits`, and the `cycle→table` trace bridge; the goldilocks `OneHotReadRaf` cycle
phase + `cache_openings` + `expected_output_claim` and the M7 `prove_read_raf_pushforward` (family-generic)
are reusable; only the dense address-phase `Val_s` is replaced by prefix/suffix. The arc (IL-1 done):
- **IL-1 ✅** `OperandPolynomial` (verifier operand-extraction MLE).
- **F1 ✅** add `VirtualPolynomial::LookupTableFlag(usize)` (the per-table selector openings).
- **P1** port the address-phase prefix/suffix math from `jolt-kernels/stage5.rs` (`InstructionReadRafAddress
  Phase`: per-table `read_prefix_polys`[`ALL_PREFIXES`]/`read_suffix_polys`, `raf_*_q` operand polys,
  `read_table_round_evals` via `combine`, `raf_round_component_evals`, `bind_high_to_low`,
  `finish_address_phase` checkpoint + `eq_eval_at_bits` group-weight update) into a goldilocks module.
- **P2** the sibling `InstructionReadRaf` `framework::SumcheckInstance` (address HighToLow → hand-off →
  cycle phase reusing the `OneHotReadRaf` shape; `cache_openings` per-chunk `[r_k_i ‖ r_cycle]`;
  `expected_output_claim` via `LookupTableKind::evaluate_mle(r_addr)` + `OperandPolynomial` + flag claims).
  **Gotcha:** prefix/suffix binds HighToLow vs the sparse read-raf's LowToHigh — hence a *sibling* instance.
- **P3** `prove_instruction_read_raf` + e2e wiring + `instruction_pushforward_family` (+`E2eProof` field) over
  the existing M7 pushforward (unchanged) + `INSTRUCTION_D=32` + parity gate.

</details>

### Then, for true parity (the remaining post-P3b work):
1. **Uni-skip Spartan (Fork 2 binding)** — the main soundness gap. Binds every self-seeded reduction point
   (bytecode/instruction `r_reduction`, the Spartan stage seeds, the memory `spartan_seeds`) to one sound
   Spartan execution via the univariate-skip outer sumcheck. Until then the stages are individually sound
   but their challenge points are not provably the Spartan-derived ones.
2. **R-integration** — commit the limb range-checks (R-core) as `R1csRangeHalf` z-leaves tied to `z(r_y)`.
3. **RAM real initial-state + dense-`RamRa` PCS discharge** — real initial memory + program-output binding;
   reconcile the dense one-hot `RamRa` with the per-chunk committed `RaDense` (RAM-via-read-raf).
4. **Stage-5 register val-evaluation** — the register-side `Val_s` closure.
5. **True intra-class WHIR batching** (perf) · ZK / BlindFold (separate track, not this crate).

---

## 13. Build, test, gotchas

```bash
source .bolt-dev-env 2>/dev/null            # MLIR/LLVM paths (harmless if unused)
# Goldilocks crate — the per-commit gate:
cargo nextest run -p jolt-prover-goldilocks --features goldilocks --cargo-quiet           # 136 green
cargo clippy -p jolt-prover-goldilocks --features goldilocks --all-targets -- -D warnings
cargo fmt -p jolt-prover-goldilocks
# The real-trace e2e + parity gate (build ONLY the named test — the full --features goldilocks set fills the disk):
cargo test -p jolt-equivalence --features goldilocks --test goldilocks_e2e
cargo test -p jolt-equivalence --features goldilocks --test goldilocks_witness_gate
# Run the FULL e2e proof+verify on the real muldiv trace (binary + bytecode + instruction read-raf at
# LOG_K=128 + both M7 pushforwards + stage-8 WHIR opens):
cargo test -p jolt-equivalence --features goldilocks --test goldilocks_e2e \
  goldilocks_real_trace_e2e_with_read_raf -- --nocapture
```
Gotchas: `F::zero()/one()` don't resolve on concrete `GoldilocksFp3` in tests (use `from_u64`); WHIR needs
`log_t ≥ 4` and non-degenerate columns; binding LowToHigh, opening points BIG_ENDIAN; the sparse read-raf
splits its point `[r_cycle ‖ r_addr]`; spongefish `pop_pattern` is strict about read-vs-squeeze ordering
(a tamper test must replay the prover's pre-round squeezes). Workspace lints: `allow_attributes="deny"` (use
`#[expect]`), `clippy::panic="deny"`, `unused_results`, `too_many_arguments` cap 7; `.unwrap()/.expect()`
only in `#[cfg(test)]`. Commit each tested piece locally (no co-author trailer; do not push; don't commit `Cargo.lock`).

---

## 14. Commit map (this implementation arc)

Real-trace e2e (P10): `61f10492d` M0 (real-trace witness) · `32060af65` M1 (binary driver verifies) ·
`795fdf57c` M2 (stage-8 R1csAux) · `df17350f3` M3a (stage-8 Inc) · `721b6c362` M3b-1 (surface P^F) ·
`d55a4351d` M3b-2 (bytecode read-raf) · `658f0f61f` M3b-3 (M7 pushforward + RaDense/Pushforward open) ·
`d985e3cce` M4 (geometry parity gate). Instruction lookups: `6d660931d` IL-1 (OperandPolynomial).
Earlier phases (framework T, R-core, P6–P9 WHIR open, sparse read-raf, bytecode Val_s, witness gate): see
`git log` + the `goldilocks-migration-plan` memory.
