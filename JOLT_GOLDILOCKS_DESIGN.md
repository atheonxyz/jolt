# Jolt × Goldilocks × WHIR-zk + LogUp\* — full integration design

**Scope.** The decision-complete plan for migrating Jolt to **Goldilocks** (`p = 2⁶⁴−2³²+1`) with
**WHIR-zk + Twist/Shout-via-LogUp\*** as the hash-based, hiding PCS. Two things are **settled** (§0):
the field is **Goldilocks base-field limbs with Fp3 challenges** (not Fp3-everywhere), and the PCS is
**WHIR paired with LogUp\*** (LogUp\* is *why* WHIR is feasible, not an optional add-on). The doc is
deliberately exhaustive — the goal is to know *everything that changes* before any code is written.
The 31-bit (BabyBear) track is deferred (and, when revisited, will use a **degree-6** extension, not
degree-5, since the quintic model is being skipped).

**Read against the real target codebase.** `jolt-core/` is the **legacy** protocol; the **new Jolt
lives in `crates/*`** and its prover/verifier are **Bolt-generated** ("generated roles"). The
foundation we build on is `main` + the jolt-v2 PR stack: **#1455** (spongefish transcript),
**#1521** (jolt-openings PCS API), #1512→#1513→#1514→#1515→#1523 (witness → bolt → generated roles →
equivalence → typed verifier). All API references below cite those PRs. **Assume #1455 is merged**
(it is not yet). Dependency crates (`../whir`, `../algebra`) are **ours to change** as needed.

---

## 0. Decisions (settled)

Two decisions are **made**; the rest of this doc is *how*, not *whether*.

1. **Field: Goldilocks base-field limbs + Fp3 challenges (DECIDED).** Witness/trace columns are
   committed and computed on as **Goldilocks base-field limbs** (`p = 2⁶⁴−2³²+1`); every Fiat-Shamir
   challenge, sumcheck round polynomial, and WHIR fold lives in the **degree-3 extension Fp3**
   (~192-bit) for soundness. We do **not** represent witnesses as full Fp3 elements
   ("Fp3-everywhere" — the commit-only benchmark's shortcut, `whir_zk::Config<Identity<Field64_3>>` /
   `encode_to_field`, `whir-pcs-bench/src/main.rs:259`). The §2.4 measurement settles it: base-field
   limbs win on commit volume, commit time, proof size, **and** sumcheck arithmetic, and the realistic
   range-check overhead is far from the break-even. The PCS API (#1521) and WHIR
   (`Basefield<Field64_3>` embedding) are *built* for this — `SourceRow` has
   `StridedU64`/`StridedI128`/`OneHot` variants precisely so a backend commits compact integers/limbs
   "without first materializing field elements" (`jolt-openings/src/sources.rs`).

2. **PCS: WHIR-zk + Twist/Shout via LogUp\* (DECIDED, inseparable).** The commitment scheme is WHIR
   (hash-based, transparent, hiding via WHIR-zk), **paired with the Twist/Shout-via-LogUp\* pushforward
   commitment — not as an optional optimization but as the reason WHIR is feasible.** Jolt's native
   one-hot `Ra` is a `K×T` sparse matrix (~337M elements, 93.5% zeros for ECDSA); committing it
   directly with a hash PCS costs `K·T` and is *worse than Dory*. LogUp\* commits the **dense**
   `ra_dense ∈ Fp^T` + one small eq-weighted pushforward `P^F` per family instead — **337M → 22M
   committed elements (15×)** — which is exactly what makes WHIR's commit competitive. WHIR and
   LogUp\* ship together; see §1A.

Everything downstream (limb decomposition §3, range-check placement §4, field layer §5, transcript §6,
the WhirScheme adapter §7, optimizations §9, stages §10) follows from these two decisions.

---

## 1. Architecture: base field + Fp3 challenges

```
JoltField / CommitmentScheme::Field   =  Fp3 = Goldilocks[X]/(X³−2)     (~191.4 bits)
   - point, eval, challenge, sumcheck round polys, Fiat-Shamir all live here
Base field  Fp                         =  Goldilocks (2⁶⁴−2³²+1)         (committed witnesses)
   - trace/witness columns, limbs; reached via Fp3::BasePrimeField
Challenge wrapper                       =  a packed Fp3 (3 base limbs), drawn from the transcript
WHIR embedding                          =  Basefield<Field64_3>  (Source=Fp, Target=Fp3)
   - commit base-field coeffs; fold/open in Fp3; cheap mixed_mul = mul_by_base_prime_field
```

This is exactly lambda_vm's split (`AIR { Field: IsFFTField + IsSubFieldOf<FieldExtension>;
FieldExtension: IsField }`, trace columns in base, `z`/`α` challenges in Fp3 —
`lambda_vm/crypto/stark/src/traits.rs:138`, `lookup.rs:1138`) and SP1's
(`PrimeField32` base columns, `ExtensionField` challenges). The **base×ext fast path** is the load
bearing operation: `Fp × Fp3 = 3 base muls` (component-wise) vs `Fp3 × Fp3 = 9 base muls`
(`lambda_vm/.../extensions_goldilocks.rs:412` vs `:297`). WHIR's `Basefield::mixed_mul =
mul_by_base_prime_field` (`whir/src/algebra/embedding.rs:151`) gives this for free on the commit side.

---

## 1A. Twist/Shout via LogUp\* — the commitment that makes WHIR feasible (DECIDED)

WHIR commits *dense* vectors (Merkle over an RS codeword); its cost is ∝ committed elements. Jolt's
native Twist/Shout commits one-hot `Ra ∈ {0,1}^{K×T}` matrices — committing those directly with WHIR
is `K·T` hashes, *worse than Dory*. So WHIR is only viable **in tandem with LogUp\*** (Wiese 2025,
`twist_shout_logup_star.pdf`), which commits the dense representation. This is settled, not a later
bolt-on: the whole point of adopting WHIR is that LogUp\* lets us commit ~15× fewer elements.

**Three axes** (`whir_logup_star_design.md`): **A** — an implicit one-hot commitment via the logup\*
*pushforward* (the novel primitive); **B** — rewire Twist/Shout's `Ra`/`Wa`/`Inc` commitments onto
Axis A (the sumcheck *structure* is untouched — only the committed objects change); **C** — the outer
dense multilinear PCS = WHIR.

**Committed-set transformation (Axis B):**

| family | Dory today (one-hot) | WHIR + LogUp\* (dense) |
|---|---|---|
| `InstructionRa`(32)/`BytecodeRa`(4)/`RamRa`(4) | one-hot `K×T` per chunk | `ra_dense ∈ Fp^T` per chunk + **one eq-weighted `P^F` per family** (3 total, §4.1) |
| `RdInc`/`RamInc` | one-hot + value | `Inc_val ∈ Fp^T` dense; reuse `wa` one-hot positions; **virtualize `wv = rv + Inc_val`** (§5.2 + fn.3) |
| `TrustedAdvice`/`UntrustedAdvice` | dense | dense (unchanged) |

Committed elements: **~337M → ~22M (15×)** — the bench's measured number, and the entire reason
WHIR's commit competes with Dory.

**The pushforward GKR opening (Figure 1 / §4.1 / §4.5.2).** To open `M̃(r_row, r_col)` for a one-hot
family, the prover commits the **eq-weighted pushforward** `P^F[k] = Σ_{j: M*[j]=k} eq(bits(j),
r_M_row)`; the LogUp\* main identity (eq. 5) is `M̃(r_row, r_col) = P̃(r_col)`. A **fan-in-2
fractional-add GKR** proves `P^F` is the genuine pushforward of `ra_dense` via
`Σ_j eq(bits(j),r_M_row)/(α − M*[j]) == Σ_k P^F[k]/(α − k)` (degree-3 per-layer sumcheck, Gruen
eq-factorization; A-side depth `log_t+log_d`, B-side depth `log_m`). The d chunks of a family are
row-concatenated into `M* ∈ Fp^{T·d}` so **one** `P^F` is committed per family (§4.1); the d input
claims share `(r_row, r_col)`, so §4.5.2 collapses them to one combined claim via an eq-weighted
linear combination (near-free). At the GKR leaves the protocol reduces to **two WHIR openings** — on
`ra_dense` (`M̃_dense`) and `P^F` (`P̃`). The GKR proof is one tail per committed-witness family on
`JoltProof` (it batches like the opening accumulator), inserted as a step between stage 7 and the
stage-8 PCS opening.

**What LogUp\* eliminates** (subsumed by GKR completeness — `GOAL.md`): the stage-6 **RA booleanity**
(`ra²−ra=0`, 20 rounds), stage-6 **Hamming booleanity** (`Σ_k ra=1`, 16 rounds), and stage-7 **Hamming
weight** (`Σ ra=T`, 4 rounds) all disappear (one-hot `ra` is never committed). They are replaced by
the per-family pushforward GKR; net soundness budget ≈ unchanged (§7.5).

**Composition with base-field limbs.** `ra_dense` (small u8 indices) and `Inc_val` (i65 → limbs +
carry) are **base** Goldilocks; `P^F` (eq-weighted sums at an Fp3 point) is **genuinely Fp3**; the GKR
runs in **Fp3** (`α`, fold randomness), with leaf denominators `α − M*[j]` = `Fp3 − base` (one cheap
base-subtraction). So the two decisions dovetail: dense witness commits in base, pushforward + GKR in
Fp3. **Three internal soundness asserts** carry into production (validated in the working prototype
`crates/whir-pcs-bench/src/gkr.rs`): the main identity (eq. 5), the GKR root histogram
`N_A·D_B == N_B·D_A`, and per-layer sumcheck consistency.

---

## 2. Why base-field limbs beat Fp3-everywhere (the evidence behind decision §0.1)

Decision §0.1 is **base-field limbs**; this section is the supporting analysis and the §2.4
measurement. Challenges are Fp3 either way — this is purely about **witness representation**. Numbers
are for the bench's real T=2¹⁹ ECDSA LogUp\* set; reduction/extension facts cited to lambda_vm and WHIR.

### 2.1 The four axes

| Axis | Goldilocks base limbs | Fp3-everywhere | Winner |
|---|---|---|---|
| **Commit volume** (T=2¹⁹) | **~195 MB** — `ra_dense` at 8 B/base-elem | **~530 MB** — every value a 24-B Fp3 (2 of 3 coeffs always 0) | **Limbs, 2.7×** |
| └ per `ra_dense` (4-bit value) | 8 B (1 base) | 24 B (66% dead) | **Limbs, 3×** |
| **Witness-multiply** (sumcheck inner loop) | `base × Fp3 = 3 muls + 3 reduce` | `Fp3 × Fp3 = 9 muls + 3 reduce + 2 dbl + overflow` | **Limbs, 3×/term** |
| └ realistic end-to-end sumcheck | baseline | ~1.5–2.2× slower | **Limbs** |
| **Columns / arithmetic row** | ~2× more (limbs + virtual carries) | fewer | **Fp3** |
| **Range-check load / MUL row** | ~14 lookups + 6 constraints | 0 (for representation) | **Fp3** |
| **p < 2⁶⁴ gap** | forces 2×u32 (never 1 element) | trivial embed `[v,0,0]` | **Fp3 (repr only)** |

Where the numbers come from:
- Fp3 = exactly **24 bytes**, mul = 3×`dot_product_3` = **9 base muls** (`extensions_goldilocks.rs:476,297`).
- base×Fp3 = **3 base muls** (`IsSubFieldOf<Degree3>::mul`, `extensions_goldilocks.rs:412`).
- lambda's prover is *architected* around this 3×: base constraints ordered first to use the
  "F×E path (3 muls)" vs extension constraints "E×E (9 muls)" (`constraints/evaluator.rs:106`,
  `lookup.rs:843`).
- Inside WHIR itself, base-committing only wins on the **commit NTT (3×, runs over Source field)**
  and **Merkle leaf/opened-row bytes (8 vs 24)**; round-1-onward folding is identical Fp3 in both
  (`whir/src/protocols/whir/prover.rs:136` lifts to Target after round 0). So **the dominant limb win
  is in Jolt's own sumchecks and commit volume, not in WHIR's folding rounds.**

### 2.2 The subtle point that settles it

Fp3-everywhere only saves work for **pure value-flow** (copy / equality / store). For *any* true
RV64 arithmetic — `wrapping_add`, mod-2⁶⁴ truncation, MUL, shifts — Fp3 addition is exact 192-bit
addition with no wraparound at 2⁶⁴, so you **still** must decompose the Fp3 result and range-check it
to get RV semantics. At that point the Fp3 representation was pure overhead (24 B, 9-mul) for that
column. Since the bulk of RV64 columns are arithmetic operands, limbs win on the columns that matter.

### 2.3 The representation policy — base-field limbs, Fp3 only where mandatory

1. **Base-field limbs** for everything that is an arithmetic operand or a hot/high-cardinality
   one-hot/`ra` column: ALU inputs/outputs, addresses, timestamps, MUL operands & 128-bit products,
   all `InstructionRa`/`BytecodeRa`/`RamRa`. 100% of the Axis-1/Axis-2 wins live here. Commit via
   `SourceRow::{StridedU64, StridedI128, OneHot}` (no Fp3 materialization).
2. **Single Fp3 embed** (`SourceRow::StridedFieldElements`, `[v,0,0]`) for: opaque value-flow columns
   that are only stored/copied/equality-checked (some advice, in-flight hashes) — limbing adds
   columns+range-checks for zero benefit and the p<2⁶⁴ gap makes a single *base* element unsafe;
   and the wide signed `RdInc`/`RamInc` (a 65-bit value needs ≥3 base limbs ≈ 24 B = Fp3 anyway, so
   pick whichever is simpler — volume is identical).
3. **Fp3 mandatory** (no limbing possible) for extension-valued data: sumcheck round polynomials,
   eq-weighted pushforwards `P^F`, Fiat-Shamir challenges, GKR accumulators.

### 2.4 Measured — the full limbed ALU trace bench (resolves the §2.3 risk)

The §2.3 open risk (does the limb range-check + carry overhead erode the commit win?) is now
**measured**, not modeled. A bench — `goldilocks-alu-bench`, currently a sibling workspace depending
on `../whir`, but movable into `crates/` now that the `digest 0.10/0.11` conflict is resolved (WHIR is
already an in-workspace dependency of `crates/whir-pcs-bench` via the `blake3 = "=1.8.3"` pin that lets
digest 0.10 and 0.11 coexist) — materializes Jolt's full committed-column inventory under LogUp\*
(40 `ra_dense` + `Inc` + carry/sign + a swept count of limb-range-check columns + 3 Fp3 `P^F`) and
commits it via WHIR under **base-Goldilocks** (`Config<Basefield<Field64_3>>`, 8 B/elem) vs
**Fp3-everywhere** (`Config<Identity<Field64_3>>`, 24 B/elem). Challenges/folds are Fp3 in both — only
the committed witness representation differs. It also microbenches the sumcheck witness-multiply
(`base×ext` vs `ext×ext`) and a small batched proof size. Commit cost is data-independent (NTT+Merkle),
so representative random values are used; per-column commits match Jolt's logical-poly model; times are
the min over N runs (Blake3 Merkle, rate 1/2, fold 4; M-series, 8 cores). Reproduce:
`cargo run --release -p goldilocks-alu-bench -- --log-t 18`.

**Results (T = 2¹⁷, 4 runs; the picture only sharpens for base at larger T):**

| Metric | base-limbs | Fp3-everywhere | base advantage |
|---|---|---|---|
| Sumcheck witness-multiply (`base×ext` vs `ext×ext`) | 8.4 ns/op | 19.0 ns/op | **2.3×** (theory 9÷3; 2.3–3.3× across runs) |
| Per-column commit @ 2¹⁷ | 2.8 ms | 9.5 ms | **3.4×** |
| Per-column commit @ 2¹⁸ | 15.5 ms | 77.4 ms | **5.0×** (advantage *grows* with T) |
| Full inventory commit (rc_extra=0) | 368 ms | 644 ms | **1.75×** |
| Full inventory committed volume | 52 MB | 128 MB | **2.45×** |
| Proof size (8-poly, 1-pt batch) | 742 KB | 1341 KB | **1.8×** |

**Range-check-overhead sweep** (extra limb-range-check `ra_dense` columns added to the base inventory):

| extra rc columns | base commit (ms) | base volume (MB) | vs Fp3 commit (644 ms) |
|---|---|---|---|
| 0 | 368 | 52 | **1.75× faster** |
| 4 | 416 | 56 | 1.55× faster |
| 8 | 440 | 60 | 1.46× faster |
| 16 | 504 | 68 | 1.28× faster |
| 32 | 607 | 84 | 1.06× faster |
| 64 | 908 | 116 | 0.71× (Fp3 wins) |

**Verdict — go base-field limbs.** Base-limbs wins on *every* axis at the realistic operating point,
and the limb range-check overhead would have to add **>~36 length-T columns on commit time (and >76 on
committed bytes)** before Fp3-everywhere catches up — i.e. nearly *doubling* the entire 40-column
lookup set. The realistic limb-range-check cost is **~4–16 extra `ra_dense` columns** (a `K=2¹⁶` range
table is ~4–8 chunks under LogUp\*; §4.2), where base-limbs is 1.3–1.6× faster to commit, 1.9–2.3× less
volume, ~1.8× smaller proof, *and* 2.3–3.3× cheaper in the sumcheck inner loop (the larger end-to-end
lever, upstream of WHIR). The per-column commit advantage also *grows* with trace length (3.4× at 2¹⁷ →
5.0× at 2¹⁸), so at production T = 2¹⁹⁺ the margin is wider still. The earlier "unless results are very
poor" caveat does not trigger: results strongly favor limbs.

*Scope of the measurement:* this is a PCS-commit + field-arithmetic + proof-size bench, not a full
end-to-end prover run. It does not execute Jolt's sumcheck stages or the actual range-check Shout
sumcheck — those would *add* to the base-limbs advantage (cheaper `base×ext` rounds) while the
range-check columns are already counted here. The inventory column counts are modeled from the ECDSA
bench's committed set plus the limb-design analysis; the per-element commit cost and field-arith ratio
are directly measured.

---

## 3. Witness representation — what is limbed, what fits, the carry mechanics

### 3.1 The p < 2⁶⁴ gap (the Goldilocks-defining constraint)

Goldilocks holds exactly `[0, p−1] = [0, 2⁶⁴−2³²]`. The top `2³²−1` values `[p, 2⁶⁴)` alias under
reduction (`u64::MAX` and `0xFFFF_FFFE` become indistinguishable witnesses —
`lambda goldilocks.rs:172`). Therefore **a raw `u64` cannot be a single canonical base element**, and
a 64-bit value is stored as **2×u32 limbs** (`DWordWL`), never recomposed to one element. lambda has
*no* `u64`-as-one-element type — every 64-bit quantity is `DWordWL`/`DWordHL`/`DWordBL`
(`config.toml:51-92`).

### 3.2 The 64/128-bit sites (audited against current code)

Jolt over BN254 today relies on every 64/128-bit value fitting in one 254-bit element; `from_u64`/
`from_i128` silently reduce (`crates/jolt-field/src/field.rs:95`). Under Goldilocks base that becomes
a silent correctness bug at these sites:

**Need limbing (the work-list):**

| Site | Where | Bits | Goldilocks fix |
|---|---|---|---|
| `RdInc`, `RamInc` | `jolt-witness/src/lib.rs:35,92,98` (`F::from_i128`); commit `jolt-prover/.../commitment.rs:511` | ~65 signed | 1 element + **1 carry bit** (or 2×u32 + sign) |
| Register `Val`, RAM `Val/init/final` | `jolt-core/.../registers/val_evaluation.rs:43`, `ram/val_check.rs:48` | 64 | 2×u32 + carry; degree of `Val=Σ inc·wa·LT` unchanged unless Inc is sign+limbs |
| **MUL/MULH\* product** `V_PRODUCT` | `jolt-r1cs/src/constraints/rv64.rs:363` (constraint 19) | **128** | **4 limbs + carry**; the single `A∘B=C` product row splits into limb-product + carry rows (worst case) |
| Lookup word outputs (And/Or/Xor, RangeCheck/LowerWord, UpperWord, shifts/Pow2/SRA/SRL/ROTR) | `jolt-lookup-tables/src/tables/{and,range_check,upper_word,...}.rs` | 64 | reconstructed word as 2×u32 + carry; **MLE weights `2^k`, k≤63 still fit (2⁶³<p)** |
| RAF operand reconstruction, all u64 `z` entries | `jolt-core/.../identity_poly.rs:45`, `rv64.rs:32` | 64–128 | 2×u32 + carry |
| SUB two's-complement bias `2⁶⁴` | `rv64.rs:72` | const | **re-express on high limb** — `2⁶⁴ ≡ 2³²−1 mod p`, so a single `2⁶⁴` coefficient is silently wrong |

**Already fit in one Goldilocks element (no limbing):** RAM/PC addresses & `unmap(k)` (≤2⁴⁸ < p,
`raf_evaluation.rs`, `identity_poly.rs:431`), timestamps/cycle index (≤2³²), all MLE weights `2^k`
for k≤63, and one-hot `Ra` columns (bit-level, field-agnostic).

**Cross-cutting blocker (independent of the limb decision):** `mul_pow_2` splits into 63-bit chunks
assuming `1<<63` is benign (`crates/jolt-field/src/field.rs:122`) — silently wrong for Goldilocks.
Must be rewritten using `2⁶⁴ ≡ 2³²−1`. Underpins RAF operand recon and timestamp scaling.

### 3.3 How Jolt range-checks today (the key enabler)

Jolt enforces "value is a valid u64" via the **Twist/Shout instruction-lookup argument** — a one-hot
`Ra` commitment of the value's bit-decomposition plus the `RangeCheck` table's MLE reconstruction
(`jolt-lookup-tables/src/tables/range_check.rs:14`, proven in stage-5 `instruction_read_raf`,
`stage5.rs:122`) — **not arithmetic range gadgets**. Booleanity (`x²−x=0`,
`hamming_booleanity.rs:30`) enforces one-hot *selector* validity, a different job. The decisive
consequence: **the bit-level one-hot machinery is field-agnostic and survives unchanged; only the
field *reconstruction of the word* breaks** under Goldilocks. We reuse this machinery for limb range
checks (§4).

### 3.4 Carry mechanics — the `2⁻³²` trick (cheap, virtual)

64-bit ADD needs **one Boolean carry per 32-bit limb**, computed `carry = 2⁻³²·(lhs[i]+rhs[i]−sum[i])`
and constrained `IS_BIT` (`lambda add.toml:25-32`). This is exact because each limb < 2³² ⇒ the
per-limb sum < 2³³ < p, and `2³² | p−1` so `2⁻³²` exists (`INV_2_32 = 18446744065119617026`). MUL
extends this to a 4-stage `2⁻³²ᵏ` carry chain over the convolution, carries range-checked to ~20 bits
(`lambda mul.rs:438`). **Carries are virtual columns** (linear combinations, *not committed*), so the
carry overhead is in *constraint degree*, not commitment volume.

### 3.5 Witness inflation summary

| Class | BN254 (today) | Goldilocks (this design) |
|---|---|---|
| u64 register/RAM value | 1 elem | 2×u32 limbs (16 B) **or** 1 elem + carry bit |
| 128-bit MUL product | 1 elem | 4 limbs + virtual carries |
| `RdInc`/`RamInc` (i65) | 1 elem | 1 elem + carry bit, or 3-limb (≈ Fp3) |
| one-hot `ra` (u8) | 1 elem | 1 base elem (8 B vs 24 B Fp3) |
| addresses / PC / timestamps | 1 elem | **1 elem** (fits) |
| net inflation | 1× | **~1.05–1.5×** (vs ~3× for 31-bit) |

This ~1.05× is the central Goldilocks advantage: most u64 slots cost **1 base element + 1 Boolean
carry**, and only the 128-bit MUL product genuinely needs 4 limbs.

---

## 4. Range-check placement — where written, where proven

The small field introduces two genuinely new checks: **(A)** a 32-bit limb `< 2³²`, and **(B)** a
Boolean carry/sign bit. Recommendation, keyed by check width (exploiting that Goldilocks needs mostly
(B), rarely (A)):

### 4.1 Boolean carries/signs → a small residual booleanity zero-check (≈ free)

- **Written:** one `carry`/`sign` base column per slot, committed as `StridedU64`/`StridedI128`. The
  recomposition `lo + 2³²·hi = v` (or `v + c·(2³²−1) = true`) is a degree-1 R1CS row in `jolt-r1cs`,
  coefficients `{1, 2³²}` (small).
- **Proven:** a small `x²−x=0` zero-check over the carry/sign columns. Note that LogUp\* **removes**
  the old stage-6 RA/Hamming booleanity sumchecks (§1A — the one-hot `ra` is never committed), so this
  is now the *only* booleanity left: reuse that machinery (`hamming_booleanity.rs:30`, degree-3 via
  `gruen_poly_deg_3`), retargeted from RA selectors to the few limb carry/sign columns. Marginal cost
  ≈ a handful of degree-3 rounds. Challenges from **Fp3** (mandatory — even for 1-bit witnesses,
  soundness is over the challenge domain).

Because Goldilocks inflation is ~1.05× (most u64 slots need only one carry bit), this covers the
*overwhelming majority* of new checks at near-zero cost — the decisive Goldilocks advantage over the
31-bit tracks (which need 3× the limbs, pushing toward heavy table-based checking).

### 4.2 Wide `< 2³²` limbs (MUL/u128 products) → reuse Jolt's Shout RangeCheck table in stage 5

- **Written:** the 32-bit limbs as base rows; under LogUp\* each becomes a dense `ra_dense ∈ Fp^T`.
- **Proven:** reuse Jolt's own `RangeCheckTable`/`LowerHalfWordTable`/`UpperWordTable`
  (`jolt-lookup-tables/src/tables/{range_check,lower_half_word,upper_word}.rs`) inside the **stage-5
  `instruction_read_raf` Shout sumcheck** — which *already* range-checks arithmetic outputs this exact
  way. **No new sumcheck instance, no degree increase**; just more one-hot rows (or, under LogUp\*, one
  extra dense column + a tiny `K≤2³²` pushforward `P^F` on the shared batched GKR).

### 4.3 What NOT to do

- **Don't** bit-decompose a 32-bit limb into 32 booleans for option-(A) (32× column blowup).
- **Don't** add a standalone SP1/lambda-style LogUp range bus as a *new* subsystem — Jolt already has
  the lookup argument (Shout); building a parallel bus duplicates infrastructure. **Adopt the small
  preprocessed range-*table* idea from SP1/lambda, but prove it through Jolt's existing Shout path.**

### 4.4 LogUp\* makes this nearly free (and LogUp\* is in — §1A)

Since LogUp\* ships with WHIR (§0.2, §1A), its pushforward GKR *is already* a fractional-sum/LogUp
argument over a dense address column. Adding a limb-range table to that machinery is therefore ~free:
one more dense `ra_dense` column + a tiny `K=2¹⁶` pushforward `P^F` ("negligible for small K" —
`whir_logup_star_design.md:106`). Options 4.1/4.2 ride the lookup infrastructure LogUp\* installs for
*all* lookups — there is no separate range subsystem to build.

### 4.5 Placement summary

| Check | Written | Proven | Mechanism | Cost |
|---|---|---|---|---|
| carry `c∈{0,1}` per u64 col | 1 base col | **stage 6 booleanity** | generalized `x²−x=0` | ~free (extra rounds) |
| sign bit (Inc) | 1 base col | stage 6 booleanity | `x²−x=0` | ~free |
| 32-bit limb `<2³²` (MUL) | dense `ra_dense` | **stage 5 instruction_read_raf** | Shout into RangeCheck/LowerHalfWord | +1 dense col + tiny pushforward |
| recomposition `lo+2³²·hi=v` | — | R1CS (`jolt-r1cs`) | degree-1 linear | negligible |

Estimated total added: **low hundreds of deg·rounds** (vs ~600+ for the deferred 31-bit track), most
of it inside sumchecks that already exist.

---

## 5. Field layer — Goldilocks base + Fp3 extension

### 5.1 Construction (arkworks fork, `../algebra`)

- **Base:** `SmallFp<GoldilocksConfig>` already exists as a test fixture
  (`algebra/test-curves/src/smallfp.rs`, gen 7) — promote to a production type.
- **Extension:** `Fp3<GoldilocksFp3Config>` via the existing `CubicExtField`/`Fp3Config` — **no new
  arkworks model** (cubic exists). Config: `NONRESIDUE=2` (`X³−2` irreducible, matches lambda + the
  bench's `Field64_3`), Frobenius `C1/C2`, two-adicity of `p³−1`, sqrt precomp. ≈ 2¹⁹².

### 5.2 Arithmetic — Montgomery-FREE (do NOT use arkworks SmallFp on the hot path)

The arkworks `SmallFp` Goldilocks backend uses **generic u128 Montgomery** (`montgomery_backend.rs:472`
— full REDC: `wrapping_mul(N_PRIME)`, second `k*MODULUS` mul, overflowing-add, cond-sub). This ignores
the prime's structure. **Port lambda's Montgomery-free reduction** (`lambda goldilocks.rs`), which is
the Plonky2/Plonky3 design:

- `reduce128(x)` using `2⁶⁴≡2³²−1`, `2⁹⁶≡−1`: 2 shifts, 1 mask, 1 sub-borrow, 1 shift-and-sub
  (replacing the mul-by-EPSILON), 1 add-carry — **~6-8 ALU ops, no extra multiply**
  (`goldilocks.rs:196`). vs ~3 u128 muls for Montgomery → **~1.5-2× on mul**.
- `add; sbb` 2-instruction modular add (`goldilocks.rs:226`, x86), non-canonical `[0,2⁶⁴)`
  representation canonicalized only at boundaries — bigger relative add win.
- Implement as a hand-coded `crates/jolt-field/src/.../goldilocks_ops.rs` (the only ops module today
  is `bn254_ops.rs`; ~200 LOC + tests).

### 5.3 Deferred-reduction accumulator (the `WideAccumulator` analog)

Jolt's BN254 prover defers Montgomery reduction across many fmadds via `WideAccumulator<Limbs<9>>`.
The Goldilocks analog: accumulate u128 products in a **192-bit (`Limbs<3>`) accumulator, reduce once**,
correcting overflow with `2¹²⁸ ≡ (2³²−1)² mod p` per overflow (`dot_product_3` is the fixed-arity
kernel, `goldilocks.rs:290`). A 192-bit lane holds ~2⁶⁴ products before flushing — amortizes
`reduce128` across the whole sumcheck inner loop. Implement `GoldilocksAccumulator` +
`GoldilocksExt3Accumulator` for the `Field::Accumulator`/`ScalarAccumulator` associated types
(`crates/jolt-field/src/field.rs:55`). **Very high impact** (mirrors the existing BN254 win).

### 5.4 `mul_by_base` (3-mul base×ext) — the sumcheck hot path

After the first sumcheck bind, polynomial values are base `Fp` and challenges are `Fp3`, so every
round is `base_value × ext_challenge` = **3 base muls** (`IsSubFieldOf<Fp3>::mul`,
`extensions_goldilocks.rs:412`), not the 9-mul full `Fp3×Fp3`. Expose `mul_by_base(&self, b: BaseF)
-> ExtF` as a first-class method on the extension/`JoltField` trait (the existing
`OptimizedMul`/`ChallengeFieldOps<F>: Mul<F,Output=F>` machinery in `jolt-core/src/field/mod.rs` is
the seam; `crates/jolt-field`'s `Field` trait must grow the same `Challenge` associated type).

### 5.5 Fp3 multiply — reduce-once dot products

`Fp3 × Fp3` = 3× `dot_product_3` = **9 base muls + 3 reduce128** (`extensions_goldilocks.rs:297`),
*not* Karatsuba (6 muls + 6 reduce) — because **reductions, not muls, dominate**, trade muls for
fewer reductions. Multiply-by-`2` (the `X³−2` reduction) is a `double`, not a real mul. Port
`dot_product_2/3` verbatim.

### 5.6 The two field-trait layers

`crates/jolt-field/src/field.rs` (`Field`, lean, BN254-only today) and `jolt-core/src/field/mod.rs`
(`JoltField`, has the `Challenge` machinery + feature gates). The migration targets `crates/jolt-field`
(the new Jolt). `crates/jolt-field`'s `Field` must grow a `Challenge` associated type and the PCS
trait's `Transcript<Challenge = Self::Field>` bound (`jolt-openings/schemes.rs`, PR #1521) becomes
`Challenge = <Self::Field as Field>::Challenge`. This is the most pervasive seam — it touches
`jolt-openings`, not just `jolt-transcript`.

---

## 6. Transcript — spongefish (post-#1455)

After #1455, both Jolt and WHIR speak **spongefish**: positional `public_message` (absorbed, no NARG
bytes — symmetric binding), `prover_message` (absorbed + NARG-serialized for replay),
`verifier_message` (squeeze), directly on `spongefish::ProverState`/`VerifierState`
(`crates/jolt-transcript/src/prover.rs`, PR #1455). WHIR adds `prover_hint`/`prover_hint_ark` for
Merkle paths kept *out of band* (not Fiat-Shamir absorbed — `whir/src/transcript/mod.rs:213`).

- **One shared transcript.** Thread a single `whir::transcript::ProverState` (which wraps
  `spongefish::ProverState` + the hint buffer) through Jolt's stage sumchecks *and* WHIR's
  commit/open. This replaces the bench's two-separate-`ProverState` hack (`main.rs:350,749`) — the
  source of its unsound Fiat-Shamir discontinuity. Expose `as_spongefish(&mut self)` on the prover
  state (only the verifier has it today, `transcript/mod.rs:289`) so Jolt's ops interleave.
- **Domain separation.** Jolt seeds the `DomainSeparator` with a Jolt protocol id that commits the
  WHIR `Config` hash as a sub-field (WHIR derives its protocol id by SHA3-512 of the ciborium-
  serialized config — `transcript/mod.rs:111`). WHIR must *not* construct its own DS in production.
- **Challenge width.** PR #1455's `OptimizedChallenge::challenge_128` is BN254-specific (squeezes a
  `u128`, lifts to `Fr`). For Goldilocks, add a **`challenge_fp3()`** decoder that draws 3 `Field64`
  limbs into one `Field64_3` (the analog) — Fp3 is only 192 bits, so you want the whole element, not a
  128-bit truncation. (`verifier_message::<Field64_3>()` via the spongefish `Codec`; `geometric_challenge`
  already draws `F: Decoding`.)
- **Hash.** Merkle leaves = Blake3 (`COPY` for ≤32-B leaves); FS sponge = Keccak/SHAKE128 (spongefish
  `StdHash`). Switch both to **Poseidon2-over-Goldilocks** only if in-circuit recursion becomes a
  goal. The BN254-hardwired `PoseidonTranscript` (`jolt-transcript/src/poseidon.rs`) is not used on
  this track.

---

## 7. WHIR-zk as the PCS — the `WhirScheme` adapter

### 7.1 WHIR internals (cost model)

WHIR is multilinear-as-univariate, little-endian, power-of-2, NTT-smooth (radix 2,3). Commit
(`irs_commit.rs:313`): interleave (folding_factor = interleaving depth) → **coset NTT over the Source
field** → Merkle over codeword rows (one row = all interleaved columns at an eval point) → OOD samples
in Fp3. Open (`whir/prover.rs:50`): two geometric-challenge RLCs (over vectors, over forms) → an
initial quadratic sumcheck folding by `2^folding_factor` → per-round commit+fold+query+grind → final
folded vector in clear → a deferred `FinalClaim` the caller checks via `mle_evaluate`. Proof =
`{narg_string (FS), hints (Merkle paths, out of band)}`.

The Goldilocks NTT engine is **already registered** for `Field64` (base), `Field64_3`, and their base
primes (`whir/src/algebra/ntt/mod.rs:28`); domain ceiling `2³²·3` — ample for Jolt (T≤2¹⁹, blowup ≤
2⁴ ⇒ codeword ≤ 2²¹). The engine uses generic Montgomery; a hand-written Goldilocks mul would speed
the NTT ~1.5-3× (§5.2).

**Base-vs-Fp3 inside WHIR** (cost model): base-committing (`Basefield<Field64_3>`) wins only on
(a) commit NTT (**3×**, runs over base `Fp` not `Fp3`), (b) Merkle leaf / opened-row bytes (**8 vs
24**), (c) round-0 fold + OOD (2×). **Round-1-onward folding, all challenges, the entire `narg_string`
are identical Fp3 in both** (the vector is lifted to Target after round 0). So WHIR's *internal* limb
win is commit + proof-size; the *sumcheck* limb win is upstream in Jolt (§2).

### 7.2 Adapter map (`crates/jolt-whir`, mirroring `jolt-dory`) over PR #1521 API

```
Field         = Field64_3 (Fp3)          commit alphabet = Field64 (base) via Basefield<Field64_3>
Output        = WhirCommitment(irs_commit::Commitment<Fp3>)   // Merkle root + OOD
OpeningHint   = irs_commit::Witness<Field64, Fp3>             // RS matrix + Merkle tree + OOD (like Dory row commitments)
Proof/BatchProof = whir::transcript::Proof { narg_string, hints }   // WHIR's batch IS its native proof
Prover/VerifierSetup = whir::Config        // transparent (no SRS) ⇒ impl VerifierSetupFromPublicParams
```

| #1521 method | WHIR mapping |
|---|---|
| `setup` | build `whir::Config::<Basefield<Fp3>>::new(size, params)`; cheap (no trusted setup) |
| `commit_batch` | the real entry — group sources by size class (num_vars), one `Config::commit` per class with `num_vectors`-interleaving (all 40 same-size RA chunks → **one** Merkle tree) |
| `open` | degenerate 1-term batch: one `MultilinearExtension{point}` form → `Config::prove` |
| `prove_batch` / `LinearOpeningScheme::prove_batch_opening` | the honest mapping — WHIR's native two-level geometric RLC (`prover.rs:131,147`) *is* the batch; let WHIR own its batching challenge |
| `verify` / `verify_batch` | `Config::verify` → then **`FinalClaim::verify(forms)`** (the deferred MLE check the adapter must complete) |
| `bind_opening_inputs` | `transcript.public_message(point); public_message(eval)` |

`CommitmentSource` rows → base limbs: `StridedU64 → 2× Field64`, `StridedI128 → 4× Field64`,
`OneHot → dense {0,1} Field64`. Recombination factors (e.g. `2³²` for the high limb) ride in the
opening relation's `eval_scale` (`BatchOutputExpression::Linear`, `jolt-openings/claims.rs`).

> **WHIR's `prove` is batched multi-vector/multi-form, not "open one poly."** Wire WHIR as a
> `LinearOpeningScheme` first; single `open` is the degenerate case.

### 7.3 ZK mapping — `ZkOpeningScheme`/`ZkLinearOpeningScheme` via `whir_zk`

- Hiding = additive mask + dual-WHIR: commit `f̂ = f + m` (witness-side WHIR) + the blinding family
  (blinding-side WHIR); the mask is woven into the RS encode (`interleaved_encode(messages, masks,
  ...)` — each codeword value = `eval(msg) + x^{len}·eval(mask)`). Blinding size `ℓ` is chosen from the
  query upper bound so leaked rows stay below `2^ℓ` (`whir_zk/mod.rs:118`).
- `HidingCommitment` = the pair of Merkle roots `(f̂ roots, blinding root)` — **no group element**
  (vs Dory's `Bn254G1 y_com`). `Blind` = the `BlindingPolynomials` family.
- **`WhirScheme` does NOT implement `EvaluationCommitmentScheme<G>`/`EvaluationCommitmentProver<G>`.**
  Those are the BlindFold-only Pedersen-`y_com` hooks ("Jolt's BlindFold integration needs… the same
  commitment generators inside its verifier R1CS. Schemes without this Dory-style evaluation
  commitment should not implement this extension trait" — `jolt-openings/schemes.rs:388`). WHIR-zk's
  hiding is self-contained; **BlindFold is dropped on this track** (and `jolt-blindfold` is already a
  stub). Any Jolt code generic over `EvaluationCommitmentScheme` must be cfg-gated off the WHIR path.

### 7.4 Batching the LogUp\*-transformed committed set

The committed set is the **LogUp\* dense set** (§1A), not one-hot: 40 `ra_dense` chunks
(`InstructionRa×32`, `BytecodeRa×4`, `RamRa×4`) + `Inc_val` (`RdInc`, `RamInc`) + advice, plus the 3
eq-weighted `P^F` pushforwards. Two size classes: large ~2¹⁹ (`ra_dense`, base Goldilocks), small
~2¹⁵ (`P^F`, Fp3, at the WHIR-zk blinding floor). Commit-time: all 40 same-length `ra_dense` →
one `Config::commit` with `num_vectors=40` (one Merkle root). Open-time (stage 8): the opening
accumulator's `ProverBatchOpeningTerm`s — including the GKR leaf openings on `ra_dense`/`P^F` (§1A) —
feed `prove_batch_opening`, which hands them to one `Config::prove` whose `vector_rlc` folds the
vectors and `constraint_rlc` folds the distinct points. **Caveat:** WHIR batches only equal-length
vectors per commitment — the 2¹⁵ and 2¹⁹ classes need separate `Config`s/`prove` calls (or pad — a
WHIR change, §8). **Note:** WHIR is now a normal in-workspace path dependency (`crates/whir-pcs-bench`
already depends on `../whir`; the `digest 0.10/0.11` conflict that once forced the two-binary bench
split is resolved via the `blake3 = "=1.8.3"` pin), so `crates/jolt-whir` links WHIR directly — no
cross-workspace proof splicing.

### 7.5 Soundness positioning of WHIR-zk

Two **independent** soundness accounts:
- **Field/Schwartz-Zippel:** challenges in Fp3 (~191 bits). Jolt's full sumcheck budget
  `Σ(deg·rounds) ≈ 2600`, dominated by stage 5 (1440). `ε ≈ 2¹¹·⁴/2¹⁹¹ ≈ 2⁻¹⁸⁰` — **~50-bit margin**
  over 128. Stage 5 *alone* needs `|F| ≥ 2¹⁴¹`, so **Fp2 (128-bit) and 128-bit-truncated challenges
  both fall ~10-11 bits short** of *provable* 128-bit. ⇒ **Fp3 is mandatory for provable 128-bit;
  don't mix challenge widths per stage** (a transcript-design hazard).
- **WHIR query/RO:** `security_level` (128) + `pow_bits` (20) + rate set the query count, **independent
  of `|F|`** above ~2¹²⁸. This is the actual 128-bit knob.

Consequences: **only witness-bearing oracles** (RA/`ra_dense`, `Inc`, **advice**) need the masked
`whir_zk` path; public/derivable polynomials (eq tables, verifier-recomputable `P^F`) use **plain
`whir`** (cheaper — no blinding family). The ~50-bit Fp3 margin means the field is never the binding
constraint, so you can spend the budget on the query side: **lower inverse rate** (smaller codeword,
faster NTT/Merkle) recovered by more queries / higher `pow_bits`. **WHIR's fold could run over Fp2**
(its soundness is query-bound) — it's Jolt's stage-5 sumcheck that forces Fp3; run the fold in Fp3 for
uniformity. An optional perf lever (`whir/verifier.rs:151` TODO): keep the *initial* fold in Fp2
before lifting to Fp3, exploiting the margin.

---

## 8. Changes inside `../whir` (the explicit ask)

| # | Change | Effort | Files |
|---|---|---|---|
| 1 | **Make `whir_zk` work without `rs_in_order`** (permuted RS) and lift the `num_vectors==1` restriction (`committer.rs:56`) so it batches the 40 RA vectors. Currently `#![cfg(feature="rs_in_order")]`, zk bench disabled. | **HIGH** | `protocols/whir_zk/{mod,committer,prover,verifier,utils}.rs` |
| 2 | **Generalize `whir_zk::Config` off the hardcoded `Identity<F>`** to `Config<M: Embedding>` so the blinded commitment can be over `Basefield<Field64_3>` (commit base, not Fp3). Today `blinded_commitment: whir::Config<Identity<F>>` (`whir_zk/mod.rs:72`) forces Fp3-everywhere for ZK. | **HIGH** | `protocols/whir_zk/mod.rs`, `committer.rs` |
| 3 | **Thread an externally-owned spongefish transcript.** Stop WHIR constructing its own `DomainSeparator`/`ProverState`; expose `as_spongefish` on the prover; let Jolt seed the protocol id. Add the Goldilocks **`challenge_fp3`** decoder. | **MED** | `transcript/mod.rs`, `jolt-transcript/src/prover.rs` |
| 4 | **Cross-size-class batching** (or a documented pad-to-class helper) — `irs_commit` requires uniform `vector_size` (`irs_commit.rs:137`); the 2¹⁵/2¹⁹ classes can't share a `Config`. | **MED** | `protocols/irs_commit.rs`, `protocols/whir/{prover,verifier}.rs` |
| 5 | **Make `Basefield<Field64_3>` base-commit the documented production path** (replace the bench's Fp3 `encode_to_field`); confirm Goldilocks NTT + 2-adicity 32. | LOW | `algebra/{fields,embedding,ntt}.rs` |
| 6 | **Strip the debug-only `Proof.pattern` field** from canonical serialization; wrap `whir::transcript::Proof` into `JoltProof`. | LOW | `transcript/mod.rs`, `jolt-whir/types.rs` |
| 7 | **(optional perf)** Hand-coded Montgomery-free Goldilocks mul/reduce in the NTT engine; Fp2 initial-fold (`verifier.rs:151` TODO). | LOW–MED | `algebra/ntt/cooley_tukey.rs`, `whir/{prover,verifier}.rs` |

---

## 9. Goldilocks optimizations — prioritized

| # | Optimization | Impact | Effort | Note |
|---|---|---|---|---|
| 1 | **Montgomery-free base reduction** (`reduce128` + non-canonical rep + `add;sbb`) — hand-coded `goldilocks_ops.rs`, NOT arkworks SmallFp u128-Montgomery | **Very high** | Low–Med | innermost op of every round; ~1.5-2× mul, bigger add win; lambda `goldilocks.rs:196,226` is the drop-in reference |
| 2 | **Deferred 192-bit accumulator** (`Limbs<3>`, `EPSILON²` overflow correction) impl `FieldAccumulator` | **Very high** | Low–Med | analog of BN254 `WideAccumulator`; amortizes `reduce128` across the inner loop |
| 3 | **`mul_by_base` (3-mul) + reduce-once Fp3 mul** | **High** | Low | post-bind rounds 3 muls not 9 (`extensions_goldilocks.rs:412`) |
| 4 | **Fix `mul_pow_2`** (use `2⁶⁴≡2³²−1`) | **High (correctness)** | Low | silently wrong today (`field.rs:122`) |
| 5 | **SIMD packed Goldilocks** (Plonky3 `PackedGoldilocks` behind a `JoltField` adapter; AVX2 = 4 lanes) | Medium (~2-3×) | High | fewer lanes than 31-bit, but offset by ~1.05× vs 3× footprint |
| 6 | **GPU kernels** (port lambda `goldilocks.cuh`/`ext3.cuh`/`ntt.cu`; 1 register/element, bit-identical CPU parity) | High (future) | High | Goldilocks is the GPU sweet spot |
| 7 | **WHIR low-rate tuning** enabled by the ~50-bit margin (lower ρ, more queries/PoW) | Medium | Low (config) | 2-adicity 32 + Fp3 margin make the field a non-constraint |

Defer 5–7 until the scalar Montgomery-free path (1–4) lands and the architecture is correct.

---

## 10. Stage-by-stage protocol under Goldilocks limbs

The IOP structure is unchanged (generated code stays generic over `<F, PCS, Transcript>`). Per stage:
(a) **every challenge → Fp3** (the `Transcript::Challenge` seam); (b) **witness reads become limb
reads + recomposition**; (c) a few degrees bump for limb/sign products; (d) commits/opens hit WHIR-zk
on the LogUp\*-transformed set. The dominant **stage-5 `instruction_read_raf` (144×10)** is the
soundness fulcrum — *all* its challenges must be Fp3 (a base-field challenge collapses it to ~2⁻²⁰).
The univariate skips (stage-1 deg-27, stage-2 deg-6) need Fp3 evaluation points. `val_evaluation`
(stages 4/5) goes degree-3→4 only if `Inc` is represented as sign+limbs; with the 1-element+carry
representation it stays degree-3.

**LogUp\* changes the stage map (§1A):** the stage-6 **RA booleanity** (20×3), stage-6 **Hamming
booleanity** (16×3) and stage-7 **Hamming weight** (4×2) are **removed** (one-hot `ra` is never
committed). A new **pushforward-GKR step** runs between stage 7 and the stage-8 opening: per family,
the fan-in-2 fractional GKR (degree-3 per layer) reducing `ra_dense`+`P^F` to two WHIR leaf openings.
Stage 6 keeps only a small residual `x²−x` booleanity over the limb carry/sign columns (§4.1). Stage 8
becomes a WHIR-zk batched multilinear open over the LogUp\* set + the GKR leaves. (Full pre-LogUp\*
22-instance table in `JOLT_SMALLFIELD_WHIR_MIGRATION.md §6`.)

---

## 11. Crate change map

| Crate | Change |
|---|---|
| `../algebra/ff` | `GoldilocksFp3Config` (no new model — `CubicExtField` exists); promote base config; **port Montgomery-free Goldilocks ops** |
| `crates/jolt-field` | add `Challenge` assoc type to `Field`; `goldilocks_ops.rs` (Mont-free); `GoldilocksAccumulator`/`Ext3Accumulator`; **fix `mul_pow_2`**; `mul_by_base` |
| `crates/jolt-transcript` | (post-#1455) `challenge_fp3` decoder; `Challenge = Fp3` |
| `crates/jolt-openings` | relax `Transcript<Challenge = Self::Field>` → `Challenge = Field::Challenge` |
| **`crates/jolt-whir` (NEW)** | `WhirScheme: CommitmentScheme + LinearOpeningScheme + ZkOpeningScheme + ZkLinearOpeningScheme` over `Basefield<Field64_3>` (in-workspace dep on `../whir`); **not** `EvaluationCommitmentScheme`. Also owns the **LogUp\* pushforward-GKR opening** (port `crates/whir-pcs-bench/src/gkr.rs`): eq-weighted `P^F`, §4.1 batching, §4.5.2 reduction, the fan-in-2 fractional GKR, and the two leaf openings → `ProverOpeningAccumulator` |
| `crates/jolt-witness` | per-field decomposition in `dense_i128_column_to_field` (→ base limbs + carry); new carry/sign/limb columns; produce **`ra_dense`** (argmax) + `Inc_val` instead of one-hot for LogUp\* |
| `crates/jolt-r1cs` | `z` gains limb columns; recomposition constraints; SUB bias re-expressed on high limb; `UniformSpartanKey` widths |
| `crates/jolt-lookup-tables` | reuse `RangeCheck`/`LowerHalfWord`/`UpperWord` for limb range-checks as LogUp\* families; decompose word-valued outputs |
| `crates/jolt-prover/stages` | Fp3 challenges; limb reads/recomposition; **remove RA/Hamming booleanity + Hamming-weight (LogUp\*)**; residual carry/sign booleanity in stage 6; pushforward-GKR step before stage 8; WHIR-zk commit/open |
| `crates/jolt-verifier` | mirror: Fp3 challenge domain; limb recomposition in claim reconstruction; pushforward-GKR verify; WHIR-zk verify |
| `crates/jolt-blindfold`, `jolt-dory`, `jolt-crypto` | untouched — the BN254/Dory/BlindFold track stays |
| `goldilocks-alu-bench` (this PR) | the §2.4 measurement crate; can move into `crates/` now the digest conflict is resolved |
| features | `--features goldilocks,whir`; mutually exclusive with BN254/Dory default; one monomorphized binary per field |

---

## 12. Milestones

- **M0 — field.** `GoldilocksFp3` + `JoltField` impl with Montgomery-free `goldilocks_ops.rs` + the
  192-bit deferred accumulator + `mul_by_base`; **fix `mul_pow_2`**; `Challenge = Fp3` + `challenge_fp3`.
  Unit-test against lambda's `goldilocks.rs`/`extensions_goldilocks.rs` and the bench's `Field64_3`.
- **M1 — transcript + WHIR PCS, no limbs yet, fibonacci e2e.** `crates/jolt-whir` over
  `whir::Config<Basefield<Field64_3>>`; commit RA naively (Fp3-embed or one-hot) to prove the
  field/transcript/PCS swap structurally correct. Single shared spongefish transcript.
- **M2 — base-field limbs + carries.** Limb decomposition at the §3 sites; residual carry/sign
  booleanity in stage 6; wide-limb range checks via the stage-5 Shout table; SUB-bias fix. `muldiv`
  under `--features host,goldilocks,whir` (primary correctness gate, WHIR-zk-hiding, non-BlindFold).
- **M3 — LogUp\* (REQUIRED, not optional).** Port `crates/whir-pcs-bench/src/gkr.rs` into
  `crates/jolt-whir`: `ra_dense`/`Inc_val` witness (replace one-hot), per-family eq-weighted `P^F`,
  §4.1 batching, §4.5.2 reduction, fan-in-2 fractional GKR, leaf openings → accumulator. **Remove**
  the RA/Hamming booleanity and Hamming-weight sumchecks. Wide-limb range checks become LogUp\*
  families. This is what makes the commit competitive (337M→22M); without it WHIR-over-one-hot is
  worse than Dory.
- **M4 — WHIR-zk hardening.** Generalize `whir_zk` off `Identity` + `rs_in_order`; batch the RA
  vectors; cross-size-class handling. RAM-heavy test.
- **M5 — perf.** SIMD `PackedGoldilocks`, WHIR low-rate tuning, optional GPU.

Sequencing note: M1 may commit `ra_dense` naively (no GKR) just to prove the field/PCS swap; M3 makes
LogUp\* real. M3 is on the critical path, not deferred — WHIR's feasibility depends on it (§0.2, §1A).

---

## 13. Open questions & risks

1. **Range-check overhead — RESOLVED (§2.4).** Measured via `../goldilocks-alu-bench`: base-limbs
   commits the full inventory 1.75× faster / 2.45× smaller and stays ahead until limb range-checks add
   >~36 columns (realistic: ~4–16). Decision confirmed: **base-field limbs**. (Remaining: a true
   end-to-end prover run, which only widens the margin.)
2. **`whir_zk` is in-progress** (`rs_in_order`-gated, `Identity`-only, `num_vectors==1`). Items #1/#2
   in §8 are the production-readiness gate for base-committed ZK — this is the single biggest schedule
   risk and lands in M4.
3. **LogUp\* GKR memory at `K=2³²` (RAM).** Even with `d=4` batching, upper GKR layers touch large
   vectors; needs a per-layer memory audit (streaming-pyramid optimization, `GOAL.md` milestone 7)
   before claiming feasibility at large traces. (LogUp\* itself is decided — this is an
   implementation-cost risk, not a whether.)
4. **Pushforward-GKR ↔ Jolt Fiat-Shamir.** The pushforward GKR proof is one transcript; confirm it
   threads cleanly through the single shared spongefish transcript (§6) and whether it can share
   challenges with WHIR's own folds (the paper is silent — `whir_logup_star_design.md:246`).
5. **Cross-size-class batching** — WHIR batches only uniform-length vectors; resolve via padding or a
   protocol change (§8 #4).
6. **`val_evaluation` degree** — confirm whether `Inc` is 1-element+carry (degree-3 preserved) or
   sign+limbs (degree-4); update the `input_claim` accordingly.
7. **Mont-free Goldilocks in the WHIR NTT** — the registered engine uses generic Montgomery; the
   ~1.5-3× NTT speedup needs the hand-coded reduction wired into `cooley_tukey.rs`.

---

## Appendix — evidence index (selected)

- **Target APIs:** transcript PR #1455 (`crates/jolt-transcript/src/{lib,prover,verifier,codec}.rs`,
  `specs/jolt-transcript-spongefish.md`); PCS PR #1521 (`crates/jolt-openings/src/{schemes,sources,
  claims}.rs`, `specs/jolt-openings-crate.md`).
- **WHIR:** commit `whir/src/protocols/irs_commit.rs:313`; encode `algebra/ntt/cooley_tukey.rs:411`;
  open/fold `protocols/whir/prover.rs:50-308`; embedding `algebra/embedding.rs:141`; NTT registry
  `algebra/ntt/mod.rs:28`; whir_zk `protocols/whir_zk/{mod:72,committer:84}`; fields `algebra/fields.rs:102`.
- **Goldilocks reference (lambda_vm):** `crypto/math/src/field/goldilocks.rs:59,172,196,226,290`;
  `extensions_goldilocks.rs:267,297,412,476`; limbs `prover/src/tables/{mul,memw}.rs`,
  `spec/src/{config,add}.toml`; LogUp `crypto/stark/src/lookup.rs`; CUDA `crypto/math-cuda/.../goldilocks.cuh`.
- **Jolt sites:** witness `crates/jolt-witness/src/lib.rs:35,92,98`; MUL `crates/jolt-r1cs/src/constraints/rv64.rs:363,72`;
  range tables `crates/jolt-lookup-tables/src/tables/{range_check,lower_half_word,upper_word}.rs`;
  booleanity `jolt-core/src/{subprotocols/booleanity.rs:13, zkvm/ram/hamming_booleanity.rs:30}`;
  stages `crates/jolt-prover/src/stages/{stage5.rs:122, stage6.rs:1843}`; `mul_pow_2` `crates/jolt-field/src/field.rs:122`.
- **Optimization sources:** Plonky2 `goldilocks_field.rs` (reduce128, add_no_canonicalize); Remco
  Bloemen "Goldilocks Reduction"; Plonky3 packed Goldilocks; WHIR eprint 2024/1586; LogUp eprint
  2022/1530 + GKR 2023/1284.
