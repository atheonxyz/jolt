# Limbed RV64 R1CS — pinned design (Phase 2, M5)

The Goldilocks port can't reuse `crates/jolt-r1cs/src/constraints/rv64.rs` as-is:
that constraint set is field-generic but **BN254-shared** (used by jolt-kernels/
equivalence/trace), and under Goldilocks (`p = 2⁶⁴−2³²+1`) **every u64-valued R1CS
variable aliases mod p** (`from_u64(v)` ≡ `from_u64(v−p)` for `v ∈ [p, 2⁶⁴)`), so a
single small-field element is unsound. This is a *new* limbed constraint set built
in `jolt-prover-goldilocks/src/r1cs/`, adapted from `rv64.rs`. The BN254 `rv64.rs`
is left untouched.

Grounded in the value semantics mapped from `jolt-core/src/zkvm/r1cs/inputs.rs`,
`crates/jolt-trace/src/r1cs_witness.rs`, and `crates/jolt-lookup-tables/`.

## Representation (user-confirmed: "Mixed")

Three limb conventions; the scalar field is `F = GoldilocksFp3`, limbs embed via
`from_u64`/`from_i128`:

1. **Unsigned u64 → 2 unsigned 32-bit limbs** `(lo, hi)`, each `∈ [0, 2³²)`,
   value `= lo + 2³²·hi`. Linear recompose; unsigned range-check `< 2³²` (M6).
2. **Signed, used only linearly → signed 2-limb** `(lo, hi)` (M4
   `i128_to_signed_limbs`; `hi` carries the sign, `lo ∈ [0,2³²)`). Linear recompose
   `lo + 2³²·hi`; **shifted** range-check `hi + 2³² ∈ [0, 2³³)` plus `lo < 2³²`.
3. **MUL operand / product → sign bit + unsigned magnitude limbs** (operand: sign +
   2 limbs; product: sign + 4 limbs). Keeps the schoolbook on clean unsigned 32-bit
   limbs; the sign is applied to the product. Boolean sign-checks (M6).

Recompose weight `2³² < 2⁶³` fits `i64`, so the existing `row::<F>` helper works for
all linear recompositions; only genuine `> i64` constants would need `row_wide`.

### Dual-use operands `LEFT_INSTRUCTION_INPUT` / `RIGHT_INSTRUCTION_INPUT` (resolved)

These appear **both** linearly (lookup-operand eq-constraints 7/8/9/10) **and**
multiplicatively (the always-present `Product = Left × Right`, constraint 19).
Grounding in jolt-core (`zkvm/r1cs/inputs.rs:281-290`): `to_instruction_inputs`
returns `(left: u64, right: i128)` for **every** instruction, and the product is
`S64::from_u64(left) × S128::from_i128(right)` — i.e. **Left is always unsigned**
and only **Right is signed** (the per-opcode MUL signedness is baked into `right`).
So:
- `LEFT_INSTRUCTION_INPUT` = **unsigned 2-limb** `(lo, hi)`. Linear value `lo+2³²·hi`
  is used directly in the eq-constraints; the same `(lo, hi)` are the MUL magnitude
  limbs (the schoolbook's `left_sign` is constant `0`).
- `RIGHT_INSTRUCTION_INPUT` = **sign + magnitude** `(sign, mlo, mhi)` for the MUL
  schoolbook, **plus a derived signed value** `RIGHT_VAL` for the linear uses:
  one degree-2 derivation `RIGHT_VAL = (1−2·sign)·(mlo + 2³²·mhi)` (realized as two
  product rows `sign·mlo`, `sign·mhi` + one linear row). The eq-constraints use
  `RIGHT_VAL` **linearly**, so they stay degree-2 in the outer sumcheck; the
  schoolbook uses `(sign, mlo, mhi)`. This is the only place a derived value var is
  needed (everything else recomposes linearly).

## Per-variable layout (`z`, per cycle)

| Variable(s) | Repr | # cols |
|---|---|---|
| `CONST` (=1) | single | 1 |
| `PC`, `UNEXPANDED_PC`, `NEXT_PC`, `NEXT_UNEXPANDED_PC` (`<2³²`, bytecode-pinned) | single | 4 |
| `RAM_ADDRESS` (`≤2⁴⁸`; = recomp(Rs1)+recomp(Imm), bounded for valid Load/Store) | single | 1 |
| 15× `FLAG_*`, `SHOULD_BRANCH`, `SHOULD_JUMP`, `BRANCH`, `NEXT_IS_NOOP`, `NEXT_IS_VIRTUAL`, `NEXT_IS_FIRST_IN_SEQUENCE` | boolean single | 21 |
| `RS1/RS2/RD_WRITE/RAM_READ/RAM_WRITE_VALUE`, `LEFT_LOOKUP_OPERAND`, `LOOKUP_OUTPUT` | unsigned 2-limb | 7×2 |
| `IMM` | signed 2-limb | 2 |
| `LEFT_INSTRUCTION_INPUT`, `RIGHT_INSTRUCTION_INPUT` | sign + 2-limb magnitude | 2×3 |
| `PRODUCT` | sign + 4-limb magnitude | 5 |
| `RIGHT_LOOKUP_OPERAND` (up to 128-bit: holds Product for MUL) | 4 limbs | 4 |

`NUM_R1CS_INPUTS` grows ~35 → ~57 (+ MUL carry columns, below). `UniformSpartanKey`
column counts update accordingly.

> **Bounded-address assumption** (carried from lambda_vm / the chip boundary):
> `RAM_ADDRESS = Rs1 + Imm ≤ 2⁴⁸` for valid Load/Store, so it fits one element. If a
> trace can violate this, promote `RAM_ADDRESS` to 2 limbs.

## Constraint transformation

> **CORRECTION (M5, locked):** an earlier draft of this section said multi-limb values
> could be related by a single **linear field recompose** (`Σ limbᵢ·2^{32i}`). That is
> **unsound**: in the field `2⁶⁴ ≡ 2³²−1` and `2⁹⁶ ≡ −1`, and even a 2-limb `u64 ≥ p`
> aliases, so `recompose(value)` equals `value mod p`, not the integer. With
> range-checked limbs `< 2³²`, two distinct `u64`s differing by `p` share a recompose,
> so `recompose(a) = recompose(b)` does **not** force `a = b` — a prover can equivocate
> with `a = b + p`. Multi-limb **equality** and **arithmetic** must therefore be done
> **limb-by-limb** (with `2⁻³²` carries for add/sub), exactly the lambda_vm pattern the
> MUL schoolbook already uses. The cost delta over the (mandatory) limbing is small
> (~15 aux carry columns + ~30 rows; R1CS is not the Jolt bottleneck) and degree stays
> 2. Only genuinely-small values stay single-element/recompose-free (see below).

Per-constraint-type rules (faithful to `rv64.rs`, retargeted limb-wise):

- **Value equality** `guard·(a − b) = 0` (constraints 2/3/4/12 and the zero-checks
  1/5/11): emit **per-limb** equality (`guard·(a_lo − b_lo)=0`, `guard·(a_hi − b_hi)=0`).
  Sound because range-checked limbs uniquely determine the integer.

- **Full-u64 lookup-operand add/sub** (7 ADD, 8 SUB): **limb-wise with `{0,1}` carries**.
  Grounded in `add.rs`/`sub.rs::to_lookup_operands`: `LeftLookupOperand = 0` always, and
  `RightLookupOperand = Left + (Right as u64)` (ADD) / `Left + (2⁶⁴ − (Right as u64))`
  (SUB), where for the 64-bit ADD/SUB/MUL arm `Right = rs2 as i128 ≥ 0`, so its magnitude
  limbs equal `Right as u64` and all carries are Boolean. ADD: `RLO = Left + Right`
  (`L₂` = high carry, `L₃ = 0`). SUB: encoded as **`RLO + Right = Left + 2⁶⁴`** (all terms
  non-negative — no signed carries, no `2⁶⁴` field constant; the `+2⁶⁴` is `+1` on the
  `L₂` carry equation). The 4-limb `RLO` limbs are range-checked `< 2³²` (M6), so `L₂∈{0,1}`.

- **`RamAddress = Rs1 + Imm` (constraint 0)**: limb-wise — limb0 carries (`{0,1}`), limb1
  is **exact** (`rs1_hi + imm_hi + c0 = addr_hi`, both `< p`, so it pins `addr_hi` directly;
  a too-large `Rs1` forces `addr_hi ≥ 2³²` which the address range-check rejects). `Imm`
  is the signed 2-limb (`imm_hi` signed); no high carry variable needed.

- **MUL lookup operand (constraint 9)** `RightLookupOperand = Product`: **per-limb**
  `RLO_i = P_i` (`to_lookup_operands` gives the unsigned `Left × (Right as u64)`, and for
  64-bit MUL both factors `≥ 0` so `Product.sign = 0`).

- **Small-value / single-element (recompose safe because the integer is `< p`)**:
  PC family (`PC, UnexpandedPC, NextPC, NextUnexpandedPC < 2³²`), flags/booleans. So
  constraints 13/15/16/17 (PC arithmetic) and 18 use ordinary recompose — e.g. 13
  `recompose(RdWrite) = UnexpandedPC + 4 − 2·IsCompressed` is a single row (result `< 2³³ < p`).

- **MUL product (constraint 19)** `Left × Right = Product`: the 4-limb schoolbook
  ([`mul.rs`], validated) on magnitudes, `Left.sign` pinned to 0 (`Left` always unsigned).

- **Boolean products (20/21)**: `ShouldBranch = recompose(LookupOutput)·Branch`
  (recompose safe — `LookupOutput ∈ {0,1}` whenever `Branch = 1`), `ShouldJump =
  Jump·(1 − NextIsNoop)`.

- **MUL product (constraint 19)** `Left × Right = Product`: the single A·B=C row
  expands to the **4-limb schoolbook** on the unsigned magnitudes
  `(Llo,Lhi) × (Rlo,Rhi) = (P0,P1,P2,P3)` with `2⁻³²` virtual carries, plus a sign
  relation `Product.sign = Left.sign ⊕ Right.sign`:
  ```
  t0 = Llo·Rlo                       P0 = t0 mod 2³²,  c0 = t0 ÷ 2³²
  t1 = Llo·Rhi + Lhi·Rlo + c0        P1 = t1 mod 2³²,  c1 = t1 ÷ 2³²
  t2 = Lhi·Rhi + c1                  P2 = t2 mod 2³²,  c2 = t2 ÷ 2³²
                                     P3 = c2
  ```
  Each `tᵢ` is a sum of ≤2 limb-products (`<2⁶⁴`) + a carry; the carries `cᵢ` are
  range-checked (`c0,c1 < 2³³`, via the carry-bit/`2⁻³²` trick using
  `decompose::carry_bit` + a range check). Partial products `limb·limb` are degree-2
  → the outer sumcheck stays degree-2. Realized as a handful of constraint rows +
  carry columns. (MULHU/MULH select `P2..P3` vs `P0..P1` per opcode, as today.)

- **Recomposition rows**: for each multi-limb value consumed elsewhere as a single
  quantity, a degree-1 row `recomp = Σ limbᵢ·2^{32i}` ties the limbs to the value.

## Degree / soundness

- Outer (Spartan) sumcheck: **degree-2 preserved** (recompose is linear; limb
  products are degree-2; sign factors land only on MUL, kept degree-2 by pushing the
  sign onto the product not the partial-products).
- `val_evaluation` (`Inc = lo + 2³²·hi`, signed 2-limb): **degree-3 preserved** (M4).
- **Soundness depends on M6 range-checks**: every limb `< 2³²` (shifted for signed
  `hi`), carries `< 2³³`, signs boolean. Without them the prover can equivocate on a
  value's limbs. So the limbed R1CS lands **coupled to M6**, not standalone.

## Validation

- **Standalone (this step):** `check_witness` on hand-built *satisfying* witnesses for
  representative ops (no-op, ADD, SUB, MUL, load/store) — validates the constraint
  algebra (mirrors `rv64.rs::noop_satisfies_constraints`). Plus the MUL schoolbook +
  carry rows vs an `i128` reference product.
- **End-to-end:** the real gate is M8 (Goldilocks+WHIR `muldiv` e2e + `jolt-equivalence`
  vs the jolt-core BN254 oracle).

## Implementation order (within M5)

1. `r1cs/rv64_limbed.rs`: limbed variable layout (the `V_*_LO/HI/…` index constants)
   + the eq-conditional constraints (recompose) + SUB-bias + recomposition rows;
   no-op/ADD/SUB `check_witness` satisfaction tests.
2. The MUL 4-limb schoolbook + carry rows + sign relation; MUL `check_witness` +
   `i128`-reference product test. (The single hardest, soundness-critical row.)
3. `UniformSpartanKey` width wiring (when the Spartan port lands).
