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

- **Eq-conditional rows** (`guard·(left − right) = 0`): replace each multi-limb value
  with its **linear recompose** (`Σ limbᵢ·2^{32i}`). The B-row gains terms but stays
  degree-1; the outer (Spartan) sumcheck stays **degree-2** overall. E.g. constraint 0
  `RamAddress = Rs1 + Imm` →
  `[(RAM_ADDRESS,1), (RS1_LO,−1), (RS1_HI,−2³²), (IMM_LO,−1), (IMM_HI,−2³²)]`.

- **SUB `+2⁶⁴` (constraint 8)**: re-express the bias on the **2⁶⁴-place limb** of the
  4-limb `RIGHT_LOOKUP_OPERAND` (i.e. `+1` on `RIGHT_LOOKUP_OPERAND_L2`), **not** a
  `from_i128(2⁶⁴)` field constant (which silently reduces to `2³²−1`). The operand is
  65-bit, so its `L2` limb (the `2⁶⁴` place) is `{0,1}`.

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
