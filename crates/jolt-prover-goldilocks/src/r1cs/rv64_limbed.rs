//! The full limbed RV64 R1CS constraint matrices — the Goldilocks analogue of the
//! BN254-shared `crates/jolt-r1cs/src/constraints/rv64.rs` (22 constraints / 38 vars).
//!
//! Over Goldilocks (`p = 2⁶⁴−2³²+1`) a single base-field element cannot hold a full
//! `u64` (`v ≥ p` aliases), and the field recompose `lo + 2³²·hi` of a multi-limb value
//! equals the value **mod p** (because `2⁶⁴ ≡ 2³²−1`, `2⁹⁶ ≡ −1`). So multi-limb
//! **equality** and **arithmetic** are done **limb-by-limb** with `2⁻³²` carries — never
//! a single recompose for full-range values. Only genuinely-small values (`PC < 2³²`,
//! flags) recompose safely. See `../../LIMBED_R1CS.md` for the full design + the
//! `to_lookup_operands` grounding (`add.rs`/`sub.rs`/`mul.rs`: `LeftLookupOperand = 0`,
//! `RightLookupOperand = Left + (Right as u64)` etc.).
//!
//! This builds the constraint *algebra*. Soundness additionally requires the M6 range
//! checks (every limb `< 2³²`, carries Boolean, signs Boolean) — without them a prover
//! can equivocate on a value's limbs. Real-op satisfying-witness validation against the
//! jolt-core oracle is the M8 e2e gate; here we validate with hand-built honest
//! witnesses (no-op / ADD / SUB / MUL / load) + tamper rejection.

use jolt_field::Field;
use jolt_r1cs::constraint::SparseRow;
use jolt_r1cs::ConstraintMatrices;

use super::mul::{push_mul_constraints, MulVars, NUM_MUL_ROWS};

/// Per-cycle `z`-variable indices for the limbed RV64 layout. Full-`u64` values are
/// 2 unsigned 32-bit limbs `[lo, hi]`; `imm` is a signed 2-limb (`hi` carries sign);
/// `right_mag`/`product` are unsigned magnitudes with a separate sign bit; PC-family
/// and flags are single elements; `rlo` (RightLookupOperand) is up to 128-bit (4 limbs).
#[derive(Clone, Copy, Debug)]
pub struct Vars {
    pub const_one: usize,

    pub pc: usize,
    pub unexpanded_pc: usize,
    pub next_pc: usize,
    pub next_unexpanded_pc: usize,

    pub should_branch: usize,
    pub should_jump: usize,
    pub branch: usize,
    pub next_is_noop: usize,
    pub next_is_virtual: usize,
    pub next_is_first_in_sequence: usize,

    pub f_add: usize,
    pub f_sub: usize,
    pub f_mul: usize,
    pub f_load: usize,
    pub f_store: usize,
    pub f_jump: usize,
    pub f_write_lookup_to_rd: usize,
    pub f_virtual: usize,
    pub f_assert: usize,
    pub f_do_not_update_pc: usize,
    pub f_advice: usize,
    pub f_is_compressed: usize,
    pub f_is_first_in_sequence: usize,
    pub f_is_last_in_sequence: usize,

    pub rs1: [usize; 2],
    pub rs2: [usize; 2],
    pub rd_write: [usize; 2],
    pub ram_read: [usize; 2],
    pub ram_write: [usize; 2],
    pub ram_address: [usize; 2],
    pub left_lookup: [usize; 2],
    pub lookup_output: [usize; 2],
    pub left: [usize; 2],
    /// Signed 2-limb (`hi` carries sign).
    pub imm: [usize; 2],

    /// `RIGHT_INSTRUCTION_INPUT` magnitude (`Right as u64`) + sign. For the 64-bit
    /// ADD/SUB/MUL arm `Right = rs2 ≥ 0`, so `right_mag` is the value and `right_sign = 0`.
    pub right_sign: usize,
    pub right_mag: [usize; 2],
    /// `Left` is always unsigned; pinned to 0 by a constraint (used by the schoolbook).
    pub left_sign: usize,

    pub product_sign: usize,
    pub product: [usize; 4],

    /// MUL schoolbook intermediates (partial products + carries).
    pub q: [usize; 4],
    pub mul_c0: usize,
    pub mul_c1: usize,
    pub mul_c2: usize,
    pub sign_prod: usize,

    /// RightLookupOperand (4 unsigned limbs; holds the 65-bit ADD/SUB sum or the
    /// 128-bit MUL product).
    pub rlo: [usize; 4],

    /// Limb-add carries (Boolean; M6).
    pub ram_addr_c0: usize,
    pub add_c0: usize,
    pub sub_c0: usize,
    pub sub_c1: usize,
}

struct Alloc(usize);
impl Alloc {
    fn one(&mut self) -> usize {
        let i = self.0;
        self.0 += 1;
        i
    }
    fn pair(&mut self) -> [usize; 2] {
        [self.one(), self.one()]
    }
    fn quad(&mut self) -> [usize; 4] {
        [self.one(), self.one(), self.one(), self.one()]
    }
}

/// Allocate the per-cycle variable layout. Returns the indices and the total var count.
pub fn layout() -> (Vars, usize) {
    let mut z = Alloc(0);
    let vars = Vars {
        const_one: z.one(),
        pc: z.one(),
        unexpanded_pc: z.one(),
        next_pc: z.one(),
        next_unexpanded_pc: z.one(),
        should_branch: z.one(),
        should_jump: z.one(),
        branch: z.one(),
        next_is_noop: z.one(),
        next_is_virtual: z.one(),
        next_is_first_in_sequence: z.one(),
        f_add: z.one(),
        f_sub: z.one(),
        f_mul: z.one(),
        f_load: z.one(),
        f_store: z.one(),
        f_jump: z.one(),
        f_write_lookup_to_rd: z.one(),
        f_virtual: z.one(),
        f_assert: z.one(),
        f_do_not_update_pc: z.one(),
        f_advice: z.one(),
        f_is_compressed: z.one(),
        f_is_first_in_sequence: z.one(),
        f_is_last_in_sequence: z.one(),
        rs1: z.pair(),
        rs2: z.pair(),
        rd_write: z.pair(),
        ram_read: z.pair(),
        ram_write: z.pair(),
        ram_address: z.pair(),
        left_lookup: z.pair(),
        lookup_output: z.pair(),
        left: z.pair(),
        imm: z.pair(),
        right_sign: z.one(),
        right_mag: z.pair(),
        left_sign: z.one(),
        product_sign: z.one(),
        product: z.quad(),
        q: z.quad(),
        mul_c0: z.one(),
        mul_c1: z.one(),
        mul_c2: z.one(),
        sign_prod: z.one(),
        rlo: z.quad(),
        ram_addr_c0: z.one(),
        add_c0: z.one(),
        sub_c0: z.one(),
        sub_c1: z.one(),
    };
    (vars, z.0)
}

/// Total constraint rows produced by [`rv64_limbed_constraints`].
pub const NUM_LIMBED_ROWS: usize = 53;

fn mul_vars(v: &Vars) -> MulVars {
    MulVars {
        const_one: v.const_one,
        left_lo: v.left[0],
        left_hi: v.left[1],
        left_sign: v.left_sign,
        right_lo: v.right_mag[0],
        right_hi: v.right_mag[1],
        right_sign: v.right_sign,
        p0: v.product[0],
        p1: v.product[1],
        p2: v.product[2],
        p3: v.product[3],
        product_sign: v.product_sign,
        q0: v.q[0],
        q1: v.q[1],
        q2: v.q[2],
        q3: v.q[3],
        c0: v.mul_c0,
        c1: v.mul_c1,
        c2: v.mul_c2,
        sign_prod: v.sign_prod,
    }
}

fn eq_row<F: Field>(
    a: &mut Vec<SparseRow<F>>,
    b: &mut Vec<SparseRow<F>>,
    c: &mut Vec<SparseRow<F>>,
    guard: SparseRow<F>,
    body: SparseRow<F>,
) {
    a.push(guard);
    b.push(body);
    c.push(Vec::new());
}

fn prod_row<F: Field>(
    a: &mut Vec<SparseRow<F>>,
    b: &mut Vec<SparseRow<F>>,
    c: &mut Vec<SparseRow<F>>,
    left: SparseRow<F>,
    right: SparseRow<F>,
    out: SparseRow<F>,
) {
    a.push(left);
    b.push(right);
    c.push(out);
}

/// Build the limbed RV64 constraint matrices over `F` (the constraints are
/// field-generic; the Goldilocks port instantiates `F` at the base field for the limb
/// algebra and at `Fp3` in the prover's sumcheck).
pub fn rv64_limbed_constraints<F: Field>() -> ConstraintMatrices<F> {
    let (v, num_vars) = layout();
    let mut a: Vec<SparseRow<F>> = Vec::with_capacity(NUM_LIMBED_ROWS);
    let mut b: Vec<SparseRow<F>> = Vec::with_capacity(NUM_LIMBED_ROWS);
    let mut c: Vec<SparseRow<F>> = Vec::with_capacity(NUM_LIMBED_ROWS);

    let one = F::from_u64(1);
    let neg_one = F::from_i64(-1);
    let w32 = F::from_u64(1u64 << 32);
    let neg_w32 = F::from_i64(-(1i64 << 32));
    let two = F::from_u64(2);
    let four = F::from_u64(4);
    let neg_four = F::from_i64(-4);

    let load_store = || vec![(v.f_load, one), (v.f_store, one)];

    // 0: RamAddress = Rs1 + Imm  (limb-wise; limb0 carries, limb1 exact). guard=Load+Store
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        load_store(),
        vec![
            (v.rs1[0], one),
            (v.imm[0], one),
            (v.ram_address[0], neg_one),
            (v.ram_addr_c0, neg_w32),
        ],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        load_store(),
        vec![
            (v.rs1[1], one),
            (v.imm[1], one),
            (v.ram_addr_c0, one),
            (v.ram_address[1], neg_one),
        ],
    );

    // 1: RamAddress = 0 if not Load/Store. guard = 1 − Load − Store
    let g_not_ls = || {
        vec![
            (v.const_one, one),
            (v.f_load, neg_one),
            (v.f_store, neg_one),
        ]
    };
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        g_not_ls(),
        vec![(v.ram_address[0], one)],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        g_not_ls(),
        vec![(v.ram_address[1], one)],
    );

    // 2: RamReadValue = RamWriteValue if Load (per-limb)
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_load, one)],
        vec![(v.ram_read[0], one), (v.ram_write[0], neg_one)],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_load, one)],
        vec![(v.ram_read[1], one), (v.ram_write[1], neg_one)],
    );

    // 3: RamReadValue = RdWriteValue if Load (per-limb)
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_load, one)],
        vec![(v.ram_read[0], one), (v.rd_write[0], neg_one)],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_load, one)],
        vec![(v.ram_read[1], one), (v.rd_write[1], neg_one)],
    );

    // 4: Rs2Value = RamWriteValue if Store (per-limb)
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_store, one)],
        vec![(v.rs2[0], one), (v.ram_write[0], neg_one)],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_store, one)],
        vec![(v.rs2[1], one), (v.ram_write[1], neg_one)],
    );

    // 5: LeftLookupOperand = 0 if Add/Sub/Mul (per-limb)
    let g_asm = || vec![(v.f_add, one), (v.f_sub, one), (v.f_mul, one)];
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        g_asm(),
        vec![(v.left_lookup[0], one)],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        g_asm(),
        vec![(v.left_lookup[1], one)],
    );

    // 6: LeftLookupOperand = LeftInstructionInput otherwise (per-limb). guard = 1−Add−Sub−Mul
    let g_not_asm = || {
        vec![
            (v.const_one, one),
            (v.f_add, neg_one),
            (v.f_sub, neg_one),
            (v.f_mul, neg_one),
        ]
    };
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        g_not_asm(),
        vec![(v.left_lookup[0], one), (v.left[0], neg_one)],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        g_not_asm(),
        vec![(v.left_lookup[1], one), (v.left[1], neg_one)],
    );

    // 7: RightLookupOperand = Left + Right if Add (limb-wise; rlo2 = high carry, rlo3 = 0)
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_add, one)],
        vec![
            (v.left[0], one),
            (v.right_mag[0], one),
            (v.rlo[0], neg_one),
            (v.add_c0, neg_w32),
        ],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_add, one)],
        vec![
            (v.left[1], one),
            (v.right_mag[1], one),
            (v.add_c0, one),
            (v.rlo[1], neg_one),
            (v.rlo[2], neg_w32),
        ],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_add, one)],
        vec![(v.rlo[3], one)],
    );

    // 8: RightLookupOperand = Left − Right + 2^64 if Sub, encoded as RLO + Right = Left + 2^64
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_sub, one)],
        vec![
            (v.rlo[0], one),
            (v.right_mag[0], one),
            (v.left[0], neg_one),
            (v.sub_c0, neg_w32),
        ],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_sub, one)],
        vec![
            (v.rlo[1], one),
            (v.right_mag[1], one),
            (v.sub_c0, one),
            (v.left[1], neg_one),
            (v.sub_c1, neg_w32),
        ],
    );
    // limb2: rlo2 + sub_c1 = 1  (the +2^64 bias; Left has 0 at the 2^64 place, no carry out)
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_sub, one)],
        vec![(v.rlo[2], one), (v.sub_c1, one), (v.const_one, neg_one)],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_sub, one)],
        vec![(v.rlo[3], one)],
    );

    // 9: RightLookupOperand = Product if Mul (per-limb; product ≥ 0 for 64-bit MUL)
    for i in 0..4 {
        eq_row(
            &mut a,
            &mut b,
            &mut c,
            vec![(v.f_mul, one)],
            vec![(v.rlo[i], one), (v.product[i], neg_one)],
        );
    }

    // 10: RightLookupOperand = RightInstructionInput otherwise (per-limb; right ≥ 0).
    //     guard = 1 − Add − Sub − Mul − Advice
    let g10 = || {
        vec![
            (v.const_one, one),
            (v.f_add, neg_one),
            (v.f_sub, neg_one),
            (v.f_mul, neg_one),
            (v.f_advice, neg_one),
        ]
    };
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        g10(),
        vec![(v.rlo[0], one), (v.right_mag[0], neg_one)],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        g10(),
        vec![(v.rlo[1], one), (v.right_mag[1], neg_one)],
    );
    eq_row(&mut a, &mut b, &mut c, g10(), vec![(v.rlo[2], one)]);
    eq_row(&mut a, &mut b, &mut c, g10(), vec![(v.rlo[3], one)]);

    // 11: LookupOutput = 1 if Assert (per-limb: lo=1, hi=0)
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_assert, one)],
        vec![(v.lookup_output[0], one), (v.const_one, neg_one)],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_assert, one)],
        vec![(v.lookup_output[1], one)],
    );

    // 12: RdWriteValue = LookupOutput if WriteLookupToRd (per-limb)
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_write_lookup_to_rd, one)],
        vec![(v.rd_write[0], one), (v.lookup_output[0], neg_one)],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_write_lookup_to_rd, one)],
        vec![(v.rd_write[1], one), (v.lookup_output[1], neg_one)],
    );

    // 13: recompose(RdWrite) = UnexpandedPC + 4 − 2·IsCompressed if Jump (single row; result < 2^33 < p)
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_jump, one)],
        vec![
            (v.rd_write[0], one),
            (v.rd_write[1], w32),
            (v.unexpanded_pc, neg_one),
            (v.const_one, neg_four),
            (v.f_is_compressed, two),
        ],
    );

    // 14: NextUnexpandedPC = LookupOutput if ShouldJump (lo equality + hi = 0)
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.should_jump, one)],
        vec![(v.next_unexpanded_pc, one), (v.lookup_output[0], neg_one)],
    );
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.should_jump, one)],
        vec![(v.lookup_output[1], one)],
    );

    // 15: NextUnexpandedPC = UnexpandedPC + Imm if ShouldBranch (recompose; PCs < 2^32)
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.should_branch, one)],
        vec![
            (v.next_unexpanded_pc, one),
            (v.unexpanded_pc, neg_one),
            (v.imm[0], neg_one),
            (v.imm[1], neg_w32),
        ],
    );

    // 16: NextUnexpandedPC = UnexpandedPC + 4 − 4·DoNotUpdate − 2·IsCompressed otherwise.
    //     guard = 1 − ShouldBranch − Jump
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![
            (v.const_one, one),
            (v.should_branch, neg_one),
            (v.f_jump, neg_one),
        ],
        vec![
            (v.next_unexpanded_pc, one),
            (v.unexpanded_pc, neg_one),
            (v.const_one, neg_four),
            (v.f_do_not_update_pc, four),
            (v.f_is_compressed, two),
        ],
    );

    // 17: NextPC = PC + 1 if Virtual − IsLastInSequence
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_virtual, one), (v.f_is_last_in_sequence, neg_one)],
        vec![(v.next_pc, one), (v.pc, neg_one), (v.const_one, neg_one)],
    );

    // 18: DoNotUpdate = 1 if NextIsVirtual − NextIsFirstInSequence
    eq_row(
        &mut a,
        &mut b,
        &mut c,
        vec![
            (v.next_is_virtual, one),
            (v.next_is_first_in_sequence, neg_one),
        ],
        vec![(v.const_one, one), (v.f_do_not_update_pc, neg_one)],
    );

    // 19: Product = Left × Right (4-limb schoolbook on magnitudes) + Left.sign pinned to 0
    push_mul_constraints(&mul_vars(&v), &mut a, &mut b, &mut c);
    prod_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.left_sign, one)],
        vec![(v.const_one, one)],
        Vec::new(),
    );

    // 20: ShouldBranch = recompose(LookupOutput) × Branch
    prod_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.lookup_output[0], one), (v.lookup_output[1], w32)],
        vec![(v.branch, one)],
        vec![(v.should_branch, one)],
    );

    // 21: ShouldJump = Jump × (1 − NextIsNoop)
    prod_row(
        &mut a,
        &mut b,
        &mut c,
        vec![(v.f_jump, one)],
        vec![(v.const_one, one), (v.next_is_noop, neg_one)],
        vec![(v.should_jump, one)],
    );

    debug_assert_eq!(a.len(), NUM_LIMBED_ROWS);
    debug_assert_eq!(NUM_MUL_ROWS, 10);

    ConstraintMatrices::new(NUM_LIMBED_ROWS, num_vars, a, b, c)
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use jolt_field::goldilocks::decompose::i128_to_signed_limbs;
    use jolt_field::goldilocks::Goldilocks;

    const MASK: u64 = 0xFFFF_FFFF;

    fn matrices() -> ConstraintMatrices<Goldilocks> {
        rv64_limbed_constraints::<Goldilocks>()
    }

    fn g(v: u64) -> Goldilocks {
        Goldilocks::from_u64(v)
    }

    /// Fill the always-active product witness (constraint 19 + Left.sign pin) for
    /// `Left × Right` with `Left` unsigned and `Right` the magnitude (`right_sign`).
    fn fill_product(w: &mut [Goldilocks], v: &Vars, left: u64, right_mag: u64, right_sign: bool) {
        let (llo, lhi) = (left & MASK, left >> 32);
        let (rlo, rhi) = (right_mag & MASK, right_mag >> 32);
        let q0 = u128::from(llo) * u128::from(rlo);
        let q1 = u128::from(llo) * u128::from(rhi);
        let q2 = u128::from(lhi) * u128::from(rlo);
        let q3 = u128::from(lhi) * u128::from(rhi);
        let (p0, c0) = (q0 & u128::from(MASK), q0 >> 32);
        let s1 = q1 + q2 + c0;
        let (p1, c1) = (s1 & u128::from(MASK), s1 >> 32);
        let s2 = q3 + c1;
        let (p2, c2) = (s2 & u128::from(MASK), s2 >> 32);
        let p3 = c2;

        w[v.left[0]] = g(llo);
        w[v.left[1]] = g(lhi);
        w[v.left_sign] = g(0);
        w[v.right_mag[0]] = g(rlo);
        w[v.right_mag[1]] = g(rhi);
        w[v.right_sign] = g(u64::from(right_sign));
        let gm = Goldilocks::from_u128;
        w[v.product[0]] = gm(p0);
        w[v.product[1]] = gm(p1);
        w[v.product[2]] = gm(p2);
        w[v.product[3]] = gm(p3);
        w[v.product_sign] = g(u64::from(right_sign)); // left_sign = 0 → product_sign = right_sign
        w[v.q[0]] = gm(q0);
        w[v.q[1]] = gm(q1);
        w[v.q[2]] = gm(q2);
        w[v.q[3]] = gm(q3);
        w[v.mul_c0] = gm(c0);
        w[v.mul_c1] = gm(c1);
        w[v.mul_c2] = gm(c2);
        w[v.sign_prod] = g(0); // left_sign · right_sign = 0
    }

    /// Base witness: const=1, normal (non-branch/jump) PC update `NextUnexpPC = UnexpPC + 4`.
    /// All values zero unless an op sets them. Satisfies constraints 16/19/20/21 and every
    /// guard that is zero for a no-op-like cycle.
    fn base(v: &Vars, num_vars: usize) -> Vec<Goldilocks> {
        let mut w = vec![g(0); num_vars];
        w[v.const_one] = g(1);
        // 16: NextUnexpPC = UnexpPC + 4 (do_not_update = 0, is_compressed = 0)
        w[v.next_unexpanded_pc] = g(4);
        fill_product(&mut w, v, 0, 0, false);
        w
    }

    fn add_limbs(w: &mut [Goldilocks], v: &Vars, left: u64, right: u64) {
        // rlo = left + right (65-bit), add_c0 = low-limb carry
        let sum = u128::from(left) + u128::from(right);
        let llo = left & MASK;
        let rlo = right & MASK;
        let c0 = (llo + rlo) >> 32;
        w[v.add_c0] = g(c0);
        w[v.rlo[0]] = g((sum & u128::from(MASK)) as u64);
        w[v.rlo[1]] = g(((sum >> 32) & u128::from(MASK)) as u64);
        w[v.rlo[2]] = g(((sum >> 64) & u128::from(MASK)) as u64);
        w[v.rlo[3]] = g((sum >> 96) as u64);
    }

    fn sub_limbs(w: &mut [Goldilocks], v: &Vars, left: u64, right: u64) {
        // rlo = left − right + 2^64; carries from RLO + Right = Left + 2^64
        let rlo_val = u128::from(left) + (1u128 << 64) - u128::from(right);
        let r0 = (rlo_val & u128::from(MASK)) as u64;
        let r1 = ((rlo_val >> 32) & u128::from(MASK)) as u64;
        let r2 = ((rlo_val >> 64) & u128::from(MASK)) as u64;
        let r3 = (rlo_val >> 96) as u64;
        w[v.rlo[0]] = g(r0);
        w[v.rlo[1]] = g(r1);
        w[v.rlo[2]] = g(r2);
        w[v.rlo[3]] = g(r3);
        let rmlo = right & MASK;
        let c0 = (u128::from(r0) + u128::from(rmlo)) >> 32;
        w[v.sub_c0] = g(c0 as u64);
        let rmhi = right >> 32;
        let c1 = (u128::from(r1) + u128::from(rmhi) + c0) >> 32;
        w[v.sub_c1] = g(c1 as u64);
    }

    fn add_witness(left: u64, right: u64) -> (Vars, Vec<Goldilocks>) {
        let (v, n) = layout();
        let mut w = base(&v, n);
        w[v.f_add] = g(1);
        fill_product(&mut w, &v, left, right, false);
        add_limbs(&mut w, &v, left, right);
        (v, w)
    }

    fn sub_witness(left: u64, right: u64) -> (Vars, Vec<Goldilocks>) {
        let (v, n) = layout();
        let mut w = base(&v, n);
        w[v.f_sub] = g(1);
        fill_product(&mut w, &v, left, right, false);
        sub_limbs(&mut w, &v, left, right);
        (v, w)
    }

    fn mul_witness(left: u64, right: u64) -> (Vars, Vec<Goldilocks>) {
        let (v, n) = layout();
        let mut w = base(&v, n);
        w[v.f_mul] = g(1);
        fill_product(&mut w, &v, left, right, false);
        // 9: RightLookupOperand = Product (the unsigned 128-bit product)
        w[v.rlo[0]] = w[v.product[0]];
        w[v.rlo[1]] = w[v.product[1]];
        w[v.rlo[2]] = w[v.product[2]];
        w[v.rlo[3]] = w[v.product[3]];
        (v, w)
    }

    fn load_witness(rs1: u64, imm: i64, val: u64) -> (Vars, Vec<Goldilocks>) {
        let (v, n) = layout();
        let mut w = base(&v, n);
        w[v.f_load] = g(1);
        // 0: ram_address = rs1 + imm
        let addr = (i128::from(rs1) + i128::from(imm)) as u128;
        w[v.rs1[0]] = g(rs1 & MASK);
        w[v.rs1[1]] = g(rs1 >> 32);
        let imm_limbs = i128_to_signed_limbs(i128::from(imm));
        w[v.imm[0]] = imm_limbs[0];
        w[v.imm[1]] = imm_limbs[1];
        let addr_lo = (addr & u128::from(MASK)) as u64;
        let addr_hi = ((addr >> 32) & u128::from(MASK)) as u64;
        w[v.ram_address[0]] = g(addr_lo);
        w[v.ram_address[1]] = g(addr_hi);
        let imm_lo = (i128::from(imm)).rem_euclid(1i128 << 32) as u64;
        let c0 = (u128::from(rs1 & MASK) + u128::from(imm_lo)) >> 32;
        w[v.ram_addr_c0] = g(c0 as u64);
        // 2/3: ram_read = ram_write = rd_write = val
        for col in [v.ram_read, v.ram_write, v.rd_write] {
            w[col[0]] = g(val & MASK);
            w[col[1]] = g(val >> 32);
        }
        (v, w)
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    #[test]
    fn layout_and_row_counts() {
        let (_, n) = layout();
        let m = matrices();
        assert_eq!(m.num_vars, n);
        assert_eq!(m.num_constraints, NUM_LIMBED_ROWS);
        assert_eq!(m.a.len(), NUM_LIMBED_ROWS);
    }

    #[test]
    fn noop_satisfies() {
        let (v, n) = layout();
        let mut w = base(&v, n);
        // no-op: do_not_update = 1 ⇒ 16 gives NextUnexpPC = UnexpPC; pin NextUnexpPC = 0
        w[v.f_do_not_update_pc] = g(1);
        w[v.next_unexpanded_pc] = g(0);
        matrices().check_witness(&w).expect("no-op must satisfy");
    }

    #[test]
    fn add_satisfies() {
        let m = matrices();
        let edges = [0u64, 1, MASK, 1 << 32, u64::MAX, u64::MAX - 1];
        for &l in &edges {
            for &r in &edges {
                let (_, w) = add_witness(l, r);
                m.check_witness(&w).expect("honest ADD must satisfy");
            }
        }
        let mut rng = Rng(0x4144_4400_0000_0001);
        for _ in 0..2000 {
            let (_, w) = add_witness(rng.next(), rng.next());
            m.check_witness(&w).expect("random ADD must satisfy");
        }
    }

    #[test]
    fn sub_satisfies() {
        let m = matrices();
        let edges = [0u64, 1, MASK, 1 << 32, u64::MAX, u64::MAX - 1];
        for &l in &edges {
            for &r in &edges {
                let (_, w) = sub_witness(l, r);
                m.check_witness(&w).expect("honest SUB must satisfy");
            }
        }
        let mut rng = Rng(0x5355_4200_0000_0001);
        for _ in 0..2000 {
            let (_, w) = sub_witness(rng.next(), rng.next());
            m.check_witness(&w).expect("random SUB must satisfy");
        }
    }

    #[test]
    fn mul_satisfies() {
        let m = matrices();
        let edges = [0u64, 1, MASK, 1 << 32, u64::MAX, u64::MAX - 1];
        for &l in &edges {
            for &r in &edges {
                let (_, w) = mul_witness(l, r);
                m.check_witness(&w).expect("honest MUL must satisfy");
            }
        }
        let mut rng = Rng(0x4D55_4C00_0000_0001);
        for _ in 0..2000 {
            let (_, w) = mul_witness(rng.next(), rng.next());
            m.check_witness(&w).expect("random MUL must satisfy");
        }
    }

    #[test]
    fn load_satisfies() {
        let m = matrices();
        // positive + negative immediates, exercising constraint 0's signed high limb + carry
        for &(rs1, imm) in &[
            (0x0000_1000u64, 0i64),
            (0x0000_1000, 8),
            (0x0000_1000, -8),
            (0xFFFF_FFFF, 1),
            (0x1_0000_0000, -1),
            (0x0000_8000_0000_0000, 0x100),
        ] {
            let (_, w) = load_witness(rs1, imm, 0xDEAD_BEEF_CAFE_F00D);
            m.check_witness(&w).expect("honest load must satisfy");
        }
    }

    #[test]
    fn tampered_witnesses_rejected() {
        let m = matrices();

        // ADD: corrupt a RightLookupOperand limb
        let (v, mut w) = add_witness(0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321);
        w[v.rlo[1]] += g(1);
        assert!(
            m.check_witness(&w).is_err(),
            "tampered ADD rlo must be rejected"
        );

        // SUB: corrupt the borrow chain
        let (v, mut w) = sub_witness(0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555);
        w[v.sub_c1] += g(1);
        assert!(
            m.check_witness(&w).is_err(),
            "tampered SUB carry must be rejected"
        );

        // MUL: corrupt a product limb (caught by both schoolbook + constraint 9)
        let (v, mut w) = mul_witness(0xDEAD_BEEF, 0xFEED_FACE);
        w[v.product[2]] += g(1);
        assert!(
            m.check_witness(&w).is_err(),
            "tampered MUL product must be rejected"
        );

        // load: corrupt the address
        let (v, mut w) = load_witness(0x4000, 16, 42);
        w[v.ram_address[0]] += g(1);
        assert!(
            m.check_witness(&w).is_err(),
            "tampered load address must be rejected"
        );

        // Left.sign must be pinned to 0
        let (v, mut w) = mul_witness(3, 5);
        w[v.left_sign] = g(1);
        assert!(
            m.check_witness(&w).is_err(),
            "non-zero Left.sign must be rejected"
        );
    }
}
