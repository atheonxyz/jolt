//! Opening accumulator — vendored from jolt-core's `poly/opening_proof.rs` +
//! `zkvm/witness.rs`, retargeted to the lean [`jolt_field::Field`] (opening points are `Vec<F>`;
//! jolt-core's `F::Challenge` collapses to `F`).
//!
//! This is the **claim store**: a map `(polynomial, sumcheck) → (opening_point, claim)` that
//! sumcheck instances read input claims from and write output claims to. The stage-8 batched PCS
//! opening (dedup/aliases/`DoryOpeningState`) and the ZK pending-claim machinery are deferred —
//! they live with the stage driver + WhirScheme opening. Non-ZK only.

use std::collections::HashMap;

use jolt_field::Field;

/// Opening-point endianness tag (`const` generic, as in jolt-core).
pub type Endianness = bool;
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;

/// A sumcheck opening point `r`, endianness-tagged so big/little-endian mixups are caught.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct OpeningPoint<const E: Endianness, F: Field> {
    pub r: Vec<F>,
}

impl<const E: Endianness, F: Field> OpeningPoint<E, F> {
    pub fn new(r: Vec<F>) -> Self {
        Self { r }
    }

    pub fn len(&self) -> usize {
        self.r.len()
    }

    pub fn is_empty(&self) -> bool {
        self.r.is_empty()
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        let (left, right) = self.r.split_at(mid);
        (Self::new(left.to_vec()), Self::new(right.to_vec()))
    }

    /// Reinterpret the point under a (possibly different) endianness, reversing if it changed.
    pub fn match_endianness<const SWAPPED_E: Endianness>(&self) -> OpeningPoint<SWAPPED_E, F> {
        let mut r = self.r.clone();
        if E != SWAPPED_E {
            r.reverse();
        }
        OpeningPoint::<SWAPPED_E, F>::new(r)
    }
}

impl<const E: Endianness, F: Field> std::ops::Index<usize> for OpeningPoint<E, F> {
    type Output = F;
    fn index(&self, index: usize) -> &F {
        &self.r[index]
    }
}

/// Identifies which sumcheck an opening was produced by. Vendored verbatim from jolt-core so the
/// ported subprotocols key openings identically. `#[repr(u8)]` ordering is not relied upon here.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
pub enum SumcheckId {
    SpartanOuter,
    SpartanProductVirtualization,
    SpartanShift,
    InstructionClaimReduction,
    InstructionInputVirtualization,
    InstructionReadRaf,
    InstructionRaVirtualization,
    RamReadWriteChecking,
    RamRafEvaluation,
    RamOutputCheck,
    RamValCheck,
    RamRaClaimReduction,
    RamHammingBooleanity,
    RamRaVirtualization,
    RegistersClaimReduction,
    RegistersReadWriteChecking,
    RegistersValEvaluation,
    BytecodeReadRaf,
    Booleanity,
    AdviceClaimReductionCyclePhase,
    AdviceClaimReduction,
    IncClaimReduction,
    HammingWeightClaimReduction,
}

/// Committed (PCS-opened) polynomials. Vendored from jolt-core `zkvm/witness.rs`.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
pub enum CommittedPolynomial {
    RdInc,
    RamInc,
    InstructionRa(usize),
    BytecodeRa(usize),
    RamRa(usize),
    TrustedAdvice,
    UntrustedAdvice,
}

/// Virtual (derived-during-proving) polynomials. Vendored subset from jolt-core `zkvm/witness.rs`;
/// the flag-carrying variants (`OpFlags`/`InstructionFlags`/`LookupTableFlag`) are added when the
/// Spartan/bytecode ports need them (they require the RISC-V flag enums).
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
pub enum VirtualPolynomial {
    PC,
    UnexpandedPC,
    NextPC,
    NextUnexpandedPC,
    NextIsNoop,
    NextIsVirtual,
    NextIsFirstInSequence,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,
    Product,
    ShouldJump,
    ShouldBranch,
    Rd,
    Imm,
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    LookupOutput,
    InstructionRaf,
    InstructionRafFlag,
    InstructionRa(usize),
    RegistersVal,
    RamAddress,
    RamRa,
    RamReadValue,
    RamWriteValue,
    RamVal,
    RamValInit,
    RamValFinal,
    RamHammingWeight,
    UnivariateSkip,
}

/// Map key: which polynomial an opening belongs to.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
pub enum PolynomialId {
    Committed(CommittedPolynomial),
    Virtual(VirtualPolynomial),
}

/// Read-only opening lookups, implemented by both prover and verifier accumulators. A sumcheck
/// instance's `input_claim` reads prior openings through this; the value is the cached claim.
pub trait OpeningAccumulator<F: Field> {
    fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F);

    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F);
}

/// Shared opening store, used by both the prover (claims it computed) and the verifier (claims it
/// read from the proof). The stage-8 PCS batching layer will wrap this with the committed
/// polynomial handles / commitments.
#[derive(Clone, Debug)]
pub struct Openings<F: Field> {
    map: HashMap<(PolynomialId, SumcheckId), (OpeningPoint<BIG_ENDIAN, F>, F)>,
    pub log_t: usize,
}

impl<F: Field> Openings<F> {
    pub fn new(log_t: usize) -> Self {
        Self {
            map: HashMap::new(),
            log_t,
        }
    }

    /// Store a committed-polynomial opening `(point, claim)` produced by `sumcheck`.
    pub fn append_dense(
        &mut self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
        point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        let _ = self.map.insert(
            (PolynomialId::Committed(polynomial), sumcheck),
            (point, claim),
        );
    }

    /// Store a virtual-polynomial opening `(point, claim)` produced by `sumcheck`.
    pub fn append_virtual(
        &mut self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        let _ = self.map.insert(
            (PolynomialId::Virtual(polynomial), sumcheck),
            (point, claim),
        );
    }

    #[expect(
        clippy::panic,
        reason = "a missing opening is a prover/verifier wiring bug, not recoverable input"
    )]
    fn get(&self, key: (PolynomialId, SumcheckId)) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        match self.map.get(&key) {
            Some((point, claim)) => (point.clone(), *claim),
            None => panic!("opening not found in accumulator: {key:?}"),
        }
    }
}

impl<F: Field> OpeningAccumulator<F> for Openings<F> {
    fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        self.get((PolynomialId::Committed(polynomial), sumcheck))
    }

    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        self.get((PolynomialId::Virtual(polynomial), sumcheck))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::goldilocks::GoldilocksFp3 as F;

    fn pt(vals: &[u64]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(vals.iter().map(|&v| F::from_u64(v)).collect())
    }

    #[test]
    fn append_and_get_round_trip() {
        let mut acc = Openings::<F>::new(8);
        acc.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            pt(&[1, 2, 3]),
            F::from_u64(42),
        );
        acc.append_virtual(
            VirtualPolynomial::Product,
            SumcheckId::SpartanOuter,
            pt(&[9]),
            F::from_u64(7),
        );

        let (p, c) = acc.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        assert_eq!(p, pt(&[1, 2, 3]));
        assert_eq!(c, F::from_u64(42));

        let (p, c) = acc
            .get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter);
        assert_eq!(p, pt(&[9]));
        assert_eq!(c, F::from_u64(7));
    }

    #[test]
    fn keys_are_distinct_per_polynomial_and_sumcheck() {
        let mut acc = Openings::<F>::new(8);
        acc.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            pt(&[1]),
            F::from_u64(10),
        );
        acc.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValCheck,
            pt(&[2]),
            F::from_u64(20),
        );
        acc.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::RamReadWriteChecking,
            pt(&[3]),
            F::from_u64(30),
        );

        assert_eq!(
            acc.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::RamReadWriteChecking
            )
            .1,
            F::from_u64(10)
        );
        assert_eq!(
            acc.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::RamValCheck
            )
            .1,
            F::from_u64(20)
        );
        assert_eq!(
            acc.get_committed_polynomial_opening(
                CommittedPolynomial::RdInc,
                SumcheckId::RamReadWriteChecking
            )
            .1,
            F::from_u64(30)
        );
    }

    #[test]
    fn match_endianness_reverses_only_on_change() {
        let p = pt(&[1, 2, 3]);
        let same = p.match_endianness::<BIG_ENDIAN>();
        assert_eq!(same.r, p.r);
        let swapped = p.match_endianness::<LITTLE_ENDIAN>();
        assert_eq!(
            swapped.r,
            vec![F::from_u64(3), F::from_u64(2), F::from_u64(1)]
        );
    }
}
