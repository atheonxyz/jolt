//! The squeeze + round-polynomial transcript seam for the single spongefish NARG.
//!
//! Both `ProverTranscript` and `VerifierTranscript` draw identical Fiat-Shamir
//! challenges, so the squeeze-only [`Challenge`] supertrait covers params
//! constructors uniformly on both sides. The round-polynomial direction splits:
//! the prover writes coeffs into the NARG ([`ProverFs::observe`]); the verifier
//! reads them back ([`VerifierFs::read_coeffs`]). The concrete impls exist only at
//! `F = GoldilocksFp3` (the prover field), so a `prove`/`verify` over these traits
//! is necessarily Fp3 and its round polys ride the same sponge WHIR commit/open
//! drive — one NARG for the whole proof. Keeping the framework generic over
//! `Fld: Field` (rather than hard-coding `F`) preserves `Fld::zero()/one()`
//! resolution via the bound and lets the math leaf-instances stay field-generic.

use jolt_field::Field;

use crate::field::{ProverTranscript, VerifierTranscript, F};

/// Squeeze Fiat-Shamir challenges. Identical behaviour on the prover and verifier
/// sides, so a params constructor that only draws challenges takes `&mut impl Challenge<F>`.
pub trait Challenge<Fld: Field> {
    /// Squeeze one challenge.
    fn challenge(&mut self) -> Fld;

    /// Squeeze `n` challenges.
    fn challenge_vector(&mut self, n: usize) -> Vec<Fld> {
        (0..n).map(|_| self.challenge()).collect()
    }
}

/// Prover side: writes field elements (sumcheck round-polynomial coeffs) into the NARG.
pub trait ProverFs<Fld: Field>: Challenge<Fld> {
    /// Append one field element as a prover message.
    fn observe(&mut self, v: Fld);
}

/// Verifier side: reads `n` consecutive prover messages back out of the NARG.
/// Returns `None` if the NARG is exhausted or a message fails to decode (malformed proof).
pub trait VerifierFs<Fld: Field>: Challenge<Fld> {
    /// Read `n` prover messages.
    fn read_coeffs(&mut self, n: usize) -> Option<Vec<Fld>>;
}

impl Challenge<F> for ProverTranscript {
    #[inline]
    fn challenge(&mut self) -> F {
        self.challenge_fp3()
    }
}

impl ProverFs<F> for ProverTranscript {
    #[inline]
    fn observe(&mut self, v: F) {
        self.observe_ext(v);
    }
}

impl Challenge<F> for VerifierTranscript<'_> {
    #[inline]
    fn challenge(&mut self) -> F {
        self.challenge_fp3()
    }
}

impl VerifierFs<F> for VerifierTranscript<'_> {
    #[inline]
    fn read_coeffs(&mut self, n: usize) -> Option<Vec<F>> {
        self.read_exts(n).ok()
    }
}
