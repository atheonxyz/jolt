//! Concrete shared spongefish transcript for the Goldilocks+WHIR prover/verifier.
//!
//! Per the Phase-2 plan (Option B), the new Goldilocks prover uses this transcript
//! **concretely** and does **not** implement `jolt_transcript::Transcript`: that
//! trait requires `Clone + Default + 'static`, which a spongefish duplex state
//! cannot satisfy (WHIR's `ProverState` is non-`Clone`; its `VerifierState<'a>`
//! borrows the proof, so it is not `'static`). The same underlying
//! `whir::transcript::{ProverState, VerifierState}` is handed to WHIR's
//! commit/open, so Jolt's sumcheck rounds and WHIR's PCS steps share ONE sponge.
//!
//! Challenges are drawn as `Fp3` through WHIR's own `verifier_message::<Field64_3>`
//! decoder + [`from_field64_3`](crate::convert::from_field64_3), so the prover,
//! the verifier, and WHIR agree on every challenge byte-for-byte.

use whir::algebra::fields::Field64_3;
use whir::transcript::codecs::Empty;
use whir::transcript::{DomainSeparator, Proof, ProverState, VerifierMessage, VerifierState};

use jolt_field::goldilocks::GoldilocksFp3;

use crate::convert::{from_field64_3, to_field64_3};

/// Fiat-Shamir protocol-id seed shared by prover and verifier.
///
/// Phase 2 (plan M2) will additionally bind the WHIR `Config` hash here so the
/// field and protocol parameters are committed into Fiat-Shamir.
const PROTOCOL: &str = "jolt-whir/goldilocks";

/// Prover-side shared transcript: owns the spongefish [`ProverState`] that WHIR
/// commit/open also drive.
pub struct ProverTranscript {
    state: ProverState,
}

impl ProverTranscript {
    /// New transcript seeded with the shared protocol id and a per-proof `session`.
    pub fn new(session: &str) -> Self {
        let ds = DomainSeparator::protocol(&PROTOCOL)
            .session(&session)
            .instance(&Empty);
        Self {
            state: ProverState::new_std(&ds),
        }
    }

    /// Absorb an `Fp3` prover message (e.g. a sumcheck round-polynomial coeff).
    #[inline]
    pub fn observe_ext(&mut self, v: GoldilocksFp3) {
        self.state.prover_message(&to_field64_3(v));
    }

    /// Squeeze an `Fp3` Fiat-Shamir challenge.
    #[inline]
    pub fn challenge_fp3(&mut self) -> GoldilocksFp3 {
        from_field64_3(self.state.verifier_message::<Field64_3>())
    }

    /// Borrow the underlying spongefish state so WHIR commit/open run on the same
    /// sponge as Jolt's sumcheck rounds.
    #[inline]
    pub fn state_mut(&mut self) -> &mut ProverState {
        &mut self.state
    }

    /// Finalize into the WHIR proof bytes (narg string + out-of-band hints).
    #[inline]
    pub fn into_proof(self) -> Proof {
        self.state.proof()
    }
}

/// Verifier-side shared transcript: wraps the borrowing spongefish [`VerifierState`].
pub struct VerifierTranscript<'a> {
    state: VerifierState<'a>,
}

impl<'a> VerifierTranscript<'a> {
    /// New verifier transcript over `proof`, seeded identically to the prover.
    pub fn new(session: &str, proof: &'a Proof) -> Self {
        let ds = DomainSeparator::protocol(&PROTOCOL)
            .session(&session)
            .instance(&Empty);
        Self {
            state: VerifierState::new_std(&ds, proof),
        }
    }

    /// Squeeze an `Fp3` Fiat-Shamir challenge (mirrors the prover).
    #[inline]
    pub fn challenge_fp3(&mut self) -> GoldilocksFp3 {
        from_field64_3(self.state.verifier_message::<Field64_3>())
    }

    /// Borrow the underlying spongefish verifier state for WHIR verify.
    #[inline]
    pub fn state_mut(&mut self) -> &mut VerifierState<'a> {
        &mut self.state
    }
}
