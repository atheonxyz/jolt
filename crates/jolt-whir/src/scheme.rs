//! `WhirScheme` — the Jolt-facing WHIR PCS over `Basefield<Field64_3>` (commit base
//! `Field64` @ 8 B, fold/open in `Fp3`), driven through the **shared** spongefish
//! transcript ([`ProverTranscript`]/[`VerifierTranscript`]).
//!
//! Per the Phase-2 plan (Option B), this is an **inherent** API rather than an
//! impl of `jolt_openings::CommitmentScheme`: that trait's `open`/`verify` pin
//! `Transcript<Challenge = Self::Field>`, and no spongefish-backed type implements
//! `jolt_transcript::Transcript` (it requires `Clone`/`'static`, which a duplex
//! sponge cannot satisfy — see [`crate::transcript`]). The method *shapes* still
//! mirror `CommitmentScheme` so promotion is mechanical if that trait is later
//! relaxed.
//!
//! Shared-transcript flow: `commit` absorbs the Merkle root + OOD samples into the
//! sponge and returns the reusable [`WhirHint`]; `open` drives WHIR's batched
//! `prove` into the *same* sponge (so the opening bytes live in the one proof);
//! the verifier replays via `receive_commitment` + `verify`, then completes the
//! deferred MLE check with [`FinalClaim::verify`](whir::protocols::whir::FinalClaim).

use std::borrow::Cow;

use whir::algebra::embedding::Basefield;
use whir::algebra::fields::Field64_3;
use whir::algebra::linear_form::{Evaluate, LinearForm, MultilinearExtension};
use whir::protocols::whir::{Commitment, Config, Witness};

use jolt_field::goldilocks::{Goldilocks, GoldilocksFp3};

use crate::convert::{column_to_field64, from_field64_3, to_field64_3};
use crate::params::whir_params;
use crate::transcript::{ProverTranscript, VerifierTranscript};

/// WHIR config specialized to the Goldilocks base-commit embedding.
pub type WhirConfig = Config<Basefield<Field64_3>>;
/// A WHIR commitment (Merkle root + out-of-domain samples), reconstructed by the
/// verifier from the shared transcript.
pub type WhirCommitment = Commitment<Field64_3>;
/// Reusable per-commit witness (RS codeword + Merkle tree + OOD) — the analogue of
/// Dory's row commitments; fed back into [`WhirScheme::open`].
pub type WhirHint = Witness<Field64_3, Basefield<Field64_3>>;

/// Verification failure for the WHIR PCS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhirError {
    /// A WHIR opening / final-claim check rejected.
    VerificationFailed,
}

/// The Jolt WHIR PCS (stateless; `Config` is the setup).
pub struct WhirScheme;

impl WhirScheme {
    /// Build the (transparent) WHIR config for a committed-column length `size`
    /// (a power of two). There is no trusted setup.
    #[must_use]
    pub fn config(size: usize) -> WhirConfig {
        WhirConfig::new(size, &whir_params())
    }

    /// Evaluate a base-Goldilocks `column` as a multilinear extension at an `Fp3`
    /// `point`, using WHIR's own embedding so the value matches what `open`/`verify`
    /// prove. (For the real prover this comes from the sumcheck instead.)
    #[must_use]
    pub fn evaluate(
        config: &WhirConfig,
        column: &[Goldilocks],
        point: &[GoldilocksFp3],
    ) -> GoldilocksFp3 {
        let vector = column_to_field64(column);
        let pt: Vec<Field64_3> = point.iter().copied().map(to_field64_3).collect();
        from_field64_3(MultilinearExtension { point: pt }.evaluate(config.embedding(), &vector))
    }

    /// Commit a base-Goldilocks column, absorbing its Merkle root + OOD samples
    /// into the shared transcript. Returns the reusable [`WhirHint`].
    pub fn commit(
        transcript: &mut ProverTranscript,
        config: &WhirConfig,
        column: &[Goldilocks],
    ) -> WhirHint {
        let vector = column_to_field64(column);
        config.commit(transcript.state_mut(), &[vector.as_slice()])
    }

    /// Open a previously committed `column` at `point`, proving `column(point) =
    /// eval`. Drives WHIR's `prove` into the shared transcript (the opening bytes
    /// land in the single proof produced by [`ProverTranscript::into_proof`]).
    pub fn open(
        transcript: &mut ProverTranscript,
        config: &WhirConfig,
        column: &[Goldilocks],
        hint: WhirHint,
        point: &[GoldilocksFp3],
        eval: GoldilocksFp3,
    ) {
        let vector = column_to_field64(column);
        let pt: Vec<Field64_3> = point.iter().copied().map(to_field64_3).collect();
        let forms: Vec<Box<dyn LinearForm<Field64_3>>> =
            vec![Box::new(MultilinearExtension { point: pt })];
        let _ = config.prove(
            transcript.state_mut(),
            vec![Cow::from(vector)],
            vec![Cow::Owned(hint)],
            forms,
            Cow::Owned(vec![to_field64_3(eval)]),
        );
    }

    /// Verifier: reconstruct a commitment from the shared transcript (in the same
    /// order the prover committed).
    pub fn receive_commitment(
        transcript: &mut VerifierTranscript,
        config: &WhirConfig,
    ) -> Result<WhirCommitment, WhirError> {
        config
            .receive_commitment(transcript.state_mut())
            .map_err(|_| WhirError::VerificationFailed)
    }

    /// Verifier: check the opening of `commitment` at `point` equals `eval`,
    /// completing the deferred MLE check.
    pub fn verify(
        transcript: &mut VerifierTranscript,
        config: &WhirConfig,
        commitment: &WhirCommitment,
        point: &[GoldilocksFp3],
        eval: GoldilocksFp3,
    ) -> Result<(), WhirError> {
        let pt: Vec<Field64_3> = point.iter().copied().map(to_field64_3).collect();
        let final_claim = config
            .verify(transcript.state_mut(), &[commitment], &[to_field64_3(eval)])
            .map_err(|_| WhirError::VerificationFailed)?;
        let form = MultilinearExtension { point: pt };
        final_claim
            .verify([&form as &dyn LinearForm<Field64_3>])
            .map_err(|_| WhirError::VerificationFailed)
    }

    /// Batch-open `columns` (all of the same committed length / one size class)
    /// at `points` via WHIR's native two-level geometric RLC — a single `prove`
    /// over N vectors × M forms that shares the sumcheck rounds and the final
    /// codeword. `evals` is the M×N matrix in **form-major** order:
    /// `evals[f * columns.len() + v] = columns[v](points[f])`.
    ///
    /// Cross-size-class batching is the caller's job: invoke `open_batch` once per
    /// distinct column length on the *same* [`ProverTranscript`] (the shared
    /// sponge keeps the classes in lockstep), with the verifier mirroring the
    /// order via [`verify_batch`](Self::verify_batch).
    pub fn open_batch(
        transcript: &mut ProverTranscript,
        config: &WhirConfig,
        columns: &[&[Goldilocks]],
        hints: Vec<WhirHint>,
        points: &[Vec<GoldilocksFp3>],
        evals: &[GoldilocksFp3],
    ) {
        assert_eq!(hints.len(), columns.len(), "one hint per committed column");
        assert_eq!(
            evals.len(),
            columns.len() * points.len(),
            "evals must be the M×N (form-major) evaluation matrix"
        );
        let vectors = columns
            .iter()
            .map(|c| Cow::from(column_to_field64(c)))
            .collect::<Vec<_>>();
        let forms: Vec<Box<dyn LinearForm<Field64_3>>> = points
            .iter()
            .map(|p| {
                let pt: Vec<Field64_3> = p.iter().copied().map(to_field64_3).collect();
                Box::new(MultilinearExtension { point: pt }) as Box<dyn LinearForm<Field64_3>>
            })
            .collect();
        let ev = evals.iter().copied().map(to_field64_3).collect::<Vec<_>>();
        let _ = config.prove(
            transcript.state_mut(),
            vectors,
            hints.into_iter().map(Cow::Owned).collect(),
            forms,
            Cow::from(ev),
        );
    }

    /// Verifier counterpart of [`open_batch`](Self::open_batch): check the N
    /// `commitments` open to the M×N `evals` matrix at `points`. `commitments`,
    /// `points`, and `evals` must use the identical order/layout as the prover.
    pub fn verify_batch(
        transcript: &mut VerifierTranscript,
        config: &WhirConfig,
        commitments: &[&WhirCommitment],
        points: &[Vec<GoldilocksFp3>],
        evals: &[GoldilocksFp3],
    ) -> Result<(), WhirError> {
        let ev = evals.iter().copied().map(to_field64_3).collect::<Vec<_>>();
        let final_claim = config
            .verify(transcript.state_mut(), commitments, &ev)
            .map_err(|_| WhirError::VerificationFailed)?;
        let forms: Vec<MultilinearExtension<Field64_3>> = points
            .iter()
            .map(|p| MultilinearExtension {
                point: p.iter().copied().map(to_field64_3).collect(),
            })
            .collect();
        final_claim
            .verify(forms.iter().map(|f| f as &dyn LinearForm<Field64_3>))
            .map_err(|_| WhirError::VerificationFailed)
    }
}
