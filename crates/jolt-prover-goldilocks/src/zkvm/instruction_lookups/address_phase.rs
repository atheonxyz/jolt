//! Address-phase prefix/suffix machinery for the instruction read-raf (P1).
//!
//! Faithfully ported from `jolt-kernels/src/stage5.rs` — the field-generic instruction read-raf
//! (`InstructionReadRafStage5State` + `InstructionReadRafAddressPhase` + the leaf helpers). On the
//! `refactor/crates` branch jolt-kernels (not the deleted jolt-core) is the read-only port oracle;
//! its `T: jolt_transcript::Transcript` bound is incompatible with the WHIR spongefish, so we port
//! the *math* into the goldilocks framework rather than call it.
//!
//! At production `XLEN = 64` the lookup index `k` is `LOG_K = 2·XLEN = 128` bits — the interleave of
//! two 64-bit operands. The dense `Val(k)` table is infeasible there; instead `Val(k)` is the
//! prefix/suffix decomposition `Σ_i prefix_i(k_high)·suffix_i(k_low)` (the `jolt-lookup-tables`
//! `LookupTableKind::combine`). This module binds the `LOG_K` address bits **HighToLow** in
//! `CHUNK_BITS`-wide phases: each phase rebuilds the per-table prefix/suffix dense polynomials from
//! the accumulated checkpoints, binds them, then folds the bound scalars into the next checkpoint
//! and restricts each lookup group's weight by `eq(chunk_challenges, chunk_of(index))`.
//!
//! This is the address half of the read-raf sumcheck; the cycle phase, hand-off, the
//! [`crate::framework::sumcheck::SumcheckInstance`] wrapper, and the verifier `expected_output_claim`
//! land in P2. **Operand-side caveat for P2:** stage5's `operand_polynomial_eval` maps `Left → offset
//! 0` whereas IL-1 [`super::OperandPolynomial`] maps `Left → offset 1`; the address phase below is a
//! byte-faithful port of stage5 (left = `uninterleave().0`), so the verifier operand reconstruction
//! in P2 must reconcile against this convention, not assume IL-1's.

use jolt_field::Field;
use jolt_lookup_tables::{
    tables::{
        prefixes::{PrefixEval, ALL_PREFIXES, NUM_PREFIXES},
        Suffixes,
    },
    uninterleave_bits, LookupBits, LookupTableKind,
};
use jolt_poly::{bind_high_to_low, UnivariatePoly};

/// Address bits bound per phase (the prefix-checkpoint granularity). `LOG_K` must be a multiple.
const ADDRESS_CHUNK_BITS: usize = 8;
/// Maximum number of suffixes any single lookup table decomposes into.
const MAX_SUFFIXES: usize = 4;

/// Per-row `(prefix evals @ 0, prefix evals @ 2)`. Mirrors jolt-kernels `PrefixPairEvals`.
type PrefixPairEvals<F> = ([PrefixEval<F>; NUM_PREFIXES], [PrefixEval<F>; NUM_PREFIXES]);

/// A distinct `(lookup_index, table, interleaved)` triple, with its summed cycle-eq weight. Mirrors
/// jolt-kernels `InstructionReadRafLookupGroup`. `phase_u_eval_sum` is restricted at each phase
/// boundary to the address bits bound so far; it starts equal to `u_eval_sum`.
#[derive(Clone, Debug)]
pub struct LookupGroup<F: Field> {
    pub lookup_index: u128,
    pub lookup_table_index: Option<usize>,
    pub is_interleaved_operands: bool,
    pub u_eval_sum: F,
    pub phase_u_eval_sum: F,
}

/// Deduplicate per-cycle lookups into [`LookupGroup`]s keyed by `(index, table, interleaved)`,
/// accumulating each cycle's eq weight `u_evals[cycle]`. Returns the groups, the per-cycle group
/// index (so the cycle phase can rebuild cycle weights), and the per-table group-index lists.
/// Mirrors jolt-kernels `instruction_read_raf_lookup_groups`.
pub fn lookup_groups_from_trace<const XLEN: usize, F: Field>(
    lookup_indices: &[u128],
    lookup_table_indices: &[Option<usize>],
    is_interleaved_operands: &[bool],
    u_evals: &[F],
) -> (Vec<LookupGroup<F>>, Vec<usize>, Vec<Vec<usize>>) {
    let trace_len = lookup_indices.len();
    debug_assert_eq!(lookup_table_indices.len(), trace_len);
    debug_assert_eq!(is_interleaved_operands.len(), trace_len);
    debug_assert!(u_evals.len() >= trace_len);

    let table_count = LookupTableKind::<XLEN>::all().len();
    let mut index_by_key: std::collections::HashMap<(u128, Option<usize>, bool), usize> =
        std::collections::HashMap::with_capacity(trace_len);
    let mut groups: Vec<LookupGroup<F>> = Vec::new();
    let mut group_indices_by_cycle = Vec::with_capacity(trace_len);

    for cycle in 0..trace_len {
        let key = (
            lookup_indices[cycle],
            lookup_table_indices[cycle],
            is_interleaved_operands[cycle],
        );
        let u_eval = u_evals[cycle];
        if let Some(&group_index) = index_by_key.get(&key) {
            groups[group_index].u_eval_sum += u_eval;
            groups[group_index].phase_u_eval_sum += u_eval;
            group_indices_by_cycle.push(group_index);
        } else {
            let group_index = groups.len();
            let _ = index_by_key.insert(key, group_index);
            groups.push(LookupGroup {
                lookup_index: key.0,
                lookup_table_index: key.1,
                is_interleaved_operands: key.2,
                u_eval_sum: u_eval,
                phase_u_eval_sum: u_eval,
            });
            group_indices_by_cycle.push(group_index);
        }
    }

    let mut groups_by_table = vec![Vec::new(); table_count];
    for (group_index, group) in groups.iter().enumerate() {
        if let Some(table_index) = group.lookup_table_index {
            groups_by_table[table_index].push(group_index);
        }
    }

    (groups, group_indices_by_cycle, groups_by_table)
}

/// Per-table read suffix polynomials for one phase. Mirrors `InstructionReadRafReadTablePhase`.
struct ReadTablePhase<F: Field, const XLEN: usize> {
    table: LookupTableKind<XLEN>,
    suffix_polys: Vec<Vec<F>>,
}

/// The dense polynomials for one `CHUNK_BITS`-wide address phase. Mirrors
/// `InstructionReadRafAddressPhase`. All vectors have length `2^CHUNK_BITS` at phase start and bind
/// HighToLow down to length 1.
struct AddressPhaseChunk<F: Field, const XLEN: usize> {
    phase: usize,
    left_operand_prefix: Vec<F>,
    right_operand_prefix: Vec<F>,
    identity_prefix: Vec<F>,
    raf_shift_half_q: Vec<F>,
    raf_left_q: Vec<F>,
    raf_right_q: Vec<F>,
    raf_shift_full_q: Vec<F>,
    raf_identity_q: Vec<F>,
    read_prefix_polys: Vec<Vec<F>>,
    read_suffix_polys: Vec<ReadTablePhase<F, XLEN>>,
}

/// Address-phase state machine for the instruction read-raf. Drives the `LOG_K = 2·XLEN` address
/// rounds of the read-raf sumcheck: `round_poly` produces the degree-2 message, `bind` ingests the
/// challenge and advances phases. The cycle phase / hand-off / opening cache are added in P2.
pub struct InstructionAddressPhase<F: Field, const XLEN: usize> {
    gamma: F,
    gamma2: F,
    active_scale: F,
    round: usize,
    address_challenges: Vec<F>,
    groups: Vec<LookupGroup<F>>,
    groups_by_table: Vec<Vec<usize>>,
    left_operand_checkpoint: F,
    right_operand_checkpoint: F,
    identity_checkpoint: F,
    read_prefix_checkpoints: Vec<PrefixEval<F>>,
    chunk: Option<AddressPhaseChunk<F, XLEN>>,
}

impl<F: Field, const XLEN: usize> InstructionAddressPhase<F, XLEN> {
    /// Total address bits = the interleaved-index width.
    pub const LOG_K: usize = 2 * XLEN;
    const CHUNK_BITS: usize = ADDRESS_CHUNK_BITS;

    pub fn new(
        groups: Vec<LookupGroup<F>>,
        groups_by_table: Vec<Vec<usize>>,
        gamma: F,
        active_scale: F,
    ) -> Self {
        debug_assert!(Self::LOG_K.is_multiple_of(Self::CHUNK_BITS));
        Self {
            gamma,
            gamma2: gamma * gamma,
            active_scale,
            round: 0,
            address_challenges: Vec::with_capacity(Self::LOG_K),
            groups,
            groups_by_table,
            left_operand_checkpoint: F::zero(),
            right_operand_checkpoint: F::zero(),
            identity_checkpoint: F::zero(),
            read_prefix_checkpoints: ALL_PREFIXES
                .iter()
                .map(|prefix| prefix.default_checkpoint::<F>())
                .collect(),
            chunk: None,
        }
    }

    pub fn num_address_rounds(&self) -> usize {
        Self::LOG_K
    }

    pub fn address_challenges(&self) -> &[F] {
        &self.address_challenges
    }

    /// The degree-2 address-round message: `(read + γ·left + γ²·(right + identity)) · active_scale`,
    /// expressed via the eval-at-{0,2}-plus-hint form (mirrors `round_poly`'s address branch).
    #[expect(
        clippy::expect_used,
        reason = "ensure_chunk just populated self.chunk; the invariant cannot be expressed to the borrow checker"
    )]
    pub fn round_poly(&mut self, previous_claim: F) -> UnivariatePoly<F> {
        debug_assert!(self.round < Self::LOG_K);
        self.ensure_chunk();
        let chunk = self
            .chunk
            .as_ref()
            .expect("address chunk built by ensure_chunk");
        let read = chunk.read_table_round_evals();
        let raf = chunk.raf_round_component_evals();
        let eval_at_0 = (read[0] + self.gamma * raf[0][0] + self.gamma2 * (raf[1][0] + raf[2][0]))
            * self.active_scale;
        let eval_at_2 = (read[1] + self.gamma * raf[0][1] + self.gamma2 * (raf[1][1] + raf[2][1]))
            * self.active_scale;
        UnivariatePoly::from_evals_and_hint(previous_claim, &[eval_at_0, eval_at_2])
    }

    pub fn bind(&mut self, challenge: F) {
        debug_assert!(self.round < Self::LOG_K);
        self.ensure_chunk();
        self.address_challenges.push(challenge);
        if let Some(chunk) = &mut self.chunk {
            chunk.bind(challenge);
        }
        if (self.round + 1).is_multiple_of(Self::CHUNK_BITS) {
            self.finish_chunk();
        }
        self.round += 1;
    }

    fn ensure_chunk(&mut self) {
        let phase = self.round / Self::CHUNK_BITS;
        if self
            .chunk
            .as_ref()
            .is_some_and(|chunk| chunk.phase == phase)
        {
            return;
        }
        self.chunk = Some(self.build_chunk(phase));
    }

    fn build_chunk(&self, phase: usize) -> AddressPhaseChunk<F, XLEN> {
        let chunk_bits = Self::CHUNK_BITS;
        let poly_len = 1usize << chunk_bits;
        let suffix_len = Self::LOG_K - (phase + 1) * chunk_bits;

        let left_operand_prefix =
            operand_prefix_poly(self.left_operand_checkpoint, chunk_bits, true);
        let right_operand_prefix =
            operand_prefix_poly(self.right_operand_checkpoint, chunk_bits, false);
        let identity_prefix = identity_prefix_poly(self.identity_checkpoint, chunk_bits);

        let read_prefix_polys = ALL_PREFIXES
            .iter()
            .map(|prefix| {
                (0..poly_len)
                    .map(|bits| {
                        prefix
                            .evaluate(
                                &self.read_prefix_checkpoints,
                                LookupBits::new(bits as u128, chunk_bits),
                                suffix_len,
                            )
                            .into_inner()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let shift_half_value = 1u128 << (suffix_len / 2);
        let shift_full_value = 1u128 << suffix_len;
        let shift_half = F::from_u128(shift_half_value);
        let shift_full = F::from_u128(shift_full_value);
        let suffix_mask = if suffix_len == 128 {
            u128::MAX
        } else {
            (1u128 << suffix_len) - 1
        };

        let mut raf_shift_half_q = vec![F::zero(); poly_len];
        let mut raf_left_q = vec![F::zero(); poly_len];
        let mut raf_right_q = vec![F::zero(); poly_len];
        let mut raf_shift_full_q = vec![F::zero(); poly_len];
        let mut raf_identity_q = vec![F::zero(); poly_len];
        for group in &self.groups {
            let index = ((group.lookup_index >> suffix_len) as usize) & (poly_len - 1);
            let suffix_bits = group.lookup_index & suffix_mask;
            let weight = group.phase_u_eval_sum;
            if group.is_interleaved_operands {
                raf_shift_half_q[index] += weight;
                let (left_suffix, right_suffix) = uninterleave_bits(suffix_bits);
                if left_suffix != 0 {
                    raf_left_q[index] += weight * F::from_u64(left_suffix);
                }
                if right_suffix != 0 {
                    raf_right_q[index] += weight * F::from_u64(right_suffix);
                }
            } else {
                raf_shift_full_q[index] += weight;
                if suffix_bits != 0 {
                    raf_identity_q[index] += weight * F::from_u128(suffix_bits);
                }
            }
        }
        if shift_half_value != 1 {
            for value in &mut raf_shift_half_q {
                *value *= shift_half;
            }
        }
        if shift_full_value != 1 {
            for value in &mut raf_shift_full_q {
                *value *= shift_full;
            }
        }

        let tables = LookupTableKind::<XLEN>::all();
        let mut read_suffix_polys = Vec::new();
        for (table_index, table) in tables.iter().enumerate() {
            if self.groups_by_table[table_index].is_empty() {
                continue;
            }
            let suffixes = table.suffixes();
            let suffix_count = suffixes.len();
            let mut one_suffix = None;
            let mut boolean_suffixes = [0usize; MAX_SUFFIXES];
            let mut boolean_suffix_count = 0usize;
            let mut valued_suffixes = [0usize; MAX_SUFFIXES];
            let mut valued_suffix_count = 0usize;
            for (suffix_index, suffix) in suffixes.iter().enumerate() {
                if matches!(suffix, Suffixes::One) {
                    one_suffix = Some(suffix_index);
                } else if suffix.is_01_valued() {
                    boolean_suffixes[boolean_suffix_count] = suffix_index;
                    boolean_suffix_count += 1;
                } else {
                    valued_suffixes[valued_suffix_count] = suffix_index;
                    valued_suffix_count += 1;
                }
            }

            let mut suffix_polys = vec![vec![F::zero(); poly_len]; suffix_count];
            for &group_index in &self.groups_by_table[table_index] {
                let group = &self.groups[group_index];
                let index = ((group.lookup_index >> suffix_len) as usize) & (poly_len - 1);
                let suffix_bits = LookupBits::new(group.lookup_index & suffix_mask, suffix_len);
                let weight = group.phase_u_eval_sum;
                if let Some(suffix_index) = one_suffix {
                    suffix_polys[suffix_index][index] += weight;
                }
                for &suffix_index in boolean_suffixes.iter().take(boolean_suffix_count) {
                    let suffix_value: u64 = suffixes[suffix_index].suffix_mle(suffix_bits);
                    debug_assert!(suffix_value == 0 || suffix_value == 1);
                    if suffix_value == 1 {
                        suffix_polys[suffix_index][index] += weight;
                    }
                }
                for &suffix_index in valued_suffixes.iter().take(valued_suffix_count) {
                    let suffix_value: u64 = suffixes[suffix_index].suffix_mle(suffix_bits);
                    if suffix_value != 0 {
                        suffix_polys[suffix_index][index] += weight * F::from_u64(suffix_value);
                    }
                }
            }
            read_suffix_polys.push(ReadTablePhase {
                table: *table,
                suffix_polys,
            });
        }

        AddressPhaseChunk {
            phase,
            left_operand_prefix,
            right_operand_prefix,
            identity_prefix,
            raf_shift_half_q,
            raf_left_q,
            raf_right_q,
            raf_shift_full_q,
            raf_identity_q,
            read_prefix_polys,
            read_suffix_polys,
        }
    }

    fn finish_chunk(&mut self) {
        let Some(chunk) = self.chunk.take() else {
            return;
        };
        self.left_operand_checkpoint = chunk.left_operand_prefix[0];
        self.right_operand_checkpoint = chunk.right_operand_prefix[0];
        self.identity_checkpoint = chunk.identity_prefix[0];
        self.read_prefix_checkpoints = chunk
            .read_prefix_polys
            .iter()
            .map(|poly| PrefixEval::from(poly[0]))
            .collect();

        let chunk_bits = Self::CHUNK_BITS;
        let start = chunk.phase * chunk_bits;
        let end = start + chunk_bits;
        let point = &self.address_challenges[start..end];
        let shift = Self::LOG_K - end;
        let mask = (1u128 << chunk_bits) - 1;
        let eq_table = (0..(1usize << chunk_bits))
            .map(|bits| eq_eval_at_bits(point, bits as u128, chunk_bits))
            .collect::<Vec<_>>();
        for group in &mut self.groups {
            let chunk_value = (group.lookup_index >> shift) & mask;
            group.phase_u_eval_sum *= eq_table[chunk_value as usize];
        }
    }
}

impl<F: Field, const XLEN: usize> AddressPhaseChunk<F, XLEN> {
    /// `[read(0), read(2)]` summed over all active tables. Mirrors `read_table_round_evals`.
    fn read_table_round_evals(&self) -> [F; 2] {
        let len = self.read_prefix_polys.first().map_or(0, Vec::len);
        debug_assert!(len > 1);
        let half = len / 2;
        let prefix_evals = (0..half)
            .map(|row| {
                (
                    self.read_prefix_evals(row, false),
                    self.read_prefix_evals(row, true),
                )
            })
            .collect::<Vec<_>>();
        self.read_suffix_polys
            .iter()
            .fold([F::zero(), F::zero()], |mut total, read_table| {
                let eval = read_table_component_eval(read_table, half, &prefix_evals);
                total[0] += eval[0];
                total[1] += eval[1];
                total
            })
    }

    fn read_prefix_evals(&self, row: usize, at_2: bool) -> [PrefixEval<F>; NUM_PREFIXES] {
        let half = self.read_prefix_polys[0].len() / 2;
        let mut values = [PrefixEval::from(F::zero()); NUM_PREFIXES];
        for (value, poly) in values.iter_mut().zip(&self.read_prefix_polys) {
            let low = poly[row];
            let eval = if at_2 {
                let high = poly[row + half];
                high + high - low
            } else {
                low
            };
            *value = PrefixEval::from(eval);
        }
        values
    }

    /// `[[left0,left2],[right0,right2],[identity0,identity2]]`. Mirrors `raf_round_component_evals`.
    fn raf_round_component_evals(&self) -> [[F; 2]; 3] {
        let (left_0, left_2) = prefix_suffix_round_evals(
            Some(&self.left_operand_prefix),
            &self.raf_shift_half_q,
            &self.raf_left_q,
        );
        let (right_0, right_2) = prefix_suffix_round_evals(
            Some(&self.right_operand_prefix),
            &self.raf_shift_half_q,
            &self.raf_right_q,
        );
        let (identity_0, identity_2) = prefix_suffix_round_evals(
            Some(&self.identity_prefix),
            &self.raf_shift_full_q,
            &self.raf_identity_q,
        );
        [
            [left_0, left_2],
            [right_0, right_2],
            [identity_0, identity_2],
        ]
    }

    fn bind(&mut self, challenge: F) {
        bind_high_to_low(&mut self.left_operand_prefix, challenge);
        bind_high_to_low(&mut self.right_operand_prefix, challenge);
        bind_high_to_low(&mut self.identity_prefix, challenge);
        bind_high_to_low(&mut self.raf_shift_half_q, challenge);
        bind_high_to_low(&mut self.raf_left_q, challenge);
        bind_high_to_low(&mut self.raf_right_q, challenge);
        bind_high_to_low(&mut self.raf_shift_full_q, challenge);
        bind_high_to_low(&mut self.raf_identity_q, challenge);
        for poly in &mut self.read_prefix_polys {
            bind_high_to_low(poly, challenge);
        }
        for read_table in &mut self.read_suffix_polys {
            for poly in &mut read_table.suffix_polys {
                bind_high_to_low(poly, challenge);
            }
        }
    }
}

/// `[read(0), read(2)]` for one table over the bound rows. Mirrors `read_table_component_eval`.
fn read_table_component_eval<F: Field, const XLEN: usize>(
    read_table: &ReadTablePhase<F, XLEN>,
    half: usize,
    prefix_evals: &[PrefixPairEvals<F>],
) -> [F; 2] {
    let mut eval_0 = F::zero();
    let mut eval_2_left = F::zero();
    let mut eval_2_right = F::zero();
    let suffix_count = read_table.suffix_polys.len();
    for row in 0..half {
        let (prefixes_0, prefixes_2) = &prefix_evals[row];
        let mut suffixes_left = [F::zero(); MAX_SUFFIXES];
        let mut suffixes_right = [F::zero(); MAX_SUFFIXES];
        for (suffix_index, poly) in read_table.suffix_polys.iter().enumerate() {
            suffixes_left[suffix_index] = poly[row];
            suffixes_right[suffix_index] = poly[row + half];
        }
        eval_0 += read_table
            .table
            .combine(prefixes_0, &suffixes_left[..suffix_count]);
        eval_2_left += read_table
            .table
            .combine(prefixes_2, &suffixes_left[..suffix_count]);
        eval_2_right += read_table
            .table
            .combine(prefixes_2, &suffixes_right[..suffix_count]);
    }
    [eval_0, eval_2_right + eval_2_right - eval_2_left]
}

/// `(eval@0, eval@2)` of `Σ_row prefix·q0 + q1`. Mirrors `prefix_suffix_round_evals`.
fn prefix_suffix_round_evals<F: Field>(prefix: Option<&[F]>, q0: &[F], q1: &[F]) -> (F, F) {
    let len = q0.len();
    debug_assert_eq!(q1.len(), len);
    debug_assert!(len > 1);
    let half = len / 2;
    let mut eval_0 = F::zero();
    let mut eval_2_left = F::zero();
    let mut eval_2_right = F::zero();
    for row in 0..half {
        let (prefix_0, prefix_2) = prefix.map_or((F::one(), F::one()), |poly| {
            debug_assert_eq!(poly.len(), len);
            let low = poly[row];
            let high = poly[row + half];
            (low, high + high - low)
        });
        eval_0 += prefix_0 * q0[row] + q1[row];
        eval_2_left += prefix_2 * q0[row] + q1[row];
        eval_2_right += prefix_2 * q0[row + half] + q1[row + half];
    }
    (eval_0, eval_2_right + eval_2_right - eval_2_left)
}

/// Operand-prefix dense poly for one chunk: `checkpoint·2^(chunk_bits/2) + operand_bits(bits)`.
/// `left = true` selects `LookupBits::uninterleave().0`. Mirrors `operand_prefix_poly`.
fn operand_prefix_poly<F: Field>(checkpoint: F, chunk_bits: usize, left: bool) -> Vec<F> {
    debug_assert!(chunk_bits.is_multiple_of(2));
    let shift = F::from_u128(1u128 << (chunk_bits / 2));
    (0..(1usize << chunk_bits))
        .map(|bits| {
            let lookup_bits = LookupBits::new(bits as u128, chunk_bits);
            let (left_bits, right_bits) = lookup_bits.uninterleave();
            let operand_bits: u64 = if left {
                left_bits.into()
            } else {
                right_bits.into()
            };
            checkpoint * shift + F::from_u64(operand_bits)
        })
        .collect()
}

/// Identity-prefix dense poly for one chunk: `checkpoint·2^chunk_bits + bits`. Mirrors
/// `identity_prefix_poly`.
fn identity_prefix_poly<F: Field>(checkpoint: F, chunk_bits: usize) -> Vec<F> {
    let shift = F::from_u128(1u128 << chunk_bits);
    (0..(1usize << chunk_bits))
        .map(|bits| checkpoint * shift + F::from_u64(bits as u64))
        .collect()
}

/// `eq(point, bits)` for a binary `bits` (MSB-first). Mirrors `eq_eval_at_bits`.
fn eq_eval_at_bits<F: Field>(point: &[F], bits: u128, num_bits: usize) -> F {
    debug_assert_eq!(point.len(), num_bits);
    point
        .iter()
        .enumerate()
        .map(|(index, &challenge)| {
            if ((bits >> (num_bits - 1 - index)) & 1) == 1 {
                challenge
            } else {
                F::one() - challenge
            }
        })
        .product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_lookup_tables::interleave_bits;
    use jolt_poly::EqPolynomial;

    const XLEN: usize = 8;
    const LOG_K: usize = 2 * XLEN;

    fn f(v: u64) -> F {
        F::from_u64(v)
    }

    /// Verifier-side operand MLE (stage5 `operand_polynomial_eval` convention: `left → offset 0`,
    /// the *opposite* of IL-1 `OperandPolynomial`). Lives in the P2 verifier; test-local for now.
    fn operand_polynomial_eval(point: &[F], left: bool) -> F {
        let offset = usize::from(!left);
        let m = point.len() / 2;
        (0..m)
            .map(|i| point[2 * i + offset] * F::from_u128(1u128 << (m - 1 - i)))
            .fold(f(0), |acc, t| acc + t)
    }

    fn identity_polynomial_eval(point: &[F]) -> F {
        let n = point.len();
        point
            .iter()
            .enumerate()
            .map(|(i, v)| *v * F::from_u128(1u128 << (n - 1 - i)))
            .fold(f(0), |acc, t| acc + t)
    }

    fn binary_point(idx: u128) -> Vec<F> {
        (0..LOG_K)
            .map(|i| f(((idx >> (LOG_K - 1 - i)) & 1) as u64))
            .collect()
    }

    #[test]
    fn operand_and_identity_prefix_match_definition() {
        // chunk_bits = 4 → 2-bit operands interleaved. uninterleave(bits) split, checkpoint folded.
        let checkpoint = f(3);
        let left = operand_prefix_poly::<F>(checkpoint, 4, true);
        let identity = identity_prefix_poly::<F>(checkpoint, 4);
        for bits in 0..16u128 {
            let lb = LookupBits::new(bits, 4);
            let (lo, _ro) = lb.uninterleave();
            let lo_u64: u64 = lo.into();
            assert_eq!(left[bits as usize], checkpoint * f(1 << 2) + f(lo_u64));
            assert_eq!(
                identity[bits as usize],
                checkpoint * f(1 << 4) + f(bits as u64)
            );
        }
    }

    /// Drive the full `LOG_K`-round address-phase sumcheck and check the reduced claim equals the
    /// reduced value `Σ_g u·eq(r_addr, idx)·[table.evaluate_mle(r_addr) + raf_mle(r_addr)]` — the
    /// table/operand MLEs at the *field* point r_addr (the value the P2 verifier reconstructs). The
    /// honest initial claim uses the integer (binary-point) values. Exercises build_chunk +
    /// round_poly + bind + finish_chunk across multiple phases, tables, groups, and the
    /// interleaved/identity raf split.
    #[test]
    fn address_phase_reduces_to_expected_mle() {
        let tables = LookupTableKind::<XLEN>::all();
        let and = 2usize; // And: a clean interleaved arithmetic table.

        // Four cycles → three groups (the first two dedup): two interleaved+And, one identity.
        let idx_a = interleave_bits(0b1011_0110u64, 0b0110_1001u64);
        let idx_b = interleave_bits(0b0011_1101u64, 0b1100_0011u64);
        let idx_c = 0b1010_0101_1001_0110u128; // non-interleaved, no table (identity raf only)
        let lookup_indices = vec![idx_a, idx_a, idx_b, idx_c];
        let lookup_table_indices = vec![Some(and), Some(and), Some(and), None];
        let is_interleaved = vec![true, true, true, false];

        // u_evals from a reduction point over log(trace_len) = 2 cycle vars.
        let r_reduction = [f(5), f(9)];
        let u_evals = EqPolynomial::<F>::evals(&r_reduction, None);
        assert_eq!(u_evals.len(), 4);

        let (groups, _by_cycle, groups_by_table) = lookup_groups_from_trace::<XLEN, F>(
            &lookup_indices,
            &lookup_table_indices,
            &is_interleaved,
            &u_evals,
        );
        assert_eq!(groups.len(), 3, "first two cycles dedup into one group");

        let gamma = f(17);
        let gamma2 = gamma * gamma;
        let read_at = |g: &LookupGroup<F>, point: &[F]| -> F {
            match g.lookup_table_index {
                Some(t) => tables[t].evaluate_mle::<F, F>(point),
                None => f(0),
            }
        };
        let raf_at = |g: &LookupGroup<F>, point: &[F]| -> F {
            if g.is_interleaved_operands {
                gamma * operand_polynomial_eval(point, true)
                    + gamma2 * operand_polynomial_eval(point, false)
            } else {
                gamma2 * identity_polynomial_eval(point)
            }
        };

        // Honest initial claim: integer read+raf values (= MLEs at the binary index point).
        let initial_claim: F = groups
            .iter()
            .map(|g| {
                let bp = binary_point(g.lookup_index);
                g.u_eval_sum * (read_at(g, &bp) + raf_at(g, &bp))
            })
            .sum();

        let mut state =
            InstructionAddressPhase::<F, XLEN>::new(groups.clone(), groups_by_table, gamma, f(1));

        let mut claim = initial_claim;
        let mut r_addr = Vec::with_capacity(LOG_K);
        for round in 0..LOG_K {
            let poly = state.round_poly(claim);
            let challenge = f((round as u64) * 7 + 3);
            r_addr.push(challenge);
            claim = poly.evaluate(challenge);
            state.bind(challenge);
        }

        let expected: F = groups
            .iter()
            .map(|g| {
                g.u_eval_sum
                    * eq_eval_at_bits(&r_addr, g.lookup_index, LOG_K)
                    * (read_at(g, &r_addr) + raf_at(g, &r_addr))
            })
            .sum();

        assert_eq!(
            claim, expected,
            "address-phase reduced claim == expected MLE at r_addr"
        );
    }
}
