# whir-pcs-bench — WHIR-ZK commit + paper-faithful LogUp\* prover

The WHIR side of the Jolt PCS commitment benchmark. Reads an integer-form
polynomial dump produced by `jolt-pcs-bench`, encodes the polynomials into a
chosen scalar field, and times three phases:

1. **`commit_ms`** — wall-clock of `whir::protocols::whir_zk::Config::commit`
   over: 40 `ra_dense` ∈ F^{2^19} + **3 eq-weighted pushforwards** `P^F`
   ∈ F^{2^15} (one per Jolt family — InstructionRa, BytecodeRa, RamRa, per
   LogUp\* §4.1) + 2 dense `RdInc`/`RamInc` ∈ F^{2^19}.
2. **`claim_eval_ms`** — wall-clock of the `d` MLE evaluations
   `M̃^(i)(r_row, r_col)` per family that feed PAZK §4.5.2's claim reduction.
   In a real protocol these come from the upstream sumcheck binding; the
   bench reports them separately so reviewers can decide how to attribute
   the cost.
3. **`gkr_ms`** — wall-clock of the LogUp\* §4 / Figure-1 prover in
   [src/gkr.rs](src/gkr.rs) per family: §4.5.2 RLC + eq-weighted P^F
   build + fan-in-2 fractional GKR proving the well-formedness identity

   ```
   ∀ family F :  Σ_j eq(bits(j), r_M_row) / (α − M^(*)_F[j])
                ==  Σ_k P^F[k] / (α − k)
   ```

   for each of the 3 families.

This binary lives in a **separate workspace** from the Jolt repo because
of a `digest 0.10` vs `0.11` trait-bound conflict on `blake3::Hasher`; see
[../jolt/crates/jolt-pcs-bench/README.md](../jolt/crates/jolt-pcs-bench/README.md)
for the architectural rationale.

## How to run

```bash
# Prerequisite: produce the polynomial dump by running the Jolt side first.
cd ../jolt
cargo run -p jolt-pcs-bench --release -- \
    --dump /tmp/jolt-pcs-bench/polys.bin --no-dory
cd ../whir-pcs-bench

# Build + run one field
cargo run --release -- \
    --field goldilocks-fp3 \
    --warmup 1 --runs 5 \
    --dump /tmp/jolt-pcs-bench/polys.bin \
    --json /tmp/whir-fp3.json
```

CLI flags:

- `--field {bn254 | goldilocks-fp3}` — scalar field for WHIR.
  - `bn254`: `Identity<Field256>` (32 B/elem, apples-to-apples with Dory).
  - `goldilocks-fp3`: `Identity<Field64_3>` (24 B/elem, 192-bit prime — the
    soundness-correct setup at 128-bit security).
- `--no-gkr` — skip the GKR phase. Useful for isolating commit-only numbers
  or for regression-checking against the prior milestone.
- `--warmup N` / `--runs N` — warmup and measured iterations.
- `--dump PATH` — input dump path.
- `--json PATH` — write a structured report.
- `--trace-chrome PATH` — emit a tracing-chrome trace for Perfetto.
- `--profile-alloc --dhat-output PATH` — capture a dhat heap profile
  (requires `cargo build --features profile-alloc`).

## What's in this crate

| File | Purpose |
|---|---|
| [src/main.rs](src/main.rs) | CLI, dump loader, integer→F encoding, commit timing loop, GKR timing loop, JSON output. |
| [src/gkr.rs](src/gkr.rs) | Self-contained fractional-GKR prover (leaves → bottom-up circuit → top-down batched sumcheck). |
| [Cargo.toml](Cargo.toml) | Pins `blake3 = "=1.8.3"` for digest-0.10 compatibility with WHIR. |

## Design

For how this binary maps to the **LogUp\* Twist/Shout** paper sections (§4.1
batching, §5.1 Shout, §5.2 Twist, Figure 1 GKR pushforward) and how the GKR
prover relates to the original **Twist/Shout** paper, see
[../jolt/crates/jolt-pcs-bench/DESIGN_WHIR.md](../jolt/crates/jolt-pcs-bench/DESIGN_WHIR.md).

That document covers the WHIR side end-to-end:
- the LogUp\* transformation in `jolt-pcs-bench/src/logup_star.rs` (which
  produces the dump this binary reads),
- the WHIR-ZK commit parameters and the size-class bucketing,
- the GKR fan-in-2 fractional circuit and batched per-layer sumcheck,
- and the bench-local `P[0]` correction that reconciles None-cycles between
  the A-side and B-side of the histogram identity.

For the **Dory** side of the comparison (and how it maps to the original
**Twist/Shout** paper) see
[../jolt/crates/jolt-pcs-bench/DESIGN_DORY.md](../jolt/crates/jolt-pcs-bench/DESIGN_DORY.md).

## Output

Stdout summary plus an optional JSON report with shape:

```json
{
  "scheme": "whir-zk",
  "field": "Goldilocks Fp3 (Field64_3)",
  "field_bytes_per_elem": 24,
  "total_field_elements": 22020096,
  "encode_seconds": 0.09,
  "params": { "security_level": 128, "pow_bits": 20, … },
  "runs_ms": [6158.7, 6159.3, 6159.6, 6172.3, 6265.9],
  "min_ms": 6158.7, "median_ms": 6159.6, "max_ms": 6265.9,
  "per_class_ms": { "15": [...3 pushforwards...], "19": [...42 ra+dense...] },
  "size_classes": { "15": 3, "19": 42 },
  "gkr": {
    "runs_ms": [2886, 2890, 2891, 2901, 2949],     // claim_eval + gkr_only per run
    "min_ms": 2886, "median_ms": 2891, "max_ms": 2949,
    "claim_eval_ms": { "min_ms":   72.5, "median_ms":   73.4, "max_ms":   74.9, "runs_ms": [...] },
    "gkr_only_ms":   { "min_ms": 2813.4, "median_ms": 2817.1, "max_ms": 2874.6, "runs_ms": [...] },
    "per_family_ms": [/* 3 entries: InstructionRa, BytecodeRa, RamRa ≈ 2089, 310, 290 ms */],
    "per_layer_ms":  [/* 24 entries: per-layer A-side sumcheck cost, layer 23 ≈ 906 ms */]
  }
}
```

The `gkr` block is omitted when `--no-gkr` is set; in that case the 3
eq-weighted pushforwards are also NOT committed (small size class
becomes empty).

For per-layer profile breakdown and dhat allocation numbers, see
[../jolt/crates/jolt-pcs-bench/PROFILING.md](../jolt/crates/jolt-pcs-bench/PROFILING.md).
