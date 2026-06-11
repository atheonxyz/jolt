#!/usr/bin/env bash
#
# Orchestrates the WHIR-vs-Dory PCS commitment benchmark on the ECDSA workload.
# Builds both binaries, runs them, then prints a combined comparison table.
#
# Usage:
#   crates/jolt-pcs-bench/run-bench.sh [--runs N] [--warmup K] [--field FIELD]
#
# FIELD selects the WHIR-side scalar field (default: bn254). Allowed:
#   bn254             — Identity<Field256>, matches Dory's field
#   goldilocks-fp3    — Identity<Field64_3>, cubic extension of Goldilocks
#   both              — runs both and reports them side-by-side

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="/tmp/jolt-pcs-bench"
DORY_JSON="${OUT_DIR}/dory.json"
WHIR_BN254_JSON="${OUT_DIR}/whir-bn254.json"
WHIR_GOLD_JSON="${OUT_DIR}/whir-goldilocks.json"
DUMP_PATH="${OUT_DIR}/polys.bin"
COMBINED_JSON="${OUT_DIR}/combined.json"

RUNS=5
WARMUP=1
FIELD="bn254"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --field) FIELD="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--runs N] [--warmup K] [--field {bn254|goldilocks-fp3|both}]"
            exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

case "$FIELD" in
    bn254|goldilocks-fp3|both) ;;
    *) echo "invalid --field: $FIELD" >&2; exit 1 ;;
esac

mkdir -p "${OUT_DIR}"

echo "=== building jolt-pcs-bench ==="
(cd "${REPO_ROOT}" && cargo build -p jolt-pcs-bench --release -q)

echo "=== building whir-pcs-bench ==="
(cd "${REPO_ROOT}" && cargo build -p whir-pcs-bench --release -q)

echo "=== Dory bench (Jolt side, BN254) ==="
"${REPO_ROOT}/target/release/jolt-pcs-bench" \
    --warmup "${WARMUP}" \
    --runs "${RUNS}" \
    --dump "${DUMP_PATH}" \
    --json "${DORY_JSON}"

# Run WHIR for the requested field(s).
WHIR_BN254_PRESENT=0
WHIR_GOLD_PRESENT=0
if [[ "$FIELD" == "bn254" || "$FIELD" == "both" ]]; then
    echo "=== WHIR-ZK bench: BN254 ==="
    "${REPO_ROOT}/target/release/whir-pcs-bench" \
        --field bn254 \
        --warmup "${WARMUP}" \
        --runs "${RUNS}" \
        --dump "${DUMP_PATH}" \
        --json "${WHIR_BN254_JSON}"
    WHIR_BN254_PRESENT=1
fi
if [[ "$FIELD" == "goldilocks-fp3" || "$FIELD" == "both" ]]; then
    echo "=== WHIR-ZK bench: Goldilocks Fp3 ==="
    "${REPO_ROOT}/target/release/whir-pcs-bench" \
        --field goldilocks-fp3 \
        --warmup "${WARMUP}" \
        --runs "${RUNS}" \
        --dump "${DUMP_PATH}" \
        --json "${WHIR_GOLD_JSON}"
    WHIR_GOLD_PRESENT=1
fi

echo
echo "=== combined report ==="
python3 - "${DORY_JSON}" "${WHIR_BN254_JSON}" "${WHIR_GOLD_JSON}" "${COMBINED_JSON}" \
        "${WHIR_BN254_PRESENT}" "${WHIR_GOLD_PRESENT}" <<'PY'
import json
import sys
from pathlib import Path

dory_path = Path(sys.argv[1])
whir_bn254_path = Path(sys.argv[2])
whir_gold_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])
have_bn254 = sys.argv[5] == "1"
have_gold = sys.argv[6] == "1"

dory = json.loads(dory_path.read_text())
workload = dory["workload"]
dory_t = dory["dory"]
dory_elements = dory["dory_total_field_elements"]

print("=== ECDSA PCS Commitment Benchmark ===")
print(f"Workload:     {workload['name']}")
print(f"Trace length: 2^{workload['log_t']} = {workload['trace_len']} cycles")
print(f"log_k_chunk:  {workload['log_k_chunk']}")
print(f"bytecode_k:   {workload['bytecode_k']} (log {workload['bytecode_k'].bit_length()-1})")
print(f"ram_k:        {workload['ram_k']} (log {workload['ram_k'].bit_length()-1})")
print(f"d-factors:    instruction_d={workload['instruction_d']}, "
      f"bytecode_d={workload['bytecode_d']}, ram_d={workload['ram_d']}")
print()

def commit_only_stats(w):
    commit = w["commit"]
    return commit["min_ms"], commit["median_ms"], commit["max_ms"]

def stats_line(label, s, elements, bytes_per=None):
    extra = f"  encode={s.get('encode_seconds', 0):.2f}s" if 'encode_seconds' in s else ""
    bpe = f"  {bytes_per}B/elem" if bytes_per else ""
    return (f"  {label:<22} min={s['min_ms']:7.1f}ms  median={s['median_ms']:7.1f}ms  "
            f"max={s['max_ms']:7.1f}ms  ({elements/1e6:.1f}M elems{bpe}){extra}")

print("Polynomial inventory:")
print(f"  Dory  (one-hot, BN254):           {dory_elements/1e6:6.1f}M field elements (32 B/elem)")

whir_payloads = []
if have_bn254:
    whir = json.loads(whir_bn254_path.read_text())
    whir_payloads.append(("WHIR-ZK BN254", whir))
    print(f"  WHIR  (LogUp*+dense, BN254):     {whir['total_field_elements']/1e6:6.1f}M "
          f"field elements ({whir['field_bytes_per_elem']} B/elem)")
if have_gold:
    whir = json.loads(whir_gold_path.read_text())
    whir_payloads.append(("WHIR-ZK Goldilocks Fp3", whir))
    print(f"  WHIR  (LogUp*+dense, Fp3-Gold):  {whir['total_field_elements']/1e6:6.1f}M "
          f"field elements ({whir['field_bytes_per_elem']} B/elem)")
print()

if whir_payloads:
    ratio_elements = dory_elements / whir_payloads[0][1]["total_field_elements"]
    print(f"  Ratio Dory/WHIR (by element):     {ratio_elements:.2f}x  "
          f"(WHIR is {100/ratio_elements:.1f}% of Dory)")
    print()

print("Timing:")
print(stats_line("Dory       (BN254)", dory_t, dory_elements))
for label, w in whir_payloads:
    # commit-only stats (claim_eval + gkr are reported separately).
    cmin, cmed, cmax = commit_only_stats(w)
    commit_synth = {"min_ms": cmin, "median_ms": cmed, "max_ms": cmax,
                    "encode_seconds": w.get("encode_seconds", 0)}
    print(stats_line(f"{label} commit", commit_synth, w['total_field_elements'],
                     w['field_bytes_per_elem']))
    if 'gkr' in w:
        g = w['gkr']
        ce = g.get('claim_eval_ms', {})
        gk = g.get('gkr_only_ms', {})
        ce_label = "  ├─ claim_eval        "
        gk_label = "  ├─ gkr (eq-weighted) "
        sum_label = "  └─ commit + LogUp*  "
        if ce:
            print(f"  {ce_label[:24]} min={ce['min_ms']:7.1f}ms  median={ce['median_ms']:7.1f}ms  "
                  f"max={ce['max_ms']:7.1f}ms  (d MLE evals per family)")
        if gk:
            print(f"  {gk_label[:24]} min={gk['min_ms']:7.1f}ms  median={gk['median_ms']:7.1f}ms  "
                  f"max={gk['max_ms']:7.1f}ms  (3 families, max A-depth=24)")
        # Top-level WHIR timings are the paired, measured end-to-end samples.
        print(f"  {sum_label[:24]} min={w['min_ms']:7.1f}ms  median={w['median_ms']:7.1f}ms  "
              f"max={w['max_ms']:7.1f}ms")
print()

print("Wall-clock ratios (WHIR / Dory):")
for label, w in whir_payloads:
    _, cmed, _ = commit_only_stats(w)
    r = cmed / dory_t["median_ms"]
    print(f"  {label:<22} commit-only:    {r:.2f}x")
    if 'gkr' in w:
        r2 = w["median_ms"] / dory_t["median_ms"]
        print(f"  {label:<22} commit + LogUp*: {r2:.2f}x")

combined = {
    "workload": workload,
    "dory": {
        "field": "BN254 Fr",
        "total_field_elements": dory_elements,
        **dory_t,
    },
    "whir": {label: w for label, w in whir_payloads},
}
out_path.write_text(json.dumps(combined, indent=2))
print(f"\nCombined JSON: {out_path}")
PY
