#!/usr/bin/env bash
#
# Orchestrates the WHIR-vs-Dory PCS commitment benchmark on the ECDSA workload.
# Builds both binaries, runs them (BN254 only), then prints a comparison table.
#
# Usage:
#   crates/jolt-pcs-bench/run-bench.sh [--runs N] [--warmup K]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="/tmp/jolt-pcs-bench"
DORY_JSON="${OUT_DIR}/dory.json"
WHIR_BN254_JSON="${OUT_DIR}/whir-bn254.json"
DUMP_PATH="${OUT_DIR}/polys.bin"
COMBINED_JSON="${OUT_DIR}/combined.json"

RUNS=5
WARMUP=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--runs N] [--warmup K]"
            exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

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

echo "=== WHIR-ZK bench: BN254 ==="
"${REPO_ROOT}/target/release/whir-pcs-bench" \
    --warmup "${WARMUP}" \
    --runs "${RUNS}" \
    --dump "${DUMP_PATH}" \
    --json "${WHIR_BN254_JSON}"

echo
echo "=== combined report ==="
python3 - "${DORY_JSON}" "${WHIR_BN254_JSON}" "${COMBINED_JSON}" <<'PY'
import json
import sys
from pathlib import Path

dory_path = Path(sys.argv[1])
whir_bn254_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

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

whir = json.loads(whir_bn254_path.read_text())

print("Polynomial inventory:")
print(f"  Dory  (one-hot, BN254):           {dory_elements/1e6:6.1f}M field elements (32 B/elem)")
print(f"  WHIR  (LogUp*+dense, BN254):     {whir['total_field_elements']/1e6:6.1f}M "
      f"field elements ({whir['field_bytes_per_elem']} B/elem)")
print()

ratio_elements = dory_elements / whir["total_field_elements"]
print(f"  Ratio Dory/WHIR (by element):     {ratio_elements:.2f}x  "
      f"(WHIR is {100/ratio_elements:.1f}% of Dory)")
print()

print("Timing:")
print(stats_line("Dory       (BN254)", dory_t, dory_elements))
cmin, cmed, cmax = commit_only_stats(whir)
commit_synth = {"min_ms": cmin, "median_ms": cmed, "max_ms": cmax,
                "encode_seconds": whir.get("encode_seconds", 0)}
print(stats_line("WHIR-ZK BN254 commit", commit_synth, whir['total_field_elements'],
                 whir['field_bytes_per_elem']))
if 'gkr' in whir:
    g = whir['gkr']
    ce = g.get('claim_eval_ms', {})
    gk = g.get('gkr_only_ms', {})
    if ce:
        print(f"  {'  ├─ claim_eval':<24} min={ce['min_ms']:7.1f}ms  median={ce['median_ms']:7.1f}ms  "
              f"max={ce['max_ms']:7.1f}ms  (d MLE evals per family)")
    if gk:
        print(f"  {'  ├─ gkr (eq-weighted)':<24} min={gk['min_ms']:7.1f}ms  median={gk['median_ms']:7.1f}ms  "
              f"max={gk['max_ms']:7.1f}ms  (3 families, max A-depth=24)")
    print(f"  {'  └─ commit + LogUp*':<24} min={whir['min_ms']:7.1f}ms  median={whir['median_ms']:7.1f}ms  "
          f"max={whir['max_ms']:7.1f}ms")
print()

print("Wall-clock ratios (WHIR / Dory):")
_, cmed, _ = commit_only_stats(whir)
print(f"  {'WHIR-ZK BN254':<22} commit-only:    {cmed / dory_t['median_ms']:.2f}x")
if 'gkr' in whir:
    print(f"  {'WHIR-ZK BN254':<22} commit + LogUp*: {whir['median_ms'] / dory_t['median_ms']:.2f}x")

combined = {
    "workload": workload,
    "dory": {
        "field": "BN254 Fr",
        "total_field_elements": dory_elements,
        **dory_t,
    },
    "whir": {"WHIR-ZK BN254": whir},
}
out_path.write_text(json.dumps(combined, indent=2))
print(f"\nCombined JSON: {out_path}")
PY
