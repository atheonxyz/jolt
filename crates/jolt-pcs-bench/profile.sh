#!/usr/bin/env bash
#
# Profiling orchestrator for jolt-pcs-bench + whir-pcs-bench (BN254 only).
#
# Captures three artifacts per configuration:
#   - samply.profraw   CPU sampling profile (samply / Firefox profiler)
#   - chrome.json      tracing span tree (https://ui.perfetto.dev/)
#   - dhat-heap.json   heap-allocation profile (https://nnethercote.github.io/dh_view/)
#
# Usage:
#   profile.sh {dory|whir-bn254|all} [--samply|--chrome|--dhat|--all]
#
# Defaults to `--all` if no profile-type flag is given.
#
# Output directory:
#   /tmp/jolt-pcs-bench/traces/<config>/
#
# View artifacts:
#   samply load <path>/samply.profraw         # opens in Firefox profiler
#   open https://ui.perfetto.dev/             # drag <path>/chrome.json
#   open https://nnethercote.github.io/dh_view/dh_view.html  # drag <path>/dhat-heap.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="/tmp/jolt-pcs-bench/traces"
DUMP_PATH="/tmp/jolt-pcs-bench/polys.bin"

# Bench parameters during profiling. Keep small to keep traces readable:
# 0 warmup + 1 run is enough to see the structure; per-poly Instant timings
# in stdout give you the raw numbers if you want them.
RUNS=1
WARMUP=0

CONFIG="${1:-}"
MODE="${2:---all}"

usage() {
    echo "Usage: $0 {dory|whir-bn254|all} [--samply|--chrome|--dhat|--all]" >&2
    exit 1
}

case "$CONFIG" in
    dory|whir-bn254|all) ;;
    *) usage ;;
esac

case "$MODE" in
    --samply|--chrome|--dhat|--all) ;;
    *) usage ;;
esac

mkdir -p "$OUT_ROOT"

# ----------------------------------------------------------------------------
# Build phases (one binary per target profile)
# ----------------------------------------------------------------------------

build_jolt() {
    local profile="$1"
    local extra_features="${2:-}"
    echo "=== building jolt-pcs-bench (profile=$profile, features='$extra_features') ==="
    if [[ -n "$extra_features" ]]; then
        (cd "$REPO_ROOT" && cargo build -p jolt-pcs-bench --profile "$profile" \
            --features "$extra_features" -q)
    else
        (cd "$REPO_ROOT" && cargo build -p jolt-pcs-bench --profile "$profile" -q)
    fi
}

build_whir() {
    local profile="$1"
    local extra_features="${2:-}"
    echo "=== building whir-pcs-bench (profile=$profile, features='$extra_features') ==="
    if [[ -n "$extra_features" ]]; then
        (cd "$REPO_ROOT" && cargo build -p whir-pcs-bench --profile "$profile" \
            --features "$extra_features" -q)
    else
        (cd "$REPO_ROOT" && cargo build -p whir-pcs-bench --profile "$profile" -q)
    fi
}

# ----------------------------------------------------------------------------
# Per-config runners
# ----------------------------------------------------------------------------

# Always materialize a fresh dump before profiling WHIR.
ensure_dump() {
    if [[ ! -f "$DUMP_PATH" ]]; then
        echo "=== materializing polynomial dump (one-time) ==="
        build_jolt release
        "$REPO_ROOT/target/release/jolt-pcs-bench" \
            --no-dory --runs 0 --warmup 0 --dump "$DUMP_PATH" >/dev/null
    fi
}

run_dory() {
    local out_dir="$OUT_ROOT/dory"
    mkdir -p "$out_dir"
    echo
    echo "=== profiling Dory (AddressMajor one-hot path, BN254) ==="

    if [[ "$MODE" == "--samply" || "$MODE" == "--all" ]]; then
        build_jolt samply
        echo "[samply] recording dory..."
        samply record --save-only \
            --output "$out_dir/samply.profraw" -- \
            "$REPO_ROOT/target/samply/jolt-pcs-bench" \
                --warmup "$WARMUP" --runs "$RUNS" --no-dump
    fi

    if [[ "$MODE" == "--chrome" || "$MODE" == "--all" ]]; then
        build_jolt release
        echo "[chrome] recording dory..."
        "$REPO_ROOT/target/release/jolt-pcs-bench" \
            --warmup "$WARMUP" --runs "$RUNS" --no-dump \
            --trace-chrome "$out_dir/chrome.json" >/dev/null
    fi

    if [[ "$MODE" == "--dhat" || "$MODE" == "--all" ]]; then
        build_jolt release profile-alloc
        echo "[dhat] recording dory..."
        # Run from REPO_ROOT (the `jolt` CLI needs a workspace Cargo.toml in
        # or above CWD to build the guest); redirect dhat output via the flag.
        ( cd "$REPO_ROOT" && \
          ./target/release/jolt-pcs-bench \
              --warmup "$WARMUP" --runs "$RUNS" --no-dump \
              --profile-alloc \
              --dhat-output "$out_dir/dhat-heap.json" >/dev/null )
    fi
}

run_whir() {
    local label="$1"          # whir-bn254
    local out_dir="$OUT_ROOT/$label"
    mkdir -p "$out_dir"
    ensure_dump
    echo
    echo "=== profiling WHIR-ZK ($label, BN254) ==="

    if [[ "$MODE" == "--samply" || "$MODE" == "--all" ]]; then
        build_whir samply
        echo "[samply] recording $label..."
        samply record --save-only \
            --output "$out_dir/samply.profraw" -- \
            "$REPO_ROOT/target/samply/whir-pcs-bench" \
                --warmup "$WARMUP" --runs "$RUNS" \
                --dump "$DUMP_PATH"
    fi

    if [[ "$MODE" == "--chrome" || "$MODE" == "--all" ]]; then
        build_whir release
        echo "[chrome] recording $label..."
        "$REPO_ROOT/target/release/whir-pcs-bench" \
            --warmup "$WARMUP" --runs "$RUNS" \
            --dump "$DUMP_PATH" \
            --trace-chrome "$out_dir/chrome.json" >/dev/null
    fi

    if [[ "$MODE" == "--dhat" || "$MODE" == "--all" ]]; then
        build_whir release profile-alloc
        echo "[dhat] recording $label..."
        "$REPO_ROOT/target/release/whir-pcs-bench" \
            --warmup "$WARMUP" --runs "$RUNS" \
            --dump "$DUMP_PATH" \
            --profile-alloc \
            --dhat-output "$out_dir/dhat-heap.json" >/dev/null
    fi
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

case "$CONFIG" in
    dory)       run_dory ;;
    whir-bn254) run_whir whir-bn254 ;;
    all)
        run_dory
        run_whir whir-bn254
        ;;
esac

echo
echo "=== artifacts written to $OUT_ROOT ==="
find "$OUT_ROOT" -type f -name '*.json' -o -name '*.profraw' | sort
echo
echo "view in:"
echo "  samply.profraw → samply load <path>"
echo "  chrome.json    → https://ui.perfetto.dev/  (drag-drop)"
echo "  dhat-heap.json → https://nnethercote.github.io/dh_view/dh_view.html"
