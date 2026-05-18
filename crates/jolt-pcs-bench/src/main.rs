// Force-link the p256 inlines so their `inventory::submit!`
// registrations are visible to the tracer at runtime.
use jolt_inlines_p256 as _;

mod dory_bench;
mod dump;
mod jolt_polys;
mod logup_star;
mod verify;
mod workload;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use dory_bench::{bench_dory, DoryBenchSummary};
use dump::dump_for_whir;
use jolt_polys::build_polynomial_set;
use logup_star::transform;
use verify::verify_transformation;
use workload::build_ecdsa_workload;

#[derive(Parser, Debug)]
#[command(name = "jolt-pcs-bench")]
struct Args {
    /// Path the polynomial dump for whir-pcs-bench is written to.
    #[arg(long, default_value = "/tmp/jolt-pcs-bench/polys.bin")]
    dump: PathBuf,

    /// Skip the Dory commit benchmark (e.g. when only dumping for WHIR).
    #[arg(long)]
    no_dory: bool,

    /// Skip writing the WHIR polynomial dump.
    #[arg(long)]
    no_dump: bool,

    /// Only run the verify-transformation invariants, no timing.
    #[arg(long)]
    verify_only: bool,

    /// Warm-up runs (not timed).
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Measured runs.
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// Optional JSON output path.
    #[arg(long)]
    json: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    let start = Instant::now();

    let workload = build_ecdsa_workload();
    println!(
        "[main] workload built: T={}, instruction_d={}, bytecode_d={}, ram_d={}",
        workload.trace_len,
        workload.one_hot_params.instruction_d,
        workload.one_hot_params.bytecode_d,
        workload.one_hot_params.ram_d,
    );

    let polys = build_polynomial_set(
        &workload.sources,
        &workload.one_hot_params,
        workload.trace_len,
    );
    let total = polys.total_field_elements();
    let one_hot_count: usize = polys
        .one_hot_families
        .iter()
        .map(|fam| fam.chunks.len())
        .sum();
    println!(
        "[main] Dory polynomial set: {} one-hot chunks across {} families, {} dense, total {} field elements ({:.1}M)",
        one_hot_count,
        polys.one_hot_families.len(),
        polys.dense.len(),
        total,
        (total as f64) / 1.0e6,
    );

    let logup = transform(&polys);
    let logup_total = logup.total_field_elements();
    let dense_field: usize = polys.dense.iter().map(|d| d.values.len()).sum();
    let whir_total = logup_total + dense_field;
    println!(
        "[main] LogUp* set: {} ra_dense + {} pushforward, total {} field elements ({:.1}M)",
        logup.ra_dense.len(),
        logup.pushforwards.len(),
        logup_total,
        (logup_total as f64) / 1.0e6,
    );
    println!(
        "[main] WHIR total (ra_dense + pushforward + RdInc + RamInc): {} field elements ({:.1}M)",
        whir_total,
        (whir_total as f64) / 1.0e6,
    );
    println!(
        "[main] Dory/WHIR field-element ratio: {:.2}x (WHIR is {:.1}% of Dory)",
        (total as f64) / (whir_total as f64),
        100.0 * (whir_total as f64) / (total as f64),
    );

    verify_transformation(&workload, &polys, &logup);

    if args.verify_only {
        println!(
            "[main] --verify-only: exiting after invariants ({:.2}s)",
            start.elapsed().as_secs_f64()
        );
        return;
    }

    if !args.no_dump {
        let dump_start = Instant::now();
        let n = dump_for_whir(
            &polys,
            &logup,
            &workload.sources,
            workload.trace_len,
            &args.dump,
        )
        .expect("dump polys");
        println!(
            "[dump] wrote {n} field elements to {} in {:.1}s",
            args.dump.display(),
            dump_start.elapsed().as_secs_f64()
        );
    }

    let mut dory_summary: Option<DoryBenchSummary> = None;
    if !args.no_dory {
        let summary = bench_dory(&polys, args.warmup, args.runs);
        let times = summary.total_times_ms();
        let (min, median, max) = min_median_max(&times);
        println!(
            "[dory] summary: min={min:.1}ms median={median:.1}ms max={max:.1}ms (setup={:.1}ms)",
            summary.setup_ms
        );
        dory_summary = Some(summary);
    }

    if let Some(path) = &args.json {
        write_json_report(path, total, whir_total, &workload, dory_summary.as_ref());
        println!("[main] wrote JSON report to {}", path.display());
    }

    println!(
        "[main] total wall time: {:.2}s",
        start.elapsed().as_secs_f64()
    );
}

fn min_median_max(times: &[f64]) -> (f64, f64, f64) {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = sorted[0];
    let median = sorted[sorted.len() / 2];
    let max = sorted[sorted.len() - 1];
    (min, median, max)
}

fn write_json_report(
    path: &std::path::Path,
    dory_total_elements: usize,
    whir_total_elements: usize,
    workload: &workload::EcdsaWorkload,
    dory: Option<&DoryBenchSummary>,
) {
    let dory_json = dory.map(|summary| {
        let times = summary.total_times_ms();
        let (min, median, max) = min_median_max(&times);
        let per_oracle_serialized: Vec<Vec<serde_json::Value>> = summary
            .runs
            .iter()
            .map(|run| {
                run.per_oracle
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "name": t.name,
                            "num_vars": t.num_vars,
                            "elapsed_ms": t.elapsed_ms,
                        })
                    })
                    .collect()
            })
            .collect();
        serde_json::json!({
            "setup_ms": summary.setup_ms,
            "setup_num_vars": summary.setup_num_vars,
            "runs_ms": times,
            "min_ms": min,
            "median_ms": median,
            "max_ms": max,
            "per_oracle_runs": per_oracle_serialized,
        })
    });

    let report = serde_json::json!({
        "scheme": "dory",
        "field": "BN254 Fr",
        "workload": {
            "name": "p256-ecdsa-verify",
            "log_t": workload.log_t,
            "trace_len": workload.trace_len,
            "bytecode_k": workload.bytecode_k,
            "ram_k": workload.ram_k,
            "log_k_chunk": workload.one_hot_params.log_k_chunk,
            "instruction_d": workload.one_hot_params.instruction_d,
            "bytecode_d": workload.one_hot_params.bytecode_d,
            "ram_d": workload.one_hot_params.ram_d,
        },
        "dory_total_field_elements": dory_total_elements,
        "whir_total_field_elements": whir_total_elements,
        "ratio_dory_over_whir": (dory_total_elements as f64) / (whir_total_elements as f64),
        "dory": dory_json,
    });
    std::fs::write(path, serde_json::to_string_pretty(&report).unwrap())
        .expect("write json");
}
