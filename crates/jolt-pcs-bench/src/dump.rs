//! Field-agnostic polynomial dump format consumed by `whir-pcs-bench`.
//!
//! Stores the underlying integer values of each committed polynomial rather
//! than pre-encoded field elements. Each WHIR-side field independently
//! encodes the integers via `F::from_u64` / `F::from_i128` at load time.
//!
//! Binary layout (version 3):
//!
//! ```text
//!   8 bytes  magic       = b"JOLTPCSB"
//!   4 bytes  version u32 = 3
//!   4 bytes  num_vectors u32
//!   per vector:
//!     1 byte    kind u8  (0=U8, 2=I128)
//!     4 bytes   label_len u32
//!     [u8]      label (UTF-8, label_len bytes)
//!     4 bytes   values_len u32 (must be a power of two)
//!     [u8]      packed values:
//!                   kind=U8   →  values_len bytes
//!                   kind=I128 → 16 * values_len bytes (LE)
//! ```
//!
//! V3 dropped the per-chunk U32 pushforward histograms that the WHIR side
//! never consumed: the §4.1 paper-faithful design rebuilds an eq-weighted
//! pushforward inside `whir-pcs-bench/src/gkr.rs::prepare_pushforwards` from
//! Fiat-Shamir-squeezed randomness, so the dump only carries raw `ra_dense`
//! indices and dense polys.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::jolt_polys::JoltPolynomialSet;
use crate::sources::CommitmentSources;

const DUMP_MAGIC: &[u8; 8] = b"JOLTPCSB";
const DUMP_VERSION: u32 = 3;

#[derive(Clone, Copy)]
enum Kind {
    U8 = 0,
    I128 = 2,
}

/// Field elements (logical count, not bytes) written to the dump.
/// This is the metric reported as "WHIR total field elements".
#[tracing::instrument(skip_all, name = "bench.dump_for_whir")]
pub(crate) fn dump_for_whir(
    polys: &JoltPolynomialSet,
    sources: &CommitmentSources,
    trace_len: usize,
    path: &Path,
) -> std::io::Result<usize> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);

    writer.write_all(DUMP_MAGIC)?;
    write_u32(&mut writer, DUMP_VERSION)?;

    let num_vectors: usize = polys
        .one_hot_families
        .iter()
        .map(|f| f.chunks.len())
        .sum::<usize>()
        + polys.dense.len();
    write_u32(&mut writer, num_vectors as u32)?;

    let mut total_elements = 0usize;

    // (1) ra_dense (U8) — one vector of length T per one-hot chunk.
    for family in &polys.one_hot_families {
        for chunk in &family.chunks {
            let label = format!("ra_dense::{}_{}", family.name, chunk.chunk);
            write_header(&mut writer, Kind::U8, &label, chunk.trace_len)?;
            for opt in &chunk.indices {
                writer.write_all(&[opt.unwrap_or(0)])?;
            }
            total_elements += chunk.trace_len;
        }
    }

    // (2) dense (I128) — RdInc, RamInc, padded to trace_len.
    for dense in &polys.dense {
        let label = format!("dense::{}", dense.name);
        write_header(&mut writer, Kind::I128, &label, trace_len)?;
        let raw: &[i128] = match dense.name {
            "RdInc" => &sources.rd_inc,
            "RamInc" => &sources.ram_inc,
            other => panic!("dump: unsupported dense oracle `{other}`"),
        };
        for j in 0..trace_len {
            let v = if j < raw.len() { raw[j] } else { 0i128 };
            writer.write_all(&v.to_le_bytes())?;
        }
        total_elements += trace_len;
    }

    writer.flush()?;
    Ok(total_elements)
}

fn write_header(
    writer: &mut impl Write,
    kind: Kind,
    label: &str,
    len: usize,
) -> std::io::Result<()> {
    writer.write_all(&[kind as u8])?;
    write_u32(writer, label.len() as u32)?;
    writer.write_all(label.as_bytes())?;
    assert!(
        len.is_power_of_two(),
        "{label} has non-power-of-2 length {len}"
    );
    write_u32(writer, len as u32)?;
    Ok(())
}

fn write_u32(writer: &mut impl Write, value: u32) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
