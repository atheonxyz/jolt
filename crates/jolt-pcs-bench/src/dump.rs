//! Field-agnostic polynomial dump format consumed by `whir-pcs-bench`.
//!
//! Stores the *underlying integer values* of each committed polynomial rather
//! than pre-encoded field elements. Each WHIR-side field independently encodes
//! the integers via `F::from_u64` / `F::from_i128` at load time.
//!
//! Binary layout (version 2):
//!
//! ```text
//!   8 bytes  magic       = b"JOLTPCSB"
//!   4 bytes  version u32 = 2
//!   4 bytes  num_vectors u32
//!   per vector:
//!     1 byte    kind u8  (0=U8, 1=U32, 2=I128)
//!     4 bytes   label_len u32
//!     [u8]      label (UTF-8, label_len bytes)
//!     4 bytes   values_len u32 (must be a power of two)
//!     [u8]      packed values:
//!                   kind=U8   →  values_len bytes
//!                   kind=U32  →  4 * values_len bytes (LE)
//!                   kind=I128 → 16 * values_len bytes (LE)
//! ```
//!
//! ## Pushforward field is legacy after the §4.1 rewrite
//!
//! The `pushforward::<family>_<chunk>` u32 entries in this dump are the
//! **unweighted per-chunk histograms** `P[k] = #{j : ra_dense[j] = k}` from
//! the original (incorrect) implementation. The paper-faithful design
//! requires the *eq-weighted* per-family pushforward
//! `P^F[k] = Σ_{j : M^(*)[j] = k} ẽq(bits(j), r_M_row)`, where `r_M_row`
//! is Fiat-Shamir-squeezed inside the WHIR transcript. That cannot be
//! pre-computed at dump time, so the WHIR side rebuilds it at runtime
//! (`whir-pcs-bench/src/gkr.rs::prepare_pushforwards`) and discards
//! these u32 entries on load. The Jolt side still emits them for backward
//! compatibility with any tooling that reads version-2 dumps.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use jolt_witness::CommitmentTraceSources;

use crate::jolt_polys::JoltPolynomialSet;
use crate::logup_star::LogUpStarSet;

const DUMP_MAGIC: &[u8; 8] = b"JOLTPCSB";
const DUMP_VERSION: u32 = 2;

#[derive(Clone, Copy)]
enum Kind {
    U8 = 0,
    U32 = 1,
    I128 = 2,
}

/// Field elements (logical count, not bytes) written to the dump.
/// This is the metric reported as "WHIR total field elements".
#[tracing::instrument(skip_all, name = "bench.dump_for_whir")]
pub fn dump_for_whir(
    polys: &JoltPolynomialSet,
    logup: &LogUpStarSet,
    sources: &CommitmentTraceSources,
    trace_len: usize,
    path: &Path,
) -> std::io::Result<usize> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(DUMP_MAGIC)?;
    write_u32(&mut writer, DUMP_VERSION)?;

    // Materialize the three vector groups in a stable order:
    //   1. ra_dense  (U8,   len = T)            — per one-hot chunk
    //   2. pushforward (U32, len = padded K)    — per one-hot chunk
    //   3. dense   (I128, len = T)              — RdInc, RamInc
    //
    // For (1) and (2), zip the source `OneHotFamily.chunks[].indices`
    // (the integer form) with the padded pushforward length from
    // `LogUpStarSet.pushforwards[i].values.len()`.

    let mut metadata: Vec<(Kind, String, usize)> = Vec::new();

    // Pair up: each family.chunk[k] corresponds 1:1 with logup.ra_dense[i]
    // and logup.pushforwards[i] in the same flat iteration order used by
    // `transform()`.
    let mut logup_idx = 0usize;
    for family in &polys.one_hot_families {
        for chunk in &family.chunks {
            // ra_dense: U8 of length T.
            metadata.push((
                Kind::U8,
                format!("ra_dense::{}_{}", family.name, chunk.chunk),
                chunk.trace_len,
            ));
            // pushforward: U32 of length = logup.pushforwards[logup_idx].values.len()
            let pf_len = logup.pushforwards[logup_idx].values.len();
            metadata.push((
                Kind::U32,
                format!("pushforward::{}_{}", family.name, chunk.chunk),
                pf_len,
            ));
            logup_idx += 1;
        }
    }
    // Dense RdInc / RamInc (already i128 in sources, but padded to trace_len).
    for dense in &polys.dense {
        metadata.push((Kind::I128, format!("dense::{}", dense.name), trace_len));
    }

    write_u32(&mut writer, metadata.len() as u32)?;

    // Now write each vector's payload in the same order.
    let mut total_elements = 0usize;
    let mut logup_idx = 0usize;
    let mut dense_idx = 0usize;
    let mut meta_iter = metadata.iter();

    for family in &polys.one_hot_families {
        for chunk in &family.chunks {
            // ra_dense
            let (_, ref label, len) = *meta_iter.next().unwrap();
            assert_eq!(len, chunk.trace_len);
            write_header(&mut writer, Kind::U8, label, len)?;
            for opt in &chunk.indices {
                let byte = opt.unwrap_or(0);
                writer.write_all(&[byte])?;
            }
            total_elements += len;

            // pushforward
            let (_, ref label, padded_len) = *meta_iter.next().unwrap();
            write_header(&mut writer, Kind::U32, label, padded_len)?;
            // Build histogram on the fly.
            let k_chunk = chunk.chunk_domain;
            let mut hist = vec![0u32; k_chunk];
            for k in chunk.indices.iter().flatten() {
                hist[*k as usize] += 1;
            }
            for &count in hist.iter() {
                writer.write_all(&count.to_le_bytes())?;
            }
            // Zero-pad up to padded_len.
            let pad_zeros = padded_len - k_chunk;
            for _ in 0..pad_zeros {
                writer.write_all(&0u32.to_le_bytes())?;
            }
            total_elements += padded_len;

            logup_idx += 1;
        }
    }
    let _ = logup_idx; // suppress unused-write warning

    for dense in &polys.dense {
        let (_, ref label, len) = *meta_iter.next().unwrap();
        assert_eq!(len, trace_len);
        write_header(&mut writer, Kind::I128, label, len)?;

        let raw: &[i128] = match dense.name {
            "RdInc" => &sources.rd_inc,
            "RamInc" => &sources.ram_inc,
            other => panic!("dump: unsupported dense oracle `{other}`"),
        };
        // Pad with zeros up to trace_len.
        for j in 0..len {
            let v = if j < raw.len() { raw[j] } else { 0i128 };
            writer.write_all(&v.to_le_bytes())?;
        }
        total_elements += len;

        dense_idx += 1;
    }
    let _ = dense_idx;

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
