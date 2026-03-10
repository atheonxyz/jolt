use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::curve::Bn254Curve;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::verifier::JoltVerifierPreprocessing;
use std::io::Cursor;
use std::path::Path;

type ProverPrep = JoltProverPreprocessing<ark_bn254::Fr, Bn254Curve, DoryCommitmentScheme>;
type VerifierPrep = JoltVerifierPreprocessing<ark_bn254::Fr, Bn254Curve, DoryCommitmentScheme>;

fn test_prover_roundtrip(bytes: &[u8]) -> Result<(), String> {
    let total = bytes.len();
    println!("\nTesting Prover Preprocessing Roundtrip");
    println!("Total bytes: {total}");

    println!("Deserializing ProverPreprocessing...");
    let prep: ProverPrep = CanonicalDeserialize::deserialize_uncompressed_unchecked(Cursor::new(bytes))
        .map_err(|e| format!("Deserialize failed: {e}"))?;
    println!("  OK");

    println!("Testing serialize -> deserialize roundtrip...");
    let mut reserialized = Vec::new();
    prep.serialize_uncompressed(&mut reserialized)
        .map_err(|e| format!("Reserialize failed: {e}"))?;

    if reserialized.len() != bytes.len() {
        let orig = bytes.len();
        let reser = reserialized.len();
        return Err(format!(
            "Prover roundtrip size mismatch: original {orig} vs reserialized {reser}"
        ));
    }

    if reserialized != bytes {
        for (i, (a, b)) in bytes.iter().zip(reserialized.iter()).enumerate() {
            if a != b {
                return Err(format!(
                    "Prover roundtrip byte mismatch at position {i}: original {a:02x} vs reserialized {b:02x}"
                ));
            }
        }
    }
    println!("  OK - bytes match exactly");

    Ok(())
}

fn test_verifier_roundtrip(bytes: &[u8]) -> Result<(), String> {
    let total = bytes.len();
    println!("\nTesting Verifier Preprocessing Roundtrip");
    println!("Total bytes: {total}");

    println!("Deserializing VerifierPreprocessing...");
    let prep: VerifierPrep = CanonicalDeserialize::deserialize_uncompressed_unchecked(Cursor::new(bytes))
        .map_err(|e| format!("Deserialize failed: {e}"))?;
    println!("  OK");

    println!("Testing serialize -> deserialize roundtrip...");
    let mut reserialized = Vec::new();
    prep.serialize_uncompressed(&mut reserialized)
        .map_err(|e| format!("Reserialize failed: {e}"))?;

    if reserialized.len() != bytes.len() {
        let orig = bytes.len();
        let reser = reserialized.len();
        return Err(format!(
            "Verifier roundtrip size mismatch: original {orig} vs reserialized {reser}"
        ));
    }

    if reserialized != bytes {
        for (i, (a, b)) in bytes.iter().zip(reserialized.iter()).enumerate() {
            if a != b {
                return Err(format!(
                    "Verifier roundtrip byte mismatch at position {i}: original {a:02x} vs reserialized {b:02x}"
                ));
            }
        }
    }
    println!("  OK - bytes match exactly");

    Ok(())
}

fn main() {
    let _ = jolt_inlines_sha2::init_inlines();
    let _ = jolt_inlines_secp256k1::init_inlines();
    let _ = jolt_inlines_keccak256::init_inlines();

    let www_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("www");

    let programs = [
        ("sha2", "sha2_prover.bin", "sha2_verifier.bin"),
        ("ecdsa", "ecdsa_prover.bin", "ecdsa_verifier.bin"),
        ("keccak", "keccak_prover.bin", "keccak_verifier.bin"),
    ];

    for (name, prover_file, verifier_file) in &programs {
        let prover_path = www_dir.join(prover_file);
        println!("\n[{name}] Reading prover preprocessing from {prover_path:?}");
        let prover_bytes = std::fs::read(&prover_path).expect("Failed to read prover file");

        match test_prover_roundtrip(&prover_bytes) {
            Ok(()) => println!("[{name}] Prover preprocessing: PASS"),
            Err(e) => {
                println!("[{name}] Prover preprocessing: FAIL - {e}");
                std::process::exit(1);
            }
        }

        let verifier_path = www_dir.join(verifier_file);
        println!("[{name}] Reading verifier preprocessing from {verifier_path:?}");
        let verifier_bytes = std::fs::read(&verifier_path).expect("Failed to read verifier file");

        match test_verifier_roundtrip(&verifier_bytes) {
            Ok(()) => println!("[{name}] Verifier preprocessing: PASS"),
            Err(e) => {
                println!("[{name}] Verifier preprocessing: FAIL - {e}");
                std::process::exit(1);
            }
        }
    }

    println!("\nAll tests passed!");
}
