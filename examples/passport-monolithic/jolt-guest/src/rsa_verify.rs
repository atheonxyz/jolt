/// RSA PKCS#1v1.5 SHA-256 signature verification for 2048-bit and 4096-bit keys.
///
/// Modular exponentiation (sig^65537 mod n) is decomposed into 17 modular
/// multiplication steps (16 squarings + 1 final multiply). Each step is
/// offloaded to an advice function that provides (quotient, remainder).
/// The guest verifies each step in-trace: a*b == q*n + r AND r < n.
///
/// This approach is sound because:
///   - Each modmul step is independently verified via check_advice!
///   - The chain of 17 verified steps reconstructs the full modexp
///   - A malicious prover cannot skip or forge any individual step

use crate::bignum;

const SHA256_DIGEST_INFO: [u8; 19] = [
    0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01,
    0x05, 0x00, 0x04, 0x20,
];

// ── Advice: per-step modular multiplication ─────────────────────────────

/// Compute a single modular multiplication step: a * b mod n.
/// Returns [q_bytes || r_bytes] where a*b = q*n + r and 0 <= r < n.
/// Runs on host during compute_advice pass; guest reads from tape in proving pass.
#[jolt::advice]
fn modmul_2048_step(
    a_bytes: &[u8; 256],
    b_bytes: &[u8; 256],
    n_bytes: &[u8; 256],
) -> jolt::UntrustedAdvice<[u8; 512]> {
    let a = bignum::be_bytes_to_limbs_2048(a_bytes);
    let b = bignum::be_bytes_to_limbs_2048(b_bytes);
    let n = bignum::be_bytes_to_limbs_2048(n_bytes);

    let product = bignum::mul_wide_2048(&a, &b);
    let (q, r) = bignum::div_rem_wide_2048(&product, &n);

    let q_bytes = bignum::limbs_to_be_bytes_2048(&q);
    let r_bytes = bignum::limbs_to_be_bytes_2048(&r);

    let mut out = [0u8; 512];
    out[..256].copy_from_slice(&q_bytes);
    out[256..].copy_from_slice(&r_bytes);
    out
}

/// Same for 4096-bit operands.
#[jolt::advice]
fn modmul_4096_step(
    a_bytes: &[u8; 512],
    b_bytes: &[u8; 512],
    n_bytes: &[u8; 512],
) -> jolt::UntrustedAdvice<[u8; 1024]> {
    let a = bignum::be_bytes_to_limbs_4096(a_bytes);
    let b = bignum::be_bytes_to_limbs_4096(b_bytes);
    let n = bignum::be_bytes_to_limbs_4096(n_bytes);

    let product = bignum::mul_wide_4096(&a, &b);
    let (q, r) = bignum::div_rem_wide_4096(&product, &n);

    let q_bytes = bignum::limbs_to_be_bytes_4096(&q);
    let r_bytes = bignum::limbs_to_be_bytes_4096(&r);

    let mut out = [0u8; 1024];
    out[..512].copy_from_slice(&q_bytes);
    out[512..].copy_from_slice(&r_bytes);
    out
}

// ── Public verification functions ───────────────────────────────────────

/// Verify RSA-2048 PKCS#1v1.5 SHA-256 signature using step-wise advice.
///
/// Decomposes sig^65537 mod n into 16 squarings + 1 multiply, each verified
/// via check_advice! against the equation a*b == q*n + r with r < n.
pub fn rsa_pkcs1v15_sha256_verify_2048(
    modulus: &[u8; 256],
    exponent: u32,
    signature: &[u8; 256],
    msg_hash: &[u8; 32],
) {
    assert_eq!(exponent, 65537, "only e=65537 is supported for step-wise RSA");

    let n_limbs = bignum::be_bytes_to_limbs_2048(modulus);
    let sig_limbs = bignum::be_bytes_to_limbs_2048(signature);
    let mut current_limbs = sig_limbs;

    // 16 squarings: sig -> sig^2 -> sig^4 -> ... -> sig^65536
    for _ in 0..16 {
        let current_bytes = bignum::limbs_to_be_bytes_2048(&current_limbs);
        let adv = modmul_2048_step(&current_bytes, &current_bytes, modulus);
        let qr: [u8; 512] = *adv;

        let mut q_bytes = [0u8; 256];
        let mut r_bytes = [0u8; 256];
        q_bytes.copy_from_slice(&qr[..256]);
        r_bytes.copy_from_slice(&qr[256..]);

        let q_limbs = bignum::be_bytes_to_limbs_2048(&q_bytes);
        let r_limbs = bignum::be_bytes_to_limbs_2048(&r_bytes);

        let modmul_ok = bignum::verify_modmul_2048(
            &current_limbs,
            &current_limbs,
            &q_limbs,
            &n_limbs,
            &r_limbs,
        );
        jolt::check_advice!(modmul_ok);
        jolt::check_advice!(bignum::lt_2048(&r_limbs, &n_limbs));

        current_limbs = r_limbs;
    }

    // Final multiply: sig^65536 * sig = sig^65537 mod n
    let current_bytes = bignum::limbs_to_be_bytes_2048(&current_limbs);
    let sig_bytes_copy = bignum::limbs_to_be_bytes_2048(&sig_limbs);
    let adv = modmul_2048_step(&current_bytes, &sig_bytes_copy, modulus);
    let qr: [u8; 512] = *adv;

    let mut q_bytes = [0u8; 256];
    let mut r_bytes = [0u8; 256];
    q_bytes.copy_from_slice(&qr[..256]);
    r_bytes.copy_from_slice(&qr[256..]);

    let q_limbs = bignum::be_bytes_to_limbs_2048(&q_bytes);
    let r_limbs = bignum::be_bytes_to_limbs_2048(&r_bytes);

    let modmul_ok =
        bignum::verify_modmul_2048(&current_limbs, &sig_limbs, &q_limbs, &n_limbs, &r_limbs);
    jolt::check_advice!(modmul_ok);
    jolt::check_advice!(bignum::lt_2048(&r_limbs, &n_limbs));

    let result_bytes = bignum::limbs_to_be_bytes_2048(&r_limbs);
    verify_pkcs1v15_padding(&result_bytes, msg_hash, 256);
}

/// Verify RSA-4096 PKCS#1v1.5 SHA-256 signature using step-wise advice.
pub fn rsa_pkcs1v15_sha256_verify_4096(
    modulus: &[u8; 512],
    exponent: u32,
    signature: &[u8; 512],
    msg_hash: &[u8; 32],
) {
    assert_eq!(exponent, 65537, "only e=65537 is supported for step-wise RSA");

    let n_limbs = bignum::be_bytes_to_limbs_4096(modulus);
    let sig_limbs = bignum::be_bytes_to_limbs_4096(signature);
    let mut current_limbs = sig_limbs;

    for _ in 0..16 {
        let current_bytes = bignum::limbs_to_be_bytes_4096(&current_limbs);
        let adv = modmul_4096_step(&current_bytes, &current_bytes, modulus);
        let qr: [u8; 1024] = *adv;

        let mut q_bytes = [0u8; 512];
        let mut r_bytes = [0u8; 512];
        q_bytes.copy_from_slice(&qr[..512]);
        r_bytes.copy_from_slice(&qr[512..]);

        let q_limbs = bignum::be_bytes_to_limbs_4096(&q_bytes);
        let r_limbs = bignum::be_bytes_to_limbs_4096(&r_bytes);

        let modmul_ok = bignum::verify_modmul_4096(
            &current_limbs,
            &current_limbs,
            &q_limbs,
            &n_limbs,
            &r_limbs,
        );
        jolt::check_advice!(modmul_ok);
        jolt::check_advice!(bignum::lt_4096(&r_limbs, &n_limbs));

        current_limbs = r_limbs;
    }

    let current_bytes = bignum::limbs_to_be_bytes_4096(&current_limbs);
    let sig_bytes_copy = bignum::limbs_to_be_bytes_4096(&sig_limbs);
    let adv = modmul_4096_step(&current_bytes, &sig_bytes_copy, modulus);
    let qr: [u8; 1024] = *adv;

    let mut q_bytes = [0u8; 512];
    let mut r_bytes = [0u8; 512];
    q_bytes.copy_from_slice(&qr[..512]);
    r_bytes.copy_from_slice(&qr[512..]);

    let q_limbs = bignum::be_bytes_to_limbs_4096(&q_bytes);
    let r_limbs = bignum::be_bytes_to_limbs_4096(&r_bytes);

    let modmul_ok =
        bignum::verify_modmul_4096(&current_limbs, &sig_limbs, &q_limbs, &n_limbs, &r_limbs);
    jolt::check_advice!(modmul_ok);
    jolt::check_advice!(bignum::lt_4096(&r_limbs, &n_limbs));

    let result_bytes = bignum::limbs_to_be_bytes_4096(&r_limbs);
    verify_pkcs1v15_padding(&result_bytes, msg_hash, 512);
}

// ── PKCS#1v1.5 padding verification ─────────────────────────────────────

/// Verify PKCS#1v1.5 padding for SHA-256.
/// Expected format: 0x00 0x01 [0xFF padding] 0x00 [DigestInfo(19)] [Hash(32)]
fn verify_pkcs1v15_padding(decrypted: &[u8], expected_hash: &[u8; 32], key_bytes: usize) {
    assert!(key_bytes >= 54, "key too small for SHA-256 PKCS#1v1.5");

    assert_eq!(decrypted[0], 0x00, "PKCS#1v1.5: byte 0 must be 0x00");
    assert_eq!(decrypted[1], 0x01, "PKCS#1v1.5: byte 1 must be 0x01");

    let padding_len = key_bytes - 54;
    for i in 0..padding_len {
        assert_eq!(
            decrypted[2 + i],
            0xFF,
            "PKCS#1v1.5: padding byte must be 0xFF"
        );
    }

    let sep_idx = 2 + padding_len;
    assert_eq!(
        decrypted[sep_idx], 0x00,
        "PKCS#1v1.5: separator must be 0x00"
    );

    let di_start = sep_idx + 1;
    for i in 0..19 {
        assert_eq!(
            decrypted[di_start + i],
            SHA256_DIGEST_INFO[i],
            "PKCS#1v1.5: DigestInfo mismatch"
        );
    }

    let hash_start = di_start + 19;
    for i in 0..32 {
        assert_eq!(
            decrypted[hash_start + i],
            expected_hash[i],
            "PKCS#1v1.5: hash mismatch"
        );
    }
}
