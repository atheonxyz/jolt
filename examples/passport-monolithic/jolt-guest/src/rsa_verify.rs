/// RSA PKCS#1v1.5 SHA-256 signature verification for 2048-bit and 4096-bit keys.
///
/// Modular exponentiation (sig^65537 mod n) is decomposed into 17 modular
/// multiplication steps (16 squarings + 1 final multiply). Each step is
/// offloaded to an advice function that provides (quotient, remainder).
/// The guest verifies each step in-trace: a*b == q*n + r AND r < n.
use crate::bignum;

const SHA256_DIGEST_INFO: [u8; 19] = [
    0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05,
    0x00, 0x04, 0x20,
];

#[jolt::advice]
fn modmul_2048_step(
    a: &[u64; 32],
    b: &[u64; 32],
    n: &[u64; 32],
) -> jolt::UntrustedAdvice<([u64; 32], [u64; 32])> {
    let product = bignum::mul_wide_2048(a, b);
    let (q, r) = bignum::div_rem_wide_2048(&product, n);
    (q, r)
}

#[jolt::advice]
fn modmul_4096_step(
    a: &[u64; 64],
    b: &[u64; 64],
    n: &[u64; 64],
) -> jolt::UntrustedAdvice<([u64; 64], [u64; 64])> {
    let product = bignum::mul_wide_4096(a, b);
    let (q, r) = bignum::div_rem_wide_4096(&product, n);
    (q, r)
}

pub fn rsa_pkcs1v15_sha256_verify_2048(
    modulus: &[u8; 256],
    exponent: u32,
    signature: &[u8; 256],
    msg_hash: &[u8; 32],
) {
    assert_eq!(
        exponent, 65537,
        "only e=65537 is supported for step-wise RSA"
    );

    let n_limbs = bignum::be_bytes_to_limbs_2048(modulus);
    let sig_limbs = bignum::be_bytes_to_limbs_2048(signature);
    let mut current_limbs = sig_limbs;

    for _ in 0..16 {
        let (q_limbs, r_limbs) = *modmul_2048_step(&current_limbs, &current_limbs, &n_limbs);
        jolt::check_advice!(bignum::verify_modsquare_2048(
            &current_limbs,
            &q_limbs,
            &n_limbs,
            &r_limbs,
        ));
        jolt::check_advice!(bignum::lt_2048(&r_limbs, &n_limbs));
        current_limbs = r_limbs;
    }

    let (q_limbs, r_limbs) = *modmul_2048_step(&current_limbs, &sig_limbs, &n_limbs);
    jolt::check_advice!(bignum::verify_modmul_2048(
        &current_limbs,
        &sig_limbs,
        &q_limbs,
        &n_limbs,
        &r_limbs,
    ));
    jolt::check_advice!(bignum::lt_2048(&r_limbs, &n_limbs));

    let result_bytes = bignum::limbs_to_be_bytes_2048(&r_limbs);
    verify_pkcs1v15_padding(&result_bytes, msg_hash, 256);
}

pub fn rsa_pkcs1v15_sha256_verify_4096(
    modulus: &[u8; 512],
    exponent: u32,
    signature: &[u8; 512],
    msg_hash: &[u8; 32],
) {
    assert_eq!(
        exponent, 65537,
        "only e=65537 is supported for step-wise RSA"
    );

    let n_limbs = bignum::be_bytes_to_limbs_4096(modulus);
    let sig_limbs = bignum::be_bytes_to_limbs_4096(signature);
    let mut current_limbs = sig_limbs;

    for _ in 0..16 {
        let (q_limbs, r_limbs) = *modmul_4096_step(&current_limbs, &current_limbs, &n_limbs);
        jolt::check_advice!(bignum::verify_modsquare_4096(
            &current_limbs,
            &q_limbs,
            &n_limbs,
            &r_limbs,
        ));
        jolt::check_advice!(bignum::lt_4096(&r_limbs, &n_limbs));
        current_limbs = r_limbs;
    }

    let (q_limbs, r_limbs) = *modmul_4096_step(&current_limbs, &sig_limbs, &n_limbs);
    jolt::check_advice!(bignum::verify_modmul_4096(
        &current_limbs,
        &sig_limbs,
        &q_limbs,
        &n_limbs,
        &r_limbs,
    ));
    jolt::check_advice!(bignum::lt_4096(&r_limbs, &n_limbs));

    let result_bytes = bignum::limbs_to_be_bytes_4096(&r_limbs);
    verify_pkcs1v15_padding(&result_bytes, msg_hash, 512);
}

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
