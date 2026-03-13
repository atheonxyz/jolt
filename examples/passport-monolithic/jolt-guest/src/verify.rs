use jolt_inlines_sha2::Sha256;

use crate::date::SimpleDate;
use crate::rsa_verify::{rsa_pkcs1v15_sha256_verify_2048, rsa_pkcs1v15_sha256_verify_4096};
use crate::types::*;

// ── Age check ──────────────────────────────────────────────────────────────

/// Port of Noir `compare_age(dg1, min_age, max_age, timestamp)`.
pub fn verify_age(dg1: &[u8; MAX_DG1_SIZE], min_age: u8, max_age: u8, timestamp: u64) {
    assert!(min_age < 100 && max_age < 100, "ages must be < 100");
    assert!(
        min_age != 0 || max_age != 0,
        "at least one age bound must be non-zero"
    );

    let current_date = SimpleDate::from_timestamp(timestamp);

    // Extract birthdate from MRZ
    let is_id = is_id_card(dg1);
    let bd_idx = if is_id {
        DG1_TO_MRZ_OFFSET + ID_CARD_MRZ_BIRTHDATE_INDEX
    } else {
        DG1_TO_MRZ_OFFSET + PASSPORT_MRZ_BIRTHDATE_INDEX
    };
    let bd_bytes: [u8; 6] = [
        dg1[bd_idx],
        dg1[bd_idx + 1],
        dg1[bd_idx + 2],
        dg1[bd_idx + 3],
        dg1[bd_idx + 4],
        dg1[bd_idx + 5],
    ];
    let birthdate = SimpleDate::from_yymmdd_bytes(&bd_bytes, current_date.year);

    // Age range checks (matching Noir logic exactly):
    //   min_age: inclusive (>=)
    //   max_age: inclusive via < birthdate + (max_age+1)
    if min_age != 0 && max_age == 0 {
        // Only minimum age check
        let min_date = birthdate.add_years(min_age as u32);
        assert!(current_date.gte(min_date), "age below minimum required age");
    } else if max_age != 0 && min_age == 0 {
        // Only maximum age check
        let max_date = birthdate.add_years(max_age as u32);
        assert!(current_date.lt(max_date), "age above maximum allowed age");
    } else {
        // Both bounds
        assert!(min_age <= max_age, "min_age must be <= max_age");
        let min_date = birthdate.add_years(min_age as u32);
        let max_date = birthdate.add_years(max_age as u32 + 1);
        assert!(current_date.gte(min_date), "age below minimum required age");
        assert!(current_date.lt(max_date), "age above maximum allowed age");
    }
}

// ── Expiry check ───────────────────────────────────────────────────────────

/// Port of Noir `check_expiry(dg1, current_date_timestamp)`.
pub fn verify_expiry(dg1: &[u8; MAX_DG1_SIZE], timestamp: u64) {
    let current_date = SimpleDate::from_timestamp(timestamp);

    let is_id = is_id_card(dg1);
    let exp_idx = if is_id {
        DG1_TO_MRZ_OFFSET + ID_CARD_MRZ_EXPIRY_DATE_INDEX
    } else {
        DG1_TO_MRZ_OFFSET + PASSPORT_MRZ_EXPIRY_DATE_INDEX
    };
    let exp_bytes: [u8; 6] = [
        dg1[exp_idx],
        dg1[exp_idx + 1],
        dg1[exp_idx + 2],
        dg1[exp_idx + 3],
        dg1[exp_idx + 4],
        dg1[exp_idx + 5],
    ];

    // Pivot year = current year + 30 (per Noir expiry logic)
    let expiry_date = SimpleDate::from_yymmdd_bytes(&exp_bytes, current_date.year + 30);

    assert!(current_date.lt(expiry_date), "document is expired");
}

// ── DG1 hash integrity ────────────────────────────────────────────────────

/// Port of Noir `check_dg1_hash_within_sod`.
/// Verifies SHA-256(dg1[0..padded_len]) matches econtent at dg1_hash_offset.
pub fn verify_dg1_hash(
    dg1: &[u8; MAX_DG1_SIZE],
    dg1_padded_length: u32,
    pv: &GuestPassportValidity,
) {
    let len = dg1_padded_length as usize;
    let offset = pv.dg1_hash_offset as usize;
    let econtent_len = pv.econtent_len as usize;

    assert!(len <= MAX_DG1_SIZE, "dg1_padded_length exceeds DG1 size");
    assert!(
        offset + 32 <= econtent_len,
        "dg1_hash_offset out of bounds in econtent"
    );

    // Verify zero-padding
    check_zero_padding(&pv.econtent, econtent_len);

    // Compute SHA-256 of actual DG1 data
    let dg1_hash = Sha256::digest(&dg1[..len]);

    // Compare with hash in econtent
    for i in 0..32 {
        assert_eq!(
            dg1_hash[i],
            pv.econtent[offset + i],
            "DG1 hash mismatch at byte {}",
            i
        );
    }
}

// ── Passport validity chain ───────────────────────────────────────────────

/// Port of Noir `check_passport_validity(passport_validity_contents)`.
/// Verifies the full certificate chain: SOD → DSC → CSC.
pub fn verify_passport_validity(pv: &GuestPassportValidity) {
    // Step 1: SOD integrity — SHA-256(eContent) in signed_attributes
    check_sod_integrity(pv);

    // Step 2: DSC signature — RSA-2048 PKCS#1v1.5 SHA-256 over signed_attributes
    verify_dsc_signature(pv);

    // Step 3: DSC pubkey in cert — byte match at offset
    verify_dsc_pubkey_in_cert(pv);

    // Step 4: CSC signature — RSA-4096 PKCS#1v1.5 SHA-256 over DSC cert
    verify_csc_signature(pv);
}

/// Step 1: Check SHA-256(eContent) matches signed_attributes at econtent_hash_offset.
pub fn check_sod_integrity(pv: &GuestPassportValidity) {
    let sa_size = pv.signed_attributes_size as usize;
    let ec_len = pv.econtent_len as usize;
    let offset = pv.econtent_hash_offset as usize;

    check_zero_padding(&pv.signed_attributes, sa_size);
    assert!(
        offset + 32 <= sa_size,
        "econtent_hash_offset out of bounds in signed_attributes"
    );

    let econtent_hash = Sha256::digest(&pv.econtent[..ec_len]);

    for i in 0..32 {
        assert_eq!(
            econtent_hash[i],
            pv.signed_attributes[offset + i],
            "eContent hash mismatch in signed_attributes at byte {}",
            i
        );
    }
}

/// Step 2: Verify DSC signature (RSA-2048) over signed_attributes.
pub fn verify_dsc_signature(pv: &GuestPassportValidity) {
    let sa_size = pv.signed_attributes_size as usize;
    let hash = Sha256::digest(&pv.signed_attributes[..sa_size]);

    rsa_pkcs1v15_sha256_verify_2048(
        &pv.dsc_pubkey,
        pv.dsc_rsa_exponent,
        &pv.dsc_signature,
        &hash,
    );
}

/// Step 3: Verify DSC pubkey bytes appear in DSC cert at the expected offset.
pub fn verify_dsc_pubkey_in_cert(pv: &GuestPassportValidity) {
    let offset = pv.dsc_pubkey_offset_in_dsc_cert as usize;
    let cert_len = pv.dsc_cert_len as usize;

    assert!(
        offset + DSC_SIG_BYTES <= cert_len,
        "DSC pubkey offset out of bounds in cert"
    );

    for i in 0..DSC_SIG_BYTES {
        assert_eq!(
            pv.dsc_cert[offset + i],
            pv.dsc_pubkey[i],
            "DSC pubkey mismatch in cert at byte {}",
            i
        );
    }
}

/// Step 4: Verify CSC signature (RSA-4096) over DSC certificate.
pub fn verify_csc_signature(pv: &GuestPassportValidity) {
    let cert_len = pv.dsc_cert_len as usize;

    check_zero_padding(&pv.dsc_cert, cert_len);

    let hash = Sha256::digest(&pv.dsc_cert[..cert_len]);

    rsa_pkcs1v15_sha256_verify_4096(
        &pv.csc_pubkey,
        pv.csc_rsa_exponent,
        &pv.dsc_cert_signature,
        &hash,
    );
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Check if the document is an ID card (TD1) vs passport (TD3).
/// ID cards have non-zero bytes at positions 93 and 94.
fn is_id_card(dg1: &[u8; MAX_DG1_SIZE]) -> bool {
    dg1[93] != 0 && dg1[94] != 0
}

/// Verify that all bytes beyond `len` are zero (matches Noir's `check_zero_padding`).
fn check_zero_padding<const N: usize>(arr: &[u8; N], len: usize) {
    for i in len..N {
        assert_eq!(arr[i], 0, "non-zero byte in padding at index {}", i);
    }
}
