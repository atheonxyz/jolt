use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

pub const MAX_DG1_SIZE: usize = 95;
pub const MAX_SIGNED_ATTRIBUTES_SIZE: usize = 200;
pub const MAX_ECONTENT_SIZE: usize = 200;
pub const DSC_SIG_BYTES: usize = 256; // RSA-2048
pub const CSC_SIG_BYTES: usize = 512; // RSA-4096
pub const MAX_TBS_SIZE: usize = 1300;

pub const DG1_TO_MRZ_OFFSET: usize = 5;
pub const MRZ_SIZE: usize = 90;
pub const PASSPORT_MRZ_BIRTHDATE_INDEX: usize = 57;
pub const ID_CARD_MRZ_BIRTHDATE_INDEX: usize = 30;
pub const PASSPORT_MRZ_EXPIRY_DATE_INDEX: usize = 65;
pub const ID_CARD_MRZ_EXPIRY_DATE_INDEX: usize = 38;

#[derive(Clone, Serialize, Deserialize)]
pub struct GuestPassportValidity {
    #[serde(with = "BigArray")]
    pub signed_attributes: [u8; MAX_SIGNED_ATTRIBUTES_SIZE],
    pub signed_attributes_size: u32,
    #[serde(with = "BigArray")]
    pub econtent: [u8; MAX_ECONTENT_SIZE],
    pub econtent_len: u32,
    pub dg1_hash_offset: u32,
    pub econtent_hash_offset: u32,

    #[serde(with = "BigArray")]
    pub dsc_signature: [u8; DSC_SIG_BYTES],
    pub dsc_rsa_exponent: u32,
    #[serde(with = "BigArray")]
    pub dsc_pubkey: [u8; DSC_SIG_BYTES],

    pub dsc_pubkey_offset_in_dsc_cert: u32,
    #[serde(with = "BigArray")]
    pub dsc_cert: [u8; MAX_TBS_SIZE],
    pub dsc_cert_len: u32,

    #[serde(with = "BigArray")]
    pub csc_pubkey: [u8; CSC_SIG_BYTES],
    #[serde(with = "BigArray")]
    pub dsc_cert_signature: [u8; CSC_SIG_BYTES],
    pub csc_rsa_exponent: u32,
}

// ── Pack/unpack helpers ──

pub fn pack_field(buf: &mut alloc::vec::Vec<u8>, field: &[u8]) {
    buf.extend_from_slice(&(field.len() as u32).to_le_bytes());
    buf.extend_from_slice(field);
    let pad = (4 - (field.len() % 4)) % 4;
    for _ in 0..pad {
        buf.push(0);
    }
}

fn pack_u32(buf: &mut alloc::vec::Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

pub fn take_slice(buf: &[u8]) -> (&[u8], &[u8]) {
    let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let pad = (4 - (len % 4)) % 4;
    (&buf[4..4 + len], &buf[4 + len + pad..])
}

pub fn take_u32(buf: &[u8]) -> (u32, &[u8]) {
    let val = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    (val, &buf[4..])
}

/// Pack DG1 bytes and all private passport crypto fields into a flat length-prefixed buffer.
/// `csc_pubkey` is NOT packed — it is a public input passed separately.
pub fn pack_preparsed_passport(dg1: &[u8], pv: &GuestPassportValidity) -> alloc::vec::Vec<u8> {
    let mut buf = alloc::vec::Vec::new();
    // 1. dg1 (variable — dg1_padded_length implicit in slice length)
    pack_field(&mut buf, dg1);
    // 2. signed_attributes (variable)
    pack_field(&mut buf, &pv.signed_attributes[..pv.signed_attributes_size as usize]);
    // 3. econtent (variable)
    pack_field(&mut buf, &pv.econtent[..pv.econtent_len as usize]);
    // 4-5. offsets
    pack_u32(&mut buf, pv.dg1_hash_offset);
    pack_u32(&mut buf, pv.econtent_hash_offset);
    // 6. dsc_signature (fixed 256)
    pack_field(&mut buf, &pv.dsc_signature);
    // 7. dsc_rsa_exponent
    pack_u32(&mut buf, pv.dsc_rsa_exponent);
    // 8. dsc_pubkey (fixed 256)
    pack_field(&mut buf, &pv.dsc_pubkey);
    // 9. dsc_pubkey_offset_in_dsc_cert
    pack_u32(&mut buf, pv.dsc_pubkey_offset_in_dsc_cert);
    // 10. dsc_cert (variable)
    pack_field(&mut buf, &pv.dsc_cert[..pv.dsc_cert_len as usize]);
    // 11. dsc_cert_signature (fixed 512)
    pack_field(&mut buf, &pv.dsc_cert_signature);
    // 12. csc_rsa_exponent
    pack_u32(&mut buf, pv.csc_rsa_exponent);
    buf
}

/// Unpack the private buffer into (dg1, dg1_padded_length, GuestPassportValidity).
/// `pv.csc_pubkey` is zeroed — caller must fill it from the public `csc_pubkey` argument.
pub fn unpack_preparsed_passport(buf: &[u8]) -> ([u8; MAX_DG1_SIZE], u32, GuestPassportValidity) {
    // 1. dg1
    let (dg1_data, buf) = take_slice(buf);
    let mut dg1 = [0u8; MAX_DG1_SIZE];
    dg1[..dg1_data.len()].copy_from_slice(dg1_data);
    let dg1_padded_length = dg1_data.len() as u32;

    // 2. signed_attributes
    let (sa_data, buf) = take_slice(buf);
    let mut signed_attributes = [0u8; MAX_SIGNED_ATTRIBUTES_SIZE];
    signed_attributes[..sa_data.len()].copy_from_slice(sa_data);
    let signed_attributes_size = sa_data.len() as u32;

    // 3. econtent
    let (ec_data, buf) = take_slice(buf);
    let mut econtent = [0u8; MAX_ECONTENT_SIZE];
    econtent[..ec_data.len()].copy_from_slice(ec_data);
    let econtent_len = ec_data.len() as u32;

    // 4-5. offsets
    let (dg1_hash_offset, buf) = take_u32(buf);
    let (econtent_hash_offset, buf) = take_u32(buf);

    // 6. dsc_signature
    let (sig_data, buf) = take_slice(buf);
    let mut dsc_signature = [0u8; DSC_SIG_BYTES];
    dsc_signature[..sig_data.len()].copy_from_slice(sig_data);

    // 7. dsc_rsa_exponent
    let (dsc_rsa_exponent, buf) = take_u32(buf);

    // 8. dsc_pubkey
    let (pk_data, buf) = take_slice(buf);
    let mut dsc_pubkey = [0u8; DSC_SIG_BYTES];
    dsc_pubkey[..pk_data.len()].copy_from_slice(pk_data);

    // 9. dsc_pubkey_offset_in_dsc_cert
    let (dsc_pubkey_offset_in_dsc_cert, buf) = take_u32(buf);

    // 10. dsc_cert
    let (cert_data, buf) = take_slice(buf);
    let mut dsc_cert = [0u8; MAX_TBS_SIZE];
    dsc_cert[..cert_data.len()].copy_from_slice(cert_data);
    let dsc_cert_len = cert_data.len() as u32;

    // 11. dsc_cert_signature
    let (cert_sig_data, buf) = take_slice(buf);
    let mut dsc_cert_signature = [0u8; CSC_SIG_BYTES];
    dsc_cert_signature[..cert_sig_data.len()].copy_from_slice(cert_sig_data);

    // 12. csc_rsa_exponent
    let (csc_rsa_exponent, _buf) = take_u32(buf);

    let pv = GuestPassportValidity {
        signed_attributes,
        signed_attributes_size,
        econtent,
        econtent_len,
        dg1_hash_offset,
        econtent_hash_offset,
        dsc_signature,
        dsc_rsa_exponent,
        dsc_pubkey,
        dsc_pubkey_offset_in_dsc_cert,
        dsc_cert,
        dsc_cert_len,
        csc_pubkey: [0u8; CSC_SIG_BYTES], // filled by caller from public arg
        dsc_cert_signature,
        csc_rsa_exponent,
    };

    (dg1, dg1_padded_length, pv)
}
