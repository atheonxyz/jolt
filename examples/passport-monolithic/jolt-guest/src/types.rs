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
pub struct GuestInputs {
    #[serde(with = "BigArray")]
    pub dg1: [u8; MAX_DG1_SIZE],
    pub dg1_padded_length: u32,
    pub current_date: u64,
    pub min_age_required: u8,
    pub max_age_required: u8,
    pub passport_validity: GuestPassportValidity,
}

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
