#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;

pub mod bignum;
pub mod date;
pub mod rsa_verify;
pub mod types;
pub mod verify;

use jolt::{end_cycle_tracking, start_cycle_tracking};

#[jolt::provable(stack_size = 8388608, heap_size = 33554432, max_trace_length = 4194304)]
fn complete_age_check(
    current_date: u64,
    min_age_required: u8,
    max_age_required: u8,
    csc_pubkey_public: alloc::vec::Vec<u8>,
    preparsed_buf: jolt::PrivateInput<alloc::vec::Vec<u8>>,
) -> u32 {
    start_cycle_tracking("unpack");
    let (dg1, dg1_padded_length, mut pv) = types::unpack_preparsed_passport(&preparsed_buf);
    pv.csc_pubkey[..csc_pubkey_public.len()].copy_from_slice(&csc_pubkey_public);
    end_cycle_tracking("unpack");

    // Step 1: Age check
    start_cycle_tracking("age_check");
    verify::verify_age(&dg1, min_age_required, max_age_required, current_date);
    end_cycle_tracking("age_check");

    // Step 2: Expiry check
    start_cycle_tracking("expiry_check");
    verify::verify_expiry(&dg1, current_date);
    end_cycle_tracking("expiry_check");

    // Step 3: DG1 hash integrity
    start_cycle_tracking("dg1_hash");
    verify::verify_dg1_hash(&dg1, dg1_padded_length, &pv);
    end_cycle_tracking("dg1_hash");

    // Step 4: Full passport validity chain (SOD → DSC → CSC)
    start_cycle_tracking("sod_integrity");
    verify::check_sod_integrity(&pv);
    end_cycle_tracking("sod_integrity");

    start_cycle_tracking("dsc_signature");
    verify::verify_dsc_signature(&pv);
    end_cycle_tracking("dsc_signature");

    start_cycle_tracking("dsc_pubkey_in_cert");
    verify::verify_dsc_pubkey_in_cert(&pv);
    end_cycle_tracking("dsc_pubkey_in_cert");

    start_cycle_tracking("csc_signature");
    verify::verify_csc_signature(&pv);
    end_cycle_tracking("csc_signature");

    1 // success
}
