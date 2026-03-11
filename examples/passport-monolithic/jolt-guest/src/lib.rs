#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;

pub mod bignum;
pub mod date;
pub mod rsa_verify;
pub mod types;
pub mod verify;

use jolt::{end_cycle_tracking, start_cycle_tracking};
use types::GuestInputs;

#[jolt::provable(stack_size = 8388608, heap_size = 33554432, max_trace_length = 4194304)]
fn complete_age_check(inputs: GuestInputs) -> u32 {
    // Step 1: Age check
    start_cycle_tracking("age_check");
    verify::verify_age(
        &inputs.dg1,
        inputs.min_age_required,
        inputs.max_age_required,
        inputs.current_date,
    );
    end_cycle_tracking("age_check");

    // Step 2: Expiry check
    start_cycle_tracking("expiry_check");
    verify::verify_expiry(&inputs.dg1, inputs.current_date);
    end_cycle_tracking("expiry_check");

    // Step 3: DG1 hash integrity
    start_cycle_tracking("dg1_hash");
    // TODO: we can split arguments further to avoid passing the entire struct
    verify::verify_dg1_hash(
        &inputs.dg1,
        inputs.dg1_padded_length,
        &inputs.passport_validity,
    );
    end_cycle_tracking("dg1_hash");

    // Step 4: Full passport validity chain (SOD → DSC → CSC)

    start_cycle_tracking("sod_integrity");
    verify::check_sod_integrity(&inputs.passport_validity);
    end_cycle_tracking("sod_integrity");

    start_cycle_tracking("dsc_signature");
    verify::verify_dsc_signature(&inputs.passport_validity);
    end_cycle_tracking("dsc_signature");

    start_cycle_tracking("dsc_pubkey_in_cert");
    verify::verify_dsc_pubkey_in_cert(&inputs.passport_validity);
    end_cycle_tracking("dsc_pubkey_in_cert");

    start_cycle_tracking("csc_signature");        
    verify::verify_csc_signature(&inputs.passport_validity);
    end_cycle_tracking("csc_signature");
    
    

    1 // success
}
