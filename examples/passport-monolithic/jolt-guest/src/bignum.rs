//! Minimal bignum arithmetic for RSA modular multiplication verification.
//!
//! Uses little-endian u64 limb representation. Only the operations needed
//! for step-wise RSA modexp verification are implemented.
//!
//! Guest verification functions (always compiled): mul_wide, verify_modmul, lt
//! Advice-only functions (compute_advice only): div_rem_wide

pub fn be_bytes_to_limbs_2048(bytes: &[u8; 256]) -> [u64; 32] {
    let mut limbs = [0u64; 32];
    be_bytes_to_limbs_inner(bytes, &mut limbs);
    limbs
}

pub fn be_bytes_to_limbs_4096(bytes: &[u8; 512]) -> [u64; 64] {
    let mut limbs = [0u64; 64];
    be_bytes_to_limbs_inner(bytes, &mut limbs);
    limbs
}

pub fn limbs_to_be_bytes_2048(limbs: &[u64; 32]) -> [u8; 256] {
    let mut bytes = [0u8; 256];
    limbs_to_be_bytes_inner(limbs, &mut bytes);
    bytes
}

pub fn limbs_to_be_bytes_4096(limbs: &[u64; 64]) -> [u8; 512] {
    let mut bytes = [0u8; 512];
    limbs_to_be_bytes_inner(limbs, &mut bytes);
    bytes
}

fn be_bytes_to_limbs_inner(bytes: &[u8], limbs: &mut [u64]) {
    let n_bytes = bytes.len();
    for i in 0..limbs.len() {
        let off = n_bytes - (i + 1) * 8;
        limbs[i] = u64::from_be_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]);
    }
}

fn limbs_to_be_bytes_inner(limbs: &[u64], bytes: &mut [u8]) {
    let n_bytes = bytes.len();
    for i in 0..limbs.len() {
        let be = limbs[i].to_be_bytes();
        let off = n_bytes - (i + 1) * 8;
        bytes[off..off + 8].copy_from_slice(&be);
    }
}

pub fn mul_wide_2048(a: &[u64; 32], b: &[u64; 32]) -> [u64; 64] {
    let mut result = [0u64; 64];
    mul_wide_inner(a, b, &mut result);
    result
}

pub fn mul_wide_4096(a: &[u64; 64], b: &[u64; 64]) -> [u64; 128] {
    let mut result = [0u64; 128];
    mul_wide_inner(a, b, &mut result);
    result
}

fn mul_wide_inner(a: &[u64], b: &[u64], result: &mut [u64]) {
    let n = a.len();
    for i in 0..result.len() {
        result[i] = 0;
    }
    for i in 0..n {
        let mut carry: u64 = 0;
        for j in 0..n {
            let prod = (a[i] as u128) * (b[j] as u128) + (result[i + j] as u128) + (carry as u128);
            result[i + j] = prod as u64;
            carry = (prod >> 64) as u64;
        }
        result[i + n] = carry;
    }
}

pub fn square_wide_2048(a: &[u64; 32]) -> [u64; 64] {
    let mut result = [0u64; 64];
    square_wide_inner(a, &mut result);
    result
}

pub fn square_wide_4096(a: &[u64; 64]) -> [u64; 128] {
    let mut result = [0u64; 128];
    square_wide_inner(a, &mut result);
    result
}

/// Squaring exploits symmetry: a[i]*a[j] == a[j]*a[i] for i != j,
/// halving the number of multiplications (528 vs 1024 for 32 limbs).
fn square_wide_inner(a: &[u64], result: &mut [u64]) {
    let n = a.len();
    for i in 0..result.len() {
        result[i] = 0;
    }

    // Off-diagonal products: accumulate a[i]*a[j] for i < j
    for i in 0..n {
        let mut carry: u64 = 0;
        for j in (i + 1)..n {
            let prod = (a[i] as u128) * (a[j] as u128) + (result[i + j] as u128) + (carry as u128);
            result[i + j] = prod as u64;
            carry = (prod >> 64) as u64;
        }
        result[i + n] = carry;
    }

    // Double the off-diagonal sum (each cross-term appears twice)
    let mut carry: u64 = 0;
    for i in 0..2 * n {
        let doubled = (result[i] as u128) * 2 + (carry as u128);
        result[i] = doubled as u64;
        carry = (doubled >> 64) as u64;
    }

    // Add diagonal products: a[i]*a[i] at positions result[2*i]
    let mut carry: u64 = 0;
    for i in 0..n {
        let prod = (a[i] as u128) * (a[i] as u128) + (result[2 * i] as u128) + (carry as u128);
        result[2 * i] = prod as u64;
        carry = (prod >> 64) as u64;
        let sum = (result[2 * i + 1] as u128) + (carry as u128);
        result[2 * i + 1] = sum as u64;
        carry = (sum >> 64) as u64;
    }
}

/// Verify a * b == q * n + r for 2048-bit operands (32 u64 limbs each).
pub fn verify_modmul_2048(
    a: &[u64; 32],
    b: &[u64; 32],
    q: &[u64; 32],
    n: &[u64; 32],
    r: &[u64; 32],
) -> bool {
    let ab = mul_wide_2048(a, b);
    let qn = mul_wide_2048(q, n);
    verify_sum_eq(&ab, &qn, r)
}

/// Verify a * b == q * n + r for 4096-bit operands (64 u64 limbs each).
pub fn verify_modmul_4096(
    a: &[u64; 64],
    b: &[u64; 64],
    q: &[u64; 64],
    n: &[u64; 64],
    r: &[u64; 64],
) -> bool {
    let ab = mul_wide_4096(a, b);
    let qn = mul_wide_4096(q, n);
    verify_sum_eq(&ab, &qn, r)
}

/// Verify a^2 == q * n + r for 2048-bit operands using optimized squaring.
pub fn verify_modsquare_2048(a: &[u64; 32], q: &[u64; 32], n: &[u64; 32], r: &[u64; 32]) -> bool {
    let a2 = square_wide_2048(a);
    let qn = mul_wide_2048(q, n);
    verify_sum_eq(&a2, &qn, r)
}

/// Verify a^2 == q * n + r for 4096-bit operands using optimized squaring.
pub fn verify_modsquare_4096(a: &[u64; 64], q: &[u64; 64], n: &[u64; 64], r: &[u64; 64]) -> bool {
    let a2 = square_wide_4096(a);
    let qn = mul_wide_4096(q, n);
    verify_sum_eq(&a2, &qn, r)
}

/// Check wide == base + ext, where ext is zero-extended to the width of wide/base.
fn verify_sum_eq(wide: &[u64], base: &[u64], ext: &[u64]) -> bool {
    let w = wide.len();
    let e = ext.len();
    let mut carry: u64 = 0;
    for i in 0..w {
        let ext_val = if i < e { ext[i] } else { 0 };
        let sum = (base[i] as u128) + (ext_val as u128) + (carry as u128);
        if (sum as u64) != wide[i] {
            return false;
        }
        carry = (sum >> 64) as u64;
    }
    carry == 0
}

pub fn lt_2048(a: &[u64; 32], b: &[u64; 32]) -> bool {
    lt_inner(a, b)
}

pub fn lt_4096(a: &[u64; 64], b: &[u64; 64]) -> bool {
    lt_inner(a, b)
}

fn lt_inner(a: &[u64], b: &[u64]) -> bool {
    for i in (0..a.len()).rev() {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    false
}

#[cfg(feature = "compute_advice")]
pub fn div_rem_wide_2048(dividend: &[u64; 64], divisor: &[u64; 32]) -> ([u64; 32], [u64; 32]) {
    let mut rem = [0u64; 33];
    let mut quo = [0u64; 32];

    for bit_pos in (0..4096).rev() {
        shift_left_1(&mut rem);
        rem[0] |= (dividend[bit_pos / 64] >> (bit_pos % 64)) & 1;

        if ge_with_extra(&rem, divisor) {
            sub_with_extra(&mut rem, divisor);
            if bit_pos < 2048 {
                quo[bit_pos / 64] |= 1u64 << (bit_pos % 64);
            }
        }
    }

    let mut remainder = [0u64; 32];
    remainder.copy_from_slice(&rem[..32]);
    (quo, remainder)
}

#[cfg(feature = "compute_advice")]
pub fn div_rem_wide_4096(dividend: &[u64; 128], divisor: &[u64; 64]) -> ([u64; 64], [u64; 64]) {
    let mut rem = [0u64; 65];
    let mut quo = [0u64; 64];

    for bit_pos in (0..8192).rev() {
        shift_left_1(&mut rem);
        rem[0] |= (dividend[bit_pos / 64] >> (bit_pos % 64)) & 1;

        if ge_with_extra(&rem, divisor) {
            sub_with_extra(&mut rem, divisor);
            if bit_pos < 4096 {
                quo[bit_pos / 64] |= 1u64 << (bit_pos % 64);
            }
        }
    }

    let mut remainder = [0u64; 64];
    remainder.copy_from_slice(&rem[..64]);
    (quo, remainder)
}

#[cfg(feature = "compute_advice")]
fn shift_left_1(limbs: &mut [u64]) {
    let mut carry = 0u64;
    for limb in limbs.iter_mut() {
        let new_carry = *limb >> 63;
        *limb = (*limb << 1) | carry;
        carry = new_carry;
    }
}

/// rem (n+1 limbs) >= div (n limbs)?
#[cfg(feature = "compute_advice")]
fn ge_with_extra(rem: &[u64], div: &[u64]) -> bool {
    let n = div.len();
    if rem[n] != 0 {
        return true;
    }
    for i in (0..n).rev() {
        if rem[i] > div[i] {
            return true;
        }
        if rem[i] < div[i] {
            return false;
        }
    }
    true
}

/// rem (n+1 limbs) -= div (n limbs), assuming rem >= div.
#[cfg(feature = "compute_advice")]
fn sub_with_extra(rem: &mut [u64], div: &[u64]) {
    let n = div.len();
    let mut borrow: u64 = 0;
    for i in 0..n {
        let (d1, b1) = rem[i].overflowing_sub(div[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        rem[i] = d2;
        borrow = b1 as u64 + b2 as u64;
    }
    rem[n] = rem[n].wrapping_sub(borrow);
}
