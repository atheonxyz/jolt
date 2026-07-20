//! Bignum arithmetic for RSA modular multiplication verification.
//!
//! Uses little-endian u64 limb representation. Only the operations needed
//! for step-wise RSA modexp verification are implemented.
//!
//! Guest verification functions (always compiled): mul_wide, verify_modmul, lt
//! Advice-only functions (compute_advice only): div_rem_wide (Knuth's Algorithm D)

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

pub fn verify_modsquare_2048(a: &[u64; 32], q: &[u64; 32], n: &[u64; 32], r: &[u64; 32]) -> bool {
    let a2 = square_wide_2048(a);
    let qn = mul_wide_2048(q, n);
    verify_sum_eq(&a2, &qn, r)
}

pub fn verify_modsquare_4096(a: &[u64; 64], q: &[u64; 64], n: &[u64; 64], r: &[u64; 64]) -> bool {
    let a2 = square_wide_4096(a);
    let qn = mul_wide_4096(q, n);
    verify_sum_eq(&a2, &qn, r)
}

fn verify_sum_eq(wide: &[u64], base: &[u64], ext: &[u64]) -> bool {
    let w = wide.len();
    let e = ext.len();
    let mut carry: u64 = 0;
    for i in 0..w {
        let ext_val = if i < e { ext[i] } else { 0 };
        let (s1, c1) = base[i].overflowing_add(ext_val);
        let (s2, c2) = s1.overflowing_add(carry);
        if s2 != wide[i] { return false; }
        carry = c1 as u64 + c2 as u64;
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

// ── Advice-only division (Knuth's Algorithm D) ──────────────────────────────
//
// Processes one 64-bit quotient limb per iteration instead of one bit.
// For 4096-bit: 65 iterations vs 8192 with bit-by-bit (~70x faster).

#[cfg(feature = "compute_advice")]
pub fn div_rem_wide_2048(dividend: &[u64; 64], divisor: &[u64; 32]) -> ([u64; 32], [u64; 32]) {
    let mut q = [0u64; 64];
    let mut r = [0u64; 32];
    knuth_div(dividend, divisor, &mut q, &mut r);
    let mut quo = [0u64; 32];
    quo.copy_from_slice(&q[..32]);
    (quo, r)
}

#[cfg(feature = "compute_advice")]
pub fn div_rem_wide_4096(dividend: &[u64; 128], divisor: &[u64; 64]) -> ([u64; 64], [u64; 64]) {
    let mut q = [0u64; 128];
    let mut r = [0u64; 64];
    knuth_div(dividend, divisor, &mut q, &mut r);
    let mut quo = [0u64; 64];
    quo.copy_from_slice(&q[..64]);
    (quo, r)
}

/// Knuth's Algorithm D: multi-precision division (dividend / divisor → quo, rem).
/// dividend has `dlen` significant limbs, divisor has `n` significant limbs.
#[cfg(feature = "compute_advice")]
fn knuth_div(dividend: &[u64], divisor: &[u64], quo: &mut [u64], rem: &mut [u64]) {
    let n = {
        let mut n = divisor.len();
        while n > 0 && divisor[n - 1] == 0 { n -= 1; }
        n
    };
    assert!(n > 0, "division by zero");

    let m_plus_n = {
        let mut l = dividend.len();
        while l > 0 && dividend[l - 1] == 0 { l -= 1; }
        l
    };
    if m_plus_n < n {
        // dividend < divisor → quo=0, rem=dividend
        for i in 0..quo.len() { quo[i] = 0; }
        for i in 0..rem.len().min(dividend.len()) { rem[i] = dividend[i]; }
        return;
    }
    let m = m_plus_n - n; // number of quotient limbs - 1

    // D1: Normalize — shift so MSB of divisor top limb is set
    let shift = divisor[n - 1].leading_zeros();

    // Shifted divisor
    let mut v = [0u64; 65];
    if shift > 0 {
        for i in (1..n).rev() {
            v[i] = (divisor[i] << shift) | (divisor[i - 1] >> (64 - shift));
        }
        v[0] = divisor[0] << shift;
    } else {
        for i in 0..n { v[i] = divisor[i]; }
    }

    // Shifted dividend (one extra limb)
    let ulen = m_plus_n + 1;
    let mut u = [0u64; 129];
    if shift > 0 {
        u[m_plus_n] = dividend[m_plus_n - 1] >> (64 - shift);
        for i in (1..m_plus_n).rev() {
            u[i] = (dividend[i] << shift) | (dividend[i - 1] >> (64 - shift));
        }
        u[0] = dividend[0] << shift;
    } else {
        for i in 0..m_plus_n { u[i] = dividend[i]; }
    }
    let _ = ulen;

    for i in 0..quo.len() { quo[i] = 0; }

    // D2-D7: Main loop
    for j in (0..=m).rev() {
        // D3: Estimate q_hat
        let u_jn = u[j + n];
        let u_jn1 = u[j + n - 1];
        let v_n1 = v[n - 1];

        let (mut q_hat, mut r_hat): (u64, u64);
        if u_jn >= v_n1 {
            q_hat = u64::MAX;
            // r_hat = u_jn - v_n1 + u_jn1 (might overflow, handled by correction)
            let (rh, _) = u_jn.overflowing_sub(v_n1);
            let (rh2, ov) = rh.overflowing_add(u_jn1);
            r_hat = rh2;
            if !ov && n >= 2 {
                // Try correction
                let v_n2 = v[n - 2];
                let u_jn2 = u[j + n - 2];
                let (qv_hi, qv_lo) = mul_u64(q_hat, v_n2);
                if qv_hi > r_hat || (qv_hi == r_hat && qv_lo > u_jn2) {
                    q_hat -= 1;
                }
            }
        } else {
            let numer = ((u_jn as u128) << 64) | (u_jn1 as u128);
            q_hat = (numer / (v_n1 as u128)) as u64;
            r_hat = (numer % (v_n1 as u128)) as u64;

            // D3 correction
            if n >= 2 {
                let v_n2 = v[n - 2];
                let u_jn2 = u[j + n - 2];
                loop {
                    let (qv_hi, qv_lo) = mul_u64(q_hat, v_n2);
                    if qv_hi < r_hat || (qv_hi == r_hat && qv_lo <= u_jn2) {
                        break;
                    }
                    q_hat -= 1;
                    let (new_r, overflow) = r_hat.overflowing_add(v_n1);
                    if overflow { break; }
                    r_hat = new_r;
                }
            }
        }

        // D4: Multiply and subtract: u[j..j+n+1] -= q_hat * v[0..n]
        let mut carry: u64 = 0;
        let mut borrow: u64 = 0;
        for i in 0..n {
            let prod = (q_hat as u128) * (v[i] as u128) + (carry as u128);
            let prod_lo = prod as u64;
            carry = (prod >> 64) as u64;
            let (d1, b1) = u[j + i].overflowing_sub(prod_lo);
            let (d2, b2) = d1.overflowing_sub(borrow);
            u[j + i] = d2;
            borrow = b1 as u64 + b2 as u64;
        }
        let (d1, b1) = u[j + n].overflowing_sub(carry);
        let (d2, b2) = d1.overflowing_sub(borrow);
        u[j + n] = d2;
        let underflow = b1 as u64 + b2 as u64;

        // D5-D6: If we subtracted too much, add back
        if underflow != 0 {
            q_hat -= 1;
            let mut c: u64 = 0;
            for i in 0..n {
                let (s1, c1) = u[j + i].overflowing_add(v[i]);
                let (s2, c2) = s1.overflowing_add(c);
                u[j + i] = s2;
                c = c1 as u64 + c2 as u64;
            }
            u[j + n] = u[j + n].wrapping_add(c);
        }

        if j < quo.len() {
            quo[j] = q_hat;
        }
    }

    // D8: De-normalize remainder
    if shift > 0 {
        for i in 0..n - 1 {
            rem[i] = (u[i] >> shift) | (u[i + 1] << (64 - shift));
        }
        rem[n - 1] = u[n - 1] >> shift;
    } else {
        for i in 0..n.min(rem.len()) { rem[i] = u[i]; }
    }
    // Zero remaining rem limbs
    for i in n..rem.len() { rem[i] = 0; }
}

#[cfg(feature = "compute_advice")]
fn mul_u64(a: u64, b: u64) -> (u64, u64) {
    let p = (a as u128) * (b as u128);
    ((p >> 64) as u64, p as u64)
}
