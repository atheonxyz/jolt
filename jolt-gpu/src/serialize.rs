use ark_bn254::{Fq, Fq12, Fq2, Fq6, G1Affine, G2Affine};
use ark_ff::biginteger::BigInt;

pub const FQ_LIMBS: usize = 8;

#[inline(always)]
pub fn fq_to_limbs(f: &Fq) -> [u32; FQ_LIMBS] {
    let mut out = [0u32; FQ_LIMBS];
    let words = (f.0).0;
    for i in 0..4 {
        out[i * 2] = words[i] as u32;
        out[i * 2 + 1] = (words[i] >> 32) as u32;
    }
    out
}

#[inline(always)]
pub fn limbs8_to_fq(limbs: &[u32]) -> Fq {
    let bigint = BigInt::<4>::new([
        (u64::from(limbs[1]) << 32) | u64::from(limbs[0]),
        (u64::from(limbs[3]) << 32) | u64::from(limbs[2]),
        (u64::from(limbs[5]) << 32) | u64::from(limbs[4]),
        (u64::from(limbs[7]) << 32) | u64::from(limbs[6]),
    ]);
    Fq::new_unchecked(bigint)
}

pub fn serialize_g1_affine(points: &[G1Affine]) -> Vec<u32> {
    let mut out = Vec::with_capacity(points.len() * 2 * FQ_LIMBS);
    for point in points {
        out.extend_from_slice(&fq_to_limbs(&point.x));
        out.extend_from_slice(&fq_to_limbs(&point.y));
    }
    out
}

pub fn serialize_g2_affine(points: &[G2Affine]) -> Vec<u32> {
    let mut out = Vec::with_capacity(points.len() * 4 * FQ_LIMBS);
    for point in points {
        out.extend_from_slice(&fq_to_limbs(&point.x.c0));
        out.extend_from_slice(&fq_to_limbs(&point.x.c1));
        out.extend_from_slice(&fq_to_limbs(&point.y.c0));
        out.extend_from_slice(&fq_to_limbs(&point.y.c1));
    }
    out
}

pub fn deserialize_fq12(words: &[u32]) -> Fq12 {
    debug_assert_eq!(words.len(), 12 * FQ_LIMBS);
    let w: Vec<Fq> = words.chunks(FQ_LIMBS).map(limbs8_to_fq).collect();
    Fq12::new(
        Fq6::new(
            Fq2::new(w[0], w[1]),
            Fq2::new(w[2], w[3]),
            Fq2::new(w[4], w[5]),
        ),
        Fq6::new(
            Fq2::new(w[6], w[7]),
            Fq2::new(w[8], w[9]),
            Fq2::new(w[10], w[11]),
        ),
    )
}
