#![allow(dead_code)]

use ark_bn254::{Fq, Fq12, Fq2, Fq6, G1Affine, G2Affine};
use ark_ff::BigInt;

pub const FQ_LIMBS: usize = 8;
pub const FP12_WORDS: usize = 96;
pub const G1_WORDS: usize = 16;
pub const G2_WORDS: usize = 32;

pub fn fq_to_limbs(f: &Fq) -> [u32; FQ_LIMBS] {
    let words = (f.0).0;
    let mut out = [0_u32; FQ_LIMBS];

    for (i, word) in words.iter().enumerate() {
        out[i * 2] = *word as u32;
        out[i * 2 + 1] = (*word >> 32) as u32;
    }

    out
}

pub fn limbs8_to_fq(limbs: &[u32]) -> Fq {
    assert_eq!(limbs.len(), FQ_LIMBS);

    let bigint = BigInt::<4>::new([
        ((limbs[1] as u64) << 32) | limbs[0] as u64,
        ((limbs[3] as u64) << 32) | limbs[2] as u64,
        ((limbs[5] as u64) << 32) | limbs[4] as u64,
        ((limbs[7] as u64) << 32) | limbs[6] as u64,
    ]);

    Fq::new_unchecked(bigint)
}

pub fn serialize_g1_affine(points: &[G1Affine]) -> Vec<u32> {
    let mut out = Vec::with_capacity(points.len() * G1_WORDS);
    for p in points {
        out.extend_from_slice(&fq_to_limbs(&p.x));
        out.extend_from_slice(&fq_to_limbs(&p.y));
    }
    out
}

pub fn serialize_g2_affine(points: &[G2Affine]) -> Vec<u32> {
    let mut out = Vec::with_capacity(points.len() * G2_WORDS);
    for p in points {
        out.extend_from_slice(&fq_to_limbs(&p.x.c0));
        out.extend_from_slice(&fq_to_limbs(&p.x.c1));
        out.extend_from_slice(&fq_to_limbs(&p.y.c0));
        out.extend_from_slice(&fq_to_limbs(&p.y.c1));
    }
    out
}

pub fn deserialize_fq12(words: &[u32]) -> Fq12 {
    assert_eq!(words.len(), FP12_WORDS);

    let fq_words: [Fq; 12] = core::array::from_fn(|i| {
        let start = i * FQ_LIMBS;
        limbs8_to_fq(&words[start..start + FQ_LIMBS])
    });

    Fq12::new(
        Fq6::new(
            Fq2::new(fq_words[0], fq_words[1]),
            Fq2::new(fq_words[2], fq_words[3]),
            Fq2::new(fq_words[4], fq_words[5]),
        ),
        Fq6::new(
            Fq2::new(fq_words[6], fq_words[7]),
            Fq2::new(fq_words[8], fq_words[9]),
            Fq2::new(fq_words[10], fq_words[11]),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::{limbs8_to_fq, serialize_g1_affine, FQ_LIMBS, G1_WORDS};
    use ark_bn254::G1Affine;
    use ark_ec::AffineRepr;

    #[test]
    fn g1_generator_roundtrip() {
        let generator = G1Affine::generator();
        let words = serialize_g1_affine(&[generator]);
        assert_eq!(words.len(), G1_WORDS);

        let x = limbs8_to_fq(&words[..FQ_LIMBS]);
        let y = limbs8_to_fq(&words[FQ_LIMBS..G1_WORDS]);
        let decoded = G1Affine::new_unchecked(x, y);
        assert_eq!(decoded, generator);
    }
}
