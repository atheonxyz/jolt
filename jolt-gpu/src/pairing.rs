use ark_bn254::{Bn254, Fq12, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use rayon::prelude::*;

use crate::context::WgpuContext;
use crate::dispatch::{
    create_staging_buffer, create_storage_buffer, create_uniform_buffer, dispatch_compute,
    div_ceil, write_buffer,
};
use crate::error::GpuError;
use crate::pipeline::ShaderRegistry;
use crate::serialize::{deserialize_fq12, serialize_g1_affine, serialize_g2_affine};

const FP12_WORDS: u32 = 96;
const MILLER_WORKGROUP_SIZE: u32 = 64;
const REDUCE_WORKGROUP_SIZE: u32 = 256;
const GPU_RATIO: f64 = 0.95;
const MIN_HYBRID_PAIRS: usize = 20;

fn gpu_miller_loop(
    ctx: &WgpuContext,
    registry: &ShaderRegistry,
    g1_flat: &[u32],
    g2_flat: &[u32],
    total_pairs: u32,
    num_g2_bases: u32,
) -> Result<(wgpu::Buffer, wgpu::CommandEncoder), GpuError> {
    let pipeline = registry
        .get_pipeline("miller_loop_kernel")
        .ok_or_else(|| GpuError::DispatchError("missing miller_loop_kernel pipeline".into()))?;

    let g1_bytes = bytemuck::cast_slice::<u32, u8>(g1_flat);
    let g2_bytes = bytemuck::cast_slice::<u32, u8>(g2_flat);
    let results_size = (total_pairs as u64) * (FP12_WORDS as u64) * 4;

    let g1_buf = create_storage_buffer(&ctx.device, g1_bytes.len() as u64, Some("miller-g1"));
    let g2_buf = create_storage_buffer(&ctx.device, g2_bytes.len() as u64, Some("miller-g2"));
    let results_buf = create_storage_buffer(&ctx.device, results_size, Some("miller-results"));

    write_buffer(&ctx.queue, &g1_buf, g1_bytes);
    write_buffer(&ctx.queue, &g2_buf, g2_bytes);

    let params = [total_pairs, num_g2_bases, 0u32, 0u32];
    let params_bytes = bytemuck::cast_slice::<u32, u8>(&params);
    let params_buf = create_uniform_buffer(&ctx.device, 16, Some("miller-params"));
    write_buffer(&ctx.queue, &params_buf, params_bytes);

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("miller-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: g1_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: g2_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: results_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("miller-encoder"),
        });

    let workgroup_count = div_ceil(total_pairs, MILLER_WORKGROUP_SIZE);
    dispatch_compute(&mut encoder, pipeline, &bind_group, workgroup_count);

    Ok((results_buf, encoder))
}

fn gpu_fp12_reduce(
    ctx: &WgpuContext,
    registry: &ShaderRegistry,
    results_buffer: &wgpu::Buffer,
    group_sizes: &[u32],
    group_offsets: &[u32],
    encoder: &mut wgpu::CommandEncoder,
) -> Result<(), GpuError> {
    let pipeline = registry
        .get_pipeline("fp12_reduce")
        .ok_or_else(|| GpuError::DispatchError("missing fp12_reduce pipeline".into()))?;

    for g in 0..group_sizes.len() {
        let mut count = group_sizes[g];
        let element_offset = group_offsets[g];
        if count <= 1 {
            continue;
        }

        while count > 1 {
            let stride = div_ceil(count, 2);
            let num_threads = count - stride;

            let params = [stride, num_threads, element_offset, 0u32];
            let params_bytes = bytemuck::cast_slice::<u32, u8>(&params);
            let params_buf = create_uniform_buffer(&ctx.device, 16, Some("reduce-params"));
            write_buffer(&ctx.queue, &params_buf, params_bytes);

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reduce-bind-group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: results_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

            let workgroup_count = div_ceil(num_threads, REDUCE_WORKGROUP_SIZE);
            dispatch_compute(encoder, pipeline, &bind_group, workgroup_count);

            count = stride;
        }
    }

    Ok(())
}

/// Returns one Fp12 per group — the Miller loop product (NOT final-exponentiated).
/// G1 points are flattened across groups; G2 bases are shared (first group, indexed
/// via modular arithmetic in the shader).
pub fn gpu_batch_multi_pairing(
    ctx: &WgpuContext,
    registry: &ShaderRegistry,
    g1_groups: &[&[G1Affine]],
    g2_groups: &[&[G2Affine]],
) -> Result<Vec<Fq12>, GpuError> {
    assert_eq!(g1_groups.len(), g2_groups.len());
    let num_groups = g1_groups.len();

    let mut g1_all = Vec::new();
    let mut group_sizes = Vec::with_capacity(num_groups);
    let mut group_offsets = Vec::with_capacity(num_groups);
    let mut offset = 0u32;

    for group in g1_groups {
        let n = group.len() as u32;
        group_sizes.push(n);
        group_offsets.push(offset);
        g1_all.extend_from_slice(group);
        offset += n;
    }
    let total_pairs = offset;

    let g2_bases = g2_groups[0];
    let num_g2_bases = g2_bases.len() as u32;

    let g1_flat = serialize_g1_affine(&g1_all);
    let g2_flat = serialize_g2_affine(g2_bases);

    let (results_buf, mut encoder) =
        gpu_miller_loop(ctx, registry, &g1_flat, &g2_flat, total_pairs, num_g2_bases)?;

    gpu_fp12_reduce(
        ctx,
        registry,
        &results_buf,
        &group_sizes,
        &group_offsets,
        &mut encoder,
    )?;

    let fp12_byte_size = (FP12_WORDS as u64) * 4;
    let readback_size = (num_groups as u64) * fp12_byte_size;
    let staging = create_staging_buffer(&ctx.device, readback_size, Some("pairing-readback"));

    for (g, &offset) in group_offsets.iter().enumerate() {
        encoder.copy_buffer_to_buffer(
            &results_buf,
            (offset as u64) * fp12_byte_size,
            &staging,
            (g as u64) * fp12_byte_size,
            fp12_byte_size,
        );
    }

    ctx.queue.submit(Some(encoder.finish()));
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    receiver
        .recv()
        .map_err(|e| GpuError::BufferError(format!("readback channel dropped: {e}")))?
        .map_err(|e| GpuError::BufferError(format!("failed to map readback buffer: {e}")))?;

    let mapped = slice.get_mapped_range();
    let result_words: &[u32] = bytemuck::cast_slice(&mapped);

    let fp12_u32s = FP12_WORDS as usize;
    let results: Vec<Fq12> = (0..num_groups)
        .map(|g| {
            let start = g * fp12_u32s;
            deserialize_fq12(&result_words[start..start + fp12_u32s])
        })
        .collect();

    drop(mapped);
    staging.unmap();

    Ok(results)
}

/// Hybrid 95/5 GPU/CPU batch multi-pairing for per-group G1/G2 pairs.
///
/// Each group pairs its own G1 and G2 vectors independently (not shared G2).
/// For groups with >= MIN_HYBRID_PAIRS, GPU handles 95% and CPU handles 5%
/// in parallel. Results are combined via Fp12 multiplication before final
/// exponentiation (done by caller).
pub fn hybrid_batch_multi_pairing(
    ctx: &WgpuContext,
    registry: &ShaderRegistry,
    groups: &[(&[G1Affine], &[G2Affine])],
) -> Result<Vec<Fq12>, GpuError> {
    let num_groups = groups.len();

    let mut gpu_counts = Vec::with_capacity(num_groups);
    let mut gpu_group_sizes = Vec::with_capacity(num_groups);
    let mut gpu_group_offsets = Vec::with_capacity(num_groups);
    let mut total_gpu_pairs = 0u32;

    for (g1, g2) in groups {
        assert_eq!(g1.len(), g2.len());
        let n = g1.len();
        let gpu_count = if n < MIN_HYBRID_PAIRS {
            n
        } else {
            ((n as f64) * GPU_RATIO)
                .round()
                .max(1.0)
                .min((n - 1) as f64) as usize
        };
        gpu_counts.push(gpu_count);
        gpu_group_sizes.push(gpu_count as u32);
        gpu_group_offsets.push(total_gpu_pairs);
        total_gpu_pairs += gpu_count as u32;
    }

    let per_group_g1: Vec<Vec<u32>> = groups
        .par_iter()
        .zip(gpu_counts.par_iter())
        .map(|((g1, _), &gc)| serialize_g1_affine(&g1[..gc]))
        .collect();

    let per_group_g2: Vec<Vec<u32>> = groups
        .par_iter()
        .zip(gpu_counts.par_iter())
        .map(|((_, g2), &gc)| serialize_g2_affine(&g2[..gc]))
        .collect();

    let mut g1_flat = Vec::with_capacity(total_gpu_pairs as usize * 16);
    let mut g2_flat = Vec::with_capacity(total_gpu_pairs as usize * 32);
    for g in 0..num_groups {
        g1_flat.extend_from_slice(&per_group_g1[g]);
        g2_flat.extend_from_slice(&per_group_g2[g]);
    }

    let (results_buf, mut encoder) = gpu_miller_loop(
        ctx,
        registry,
        &g1_flat,
        &g2_flat,
        total_gpu_pairs,
        total_gpu_pairs,
    )?;

    gpu_fp12_reduce(
        ctx,
        registry,
        &results_buf,
        &gpu_group_sizes,
        &gpu_group_offsets,
        &mut encoder,
    )?;

    let fp12_byte_size = (FP12_WORDS as u64) * 4;
    let readback_size = (num_groups as u64) * fp12_byte_size;
    let staging = create_staging_buffer(&ctx.device, readback_size, Some("hybrid-readback"));

    for (g, &offset) in gpu_group_offsets.iter().enumerate() {
        encoder.copy_buffer_to_buffer(
            &results_buf,
            (offset as u64) * fp12_byte_size,
            &staging,
            (g as u64) * fp12_byte_size,
            fp12_byte_size,
        );
    }

    // Submit GPU work, then immediately start CPU tail computation in parallel
    ctx.queue.submit(Some(encoder.finish()));

    let cpu_millers: Vec<Option<Fq12>> = groups
        .par_iter()
        .zip(gpu_counts.par_iter())
        .map(|((g1, g2), &gc)| {
            if gc >= g1.len() {
                None
            } else {
                Some(Bn254::multi_miller_loop(g1[gc..].iter().copied(), g2[gc..].iter().copied()).0)
            }
        })
        .collect();

    // Now wait for GPU to finish
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    receiver
        .recv()
        .map_err(|e| GpuError::BufferError(format!("readback channel dropped: {e}")))?
        .map_err(|e| GpuError::BufferError(format!("failed to map readback buffer: {e}")))?;

    let mapped = slice.get_mapped_range();
    let result_words: &[u32] = bytemuck::cast_slice(&mapped);

    let fp12_u32s = FP12_WORDS as usize;
    let results: Vec<Fq12> = (0..num_groups)
        .map(|g| {
            let start = g * fp12_u32s;
            let gpu_miller = deserialize_fq12(&result_words[start..start + fp12_u32s]);
            match cpu_millers[g] {
                Some(cpu_fp12) => gpu_miller * cpu_fp12,
                None => gpu_miller,
            }
        })
        .collect();

    drop(mapped);
    staging.unmap();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::WgpuContext;
    use crate::error::GpuError;
    use crate::pipeline::ShaderRegistry;
    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;
    use ark_std::UniformRand;

    #[test]
    fn test_miller_loop_single_group() {
        let ctx = match WgpuContext::new() {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) | Err(GpuError::NoDevice(_)) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let registry = ShaderRegistry::new(&ctx.device)
            .unwrap_or_else(|err| panic!("failed to build shader registry: {err}"));

        let mut rng = ark_std::test_rng();
        let n = 8;

        let g1_points: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let g2_points: Vec<G2Affine> = (0..n).map(|_| G2Affine::rand(&mut rng)).collect();

        let gpu_results =
            gpu_batch_multi_pairing(&ctx, &registry, &[&g1_points], &[&g2_points]).unwrap();

        assert_eq!(gpu_results.len(), 1);

        let cpu_result =
            Bn254::multi_miller_loop(g1_points.iter().copied(), g2_points.iter().copied()).0;

        assert_eq!(
            gpu_results[0], cpu_result,
            "GPU and CPU Miller loop results must match"
        );
    }

    #[test]
    fn test_hybrid_batch_multi_pairing() {
        let ctx = match WgpuContext::new() {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) | Err(GpuError::NoDevice(_)) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let registry = ShaderRegistry::new(&ctx.device)
            .unwrap_or_else(|err| panic!("failed to build shader registry: {err}"));

        let mut rng = ark_std::test_rng();
        let n = 40; // > MIN_HYBRID_PAIRS to trigger 95/5 split

        let g1: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let g2: Vec<G2Affine> = (0..n).map(|_| G2Affine::rand(&mut rng)).collect();

        let hybrid_results = hybrid_batch_multi_pairing(&ctx, &registry, &[(&g1, &g2)]).unwrap();

        assert_eq!(hybrid_results.len(), 1);

        let cpu_result = Bn254::multi_miller_loop(g1.iter().copied(), g2.iter().copied()).0;

        assert_eq!(
            hybrid_results[0], cpu_result,
            "Hybrid GPU/CPU pairing must match full CPU pairing"
        );
    }

    #[test]
    fn test_miller_loop_multiple_groups() {
        let ctx = match WgpuContext::new() {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) | Err(GpuError::NoDevice(_)) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let registry = ShaderRegistry::new(&ctx.device)
            .unwrap_or_else(|err| panic!("failed to build shader registry: {err}"));

        let mut rng = ark_std::test_rng();

        // Create 3 groups with different sizes
        let g1_group1: Vec<G1Affine> = (0..4).map(|_| G1Affine::rand(&mut rng)).collect();
        let g1_group2: Vec<G1Affine> = (0..8).map(|_| G1Affine::rand(&mut rng)).collect();
        let g1_group3: Vec<G1Affine> = (0..12).map(|_| G1Affine::rand(&mut rng)).collect();

        // Use same G2 bases for all groups (gpu_batch_multi_pairing uses g2_groups[0] as shared)
        let g2_bases: Vec<G2Affine> = (0..12).map(|_| G2Affine::rand(&mut rng)).collect();

        let gpu_results = gpu_batch_multi_pairing(
            &ctx,
            &registry,
            &[&g1_group1, &g1_group2, &g1_group3],
            &[&g2_bases, &g2_bases, &g2_bases],
        )
        .unwrap();

        assert_eq!(gpu_results.len(), 3);

        // Verify each group's result matches CPU multi_miller_loop individually
        let cpu_result1 =
            Bn254::multi_miller_loop(g1_group1.iter().copied(), g2_bases.iter().copied()).0;
        let cpu_result2 =
            Bn254::multi_miller_loop(g1_group2.iter().copied(), g2_bases.iter().copied()).0;
        let cpu_result3 =
            Bn254::multi_miller_loop(g1_group3.iter().copied(), g2_bases.iter().copied()).0;

        assert_eq!(
            gpu_results[0], cpu_result1,
            "Group 1 GPU result must match CPU"
        );
        assert_eq!(
            gpu_results[1], cpu_result2,
            "Group 2 GPU result must match CPU"
        );
        assert_eq!(
            gpu_results[2], cpu_result3,
            "Group 3 GPU result must match CPU"
        );
    }

    #[test]
    fn test_miller_loop_single_pair() {
        let ctx = match WgpuContext::new() {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) | Err(GpuError::NoDevice(_)) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let registry = ShaderRegistry::new(&ctx.device)
            .unwrap_or_else(|err| panic!("failed to build shader registry: {err}"));

        let mut rng = ark_std::test_rng();

        let g1_point = vec![G1Affine::rand(&mut rng)];
        let g2_point = vec![G2Affine::rand(&mut rng)];

        let gpu_results =
            gpu_batch_multi_pairing(&ctx, &registry, &[&g1_point], &[&g2_point]).unwrap();

        assert_eq!(gpu_results.len(), 1);

        let cpu_result =
            Bn254::multi_miller_loop(g1_point.iter().copied(), g2_point.iter().copied()).0;

        assert_eq!(
            gpu_results[0], cpu_result,
            "Single pair GPU result must match CPU"
        );
    }

    #[test]
    fn test_hybrid_small_group_no_split() {
        let ctx = match WgpuContext::new() {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) | Err(GpuError::NoDevice(_)) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let registry = ShaderRegistry::new(&ctx.device)
            .unwrap_or_else(|err| panic!("failed to build shader registry: {err}"));

        let mut rng = ark_std::test_rng();
        let n = 10; // < MIN_HYBRID_PAIRS, so 100% goes to GPU, no CPU tail

        let g1: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let g2: Vec<G2Affine> = (0..n).map(|_| G2Affine::rand(&mut rng)).collect();

        let hybrid_results = hybrid_batch_multi_pairing(&ctx, &registry, &[(&g1, &g2)]).unwrap();

        assert_eq!(hybrid_results.len(), 1);

        let cpu_result = Bn254::multi_miller_loop(g1.iter().copied(), g2.iter().copied()).0;

        assert_eq!(
            hybrid_results[0], cpu_result,
            "Hybrid result with no split must match full CPU result"
        );
    }

    #[test]
    fn test_gpu_determinism() {
        let ctx = match WgpuContext::new() {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) | Err(GpuError::NoDevice(_)) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let registry = ShaderRegistry::new(&ctx.device)
            .unwrap_or_else(|err| panic!("failed to build shader registry: {err}"));

        let mut rng = ark_std::test_rng();
        let n = 8;

        let g1_points: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let g2_points: Vec<G2Affine> = (0..n).map(|_| G2Affine::rand(&mut rng)).collect();

        // Run gpu_batch_multi_pairing twice with identical inputs
        let result1 =
            gpu_batch_multi_pairing(&ctx, &registry, &[&g1_points], &[&g2_points]).unwrap();
        let result2 =
            gpu_batch_multi_pairing(&ctx, &registry, &[&g1_points], &[&g2_points]).unwrap();

        assert_eq!(result1.len(), 1);
        assert_eq!(result2.len(), 1);

        // Assert results are bitwise equal
        assert_eq!(
            result1[0], result2[0],
            "GPU results must be deterministic (bitwise equal)"
        );
    }
}
