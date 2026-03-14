use std::sync::mpsc;

use ark_bn254::{Bn254, Fq12, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use ark_std::One;
use rayon::prelude::*;

use crate::{
    context::WgpuContext,
    dispatch::{create_bind_group, div_ceil, write_buffer},
    error::GpuError,
    pipeline::ShaderRegistry,
    serialize::{deserialize_fq12, serialize_g1_affine, serialize_g2_affine, FP12_WORDS},
};

const MILLER_WORKGROUP_SIZE: u32 = 64;
const REDUCE_WORKGROUP_SIZE: u32 = 256;

/// In-flight GPU readback handle. The staging buffer has been mapped async
/// but `device.poll` has not yet been called — GPU work is still running.
struct GpuPairingReadback {
    staging_buffer: wgpu::Buffer,
    rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

struct GpuPairingDispatch {
    group_sizes: Vec<u32>,
    /// Present only when there are non-empty groups with actual GPU work.
    readback: Option<GpuPairingReadback>,
}

fn gpu_dispatch_pairing(
    ctx: &WgpuContext,
    registry: &ShaderRegistry,
    groups: &[(&[G1Affine], &[G2Affine])],
) -> Result<GpuPairingDispatch, GpuError> {
    let mut group_sizes = Vec::with_capacity(groups.len());
    let mut group_offsets = Vec::with_capacity(groups.len());
    let mut total_pairs = 0_u32;

    for &(g1s, g2s) in groups {
        if g1s.len() != g2s.len() {
            return Err(GpuError::DispatchError(
                "group G1/G2 length mismatch".to_string(),
            ));
        }
        let size = g1s.len() as u32;
        group_sizes.push(size);
        group_offsets.push(total_pairs);
        total_pairs = total_pairs.saturating_add(size);
    }

    if total_pairs == 0 {
        return Ok(GpuPairingDispatch {
            group_sizes,
            readback: None,
        });
    }

    let mut g1_flat = Vec::with_capacity(total_pairs as usize);
    let mut g2_flat = Vec::with_capacity(total_pairs as usize);
    for &(g1s, g2s) in groups {
        g1_flat.extend_from_slice(&serialize_g1_affine(g1s));
        g2_flat.extend_from_slice(&serialize_g2_affine(g2s));
    }

    let g1_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pairing-g1-buffer"),
        size: (g1_flat.len() * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let g2_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pairing-g2-buffer"),
        size: (g2_flat.len() * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let results_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pairing-results-buffer"),
        size: total_pairs as u64 * FP12_WORDS as u64 * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pairing-miller-params-buffer"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    write_buffer(&ctx.queue, &g1_buffer, bytemuck::cast_slice(&g1_flat));
    write_buffer(&ctx.queue, &g2_buffer, bytemuck::cast_slice(&g2_flat));
    let miller_params = [total_pairs, total_pairs, 0_u32, 0_u32];
    write_buffer(
        &ctx.queue,
        &params_buffer,
        bytemuck::cast_slice(&miller_params),
    );

    let miller_pipeline = registry
        .get_pipeline("bn254_miller")
        .ok_or_else(|| GpuError::DispatchError("missing bn254_miller pipeline".to_string()))?;
    let reduce_pipeline = registry
        .get_pipeline("bn254_reduce")
        .ok_or_else(|| GpuError::DispatchError("missing bn254_reduce pipeline".to_string()))?;

    let miller_layout = miller_pipeline.get_bind_group_layout(0);
    let miller_bind_group = create_bind_group(
        &ctx.device,
        &miller_layout,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: g1_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: g2_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: results_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    );

    let fp12_bytes = (FP12_WORDS * std::mem::size_of::<u32>()) as u64;
    let non_empty_count = group_sizes.iter().filter(|&&s| s > 0).count();
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pairing-group-results-staging"),
        size: non_empty_count as u64 * fp12_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pairing-batch-multi-pairing-encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pairing-miller-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(miller_pipeline);
        pass.set_bind_group(0, &miller_bind_group, &[]);
        pass.dispatch_workgroups(div_ceil(total_pairs, MILLER_WORKGROUP_SIZE), 1, 1);
    }

    let reduce_layout = reduce_pipeline.get_bind_group_layout(0);
    for (group_idx, &group_size) in group_sizes.iter().enumerate() {
        let mut count = group_size;
        while count > 1 {
            let stride = div_ceil(count, 2);
            let num_threads = count - stride;
            if num_threads == 0 {
                break;
            }

            let reduce_params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pairing-reduce-params-buffer"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let reduce_params = [stride, num_threads, group_offsets[group_idx], 0_u32];
            write_buffer(
                &ctx.queue,
                &reduce_params_buffer,
                bytemuck::cast_slice(&reduce_params),
            );

            let reduce_bind_group = create_bind_group(
                &ctx.device,
                &reduce_layout,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: results_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: reduce_params_buffer.as_entire_binding(),
                    },
                ],
            );

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pairing-reduce-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(reduce_pipeline);
                pass.set_bind_group(0, &reduce_bind_group, &[]);
                pass.dispatch_workgroups(div_ceil(num_threads, REDUCE_WORKGROUP_SIZE), 1, 1);
            }

            count = stride;
        }
    }

    let mut copy_slot = 0_u64;
    for (idx, &group_size) in group_sizes.iter().enumerate() {
        if group_size == 0 {
            continue;
        }
        let src_offset = group_offsets[idx] as u64 * fp12_bytes;
        let dst_offset = copy_slot * fp12_bytes;
        encoder.copy_buffer_to_buffer(
            &results_buffer,
            src_offset,
            &staging_buffer,
            dst_offset,
            fp12_bytes,
        );
        copy_slot += 1;
    }

    // Submit — non-blocking. GPU starts executing immediately.
    ctx.queue.submit(Some(encoder.finish()));

    // Register map callback. Fires only during device.poll().
    let (tx, rx) = mpsc::channel();
    staging_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

    Ok(GpuPairingDispatch {
        group_sizes,
        readback: Some(GpuPairingReadback { staging_buffer, rx }),
    })
}

fn gpu_collect_pairing(
    ctx: &WgpuContext,
    dispatch: GpuPairingDispatch,
) -> Result<Vec<Fq12>, GpuError> {
    let num_groups = dispatch.group_sizes.len();
    let mut out = vec![Fq12::one(); num_groups];

    let Some(readback) = dispatch.readback else {
        return Ok(out);
    };

    let _ = ctx.device.poll(wgpu::Maintain::Wait);
    match readback.rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => return Err(GpuError::BufferError(err.to_string())),
        Err(err) => return Err(GpuError::BufferError(err.to_string())),
    }

    let bytes = readback.staging_buffer.slice(..).get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&bytes);

    let mut read_idx = 0_usize;
    for (idx, &group_size) in dispatch.group_sizes.iter().enumerate() {
        if group_size == 0 {
            continue;
        }
        let start = read_idx * FP12_WORDS;
        out[idx] = deserialize_fq12(&words[start..start + FP12_WORDS]);
        read_idx += 1;
    }

    drop(bytes);
    readback.staging_buffer.unmap();

    Ok(out)
}

pub fn gpu_batch_multi_pairing(
    ctx: &WgpuContext,
    registry: &ShaderRegistry,
    groups: &[(&[G1Affine], &[G2Affine])],
) -> Result<Vec<Fq12>, GpuError> {
    if groups.is_empty() {
        return Ok(Vec::new());
    }
    let dispatch = gpu_dispatch_pairing(ctx, registry, groups)?;
    gpu_collect_pairing(ctx, dispatch)
}

pub fn hybrid_batch_multi_pairing(
    ctx: &WgpuContext,
    registry: &ShaderRegistry,
    groups: &[(&[G1Affine], &[G2Affine])],
    gpu_ratio: f64,
) -> Result<Vec<Fq12>, GpuError> {
    if groups.is_empty() {
        return Ok(Vec::new());
    }

    let mut gpu_groups = Vec::with_capacity(groups.len());
    let mut cpu_groups = Vec::with_capacity(groups.len());

    for &(g1s, g2s) in groups {
        if g1s.len() != g2s.len() {
            return Err(GpuError::DispatchError(
                "group G1/G2 length mismatch".to_string(),
            ));
        }

        let n = g1s.len();
        let gpu_count = if n >= 20 {
            ((n as f64 * gpu_ratio).round() as usize).clamp(1, n - 1)
        } else {
            n
        };

        gpu_groups.push((&g1s[..gpu_count], &g2s[..gpu_count]));
        cpu_groups.push((&g1s[gpu_count..], &g2s[gpu_count..]));
    }

    let dispatch = gpu_dispatch_pairing(ctx, registry, &gpu_groups)?;

    let cpu_results: Vec<Fq12> = cpu_groups
        .par_iter()
        .map(|&(cpu_g1s, cpu_g2s)| {
            if cpu_g1s.is_empty() {
                Fq12::one()
            } else {
                Bn254::multi_miller_loop(cpu_g1s.iter().copied(), cpu_g2s.iter().copied()).0
            }
        })
        .collect();

    let gpu_results = gpu_collect_pairing(ctx, dispatch)?;

    let combined = gpu_results
        .iter()
        .zip(cpu_results.iter())
        .map(|(&gpu, &cpu)| gpu * cpu)
        .collect();

    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::{gpu_batch_multi_pairing, hybrid_batch_multi_pairing};
    use crate::{pipeline::ShaderRegistry, WgpuContext};
    use ark_bn254::{Bn254, G1Affine, G1Projective, G2Affine, G2Projective};
    use ark_ec::{pairing::Pairing, CurveGroup};
    use ark_std::{test_rng, UniformRand};

    fn random_pairs(count: usize) -> (Vec<G1Affine>, Vec<G2Affine>) {
        let mut rng = test_rng();
        let g1s = (0..count)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect();
        let g2s = (0..count)
            .map(|_| G2Projective::rand(&mut rng).into_affine())
            .collect();
        (g1s, g2s)
    }

    #[test]
    fn gpu_miller_loop_matches_cpu_single_group() {
        let Ok(ctx) = WgpuContext::new() else {
            return;
        };
        let Ok(registry) = ShaderRegistry::new(&ctx) else {
            return;
        };

        let (g1s, g2s) = random_pairs(4);
        let gpu = gpu_batch_multi_pairing(&ctx, &registry, &[(&g1s, &g2s)]).expect("gpu pairing");
        let cpu = Bn254::multi_miller_loop(g1s.iter().copied(), g2s.iter().copied()).0;
        assert_eq!(gpu[0], cpu);
    }

    #[test]
    fn gpu_miller_loop_matches_cpu_multi_group() {
        let Ok(ctx) = WgpuContext::new() else {
            return;
        };
        let Ok(registry) = ShaderRegistry::new(&ctx) else {
            return;
        };

        let (g1_a, g2_a) = random_pairs(8);
        let (g1_b, g2_b) = random_pairs(4);

        let gpu = gpu_batch_multi_pairing(&ctx, &registry, &[(&g1_a, &g2_a), (&g1_b, &g2_b)])
            .expect("gpu pairing");
        let cpu_a = Bn254::multi_miller_loop(g1_a.iter().copied(), g2_a.iter().copied()).0;
        let cpu_b = Bn254::multi_miller_loop(g1_b.iter().copied(), g2_b.iter().copied()).0;

        assert_eq!(gpu[0], cpu_a);
        assert_eq!(gpu[1], cpu_b);
    }

    #[test]
    fn hybrid_matches_cpu_result() {
        let Ok(ctx) = WgpuContext::new() else {
            return;
        };
        let Ok(registry) = ShaderRegistry::new(&ctx) else {
            return;
        };

        let (g1s, g2s) = random_pairs(100);
        let hybrid =
            hybrid_batch_multi_pairing(&ctx, &registry, &[(&g1s, &g2s)], 0.95).expect("hybrid");
        let cpu = Bn254::multi_miller_loop(g1s.iter().copied(), g2s.iter().copied()).0;

        assert_eq!(hybrid[0], cpu);
    }
}
