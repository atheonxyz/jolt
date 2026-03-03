// Standalone test: verify g2_mul_by_char on GPU matches Rust reference
async function testMulByChar() {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const commonResp = await fetch('/shaders/bn254_common.wgsl');
  const commonSrc = await commonResp.text();

  const testShader = commonSrc + `
@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;

fn load_g2(offset: u32) -> G2Affine {
    var q: G2Affine;
    for (var i = 0u; i < 8u; i = i + 1u) {
        q.x.c0.limbs[i] = input_data[offset + i];
        q.x.c1.limbs[i] = input_data[offset + 8u + i];
        q.y.c0.limbs[i] = input_data[offset + 16u + i];
        q.y.c1.limbs[i] = input_data[offset + 24u + i];
    }
    return q;
}

fn store_g2(q: G2Affine, offset: u32) {
    for (var i = 0u; i < 8u; i = i + 1u) {
        output_data[offset + i] = q.x.c0.limbs[i];
        output_data[offset + 8u + i] = q.x.c1.limbs[i];
        output_data[offset + 16u + i] = q.y.c0.limbs[i];
        output_data[offset + 24u + i] = q.y.c1.limbs[i];
    }
}

@compute @workgroup_size(1)
fn test_main(@builtin(global_invocation_id) gid: vec3u) {
    let q = load_g2(0u);
    let q1 = g2_mul_by_char(q);
    store_g2(q1, 0u);
}
`;

  const module = device.createShaderModule({ code: testShader });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'test_main' }
  });

  const g2Gen = new Uint32Array([
    45883430, 2390996433, 1232798066, 3706394933, 2541820639, 4223149639, 2945863739, 425146433,
    2823577920, 2947838845, 1476581572, 1615060314, 1386229638, 166285564, 988445547, 352252035,
    2288773622, 1637743261, 4120812408, 4269789847, 589004286, 4288551522, 2929607174, 687701739,
    3340261102, 1678334806, 847068347, 3696752930, 859115638, 1442395582, 2482857090, 228892902
  ]);

  const inputBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outputBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inputBuf, 0, g2Gen);

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
    ]
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(1);
  pass.end();
  enc.copyBufferToBuffer(outputBuf, 0, readBuf, 0, 128);
  device.queue.submit([enc.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  const expected = new Uint32Array([
    0x6d0d8acf, 0x6244d29d, 0xf88dbcf6, 0x60190a36, 0xa74b4532, 0x2ec4693b, 0x568104e7, 0x2b2271a1,
    0x7543f52b, 0x3b92c2ca, 0xd7bf2769, 0x3e3e70cf, 0xfa23a8ba, 0x866d2618, 0xbd041be7, 0x05870fdc,
    0x7d512efc, 0x40d316f1, 0x90ef92be, 0x01ee8002, 0x8be26818, 0xaaf14b7d, 0x67032ab2, 0x0fdd29ec,
    0xb83e939f, 0x2cc454e2, 0x59e9a041, 0xdbe96c73, 0x4a62d6c9, 0xe438faca, 0xb18fef88, 0x0a965e89
  ]);

  let ok = true;
  const diffs = [];
  for (let i = 0; i < 32; i++) {
    if (result[i] !== expected[i]) {
      ok = false;
      diffs.push('[' + i + '] exp=0x' + expected[i].toString(16).padStart(8,'0') + ' got=0x' + result[i].toString(16).padStart(8,'0'));
    }
  }

  device.destroy();
  return { ok, diffs: diffs.slice(0, 8), gotFirst8: Array.from(result.slice(0, 8)).map(x => '0x' + x.toString(16).padStart(8, '0')) };
}

window.__testMulByChar = testMulByChar;
