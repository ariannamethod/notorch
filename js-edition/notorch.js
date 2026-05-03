// notorch.js — "The logic of memory without the weight of framework"
// Pure JS/WebGPU AI engine. No dependencies.
// (C) 2026 Arianna Method contributors

export class Tensor {
  constructor(data, shape) {
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape = shape;
    this.len = this.data.length;
    this.gpuBuffer = null;
  }

  static zeros(shape) {
    const len = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(len), shape);
  }

  static rand(shape, scale = 1.0) {
    const len = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(len);
    for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * scale;
    return new Tensor(data, shape);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// WGSL SHADERS
// ═══════════════════════════════════════════════════════════════════════════

const MATMUL_WGSL = `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Uniforms { M: u32, N: u32, K: u32 };
@group(0) @binding(3) var<uniform> u: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    if (row >= u.M || col >= u.N) { return; }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < u.K; k = k + 1u) {
        sum = sum + A[row * u.K + k] * B[k * u.N + col];
    }
    C[row * u.N + col] = sum;
}
`;

// ═══════════════════════════════════════════════════════════════════════════
// ENGINE
// ═══════════════════════════════════════════════════════════════════════════

export class Notorch {
  constructor() {
    this.device = null;
    this.hasWebGPU = false;
    this.pipelines = new Map();
  }

  async init() {
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        this.device = await adapter.requestDevice();
        this.hasWebGPU = true;
        console.log("notorch.js: WebGPU active");
      } catch (e) {
        console.warn("notorch.js: WebGPU init failed", e);
      }
    }
    return this.hasWebGPU;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // HIGH LEVEL API
  // ═══════════════════════════════════════════════════════════════════════════

  async matmul(a, b) {
    if (this.hasWebGPU) return this.matmulGPU(a, b);
    return Notorch.matmulCPU(a, b);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // CPU OPS (Optimized with fround)
  // ═══════════════════════════════════════════════════════════════════════════

  static matmulCPU(a, b) {
    const [M, K] = a.shape;
    const [K2, N] = b.shape;
    const out = Tensor.zeros([M, N]);
    const d_a = a.data; const d_b = b.data; const d_out = out.data;

    for (let i = 0; i < M; i++) {
      const iK = i * K;
      for (let j = 0; j < N; j++) {
        let sum = Math.fround(0.0);
        for (let k = 0; k < K; k++) {
          sum = Math.fround(sum + Math.fround(d_a[iK + k] * d_b[k * N + j]));
        }
        d_out[i * N + j] = sum;
      }
    }
    return out;
  }

  static softmax(a) {
    const d = a.data; const len = a.len; const out = new Float32Array(len);
    let max = -Infinity;
    for (let i = 0; i < len; i++) if (d[i] > max) max = d[i];
    let sum = 0;
    for (let i = 0; i < len; i++) {
      out[i] = Math.exp(d[i] - max);
      sum += out[i];
    }
    for (let i = 0; i < len; i++) out[i] /= sum;
    return new Tensor(out, a.shape);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // GPU OPS
  // ═══════════════════════════════════════════════════════════════════════════

  async matmulGPU(a, b) {
    const [M, K] = a.shape;
    const [K2, N] = b.shape;
    const out = Tensor.zeros([M, N]);

    const bufA = this.createGPUBuffer(a.data, GPUBufferUsage.STORAGE);
    const bufB = this.createGPUBuffer(b.data, GPUBufferUsage.STORAGE);
    const bufC = this.createGPUBuffer(out.data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const bufU = this.createGPUBuffer(new Uint32Array([M, N, K]), GPUBufferUsage.UNIFORM);

    const pipeline = await this.getPipeline("matmul", MATMUL_WGSL);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufC } },
        { binding: 3, resource: { buffer: bufU } }
      ]
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(N / 8), Math.ceil(M / 8));
    passEncoder.end();

    const readBuffer = this.device.createBuffer({
      size: out.len * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    commandEncoder.copyBufferToBuffer(bufC, 0, readBuffer, 0, out.len * 4);
    
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    out.data.set(new Float32Array(readBuffer.getMappedRange()));
    readBuffer.unmap();

    // Cleanup
    bufA.destroy(); bufB.destroy(); bufC.destroy(); bufU.destroy(); readBuffer.destroy();

    return out;
  }

  createGPUBuffer(data, usage) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage,
      mappedAtCreation: true
    });
    new data.constructor(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  }

  async getPipeline(name, code) {
    if (this.pipelines.has(name)) return this.pipelines.get(name);
    const module = this.device.createShaderModule({ code });
    const pipeline = await this.device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' }
    });
    this.pipelines.set(name, pipeline);
    return pipeline;
  }
}
