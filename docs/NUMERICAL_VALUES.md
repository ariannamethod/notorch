# Learning with owned values

Seven stateless CPU kernels expose canonical NoTorch arithmetic to AML and
other callers. The caller owns parameter vectors, activations, gradients and
publication. Layers and their reverse pass are composed in the caller's
language. These kernels allocate nothing, retain no pointers and never access
the process-global autograd tape, optimizer state or legacy random stream.

## Contract

All functions return `0` on success and `-1` on invalid input or nonfinite
arithmetic. Every input float must be finite. Pointers name contiguous CPU
`float` buffers; the caller supplies their capacities. Output storage must be
disjoint from inputs and, for normal sampling, the state. Inputs can share
read-only storage with each other.

Every vector, matrix and packed output has a positive length no greater than
`NT_MAX_ELEMENTS`. Dimensions are checked with a division bound before their
product is formed, including on 32-bit hosts. No pointer remains borrowed
after a function returns.

**Treat output as unpublished scratch and discard it on every failure.**
Argument and input validation precede output writes. Arithmetic failure can
leave a partial result. Input buffers are unchanged. The caller publishes a
new parameter/activation array only after success; this supports transactional
language bindings without an extra allocation inside NoTorch.

| Function | Inputs | Output layout |
|---|---|---|
| `nt_linear_values(w,b,x,rows,cols,out)` | Row-major `w[rows*cols]`, `b[rows]`, `x[cols]` | `out[rows]` |
| `nt_linear_vjp_values(w,x,dy,rows,cols,out)` | Same weight/input layout, `dy[rows]` | `[dW(rows*cols), db(rows), dx(cols)]` |
| `nt_tanh_values(x,n,out)` | `x[n]` | `tanh(x)`, length `n` |
| `nt_tanh_vjp_values(y,dy,n,out)` | Saved activation `y[n]` in `[-1,1]`, upstream `dy[n]` | `dy * (1-y*y)`, length `n` |
| `nt_mse_grad_values(pred,target,n,out)` | Prediction/target vectors of length `n` | `[mean_loss, d_pred(n)]` |
| `nt_sgd_values(params,grad,n,lr,out)` | Parameter/gradient vectors, finite `lr >= 0` | `params-lr*grad`, length `n` |
| `nt_rng_normal_values(state,n,out)` | Owned PCG32 `uint64_t` state | `n` standard normal values |

Linear accumulation starts with the bias, then visits columns in order.
The VJP uses `dW[r,c]=dy[r]*x[c]`, `db[r]=dy[r]`, and a row-order sum for
`dx[c]`. MSE is the mean of squared differences, with
`d_pred[i]=2*(pred[i]-target[i])/n`; its packed output requires `n+1` slots.
Scalar float arithmetic and `tanhf` match the native float32 substrate.
Optimizer policy, clipping, clamps and statistics belong to the caller.

## Normal initialization

Each normal value consumes exactly two words from the existing
[owned PCG32 stream](SAMPLING.md). With raw unsigned words `a` and `b`:

```
u1 = (a + 1.0) / 4294967297.0
u2 = b / 4294967296.0
z  = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
```

The transform uses double arithmetic and stores a float result. `u1` is
strictly between zero and one. There is no hidden spare-value cache: drawing
17 values at once and drawing 3 followed by 14 produce identical values and
state on the same platform. The state is copied locally and commits only after
the complete call succeeds. Invalid calls preserve it. Integer state evolution
is portable; the transform follows the platform's `log`, `sqrt` and `cos`.

## Ownership in a language binding

Allocate a fresh output container, invoke the kernel with borrowed input data,
and publish that container on success. For Gaussian initialization, use a local
copy of the language's serialized stream and publish its new state after the
output container is ready. No native model identity, pointer handle, lock or
tape needs to survive between calls. Independent model vectors and stream
states can run concurrently; copying them captures the full learning state.

## Verification

```
make check_numerical_values BLAS_FLAGS= BLAS_LIBS= X86_SIMD=0 ARM_SIMD=0
python3 tests/test_numerical_values_mutations.py
```

The compiled gate checks bias-first accumulation, every packed gradient,
mean-loss scaling, functional SGD, input preservation, invalid dimensions,
nonfinite inputs, bounded saved activations and overflow rejection. A complete
5→8→1 tanh network is assembled from the primitives in the test: all 57
gradients match central finite differences, and 32 SGD steps match independent
Python binary64 predictions, losses and parameter vectors.

The normal gate compares 256 independently generated Python values and exact
64-bit post-draw states, covers raw state zero and the maximum-output PCG
state, checks chunk invariance and legacy-stream isolation, and measures mean
and second moment for a fixed population of 65,536 samples. Eight threads each
perform 64 training steps with separate model/state vectors; their results are
byte-identical to serial runs while a live legacy tape remains byte-unchanged.

The committed reference header requires no Python at test time. Regenerate it:

```
python3 tests/numerical_values_reference.py > tests/numerical_values_reference.h
```

Three isolated mutations remove the linear bias, reverse the tanh derivative
sign, and advance a normal stream before input validation. The gate rejects
all three; the mutation script changes temporary copies only.
