# Owned random streams and categorical sampling

NoTorch owns a deterministic sampling stream in a caller-supplied `uint64_t`.
Copying the integer creates an independent replay point. All six operations
are allocation-free and independent of the legacy `nt_seed` tensor stream,
Chuck's noise, and libc `rand`.

```c
uint64_t rng;
nt_rng_seed(&rng, 42);
uint64_t saved = rng;
float weights[] = {1, 2, 5};
int chosen;
if (nt_rng_categorical(&rng, weights, 3, 0.8f, &chosen) == 0) {
    /* chosen is a valid index; rng has advanced exactly one word. */
}
rng = saved;  /* replay */
```

## API

| Call | Contract |
| --- | --- |
| `nt_rng_seed(state, seed)` | Initialize any 64-bit seed using the standard two-step PCG procedure. NULL is a no-op. |
| `nt_rng_u32(state)` | Advance once and return a 32-bit word. NULL returns zero. |
| `nt_rng_uniform(state)` | Advance once; return the upper 24 bits times `2^-24`, exactly in float32 `[0,1)`. NULL returns zero. |
| `nt_rng_index(state, bound, out)` | Choose an unbiased integer in `[0,bound)`. Any positive `uint32_t` bound is accepted. Rejection may consume several words; bound one still consumes one. |
| `nt_categorical_index(weights, n, temperature, draw, out)` | Select from explicit finite `draw` in `[0,1)` without any RNG state. |
| `nt_rng_categorical(state, weights, n, temperature, out)` | Select using one full 32-bit word divided by `2^32` in double precision. Singleton support still consumes one word. |

The three checked calls return zero on success and `-1` on invalid inputs.
On error, output and RNG state are unchanged. Output storage must not overlap
state or input storage. Weight arrays are borrowed and remain unchanged.
Each thread owns its state; a host sharing the same mutable state supplies
its own synchronization.

The generator is PCG32 XSH-RR with multiplier `6364136223846793005` and fixed
sequence 54, hence increment 109. Every 64-bit state is valid, including zero.
The seed procedure starts at zero, advances once, adds the seed modulo
`2^64`, and advances once more. The first six words for seed 42 are:

```text
a15c02b7 7b47f409 ba1d3330 83d2f293 bfa4784b cbed606e
```

These are the [published PCG minimal-C reference words](https://www.pcg-random.org/using-pcg-c-basic.html).
For bounded sampling, discard words below `2^32 mod bound` before taking the
remainder. Starting from raw state zero with bound three rejects two initial
zero words, accepts `47c28b93`, returns two, and ends at state
`0b98f44445d432cb`.

The uniform float API deliberately exposes 24 bits; the categorical API uses
all 32 bits. This distinction prevents float conversion from rounding a draw
up to one while retaining finer categorical resolution. Integer words and
state transitions have portable fixed-width semantics. Categorical boundaries
use the platform's double `log`/`exp`; engine comparisons name their tested
platform and numerical boundary.

## Weight law

`n` must be positive. Weights must all be finite and nonnegative, with at
least one positive entry. Temperature must be finite and strictly positive;
zero does not request greedy selection. All inputs are checked before any
externally visible state or output changes.

The categorical law is proportional to `weight^(1/temperature)`. NoTorch
evaluates the equivalent scaled mass

```text
mass[i] = exp(log(double(weight[i]) / maximum_weight) / double(temperature))
```

for positive weights and uses zero mass otherwise. Exponents cannot be
positive; at least one mass is exactly one. The entire float32 weight ratio
range fits in double. Double accumulation avoids intermediate overflow from
raising counts directly to a large inverse temperature. Smaller masses may
underflow to zero at low temperature, leaving the maximal weights active.

Selection uses strict `target < cumulative`, skips zero masses, and returns
the final positive-mass index if rounding places a draw just below one on the
last boundary. Leading, interior, or trailing zero weights are never selected.
All sums follow the caller's order, which fixes behavior at cumulative ties.

## Verification

```sh
make check_sampling BLAS_FLAGS= BLAS_LIBS=
```

The C gate uses the published six-word vector and 384 checked-in reference
steps generated independently by Python integer arithmetic and 80-digit
Decimal direct powers. Six seeds include zero, 42, and `UINT64_MAX`; mixed
operations exercise uniform floats, small and large bounds, five temperatures,
and complete state after every call. Regenerate the fixture only for review:

```sh
python3 tests/make_sampling_reference.py
```

The ordinary gate needs no Python. Additional checks cover rejected-word
state transitions, exact CDF ties and zero holes, singleton consumption,
`FLT_TRUE_MIN`/`FLT_MAX` weights and temperatures, power-of-two scaling,
invalid-input preservation, snapshot replay, and bidirectional isolation from
libc and the legacy tensor stream.

The scalar C gate passes **7,215 checks**. ASan/UBSan passes the same gate with
`ASAN_OPTIONS=detect_leaks=0`. The mutation harness compiles an unchanged
baseline and three temporary source copies; bypassing bounded rejection,
making CDF comparison inclusive, and publishing state before input validation
each fail the corresponding assertion:

```sh
python3 tests/test_sampling_mutations.py
```

The first gate run caught a mistyped raw-state fixture intended to produce
`UINT32_MAX`; correcting it to `07fffe0000000000` fixed the test. The published
vector and all independent reference steps passed on that first run.

PCG attribution and adapted-code terms are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
