## Summary

`nt_tape_clear()` likely fails to fully drop tensor references, causing GPU mirror buffers (`t->d_data`) to persist across training steps. Symptom observed during DoE LoRA SFT on RunPod A100 (2026-05-09): GPU memory grew 2 GB → 34.7 GB across 1450 steps (~22 MB/step), accompanied by 1.21 million `[GPU] ptr_map full — buffer leak` warnings.

This is a static-analysis hypothesis. Verification requires GPU reproduction.

## Suspected refcount path

`notorch.c` `nt_tape_record` (lines 323-327):

```c
e->output = output;
nt_tensor_ref(output);   // → refcount++
```

`nt_tensor_new` returns a tensor with `refcount = 1`. After `nt_tape_record`, the tensor's `refcount = 2`.

`nt_tape_clear` (lines 284-296) releases one reference per entry:

```c
if (g_tape.entries[i].output)
    nt_tensor_free(g_tape.entries[i].output);
```

`nt_tensor_free` (lines 194-203):

```c
t->refcount--;
if (t->refcount <= 0) {
    free(t->data);
#ifdef USE_CUDA
    if (t->d_data) { gpu_free(t->d_data); t->d_data = NULL; }
#endif
    free(t);
}
```

If the op forward function that called `nt_tape_record` does not also call `nt_tensor_free(output)` to release its allocator-side reference, tape clear brings `refcount` down to 1 — never 0 — and the GPU mirror is never returned to the pool. Each step's intermediates accumulate.

## Why "ptr_map full" alone is not the leak

`g_ptr_map` (notorch_cuda.cu line 117, size 65536) is an open-addressed hash mapping live GPU pointers → bucket. Pool architecture is sound: when `ptr_map_set` fails (table full), the allocation succeeds anyway — the pointer is simply not tracked by the bucket cache. `gpu_free` later finds `bucket = -1` from `get_and_clear` and falls through to direct `cudaFree(d_ptr)`. So the warning by itself does not lose memory.

The actual leak is upstream: pointers reach the pool but the corresponding tensors are never refcounted to zero, so `gpu_free` is never called for them. The `ptr_map full` warnings are a *symptom* of that upstream leak — once 65K live tensors accumulate, every new alloc fails to register and the pool stops caching, but the existing leaked allocations stay live in CUDA heap.

## Secondary suspect

`g_wcache` (notorch_cuda.cu lines 22-92) is a persistent weight cache — by design, only freed in `gpu_alloc_cache_clear()` or `gpu_shutdown`. If ephemeral intermediates ever route through this cache (incorrectly registered as "weights"), they accumulate without bound. Reading: `nt_tensor_register_weight(...)` callsites in op functions, look for any case where a non-weight tensor gets registered.

## Reproduction plan

1. Build `notorch_test_gpu` (`make gpu`) on A100 SXM 80GB pod.
2. Run a 50-step micro-training loop (LoRA on tiny mini-Resonance, batch 1, seq 64).
3. Capture `[GPU] ptr_map full` warning count + `nvidia-smi` memory growth between steps 0 and 50.
4. If growth > 0 MB/step, hypothesis confirmed — apply fix (a) or (b) and re-run.

## Proposed fixes (need verification)

**Option A — release allocator ref after tape_record:**
At every op forward callsite, add `nt_tensor_free(output)` after `nt_tape_record`. The tape retains its ref via `nt_tensor_ref`; allocator's ref is released; tape clear brings refcount → 0.

**Option B — transfer ownership in tape_record:**
Remove `nt_tensor_ref(output)` from `nt_tape_record`. Tape "owns" the tensor; tape_clear's `nt_tensor_free` is the only release path.

(B) is cleaner architecturally but requires no callsite still expects to keep using `output` after `tape_record` (need audit). (A) is local but more verbose.

## Success criteria

- `[GPU] ptr_map full` warnings: 0 across 1000-step micro-test.
- GPU memory steady-state: bounded by `weights + activations + optimizer state`, not growing per-step.
- `notorch_test` (CPU) still 47/47 PASS.

## References

- `feedback_doe_lora_5_attempts_failed_2026_05_09.md` — original symptom report (1450 steps, 1.21M warnings, 2→34.7 GB).
- `feedback_lora_resonance_200m_failed_2026_05_09.md` — adjacent Run A1 byte-identical-ckpts incident (different bug, same session).
- Related fix already in main: `3d46007` raised `g_ptr_map` from 8K → 64K (mitigation, not root-cause cure).
