/* gguf_layout.c — print the layout the C side actually has, so a binding can
 * be checked against it instead of against a guess. A ctypes Structure whose
 * offsets have drifted reads neighbouring fields and reports them as data,
 * which is the kind of wrong that looks right.
 *
 * Build: cc -O2 -I. tests/gguf_layout.c -o /tmp/gguf_layout
 */
#include "gguf.h"
#include <stddef.h>
#include <stdio.h>

int main(void) {
    printf("tensor_info %zu\n", sizeof(gguf_tensor_info));
    printf("tensor_info.name %zu\n", offsetof(gguf_tensor_info, name));
    printf("tensor_info.ndim %zu\n", offsetof(gguf_tensor_info, ndim));
    printf("tensor_info.shape %zu\n", offsetof(gguf_tensor_info, shape));
    printf("tensor_info.dtype %zu\n", offsetof(gguf_tensor_info, dtype));
    printf("tensor_info.offset %zu\n", offsetof(gguf_tensor_info, offset));
    printf("tensor_info.n_elements %zu\n", offsetof(gguf_tensor_info, n_elements));
    printf("kv %zu\n", sizeof(gguf_kv));
    printf("kv.type %zu\n", offsetof(gguf_kv, type));
    printf("kv.val %zu\n", offsetof(gguf_kv, val));
    printf("file %zu\n", sizeof(gguf_file));
    printf("file.n_tensors %zu\n", offsetof(gguf_file, n_tensors));
    printf("file.kv %zu\n", offsetof(gguf_file, kv));
    printf("file.tensors %zu\n", offsetof(gguf_file, tensors));
    printf("file.data %zu\n", offsetof(gguf_file, data));
    printf("file.data_size %zu\n", offsetof(gguf_file, data_size));
    printf("file.n_layers %zu\n", offsetof(gguf_file, n_layers));
    printf("file.rope_freq_base %zu\n", offsetof(gguf_file, rope_freq_base));
    printf("file.arch %zu\n", offsetof(gguf_file, arch));
    return 0;
}
