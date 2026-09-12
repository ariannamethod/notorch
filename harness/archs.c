#include "harness/archs.h"
#include <string.h>

const nt_arch *const nt_archs[] = {
    &nt_arch_gemma4,
    &nt_arch_olmoe,
    &nt_arch_mamba,
    &nt_arch_resonance,
    &nt_arch_janus,
    &nt_arch_llama,
    NULL,
};

const nt_arch *nt_pick_arch(const char *arch) {
    const nt_arch *fallback = NULL;
    for (const nt_arch *const *p = nt_archs; *p; p++) {
        const nt_arch *a = *p;
        if (!a->names) { fallback = a; continue; }
        for (const char *const *n = a->names; *n; n++)
            if (strcmp(*n, arch) == 0) return a;
    }
    return fallback;
}
