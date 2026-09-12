/* archs.h — the table of families, and the one lookup into it.
 *
 * This used to live in main.c, which meant a body wanting the harness had to
 * either copy the table or compile main.c and inherit its main(). Neither is a
 * dependency; both are a fork waiting to happen. The table is the harness's
 * answer to "what can this read", so it belongs in the library, not in the CLI
 * that happens to be the first caller.
 */
#ifndef NT_HARNESS_ARCHS_H
#define NT_HARNESS_ARCHS_H

#include "harness/arch.h"

/* Exact name lookup. `arch` is the GGUF's general.architecture. Returns NULL
 * for NULL or an architecture no family explicitly claims. */
const nt_arch *nt_pick_arch(const char *arch);

/* The table itself, for a caller that wants to enumerate rather than look up —
 * printing what is supported without duplicating the registry. NULL-terminated. */
extern const nt_arch *const nt_archs[];

#endif
