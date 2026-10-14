/* Test-only allocation interception, injected into the three loader objects.
 * Production sources and installed headers carry no allocation hook state. */
#ifndef NT_PEN7_FAULT_ALLOC_H
#define NT_PEN7_FAULT_ALLOC_H
#include <stdlib.h>
void *nt_pen7_fault_malloc(size_t bytes);
void *nt_pen7_fault_calloc(size_t count, size_t bytes);
void nt_pen7_fault_free(void *ptr);
#ifdef NT_PEN7_FAULT_REDIRECT
#define malloc nt_pen7_fault_malloc
#define calloc nt_pen7_fault_calloc
#define free nt_pen7_fault_free
#endif
#endif
