// Read-only SPA ABI manifest for foreign-function bindings.
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef NOTORCH_SPA_BINDING_H
#define NOTORCH_SPA_BINDING_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint32_t nt_spa_binding_version(void);
// Returns SIZE_MAX for an unknown key or NULL. Keys describe the compiled
// library's constants and every public SPA value type's size/alignment/fields.
size_t nt_spa_binding_layout(const char *key);
#ifdef __cplusplus
}
#endif
#endif
