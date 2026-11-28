/* publish_new.h — make a finished file appear under its final name, never over another one.
 *
 * Spores, replay episodes and the converters' GGUF outputs are written to a sibling temporary
 * and published with this. */
#ifndef NT_PUBLISH_NEW_H
#define NT_PUBLISH_NEW_H

#include <errno.h>
#include <fcntl.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#include <unistd.h>

/* 0 when tmp now also stands at path, which must not have existed; -1 otherwise.
 *
 * link refuses an existing destination and publishes every byte at once. Android's SELinux
 * policy denies hard links in app data — Termux gets EACCES from link() and from ln — while
 * rename stays allowed, so there renameat2 with RENAME_NOREPLACE keeps the same refusal.
 * Bionic wraps renameat2 only from API 30, hence the raw syscall. A caller's unlink(tmp)
 * then finds nothing, which is harmless. */
static inline int nt_publish_new(const char *tmp, const char *path) {
    if (!link(tmp, path)) return 0;
#if defined(__linux__) && defined(SYS_renameat2)
    if ((errno == EACCES || errno == EPERM) &&
        !syscall(SYS_renameat2, AT_FDCWD, tmp, AT_FDCWD, path, 1u /* RENAME_NOREPLACE */))
        return 0;
#endif
    return -1;
}

#endif
