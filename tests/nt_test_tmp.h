/* nt_test_tmp.h — where a test may leave its fixtures.
 *
 * $TMPDIR when it is set, /tmp otherwise. Android has no /tmp at all: Termux points TMPDIR at
 * $PREFIX/tmp, and a fixture template spelled "/tmp/..." fails there in mkstemp before the
 * test reaches the code it is about. */
#ifndef NT_TEST_TMP_H
#define NT_TEST_TMP_H

#include <stdio.h>
#include <stdlib.h>

#define NT_TMP_MAX 256

static inline const char *nt_test_tmpdir(void) {
    const char *d = getenv("TMPDIR");
    return d && *d ? d : "/tmp";
}

/* buf = "<tmpdir>/<name>". A trailing XXXXXX in name is left for mkstemp or mkdtemp. A path
 * that does not fit stops the test here instead of letting a cut-off template reach them. */
static inline char *nt_test_tmp(char *buf, size_t n, const char *name) {
    int k = snprintf(buf, n, "%s/%s", nt_test_tmpdir(), name);
    if (k < 0 || (size_t)k >= n) {
        fprintf(stderr, "temporary path too long: %s/%s\n", nt_test_tmpdir(), name);
        exit(2);
    }
    return buf;
}

/* The same, for a test that hands the path to system(). Such a test single-quotes every path
 * it builds from buf, so spaces and shell syntax in TMPDIR pass through unchanged; a single
 * quote cannot, and stops the test here. */
static inline char *nt_test_tmp_sh(char *buf, size_t n, const char *name) {
    nt_test_tmp(buf, n, name);
    for (const char *p = buf; *p; p++)
        if (*p == '\'') {
            fprintf(stderr, "temporary path holds a single quote: %s\n", buf);
            exit(2);
        }
    return buf;
}

#endif
