/* logo.h — the banner, and the one rule about it.
 *
 * It goes to stderr. stdout carries what the model said and nothing else, so
 * `notorch model.gguf "prompt" > out.txt` writes text and not decoration. */
#ifndef NT_HARNESS_LOGO_H
#define NT_HARNESS_LOGO_H

#include <stdio.h>
#include <unistd.h>

static const char *const NT_LOGO[] = {
    "             _                 _     ",
    " _ __   ___ | |_ ___  _ __ ___| |__  ",
    "| '_ \\ / _ \\| __/ _ \\| '__/ __| '_ \\ ",
    "| | | | (_) | || (_) | | | (__| | | |",
    "|_| |_|\\___/ \\__\\___/|_|  \\___|_| |_|",
};

/* Quiet when asked, and quiet when nobody is watching — a pipe or a log file
 * gets the work, not the letterhead. */
static void nt_logo(int quiet) {
    if (quiet || !isatty(2)) return;
    for (unsigned i = 0; i < sizeof(NT_LOGO) / sizeof(NT_LOGO[0]); i++)
        fprintf(stderr, "%s\n", NT_LOGO[i]);
    fprintf(stderr, "  neural networks in C\n\n");
}

#endif
