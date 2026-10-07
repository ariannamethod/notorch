#!/bin/sh
# Use a private comma locale where localedef/source data are available.
# Existing system locale archives are never changed.
set -eu
binary=${1:-./test_chuck_architect_state}
if [ -n "${NT_CHUCK_TEST_LOCALE:-}" ]; then
    exec "$binary"
fi
if command -v localedef >/dev/null 2>&1 &&
   [ -r /usr/share/i18n/locales/de_DE ] &&
   { [ -r /usr/share/i18n/charmaps/UTF-8.gz ] || [ -r /usr/share/i18n/charmaps/UTF-8 ]; }; then
    private=$(mktemp -d "${TMPDIR:-/tmp}/notorch-locale-XXXXXX")
    trap 'rm -rf "$private"' EXIT HUP INT TERM
    if localedef --no-archive -i de_DE -f UTF-8 "$private/de_DE.UTF-8" >"$private/build.log" 2>&1; then
        LOCPATH="$private${LOCPATH:+:$LOCPATH}" NT_CHUCK_TEST_LOCALE=de_DE.UTF-8 "$binary"
        exit $?
    fi
    printf '%s\n' 'Private comma locale build unavailable; checking installed locales.'
    cat "$private/build.log"
fi
"$binary"
