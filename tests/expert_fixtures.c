/* Packed decoder fixtures for the CLI restart test. */
#include "decode_fixtures.h"

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    return fixture(argv[1], 0) || fixture_gemma3(argv[2]);
}
