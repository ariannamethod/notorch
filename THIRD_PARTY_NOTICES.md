# Third-party notices

## PCG32 sampling

The PCG32 XSH-RR transition, two-step seeding, and bounded-rejection algorithm
in `notorch.c` adapt the [PCG minimal C implementation](https://github.com/imneme/pcg-c-basic)
by Melissa O'Neill.

Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>

These portions are available under the Apache License, Version 2.0. The full
license is in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). Upstream provides
the work on an “AS IS” basis without warranties or conditions of any kind,
either express or implied; the license sets out the applicable permissions
and limitations.

NoTorch modifications fix the sequence to 54, expose caller-owned `uint64_t`
state, handle NULL inputs, preserve state and output on rejected checked
calls, and integrate floating-point and categorical sampling. The categorical
weight calculation and NoTorch API integration are Arianna Method code.

PCG is compiled into NoTorch and introduces no separate runtime library.
The surrounding NoTorch source retains its declared LGPL-3.0-or-later license.
