"""Build the SPA Python runtime and its native upstream arithmetic together."""

import os
import shlex
import shutil
import sys
import sysconfig

from setuptools import Extension, setup


if sys.platform not in ("linux", "darwin"):
    raise RuntimeError("The SPA native package currently builds on Linux and macOS")

# Relocatable Python installations can name a compiler that is absent from
# the machine running pip. Respect an explicit CC; otherwise use the system C
# compiler when that recorded build compiler is unavailable.
recorded_cc = shlex.split(sysconfig.get_config_var("CC") or "cc")[0]
if "CC" not in os.environ and shutil.which(recorded_cc) is None and shutil.which("cc"):
    os.environ["CC"] = "cc"

setup(
    name="notorch-spa",
    version="1.0.0",
    description="Sentence Phonon Agent: native sentence perception, actions and acquired experience",
    author="Arianna Method contributors",
    url="https://github.com/ariannamethod/notorch",
    license="GPL-3.0-or-later",
    license_files=["LICENSE"],
    python_requires=">=3.10",
    package_dir={"": "python"},
    py_modules=["SPA", "spa_engine"],
    ext_modules=[Extension(
        # A ctypes library with a platform-specific filename. SPA locates it
        # without importing it; it exports the ordinary C API, not PyInit_*.
        "_notorch_spa_native",
        sources=["notorch.c", "spa_agent.c", "spa_binding.c"],
        include_dirs=["."],
        depends=["notorch.h", "spa_agent.h", "spa_binding.h",
                 "chuck_architect.h", "chuck_architect_impl.h"],
        extra_compile_args=["-O2", "-std=gnu11", "-pthread"],
        extra_link_args=["-pthread"],
        libraries=["m"],
    )],
)
