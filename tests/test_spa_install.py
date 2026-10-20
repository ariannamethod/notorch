"""Build and install SPA away from the checkout, then rebuild its source archive.

Requires the local Python build tools: setuptools >=77 and pip. This gate
uses no package index and leaves all wheels, binaries and installs in a
temporary directory. Run: python3 tests/test_spa_install.py
"""

from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def run(args, cwd, env=None):
    result = subprocess.run(args, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise AssertionError(f"Command failed ({result.returncode}): {args!r}\n{result.stdout}")
    return result.stdout


def copy_build_sources(destination):
    # Copy the actual C include closure, so a missing MANIFEST entry is exposed
    # by the separate source-distribution rebuild below.
    pending = ["notorch.c", "spa_agent.c", "spa_binding.c"]
    copied = set()
    while pending:
        relative = pending.pop()
        if relative in copied:
            continue
        copied.add(relative)
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        for name in re.findall(r'^\s*#\s*include\s+"([^"]+)"', source.read_text(), re.M):
            pending.append(str(Path(relative).parent / name))
    for relative in ("setup.py", "pyproject.toml", "MANIFEST.in", "README.md",
                     "LICENSE", "ARIANNA_METHOD_MANIFESTO.md",
                     "python/SPA.py", "python/spa_engine.py"):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


SMOKE = r'''
import importlib.util
import os
import pathlib
import sys
import SPA

root = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(SPA.__file__).resolve().parent == root, SPA.__file__
assert importlib.util.find_spec("notorch") is None, "installed SPA depends on a notorch shim"
assert hasattr(SPA, "Engine")
native = SPA.Native()
assert pathlib.Path(native.path).resolve().parent == root, native.path
assert "_notorch_spa_native" not in sys.modules, "ctypes library was imported as a Python extension"
os.environ["NOTORCH_LIB"] = str(root / "missing-override.so")
try:
    SPA.Native()
except OSError:
    pass
else:
    raise AssertionError("explicit library override was silently ignored")
assert SPA.Native(native.path).path == native.path
del os.environ["NOTORCH_LIB"]
agent = SPA.Agent(SPA.Config.default(native=native, mode=SPA.Mode.LEARNED,
                                     exploration=0, seed=42), native=native)
observation = native.perceive([[1., 0.], [0., 1.], [-1., 0.]], 1)
decision = agent.choose(observation)
before = SPA.Metrics(.2, .3, .4, .4, .6, .1, .4)
after = SPA.Metrics(.3, .4, .5, .5, .5, .1, .5)
receipt = agent.observe(decision.sequence, decision.action,
                        SPA.Consequence(before, after, 0))
assert receipt.learned and agent.state.observations == 1
assert agent.state.updates == 1 and receipt.reward > 0
assert len(native.embed_sentence([0, 1], [[1., 0.], [0., 1.]])) == 2
print("installed SPA: native perception, learned consequence and Engine available")
'''


class InstallTest(unittest.TestCase):
    def test_wheel_and_source_distribution(self):
        with tempfile.TemporaryDirectory(prefix="spa-install-") as directory:
            scratch = Path(directory)
            source = scratch / "source"
            source.mkdir()
            copy_build_sources(source)
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.pop("NOTORCH_LIB", None)
            env["PIP_NO_INDEX"] = "1"
            env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
            wheels = scratch / "wheels"
            wheels.mkdir()
            run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps",
                 "--no-build-isolation", "--wheel-dir", str(wheels)], source, env)
            wheel, = wheels.glob("*.whl")
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                self.assertIn("SPA.py", names)
                self.assertIn("spa_engine.py", names)
                self.assertNotIn("notorch.py", names)
                self.assertEqual(len([n for n in names if n.startswith("_notorch_spa_native.")]), 1)
                self.assertFalse(any(n.startswith("experiments/") for n in names))
            installed = scratch / "installed"
            run([sys.executable, "-m", "pip", "install", "--no-deps", "--target",
                 str(installed), str(wheel)], scratch, env)
            # -S excludes user/global site-packages: the wheel must carry its
            # whole runtime, with no checkout or already-installed shim fallback.
            # A stale adjacent libnotorch must not displace the wheel's own code.
            (scratch / ("libnotorch.dylib" if sys.platform == "darwin" else "libnotorch.so")).write_bytes(
                b"deliberately invalid neighbouring library")
            smoke_env = dict(env, PYTHONPATH=str(installed))
            run([sys.executable, "-S", "-c", SMOKE, str(installed)], scratch, smoke_env)
            run([sys.executable, "setup.py", "sdist", "--dist-dir", str(scratch / "sdist")],
                source, env)
            sdist, = (scratch / "sdist").glob("*.tar.gz")
            unpack = scratch / "unpacked"
            unpack.mkdir()
            with tarfile.open(sdist) as archive:
                for member in archive.getmembers():
                    path = Path(member.name)
                    self.assertFalse(path.is_absolute() or ".." in path.parts)
                    self.assertFalse(member.issym() or member.islnk())
                archive.extractall(unpack, **({"filter": "data"}
                                              if hasattr(tarfile, "data_filter") else {}))
            extracted, = unpack.iterdir()
            rebuilt = scratch / "rebuilt"
            rebuilt.mkdir()
            run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps",
                 "--no-build-isolation", "--wheel-dir", str(rebuilt)], extracted, env)
            rebuilt_wheel, = rebuilt.glob("*.whl")
            second = scratch / "installed-sdist"
            run([sys.executable, "-m", "pip", "install", "--no-deps", "--target",
                 str(second), str(rebuilt_wheel)], scratch, env)
            run([sys.executable, "-S", "-c", SMOKE, str(second)], scratch,
                dict(env, PYTHONPATH=str(second)))


if __name__ == "__main__":
    unittest.main()
