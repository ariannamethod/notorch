#!/usr/bin/env python3
"""Compile a native sentinel through the production living source boundary."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "chuck_living_build", ROOT / "experiments/chuck_loss_architect/living/run.py")
living = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(living)


class LivingBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="chuck-build-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "fixture"
        self.here = self.source / "experiments/chuck_loss_architect/living"
        self.here.mkdir(parents=True)
        for name in living.SOURCE_FILES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        (self.source / "notorch.h").write_text(
            "int snapshot_value(void);\n#define SNAPSHOT_BIAS 0\n")
        (self.source / "notorch.c").write_text("int snapshot_value(void) { return 41; }\n")
        (self.source / "examples/chuck_architect_train.c").write_text(
            '#include "notorch.h"\n#include <stdio.h>\n'
            'int main(void) { printf("%d\\n", snapshot_value() + SNAPSHOT_BIAS); return 0; }\n')
        (self.here / "run.py").write_text("# orchestration identity fixture\n")
        (self.here / "protocol.json").write_text('{"fixture":true}\n')
        # This tiny synthetic repository has no remote or NoTorch checkout.
        self.git("init", "-q")
        self.git("add", ".")
        self.commit()
        self.source_commit = self.git("rev-parse", "HEAD").strip()
        for owner, name, value in ((living, "ROOT", self.source), (living, "HERE", self.here),
                                   (living.scenarios, "ROOT", self.source)):
            patch = mock.patch.object(owner, name, value)
            patch.start(); self.addCleanup(patch.stop)

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.source, text=True)

    def commit(self):
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "Native build fixture")

    def build(self, name="run", dirty=False):
        output = self.directory / name
        output.mkdir()
        durable = living.DurableArtifacts(output)
        hashes, command = living.build_snapshot(output, durable, self.source_commit, dirty)
        result = subprocess.run([str(output / "host_runner")], check=True,
                                capture_output=True, text=True)
        return output, hashes, command, result.stdout

    def test_late_edits_cannot_change_both_translation_units_or_headers(self):
        original_build = living.scenarios.build

        def edit_then_build(binary, *args, **kwargs):
            (self.source / "notorch.c").write_text("int snapshot_value(void) { return 90; }\n")
            (self.source / "notorch.h").write_text(
                "int snapshot_value(void);\n#define SNAPSHOT_BIAS 7\n")
            trainer = self.source / "examples/chuck_architect_train.c"
            trainer.write_text(trainer.read_text().replace("SNAPSHOT_BIAS", "SNAPSHOT_BIAS + 2"))
            self.git("add", "."); self.commit()
            return original_build(binary, *args, **kwargs)

        with mock.patch.object(living.scenarios, "build", side_effect=edit_then_build):
            output, hashes, command, actual = self.build()
        self.assertEqual(actual, "41\n")
        self.assertNotEqual(self.git("rev-parse", "HEAD").strip(), self.source_commit)
        self.assertEqual(command[command.index("-I") + 1], str(output / "source"))
        for name in ("notorch.c", "examples/chuck_architect_train.c"):
            self.assertIn(str(output / "source" / name), command)
        for name, digest in hashes.items():
            self.assertEqual(living.v1.digest(output / "source" / name), digest)

    def test_wrong_root_mutant_runs_but_fails_native_identity(self):
        original_build = living.scenarios.build

        def wrong_root(binary, *args, **kwargs):
            (self.source / "notorch.c").write_text("int snapshot_value(void) { return 99; }\n")
            return original_build(binary)  # Deliberate restoration of the reviewed defect.

        with mock.patch.object(living.scenarios, "build", side_effect=wrong_root):
            _, _, _, actual = self.build()
        self.assertEqual(actual, "99\n")  # The mutant compiled and exited successfully.
        with self.assertRaises(AssertionError):
            self.assertEqual(actual, "41\n")

    def test_change_between_clean_check_and_copy_is_refused(self):
        (self.source / "notorch.h").write_text("#define SNAPSHOT_BIAS 99\n")
        with mock.patch.object(living.scenarios, "build") as compiler:
            with self.assertRaisesRegex(ValueError, "source changed.*notorch.h"):
                self.build()
            compiler.assert_not_called()

    def test_dirty_smoke_records_and_builds_its_actual_bytes(self):
        (self.source / "notorch.c").write_text("int snapshot_value(void) { return 43; }\n")
        output, hashes, _, actual = self.build(dirty=True)
        self.assertEqual(actual, "43\n")
        self.assertEqual(hashes["notorch.c"], living.v1.digest(output / "source/notorch.c"))

    def test_legacy_build_and_explicit_mutant_source_remain_supported(self):
        binary = self.directory / "default"
        living.scenarios.build(binary)
        self.assertEqual(subprocess.check_output([str(binary)], text=True), "41\n")
        override = self.directory / "override.c"
        override.write_text('#include "notorch.h"\n#include <stdio.h>\n'
                            'int main(void) { printf("%d\\n", snapshot_value() + 1); return 0; }\n')
        living.scenarios.build(binary, override)
        self.assertEqual(subprocess.check_output([str(binary)], text=True), "42\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
