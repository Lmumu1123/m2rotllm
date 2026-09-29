"""Local CLI and shared-contact integrity tests using temporary synthetic files.

These tests exercise the data contract, not diagnosis accuracy or real capture
validity. Run with ``python -m unittest test_pipeline -v``.
"""
from pathlib import Path
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import build_manifest


ROOT = Path(__file__).resolve().parent


def synthetic_frame():
    """Known distance bin under the documented historical I-iQ convention."""
    fast = 2 * np.pi * 20 * np.arange(256)[None, None, :] / 512
    slow = .08 * np.sin(2 * np.pi * 60 * np.arange(192)[:, None, None] / 5000)
    phase_rx = .1 * np.arange(4)[None, :, None]
    paired = (2000 * np.exp(1j * (fast + slow + phase_rx))).reshape(-1, 2)
    words = np.empty((len(paired), 4), dtype="<i2")
    words[:, 0] = np.rint(paired[:, 0].real)
    words[:, 1] = np.rint(paired[:, 1].real)
    words[:, 2] = np.rint(-paired[:, 0].imag)
    words[:, 3] = np.rint(-paired[:, 1].imag)
    return words.tobytes()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.output = self.root / "processed"
        self.manifest = self.root / "manifest.csv"

    def run_cli(self, script, *args, expected=0):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, expected,
                         f"Command failed: {script} {args}\n{result.stdout}\n{result.stderr}")
        return result

    def create_two_recordings(self):
        data = synthetic_frame() * 2
        for rep in (1, 2):
            (self.raw / f"20260927-1830{rep:02d}-normal-1000r-40cm.bin").write_bytes(data)
        contact = self.raw / "2026_9_27_18-00-00-normal-1000r.DAT"
        contact.write_text("".join(
            f'{"!" if i == 1 else ""}{i},{np.sin(i / 20):.6f},{np.cos(i / 30):.6f},0.3;'
            for i in range(1, 4097)), encoding="ascii")
        self.run_cli("build_manifest.py", "--data-root", self.raw, "--output", self.manifest)
        return contact

    def process_args(self):
        return ("process", "--data-root", self.raw, "--manifest", self.manifest,
                "--output", self.output, "--radar-format", "headerless_reordered_adc")

    def upload_mtimes(self):
        return {p.relative_to(self.output).as_posix(): p.stat().st_mtime_ns
                for p in (self.output / "upload").rglob("*") if p.is_file()}

    def test_shared_contact_resume_missing_groups_and_report(self):
        contact_source = self.create_two_recordings()
        self.run_cli("preprocess_local.py", *self.process_args())
        contacts = self.output / "upload" / "contacts"
        radar = self.output / "upload" / "radar"
        self.assertEqual(len(list(contacts.glob("*/contact.h5"))), 1)
        self.assertEqual(len(list(contacts.glob("*/contact_original.DAT.gz"))), 1)
        self.assertEqual(len(list(radar.glob("*/radar.h5"))), 2)
        catalog = json.loads((self.output / "export_catalog.json").read_text())
        self.assertEqual(len(catalog["expected_radar"]), 2)
        self.assertEqual(len(set(catalog["expected_radar"].values())), 1)
        for path in radar.glob("*/recording.json"):
            recording = json.loads(path.read_text())
            self.assertEqual(recording["pairing_level"], "continuous_session")
            self.assertFalse(recording["synchronized_window_pair"])
            self.assertTrue((path.parent / recording["contact_file"]).is_file())
        before = self.upload_mtimes()
        self.run_cli("preprocess_local.py", *self.process_args())
        self.assertEqual(before, self.upload_mtimes(), "Resume must not rewrite completed exports")
        self.run_cli("preprocess_local.py", "verify", "--output", self.output)

        # An absent whole contact group has no DONE to discover by rglob; the
        # catalog and radar reference must still make verification reject it.
        shutil.rmtree(contacts)
        self.run_cli("preprocess_local.py", "verify", "--output", self.output, expected=2)
        result = json.loads((self.output / "verification.json").read_text())
        self.assertTrue(result["missing_export_groups"])
        self.assertEqual(len(result["radar_with_missing_contact"]), 2)
        self.run_cli("preprocess_local.py", *self.process_args())
        self.assertEqual(len(list(contacts.glob("*/contact.h5"))), 1)
        self.run_cli("preprocess_local.py", "verify", "--output", self.output)

        # The expected-radar catalog also catches an entire absent radar group.
        victim = sorted(p for p in radar.iterdir() if p.is_dir())[0]
        removed = self.root / "temporarily_removed_radar"
        victim.rename(removed)
        self.run_cli("preprocess_local.py", "verify", "--output", self.output, expected=2)
        result = json.loads((self.output / "verification.json").read_text())
        self.assertIn(f"radar/{victim.name}/DONE.json", result["missing_export_groups"])
        removed.rename(victim)

        self.run_cli("make_qc_report.py", "--output", self.output)
        page = (self.output / "qc_report.html").read_text(encoding="utf-8")
        self.assertIn('<html lang="zh-CN">', page)
        self.assertIn("normal_1000r_40cm_rep1", page)
        self.assertIn("normal_1000r_40cm_rep2", page)
        self.assertIn("normal_1000r_session", page)
        self.assertEqual(page.count("<svg "), 2)

        # Source changes must not silently reuse outputs built from the old DAT.
        before = self.upload_mtimes()
        with contact_source.open("a", encoding="ascii") as stream:
            stream.write("!1,0.1,0.2,0.3;")
        self.run_cli("preprocess_local.py", *self.process_args(), expected=2)
        self.assertEqual(before, self.upload_mtimes())

    def test_full_manifest_and_ambiguous_contact_candidates(self):
        aliases = {"normal": "normal", "inBroken": "in", "outBroken": "out", "roll": "ball"}
        for label in build_manifest.LABELS:
            for rpm in (1000, 2000, 3000):
                (self.raw / f"2026_9_27_20-48-50-{aliases[label]}-{rpm}r.DAT").touch()
                for cm in (20, 40, 80):
                    for rep in range(1, 5):
                        (self.raw / f"20260927-1830{rep:02d}-{aliases[label]}-{rpm}r-{cm}cm.bin").touch()
        rows, review, summary = build_manifest.generate(self.raw)
        self.assertEqual(len(rows), 144)
        self.assertEqual(len({row["recording_id"] for row in rows}), 144)
        self.assertEqual(summary["unique_mapped_contact_ids"], 12)
        self.assertTrue(summary["all_expected_radar_conditions_have_four"])
        self.assertFalse(review)
        build_manifest.write_csv(self.manifest, rows, build_manifest.FIELDS)
        with self.manifest.open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 144)
        (self.raw / "2026_9_27_21-48-50-out-2000r.DAT").touch()
        (self.raw / "adc_data_Raw_0.BIN").touch()
        rows, review, summary = build_manifest.generate(self.raw)
        ambiguous = [row for row in rows if row["label"] == "outBroken" and row["rpm"] == 2000]
        self.assertEqual(len(ambiguous), 12)
        self.assertTrue(all(not row["contact_path"] and not row["contact_id"]
                            and not row["paired_confirmed"] for row in ambiguous))
        self.assertEqual(summary["review_issue_counts"]["multiple_contact_candidates"], 2)
        self.assertEqual(summary["review_issue_counts"]["unrecognized_filename"], 1)
        self.assertTrue(summary["review_required"])


if __name__ == "__main__":
    unittest.main()
