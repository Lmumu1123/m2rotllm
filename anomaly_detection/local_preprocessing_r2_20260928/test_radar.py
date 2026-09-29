"""Synthetic data-contract tests; these do not validate real capture quality."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from radar_r2 import parse_cfg, process_radar


CFG = """sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 1 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 77 120 5 80 0 0 49.99 1 256 3430 0 0 48
chirpCfg 0 0 0 0 0 0 0 1
frameCfg 0 0 192 0 40 1 0
lvdsStreamCfg -1 0 1 0
sensorStart
"""


def synthetic_frame():
    """Positive FFT bin 20 under historical I-iQ; 60-Hz phase modulation."""
    fast_phase = 2 * np.pi * 20 * np.arange(256)[None, None, :] / 512
    vibration = .08 * np.sin(2 * np.pi * 60 * np.arange(192)[:, None, None] / 5000)
    antenna_phase = np.arange(4)[None, :, None] * .1
    signal = 2000 * np.exp(1j * (fast_phase + vibration + antenna_phase))
    paired = signal.reshape(-1, 2)
    words = np.empty((len(paired), 4), dtype="<i2")
    words[:, 0] = np.rint(paired[:, 0].real)
    words[:, 1] = np.rint(paired[:, 1].real)
    words[:, 2] = np.rint(-paired[:, 0].imag)
    words[:, 3] = np.rint(-paired[:, 1].imag)
    return words.tobytes()


class RadarExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = self.root / "R2.cfg"
        self.cfg.write_text(CFG, encoding="utf-8")
        self.frame = synthetic_frame()

    def write_frames(self, name, count, tail=b""):
        path = self.root / name
        with path.open("wb") as stream:
            for _ in range(count):
                stream.write(self.frame)
            stream.write(tail)
        return path

    def test_r2_timing_and_fifty_frame_window(self):
        cfg = parse_cfg(self.cfg)
        self.assertAlmostEqual(cfg["within_frame_sample_rate_hz"], 5000)
        self.assertEqual(cfg["window_frames"], 50)
        self.assertEqual(cfg["missing_nominal_grid_slots_per_frame"], 8)
        self.assertAlmostEqual(cfg["nominal_frame_gap_s"], .0016)
        raw = self.write_frames("fifty.bin", 50)
        out = self.root / "fifty.h5"
        qc = process_radar(raw, self.cfg, out, 40, batch_frames=3)
        self.assertEqual(qc["complete_two_second_windows"], 1)
        self.assertEqual(qc["frames_exported"], 50)
        self.assertAlmostEqual(qc["nominal_export_duration_s"], 2)
        self.assertAlmostEqual(qc["nominal_last_observed_chirp_s"], 1.9982)
        self.assertEqual(qc["selected_center_bin"], 20)
        with h5py.File(out, "r") as handle:
            self.assertEqual(handle["iq"].shape, (50, 192, 4, 5))
            self.assertEqual(handle["iq"].dtype, np.dtype("complex64"))
            self.assertEqual(int(handle["valid_chirp"][:].sum()), 9600)
            observed = (handle["frame_start_s"][:][:, None] + handle["chirp_offset_s"][:][None, :]).ravel()
            ticks = np.rint(observed * 5000).astype(int)
            self.assertEqual(len(set(ticks)), 9600)
            self.assertEqual(10000 - len(ticks), 400)
            self.assertAlmostEqual(observed[-1], 1.9982)
            np.testing.assert_array_equal(handle["window_first_frame"][:], [0])
            np.testing.assert_array_equal(handle["window_frame_count"][:], [50])
            self.assertTrue(bool(handle.attrs["export_complete"]))
            self.assertEqual(json.loads(handle.attrs["qc_json"])["source_hash_scope"],
                             "concatenated_complete_sources")
        with self.assertRaises(FileExistsError):
            process_radar(raw, self.cfg, out, 40)

    def test_arbitrary_part_boundary_tail_and_hash(self):
        data = self.frame * 3 + b"abc"
        first, second = self.root / "part0.bin", self.root / "part1.bin"
        first.write_bytes(data[:10001])
        second.write_bytes(data[10001:])
        out, wide = self.root / "split.h5", self.root / "wide.h5"
        qc = process_radar([first, second], self.cfg, out, 40,
                           archive_h5=wide, batch_frames=2)
        self.assertEqual(qc["frames_exported"], 3)
        self.assertEqual(qc["trailing_incomplete_frame_bytes"], 3)
        self.assertEqual(qc["selected_center_bin"], 20)
        self.assertEqual(qc["source_sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(qc["source_parts"][0]["sha256"], hashlib.sha256(data[:10001]).hexdigest())
        self.assertEqual(qc["source_parts"][1]["sha256"], hashlib.sha256(data[10001:]).hexdigest())
        with h5py.File(out, "r") as handle, h5py.File(wide, "r") as archive:
            self.assertEqual(len(handle["window_first_frame"]), 0)
            self.assertEqual(archive["iq"].shape[:3], (3, 192, 4))
            wide_bins = archive["range_bin_indices"][:]
            where = [int(np.flatnonzero(wide_bins == b)[0]) for b in handle["range_bin_indices"][:]]
            np.testing.assert_array_equal(archive["iq"][:][..., where], handle["iq"][:])

    def test_wrong_conjugation_is_flagged(self):
        raw = self.write_frames("sign.bin", 3)
        proper = process_radar(raw, self.cfg, self.root / "proper.h5", 40)
        wrong = process_radar(raw, self.cfg, self.root / "wrong.h5", 40, iq_order="iiqq_pos")
        self.assertLess(proper["pilot_negative_to_positive_dynamic_ratio"], 1e-4)
        self.assertGreater(wrong["pilot_negative_to_positive_dynamic_ratio"], 4)
        self.assertIn("negative_frequency_mirror_dominates_check_IQ_convention", wrong["quality_flags"])

    def test_pilot_hash_is_only_exported_prefix(self):
        raw = self.write_frames("pilot_source.bin", 3, b"abc")
        qc = process_radar(raw, self.cfg, self.root / "pilot.h5", 40, max_frames=1)
        self.assertEqual(qc["status"], "pilot_partial")
        self.assertEqual(qc["frames_available"], 3)
        self.assertEqual(qc["frames_exported"], 1)
        self.assertEqual(qc["hashed_bytes"], len(self.frame))
        self.assertEqual(qc["source_hash_scope"], "processed_prefix")
        self.assertEqual(qc["source_sha256"], hashlib.sha256(self.frame).hexdigest())
        self.assertNotEqual(qc["source_sha256"], hashlib.sha256(self.frame * 3 + b"abc").hexdigest())

    def test_zero_chirp_remains_in_timing_but_is_masked(self):
        frame = bytearray(self.frame)
        chirp_bytes = len(frame) // 192
        frame[10 * chirp_bytes:11 * chirp_bytes] = bytes(chirp_bytes)
        raw = self.root / "zero.bin"
        raw.write_bytes(frame)
        out = self.root / "zero.h5"
        qc = process_radar(raw, self.cfg, out, 40)
        self.assertEqual(qc["all_zero_ADC_chirps"], 1)
        with h5py.File(out, "r") as handle:
            self.assertFalse(handle["valid_chirp"][0, 10])
            self.assertEqual(handle["iq"].shape[1], 192)
            self.assertAlmostEqual(handle["chirp_offset_s"][10], .002)
            self.assertTrue(np.all(handle["iq"][0, 10] == 0))

    def test_unsupported_cfg_is_rejected(self):
        for replacement in (CFG.replace("channelCfg 15 1 0", "channelCfg 15 7 0"),
                            CFG.replace("frameCfg 0 0 192 0 40 1 0", "frameCfg 0 0 192 0 39 1 0"),
                            CFG.replace("lvdsStreamCfg -1 0 1 0", "lvdsStreamCfg -1 1 1 0")):
            self.cfg.write_text(replacement, encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_cfg(self.cfg)


if __name__ == "__main__":
    unittest.main()
