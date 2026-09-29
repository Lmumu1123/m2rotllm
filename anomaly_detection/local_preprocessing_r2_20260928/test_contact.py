"""Synthetic contact-parser tests; no recording data or network required."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from contact_packets import _tokens, process_contact


def packet_bytes(samples=4, constant=False, marker=True):
    rows = []
    for index in range(1, samples + 1):
        xyz = (5, 5, 5) if constant else (index, index + 1, index + 2)
        prefix = "!" if marker and index == 1 else ""
        rows.append(f"{prefix}{index},{xyz[0]},{xyz[1]},{xyz[2]};")
    return "".join(rows).encode("ascii")


class ContactExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def export(self, data, name="fixture", sample_rate_hz=4, packet_samples=4):
        source = self.directory / (name + ".DAT")
        destination = self.directory / (name + ".h5")
        source.write_bytes(data)
        qc = process_contact(source, destination, sample_rate_hz=sample_rate_hz,
                             packet_samples=packet_samples)
        return source, destination, qc

    def test_unterminated_last_record_does_not_complete_packet(self):
        complete = packet_bytes()
        _, destination, qc = self.export(complete + complete[:-1])
        self.assertEqual(qc["complete_packets"], 1)
        self.assertEqual(qc["incomplete_candidates"], 1)
        self.assertEqual(qc["issue_reason_counts"]["unterminated_tail"], 1)
        with h5py.File(destination) as output:
            self.assertEqual(output["raw_xyz"].shape, (1, 4, 3))

    def test_bad_prefix_and_counter_do_not_join_damaged_packets(self):
        prefixed = b"~" + packet_bytes()
        interrupted = b"!1,1,2,3;3,3,4,5;4,4,5,6;"
        complete = packet_bytes()
        raw = prefixed + interrupted + complete
        _, destination, qc = self.export(raw)
        self.assertEqual(qc["complete_packets"], 1)
        self.assertEqual(qc["issue_reason_counts"]["invalid_record"], 1)
        self.assertEqual(qc["issue_reason_counts"]["counter_discontinuity"], 1)
        self.assertEqual(qc["orphan_valid_rows"], 4)
        with h5py.File(destination) as output:
            self.assertEqual(int(output["byte_start"][0]), len(prefixed + interrupted))
            self.assertEqual(int(output["byte_end_exclusive"][0]), len(raw))
            self.assertEqual(int(output["packet_ordinal_observed"][0]), 2)

    def test_constant_packet_is_retained_and_flagged(self):
        _, destination, qc = self.export(packet_bytes(constant=True))
        self.assertEqual(qc["complete_packets"], 1)
        self.assertEqual(qc["packets_constant_all_axes"], 1)
        with h5py.File(destination) as output:
            np.testing.assert_array_equal(output["raw_xyz"][0], np.full((4, 3), 5.0))
            self.assertEqual(output["raw_xyz"].dtype, np.dtype("float64"))
            self.assertTrue(output["packet_stats/constant_axis"][0].all())
            self.assertTrue(np.isnan(output["packet_stats/kurtosis_pearson_xyz"][0]).all())

    def test_tokenizer_offsets_and_hash_are_independent_of_chunk_boundaries(self):
        class CappedReader(io.BytesIO):
            def __init__(self, data, cap):
                super().__init__(data)
                self.cap = cap

            def read(self, size=-1):
                return super().read(min(size, self.cap))

        raw = b"header;!1,1,2,3;2,4,5,6;;tail"
        expected = []
        offset = 0
        parts = raw.split(b";")
        for index, token in enumerate(parts):
            expected.append((token, offset, index < len(parts) - 1, len(token)))
            offset += len(token) + 1
        for cap in (1, 2, 7, 17, 1024):
            with self.subTest(read_cap=cap):
                digest = hashlib.sha256()
                result = list(_tokens(CappedReader(raw, cap), digest))
                self.assertEqual(result, expected)
                self.assertEqual(digest.hexdigest(), hashlib.sha256(raw).hexdigest())

    def test_actual_megabyte_boundary_and_oversized_token(self):
        # Place the first valid record across the parser's 1 MiB read boundary.
        prefix = b"x" * (1024 * 1024 - 5) + b";"
        complete = packet_bytes()
        _, destination, qc = self.export(prefix + complete)
        self.assertEqual(qc["complete_packets"], 1)
        self.assertEqual(qc["issue_reason_counts"]["oversized_record"], 1)
        self.assertEqual(qc["source_sha256"], hashlib.sha256(prefix + complete).hexdigest())
        with h5py.File(destination) as output:
            self.assertEqual(int(output["byte_start"][0]), len(prefix))

    def test_4000_hz_centered_window_retains_all_4096_samples(self):
        data = packet_bytes(samples=4096)
        _, destination, qc = self.export(data, sample_rate_hz=4000, packet_samples=4096)
        self.assertEqual(qc["window_slice"], [48, 4048])
        self.assertEqual(qc["source_sha256"], hashlib.sha256(data).hexdigest())
        self.assertIsNone(qc["recording_wall_duration_s"])
        with h5py.File(destination) as output:
            self.assertEqual(output["raw_xyz"].shape, (1, 4096, 3))
            np.testing.assert_array_equal(output.attrs["window_slice"], [48, 4048])
            np.testing.assert_array_equal(output["raw_xyz"][0, 0], [1, 2, 3])
            np.testing.assert_array_equal(output["raw_xyz"][0, -1], [4096, 4097, 4098])

    def test_no_overwrite(self):
        source, destination, _ = self.export(packet_bytes())
        before = hashlib.sha256(destination.read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            process_contact(source, destination, sample_rate_hz=4, packet_samples=4)
        self.assertEqual(hashlib.sha256(destination.read_bytes()).hexdigest(), before)

    def test_short_packet_is_preserved_without_fabricating_one_second_window(self):
        _, destination, qc = self.export(packet_bytes(), sample_rate_hz=8)
        self.assertFalse(qc["one_second_window_available"])
        self.assertIsNone(qc["window_slice"])
        with h5py.File(destination) as output:
            self.assertNotIn("window_slice", output.attrs)
            self.assertEqual(output["raw_xyz"].shape, (1, 4, 3))


if __name__ == "__main__":
    unittest.main()
