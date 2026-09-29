"""Scientific/interface checks independent of real bearing labels."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from preprocess import dcn, parse_packets, masked_power, decode_range, FREQ
from server_encode_contact import make_pairs
import json


class PreprocessingTests(unittest.TestCase):
    def test_dcn_physical_frequency_and_normalization(self):
        t = (np.arange(4000)+.5)/4000
        x = np.cos(2*np.pi*125*t)[:, None]*np.array([[1., 2., 3.]])
        out = dcn(x+np.array([[.1, -.2, -1.]]))
        np.testing.assert_array_equal(abs(out).argmax(axis=1), [250]*3)
        np.testing.assert_allclose(np.mean(out**2, axis=1), .0001, rtol=1e-5)
        np.testing.assert_allclose(dcn(x), out, atol=1e-7)
        self.assertTrue(np.all(out[:, 4000:] == 0))
        with self.assertRaises(ValueError): dcn(np.zeros((4000, 3)))
        with self.assertRaises(ValueError): dcn(np.zeros((4096, 3)))

    def test_packet_gaps_corruption_and_unterminated_tail(self):
        def packet(skip=None):
            return b''.join(f'{"!" if i == 1 else ""}{i},0.1,0.2,-0.9;'.encode()
                            for i in range(1, 4097) if i != skip)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'a.DAT'
            p.write_bytes(b'partial;' + packet() + packet(100) + packet()[:-1])
            packs, qc = parse_packets(p)
        self.assertEqual(len(packs), 1)
        self.assertEqual(packs[0][1].shape, (4096, 3))
        self.assertTrue(any(x['reason'] == 'counter_discontinuity' for x in qc['issues']))

    def test_masked_sinusoid_uses_actual_times(self):
        t = np.arange(4000)/2000
        mask = np.tile(np.arange(200) < 192, 20)
        y = 2*np.cos(2*np.pi*123.5*t) + 3*np.sin(2*np.pi*123.5*t) + 17
        y[~mask] = 1e9
        p = masked_power(y, mask)[:, 0]
        k = np.flatnonzero(FREQ == 123.5)[0]
        self.assertEqual(p.argmax(), k)
        self.assertAlmostEqual(p[k], 6.5, places=9)
        # Direct least squares cross-check at several off-peak frequencies.
        for f in [5., 50., 123.5, 400., 800.]:
            design = np.column_stack([np.ones(mask.sum()), np.cos(2*np.pi*f*t[mask]), np.sin(2*np.pi*f*t[mask])])
            beta = np.linalg.lstsq(design, y[mask], rcond=None)[0]
            self.assertAlmostEqual(p[np.flatnonzero(FREQ == f)[0]], (beta[1]**2+beta[2]**2)/2, places=9)

    def test_2i2q_layout_range_peak(self):
        z = (1000*np.exp(-2j*np.pi*20*np.arange(256)/256))[None, None, :]
        z = np.tile(z, (2, 4, 1)).reshape(-1, 2)
        raw = np.column_stack([z[:, 0].real, z[:, 1].real, z[:, 0].imag, z[:, 1].imag]).round().astype('<i2').ravel()
        rf = decode_range(raw)
        self.assertEqual(rf.shape, (2, 4, 512))
        np.testing.assert_array_equal(abs(rf).argmax(-1), np.full((2, 4), 40))

    def test_reference_isolation_and_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath('summary.json').write_text(json.dumps([
                dict(output='n.npz', modality='contact', label=0, windows=2),
                dict(output='q.npz', modality='contact', label=1, windows=3)]))
            assignments = {'n.npz': dict(condition_id='c', independent_run_id='n', split='train'),
                           'q.npz': dict(condition_id='c', independent_run_id='q', split='test')}
            pairs, missing = make_pairs(root, assignments)
            self.assertEqual(len(pairs), 3)
            self.assertEqual(missing[0]['file'], 'n.npz')
            assignments['n.npz']['split'] = 'test'
            self.assertEqual(len(make_pairs(root, assignments)[0]), 0)
            assignments['n.npz']['split'] = 'train'
            assignments['q.npz']['independent_run_id'] = 'n'
            with self.assertRaises(ValueError): make_pairs(root, assignments)


if __name__ == '__main__': unittest.main()
