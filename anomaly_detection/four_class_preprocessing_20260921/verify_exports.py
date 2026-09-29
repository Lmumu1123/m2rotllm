"""Read every exported array and check dimensions, finite values and provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
from preprocess import save_json, sha256
from server_radar_io import spectral_inputs, frame_inputs


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('results', type=Path)
    args = a.parse_args()
    root = args.results
    summary = json.loads((root/'summary.json').read_text())
    checks = []
    for r in summary:
        path = root/r['output']
        n = r['windows']
        side = json.loads(path.with_suffix('.json').read_text())
        assert len(side['windows']) == n
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                value = z[key]
                if np.issubdtype(value.dtype, np.number):
                    assert np.isfinite(value).all(), (path, key)
            assert (z['labels'] == r['label']).all()
            if r['modality'] == 'contact':
                assert z['raw_xyz'].shape == (n, 4000, 3)
                assert z['dcn'].shape == (n, 3, 24000)
                assert np.all(z['dcn'][:, :, 4000:] == 0)
                np.testing.assert_allclose(np.mean(z['dcn']**2, axis=-1), .0001, rtol=1e-5)
                assert len(set(w['byte_start'] for w in side['windows'])) == n
            else:
                assert z['iq_frames'].shape == (n, 20, 192, 12, 2)
                assert z['phase_rate_frames'].shape == (n, 20, 191, 12)
                t = z['observed_time_s']
                np.testing.assert_allclose(t[1:, 0]-t[:-1, -1], .0045)
                assert z['valid_phase_mask'].shape == (n, 20, 191)
                expected = z['valid_chirp_mask'][:, :, 1:] & z['valid_chirp_mask'][:, :, :-1]
                np.testing.assert_array_equal(expected, z['valid_phase_mask'])
        if r['modality'] == 'radar':
            assert spectral_inputs(path)[0].shape == (n, 1591)
            assert frame_inputs(path)[0].shape == (n, 20, 12, 2, 192)
        checks.append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha256(path), status='pass'))
    bags = json.loads((root/'recording_bags.json').read_text())
    assert len(bags) == len([r for r in summary if r['modality'] == 'contact'])
    assert all(not b['synchronized_window_pairs'] for b in bags)
    report = dict(status='pass', arrays_checked='all numeric arrays, every NPZ', files=checks,
        total_contact_windows=sum(r['windows'] for r in summary if r['modality'] == 'contact'),
        total_radar_windows=sum(r['windows'] for r in summary if r['modality'] == 'radar'),
        four_class_contact_windows=sum(r['windows'] for r in summary if r['modality'] == 'contact' and r['label'] >= 0),
        four_class_radar_windows=sum(r['windows'] for r in summary if r['modality'] == 'radar' and r['label'] >= 0),
        server_model_executed=False, reference_pairs_assigned=False,
        radar_geometry_confirmed=all(r.get('range_status') == 'geometry_supplied' for r in summary if r['modality'] == 'radar'))
    save_json(root/'verification.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))


if __name__ == '__main__': main()
