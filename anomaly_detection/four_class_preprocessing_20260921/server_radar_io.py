"""NumPy-only input adapters. No untrained network masquerades as pretrained encoder.

Feed spectral_inputs() to a 1591-input radar MLP, or frame_inputs() to an encoder
that processes each 192-chirp frame independently, masks pooling, then aggregates
20 frames. Learn the output projection to contact h[128] on training runs.
"""
import numpy as np


def spectral_inputs(path, variant='frame_log_shape'):
    if variant not in ('frame_log_shape', 'frame_log_power', 'coherent_log_shape'):
        raise ValueError('Unknown spectral branch')
    with np.load(path, allow_pickle=False) as z:
        x, y, f = z[variant], z['labels'], z['frequency_hz']
    if x.ndim != 2 or x.shape[1] != 1591 or not np.isfinite(x).all():
        raise ValueError('Invalid radar feature array')
    return x, y, f


def frame_inputs(path):
    with np.load(path, allow_pickle=False) as z:
        # [window, frame, RX/bin, I/Q, time]; gaps are never concatenated.
        iq = z['iq_frames'].transpose(0, 1, 3, 4, 2)
        mask = z['valid_chirp_mask']
        times = z['observed_time_s']
        labels = z['labels']
    return iq, mask, times, labels
