"""Analytical checks only. No real data, model training, or hardware validation."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import hadamard

ROOT = Path(__file__).resolve().parents[1]
c = 299_792_458.0
tc = (420 + 80) * 1e-6
tf = .1
n = 192
fs_adc = 3_430_000.
slope = 49.99e12
adc_duration = 256 / fs_adc
bandwidth = slope * adc_duration
frame_bytes = 256 * 4 * 4 * n
times = np.arange(2)[:, None] * tf + np.arange(n)[None, :] * tc
assert np.isclose(1/tc, 2000)
assert np.isclose(tf-n*tc, .004)
assert np.isclose(times[1, 0]-times[0, -1], .0045)
assert frame_bytes == 786432

rng = np.random.default_rng(42)
d = 128
x = rng.normal(size=(4096, d)) * np.geomspace(.2, 4, d)
mu = x.mean(0)
cov = np.cov(x-mu, rowvar=False)
eig, u = np.linalg.eigh(cov)
h = hadamard(d) / np.sqrt(d)
sigma = np.sqrt(eig.mean())
w = h @ u.T / sigma
z = (x-mu) @ w.T
diag_err = float(np.max(np.abs(np.diag(np.cov(z, rowvar=False))-1)))
offdiag = np.cov(z, rowvar=False) - np.eye(d)
e = rng.normal(size=(16, d))
distance_err = float(np.max(np.abs(np.sum((e @ w.T)**2, axis=1)
                                 - np.sum(e**2, axis=1)/sigma**2)))
inverse_err = float(np.max(np.abs(mu+sigma*z @ h @ u.T-x)))
assert diag_err < 1e-10
assert distance_err < 1e-10
assert inverse_err < 1e-10

# Algebraic spectral-coherence identity for one frequency pair, not an estimator.
sp = rng.uniform(.2, 3, 200)
sm = rng.uniform(.2, 3, 200)
coh = .6*np.exp(1j*rng.uniform(-np.pi, np.pi, 200))
sc = coh*np.sqrt(sp*sm)
hp = rng.uniform(.1, 4, 200)*np.exp(1j*rng.uniform(-np.pi, np.pi, 200))
hm = rng.uniform(.1, 4, 200)*np.exp(1j*rng.uniform(-np.pi, np.pi, 200))
coh_filtered = hp*hm.conj()*sc/np.sqrt(abs(hp)**2*sp*abs(hm)**2*sm)
coh_err = float(np.max(np.abs(abs(coh_filtered)-abs(coh))))
assert coh_err < 1e-12

out = {
    "status": "analytical_checks_passed_not_hardware_or_model_validation",
    "hardware_reported_by_user": "AWR1843 + DCA1000",
    "assumptions": ["Standard SDK CLI meanings", "Configuration successfully applied",
                    "4 RX, 256 complex samples per chirp, int16 I and Q",
                    "ADC payload only, no headers, no packet loss"],
    "radar": {
        "chirp_interval_us": tc*1e6, "in_frame_slow_rate_hz": 1/tc,
        "frame_period_ms": tf*1e3, "frame_rate_hz": 1/tf,
        "chirps_per_frame": n, "nominal_chirp_block_ms": n*tc*1e3,
        "nominal_inter_frame_idle_ms": (tf-n*tc)*1e3,
        "last_to_next_first_chirp_ms": (times[1,0]-times[0,-1])*1e3,
        "missing_grid_slots_per_frame": round(tf/tc)-n,
        "mean_chirps_per_second_not_uniform_sample_rate": n/tf,
        "single_frame_fft_spacing_hz": (1/tc)/n,
        "adc_duration_us": adc_duration*1e6,
        "adc_end_within_ramp_us": 5+adc_duration*1e6,
        "effective_sweep_bandwidth_ghz": bandwidth/1e9,
        "ideal_range_resolution_cm": c/(2*bandwidth)*100,
        "wavelength_at_start_mm": c/77e9*1000,
        "frame_adc_bytes": frame_bytes,
        "mean_adc_MB_per_s": frame_bytes/tf/1e6,
        "adc_GB_for_270_120s_runs": frame_bytes/tf*270*120/1e9,
        "beat_khz_at_distances_m": {str(r):2*slope*r/c/1e3 for r in [.5, 1., 1.5, 2.]},
        "range_at_350khz_m": 350e3*c/(2*slope),
    },
    "accelerometer": {"assumed_rate_hz": 5000, "samples_per_axis": 4096,
                      "packet_duration_s": 4096/5000,
                      "fft_spacing_hz": 5000/4096,
                      "dct_II_basis_spacing_hz": 5000/(2*4096)},
    "phi_s_check": {"max_coordinate_variance_error": diag_err,
                    "max_scaled_distance_error": distance_err,
                    "max_inverse_error": inverse_err,
                    "max_off_diagonal_covariance": float(np.max(abs(offdiag))),
                    "interpretation": "Equal coordinate variance; not complete whitening or structural invariance"},
    "cyclic_coherence_check": {"max_modulus_error_ideal_nonzero_LTI": coh_err,
                               "interpretation": "Algebraic identity only; not a finite-record estimator validation"},
}
(ROOT / "核算结果.json").write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n")
print(json.dumps(out, indent=2, ensure_ascii=False))
