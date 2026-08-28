"""
Cross-modal time alignment for mmWave radar IQ and tactile sensor data.

- Extract a 1D radar slow-time proxy from TI mmWave ADC IQ data (.bin)
- Extract a 1D tactile feature sequence from CSV
- Align them with coarse delay search + constrained DTW
"""

__all__ = ["align_cross_modal", "dtw_alignment"]
