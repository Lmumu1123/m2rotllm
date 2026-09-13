"""Cross-modal alignment utilities for mmWave radar and tactile data.

This directory previously only contained raw data. We add code here to:
- Extract a 1D radar slow-time proxy from TI mmWave ADC IQ data (.bin)
- Extract a 1D tactile feature sequence from CSV
- Align them with timestamp calibration, coarse delay search, or constrained DTW
"""

__all__ = ["align_cross_modal", "dtw_alignment", "rd_alignment_test"]
