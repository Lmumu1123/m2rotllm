# Current project status

The repository combines two reproducibility tracks:

1. RotLLM SFN/LLM code with an MBHM adapter and a 10-class bearing-only training/evaluation runner.
2. Cross-modal mmWave radar and tactile validation code.

The cross-modal work was corrected after checking the raw files. The radar file is real ADC data with 512 real samples per RX channel, not interleaved complex I/Q; the loop rate is 666.67 Hz; and the tactile CSV has a one-column row/header offset. The corrected analysis uses a target range-bin phase displacement and timestamp alignment.

The final interpretation is deliberately conservative: slow-trend correspondence is strong and the radar-to-tactile correlation hierarchy is X > Z >> Y at the reported scales, but the initial magnitude-proxy + DTW score was inflated by parsing/time-scale errors and DTW overfitting. This hierarchy is not the same as raw tactile displacement variance, which is largest on Y in the recorded stream. New experiments should improve radar SNR and evaluate DTW on data without trustworthy timestamps, with circular-shift null tests included.

See `docs/REPRODUCTION_GUIDE.md` for commands and `docs/ALIGNMENT_STATUS.md` for the recorded measurements.
