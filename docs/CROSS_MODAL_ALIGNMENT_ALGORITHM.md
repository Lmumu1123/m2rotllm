# Corrected cross-modal alignment algorithm

This document describes the current interpretation of the mmWave radar and tactile experiments. The runnable implementation is in `cross_modal_alignment/`.

## 1. Data model

The experiment contains two independently sampled streams:

- tactile features from `data_0.csv` or timestamps/features decoded from `data_0.bin`;
- TI mmWave radar ADC samples from a raw `.bin` file.

The recorded radar file is real ADC data, not interleaved complex I/Q. Its verified layout is:

```text
4 RX × 512 real int16 samples/chirp
3 chirps/loop (TX0/TX1/TX2)
666.67 loops/second (approximately 1.5 ms/loop)
```

The corrected parser is `rd_alignment_test.py`. The older `align_cross_modal.py` is retained as a baseline for the initial magnitude-proxy experiment and must not be used to support the final alignment claim.

## 2. Corrected radar feature

For each TX0 chirp, apply a Hann window and evaluate the DFT at a target range bin. The default target bin is selected from the average range profile; the recorded run selected bin 13. After fixed phase rotation and averaging across RX channels, the complex slow-time value is `z[k]`.

The radial displacement proxy is:

```text
d[k] = unwrap(angle(z[k])) × lambda / (4π)
```

where `lambda = c / 77 GHz`. The main comparison uses short-window standard deviation of `d[k]` as a displacement envelope, plus the legacy magnitude and Doppler features for diagnosis.

## 3. Tactile parsing

The exported CSV has one extra field at the end of each row. When pandas reads it, the actual time string becomes the dataframe index and the feature names are shifted by one position. `load_tactile_fixed()` removes the trailing field and restores the header alignment before selecting:

```text
X位移幅值(um), Y位移幅值(um), Z位移幅值(um)
X速度幅值(mm/s), X加速度幅值(g)
```

This correction is essential: the initial script labelled a Y-axis column as X.

## 4. Alignment and evaluation

The corrected experiment evaluates three distinct questions:

1. **Timestamp alignment:** use the radar filename clock and tactile clock to place both streams on a common wall-clock axis.
2. **Lag diagnostics:** scan a limited lag range and compare the beginning and end of the overlap to detect an offset or clock drift.
3. **Elastic alignment stress test:** run coarse lag + constrained DTW only as a diagnostic, and compare it with circularly shifted null inputs. A high DTW score alone is not evidence of physical alignment.

The direct, timestamp-aligned Pearson correlation is reported at smoothing scales of 0, 0.5, 2, and 10 seconds. Slow trends are the meaningful comparison because the tactile stream is a vendor-generated envelope while radar phase displacement still contains fast carrier motion.

## 5. Current conclusions

The corrected run found:

- the radar stream contains about 86,528 loops (about 129.8 seconds) and the tactile stream about 153.5 seconds;
- the radar–tactile clock offset is about 13.6 seconds for the recorded pair;
- phase-displacement envelope and tactile X displacement show strong slow-trend agreement, reaching roughly `r=0.97` after 10-second smoothing;
- at the measured slow scales, cross-modal correlation is strongest for tactile X displacement, followed by Z, with Y much weaker; this is a cross-modal result, not a claim that the raw tactile displacement variance is largest on X;
- the old full-ADC magnitude proxy is noisy and can have the opposite sign;
- the initial `0.644` DTW improvement is not a valid standalone result because circularly shifted inputs can also obtain high DTW scores.

Therefore, use direct timestamp calibration plus the phase-displacement envelope as the current baseline. Re-evaluate DTW on data without reliable timestamps and always include the circular-shift null test.
