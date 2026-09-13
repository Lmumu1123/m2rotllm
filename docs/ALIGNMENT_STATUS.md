# Cross-modal alignment status

This is the current report for the recorded 2026-08 experiments.

## Corrected facts

- Radar: 4 RX × 512 real ADC samples/chirp, 3 chirps/loop, 666.67 loops/s.
- Radar duration: 86,528 loops, approximately 129.8 seconds.
- Tactile stream: approximately 6,420 feature rows and 153.5 seconds.
- Recorded clock offset for the main pair: approximately 13.6 seconds.
- Correct target-range-bin run: bin 13.

## Findings

The target-bin phase-displacement envelope agrees with tactile displacement at slow scales. In the recorded run, the radar-to-tactile correlation hierarchy was X > Z >> Y at the reported slow scales; the 10-second smoothed X correlation was approximately `0.97`. This should not be confused with raw tactile displacement variance: the recorded contact stream itself varies most on Y. Raw-scale correlation was low because the tactile stream is a vendor envelope and radar phase displacement has substantially higher fast variation/noise.

The initial report that coarse lag plus DTW raised the correlation to `0.644` has been withdrawn. The original code parsed the radar incorrectly, used the wrong time scale, and selected a shifted tactile column. More importantly, elastic DTW can produce high scores on circularly shifted null inputs. That score is retained only as a historical baseline.

## Recommended next steps

- Improve radar SNR through multi-chirp coherent accumulation and range-bin aggregation.
- Use timestamp calibration as the baseline whenever a reliable clock is available.
- Use phase displacement rather than the full-ADC magnitude proxy for vibration comparison.
- Treat DTW as an analysis tool, not a quality metric, unless a circular-shift null test is reported.
- Validate on start/stop, variable-speed, and fault-condition recordings.
