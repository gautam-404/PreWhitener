The `prewhitener.py` module is designed for pre-whitening a light curve by iteratively fitting sinusoids to the peaks in the amplitude spectrum. Starting with the highest peak, the function fits and subsequently removes it from the spectrum. This iterative process continues until a specified Signal-to-Noise Ratio (SNR) threshold is reached.

Key thresholds and parameters incorporated within this module include:
- `snr_threshold`: Dictates the SNR threshold requisite for peak extraction. The pre-whitening continues, iteratively removing peaks, until no peaks surpassing this SNR threshold remain. A quintessential value for this threshold, satisfactory for most δ Sct stars, stands at 5. In the case of high amplitude δ Sct stars, this value needs to be much lower.
  
- `harmonic_tolerance`: Determines the proximity a frequency should exhibit to an integer multiple of another frequency to be deemed its harmonic. This tolerance is often set to a small fraction, such as 0.01 or 1%. This ensures that if, for example, one frequency is approximately twice (within 1% deviation) another frequency, it is considered its second harmonic.

- `nearby_tolerance`: Specifies the permissible closeness between two frequencies before they are deemed overlapping. Ideally, this tolerance should be the Rayleigh resolution, which is the reciprocal of the observational time span, ensuring that indistinguishably close frequencies are not detected as significant peaks.
