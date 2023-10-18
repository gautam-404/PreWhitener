This module is designed for pre-whitening a light curve by iteratively fitting sinusoids to the peaks in the amplitude spectrum. Starting with the highest peak, the function fits and subsequently removes it from the spectrum. This iterative process continues until a specified Signal-to-Noise Ratio (SNR) threshold is reached.

Install with `pip install git+https://github.com/gautam-404/PreWhitener.git`

## Usage

```python
from PreWhitener import PreWhitener

# Create a PreWhitener object by specifying the KIC/TIC/HD ID
pw = PreWhitener(name='HDxxxxx')
pw.auto()
```

## Available parameters
`name`: Default=None, Type: `str`<br>
KIC/TIC/HD ID, will be used to download light curve if no `lc` is provided. 

`lc`: Default=None, Type: `lightkurve.lightcurve.LightCurve` or `pandas.DataFrame` or tuple<br>
Light curve, if already available. If not provided, the light curve will be downloaded using the `name` parameter.

`max_iterations`: Default=100, Type: `int`<br>
Maximum number of iterations.

`snr_threshold`: Default=5, Type: `float`<br>
Signal-to-Noise Ratio threshold.

`flag_harmonics`: Default=True, Type: `bool`<br>
Flag harmonics of the fitted frequencies.

`harmonic_tolerance`: Default=0.001, Type: `float`<br>
Tolerance for flagging harmonics.

`frequency_resolution`: Default=4/27, Type: `float`<br>
Frequency resolution of the amplitude spectrum.

`fbounds`: Default=None, Type: `tuple` or `list`<br>
Frequency bounds for the amplitude spectrum.

`nyq_mult`: Default=1, Type: `int`<br>
Multiplier for the Nyquist frequency.

`oversample_factor`: Default=5, Type: `int`<br>
Oversampling factor for the amplitude spectrum.

`mode`: Default='amplitude', Type: `str`<br>
Mode of the spectrum. Can be 'amplitude' or 'power'.

## Available methods
```python
pw.auto(make_plot=True, save=True) ## Automatically pre-whiten the light curve
pw.interate()   ## Iteratively pre-whiten the light curve once a PreWhitening object has been created
pw.post_pw(make_plot=True, save=True)   ## Post-processing of the pre-whitened light curve
pw.post_pw_plot()   ## Plot the pre-whitened light curve and the identified peaks in the amplitude spectrum
pw.amplitude_power_spectrum()   ## Get the amplitude/power spectrum
pw.get_lightcurve()   ## Get the pre-whitened light curve
```

