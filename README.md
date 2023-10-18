The `prewhitener.py` module is designed for pre-whitening a light curve by iteratively fitting sinusoids to the peaks in the amplitude spectrum. Starting with the highest peak, the function fits and subsequently removes it from the spectrum. This iterative process continues until a specified Signal-to-Noise Ratio (SNR) threshold is reached.

Install with `pip install git+https://github.com/gautam-404/PreWhitener.git`

## Usage

```python
from prewhitener import PreWhitener

# Create a PreWhitener object by specifying the KIC/TIC/HD ID
pw = PW.PreWhitener(name='HDxxxxx')
pw.auto()
```

## Available parameters
```python
PW.PreWhitener(name=None, ## KIC/TIC/HD ID, will be used to download light curve if no `lc`` is provided. Type: `str`
                lc=None, ## Light curve, if already available. Type: `lightkurve.lightcurve.LightCurve` or `pandas.DataFrame` or tuple
                max_iterations=100,  ## Maximum number of iterations. Type: `int`
                snr_threshold=5,    ## Signal-to-Noise Ratio threshold. Type: `float`
                flag_harmonics=True,    ## Flag harmonics of the fitted frequencies. Type: `bool`
                harmonic_tolerance=0.001,   ## Tolerance for flagging harmonics. Type: `float`
                frequency_resolution=4/27,  ## Frequency resolution of the amplitude spectrum. Type: `float`
                fbounds=None,   ## Frequency bounds for the amplitude spectrum. Type: `tuple` or `list`
                nyq_mult=1,    ## Multiplier for the Nyquist frequency. Type: `int`
                oversample_factor=5,    ## Oversampling factor for the amplitude spectrum. Type: `int`
                mode='amplitude')   ## Mode of the spectrum. Can be 'amplitude' or 'power'. Type: `str`
```

## Available methods
```python
pw.auto(make_plot=True, save=True) ## Automatically pre-whiten the light curve
pw.interate()   ## Iteratively pre-whiten the light curve once a PreWhitening object has been created
pw.post_pw(make_plot=True, save=True)   ## Post-processing of the pre-whitened light curve
pw.post_pw_plot()   ## Plot the pre-whitened light curve and the identified peaks in the amplitude spectrum
pw.amplitude_power_spectrum()   ## Get the amplitude/power spectrum
pw.get_lightcurve()   ## Get the pre-whitened light curve
```

