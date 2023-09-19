import copy
import os, shutil
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, peak_prominences


def amp_spectrum(t, y, fmin=None, fmax=None, nyq_mult=1., oversample_factor=5.):
    tmax = t.max()
    tmin = t.min()
    df = 1.0 / (tmax - tmin)
    
    if fmin is None:
        fmin = df
    if fmax is None:
        fmax = (0.5 / np.median(np.diff(t)))*nyq_mult

    freq = np.arange(fmin, fmax, df / oversample_factor)
    
    model = LombScargle(t, y)
    sc = model.power(freq, method="fast", normalization="psd")

    amp = np.sqrt(4./len(t)) * np.sqrt(sc)

    return freq, amp

# Define a sinusoidal function to fit the peaks
def sinusoidal_model(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C

def prewhitener(time, flux, f_sigma=3, max_iterations=5, fmin=None, fmax=90, nyq_mult=1, oversample_factor=5, name='star'):
    if not os.path.exists(f'pw/{name}'):
        os.makedirs(f'pw/{name}')
    else:
        shutil.rmtree(f'pw/{name}')
        os.makedirs(f'pw/{name}')

    flux_i = copy.deepcopy(flux)
    freqs, amps = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
    freqs_i = copy.deepcopy(freqs)
    amps_i = copy.deepcopy(amps)
    peaks = np.array([], dtype=int)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    peak_freqs = np.array([])
    peak_amps = np.array([])
    for n in range(max_iterations):
        freqs_i, amps_i = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)

        # Find peaks in the periodogram
        peaks_tmp = find_peaks(amps_i)[0]
        peaks_widths_i = peak_widths(amps_i, peaks=peaks_tmp, rel_height=0.7)[0]
        width = np.median(peaks_widths_i)
        distance = width/(np.median(np.diff(freqs_i))-10*np.std(np.diff(freqs_i)))
        # prominence = np.median(sp.signal.peak_prominences(amps_i, peaks=peaks_tmp, wlen=(width+3*np.std(width)))[0])
        prominence = np.median(peak_prominences(amps_i, peaks=peaks_tmp)[0])

        peaks_i = find_peaks(amps_i, height=np.median(amps_i)+f_sigma*np.std(amps_i), width=width, prominence=prominence, distance=distance)[0]

        ## If no peaks are found, break the loop
        if len(peaks_i) == 0:
            break

        peaks = np.append(peaks, peaks_i)

        peak_freqs_i = []
        peak_amps_i = []
        for i, freq in enumerate(freqs_i[peaks_i]):
            omega = 2 * np.pi * freq
            p0 = [amps_i[peaks_i][i], omega, 0, 0]
            params, pcov = curve_fit(sinusoidal_model, time, flux_i, p0=p0)
            if params[0] < 0:
                params[0] *= -1.
                params[2] += np.pi

            peak_freqs_i.append(params[1]/(2*np.pi))
            peak_amps_i.append(params[0])
            # peak_freqs_i.append(freq)
            # peak_amps_i.append(amps_i[peaks_i][i]) 
            
            # # Subtract the fitted sinusoid from the original signal
            flux_i -= sinusoidal_model(time, *params)
            freqs_i, amps_i = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)

        # Plot the periodogram and the folded light curve
        ax.cla()
        ax.plot(freqs_i, amps_i)
        ax.set_title(f"Pre-whitening iteration {n+1}")
        ax.set_xlabel("Frequency (1/day)")
        ax.set_ylabel("Power")
        plt.savefig(f'pw/{name}/pg_{n+1}')
        peak_freqs = np.append(peak_freqs, peak_freqs_i)
        peak_amps = np.append(peak_amps, peak_amps_i)

    ## freqs of these sorted peaks
    peak_freqs = peak_freqs[np.argsort(peak_amps)[::-1]]
    ## sort frequencies in ascending order
    peak_amps = sorted(peak_amps, reverse=True)

    # # # Harmonic ratio checking
    tolerance = 0.01 
    harmonics_idx = []
    for i in range(len(peak_freqs)):
        for j in range(i+1, len(peak_freqs)):
            ratio = peak_freqs[j]/peak_freqs[i]
            if np.abs(ratio - round(ratio)) < tolerance:
                harmonics_idx.append(j)
    peak_freqs = np.delete(peak_freqs, harmonics_idx)
    peak_amps = np.delete(peak_amps, harmonics_idx)

    print('Done!')
    return peaks, peak_freqs, peak_amps, freqs, amps

if __name__ == "__main__":
    star = 'TIC171591531'
    # star = 'HD20203'
    # star = 'HD47129'
    # star = 'V647Tau'
    lk_search = lk.search_lightcurve(star, mission="TESS", cadence=120)
    lc = lk_search[0].download().remove_nans().remove_outliers()

    # Extract time and flux from the light curve
    time = lc.time.value
    flux = lc.flux.value

    # Pre-whiten the light curve
    peaks, peak_freqs, peak_amps, freqs, amps = prewhitener(time, flux, f_sigma=5, max_iterations=10, name=star)

    df = pd.DataFrame({'freq': peak_freqs, 'amp': peak_amps}).sort_values(by='freq')
    df.to_csv(f'pw/{star}_frequencies.csv', index=False)

