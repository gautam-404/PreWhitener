import copy
import os
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

def sinusoidal_model(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C

def prewhitener(time, flux, max_iterations=100, name='star', remove_harmonics=True):
    os.makedirs(f'pw/{name}', exist_ok=True)
    flux_i = copy.deepcopy(flux)
    freqs, amps = amp_spectrum(t=time, y=flux_i, fmin=None, fmax=90, nyq_mult=1., oversample_factor=5)
    freqs_i = copy.deepcopy(freqs)
    amps_i = copy.deepcopy(amps)
    peaks = np.array([], dtype=int)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for n in range(max_iterations):
        power_i = LombScargle(time, flux_i, normalization="psd").power(freqs_i)
        amps_i = 2 * (abs(power_i) / len(time)) ** 0.5

        # Find peaks in the periodogram
        peaks_tmp, _ = find_peaks(amps_i, height=np.median(amps_i)+3*np.std(amps_i))
        peaks_widths_i = peak_widths(amps_i, peaks=peaks_tmp)[0]
        width = np.mean(peaks_widths_i)
        distance = width/np.median(np.diff(freqs_i))
        prominence = np.median(peak_prominences(amps_i, peaks=peaks_tmp)[0])

        peaks_i, _ = find_peaks(amps_i, height=np.median(amps_i)+3*np.std(amps_i), width=width, prominence=prominence, distance=distance)
        peaks = np.append(peaks, peaks_i)

        # If no peaks are found, break the loop
        if len(peaks_i) == 0:
            break

        highest_peak_id = np.argmax(amps_i[peaks_i])
        highest_peak_frequency = freqs_i[peaks_i[highest_peak_id]]
        
        # Fit a sinusoid to the original data at the frequency of the highest peak
        omega = 2 * np.pi * highest_peak_frequency
        p0 = [np.median(amps_i), omega, 0.5, 0.5]
        params, pcov = curve_fit(sinusoidal_model, time, flux_i, p0=p0)
        # perr = np.sqrt(np.diag(pcov))

        # # Subtract the fitted sinusoid from the original signal
        flux_i -= sinusoidal_model(time, *params)

        # Plot the periodogram
        ax.cla()
        ax.plot(freqs_i, amps_i)
        ax.set_title(f"Pre-whitening iteration {n+1}")
        ax.set_xlabel("Frequency (1/day)")
        ax.set_ylabel("Amplitude")
        plt.savefig(f'pw/{name}/pg_{n+1}.png')

    ## peak amps and freqs
    peak_amps = sorted(amps[peaks], reverse=True)
    peak_freqs = freqs[peaks][np.argsort(amps[peaks])[::-1]]

    if remove_harmonics:
        ## Harmonic ratio checking
        tolerance = 0.01  
        harmonics_idx = []
        for i in range(len(peak_freqs)):
            for j in range(i+1, len(peak_freqs)):
                ratio = peak_freqs[j]/peak_freqs[i]
                if np.abs(ratio - round(ratio)) < tolerance:
                    harmonics_idx.append(j)
        peak_freqs = np.delete(peak_freqs, harmonics_idx)
        peak_amps = np.delete(peak_amps, harmonics_idx)
        
    print(f'Found {len(peak_freqs)} frequencies')
    print('Done!')
    return peak_freqs, peak_amps

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
    peak_freqs, peak_amps = prewhitener(time, flux, max_iterations=100, name=star)
    df = pd.DataFrame({'freq': peak_freqs, 'amp': peak_amps}).sort_values(by='freq')
    df.to_csv(f'pw/{star}_frequencies.csv', index=False)

