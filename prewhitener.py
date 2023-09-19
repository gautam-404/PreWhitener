import copy
import os, shutil
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from lmfit import Model
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


def is_harmonic(f1, f2, tolerance=0.01):
    ratio = f2 / f1
    closest_integer = round(ratio)
    is_harmonic = abs(ratio - closest_integer) < tolerance

    # Check for sub-harmonics
    sub_ratio = f1 / f2  # inverse of the original ratio
    closest_sub_integer = round(sub_ratio)
    is_sub_harmonic = abs(sub_ratio - closest_sub_integer) < tolerance
    return is_harmonic or is_sub_harmonic


# Sinusoidal function to fit the peaks
def sinusoidal_model(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C

def prewhitener(time, flux, f_sigma=3, remove_harmonics=True, max_iterations=5, fmin=None, fmax=90, nyq_mult=1, oversample_factor=5, name='star'):
    if not os.path.exists(f'pw/{name}'):
        os.makedirs(f'pw/{name}')
    else:
        shutil.rmtree(f'pw/{name}')
        os.makedirs(f'pw/{name}')

    ## Normalize the flux
    flux = flux/np.median(flux)
    flux_i = copy.deepcopy(flux)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = ax[0], ax[1]
    peak_freqs = []
    peak_amps = []
    peaks = []

    ## Initial amplitude spectrum
    freqs_i, amps_i = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
    amps_i *= 1000 # convert to ppt
    for n in range(max_iterations):
        ## Find all peaks to calculate the median prominence and width
        peaks_tmp = find_peaks(amps_i)[0]

        prominence_data = peak_prominences(amps_i, peaks=peaks_tmp)
        prominence = np.median(prominence_data[0])
        peaks_widths_i = peak_widths(amps_i, peaks=peaks_tmp, rel_height=0.5, prominence_data=prominence_data)[0]
        width = np.median(peaks_widths_i)  ## median fwhm 
        distance = width/(np.median(np.diff(freqs_i)))

        ## Find all peaks that fit the above criteria
        peaks_i = find_peaks(amps_i, height=np.median(amps_i)+f_sigma*np.std(amps_i), 
                             width=width, 
                             prominence=prominence,
                             distance=distance)[0].tolist()
        
        ## If no peaks are found, break the loop
        if len(peaks_i) == 0:
            break
            
        # Periodogram before pre-whitening
        ax1.cla()
        ax1.plot(freqs_i, amps_i)
        ax1.scatter(freqs_i[peaks_i], amps_i[peaks_i], c='r', s=10, label='Frequecies to be extracted')
        ax1.set_title("Before")
        ax1.set_xlabel("Frequency (1/day)")
        ax1.set_ylabel("Amplitude (ppt)")
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        double_digit = "0" if n<9 else ""
        plt.savefig(f'pw/{name}/pg_{double_digit}{n+1}', bbox_inches='tight')

        ## Append the peaks to the list
        # peaks = np.append(peaks, peaks_i)
        peaks += peaks_i


        ## Fitting the sinusoids and subtracting them from the original signal
        for freq, amp in zip(freqs_i[peaks_i], amps_i[peaks_i]):
            omega = 2 * np.pi * freq
            p0 = [amp, omega, 0, 0]
            params, pcov = curve_fit(sinusoidal_model, time, flux_i, p0=p0)
            # ## Negative amp corrections. Flip sign, add pi to phase
            # if params[0] < 0:
            #     params[0] *= -1.
            #     params[2] += np.pi
            
            peak_freqs.append(params[1]/(2*np.pi))
            peak_amps.append(params[0])
            flux_i -= sinusoidal_model(time, *params)

        # Periodogram after pre-whitening
        ax2.cla()
        freqs_i, amps_i = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
        amps_i *= 1000 # convert to ppt
        ax2.plot(freqs_i, amps_i)
        ax2.set_title("After")
        ax2.set_xlabel("Frequency (1/day)")
        ax2.set_ylabel("Amplitude (ppt)")
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
        double_digit = "0" if n<9 else ""
        plt.savefig(f'pw/{name}/pg_{double_digit}{n+1}', bbox_inches='tight')
    plt.close()

    freq_amp = pd.DataFrame({'freq': peak_freqs, 'amp': peak_amps})

    ## Sorting the peaks by amplitude
    freq_amp = freq_amp.sort_values(by='freq', ascending=False)
    if remove_harmonics:
        # # # Harmonic ratio checking
        tolerance = 0.001 
        harmonics_idx = []
        for i in range(len(freq_amp)):
            for j in range(i+1, len(freq_amp)):
                if is_harmonic(freq_amp.iloc[i]['freq'], freq_amp.iloc[j]['freq'], tolerance=tolerance):
                    harmonics_idx.append(j)
        freq_amp = freq_amp.drop(index=harmonics_idx)
    
    # Final periodogram after pre-whitening
    freqs, amps = amp_spectrum(t=time, y=flux/np.median(flux), fmin=0, fmax=90, nyq_mult=1, oversample_factor=5)
    amps *= 1000 # convert to ppt
    plt.figure()
    plt.scatter(peak_freqs, peak_amps, s=10, color='red')
    plt.plot(freqs, amps)
    plt.title("Lomb-Scargle Periodogram with peaks")
    plt.xlabel("Frequency (1/day)")
    plt.ylabel("Amplitude (ppt)")
    plt.savefig(f'pw/{name}/pg_final', bbox_inches='tight')

    # Save the frequencies and amplitudes
    freq_amp.to_csv(f'pw/{star}_frequencies.csv', index=False)
    print('Done!')
    return peaks, freq_amp.freq.values, freq_amp.amp.values

if __name__ == "__main__":
    # star = 'TIC171591531'
    star = 'TIC17372709'
    # star = 'HD20203'
    # star = 'HD47129'
    # star = 'V647Tau'

    # lk_search = lk.search_lightcurve(star, mission="TESS", cadence=120)
    # lc = lk_search[0].download().remove_nans().remove_outliers()

    lc_collection = lk.search_lightcurve(star, mission="TESS", cadence=120, author="SPOC").download_all()
    lc = lc_collection[0].normalize() # defaults to pdcsap_flux now.
    for l in lc_collection[1:]:
        lc = lc.append(l.normalize())
    lc = lc.remove_nans().remove_outliers()

    # Extract time and flux from the light curve
    time = lc.time.value
    flux = lc.flux.value

    # Pre-whiten the light curve
    peaks, peak_freqs, peak_amps = prewhitener(time, flux, f_sigma=5, remove_harmonics=True, max_iterations=20, name=star)

