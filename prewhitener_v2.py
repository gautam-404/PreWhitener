import copy
import os, shutil
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from lightkurve import periodogram
from astropy import units
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.ndimage import median_filter


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


def is_harmonic(f1, f2, tolerance):
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

def prewhitener(time, flux, max_iterations=100, snr_threshold=5, nearby_peaks_tolerance=0.1,
                remove_harmonics=True, harmonic_tolerance=0.001,  
                fmin=5, fmax=72, nyq_mult=1, oversample_factor=5, name='star'):
    if not os.path.exists(f'pw/{name}'):
        os.makedirs(f'pw/{name}')
    else:
        shutil.rmtree(f'pw/{name}')
        os.makedirs(f'pw/{name}')

    ## Normalize the flux
    flux = flux/np.median(flux)
    flux_i = copy.deepcopy(flux)

    peak_freqs = []
    peak_amps = []
    peaks = []

    ## Initial amplitude spectrum
    freqs_i, amps_i = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
    amps_i *= 1000 # convert to ppt
    for n in range(max_iterations):
        ## Find all peaks
        peaks_i = find_peaks(amps_i)[0]
        
        ## If no peaks are found, break the loop
        if len(peaks_i) == 0:
            print('No more peaks found')
            break

        ## Fit and remove the highest peak
        amp = np.max(amps_i)
        freq = freqs_i[np.argmax(amps_i)]
        omega = 2 * np.pi * freq
        p0 = [amp, omega, 0.5, 0.5]
        params, pcov = curve_fit(sinusoidal_model, time, flux_i, p0=p0)
        ## Negative amp corrections. Flip sign, add pi to phase
        if params[0] < 0:
            params[0] *= -1.
            params[2] += np.pi
        peak_freqs.append(params[1]/(2*np.pi))
        # peak_amps.append(params[0]*1000)
        peak_amps.append(amp)
        flux_i -= sinusoidal_model(time, *params)


        ### New amplitude spectrum ###
        freqs_i, amps_i = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
        amps_i *= 1000 # convert to ppt

        ### SNR stopping condition ###
        noise = median_filter(amps_i, size=int(len(amps_i)*(freqs_i[1]-freqs_i[0])/(freqs_i[-1]-freqs_i[0])))
        snr = np.max(amps_i) / np.median(noise)
        if snr < snr_threshold:
            print('SNR threshold reached')
            break

    ## Creating a dataframe with the peaks
    freq_amp = pd.DataFrame({'freq': peak_freqs, 'amp': peak_amps})

    ## Sorting the peaks by amplitude
    freq_amp = freq_amp.sort_values(by='freq', ascending=False)
    if remove_harmonics:
        # # # Harmonic ratio checking
        harmonics_idx = []
        for i in range(len(freq_amp)):
            for j in range(i+1, len(freq_amp)):
                if is_harmonic(freq_amp.iloc[i]['freq'], freq_amp.iloc[j]['freq'], tolerance=harmonic_tolerance):
                    harmonics_idx.append(j)
        freq_amp = freq_amp.drop(index=harmonics_idx)

    ## Remove overlapping or very nearby peaks, keeping the highest amplitude one
    freq_amp = freq_amp.sort_values(by=['freq', 'amp'], ascending=False)
    freq_amp = freq_amp.reset_index(drop=True)
    to_drop = []
    for i in range(len(freq_amp)-1):
        if freq_amp.iloc[i]['freq'] - freq_amp.iloc[i+1]['freq'] < nearby_peaks_tolerance:
            if freq_amp.iloc[i]['amp'] < freq_amp.iloc[i+1]['amp']:
                to_drop.append(i)
            else:
                to_drop.append(i+1)
    freq_amp = freq_amp.drop(index=to_drop)
    
    # Final periodogram after pre-whitening
    freqs, amps = amp_spectrum(t=time, y=flux, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
    amps *= 1000 # convert to ppt
    plt.figure(figsize=(20, 5))
    plt.scatter(freq_amp.freq.values, freq_amp.amp.values, 'x', s=10, color='red', zorder=2)
    plt.plot(freqs, amps, zorder=1)
    plt.title("Lomb-Scargle Periodogram with peaks")
    plt.xlabel("Frequency (1/day)")
    plt.ylabel("Amplitude (ppt)")
    plt.savefig(f'pw/{name}/pg_final', bbox_inches='tight')

    # Save the frequencies and amplitudes
    freq_amp.to_csv(f'pw/{name}/frequencies.csv', index=False)
    print(f'Done! {name}')
    return peaks, freq_amp.freq.values, freq_amp.amp.values

if __name__ == "__main__":
    # stars = [189127221,193893464,469421586,158374262,237162793,20534584,235612106,522220718,15997013,120893795]
    stars = [17372709]
    for star in stars:
        lc_collection = lk.search_lightcurve("TIC"+str(star), mission="TESS", cadence=120, author="SPOC").download_all()
        if lc_collection is None:
            print (f"No 2-min LK for TIC{star}, try FFI data...")
            lc_collection = lk.search_lightcurve("TIC"+str(star), mission="TESS", cadence=600, author="TESS-SPOC").download_all()
        if lc_collection is None:
            print (f"No FFI LK for TIC{star}, passing...")
            pass
        else:
            lc = lc_collection[0].normalize() # defaults to pdcsap_flux now.
            for l in lc_collection[1:]:
                lc = lc.append(l.normalize())
            lc = lc.remove_nans().remove_outliers()

            # Extract time and flux from the light curve
            time, flux = lc.time.value, lc.flux.value

            # print(len(time))
            # Pre-whiten the light curve
            peaks, peak_freqs, peak_amps = prewhitener(time, flux, 
                                                snr_threshold=5,
                                                nearby_peaks_tolerance=0.1,
                                                remove_harmonics=True, name='TIC'+str(star))

    


