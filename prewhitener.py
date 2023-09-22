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

def harmonics_check(df, harmonic_tolerance=0.01):
    df = df.sort_values(by='freq', ascending=False)
    df = df.reset_index(drop=True)
    harmonics_ids = []
    for i in range(len(df)-1):
        for j in range(i+1, len(df)-1):
            ratio = df.iloc[i]['freq']/df.iloc[j]['freq']
            closest_integer = round(ratio)
            if abs(ratio-closest_integer) < harmonic_tolerance:
                if df.iloc[i]['amp'] > df.iloc[j]['amp']:
                    harmonics_ids.append(j)
                else:
                    harmonics_ids.append(i)
    df['harmonic'] = [1 if i in harmonics_ids else 0 for i in range(len(df))]
    return df
    # return df.drop(index=harmonics_ids)

def remove_overlapping_freqs(df, nearby_tolerance=0.01):
    df = df.sort_values(by=['freq', 'amp'], ascending=False)
    df = df.reset_index(drop=True)
    to_drop = []
    for i in range(len(df)-1):
        if df.iloc[i]['freq'] - df.iloc[i+1]['freq'] < 0.2:
            if df.iloc[i]['amp'] < df.iloc[i+1]['amp']:
                to_drop.append(i)
            else:
                to_drop.append(i+1)
    return df.drop(index=to_drop)


# Sinusoidal function to fit the peaks
def sinusoidal_model(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C

def prewhitener_single(time, flux, max_iterations=100, snr_threshold=5,
                flag_harmonics=True, harmonic_tolerance=0.001,  
                fmin=5, fmax=72, nyq_mult=1, oversample_factor=5, name='star'):
    if not os.path.exists(f'pw/{name}'):
        os.makedirs(f'pw/{name}')
    else:
        shutil.rmtree(f'pw/{name}')
        os.makedirs(f'pw/{name}')

    ## Normalize the flux
    flux_i = copy.deepcopy(flux)

    peak_freqs = []
    peak_amps = []

    ## Initial amplitude spectrum
    freqs_i, amps_i = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
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
        peak_amps.append(params[0]*1000)
        # peak_amps.append(amp)
        flux_i -= sinusoidal_model(time, *params)

        ### New amplitude spectrum ###
        freqs_i, amps_i = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
        amps_i *= 1000 # convert to ppt

        ### SNR stopping condition ###
        noise = median_filter(amps_i, size=int(len(amps_i)*(freqs_i[1]-freqs_i[0])))
        snr = np.max(amps_i) / np.median(noise)
        if snr < snr_threshold:
            # print('SNR threshold reached')
            break

    freq_amp = pd.DataFrame({'freq': peak_freqs, 'amp': peak_amps})

    if flag_harmonics:
        freq_amp = harmonics_check(freq_amp, harmonic_tolerance=harmonic_tolerance)

    ## Remove overlapping or very nearby peaks, keep the highest amplitude one
    freq_amp = remove_overlapping_freqs(freq_amp, nearby_tolerance=0.01)

    # Final periodogram after pre-whitening
    freqs, amps = amp_spectrum(t=time, y=flux, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
    amps *= 1000 # convert to ppt
    plt.figure(figsize=(20, 5))
    plt.scatter(freq_amp.freq.values, freq_amp.amp.values, marker='x', s=10, color='red', zorder=2)
    plt.plot(freqs, amps, zorder=1)
    plt.title("Lomb-Scargle Periodogram with peaks")
    plt.xlabel("Frequency (1/day)")
    plt.ylabel("Amplitude (ppt)")
    plt.savefig(f'pw/{name}/pg_final', bbox_inches='tight')

    # Save the frequencies and amplitudes
    freq_amp.to_csv(f'pw/{name}/frequencies.csv', index=False)
    print(f'Done {name}')
    return freq_amp.freq.values, freq_amp.amp.values


def prewhitener_multi(time, flux, max_iterations=100, snr_threshold=5, f_sigma=3,
                flag_harmonics=True, harmonic_tolerance=0.001,  
                fmin=5, fmax=72, nyq_mult=1, oversample_factor=5, name='star'):
    if not os.path.exists(f'pw/{name}'):
        os.makedirs(f'pw/{name}')
    else:
        shutil.rmtree(f'pw/{name}')
        os.makedirs(f'pw/{name}')

    ## Normalize the flux
    flux_i = copy.deepcopy(flux)

    peak_freqs = []
    peak_amps = []
    peaks = []

    ## Initial amplitude spectrum
    freqs_i, amps_i = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
    for n in range(max_iterations):
        ## Find all peaks to calculate the median prominence and width
        peaks_tmp = find_peaks(amps_i)[0]

        prominence_data = peak_prominences(amps_i, peaks=peaks_tmp)
        prominence = np.median(prominence_data[0])
        peaks_widths_i = peak_widths(amps_i, peaks=peaks_tmp, rel_height=0.5, prominence_data=prominence_data)[0]
        width = np.median(peaks_widths_i)  ## median fwhm 
        # distance = width/(np.median(np.diff(freqs_i)))
        # distance = len(amps_i)/width
        fs = 720
        distance = fs / (1/27)

        ## Find all peaks that fit the above criteria
        peaks_i = find_peaks(amps_i, height=np.median(amps_i)+f_sigma*np.std(amps_i), 
                             width=width, 
                             prominence=prominence,
                             distance=distance)[0].tolist()
        
        ## If no peaks are found, break the loop
        if len(peaks_i) == 0:
            print('No more peaks found')
            break

        ## Add the peaks to the list
        peaks += peaks_i
        ## Fitting the sinusoids and subtracting them from the original signal
        for freq, amp in zip(freqs_i[peaks_i], amps_i[peaks_i]):
            omega = 2 * np.pi * freq
            p0 = [amp, omega, 0.5, 0.5]
            params, pcov = curve_fit(sinusoidal_model, time, flux_i, p0=p0)
            ## Negative amp corrections. Flip sign, add pi to phase
            if params[0] < 0:
                params[0] *= -1.
                params[2] += np.pi

            flux_i -= sinusoidal_model(time, *params)
            ### New amplitude spectrum ###
            freqs_i, amps_i = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
            amps_i *= 1000 # convert to ppt

            ### SNR stopping condition ###
            noise = median_filter(amps_i, size=int(len(amps_i)*(freqs_i[1]-freqs_i[0])))
            snr = np.max(amps_i) / np.median(noise)
            if snr < snr_threshold:
                # print('SNR threshold reached')
                break_now = True
                break
            else:
                peak_freqs.append(params[1]/(2*np.pi))
                peak_amps.append(params[0]*1000)
        else:
            continue
        break

    freq_amp = pd.DataFrame({'freq': peak_freqs, 'amp': peak_amps})

    if flag_harmonics:
        freq_amp = harmonics_check(freq_amp, harmonic_tolerance=harmonic_tolerance)

    ## Remove overlapping or very nearby peaks, keep the highest amplitude one
    freq_amp = remove_overlapping_freqs(freq_amp, nearby_tolerance=0.01)

    # Final periodogram after pre-whitening
    freqs, amps = amp_spectrum(t=time, y=flux, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
    amps *= 1000 # convert to ppt
    plt.figure(figsize=(20, 5))
    plt.scatter(freq_amp.freq.values, freq_amp.amp.values, marker='x', s=10, color='red', zorder=2)
    plt.plot(freqs, amps, zorder=1)
    plt.title("Lomb-Scargle Periodogram with peaks")
    plt.xlabel("Frequency (1/day)")
    plt.ylabel("Amplitude (ppt)")
    plt.savefig(f'pw/{name}/pg_final', bbox_inches='tight')
    plt.close()

    # Save the frequencies and amplitudes
    freq_amp.to_csv(f'pw/{name}/frequencies.csv', index=False)
    print(f'Done {name}')
    return freq_amp.freq.values, freq_amp.amp.values

if __name__ == "__main__":
    # stars = [189127221,193893464,469421586,158374262,237162793,20534584,235612106,522220718,15997013,120893795]
    stars = pd.read_csv('cepher_pulsating_tics.csv')['TIC'].values
    # stars = [17372709]
    for star in stars:
        lc_collection = lk.search_lightcurve("TIC"+str(star), mission="TESS", cadence=120, author="SPOC").download_all()
        f_max = 90
        if lc_collection is None:
            print (f"No 2-min LK for TIC{star}, try FFI data...")
            lc_collection = lk.search_lightcurve("TIC"+str(star), mission="TESS", cadence=600, author="TESS-SPOC").download_all()
            f_max = 72
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

            # Pre-whiten the light curve
            peak_freqs, peak_amps = prewhitener_single(time, flux, fmax=f_max,
                                               snr_threshold=5,
                                               flag_harmonics=True, name='TIC'+str(star))

    


