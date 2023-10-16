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
    '''
    Calculate the amplitude spectrum of the time series y(t)
    
    Parameters
    ----------
    t : array_like
        The time series
    y : array_like
        Flux or magnitude time series
    fmin : float
        Minimum frequency to calculate the amplitude spectrum
    fmax : float
        Maximum frequency to calculate the amplitude spectrum
    nyq_mult : float
        Multiple of the Nyquist frequency to use as the maximum frequency
    oversample_factor : float
        Oversample factor for the frequency grid

    Returns
    -------
    freq : array_like
        Frequency grid
    amp : array_like
        Amplitude spectrum

    Written by Simon J. Murphy
    '''
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
    '''
    Harmonics check for the frequencies in the amplitude spectrum.
    If two frequencies are within the harmonic tolerance, the lower
    frequency is kept and the higher one is labelled as a harmonic.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'freq' and 'amp'
    harmonic_tolerance : float
        Harmonic tolerance in frequency units
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns 'freq', 'amp' and 'label'
    '''
    df = df.sort_values(by='freq', ascending=True)
    df = df.reset_index(drop=True)
    harmonic_idx = []
    for i in range(len(df)-1):
        for j in range(i+1, len(df)):
            ratio = df.iloc[j]['freq']/df.iloc[i]['freq']
            closest_integer = round(ratio)
            if abs(ratio-closest_integer) < harmonic_tolerance and closest_integer > 1:
                df.loc[j, 'label'] = f'H{closest_integer}F{i}'
                harmonic_idx.append(j)
    base_idx = ~df.index.isin(harmonic_idx)
    df.loc[base_idx, 'label'] = [f'F{i}' for i in range(0, sum(base_idx))]
    return df

def remove_overlapping_freqs(df, nearby_tolerance=0.01):
    '''
    Remove overlapping or very nearby frequencies from the amplitude spectrum.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'freq' and 'amp'
    nearby_tolerance : float
        Tolerance in frequency units

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns 'freq' and 'amp' with overlapping frequencies removed
    '''
    df = df.sort_values(by=['freq', 'amp'], ascending=False)
    df = df.reset_index(drop=True)
    to_drop = []
    for i in range(len(df)-1):
        if df.iloc[i]['freq'] - df.iloc[i+1]['freq'] < nearby_tolerance:
            if df.iloc[i]['amp'] < df.iloc[i+1]['amp']:
                to_drop.append(i)
            else:
                to_drop.append(i+1)
    return df.drop(index=to_drop)

def sinusoidal_model(t, A, omega, phi, C):
    '''
    Sinusoidal function to fit the peaks

    Parameters
    ----------
    t : array_like
        Time series
    A : float
        Amplitude
    omega : float
        Angular frequency
    phi : float
        Phase
    C : float
        Constant

    Returns
    -------
    y : array_like
        Sinusoidal function
    '''
    return A * np.sin(omega * t + phi) + C

def prewhitener_single(time, flux, max_iterations=100, snr_threshold=5,
                flag_harmonics=True, harmonic_tolerance=0.01, nearby_tolerance=0.001,
                fmin=5, fmax=72, nyq_mult=1, oversample_factor=5, name='star'):
    '''
    Pre-whitening the light curve by fitting sinusoids to the peaks in the amplitude spectrum.
    The highest peak is fit and removed, and the process is repeated until the SNR threshold is reached.

    Parameters
    ----------
    time : array_like
        Time series
    flux : array_like
        Flux or magnitude time series
    max_iterations : int
        Maximum number of iterations
    snr_threshold : float
        SNR threshold to stop the pre-whitening
    flag_harmonics : bool
        Flag to check for harmonics
    harmonic_tolerance : float
        Harmonic tolerance in frequency units
    nearby_tolerance : float
        Tolerance for overlapping or very nearby frequencies in frequency units
    fmin : float
        Minimum frequency to calculate the amplitude spectrum
    fmax : float
        Maximum frequency to calculate the amplitude spectrum
    nyq_mult : float
        Multiple of the Nyquist frequency to use as the maximum frequency   
    oversample_factor : float
        Oversample factor for the frequency grid
    name : str
        Name of the star. Used to save the plots and frequencies
    '''
    if not os.path.exists(f'pw/{name}'):
        os.makedirs(f'pw/{name}')
    else:
        shutil.rmtree(f'pw/{name}')
        os.makedirs(f'pw/{name}')

    flux_i = copy.deepcopy(flux)

    peak_freqs = []
    peak_amps = []

    ## Initial amplitude spectrum
    freqs_i, amps_i = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
    amps_i *= 1000 # convert to ppt
    median_window_size = int(len(amps_i)*np.median(np.diff(freqs_i)))
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
        noise = median_filter(amps_i, size=median_window_size)
        snr = np.max(amps_i) / np.median(noise)
        if snr < snr_threshold:
            # print('SNR threshold reached')
            break

    freq_amp = pd.DataFrame({'freq': peak_freqs, 'amp': peak_amps}).sort_values(by='freq')

    ## Remove overlapping or very nearby peaks, keep the highest amplitude one
    freq_amp = remove_overlapping_freqs(freq_amp, nearby_tolerance=nearby_tolerance)
              
    if flag_harmonics:
        freq_amp = harmonics_check(freq_amp, harmonic_tolerance=harmonic_tolerance)

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
                flag_harmonics=True, harmonic_tolerance=0.01, nearby_tolerance=0.01,
                fmin=5, fmax=72, nyq_mult=1, oversample_factor=5, name='star'):
    '''
    Pre-whitening the light curve by fitting sinusoids to the peaks in the amplitude spectrum.
    Multiple peaks are fit and removed, and the process is repeated until the SNR threshold is reached.

    Parameters
    ----------
    time : array_like
        Time series
    flux : array_like
        Flux or magnitude time series
    max_iterations : int
        Maximum number of iterations
    snr_threshold : float
        SNR threshold to stop the pre-whitening
    f_sigma : float
        Number of standard deviations above the mean to use as the minimum height for the peaks
    flag_harmonics : bool
        Flag to check for harmonics
    harmonic_tolerance : float
        Harmonic tolerance in frequency units
    nearby_tolerance : float
        Tolerance for overlapping or very nearby frequencies in frequency units
    fmin : float
        Minimum frequency to calculate the amplitude spectrum
    fmax : float
        Maximum frequency to calculate the amplitude spectrum
    nyq_mult : float
        Multiple of the Nyquist frequency to use as the maximum frequency   
    oversample_factor : float
        Oversample factor for the frequency grid
    name : str
        Name of the star. Used to save the plots and frequencies
    '''
    if not os.path.exists(f'pw/{name}'):
        os.makedirs(f'pw/{name}')
    else:
        shutil.rmtree(f'pw/{name}')
        os.makedirs(f'pw/{name}')

    flux_i = copy.deepcopy(flux)

    peak_freqs = []
    peak_amps = []
    peaks = []

    ## Initial amplitude spectrum
    freqs_i, amps_i = amp_spectrum(t=time, y=flux_i, fmin=fmin, fmax=fmax, nyq_mult=nyq_mult, oversample_factor=oversample_factor)
    amps_i *= 1000 # convert to ppt
    median_window_size = int(len(amps_i)*np.median(np.diff(freqs_i)))
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
            noise = median_filter(amps_i, size=median_window_size)
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

    freq_amp = pd.DataFrame({'freq': peak_freqs, 'amp': peak_amps}).sort_values(by='freq')

    ## Remove overlapping or very nearby peaks, keep the highest amplitude one
    freq_amp = remove_overlapping_freqs(freq_amp, nearby_tolerance=nearby_tolerance)
              
    if flag_harmonics:
        freq_amp = harmonics_check(freq_amp, harmonic_tolerance=harmonic_tolerance)

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

    


