import copy
import os, shutil
import lightkurve as lk
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import lightkurve as lk
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
import inspect

class PreWhitener:
    '''
    The main class for conduction pre-whitening analysis

    Attributes
    ----------
    name : str
        Name of the star. If lightkurve searchable (e.g. TIC, HD, KIC), will download the light curve.
    lc : lightkurve.LightCurve or pandas.DataFrame or tuple
        If lightkurve.LightCurve, will use the time and flux attributes
        If pandas.DataFrame, will use the time and flux columns
        If tuple, will use the first and second elements as time and flux
    max_iterations : int
        Maximum number of iterations to perform
    snr_threshold : float
        Signal-to-noise threshold for stopping iterations
    flag_harmonics : bool
        Flag harmonics of detected frequencies
    harmonic_tolerance : float
        Tolerance for flagging harmonics
    frequency_resolution : float
        Frequency resolution of the periodogram
    fbounds : tuple
        (fmin, fmax) frequency bounds
    nyq_mult : float
        Multiple of the Nyquist frequency to use as the maximum frequency
    oversample_factor : float
        Oversample factor for the frequency grid
    mode : str
        Mode of the periodogram ('amplitude' or 'power')
    data_iter : array_like
        Copy of the original data for iterative pre-whitening
    pg_og : lightkurve.periodogram.Periodogram
        Original periodogram object
    pg_iter : lightkurve.periodogram.Periodogram
        Copy of the original periodogram object for iterative pre-whitening
    iteration : int
        Current iteration number
    stop_iteration : bool
        Flag to stop iterations
    peak_freqs : list
        List of detected peak frequencies
    peak_amps : list
        List of amplitudes of detected peak frequencies
    f_container : array_like
        Significant peak frequencies and amplitudes in a pandas.DataFrame
    '''

    def __init__(self, name=None, lc=None, max_iterations=100, snr_threshold=5,
                flag_harmonics=True, harmonic_tolerance=0.001, frequency_resolution=4/27, 
                fbounds=None, nyq_mult=1, oversample_factor=5, mode='amplitude'):
        '''
        Constructor for PreWhitener object

        Parameters
        ----------
        name : str
            Name of the star. If lightkurve searchable (e.g. TIC, HD, KIC), will download the light curve.
        lc : lightkurve.LightCurve or pandas.DataFrame or tuple
            If lightkurve.LightCurve, will use the time and flux attributes
            If pandas.DataFrame, will use the time and flux columns
            If tuple, will use the first and second elements as time and flux
        max_iterations : int
            Maximum number of iterations to perform
        snr_threshold : float
            Signal-to-noise threshold for stopping iterations
        flag_harmonics : bool
            Flag harmonics of detected frequencies
        harmonic_tolerance : float
            Tolerance for flagging harmonics
        frequency_resolution : float
            Frequency resolution of the periodogram
        fbounds : tuple
            (fmin, fmax) frequency bounds
        nyq_mult : float
            Multiple of the Nyquist frequency to use as the maximum frequency
        oversample_factor : float
            Oversample factor for the frequency grid
        mode : str
            Mode of the periodogram ('amplitude' or 'power')
        '''
        self.name = name
        if lc is None:
            if name is not None:
                if not self.get_lightcurve():
                    raise ValueError(f'No lightkurve data found for {self.name}.\n\
                                     Provide a lightkurve searchable ID as `name` (e.g. TIC, HD, KIC) or provide a lightkurve.LightCurve or pandas.DataFrame or tuple as `lc`')
            else: 
                raise ValueError('Provide a lightkurve searchable ID as `name` (e.g. TIC, HD, KIC) or provide a lightkurve.LightCurve or pandas.DataFrame or tuple as `lc`')
        else:
            # self.fbounds = (0, 72) if fbounds is None else fbounds
            if isinstance(lc, lk.LightCurve):
                self.t, self.data = lc.time.value, lc.flux.value
            elif isinstance(lc, pd.DataFrame):
                self.t, self.data = lc['time'].values, lc['flux'].values
            elif isinstance(lc, tuple):
                self.t, self.data = lc[0], lc[1]
            else:
                raise ValueError('lc must be lightkurve.LightCurve or pandas.DataFrame or tuple\n\
                                Or provide lightkurve searchable ID as name (e.g. TIC, HD, KIC)')
        
        self.lc = lk.LightCurve(time=self.t, flux=self.data)
        self.fbounds = fbounds if fbounds is not None else (0, 72 if self.nyquist_frequency() < 200 else 90)
        
        self.data_iter = copy.deepcopy(self.data)
        self.max_iterations = max_iterations
        self.snr_threshold = snr_threshold
        self.flag_harmonics = flag_harmonics
        self.harmonic_tolerance = harmonic_tolerance
        self.frequency_resolution = frequency_resolution
        self.fmin, self.fmax = self.fbounds if self.fbounds is not None else (self.fmin, self.fmax)
        self.nyq_mult = nyq_mult
        self.oversample_factor = oversample_factor
        self.mode = mode

        self.pg_og = Periodogram(self.t, self.data, fbounds=fbounds, nyq_mult=nyq_mult, oversample_factor=oversample_factor, mode=mode)
        self.pg_iter = copy.deepcopy(self.pg_og)

        self.iteration = 0
        self.stop_iteration = False
        self.peak_freqs = []
        self.peak_amps = []
        self.f_container = None

        if not os.path.exists(f'pw/{self.name}'):
            os.makedirs(f'pw/{self.name}')
        else:
            shutil.rmtree(f'pw/{self.name}')
            os.makedirs(f'pw/{self.name}')

    def get_lightcurve(self):
        '''
        Get lightkurve data for the star
        '''
        print(f'Getting lightkurve data for {self.name}')
        lc_collection = lk.search_lightcurve(self.name, mission="TESS", cadence=120, author="SPOC").download_all()
        # self.fbounds = (0, 90) if self.fbounds is None else self.fbounds
        if lc_collection is None:
            print (f"No 2-min LK for {self.name}, try FFI data...")
            lc_collection = lk.search_lightcurve(self.name, mission="TESS", cadence=600, author="TESS-SPOC").download_all()
            # self.fbounds = (0, 72) if self.fbounds is None else self.fbounds
        if lc_collection is None:
            print (f"No FFI LK for {self.name}, passing...")
            return False
        else:
            lc = lc_collection[0].normalize() # defaults to pdcsap_flux now.
            for l in lc_collection[1:]:
                lc = lc.append(l.normalize())
            lc = lc.remove_nans().remove_outliers()

            # Extract time and flux from the light curve
            self.t, self.data = lc.time.value, lc.flux.value
            return True
        
    def nyquist_frequency(self):
        '''
        Calculate the Nyquist frequency
        '''
        return 1/(2*np.median(np.diff(self.t)))


    def noise_level(self):
        '''
        Calculate the noise level of the light curve
        '''
        return np.median(self.pg_og.amps)*self.snr_threshold if self.mode == 'amplitude' else np.median(self.pg_og.powers)*self.snr_threshold

    def iterate(self):
        '''
        Perform a single iteration of pre-whitening

        Returns
        -------
        freqs : array_like
            Frequency grid
        amps : array_like
            Amplitude spectrum
        powers : array_like
            Power spectrum
        '''
        if self.iteration == 0:
            self.pg_iter = copy.deepcopy(self.pg_og)
        self.pg_iter.amplitude_power_spectrum(self.t, self.data_iter)
        freqs_i = self.pg_iter.freqs
        if self.mode == 'amplitude':
            y_i = self.pg_iter.amps
        elif self.mode == 'power':
            y_i = self.pg_iter.powers

        if self.iteration < self.max_iterations:
            y_max = np.max(y_i)
            freq = freqs_i[np.argmax(y_i)]

            ### SNR stopping condition ###
            if y_max < self.noise_level():
                print('SNR threshold reached')
                self.stop_iteration = True
                return
        
            omega = 2 * np.pi * freq
            p0 = [y_max, omega, 0.5, 0.5]

            params, pcov = curve_fit(self.sinusoidal_model, self.t, self.data_iter, p0=p0)
            ## Negative amp corrections. Flip sign, add pi to phase
            if params[0] < 0:
                params[0] *= -1
                params[2] += np.pi
            
            self.peak_freqs.append(params[1]/(2*np.pi))
            # self.peak_amps.append(params[0])
            self.peak_amps.append(y_max)
            self.data_iter -= self.sinusoidal_model(self.t, *params)
            self.iteration += 1

    def auto(self, make_plot=True, save=True):
        '''
        Auto iterator for pre-whitening
        '''
        for i in tqdm(range(self.max_iterations), desc='Pre-whitening'):
            self.iterate()
            if self.stop_iteration:
                break
        
        self.post_pw(make_plot=make_plot, save=save)
        print(f'Pre-whitening complete after {self.iteration} iterations')


    def post_pw(self, make_plot=True, save=True):
        '''
        Post pre-whitening analysis
        '''
        self.f_container = pd.DataFrame({'freq': self.peak_freqs, 'amp': self.peak_amps}).sort_values(by='freq')

        ## Remove frequencies with amplitude less than the local SNR.
        self.f_container = self.remove_based_on_local_snr(self.f_container, resolution=3)

        ## Remove overlapping or very nearby peaks, keep the highest amplitude one
        self.f_container = self.remove_overlapping_freqs(self.f_container, nearby_tolerance=self.frequency_resolution)
                
        if self.flag_harmonics:
            self.f_container = self.harmonics_check(self.f_container, harmonic_tolerance=self.harmonic_tolerance)
        
        if save:
            self.f_container.to_csv(f'pw/{self.name}/frequencies.csv', index=False)

        if make_plot:
            self.post_pw_plot(save=save)

    def post_pw_plot(self, ax=None, save=True, **kwargs):
        '''
        Post pre-whitening plot

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes
        save : bool
            If True, save the plot
        **kwargs : dict
            Keyword arguments for matplotlib.pyplot.plot and matplotlib.pyplot.scatter bunched together

        Returns
        -------
        ax : matplotlib.axes._axes.Axes
        '''
        ax = ax if ax is not None else plt.gca()

        ## Separate scatter and plot kwargs ##
        scatter_args = list(inspect.signature(plt.scatter).parameters)
        scatter_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in scatter_args}
        plot_args = list(inspect.signature(plt.plot).parameters)
        plot_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in plot_args}


        if isinstance(self.f_container, pd.DataFrame):
            if self.mode == 'amplitude':
                ax.plot(self.pg_og.freqs, self.pg_og.amps*1000, **plot_dict)
                ax.scatter(self.f_container['freq'], self.f_container['amp']*1000, marker='x', color='maroon', s=10, linewidths=1, zorder=2, **scatter_dict)
                ax.set_ylabel("Amplitude (ppt)")
            if self.mode == 'power':
                ax.plot(self.pg_og.freqs, self.pg_og.powers, **plot_dict)
                ax.scatter(self.f_container['freq'], self.f_container['amp'], marker='x', color='maroon', s=10, linewidths=1, zorder=2, **scatter_dict)
                ax.set_ylabel("Power (ppt)")
            ax.set_xlabel("Frequency (1/day)")
            ax.set_xlim(self.fmin, self.fmax)
            if save:
                plt.savefig(f'pw/{self.name}/prewhitening.png', dpi=300)
            return ax
        else:
            raise ValueError('No frequencies found. Try running post_pw() first')

    # Sinusoidal function to fit the peaks
    def sinusoidal_model(self, t, A, omega, phi, C):
        '''
        Sinusoidal model
        '''
        return A * np.sin(omega * t + phi) + C

    def harmonics_check(self, df, harmonic_tolerance=0.01):
        '''
        Check for and flag harmonics of detected frequencies

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with columns 'freq' and 'amp'
        harmonic_tolerance : float
            Tolerance for flagging harmonics
        
        Returns
        -------
        df : pandas.DataFrame with a new column 'label'
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

    def remove_overlapping_freqs(self, df, nearby_tolerance=0.01):
        '''
        Function to remove overlapping or very nearby peaks, keeps the highest amplitude one

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with columns 'freq' and 'amp'
        harmonic_tolerance : float
            Tolerance for flagging harmonics
        
        Returns
        -------
        df : pandas.DataFrame
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

    def remove_based_on_local_snr(self, df, resolution=3):
        '''
        Remove frequencies based on the local SNR.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with columns 'freq' and 'amp'
        snr_threshold : float
            SNR threshold

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with columns 'freq' and 'amp' with frequencies below the local SNR threshold removed
        '''
        df = df.sort_values(by=['freq', 'amp'], ascending=False)
        df = df.reset_index(drop=True)
        to_drop = []
        for i in range(len(df)-1):
            freq = df.iloc[i]['freq']
            amp = df.iloc[i]['amp']
            local_noise = np.median(df[(df['freq'] > freq - resolution) & (df['freq'] < freq + resolution)]['amp'])
            if amp < local_noise:
                to_drop.append(i)
        return df.drop(index=to_drop)
        

        


class Periodogram:
    '''
    Periodogram object for storing and operating on Lomb-Scargle periodograms

    Attributes:
        freqs : array_like
            Frequency grid
        amps : array_like or None
            Amplitude spectrum if mode == 'amplitude'
        powers : array_like or None
            Power spectrum if mode == 'power'

    '''
    def __init__(self, t, data, fbounds=None, nyq_mult=1., oversample_factor=5., mode='amplitude'):
        '''
        Constructor for Periodogram object
        '''
        self.freqs = None
        self.amps = None
        self.powers = None
        self.fbounds = fbounds
        self.nyq_mult = nyq_mult
        self.oversample_factor = oversample_factor
        self.mode = mode
        self.amplitude_power_spectrum(t, data)

    def amplitude_power_spectrum(self, t, data):
        '''
        Calculate the amplitude spectrum of the time series y(t)
        
        Parameters
        ----------
        t : array_like
            The time series
        y : array_like
            Flux or magnitude time series
        fbounds : tuple
            (fmin, fmax) frequency bounds
        nyq_mult : float
            Multiple of the Nyquist frequency to use as the maximum frequency
        oversample_factor : float
            Oversample factor for the frequency grid
        mode : str
            'amplitude' or 'power'

        Returns
        -------
        freqs : array_like
            Frequency grid
        amps : array_like
            Amplitude spectrum
        '''
        tmax = t.max()
        tmin = t.min()
        fmin, fmax = self.fbounds if self.fbounds is not None else (None, None)
        df = 1.0 / (tmax - tmin)
        
        if fmin is None:
            fmin = df
        if fmax is None:
            fmax = (0.5 / np.median(np.diff(t)))*self.nyq_mult

        freqs = np.arange(fmin, fmax, df / self.oversample_factor)
        
        model = LombScargle(t, data)
        sc = model.power(freqs, method="fast", normalization="psd")

        if self.mode == 'amplitude':
            amps = np.sqrt(4./len(t)) * np.sqrt(sc)
            self.freqs = freqs
            self.amps = amps
        elif self.mode == 'power':
            powers = np.sqrt(4./len(t))**2 * sc
            self.freqs = freqs
            self.powers = powers

    def plot(self, ax=None, mode='amplitude', **kwargs):
        '''
        Plot the periodogram
        
        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes to plot on
        mode : str
            'amplitude' or 'power'
        show_peaks : bool
            If True, plot the identified peaks
        '''
        if ax is None:
            ax = plt.gca()

        if mode == 'amplitude':
            ax.plot(self.freqs, self.amps, **kwargs) 
        elif mode == 'power':
            ax.plot(self.freqs, self.powers, **kwargs)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Amplitude' if mode == 'amplitude' else 'Power')
        return ax
    
