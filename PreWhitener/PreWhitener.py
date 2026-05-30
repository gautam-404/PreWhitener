import copy
import os, shutil
import lightkurve as lk
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

from .Periodogram import Periodogram

class PreWhitener:
    """
    The main class for conducting pre-whitening analysis.
    """
    # Attributes 
    name : str
    """Name of the star. If lightkurve searchable (e.g. TIC, HD, KIC), will download the light curve."""
    lc : (lk.LightCurve or pd.DataFrame or tuple)
    """If lightkurve.LightCurve, will use the time and flux attributes. If pandas.DataFrame, will use the time and flux columns. If tuple, will use the first and second elements as time and flux."""
    max_iterations : int
    """Maximum number of iterations to perform."""
    snr_threshold : float
    """Signal-to-noise threshold for stopping iterations."""
    flag_harmonics : bool
    """Flag harmonics of detected frequencies."""
    harmonic_tolerance : float
    """Tolerance for flagging harmonics."""
    frequency_resolution : float
    """Frequency resolution of the periodogram."""
    fbounds : tuple
    """(fmin, fmax) frequency bounds."""
    nyq_mult : int
    """Multiple of the Nyquist frequency to use as the maximum frequency."""
    oversample_factor : int
    """Oversample factor for the frequency grid."""
    normalization : str
    """Mode of the periodogram ('amplitude' or 'psd')."""
    t : np.ndarray
    """1D array time series."""
    data : np.ndarray
    """1D array flux/magnitude time series."""
    data_iter : np.ndarray
    """1D array opy of the original data for iterative pre-whitening."""
    pg : Periodogram
    """Original periodogram object."""
    pg_iter : Periodogram
    """Copy of the original periodogram object for iterative pre-whitening."""
    iteration : int
    """Current iteration number."""
    stop_iteration : bool
    """Flag to stop iterations."""
    peak_freqs : list
    """List of detected peak frequencies."""
    peak_amps : list
    """List of amplitudes of detected peak frequencies."""
    freq_container : pd.DataFrame
    """Significant peak frequencies and amplitudes in a pandas.DataFrame."""

    def __init__(self, name: str = None, lc: (lk.LightCurve or pd.DataFrame or tuple)=None, max_iterations: int = 100, snr_threshold: float = 5,
                fbounds: tuple = None, nyq_mult: int = 1, oversample_factor: int = 5, normalization: str = 'amplitude', noise_level: float = None,
                noise_level_type: str = 'median_snr', noise_kwargs: dict = None, wdir: str = '.'):
        """
        Constructor for PreWhitener object.

        Parameters
        ----------
        name : str
            Name of the star. If lightkurve searchable (e.g. TIC, HD, KIC), will download the light curve.
        lc : (lightkurve.LightCurve or pandas.DataFrame or tuple)
            If lightkurve.LightCurve, will use the time and flux attributes. If pandas.DataFrame, will use the time and flux columns. If tuple, will use the first and second elements as time and flux.
        max_iterations : int
            Maximum number of iterations to perform.
        snr_threshold : float
            Signal-to-noise threshold for stopping iterations.
        fbounds : tuple
            (fmin, fmax) frequency bounds.
        nyq_mult : int
            Multiple of the Nyquist frequency to use as the maximum frequency.
        oversample_factor : int
            Oversample factor for the frequency grid.
        normalization : str
            Mode of the periodogram ('amplitude' or 'psd').
        noise_level : float, optional
            Explicit global stopping threshold. If provided, this value takes precedence over `noise_level_type`.
        noise_level_type : str, optional, default: 'median_snr'
            Noise estimator used when `noise_level` is not provided.
            Supported values:
            - 'median_snr': global threshold = median(spectrum) * snr_threshold.
            - 'mad_snr': global threshold = median + snr_threshold * (1.4826 * MAD).
            - 'lower_tail_median_snr': use only the lowest-amplitude `tail_frac` of bins, then median * snr_threshold.
              Requires `tail_frac` in `noise_kwargs`.
            - 'quiet_band_median_snr': estimate noise from a quiet frequency band only, then median * snr_threshold.
              Requires `f_noise_min`in `noise_kwargs`. `f_noise_max` is optional.
            - 'local_window_median_snr': dynamic local threshold around each candidate peak using a frequency window.
              To be used when manually iterating, instead of using `auto()`. Requires `window`. `exclude_width` is optional in `noise_kwargs`.
        noise_kwargs : dict, optional
            Additional keyword arguments required by selected `noise_level_type`.
        wdir : str
            Working directory to save results. Default is current directory.
        """
        self.name = name
        self.wdir = os.path.abspath(wdir)
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
                self.lc = lc
            elif isinstance(lc, pd.DataFrame):
                self.lc = lk.LightCurve(time=lc['time'].values, flux=lc['flux'].values)
            elif isinstance(lc, tuple):
                self.lc = lk.LightCurve(time=lc[0], flux=lc[1])
            else:
                raise ValueError('lc must be lightkurve.LightCurve or pandas.DataFrame or tuple\n\
                                Or provide lightkurve searchable ID as name (e.g. TIC, HD, KIC)')
        
        self.nyq_mult = nyq_mult
        if fbounds is not None:
            self.fbounds = fbounds
        else:
            nyquist_limit = self.nyquist_frequency() * self.nyq_mult
            default_fmax = 72 if nyquist_limit < 200 else 90
            self.fbounds = (0, min(default_fmax, nyquist_limit))
        normalization_alias = {'power': 'psd'}
        self.normalization = normalization_alias.get(normalization, normalization)

        self.data_iter = copy.deepcopy(self.data - np.median(self.data))
        self.max_iterations = max_iterations
        self.snr_threshold = snr_threshold
        self.noise_level_type = noise_level_type
        self.noise_kwargs = {} if noise_kwargs is None else noise_kwargs
        self.fmin, self.fmax = self.fbounds if self.fbounds is not None else (self.fmin, self.fmax)
        self.oversample_factor = oversample_factor
        self.normalization = self.normalization if self.normalization in ['amplitude', 'psd'] else 'amplitude'

        self.pg = Periodogram(self.lc.time.value, self.lc.flux, fbounds=self.fbounds, nyq_mult=self.nyq_mult, oversample_factor=self.oversample_factor, normalization=self.normalization, wdir=self.wdir)
        self.pg_iter = copy.deepcopy(self.pg)
        self.noise_level_override = noise_level is not None
        if self.noise_level_override:
            self.noise_level = noise_level
        else:
            freqs_0, y_0 = self.get_spectrum_xy(self.pg)
            self.noise_level = self.compute_noise_level(freqs_0, y_0, noise_level_type=self.noise_level_type,
                                                        snr_threshold=self.snr_threshold, noise_kwargs=self.noise_kwargs)

        self.iteration = 0
        self.stop_iteration = False
        self.peak_freqs = []
        self.peak_amps = []
        self.peak_powers = []
        self.peak_phases = []
        self.freq_container = None
        # Populated by _attach_uncertainties() in post_pw().
        self.noise_sigma_residual = None
        self.noise_amp_mean = None

        self.output_dir = os.path.join(self.wdir, 'pw', str(self.name))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)

    def get_lightcurve(self) -> bool:
        """
        Get lightkurve data for the star
        """
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
            self.lc = lc
            return True
    
    @property
    def t(self):
        return self.lc.time.value

    @t.setter
    def t(self):
        raise ValueError('Cannot set time directly. Use lc instead.')
    
    @property
    def data(self):
        return self.lc.flux.value

    @data.setter
    def data(self):
        raise ValueError('Cannot set data directly. Use lc instead.')

    @property
    def t_ref(self) -> float:
        """
        Reference epoch used when fitting per-iteration sinusoids. Set to the mean
        of the time array so that the fitted phase is decorrelated from frequency
        (standard recipe; see Montgomery & O'Donoghue 1999, ADS 1999DSSN...13...28M).
        Reported phases in the output file are phases at ``t_ref``, and any
        reconstruction of a sinusoid from the saved values must use
        ``A * sin(2*pi*freq*(t - t_ref) + phase)``.
        """
        return float(np.mean(self.t))
   
    def nyquist_frequency(self) -> float:
        """
        Calculate the Nyquist frequency
        """
        return 1/(2*np.median(np.diff(self.t)))

    ## helper
    def get_spectrum_xy(self, pg_obj: Periodogram) -> tuple:
        if self.normalization == 'amplitude':
            return pg_obj.freqs, pg_obj.amps
        if self.normalization == 'psd':
            return pg_obj.freqs, pg_obj.powers
        raise ValueError(f'Unsupported normalization: {self.normalization}')

    ## helper
    def validate_xy(self, freqs: np.ndarray, y: np.ndarray) -> tuple:
        freqs_arr = np.asarray(freqs, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        finite_mask = np.isfinite(freqs_arr) & np.isfinite(y_arr)
        freqs_valid = freqs_arr[finite_mask]
        y_valid = y_arr[finite_mask]
        if len(y_valid) == 0:
            raise ValueError('No finite periodogram values available for noise level estimation.')
        return freqs_valid, y_valid

    ## helper
    def compute_noise_level(self, freqs: np.ndarray, y: np.ndarray, noise_level_type: str, snr_threshold: float, noise_kwargs: dict) -> float:
        freqs_valid, y_valid = self.validate_xy(freqs, y)
        if noise_level_type == 'median_snr':
            return np.median(y_valid) * snr_threshold
        if noise_level_type == 'mad_snr': ## median absolute deviation
            y_med = np.median(y_valid)
            mad = np.median(np.abs(y_valid - y_med))
            sigma = 1.4826 * mad            ### For a normal distribution, MAD ≈ 0.67449 * sigma. So sigma is MAD / 0.67449 ≈ 1.4826 * MAD.
            return y_med + snr_threshold * sigma
        if noise_level_type == 'lower_tail_median_snr':
            tail_frac = noise_kwargs.get('tail_frac', None)
            if tail_frac is None:
                raise ValueError("noise_level_type='lower_tail_median_snr' requires noise_kwargs['tail_frac'].")
            if not (0 < float(tail_frac) <= 1):
                raise ValueError("noise_kwargs['tail_frac'] must be in the range (0, 1].")
            n_tail = max(1, int(np.floor(len(y_valid) * float(tail_frac))))
            y_tail = np.partition(y_valid, n_tail - 1)[:n_tail]
            return np.median(y_tail) * snr_threshold
        if noise_level_type == 'quiet_band_median_snr':
            f_noise_min = noise_kwargs.get('f_noise_min', None)
            if f_noise_min is None:
                raise ValueError("noise_level_type='quiet_band_median_snr' requires noise_kwargs['f_noise_min'].")
            f_noise_max = noise_kwargs.get('f_noise_max', self.fmax)
            band_mask = (freqs_valid >= float(f_noise_min)) & (freqs_valid <= float(f_noise_max))
            if np.sum(band_mask) < 3:
                raise ValueError('Quiet-band mask has fewer than 3 samples; adjust f_noise_min/f_noise_max.')
            return np.median(y_valid[band_mask]) * snr_threshold
        if noise_level_type == 'local_window_median_snr':
            raise ValueError("noise_level_type='local_window_median_snr' must be evaluated with compute_local_noise_level.")
        raise ValueError(f"Unsupported noise_level_type: {noise_level_type}")

    ## helper
    def compute_local_noise_level(self, freqs: np.ndarray, y: np.ndarray, freq_peak: float, noise_level_type: str, snr_threshold: float, noise_kwargs: dict) -> float:
        if noise_level_type != 'local_window_median_snr':
            return self.compute_noise_level(freqs, y, noise_level_type=noise_level_type, snr_threshold=snr_threshold, noise_kwargs=noise_kwargs)
        window = noise_kwargs.get('window', None)
        if window is None:
            raise ValueError("noise_level_type='local_window_median_snr' requires noise_kwargs['window'].")
        window = float(window)
        if window <= 0:
            raise ValueError("noise_kwargs['window'] must be > 0.")
        exclude_width = float(noise_kwargs.get('exclude_width', 0.0))
        if exclude_width < 0:
            raise ValueError("noise_kwargs['exclude_width'] must be >= 0.")
        freqs_valid, y_valid = self.validate_xy(freqs, y)
        local_mask = np.abs(freqs_valid - float(freq_peak)) <= window
        if exclude_width > 0:
            local_mask &= np.abs(freqs_valid - float(freq_peak)) >= exclude_width
        if np.sum(local_mask) < 3:
            raise ValueError('Local-window mask has fewer than 3 samples; adjust window/exclude_width.')
        return np.median(y_valid[local_mask]) * snr_threshold


    # def noise_level(self) -> float:
    #     """
    #     Calculate the noise level of the light curve
    #     """
    #     return np.median(self.pg.amps)*self.snr_threshold if self.normalization == 'amplitude' else np.median(self.pg.powers)*self.snr_threshold

    def iterate(self) -> None:
        """
        Perform a single iteration of pre-whitening
        """
        if self.iteration == 0:
            self.pg_iter = copy.deepcopy(self.pg)
        self.pg_iter.amplitude_power_spectrum(self.t, self.data_iter)
        freqs_i = self.pg_iter.freqs
        if self.normalization == 'amplitude':
            y_i = self.pg_iter.amps
        elif self.normalization == 'psd':
            y_i = self.pg_iter.powers
        else:
            raise ValueError(f'Unsupported normalization: {self.normalization}')

        if y_i is None:
            raise ValueError(f'No periodogram values available for normalization={self.normalization}')

        if self.iteration < self.max_iterations:
            y_max = np.max(y_i)
            freq = freqs_i[np.argmax(y_i)]

            ### SNR stopping condition ###
            if self.noise_level_override:
                noise_level_i = self.noise_level
            elif self.noise_level_type == 'local_window_median_snr':
                noise_level_i = self.compute_local_noise_level(freqs_i, y_i, freq_peak=freq, noise_level_type=self.noise_level_type,
                                                                snr_threshold=self.snr_threshold, noise_kwargs=self.noise_kwargs)
            else:
                noise_level_i = self.compute_noise_level(freqs_i, y_i, noise_level_type=self.noise_level_type,
                                                          snr_threshold=self.snr_threshold, noise_kwargs=self.noise_kwargs)
            if y_max < noise_level_i:
                print('SNR threshold reached')
                self.stop_iteration = True
                return
        
            omega = 2 * np.pi * freq
            p0 = [y_max, omega, 0.5]
            freq_step = np.median(np.diff(freqs_i))
            omega_margin = 2 * np.pi * max(freq_step, 1e-8)
            lower_bounds = [0.0, omega - omega_margin, -2*np.pi]
            upper_bounds = [np.inf, omega + omega_margin, 2*np.pi]

            # Fit on time shifted to the mean epoch so the fitted phase is
            # decorrelated from frequency (Montgomery & O'Donoghue 1999,
            # ADS 1999DSSN...13...28M). The same shifted time must be used when
            # subtracting the fitted sinusoid from data_iter.
            t_shifted = self.t - self.t_ref
            params, _ = curve_fit(self.sinusoidal_model, t_shifted, self.data_iter, p0=p0, bounds=(lower_bounds, upper_bounds))
            ## Negative amp corrections. Flip sign, add pi to phase
            if params[0] < 0:
                params[0] *= -1
                params[2] += np.pi
            # Wrap phase to (-pi, pi].
            phase = (params[2] + np.pi) % (2 * np.pi) - np.pi

            # Post-fit SNR gate. The pre-fit check uses the periodogram peak
            # height y_max, which can include window-function leakage or
            # numerical residue from imperfectly subtracted peaks. If
            # curve_fit actually converged well below the noise threshold,
            # the apparent periodogram peak was not a coherent sinusoid;
            # stop iterating so we don't accumulate degenerate near-zero-
            # amplitude fits at the same frequency.
            fitted_val = params[0] if self.normalization == 'amplitude' else params[0]**2
            if fitted_val < noise_level_i:
                print('Fitted amplitude below SNR threshold; stopping.')
                self.stop_iteration = True
                return

            self.peak_freqs.append(params[1]/(2*np.pi))
            self.peak_phases.append(phase)
            if self.normalization == 'amplitude':
                self.peak_amps.append(params[0])
            elif self.normalization == 'psd':
                # Store the actual power P = A**2 so the 'pow' column matches
                # the periodogram normalization (powers = (4/N) * LS_psd).
                self.peak_powers.append(params[0]**2)
            self.data_iter -= self.sinusoidal_model(t_shifted, *params)
            # self.data_iter = np.nan_to_num(self.data_iter, nan=0.0, posinf=0.0, neginf=0.0)
            self.iteration += 1
        self.freq_container = self.init_freq_container()

    def auto(self, make_plot: bool = True, save: bool = True, remove_overlapping: bool = True, remove_local_snr: bool = False, local_snr_resolution: float = 3,
                flag_harmonics: bool = True, harmonic_tolerance: float = 0.001, frequency_resolution: float = 4/27) -> None:
        """
        Auto iterator for pre-whitening

        Parameters
        ----------
        make_plot : bool, optional, default: True
            If True, make a plot of the pre-whitened light curve
        save : bool, optional, default: True
            If True, save the pre-whitened light curve
        remove_overlapping : bool, optional, default: True
            If True, remove overlapping or very nearby peaks, keeps the highest amplitude one
        remove_local_snr : bool, optional, default: True
            If True, remove frequencies with amplitude less than the local SNR. 
            Local SNR is defined as the median amplitude of frequencies within 3 cycles/day of the peak frequency.
        local_snr_resolution : float, optional, default: 3
            Resolution for the local SNR. The local SNR is defined as the median amplitude of frequencies within `local_snr_resolution` cycles/day of the peak frequency.
        flag_harmonics : bool, optional, default: True
            If True, flag harmonics of detected frequencies
        harmonic_tolerance : float, optional, default: 0.001
            Tolerance for flagging harmonics
        frequency_resolution : float, optional, default: 4/27
            Frequency resolution of the periodogram
        """
        for i in tqdm(range(self.max_iterations), desc='Pre-whitening'):
            self.iterate()
            if self.stop_iteration:
                break
        
        self.post_pw(make_plot=make_plot, save=save, remove_overlapping=remove_overlapping, remove_local_snr=remove_local_snr, local_snr_resolution=local_snr_resolution,
                    flag_harmonics=flag_harmonics, harmonic_tolerance=harmonic_tolerance, frequency_resolution=frequency_resolution)
        print(f'Pre-whitening complete after {self.iteration} iterations')

    def init_freq_container(self) -> pd.DataFrame:
        """
        Convert the pre-whitened light curve to a pandas.DataFrame.
        Phase is recorded at the mean-epoch reference ``self.t_ref``.
        """
        if self.normalization == 'amplitude':
            df = pd.DataFrame({'freq': self.peak_freqs,
                               'amp': self.peak_amps,
                               'phase': self.peak_phases}).sort_values(by='freq', ascending=True)
        elif self.normalization == 'psd':
            df = pd.DataFrame({'freq': self.peak_freqs,
                               'pow': self.peak_powers,
                               'phase': self.peak_phases}).sort_values(by='freq', ascending=True)
        df = df.reset_index(drop=True)
        df['label'] = [f'F{i+1}' for i in range(len(df))]
        return df

    def post_pw(self, make_plot: bool = True, save: bool = True, remove_overlapping: bool = True, remove_local_snr: bool = True, local_snr_resolution: float = 3, 
                flag_harmonics: bool = True, harmonic_tolerance: float = 0.001, frequency_resolution: float = 4/27) -> None:
        """
        Post pre-whitening analysis

        Parameters
        ----------
        make_plot : bool, optional, default: True
            If True, make a plot of the pre-whitened light curve
        save : bool, optional, default: True
            If True, save the pre-whitened light curve
        remove_overlapping : bool, optional, default: True
            If True, remove overlapping or very nearby peaks, keeps the highest amplitude one
        remove_local_snr : bool, optional, default: True
            If True, remove frequencies with amplitude less than the local SNR.
            Local SNR is defined as the median amplitude of frequencies within 3 cycles/day of the peak frequency.
        local_snr_resolution : float, optional, default: 3
            Resolution for the local SNR. The local SNR is defined as the median amplitude of frequencies within `local_snr_resolution` cycles/day of the peak frequency.
        flag_harmonics : bool, optional, default: True
            If True, flag harmonics of detected frequencies
        harmonic_tolerance : float, optional, default: 0.001
            Tolerance for flagging harmonics
        frequency_resolution : float, optional, default: 4/27
            Frequency resolution of the periodogram
        """
        if self.freq_container is None:
            raise ValueError('No frequencies found. Try running auto()/interate() first or decrease the SNR threshold')
        self.flag_harmonics = flag_harmonics
        self.harmonic_tolerance = harmonic_tolerance
        self.frequency_resolution = frequency_resolution
        self.remove_local_snr = remove_local_snr
        self.remove_overlapping = remove_overlapping

        if self.remove_local_snr:
            ## Remove frequencies with amplitude less than the local SNR.
            self.freq_container = self.remove_based_on_local_snr(self.freq_container, resolution=local_snr_resolution)

        if self.remove_overlapping:
            ## Remove overlapping or very nearby peaks, keep the highest amplitude one
            self.freq_container = self.remove_overlapping_freqs(self.freq_container, nearby_tolerance=self.frequency_resolution)
                
        if self.flag_harmonics:
            self.freq_container = self.harmonics_check(self.freq_container, harmonic_tolerance=self.harmonic_tolerance)

        # Attach lower-limit 1-sigma uncertainties on freq, amplitude/power,
        # and phase, plus amplitude-spectrum SNR (see _attach_uncertainties).
        self._attach_uncertainties()

        if save:
            self.freq_container.to_csv(os.path.join(self.output_dir, 'frequencies.csv'), index=False)
            self._save_metadata()

        if make_plot:
            self.post_pw_plot(save=save)

    def _attach_uncertainties(self) -> None:
        """
        Attach lower-limit 1-sigma uncertainties on frequency, amplitude (or
        power), and phase to ``self.freq_container``, plus the amplitude-
        spectrum SNR. Follows the coherent white-noise sinusoid result of:

          - Montgomery & O'Donoghue 1999, *A derivation of the errors for
            least squares fitting to time series data*, DSSN 13, 28
            (ADS 1999DSSN...13...28M)
          - Kjeldsen & Bedding 2012, *Kepler, CoRoT and MOST: Time-Series
            Photometry from Space*, IAU Symp. 7, S285, 17
            (DOI 10.1017/S1743921312000142)

        With ``sigma_m = std(data_iter)`` the time-domain residual scatter
        after pre-whitening, ``N`` the number of points, ``T`` the baseline,
        and per-peak amplitude ``a``, the amplitude-spectrum mean noise is
        ``<A_noise> = sqrt(pi/N) * sigma_m`` (KB12 eq. 3.4), giving

            sigma_f   = sqrt(6/pi^3) * <A_noise> / (a * T)
                      = sqrt(6) / (pi * sqrt(N)) * sigma_m / (T * a)
            sigma_A   = sqrt(2/N) * sigma_m
            sigma_phi = sqrt(2/N) * sigma_m / a

        For the ``normalization='psd'`` branch, ``pow = A**2`` is reported and
        the power uncertainty is propagated as ``sigma_pow = 2 * A * sigma_A``.

        ``snr`` is reported as the amplitude-spectrum SNR
        ``a / <A_noise>`` using the *global* ``<A_noise>`` from the final
        residual (the convention used by the formulas above).

        Caveats: these are theoretical lower limits. Correlated noise,
        unresolved peaks, and intrinsic amplitude/phase modulation will
        inflate the true uncertainty above these values; downstream
        covariance terms (emulator, model floor, mode-ID) must be added
        separately. See logbook 2026-05-28 for the Velociscuti context.
        """
        if self.freq_container is None or len(self.freq_container) == 0:
            return

        N = len(self.t)
        T = float(self.t.max() - self.t.min())
        sigma_m = float(np.std(self.data_iter, ddof=1))
        A_noise = np.sqrt(np.pi / N) * sigma_m

        self.noise_sigma_residual = sigma_m
        self.noise_amp_mean = A_noise

        if self.normalization == 'amplitude':
            amp = self.freq_container['amp'].to_numpy(dtype=float)
        else:
            # 'pow' column stores A**2; recover the time-domain amplitude.
            amp = np.sqrt(np.clip(self.freq_container['pow'].to_numpy(dtype=float), 0.0, None))

        # Guard against zero amplitudes which would blow up sigma_f, sigma_phi.
        safe_amp = np.where(amp > 0, amp, np.nan)

        sigma_A = np.full_like(safe_amp, np.sqrt(2.0 / N) * sigma_m)
        sigma_f = np.sqrt(6.0 / np.pi**3) * A_noise / (safe_amp * T)
        sigma_phi = np.sqrt(2.0 / N) * sigma_m / safe_amp
        snr = safe_amp / A_noise

        self.freq_container['freq_err'] = sigma_f
        self.freq_container['phase_err'] = sigma_phi
        self.freq_container['snr'] = snr
        if self.normalization == 'amplitude':
            self.freq_container['amp_err'] = sigma_A
            col_order = ['label', 'freq', 'freq_err', 'amp', 'amp_err',
                         'phase', 'phase_err', 'snr']
        else:
            # power = A**2  =>  sigma_pow = 2 * A * sigma_A.
            self.freq_container['pow_err'] = 2.0 * safe_amp * sigma_A
            col_order = ['label', 'freq', 'freq_err', 'pow', 'pow_err',
                         'phase', 'phase_err', 'snr']
        self.freq_container = self.freq_container[col_order]

    def _save_metadata(self) -> None:
        """
        Write a small sidecar ``metadata.csv`` with the global quantities
        needed to interpret and reuse the per-peak values: the mean-epoch
        reference ``t_ref`` (phases in frequencies.csv are at this epoch),
        the baseline ``T``, number of points ``N``, the time-domain residual
        scatter ``sigma_m``, and the amplitude-spectrum mean noise
        ``A_noise = sqrt(pi/N)*sigma_m`` (KB12 eq. 3.4).
        """
        meta = pd.DataFrame({
            't_ref': [self.t_ref],
            'T': [float(self.t.max() - self.t.min())],
            'N': [len(self.t)],
            'sigma_m': [self.noise_sigma_residual],
            'A_noise': [self.noise_amp_mean],
        })
        meta.to_csv(os.path.join(self.output_dir, 'metadata.csv'), index=False)

    def post_pw_plot(self, ax: matplotlib.axes._axes.Axes = None, save: bool = True, plot_kwargs: dict = {}, scatter_kwargs: dict = {}) -> matplotlib.axes._axes.Axes:
        """
        Post pre-whitening plot

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes  
            The axes to plot on. If None, will create a new figure and axes.
        save : bool  
            If True, save the plot  
        plot_kwargs : dict  
            Keyword arguments for the plot  
        scatter_kwargs : dict   
            Keyword arguments for the scatter plot  

        Returns
        -------
        ax : matplotlib.axes._axes.Axes 
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        if isinstance(self.freq_container, pd.DataFrame):
            if self.normalization == 'amplitude':
                ax.plot(self.pg.freqs, self.pg.amps, **plot_kwargs)
                ax.scatter(self.freq_container['freq'], self.freq_container['amp'], marker='x', color='maroon', s=10, linewidths=1, zorder=2, **scatter_kwargs)
                ax.set_ylabel("Amplitude")
            if self.normalization == 'psd':
                ax.plot(self.pg.freqs, self.pg.powers, **plot_kwargs)
                ax.scatter(self.freq_container['freq'], self.freq_container['pow'], marker='x', color='maroon', s=10, linewidths=1, zorder=2, **scatter_kwargs)
                ax.set_ylabel("Power")
            ax.set_xlabel("Frequency (1/day)")
            ax.set_xlim(self.fmin, self.fmax)
            if save:
                plt.savefig(os.path.join(self.output_dir, 'prewhitening.png'), dpi=300)
            return ax
        else:
            raise ValueError('No frequencies found. Try running post_pw() first')

    # Sinusoidal function to fit the peaks
    def sinusoidal_model(self, t: np.ndarray, A: float, omega: float, phi: float) -> np.ndarray:
        """
        Sinusoidal model
        """
        return A * np.sin(omega * t + phi)

    def harmonics_check(self, df: pd.DataFrame, harmonic_tolerance: float = 0.01) -> pd.DataFrame:
        """
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
        """
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

    def remove_overlapping_freqs(self, df: pd.DataFrame, nearby_tolerance: float = 0.01) -> pd.DataFrame:
        """
        Collapse clusters of peaks separated by less than ``nearby_tolerance``,
        keeping only the highest-amplitude (or highest-power, in psd mode)
        member of each cluster.

        A cluster is grown by adding consecutive (frequency-sorted) peaks that
        lie within ``nearby_tolerance`` of the cluster's lowest-frequency
        member. Using the cluster *start* as the reference (rather than the
        last-added peak) avoids the chain effect where intermediate peaks
        merge two genuinely distinct modes into a single super-cluster.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with at least 'freq' and ('amp' for amplitude mode or
            'pow' for psd mode).
        nearby_tolerance : float
            Frequency separation below which two peaks are treated as the
            same coherent signal.

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame with overlapping peaks removed and the
            original column ordering preserved.
        """
        if len(df) == 0:
            return df
        val_col = 'pow' if self.normalization == 'psd' else 'amp'
        df = df.sort_values(by='freq', ascending=True).reset_index(drop=True)
        freqs = df['freq'].to_numpy()
        vals = df[val_col].to_numpy()
        keep = np.ones(len(df), dtype=bool)
        i = 0
        while i < len(df):
            j = i + 1
            while j < len(df) and (freqs[j] - freqs[i]) < nearby_tolerance:
                j += 1
            if j > i + 1:
                cluster = np.arange(i, j)
                best = cluster[int(np.argmax(vals[cluster]))]
                keep[cluster] = False
                keep[best] = True
            i = j
        return df.iloc[keep].reset_index(drop=True)

    def remove_based_on_local_snr(self, df: pd.DataFrame, resolution: float = 3) -> pd.DataFrame:
        """
        Remove frequencies based on the local SNR.  

        Parameters
        ----------
        df : pandas.DataFrame  
            DataFrame with columns 'freq' and 'amp'  
        resolution : float
            Resolution for the local SNR. The local SNR is defined as the median amplitude of frequencies within `resolution` cycles/day of the peak frequency.

        Returns
        -------
        df : pandas.DataFrame  
            DataFrame with columns 'freq' and 'amp' with frequencies below the local SNR threshold removed  
        """
        if self.normalization == 'psd':
            df = df.sort_values(by=['freq', 'pow'], ascending=False)
        else:
            df = df.sort_values(by=['freq', 'amp'], ascending=False)
        df = df.reset_index(drop=True)
        to_drop = []
        for i in range(len(df)-1):
            freq = df.iloc[i]['freq']
            if self.normalization == 'psd':
                local_noise = np.median(df[(df['freq'] > freq - resolution) & (df['freq'] < freq + resolution)]['pow'])
                if df.iloc[i]['pow'] < local_noise:
                    to_drop.append(i)
            elif self.normalization == 'amplitude':
                local_noise = np.median(df[(df['freq'] > freq - resolution) & (df['freq'] < freq + resolution)]['amp'])
                if df.iloc[i]['amp'] < local_noise:
                    to_drop.append(i)
        return df.drop(index=to_drop)
    
    def __repr__(self):
        if self.name is not None:
            return f"PreWhitener(name='{self.name}')"
        else:
            return f'PreWhitener(lc=({self.t}, {self.data}))'