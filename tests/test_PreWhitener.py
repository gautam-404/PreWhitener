import unittest
import pandas as pd
import numpy as np
import lightkurve as lk

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from PreWhitener.PreWhitener import PreWhitener

class TestPreWhitener(unittest.TestCase):

    def test_init(self):
        # Test initialization with lightkurve.LightCurve
        pw = PreWhitener(name='TIC 7903477')
        lc = pw.lc
        self.assertEqual(pw.name, None)
        np.testing.assert_array_equal(pw.t, lc.time.value)
        np.testing.assert_array_equal(pw.data, lc.flux.value)
        self.assertEqual(pw.max_iterations, 100)
        self.assertEqual(pw.snr_threshold, 5)
        self.assertEqual(pw.flag_harmonics, True)
        self.assertEqual(pw.harmonic_tolerance, 0.001)
        self.assertEqual(pw.frequency_resolution, 4/27)
        self.assertEqual(pw.fmin, 0)
        self.assertLessEqual(pw.fmax, pw.nyquist_frequency())
        self.assertEqual(pw.nyq_mult, 1)
        self.assertEqual(pw.oversample_factor, 5)
        self.assertEqual(pw.mode, 'amplitude')
        self.assertEqual(pw.iteration, 0)
        self.assertEqual(pw.stop_iteration, False)
        self.assertEqual(pw.peak_freqs, [])
        self.assertEqual(pw.peak_amps, [])
        self.assertEqual(pw.f_container, None)

        # Test initialization with pandas.DataFrame
        df = pd.DataFrame({'time': np.arange(0, 10, 0.1), 'flux': np.random.randn(100)})
        pw = PreWhitener(lc=df)
        self.assertEqual(pw.name, None)
        np.testing.assert_array_equal(pw.t, df['time'].values)
        np.testing.assert_array_equal(pw.data, df['flux'].values)
        self.assertEqual(pw.max_iterations, 100)
        self.assertEqual(pw.snr_threshold, 5)
        self.assertEqual(pw.flag_harmonics, True)
        self.assertEqual(pw.harmonic_tolerance, 0.001)
        self.assertEqual(pw.frequency_resolution, 4/27)
        self.assertEqual(pw.fmin, 0)
        self.assertEqual(pw.fmax, 1/(2*np.median(np.diff(df['time'].values))))
        self.assertEqual(pw.nyq_mult, 1)
        self.assertEqual(pw.oversample_factor, 5)
        self.assertEqual(pw.mode, 'amplitude')
        self.assertEqual(pw.iteration, 0)
        self.assertEqual(pw.stop_iteration, False)
        self.assertEqual(pw.peak_freqs, [])
        self.assertEqual(pw.peak_amps, [])
        self.assertEqual(pw.f_container, None)

        # Test initialization with tuple
        t = np.arange(0, 10, 0.1)
        f = np.random.randn(100)
        pw = PreWhitener(lc=(t, f))
        self.assertEqual(pw.name, None)
        np.testing.assert_array_equal(pw.t, t)
        np.testing.assert_array_equal(pw.data, f)
        self.assertEqual(pw.max_iterations, 100)
        self.assertEqual(pw.snr_threshold, 5)
        self.assertEqual(pw.flag_harmonics, True)
        self.assertEqual(pw.harmonic_tolerance, 0.001)
        self.assertEqual(pw.frequency_resolution, 4/27)
        self.assertEqual(pw.fmin, 0)
        self.assertEqual(pw.fmax, 1/(2*np.median(np.diff(t))))
        self.assertEqual(pw.nyq_mult, 1)
        self.assertEqual(pw.oversample_factor, 5)
        self.assertEqual(pw.mode, 'amplitude')
        self.assertEqual(pw.iteration, 0)
        self.assertEqual(pw.stop_iteration, False)
        self.assertEqual(pw.peak_freqs, [])
        self.assertEqual(pw.peak_amps, [])
        self.assertEqual(pw.f_container, None)

    def test_get_lightcurve(self):
        # Test getting lightkurve data with valid name
        pw = PreWhitener(name='TIC 7903477')
        self.assertTrue(pw.get_lightcurve())

        # Test getting lightkurve data with invalid name
        pw = PreWhitener(name='invalid_name')
        self.assertFalse(pw.get_lightcurve())

    def test_iterative_prewhitening(self):
        # Test iterative pre-whitening with simulated data
        t = np.arange(0, 10, 0.1)
        f = np.sin(2*np.pi*0.5*t) + np.sin(2*np.pi*1.5*t) + np.sin(2*np.pi*2.5*t) + np.random.randn(len(t))
        pw = PreWhitener(lc=(t, f), max_iterations=3, snr_threshold=3)
        for i in range(pw.max_iterations):
            pw.iterate()
        self.assertEqual(pw.iteration, 3)
        self.assertEqual(len(pw.peak_freqs), 3)

if __name__ == '__main__':
    unittest.main()