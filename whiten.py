import lightkurve as lk
import pywhiten
import pandas as pd
import os

# star = 'HD47129'
# star = 'HD20203'
# star = 'TIC171591531'
# star = 'HD28548'
star = 'V647Tau'

lk_search = lk.search_lightcurve(star, mission="TESS", cadence=120)
print(lk_search)
lc = lk_search[0].download().remove_nans().remove_outliers()
lc.to_pandas().to_csv(f'stars/{star}_lk.csv')
t, m = lc.time.value, lc.flux.value

config_dict = {'periodograms' : {'upper_limit':90}, 
               'autopw' : {'peak_selection_method' : 'slf', 'new_lc_generation_method' : 'mf',
                           'peak_selection_highest_override' : 1, 'peak_selection_cutoff_sig' : 3,
                           'cutoff_iteration' : 50, 'autopw.bounds' : {'phase_fit_rejection_criterion' : 0.1,
                                                                       'freq_lower_coeff' : 0.8,
                                                                       'freq_upper_coeff' : 1.2,
                                                                       'amp_lower_coeff' : 0.8,
                                                                       'amp_upper_coeff' : 1.2,
                                                                       'phase_lower' : -100,
                                                                       'phase_upper' : 100}},
                'output' : {'show_peak_selection_plots' : True}}
pywhitener = pywhiten.PyWhitener(time=list(t), data=list(m), cfg=config_dict)

# pywhitener.auto()
try:
    pywhitener.auto()
except Exception as e:
    print(e)
finally:
    pywhitener.post_pw(residual_lc_idx=-2)


