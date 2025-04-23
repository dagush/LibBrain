import numpy as np
from scipy.stats import permutation_test
from statannotations.stats.StatTest import StatTest


def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def stat_permutation_test(group_data1, group_data2, **stats_params):
    alternative = stats_params['alternative'] if 'alternative' in stats_params else 'two-sided'
    res = permutation_test((group_data1, group_data2), statistic, n_resamples=10000,
                           vectorized=True, alternative=alternative)
    return res.statistic, res.pvalue


def custom_permutation():
    custom_long_name = 'Permutation test'
    custom_short_name = 'Permutation'
    custom_func = stat_permutation_test
    custom_test = StatTest(custom_func, custom_long_name, custom_short_name)
    return custom_test

