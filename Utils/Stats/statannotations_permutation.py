from statannotations.stats.StatTest import StatTest

from Utils.Stats.permutation_test import stat_permutation_test


def custom_permutation():
    """
    Return a statannotations-compatible wrapper around the standalone
    permutation test defined in permutation_test.py.
    """
    custom_long_name = "Permutation test"
    custom_short_name = "Permutation"

    custom_test = StatTest(
        stat_permutation_test,
        custom_long_name,
        custom_short_name,
    )

    return custom_test