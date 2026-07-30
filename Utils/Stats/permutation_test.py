"""
permutation_test.py

A small, explicit wrapper around ``scipy.stats.permutation_test`` for
comparing two independent samples.

The default statistic is the difference between sample means:

    mean(x) - mean(y)

This module is intentionally independent of ``statannotations`` and can be
used directly or registered in another pairwise-comparison framework.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import permutation_test as scipy_permutation_test


StatisticFunction = Callable[..., float | np.ndarray]


def mean_difference(
    x: np.ndarray,
    y: np.ndarray,
    axis: int = 0,
) -> float | np.ndarray:
    """Return the difference between the sample means: ``mean(x) - mean(y)``."""
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def stat_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
    *,
    statistic: StatisticFunction = mean_difference,
    n_resamples: int = 10_000,
    permutation_type: str = "independent",
    vectorized: bool | None = True,
    batch: int | None = None,
    rng=None,
) -> tuple[float, float]:
    """
    Perform a two-sample permutation test using SciPy.

    Parameters
    ----------
    x, y
        One-dimensional numerical arrays containing the two samples.

    alternative
        Alternative hypothesis: ``"two-sided"``, ``"less"``, or
        ``"greater"``.

    statistic
        Statistic computed for the observed and permuted samples. By default,
        the difference in sample means, ``mean(x) - mean(y)``.

        When ``vectorized=True``, the function must accept an ``axis`` keyword
        argument, as ``mean_difference`` does.

    n_resamples
        Number of random permutations. The default, 10,000, preserves the
        behavior of the original statannotations extension.

    permutation_type
        Permutation strategy passed to ``scipy.stats.permutation_test``.
        The default is ``"independent"``.

    vectorized
        Whether ``statistic`` supports vectorized evaluation. The default is
        ``True`` because ``mean_difference`` is vectorized.

    batch
        Number of permutations processed per vectorized call. ``None`` lets
        SciPy choose its default behavior.

    rng
        Optional random-number generator or seed passed to SciPy. Supplying a
        fixed value makes Monte Carlo permutation results reproducible.

    Returns
    -------
    statistic_value, p_value : tuple[float, float]
        Observed test statistic and permutation p-value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must both be one-dimensional arrays.")

    if len(x) == 0 or len(y) == 0:
        raise ValueError("x and y must both contain at least one observation.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(
            "x and y must contain only finite values. Remove NaNs and infinities "
            "before calling stat_permutation_test()."
        )

    valid_alternatives = {"two-sided", "less", "greater"}
    if alternative not in valid_alternatives:
        raise ValueError(
            f"alternative must be one of {sorted(valid_alternatives)}; "
            f"got {alternative!r}."
        )

    result = scipy_permutation_test(
        (x, y),
        statistic,
        permutation_type=permutation_type,
        vectorized=vectorized,
        n_resamples=n_resamples,
        batch=batch,
        alternative=alternative,
        # rng=rng,
    )

    return float(result.statistic), float(result.pvalue)


# Convenient alias for direct use.
permutation_test = stat_permutation_test


__all__ = [
    "mean_difference",
    "permutation_test",
    "stat_permutation_test",
]
