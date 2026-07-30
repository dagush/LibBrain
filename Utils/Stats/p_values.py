"""
p_values.py

Explicit pairwise statistical comparisons for data stored as:

    {
        "group_A": array_like,
        "group_B": array_like,
        ...
    }

The module uses:
    - NumPy for array handling
    - SciPy for statistical tests
    - statsmodels for multiple-testing correction
    - pandas for tidy tabular output

Supported tests
---------------
Independent samples:
    - "mann-whitney"  : Mann-Whitney U test
    - "welch"         : Welch's unequal-variance t-test
    - "ttest"         : Student's independent-samples t-test
    - "ks"            : Two-sample Kolmogorov-Smirnov test

Paired samples:
    - "wilcoxon"      : Wilcoxon signed-rank test
    - "paired-ttest"  : Paired-samples t-test

Supported multiple-testing corrections are those implemented by
statsmodels.stats.multitest.multipletests, including:
    - "bonferroni"
    - "sidak"
    - "holm-sidak"
    - "holm"
    - "simes-hochberg"
    - "hommel"
    - "fdr_bh"   (Benjamini-Hochberg)
    - "fdr_by"   (Benjamini-Yekutieli)
    - "fdr_tsbh"
    - "fdr_tsbky"

Example
-------
>>> import numpy as np
>>> from Utils.Stats.p_values import compare_groups
>>>
>>> data = {
...     "CTRL": np.array([1.1, 1.3, 1.2, 1.5]),
...     "MCI":  np.array([1.5, 1.8, 1.7, 1.9, 2.0]),
...     "AD":   np.array([2.1, 2.4, 2.2]),
... }
>>>
>>> results = compare_groups(
...     data,
...     test="mann-whitney",
...     correction="fdr_bh",
...     alpha=0.05,
... )
>>> print(results)
"""

from __future__ import annotations

from itertools import combinations
import inspect
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ArrayLike = Sequence[float] | np.ndarray
TestFunction = Callable[..., object]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _as_1d_float_array(values: ArrayLike, group_name: str) -> np.ndarray:
    """Convert input values to a one-dimensional float NumPy array."""
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(
            f"Group {group_name!r} must contain a one-dimensional array; "
            f"got shape {array.shape}."
        )

    return array


def _clean_independent_samples(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove non-finite observations independently from two samples."""
    return x[np.isfinite(x)], y[np.isfinite(y)]


def _clean_paired_samples(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove paired observations whenever either member of the pair is non-finite.

    Paired tests require x and y to have the same original length because
    observation i in x must correspond to observation i in y.
    """
    if len(x) != len(y):
        raise ValueError(
            "Paired tests require equal-length arrays before NaN removal; "
            f"received lengths {len(x)} and {len(y)}."
        )

    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _extract_test_result(result: object) -> tuple[float, float]:
    """Normalize common statistical-test outputs to ``(statistic, p_value)``.

    Supported return styles are:

    - a two-item tuple/list: ``(statistic, p_value)``;
    - a SciPy-style result object with ``.statistic`` and ``.pvalue``;
    - a statannotations result object with ``.stat_value`` and ``.pvalue``.
    """
    if isinstance(result, (tuple, list)) and len(result) == 2:
        statistic, p_value = result
    elif hasattr(result, "stat_value") and hasattr(result, "pvalue"):
        statistic = result.stat_value
        p_value = result.pvalue
    elif hasattr(result, "statistic") and hasattr(result, "pvalue"):
        statistic = result.statistic
        p_value = result.pvalue
    else:
        raise TypeError(
            "The statistical test must return either (statistic, p_value), "
            "a SciPy-style object with '.statistic' and '.pvalue', or a "
            "statannotations result with '.stat_value' and '.pvalue'."
        )

    return float(statistic), float(p_value)


def _call_test(
    test_function: Callable[..., object],
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    alternative: str,
    test_params: Mapping[str, Any] | None = None,
) -> tuple[float, float]:
    """Call a plain function or callable object using compatible keywords.

    This supports, among others:

    - built-in wrappers in this module;
    - plain functions returning ``(statistic, p_value)``;
    - SciPy statistical functions returning result objects;
    - ``statannotations.stats.StatTest`` instances, whose ``__call__`` method
      accepts ``alpha`` and ``**stat_params`` and returns a StatResult object.
    """
    if not callable(test_function):
        raise TypeError("test must be a registered test name or a callable object.")

    kwargs: dict[str, Any] = dict(test_params or {})

    try:
        signature = inspect.signature(test_function)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        parameters = signature.parameters
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

        if "alternative" in parameters or accepts_var_kwargs:
            kwargs.setdefault("alternative", alternative)

        if "alpha" in parameters or accepts_var_kwargs:
            kwargs.setdefault("alpha", alpha)
    else:
        # Best generic fallback for opaque callables.
        kwargs.setdefault("alternative", alternative)

    result = test_function(x, y, **kwargs)
    return _extract_test_result(result)


# ---------------------------------------------------------------------
# Statistical tests
#
# Each function returns:
#     statistic, raw_p_value
# ---------------------------------------------------------------------

def _mann_whitney(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str,
) -> tuple[float, float]:
    """Mann-Whitney U test for two independent samples."""
    result = stats.mannwhitneyu(
        x,
        y,
        alternative=alternative,
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def _welch_ttest(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str,
) -> tuple[float, float]:
    """Welch's t-test for two independent samples with unequal variances."""
    result = stats.ttest_ind(
        x,
        y,
        equal_var=False,
        alternative=alternative,
    )
    return float(result.statistic), float(result.pvalue)


def _student_ttest(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str,
) -> tuple[float, float]:
    """Student's t-test for two independent samples with equal variances."""
    result = stats.ttest_ind(
        x,
        y,
        equal_var=True,
        alternative=alternative,
    )
    return float(result.statistic), float(result.pvalue)


def _wilcoxon(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str,
) -> tuple[float, float]:
    """Wilcoxon signed-rank test for paired samples."""
    result = stats.wilcoxon(
        x,
        y,
        alternative=alternative,
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def _paired_ttest(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str,
) -> tuple[float, float]:
    """Paired-samples t-test."""
    result = stats.ttest_rel(
        x,
        y,
        alternative=alternative,
    )
    return float(result.statistic), float(result.pvalue)


def _ks_2sample(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test."""
    result = stats.ks_2samp(
        x,
        y,
        alternative=alternative,
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------
# Test registry
#
# To add another test, define a function with signature:
#
#     func(x, y, alternative) -> (statistic, p_value)
#
# and add it here with the corresponding paired/unpaired flag.
# ---------------------------------------------------------------------

TESTS: dict[str, dict[str, object]] = {
    "mann-whitney": {
        "function": _mann_whitney,
        "paired": False,
        "display_name": "Mann-Whitney U",
    },
    "welch": {
        "function": _welch_ttest,
        "paired": False,
        "display_name": "Welch's t-test",
    },
    "ttest": {
        "function": _student_ttest,
        "paired": False,
        "display_name": "Student's t-test",
    },
    "wilcoxon": {
        "function": _wilcoxon,
        "paired": True,
        "display_name": "Wilcoxon signed-rank",
    },
    "paired-ttest": {
        "function": _paired_ttest,
        "paired": True,
        "display_name": "Paired t-test",
    },
    "ks": {
        "function": _ks_2sample,
        "paired": False,
        "display_name": "Kolmogorov-Smirnov",
    },
}


TEST_ALIASES = {
    "mannwhitney": "mann-whitney",
    "mann_whitney": "mann-whitney",
    "mann-whitney-u": "mann-whitney",
    "mw": "mann-whitney",
    "welch-t": "welch",
    "welch_t": "welch",
    "welch-ttest": "welch",
    "student": "ttest",
    "student-t": "ttest",
    "student_t": "ttest",
    "independent-ttest": "ttest",
    "independent_ttest": "ttest",
    "wilcoxon-signed-rank": "wilcoxon",
    "wilcoxon_signed_rank": "wilcoxon",
    "paired_ttest": "paired-ttest",
    "paired-t": "paired-ttest",
    "ttest_rel": "paired-ttest",
    "kolmogorov-smirnov": "ks",
    "ks-2sample": "ks",
    "ks_2samp": "ks",
}


def available_tests() -> pd.DataFrame:
    """Return a table describing the statistical tests available."""
    rows = []

    for name, information in TESTS.items():
        rows.append(
            {
                "test": name,
                "display_name": information["display_name"],
                "paired": information["paired"],
            }
        )

    return pd.DataFrame(rows)


def register_test(
    name: str,
    function: TestFunction,
    *,
    paired: bool,
    display_name: str | None = None,
) -> None:
    """
    Register a custom two-sample statistical test.

    The custom function must have this signature:

        function(x, y, alternative) -> (statistic, p_value)

    Parameters
    ----------
    name
        Name used in ``compare_groups(test=name)``.

    function
        Callable accepting two finite one-dimensional NumPy arrays and an
        alternative hypothesis string. It must return ``(statistic, p_value)``.

    paired
        Whether the test requires paired observations.

    display_name
        Optional human-readable test name.
    """
    normalized_name = name.strip().lower()

    if not normalized_name:
        raise ValueError("Test name cannot be empty.")

    if not callable(function):
        raise TypeError("function must be callable.")

    TESTS[normalized_name] = {
        "function": function,
        "paired": bool(paired),
        "display_name": display_name or normalized_name,
    }


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

def compare_groups(
    data: Mapping[str, ArrayLike],
    *,
    test: str | Callable[..., object] = "mann-whitney",
    correction: str | None = "fdr_bh",
    alpha: float = 0.05,
    alternative: str = "two-sided",
    pairs: Iterable[tuple[str, str]] | None = None,
    paired: bool | None = None,
    test_name: str | None = None,
    test_params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Perform pairwise statistical comparisons between groups.

    Parameters
    ----------
    data
        Mapping from group names to one-dimensional numerical arrays.
        Arrays may have unequal lengths for independent-samples tests.

    test
        Statistical test to use. This may be either a registered test name or
        any callable object accepting ``(x, y, ...)``. Supported callable
        outputs include ``(statistic, p_value)``, SciPy-style result objects
        with ``.statistic``/``.pvalue``, and statannotations results with
        ``.stat_value``/``.pvalue``. Built-in choices are:

        - ``"mann-whitney"``
        - ``"welch"``
        - ``"ttest"``
        - ``"wilcoxon"``
        - ``"paired-ttest"``
        - ``"ks"``

        Several common aliases are also accepted.

    correction
        Multiple-testing correction passed directly to
        ``statsmodels.stats.multitest.multipletests``.

        Common choices include:

        - ``None``             : no correction
        - ``"bonferroni"``
        - ``"sidak"``
        - ``"holm"``
        - ``"holm-sidak"``
        - ``"fdr_bh"``        : Benjamini-Hochberg
        - ``"fdr_by"``        : Benjamini-Yekutieli

    alpha
        Significance level used to compute the ``reject`` column.

    alternative
        Alternative hypothesis. Usually one of:

        - ``"two-sided"``
        - ``"less"``
        - ``"greater"``

        The exact interpretation depends on the chosen statistical test.

    pairs
        Optional iterable of explicit ``(group1, group2)`` pairs.
        If omitted, every unique pair of groups is compared.

    paired
        For a custom callable, whether the observations are paired. The default
        is ``False``. For registered tests, their built-in paired/unpaired
        setting is used unless this argument explicitly overrides it.

    test_name
        Optional display name used in the output table for a custom callable.

    test_params
        Optional extra keyword arguments passed to the statistical callable.
        For example, these may include resampling settings for a permutation
        test. ``alpha`` and ``alternative`` are added automatically when the
        callable accepts them, unless already supplied here.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with columns:

        - group1
        - group2
        - test
        - paired
        - n1
        - n2
        - mean1
        - mean2
        - median1
        - median2
        - statistic
        - p_raw
        - correction
        - p_corrected
        - alpha
        - reject

    Notes
    -----
    Non-finite observations (NaN, +inf, -inf) are removed automatically.

    For independent-samples tests, observations are removed independently
    from each group.

    For paired tests, the two original arrays must have equal length, and
    a pair is removed if either member of that pair is non-finite.
    """
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping of group names to arrays.")

    if len(data) < 2:
        raise ValueError("At least two groups are required.")

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1; got {alpha}.")

    valid_alternatives = {"two-sided", "less", "greater"}
    if alternative not in valid_alternatives:
        raise ValueError(
            f"alternative must be one of {sorted(valid_alternatives)}; "
            f"got {alternative!r}."
        )

    # Resolve either a registered test name or an arbitrary callable object.
    if isinstance(test, str):
        test_key = test.strip().lower()
        test_key = TEST_ALIASES.get(test_key, test_key)

        if test_key not in TESTS:
            raise ValueError(
                f"Unknown test {test!r}. Available tests are: "
                f"{sorted(TESTS.keys())}."
            )

        test_information = TESTS[test_key]
        test_function = test_information["function"]
        is_paired = bool(test_information["paired"]) if paired is None else bool(paired)
        display_name = test_name or str(test_information["display_name"])
    elif callable(test):
        test_function = test
        is_paired = False if paired is None else bool(paired)
        display_name = test_name or getattr(test, "__name__", test.__class__.__name__)
    else:
        raise TypeError(
            "test must be either the name of a registered test or a callable object."
        )

    # Convert all groups once.
    arrays = {
        str(group_name): _as_1d_float_array(values, str(group_name))
        for group_name, values in data.items()
    }

    group_names = list(arrays.keys())

    # Determine which pairs to compare.
    if pairs is None:
        comparison_pairs = list(combinations(group_names, 2))
    else:
        comparison_pairs = list(pairs)

        if not comparison_pairs:
            raise ValueError("pairs cannot be empty.")

        for group1, group2 in comparison_pairs:
            if group1 not in arrays:
                raise KeyError(f"Unknown group in pairs: {group1!r}.")
            if group2 not in arrays:
                raise KeyError(f"Unknown group in pairs: {group2!r}.")
            if group1 == group2:
                raise ValueError(
                    f"A group cannot be compared with itself: {group1!r}."
                )

    rows = []

    for group1, group2 in comparison_pairs:
        x = arrays[group1]
        y = arrays[group2]

        if is_paired:
            x_clean, y_clean = _clean_paired_samples(x, y)
        else:
            x_clean, y_clean = _clean_independent_samples(x, y)

        if len(x_clean) == 0 or len(y_clean) == 0:
            raise ValueError(
                f"Comparison {group1!r} vs {group2!r} has no finite "
                "observations after cleaning."
            )

        statistic, p_raw = _call_test(
            test_function,
            x_clean,
            y_clean,
            alpha=alpha,
            alternative=alternative,
            test_params=test_params,
        )

        rows.append(
            {
                "group1": group1,
                "group2": group2,
                "test": display_name,
                "paired": is_paired,
                "n1": len(x_clean),
                "n2": len(y_clean),
                "mean1": float(np.mean(x_clean)),
                "mean2": float(np.mean(y_clean)),
                "median1": float(np.median(x_clean)),
                "median2": float(np.median(y_clean)),
                "statistic": statistic,
                "p_raw": p_raw,
            }
        )

    results = pd.DataFrame(rows)

    raw_pvalues = results["p_raw"].to_numpy(dtype=float)

    if not np.all(np.isfinite(raw_pvalues)):
        invalid_rows = results.index[~np.isfinite(raw_pvalues)].tolist()
        raise ValueError(
            "One or more statistical tests returned a non-finite p-value. "
            f"Affected result rows: {invalid_rows}."
        )

    if correction is None:
        results["correction"] = None
        results["p_corrected"] = raw_pvalues
        results["reject"] = raw_pvalues < alpha
    else:
        reject, p_corrected, _, _ = multipletests(
            raw_pvalues,
            alpha=alpha,
            method=correction,
        )

        results["correction"] = correction
        results["p_corrected"] = p_corrected
        results["reject"] = reject

    results["alpha"] = alpha

    # Put inferential results together at the end.
    column_order = [
        "group1",
        "group2",
        "test",
        "paired",
        "n1",
        "n2",
        "mean1",
        "mean2",
        "median1",
        "median2",
        "statistic",
        "p_raw",
        "correction",
        "p_corrected",
        "alpha",
        "reject",
    ]

    return results[column_order]


__all__ = [
    "TESTS",
    "available_tests",
    "compare_groups",
    "register_test",
]
