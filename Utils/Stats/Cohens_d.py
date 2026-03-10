# --------------------------------------------------------------------------------------
# Cohen's d calculation
# from https://www.askpython.com/python/examples/cohens-d-python
# --------------------------------------------------------------------------------------

# importing numpy for using in-built functions
import numpy as np
import pandas as pd
import itertools

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# Sawilowsky (2009) extension of Cohen’s conventions
def cohens_d_label(d):
    d = abs(d)
    if d < 0.01:
        return "negligible"
    elif d < 0.2:
        return "very small"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    elif d < 1.2:
        return "large"
    elif d < 2.0:
        return "very large"
    else:
        return "huge"

# --------------------------------------------------------------------------------------
# One sample
# --------------------------------------------------------------------------------------
def cohen_d_onesample(x):
    return np.mean(x) / np.std(x, ddof=1)

def bootstrap_ci(x, stat_func, n_boot=5000, alpha=0.05, rng=None):
    rng = np.random.default_rng(rng)
    stats = [
        stat_func(rng.choice(x, size=len(x), replace=True))
        for _ in range(n_boot)
    ]
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper

def effect_size_table(data_dict, alpha=0.05, n_boot=5000):
    rows = []

    for label, x in data_dict.items():
        x = np.asarray(x)
        d = cohen_d_onesample(x)
        ci_low, ci_high = bootstrap_ci(
            x, cohen_d_onesample, n_boot=n_boot, alpha=alpha
        )

        rows.append({
            "Outcome": label,
            "Cohen's d": round(d, 3),
            "95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
            "Interpretation": cohens_d_label(d)
        })

    df = pd.DataFrame(rows)
    return df

# --------------------------------------------------------------------------------------
# Two samples
# --------------------------------------------------------------------------------------
def cohens_d(group1, group2):
    # Calculating means of the two groups
    mean1, mean2 = np.mean(group1), np.mean(group2)

    # Calculating pooled standard deviation
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

    # Calculating Cohen's d
    d = (mean1 - mean2) / pooled_std

    return d

def bootstrap_ci_two_sample(x, y, stat_func, n_boot=5000, alpha=0.05, paired=False, rng=None):
    rng = np.random.default_rng(rng)
    stats = []

    for _ in range(n_boot):
        if paired:
            idx = rng.choice(len(x), size=len(x), replace=True)
            xb, yb = x[idx], y[idx]
        else:
            xb = rng.choice(x, size=len(x), replace=True)
            yb = rng.choice(y, size=len(y), replace=True)

        stats.append(stat_func(xb, yb))

    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper

def pairwise_effect_size_table(
    data_dict,
    cohens_d_func=cohens_d,
    paired=False,
    alpha=0.05,
    n_boot=5000
):
    rows = []

    for (label1, x), (label2, y) in itertools.combinations(data_dict.items(), 2):
        x, y = np.asarray(x), np.asarray(y)

        if paired and len(x) != len(y):
            raise ValueError(f"Paired design requires equal lengths: {label1}, {label2}")

        d = cohens_d_func(x, y)
        ci_low, ci_high = bootstrap_ci_two_sample(
            x, y, cohens_d_func,
            paired=paired,
            alpha=alpha,
            n_boot=n_boot
        )

        rows.append({
            "Comparison": f"{label1} vs {label2}",
            "Cohen's d": round(d, 3),
            "95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
            "Interpretation": cohens_d_label(d)
        })

    return pd.DataFrame(rows)