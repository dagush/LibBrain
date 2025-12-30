# --------------------------------------------------------------------------------------
# Generic utilities to harmonize DataFrames with subjects coming from different sites
#
# By Gustavo Patow
# --------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from neuroHarmonize import harmonizationLearn, harmonizationApply
from typing import List, Optional


# --------------------------------------------------------------------------------------
# Convenience analysis and reporting methods
# --------------------------------------------------------------------------------------
def check_sites(df: pd.DataFrame,
                site_col: str,):
    counts = df[site_col].value_counts()
    return counts


def report_sites(df_w, DL):
    groups = DL.get_groupLabels()
    subjects = DL.get_classification()
    site_counts = check_sites(df_w, 'site')
    for site,count in site_counts.items():
        group_counts = {group: 0 for group in groups}
        rows = df_w[df_w['site']==site]
        for index, row in rows.iterrows():
            group_counts[subjects[row['id']]] += 1
        group_text = {g: group_counts[g] for g in groups}
        print(f"Site {site} ({count}): {group_text}")


# --------------------------------------------------------------------------------------
# Harmonization code!
# --------------------------------------------------------------------------------------
def remove_NaN(df: pd.DataFrame):
    if df.isnull().any().any():
        df = df.dropna()
    return df


def harmonize_dataset(
        df_wide: pd.DataFrame,
        feature_cols: List[str],
        site_col: str,
        covariate_cols: Optional[List[str]] = None,
        harmonization_level: str = "global"  # "global" or "region"
) -> pd.DataFrame:
    """
    Applies ComBat harmonization to a wide-format DataFrame of features.

    Parameters:
    - df_wide (pd.DataFrame): Input DataFrame in wide format (subjects as rows,
                              features and metadata as columns).
    - feature_cols (List[str]): List of column names corresponding to the
                                numerical features (e.g., region scores) to be harmonized.
    - site_col (str): The column name identifying the acquisition site/batch ID
                      for each subject.
    - covariate_cols (Optional[List[str]]): List of column names corresponding to
                                          biological/clinical covariates (e.g., age, sex)
                                          to preserve during harmonization. Defaults to None.
    - harmonization_level:
                    "global" (default): estimate a single site effect per subject and
                    apply it uniformly to all features.
                    "region": estimate region-specific site effects (ComBat default behavior).

    Returns:
    - pd.DataFrame: A new DataFrame with the harmonized features replacing the originals,
                    while keeping all original metadata columns intact.
    """

    # 1. Input Validation and Setup
    if site_col not in df_wide.columns:
        raise ValueError(f"Site column '{site_col}' not found in DataFrame.")

    if harmonization_level not in ["global", "region"]:
        raise ValueError("harmonization_level must be 'global' or 'region'")

    all_covariate_cols = [site_col] + (covariate_cols if covariate_cols else [])
    for col in feature_cols + all_covariate_cols:
        if col not in df_wide.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    # Ensure site column is treated as a category
    df_wide[site_col] = df_wide[site_col].astype('category')

    # Remove the sites with only 1 subject
    site_stats = check_sites(df_wide, site_col)
    discarded_sites = [s for s,v in site_stats.items() if v==1]
    df_final = df_wide[~df_wide['site'].isin(discarded_sites)]

    # remove NaNs...
    df_final = remove_NaN(df_final)

    # Prepare data matrix and covariates matrix
    if harmonization_level == "region":
        # Region-based harmonization (current behavior)
        data_to_harmonize = df_final[feature_cols].values
    elif harmonization_level == "global":
        # True global harmonization:
        # collapse regions into a single global summary per subject
        global_signal = df_final[feature_cols].mean(axis=1).values.reshape(-1, 1)
        data_to_harmonize = global_signal

    covariates_df = df_final[all_covariate_cols]

    print(f"Starting harmonization for {len(feature_cols)} features and {len(df_final)} subjects...")
    print(f"Site column used: '{site_col}'")
    if covariate_cols:
        print(f"Covariates preserved: {covariate_cols}")

    covariates_df = covariates_df.rename(columns={site_col: 'SITE'})

    # Convert categorical covariates to numeric (required by neuroHarmonize)
    covariates_df = pd.get_dummies(
        covariates_df,
        columns=[col for col in covariate_cols if covariates_df[col].dtype == object],
        drop_first=True
    )

    # -----------------------------------------------------
    # 2. Perform ComBat harmonization
    model, harmonized_data_np = harmonizationLearn(
        data_to_harmonize,
        covariates_df,
    )
    print("Harmonization complete.")
    # -----------------------------------------------------

    # 3. Reconstruct the output DataFrame
    # Create a copy of the original DF to avoid modifying the input DF in place
    df_harmonized = df_final.copy()

    # Replace the original feature columns with the new harmonized values
    if harmonization_level == "region":
        df_harmonized[feature_cols] = harmonized_data_np
    elif harmonization_level == "global":
        # Compute per-subject correction factor
        original_global = global_signal.flatten()
        harmonized_global = harmonized_data_np.flatten()

        # Avoid division by zero
        eps = 1e-8
        scaling = harmonized_global / (original_global + eps)

        # Apply the same correction to all regions
        df_harmonized[feature_cols] = df_final[feature_cols].values * scaling[:, None]

    print("Harmonized values integrated back into DataFrame.")
    return df_harmonized


# --------------------------------------------------------------------------------------
# --- Example Usage (Demonstration with dummy data) ---
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Create a dummy wide DataFrame for demonstration
    np.random.seed(42)
    sites = ['SiteA'] * 10 + ['SiteB'] * 10 + ['SiteC'] * 10
    subjects = [f'Sub{i:02d}' for i in range(30)]
    age = np.random.randint(20, 50, 30)
    # Simulate site bias: SiteB has slightly higher values overall, SiteC slightly lower
    region1_scores = np.random.rand(30) + np.array([0.1] * 10 + [0.5] * 10 + [0.0] * 10)
    region2_scores = np.random.rand(30) + np.array([0.2] * 10 + [0.4] * 10 + [-0.1] * 10)

    dummy_df_wide = pd.DataFrame({
        'Subject_ID': subjects,
        'Site_ID': sites,
        'Age': age,
        'R001': region1_scores,
        'R002': region2_scores
    })

    print("--- Original Data Head (showing Site bias) ---")
    print(dummy_df_wide[['Site_ID', 'R001', 'R002']].head(15))

    # Use the generic function
    harmonized_df = harmonize_dataset(
        df_wide=dummy_df_wide,
        feature_cols=['R001', 'R002'],
        site_col='Site_ID',
        covariate_cols=['Age']  # Preserve the Age effect
    )

    print("\n--- Harmonized Data Head (bias removed) ---")
    # Visually inspect that the mean scores across sites look more similar now
    print(harmonized_df[['Site_ID', 'R001', 'R002']].head(15))

    print("\n--- Mean values per site (Original vs Harmonized) ---")
    original_means = dummy_df_wide.groupby('Site_ID')[['R001', 'R002']].mean()
    harmonized_means = harmonized_df.groupby('Site_ID')[['R001', 'R002']].mean()
    print("Original Means:\n", original_means)
    print("Harmonized Means:\n", harmonized_means)  # These means should be much closer together

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------EOF