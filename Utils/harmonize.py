# --------------------------------------------------------------------------------------
# Generic utilities to harmonize DataFrames with subjects comming from diferent sites
#
# By Gustavo Patow
# --------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from neuroHarmonize import harmonizationLearn, harmonizationApply
from typing import List, Optional


def check_sites(df: pd.DataFrame,
                 site_col: str,):
    counts = df[site_col].value_counts()
    return counts


def harmonize_dataset(
        df_wide: pd.DataFrame,
        feature_cols: List[str],
        site_col: str,
        covariate_cols: Optional[List[str]] = None
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

    Returns:
    - pd.DataFrame: A new DataFrame with the harmonized features replacing the originals,
                    while keeping all original metadata columns intact.
    """

    # 1. Input Validation and Setup
    if site_col not in df_wide.columns:
        raise ValueError(f"Site column '{site_col}' not found in DataFrame.")

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

    # Prepare data matrix and covariates matrix
    data_to_harmonize = df_final[feature_cols]
    covariates_df = df_final[all_covariate_cols]

    print(f"Starting harmonization for {len(feature_cols)} features and {len(df_final)} subjects...")
    print(f"Site column used: '{site_col}'")
    if covariate_cols:
        print(f"Covariates preserved: {covariate_cols}")

    covariates_df = covariates_df.rename(columns={site_col: 'SITE'})

    # 2. Perform ComBat harmonization
    model, harmonized_data_np = harmonizationLearn(
        data_to_harmonize.values,
        covariates_df,
    )
    print("Harmonization complete.")

    # 3. Reconstruct the output DataFrame
    # Create a copy of the original DF to avoid modifying the input DF in place
    df_harmonized = df_final.copy()

    # Replace the original feature columns with the new harmonized values
    df_harmonized[feature_cols] = harmonized_data_np

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