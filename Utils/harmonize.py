# --------------------------------------------------------------------------------------
# Generic utilities to harmonize DataFrames with subjects coming from different sites
#
# By Gustavo Patow
# --------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from neuroHarmonize import harmonizationLearn, harmonizationApply
from typing import List, Optional

import Utils.dataframe_builder as dfb

from Utils.harmonization_sanity_check import sanity_check_long_df


# --------------------------------------------------------------------------------------
# Convenience analysis and reporting methods
# --------------------------------------------------------------------------------------
def check_sites_l(df_l: pd.DataFrame,
                  id_col: str = 'id',
                  site_col: str = 'site',):
    # subjects_per_site = (
    #     df_l
    #     .drop_duplicates(subset=[id_col, site_col])
    #     .groupby(site_col)
    #     .size()
    #     .rename("n_subjects")
    #     .reset_index()
    # )
    subjects_per_site = (
        df_l.groupby(site_col)[id_col].nunique().reset_index(name="n_subjects")
    )
    return subjects_per_site


def check_sites_w(df_w: pd.DataFrame,
                  site_col: str = 'site',):
    counts = df_w[site_col].value_counts()
    return counts


def report_sites(df_w, DL):
    groups = DL.get_groupLabels()
    subjects = DL.get_classification()
    site_counts = check_sites_w(df_w, 'site')
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
        df_l: pd.DataFrame,  # DataFrame in Long format
        id_col: str,
        obs_name: str,
        site_col: str,
        region_col: str,
        metadata: List[str],
        harmonization_level: str = "global",  # "global" or "parcel"
) -> pd.DataFrame:  # returns DataFrame in Lonf format
    """
    Applies ComBat harmonization to a long-format DataFrame of features.

    Parameters:
    - df_l (pd.DataFrame): Input DataFrame in long format
    - id_col (str): Column name identifying ID column in df_l
    - obs_name (str): Column name identifying observable column in df_l
    - site_col (str): The column name identifying the acquisition site/batch ID
                      for each subject.
    - metadata (List[str]): List of column names corresponding to
                            biological/clinical covariates (e.g., age, sex)
                            to preserve during harmonization + site_col
    - harmonization_level:
                    "global" (default): estimate a single site effect per subject and
                    apply it uniformly to all features.
                    "parcel": estimate region-specific site effects (ComBat default behavior).
                    "region: RSN-specific site effects.

    Returns:
    - pd.DataFrame: A new DataFrame with the harmonized features replacing the originals,
                    while keeping all original metadata columns intact, in long format.
    """
    # --------------------------------------
    # 1. Input validation and setup
    if site_col not in df_l.columns:
        raise ValueError(f"Site column '{site_col}' not found in DataFrame.")

    if harmonization_level not in ["global", "parcel", "region"]:
        raise ValueError("harmonization_level must be 'global' or 'parcel'")

    for col in metadata:
        if col not in df_l.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    site_per_subject = df_l.groupby(id_col)[site_col].nunique()
    if site_per_subject.max() > 1:
        raise ValueError("Some subjects appear in more than one site!")

    # Ensure site column is treated as a category
    df_l[site_col] = df_l[site_col].astype('category')

    # keep only the rows we are interested in...
    df_l = df_l[df_l["observable"] == obs_name]

    # Remove the sites with only 1 subject
    valid_sites = (
        df_l.groupby(site_col)[id_col]
        .nunique()
        .loc[lambda s: s >= 2]
        .index
    )
    df_l = df_l[df_l["site"].isin(valid_sites)]

    # --------------------------------------
    # 2. Arrange features and covariates
    if harmonization_level == "parcel" or harmonization_level == "global":
        feature_cols = df_l['parcel'].unique()
    elif harmonization_level == "region":
        feature_cols = df_l[region_col].unique()
    print(f"Starting harmonization for {len(feature_cols)} features and {len(df_l[id_col])} subjects...")
    print(f"Site column used: '{site_col}'")

    covariate_cols = [i for i in metadata if i != site_col]
    if covariate_cols:
        print(f"Covariates preserved: {covariate_cols}")

    # --------------------------------------
    # 3. Prepare data matrix and covariates matrix
    if harmonization_level == "parcel" or harmonization_level == "global":
        df_wide = df_l.pivot_table(
            index=[id_col] + metadata,
            columns=["parcel"],
            values="value"
        ).reset_index()

        if harmonization_level == "parcel":
            # Parcel-based harmonization:
            data_to_harmonize = df_wide[feature_cols].values
        elif harmonization_level == "global":
            # True global harmonization:
            # Collapse regions into a single global summary per subject
            global_signal = df_wide[feature_cols].mean(axis=1).values.reshape(-1, 1)
            data_to_harmonize = global_signal
    elif harmonization_level == "region":
        # Regional (e.g., RSN-based) harmonization:
        # Aggregate parcel-level values → region-level means
        region_means = (
            df_l
            .groupby([id_col, region_col], as_index=False)
            .agg(region_value=("value", "mean"))
        )
        # Pivot to wide format
        df_wide = (
            region_means
            .pivot(index=id_col, columns=region_col, values="region_value")
            .reset_index()
        )
        # Add patient-level metadata (site, age, etc.)
        patient_info = (
            df_l[[id_col] + metadata]
            .drop_duplicates()
        )
        df_wide = df_wide.merge(patient_info, on=id_col, how="left")
        data_to_harmonize = df_wide[feature_cols].values
    else:
        raise ValueError(f"Unknown harmonization level '{harmonization_level}'")

    # remove NaNs, just in case...
    df_wide = remove_NaN(df_wide)

    # --------------------------------------
    # 2. Arrange covariates
    covariates_df = df_wide[metadata]
    covariates_df = covariates_df.rename(columns={site_col: 'SITE'})
    # Convert categorical covariates to numeric (required by neuroHarmonize)
    covariates_df = pd.get_dummies(
        covariates_df,
        columns=[col for col in covariate_cols if covariates_df[col].dtype == object],
        drop_first=True
    )

    # --------------------------------------
    # 3. Sanity checks...
    sanity_check_long_df(
        df_l,
        id_col=id_col,
        site_col=site_col,
        region_col=region_col,
        metadata=metadata,
        obs_name=obs_name,
        harmonization_level=harmonization_level,
    )

    # -----------------------------------------------------
    # 4. Perform ComBat harmonization
    model, harmonized_data_np = harmonizationLearn(
        data_to_harmonize,
        covariates_df,
    )
    print("Harmonization complete.")

    # -----------------------------------------------------
    # 5. Reconstruct the output DataFrame
    # Create a copy of the original DF to avoid modifying the input DF in place
    df_harmonized = df_wide.copy()

    # Replace the original feature columns with the new harmonized values
    if harmonization_level == "parcel" or harmonization_level == "global":
        if harmonization_level == "parcel":
            # -------------------------------------------------
            # Replace with globally-corrected data
            df_harmonized[feature_cols] = harmonized_data_np
        elif harmonization_level == "global":
            # -------------------------------------------------
            # Compute per-subject correction factor
            original_global = global_signal.flatten()
            harmonized_global = harmonized_data_np.flatten()
            # Avoid division by zero
            eps = 1e-8
            scaling = harmonized_global / (original_global + eps)
            # Apply the same correction to all regions
            df_harmonized[feature_cols] = df_wide[feature_cols].values * scaling[:, None]
        obs_l = dfb.wide_to_long(df_harmonized, metadata=metadata, obs_name=obs_name)
    elif harmonization_level == "region":
        # -------------------------------------------------
        # RSN-level harmonization → parcel-level projection
        eps = 1e-8
        # Original and harmonized RSN values
        original_region = df_wide[feature_cols].values
        harmonized_region = harmonized_data_np
        # Subject × RSN scaling factors
        region_scaling = harmonized_region / (original_region + eps)
        # Start from original long dataframe (parcel-level)
        df_parcel = (
            df_l[[id_col, "parcel", region_col, "value"] + metadata]
            .drop_duplicates()
            .copy()
        )
        # Map subject → row index in df_wide
        subj_index = {sid: i for i, sid in enumerate(df_wide[id_col].values)}
        region_index = {r: j for j, r in enumerate(feature_cols)}
        # Apply RSN scaling to each parcel
        def apply_region_scaling(row):
            i = subj_index[row[id_col]]
            j = region_index[row[region_col]]
            return row["value"] * region_scaling[i, j]

        df_parcel["value"] = df_parcel.apply(apply_region_scaling, axis=1)
        obs_l = df_parcel

    print("Harmonized values integrated back into DataFrame.")

    # --------------------------------------
    # Return final df, in Long format
    return obs_l


# --------------------------------------------------------------------------------------
# --- Example Usage (Demonstration with dummy data) ---
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    print("Running harmonization...")
    # # Create a dummy wide DataFrame for demonstration
    # np.random.seed(42)
    # sites = ['SiteA'] * 10 + ['SiteB'] * 10 + ['SiteC'] * 10
    # subjects = [f'Sub{i:02d}' for i in range(30)]
    # age = np.random.randint(20, 50, 30)
    # # Simulate site bias: SiteB has slightly higher values overall, SiteC slightly lower
    # region1_scores = np.random.rand(30) + np.array([0.1] * 10 + [0.5] * 10 + [0.0] * 10)
    # region2_scores = np.random.rand(30) + np.array([0.2] * 10 + [0.4] * 10 + [-0.1] * 10)
    #
    # dummy_df_wide = pd.DataFrame({
    #     'Subject_ID': subjects,
    #     'Site_ID': sites,
    #     'Age': age,
    #     'R001': region1_scores,
    #     'R002': region2_scores
    # })
    #
    # print("--- Original Data Head (showing Site bias) ---")
    # print(dummy_df_wide[['Site_ID', 'R001', 'R002']].head(15))
    #
    # # Use the generic function
    # harmonized_df = harmonize_dataset(
    #     df_wide=dummy_df_wide,
    #     feature_cols=['R001', 'R002'],
    #     site_col='Site_ID',
    #     covariate_cols=['Age']  # Preserve the Age effect
    # )
    #
    # print("\n--- Harmonized Data Head (bias removed) ---")
    # # Visually inspect that the mean scores across sites look more similar now
    # print(harmonized_df[['Site_ID', 'R001', 'R002']].head(15))
    #
    # print("\n--- Mean values per site (Original vs Harmonized) ---")
    # original_means = dummy_df_wide.groupby('Site_ID')[['R001', 'R002']].mean()
    # harmonized_means = harmonized_df.groupby('Site_ID')[['R001', 'R002']].mean()
    # print("Original Means:\n", original_means)
    # print("Harmonized Means:\n", harmonized_means)  # These means should be much closer together

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------EOF