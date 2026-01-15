import numpy as np
import pandas as pd

# Usage:
# sanity_check_long_df(
#     df_l,
#     id_col=id_col,
#     site_col=site_col,
#     region_col=region_col,
#     metadata=metadata,
#     obs_name=obs_name,
#     harmonization_level=harmonization_level,
# )

def sanity_check_long_df(
    df: pd.DataFrame,
    *,
    id_col: str,
    site_col: str,
    parcel_col: str = "parcel",
    value_col: str = "value",
    observable_col: str = "observable",
    obs_name: str,
    region_col: str | None = None,
    metadata: list[str] | None = None,
    harmonization_level: str = "parcel",
    min_subjects_per_site: int = 2,
    verbose: bool = True,
):
    """
    Sanity checks for long-format fMRI data prior to harmonization.
    Designed to match the harmonize_dataset() pipeline exactly.
    """

    metadata = metadata or []

    # --------------------------------------------------
    # 1. Column existence
    required_cols = {
        id_col, site_col, parcel_col, value_col, observable_col
    }
    if harmonization_level == "region":
        if region_col is None:
            raise ValueError("region_col must be provided for region harmonization")
        required_cols.add(region_col)

    required_cols |= set(metadata)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if verbose:
        print("✔ Required columns present")

    # --------------------------------------------------
    # 2. Observable filtering sanity
    if obs_name not in df[observable_col].unique():
        raise ValueError(
            f"Observable '{obs_name}' not found in column '{observable_col}'"
        )

    df = df[df[observable_col] == obs_name]

    if df.empty:
        raise ValueError(
            f"No rows remain after filtering observable == '{obs_name}'"
        )

    if verbose:
        print(f"✔ Observable '{obs_name}' present and filtered correctly")

    # --------------------------------------------------
    # 3. Duplicate rows
    if harmonization_level == 'global' or harmonization_level == 'parcel':
        subset = [id_col, parcel_col, observable_col]
    else:
        subset = [id_col, site_col, parcel_col, region_col]
    dup = df.duplicated(
        subset=subset,
    ).sum()
    if dup > 0:
        raise ValueError(
            f"Found {dup} duplicated (subject, parcel, observable) rows"
        )

    if verbose:
        print("✔ No duplicated (subject, parcel, observable) rows")

    # --------------------------------------------------
    # 4. One site per subject
    sites_per_subject = df.groupby(id_col)[site_col].nunique()
    bad = sites_per_subject[sites_per_subject > 1]
    if not bad.empty:
        raise ValueError(
            "Some subjects appear in multiple sites:\n"
            f"{bad}"
        )

    if verbose:
        print("✔ Each subject belongs to exactly one site")

    # --------------------------------------------------
    # 5. Minimum subjects per site
    site_counts = df.groupby(site_col)[id_col].nunique()
    site_counts = site_counts[site_counts >= 1]
    small_sites = site_counts[site_counts < min_subjects_per_site]
    if not small_sites.empty:
        raise ValueError(
            "Sites with insufficient subjects:\n"
            f"{small_sites}"
        )

    if verbose:
        print("✔ All sites have sufficient subjects")

    # --------------------------------------------------
    # 6. Value sanity
    if not np.isfinite(df[value_col]).all():
        raise ValueError("Found NaN or infinite values in data")

    if verbose:
        print("✔ No NaN or Inf values")

    # --------------------------------------------------
    # 7. Parcel coverage consistency
    parcels_per_subject = df.groupby(id_col)[parcel_col].nunique()
    if parcels_per_subject.nunique() != 1:
        raise ValueError(
            "Subjects have inconsistent number of parcels:\n"
            f"{parcels_per_subject.value_counts()}"
        )

    if verbose:
        print("✔ Consistent parcel coverage across subjects")

    # --------------------------------------------------
    # 8. Region (RSN) consistency — only if needed
    if harmonization_level == "region":
        # Right now, I do not have this information in the DataFrame, every RSN starts from 0...
        # parcel_to_region = (
        #     df[[parcel_col, region_col]]
        #     .drop_duplicates()
        #     .groupby(parcel_col)[region_col]
        #     .nunique()
        # )
        # bad_parcels = parcel_to_region[parcel_to_region > 1]
        # if not bad_parcels.empty:
        #     raise ValueError(
        #         "Some parcels belong to multiple regions:\n"
        #         f"{bad_parcels}"
        #     )
        # if verbose:
        #     print("✔ Each parcel maps to exactly one region")

        regions_per_subject = df.groupby(id_col)[region_col].nunique()
        if regions_per_subject.nunique() != 1:
            raise ValueError(
                "Subjects have inconsistent number of regions:\n"
                f"{regions_per_subject.value_counts()}"
            )

        if verbose:
            print("✔ Consistent region coverage across subjects")

    # --------------------------------------------------
    # 9. Variance checks (critical for ComBat)
    parcel_var = (
        df.groupby(parcel_col)[value_col]
        .var()
        .fillna(0)
    )

    zero_var = parcel_var[parcel_var == 0]
    if not zero_var.empty:
        raise ValueError(
            f"Found {len(zero_var)} parcels with zero variance"
        )

    if verbose:
        print("✔ All parcels have non-zero variance")

    # --------------------------------------------------
    # 10. Metadata sanity
    for col in metadata:
        if df[col].isnull().any():
            raise ValueError(f"Metadata column '{col}' contains NaNs")

    if verbose and metadata:
        print("✔ Metadata columns are complete")

    if verbose:
        print("\nSanity check passed ✔\n")

    return True
