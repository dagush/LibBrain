# --------------------------------------------------------------------------------------
# Generic utilities to build long- and wide-format pandas DataFrames
# from per-entity observable files.
#
# The module is agnostic to:
# - domain (neuroscience, finance, physics, ...)
# - number of observables
# - scalar vs vector observables
#
# By Gustavo Patow
# --------------------------------------------------------------------------------------
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Union

import pandas as pd
import numpy as np



# ----------------------------
# Core transformation logic
# ----------------------------
def as_dict_iter(data, nested):
    if nested:
        yield from data.items()   # (outer_key, inner_dict)
    else:
        yield None, data          # single “anonymous” dict

def observables_to_long_dataframe(
    entity_id: str,
    observables: Dict[str, Union[float, np.ndarray]],
    metadata: Optional[Dict[str, Union[str, int, float]]] = None,
    index_name: str = "parcel",
    use_RSN: bool = False,
  ) -> pd.DataFrame:
    """
    Convert a dictionary of observables into a long-format DataFrame.

    Parameters
    ----------
    entity_id : str
        Identifier of the entity (e.g., subject ID).
    observables : dict
        Mapping observable_name -> scalar or vector.
    metadata : dict, optional
        Additional metadata to attach to every row.
    index_name : str
        Name of the index dimension (e.g., parcel, channel).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame.
    """
    def process_single_row(obs, RSN):
        rows = []
        for obs_name, value in obs.items():
            if np.isscalar(value):
                rows.append({
                    "id": entity_id,
                    "observable": obs_name,
                    index_name: None,
                    "value": value,
                    "RSN": RSN,
                })
            else:
                for idx, v in enumerate(value):
                    # TODO: correct the idx number in the case of RSNs to have the right global parcel number
                    rows.append({
                        "id": entity_id,
                        "observable": obs_name,
                        index_name: idx,
                        "value": v,
                        "RSN": RSN,
                    })
        return rows

    rows = [
        process_single_row(row, key)
        for key, row in as_dict_iter(observables, use_RSN)
    ] if use_RSN else process_single_row(observables, 'Whole-Brain')

    if use_RSN:
        all_rows = [
            x
            for row in rows
            for x in row
        ]
    else:
        all_rows = rows
    df = pd.DataFrame(all_rows)

    if metadata:
        for key, val in metadata.items():
            df[key] = val

    return df


# ----------------------------
# Batch processing utilities
# ----------------------------

def build_long_dataframe_from_entities(
    entity_ids: Iterable[str],
    observable_loader: Callable[[str], Dict[str, Union[float, np.ndarray]]],
    metadata_loader: Optional[Callable[[str], Dict[str, Union[str, int, float]]]] = None,
    index_name: str = "parcel",
    use_RSN: bool = False,
  ) -> pd.DataFrame:
    """
    Build a long-format DataFrame from multiple entities.

    Parameters
    ----------
    entity_ids : iterable of str
        IDs of entities to process.
    observable_loader : callable
        Function(entity_id) -> observables dict.
    metadata_loader : callable, optional
        Function(entity_id) -> metadata dict.
    index_name : str
        Name for vector index dimension.

    Returns
    -------
    pd.DataFrame
    """

    # TODO: To have the right numbering of the parcels for the "region" case (i.e., RSNs), here
    # we should load the RSN indexes file, and convert the parcel number to the correct index
    # in the parcellation we are using...

    dfs = []

    for eid in entity_ids:
        observables = observable_loader(eid)
        metadata = metadata_loader(eid) if metadata_loader else None

        df = observables_to_long_dataframe(
            entity_id=eid,
            observables=observables,
            metadata=metadata,
            index_name=index_name,
            use_RSN=use_RSN,
        )
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    return final_df


# ----------------------------
# Format conversions
# ----------------------------

def long_to_wide(
        df: pd.DataFrame,
        label: str,
        metadata: list[str],
        id_col: str = 'id',
        columns_name: str = "parcel",
        values_name: str = "value",
        observable_col: str = "observable",
    ) -> pd.DataFrame:
    """
    Pivot a long-format dataframe to wide format, for a given observable (should be
    generalized for all observables in the dataframe)
    """
    widened = df[df[observable_col]==label].pivot_table(
        index=[id_col] + metadata,
        columns=[columns_name],
        values=values_name
    ).reset_index()
    return widened


def wide_to_long(
    df: pd.DataFrame,
    # -------------- input cols
    entity_col: str = "id",
    index_col: str = "parcel",
    metadata: Iterable[str] = [],
    # -------------- output cols
    observable_col: str = "observable",
    obs_name: str = "observable",
    value_col: str = "value",
  ) -> pd.DataFrame:
    """
    Convert a wide-format DataFrame into a long-format DataFrame,
    supporting both global and indexed (vector) observables, for
    ONE specific observable (should be generalized).

    Wide-format assumptions
    -----------------------
    - One row per entity
    - Columns follow either:
        * <observable>_<index>  (vector observables)
        * <observable>          (global observables)

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame.
    # -------------- input cols
    entity_col: str = "id",
        Name of entity identifier column.
    index_col: str = "parcel",
        Name of index identifier column.
    metadata: Iterable[str] = [],
        Additional metadata to attach to every row.
    # -------------- output cols
    observable_col: str = "observable",
        Name of observable identifier column.
    obs_name: str = "observable",
        Name of observable itself.
    value_col: str = "value",
        Name of value column.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame.
    """
    rows = []

    # vector_pattern = re.compile(rf"(.+){sep}(\d+)$")  # this will probably be needed for the generalization...

    for _, row in df.iterrows():
        entity_id = row[entity_col]

        for col, val in row.items():
            if col == entity_col or col in metadata:
                continue

            # match = vector_pattern.match(col)  # same...
            # if match:
            #     obs_name = match.group(1)
            #     idx = int(match.group(2))
            # else:
            #     obs_name = col
            #     idx = None

            rows.append({
                entity_col: entity_id,
                observable_col: obs_name,
                index_col: col,
                value_col: val
            } | {m: row[m] for m in metadata})

    return pd.DataFrame(rows)


# ----------------------------
# IO
# ----------------------------

def save_dataframe(
        df: pd.DataFrame,
        path: Union[str, Path]
  ) -> None:
    """
    Save DataFrame to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix in (".pkl", ".pickle"):
        df.to_pickle(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load DataFrame from disk.
    """
    path = Path(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

# --------------------------------------------------------------------------------------
# --- Example Usage (Demonstration with dummy data) ---
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Create a dummy wide DataFrame for demonstration
    def make_dummy_long_dataframe():
        """
        Create a dummy long-format DataFrame with:
        - 2 subjects
        - 2 regional observables (3 regions)
        - 2 global observables
        """
        rows = []

        subjects = ["sub-01", "sub-02"]
        regions = 3

        for sid in subjects:
            # Global observables
            rows.append({
                "entity_id": sid,
                "observable": "age",
                "index": None,
                "value": 70 if sid == "sub-01" else 65
            })

            rows.append({
                "entity_id": sid,
                "observable": "mean_fc",
                "index": None,
                "value": 0.42 if sid == "sub-01" else 0.37
            })

            # Regional observables
            for r in range(regions):
                rows.append({
                    "entity_id": sid,
                    "observable": "fc",
                    "index": r,
                    "value": np.round(0.1 * r + (0.01 if sid == "sub-01" else 0.02), 3)
                })

                rows.append({
                    "entity_id": sid,
                    "observable": "alff",
                    "index": r,
                    "value": np.round(1.0 + r + (0.1 if sid == "sub-01" else 0.2), 3)
                })

        return pd.DataFrame(rows)

    # -------------------------------
    # Create dummy long dataframe
    df_long_original = make_dummy_long_dataframe()
    # Now, we should test conversions between wide and long formats, and back...

    print("done!")
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------EOF