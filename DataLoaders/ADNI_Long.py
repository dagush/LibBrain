"""
ADNI_Long data loader
=====================

Loader for the parcellated ADNI3 longitudinal dataset:

    ADNI-Long/ADNI3_LONG_2VISITS_ALLIMGS/

The loader assumes the Section 6.2 layout described in the dataset document:
sub-*/ses-M*/{anat,pet,func}/ parcellated .npy files, plus
sub-*/ses-longitudinal/anat/ GMvolAPC files and participants.tsv.

Default choices:
    - parcellation: Schaefer400
    - amyloid: CL + PVC when available
    - tau: PVC when available
    - fMRI: fmri_prepro_denoised_bp_008_08
    - GMV template: template_all
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from DataLoaders.baseDataLoader import DataLoader


class ADNI_Long(DataLoader):
    """
    DataLoader for ADNI-Long / ADNI3_LONG_2VISITS_ALLIMGS.

    Data are organized as:
        self.timeseries[subject][session]
        self.abeta[subject][session]
        self.tau[subject][session]
        self.gmvol[subject][session]
        self.gmvol_apc[subject]
        self.subject_metadata[subject]
        self.session_metadata[subject]

    The main public method `get_subjectData(subjectID)` returns all available data
    for one subject in a single dictionary.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        parcellation: str = "Schaefer400",
        template_name: str = "template_all",
        fmri_deriv_name: str = "fmri_prepro_denoised_bp_008_08",
        prefer_cl: bool = True,
        prefer_pvc: bool = True,
        load_data: bool = True,
        strict: bool = False,
        verbose: bool = True,
    ):
        self.parcellation = parcellation
        self.template_name = template_name
        self.fmri_deriv_name = fmri_deriv_name
        self.prefer_cl = prefer_cl
        self.prefer_pvc = prefer_pvc
        self.strict = strict
        self.verbose = verbose

        self.groups = ["CN", "MCI", "AD"]

        self.subjects: List[str] = []
        self.sessions: Dict[str, List[str]] = {}

        self.timeseries: Dict[str, Dict[str, np.ndarray]] = {}
        self.abeta: Dict[str, Dict[str, np.ndarray]] = {}
        self.tau: Dict[str, Dict[str, np.ndarray]] = {}
        self.gmvol: Dict[str, Dict[str, np.ndarray]] = {}
        self.gmvol_apc: Dict[str, np.ndarray] = {}

        self.subject_metadata: Dict[str, Dict[str, Any]] = {}
        self.session_metadata: Dict[str, pd.DataFrame] = {}

        self.missing_files: List[Dict[str, str]] = []

        self.set_basePath(path)
        self._discover_subjects_and_sessions()
        self._load_metadata()

        if load_data:
            self._load_all_subjects()

        if self.verbose:
            print(self.summary())

    # ------------------------------------------------------------------
    # Basic dataset information
    # ------------------------------------------------------------------
    def name(self) -> str:
        return "ADNI_Long"

    def TR(self) -> float:
        return 3.0

    def N(self) -> int:
        m = re.search(r"(\d+)$", self.parcellation)
        if m is not None:
            return int(m.group(1))
        if self.parcellation.lower() == "glasser":
            return 360
        if self.parcellation.upper() == "DBS80":
            return 80
        return -1

    def get_groupLabels(self) -> List[str]:
        return self.groups

    # ------------------------------------------------------------------
    # Paths and discovery
    # ------------------------------------------------------------------
    def set_basePath(self, path: Optional[str]) -> None:
        if path is None:
            try:
                from DataLoaders.WorkBrainFolder import WorkBrainDataFolder
                root = Path(WorkBrainDataFolder)
            except Exception:
                root = Path(".")
        else:
            root = Path(path)

        # Accept either the parent folder or the dataset folder itself.
        if root.name == "ADNI3_LONG_2VISITS_ALLIMGS":
            self.base_folder = root
        else:
            self.base_folder = root / "ADNI-Long" / "ADNI3_LONG_2VISITS_ALLIMGS"

        self.participants_tsv = self.base_folder / "participants.tsv"
        self.anat_templates_folder = self.base_folder / "anat_templates"

    def _discover_subjects_and_sessions(self) -> None:
        subject_dirs = sorted(self.base_folder.glob("sub-*"))
        self.subjects = [p.name for p in subject_dirs if p.is_dir()]

        for sub in self.subjects:
            sub_dir = self.base_folder / sub
            ses_dirs = sorted(
                p.name for p in sub_dir.glob("ses-M*")
                if p.is_dir() and p.name != "ses-longitudinal"
            )
            self.sessions[sub] = ses_dirs

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _load_metadata(self) -> None:
        self.participants = None
        if self.participants_tsv.exists():
            self.participants = pd.read_csv(self.participants_tsv, sep="\t")
            id_col = self._guess_id_column(self.participants)
            if id_col is not None:
                for _, row in self.participants.iterrows():
                    sid = str(row[id_col])
                    if not sid.startswith("sub-"):
                        sid = f"sub-{sid}"
                    self.subject_metadata[sid] = row.to_dict()

        for sub in self.subjects:
            f = self.base_folder / sub / f"{sub}_sessions.tsv"
            if f.exists():
                self.session_metadata[sub] = pd.read_csv(f, sep="\t")

    @staticmethod
    def _guess_id_column(df: pd.DataFrame) -> Optional[str]:
        candidates = ["participant_id", "subject_id", "sub", "PTID", "RID"]
        for c in candidates:
            if c in df.columns:
                return c
        return df.columns[0] if len(df.columns) else None

    # ------------------------------------------------------------------
    # File selection helpers
    # ------------------------------------------------------------------
    def _record_missing(self, sub: str, ses: str, modality: str, pattern: str) -> None:
        self.missing_files.append({
            "subject": sub,
            "session": ses,
            "modality": modality,
            "pattern": pattern,
        })
        if self.strict:
            raise FileNotFoundError(f"Missing {modality} file for {sub}/{ses}: {pattern}")

    @staticmethod
    def _load_npy(path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None:
            return None
        return np.load(path)

    @staticmethod
    def _select_one(files: List[Path]) -> Optional[Path]:
        if len(files) == 0:
            return None
        if len(files) > 1:
            # Deterministic behaviour; exact patterns should normally return one.
            return sorted(files)[0]
        return files[0]

    def _find_func_file(self, sub: str, ses: str) -> Optional[Path]:
        folder = self.base_folder / sub / ses / "func"
        pattern = f"{sub}_{ses}_task-rest_bold_{self.parcellation}_{self.fmri_deriv_name}.npy"
        return self._select_one(list(folder.glob(pattern)))

    def _find_gmvol_file(self, sub: str, ses: str) -> Optional[Path]:
        folder = self.base_folder / sub / ses / "anat"
        pattern = f"{sub}_{ses}_GMvol_{self.parcellation}_{self.template_name}.npy"
        return self._select_one(list(folder.glob(pattern)))

    def _find_gmvol_apc_file(self, sub: str) -> Optional[Path]:
        ses = "ses-longitudinal"
        folder = self.base_folder / sub / ses / "anat"
        pattern = f"{sub}_{ses}_GMvolAPC_{self.parcellation}_{self.template_name}.npy"
        return self._select_one(list(folder.glob(pattern)))

    def _find_abeta_file(self, sub: str, ses: str) -> Optional[Path]:
        """Prefer amyloid CL + PVC files when available."""
        folder = self.base_folder / sub / ses / "pet"
        cl = "_CL" if self.prefer_cl else ""
        pvc = "_pvc" if self.prefer_pvc else ""

        patterns = []
        # Highest priority: requested CL/PVC setting.
        patterns.append(f"{sub}_{ses}_trc-*_pet_{self.parcellation}{cl}{pvc}.npy")
        # Fallbacks, still preferring CL first, PVC second.
        if self.prefer_cl and self.prefer_pvc:
            patterns += [
                f"{sub}_{ses}_trc-*_pet_{self.parcellation}_CL.npy",
                f"{sub}_{ses}_trc-*_pet_{self.parcellation}_pvc.npy",
                f"{sub}_{ses}_trc-*_pet_{self.parcellation}.npy",
            ]
        elif self.prefer_cl:
            patterns += [
                f"{sub}_{ses}_trc-*_pet_{self.parcellation}_CL_pvc.npy",
                f"{sub}_{ses}_trc-*_pet_{self.parcellation}.npy",
            ]
        elif self.prefer_pvc:
            patterns += [
                f"{sub}_{ses}_trc-*_pet_{self.parcellation}_CL_pvc.npy",
                f"{sub}_{ses}_trc-*_pet_{self.parcellation}.npy",
            ]

        # Amyloid tracers in this dataset are FBP/FBB; exclude tau tracer explicitly.
        for pat in patterns:
            files = [p for p in folder.glob(pat) if "18FAV1451" not in p.name]
            selected = self._select_one(files)
            if selected is not None:
                return selected
        return None

    def _find_tau_file(self, sub: str, ses: str) -> Optional[Path]:
        """Prefer tau PVC files when available. Tau is stored as SUVR in the example."""
        folder = self.base_folder / sub / ses / "pet"
        pvc = "_pvc" if self.prefer_pvc else ""
        patterns = [
            f"{sub}_{ses}_trc-18FAV1451_pet_{self.parcellation}{pvc}.npy",
            f"{sub}_{ses}_trc-18FAV1451_pet_{self.parcellation}.npy",
        ]
        if not self.prefer_pvc:
            patterns.append(f"{sub}_{ses}_trc-18FAV1451_pet_{self.parcellation}_pvc.npy")

        for pat in patterns:
            selected = self._select_one(list(folder.glob(pat)))
            if selected is not None:
                return selected
        return None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_all_subjects(self) -> None:
        for sub in self.subjects:
            self.timeseries[sub] = {}
            self.abeta[sub] = {}
            self.tau[sub] = {}
            self.gmvol[sub] = {}

            for ses in self.sessions[sub]:
                self._load_subject_session(sub, ses)

            gmvol_apc_file = self._find_gmvol_apc_file(sub)
            if gmvol_apc_file is None:
                self._record_missing(sub, "ses-longitudinal", "GMvolAPC", "GMvolAPC")
            else:
                self.gmvol_apc[sub] = np.load(gmvol_apc_file)

    def _load_subject_session(self, sub: str, ses: str) -> None:
        loaders = {
            "timeseries": (self._find_func_file, self.timeseries),
            "ABeta": (self._find_abeta_file, self.abeta),
            "Tau": (self._find_tau_file, self.tau),
            "GMvol": (self._find_gmvol_file, self.gmvol),
        }

        for label, (finder, target) in loaders.items():
            f = finder(sub, ses)
            if f is None:
                self._record_missing(sub, ses, label, label)
            else:
                target[sub][ses] = np.load(f)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_subjects(self) -> List[str]:
        return list(self.subjects)

    def get_sessions(self, subjectID: str) -> List[str]:
        return list(self.sessions.get(subjectID, []))

    def get_subject_count(self) -> int:
        return len(self.subjects)

    def get_classification(self) -> Dict[str, str]:
        """
        Best-effort classification from participants.tsv.
        Returns subject -> diagnostic label when a suitable column is found.
        """
        candidates = [
            "diagnosis_abpos_ses0", "diagnosis_ses0", "DX_bl",
            "diagnosis", "group", "Group", "DX",
        ]
        out = {}
        for sub, meta in self.subject_metadata.items():
            for c in candidates:
                if c in meta and pd.notna(meta[c]):
                    out[sub] = str(meta[c])
                    break
        return out

    def get_subjectData(self, subjectID: str) -> Dict[str, Any]:
        if subjectID not in self.subjects:
            raise KeyError(f"Unknown subjectID: {subjectID}")

        data = {
            subjectID: {
                "sessions": {},
                "longitudinal": {},
                "meta": self.subject_metadata.get(subjectID, {}),
                "session_meta": self.session_metadata.get(subjectID, None),
            }
        }

        for ses in self.sessions[subjectID]:
            data[subjectID]["sessions"][ses] = {
                "timeseries": self.timeseries.get(subjectID, {}).get(ses),
                "ABeta": self.abeta.get(subjectID, {}).get(ses),
                "Tau": self.tau.get(subjectID, {}).get(ses),
                "GMvol": self.gmvol.get(subjectID, {}).get(ses),
            }

        data[subjectID]["longitudinal"] = {
            "GMvolAPC": self.gmvol_apc.get(subjectID)
        }

        return data

    def get_available_modalities(self, subjectID: str, session: str) -> Dict[str, bool]:
        return {
            "timeseries": session in self.timeseries.get(subjectID, {}),
            "ABeta": session in self.abeta.get(subjectID, {}),
            "Tau": session in self.tau.get(subjectID, {}),
            "GMvol": session in self.gmvol.get(subjectID, {}),
            "GMvolAPC": subjectID in self.gmvol_apc,
        }

    def summary(self) -> str:
        n_sessions = sum(len(v) for v in self.sessions.values())
        n_ts = sum(len(v) for v in self.timeseries.values())
        n_ab = sum(len(v) for v in self.abeta.values())
        n_tau = sum(len(v) for v in self.tau.values())
        n_gmv = sum(len(v) for v in self.gmvol.values())
        n_apc = len(self.gmvol_apc)

        return (
            f"{self.name()} | subjects={len(self.subjects)}, sessions={n_sessions}, "
            f"parcellation={self.parcellation}, template={self.template_name}\n"
            f"Loaded: fMRI={n_ts}, ABeta={n_ab}, Tau={n_tau}, GMvol={n_gmv}, GMvolAPC={n_apc}. "
            f"Missing entries={len(self.missing_files)}"
        )

# ================================================================================================================
print('Done loading Raw Data!')
# =========================  debug
if __name__ == '__main__':
    # ---- test DBS 80
    baseDL = ADNI_Long(parcellation='DBS80')  # DBS80, glasser, Schaefer 100, 200, 400, 600 and 1000
    sujes = baseDL.get_classification()
    gCtrl = baseDL.get_groupSubjects('CN')
    s1 = baseDL.get_subjectData(gCtrl[0])
    # avg_SC = baseDL.get_AvgSC_ctrl()
    gAD = baseDL.get_groupSubjects('AD')
    s2 = baseDL.get_subjectData(gAD[0])
    print('done ADNI-Long! ;-)')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF