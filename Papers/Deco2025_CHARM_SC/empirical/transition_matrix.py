"""
Deco2025_CHARM_SC/empirical/transition_matrix.py
----------------------------------------
Empirical parcel state distribution from BOLD activity events.

Computes Pstatesemp — the empirical probability of each parcel being
active — from filtered BOLD timeseries, following the event-detection
pipeline in Model_subjects.m.

Pipeline (per subject group):
    1. Detrend, bandpass-filter BOLD for each parcel
    2. Detect activity events: timepoints where signal > mean + std
       and where the event starts (rising edge, not sustained activity)
    3. Build an empirical transition matrix Pm2 counting
       "which parcel fires next after parcel i?"
    4. Row-normalise Pm2 → Pmatrixemp
    5. Raise to power 50 → Pmatrixemp^50[0,:] = stationary distribution

Reference:
    Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics
    reveal low-dimensional manifolds of critical brain dynamics.
    Physical Review E, 111(1). https://doi.org/10.1103/physreve.111.014410

    Model_subjects.m — original MATLAB code by Gustavo Deco.
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
from numpy import linalg as LA
from scipy import stats

from neuronumba.tools.filters import BandPassFilter


class EmpiricalTransitionMatrix:
    """
    Computes the empirical parcel state distribution from BOLD timeseries.

    Parameters
    ----------
    tr_seconds : float
        Repetition time in seconds. Used to construct the bandpass filter.
        Default: 0.72 (HCP dataset).
    flp : float
        Low-pass frequency in Hz. Default: 0.008.
    fhi : float
        High-pass frequency in Hz. Default: 0.08.
    cut : int
        Number of timepoints to remove from each end after filtering.
        Default: 50 (matching MATLAB: signal_filt2(50:end-50)).
    diffusion_steps : int
        Power to raise the transition matrix to for stationary distribution.
        Default: 50 (matching MATLAB: Pmatrixemp^50).
    exclude_parcels : list of int or None
        0-indexed parcel indices to exclude from the output.
        Default: [554, 907] (Schaefer1000 NaN parcels).
    """

    def __init__(
        self,
        tr_seconds:      float = 0.72,
        flp:             float = 0.008,
        fhi:             float = 0.08,
        cut:             int   = 50,
        diffusion_steps: int   = 50,
        exclude_parcels: Optional[list[int]] = None,
    ):
        self.tr_seconds      = tr_seconds
        self.flp             = flp
        self.fhi             = fhi
        self.cut             = cut
        self.diffusion_steps = diffusion_steps
        self.exclude_parcels = exclude_parcels if exclude_parcels is not None \
                               else [554, 907]

        # BandPassFilter expects tr in milliseconds
        self._bpf = BandPassFilter(
            k=2,
            tr=tr_seconds * 1000.0,    # seconds → milliseconds
            flp=flp,
            fhi=fhi,
            remove_artifacts=False,    # MATLAB code does not remove artifacts here
            apply_demean=True,
            apply_detrend=True,
        )

    def compute(
        self,
        timeseries_list: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute empirical stationary distribution from a list of subject
        timeseries.

        Corresponds to the inner loop of Model_subjects.m:
            for sub2=sub:sub+NGroup-1
                ... detect events ...
                ... accumulate Pm2 ...
            Pmatrixemp = D^{-1} Pm2
            Pstatesemp = Pmatrixemp^50[0,:]

        Parameters
        ----------
        timeseries_list : list of np.ndarray, each shape (N, T)
            BOLD timeseries for a group of subjects, in Neuroreduce
            convention (N parcels × T timepoints). Raw, unfiltered.

        Returns
        -------
        p_states_emp : np.ndarray, shape (N_valid,)
            Empirical stationary distribution with excluded parcels removed.
        """
        if not timeseries_list:
            raise ValueError("timeseries_list must not be empty.")

        N = timeseries_list[0].shape[0]
        Pm2 = np.zeros((N, N))   # accumulate transition counts across subjects

        for ts in timeseries_list:
            if ts.shape[0] != N:
                raise ValueError(
                    f"All timeseries must have {N} parcels. "
                    f"Got shape {ts.shape}."
                )
            events = self._detect_events(ts)
            Pm2   += self._count_transitions(events)

        # Row-normalise Pm2 → Pmatrixemp
        # Handle parcels with zero row-sum (no events detected):
        # two specific parcels in Schaefer1000 are known to be problematic.
        row_sums = np.sum(Pm2, axis=1)

        # Fix zero row-sums: copy from the previous parcel's row sum.
        # Matches the MATLAB hard-coded fix:
        #   Dmatrix(555,555) = Dmatrix(554,554)
        #   Dmatrix(908,908) = Dmatrix(907,907)
        # These two parcels (0-indexed: 554, 907) have NaN BOLD in the
        # Schaefer 1000 atlas and never generate events.
        for idx in self.exclude_parcels:
            if 0 < idx < N and row_sums[idx] == 0:
                row_sums[idx] = row_sums[idx - 1]

        # Replace any remaining zeros to avoid division by zero
        zero_mask = row_sums == 0
        if zero_mask.any():
            warnings.warn(
                f"{zero_mask.sum()} parcels had no detected events. "
                "Their rows will be set to uniform transition probabilities.",
                RuntimeWarning,
                stacklevel=2,
            )
            row_sums[zero_mask] = N   # uniform fallback

        D            = np.diag(row_sums)
        Pmatrixemp   = LA.inv(D) @ Pm2           # (N, N) row-stochastic

        # Stationary distribution: first row of P^diffusion_steps.
        # See base_geometry.py for explanation of why row 0 is taken.
        P_n          = LA.matrix_power(Pmatrixemp, self.diffusion_steps)
        p_states     = P_n[0, :]                 # (N,)

        # Remove excluded parcels
        return self._remove_excluded(p_states, N)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _detect_events(self, ts: np.ndarray) -> np.ndarray:
        """
        Detect activity onset events from a BOLD timeseries.

        Events are timepoints where:
            1. The filtered signal exceeds mean + std  (ev1)
            2. The previous timepoint did NOT exceed this threshold  (rising edge)
        This detects the START of sustained activations, not sustained activity.

        Matches the MATLAB logic:
            ev1 = tss > std(tss) + mean(tss);   % above threshold
            ev2 = [0 ev1(1:end-1)];              % shifted by one
            events(seed,:) = (ev1 - ev2) > 0;   % rising edge only

        Parameters
        ----------
        ts : np.ndarray, shape (N, T)
            Raw BOLD timeseries, Neuroreduce (N, T) convention.

        Returns
        -------
        events : np.ndarray, shape (N, T_trimmed), dtype bool
            Binary event matrix. events[i,t] = True if parcel i has an
            activity onset at timepoint t.
        """
        N, T = ts.shape

        # BandPassFilter expects (T, N) — transpose in, transpose out
        signal_filt = self._bpf.filter(ts.T).T   # (N, T)

        # Trim cut timepoints from each end (MATLAB: signal_filt2(50:end-50))
        # Assumption: 0-indexed slice [cut-1 : T-cut] matches MATLAB's
        # 1-indexed (50:end-50) which gives indices 50..T-50 inclusive.
        signal_trimmed = signal_filt[:, self.cut - 1: T - self.cut]  # (N, T')
        T_trimmed = signal_trimmed.shape[1]

        events = np.zeros((N, T_trimmed), dtype=bool)
        for seed in range(N):
            tss  = signal_trimmed[seed, :]
            # Threshold: mean + std (MATLAB: tss > std(tss) + mean(tss))
            threshold = np.mean(tss) + np.std(tss)
            ev1  = tss > threshold                           # above threshold
            ev2  = np.concatenate([[False], ev1[:-1]])       # shifted by 1
            events[seed, :] = (ev1.astype(int) - ev2.astype(int)) > 0

        return events

    def _count_transitions(self, events: np.ndarray) -> np.ndarray:
        """
        Build a transition count matrix from a binary event matrix.

        For each event at parcel i at time t, find the NEXT timepoint
        t2 > t where any parcel fires. Count a transition i → j for
        each firing parcel j at t2.

        If no parcel fires after t, count a self-transition i → i.

        Matches the MATLAB double loop:
            for seed=1:N
                lisev = find(events(seed,:)==1)
                for t=lisev
                    for t2=t+1:end
                        lista = find(events(:,t2)==1)
                        if isempty(lista): Pm2(seed,seed)++
                        else: Pm2(seed,lista)++; break

        Parameters
        ----------
        events : np.ndarray, shape (N, T), dtype bool

        Returns
        -------
        Pm2 : np.ndarray, shape (N, N)
            Transition count matrix for this subject.
        """
        N, T   = events.shape
        Pm2    = np.zeros((N, N))

        for seed in range(N):
            event_times = np.where(events[seed, :])[0]
            for t in event_times:
                # Find the next timepoint where any parcel fires
                found = False
                for t2 in range(t + 1, T):
                    firing_parcels = np.where(events[:, t2])[0]
                    if len(firing_parcels) > 0:
                        Pm2[seed, firing_parcels] += 1
                        found = True
                        break
                if not found:
                    # No subsequent event — count self-transition
                    Pm2[seed, seed] += 1

        return Pm2

    def _remove_excluded(self, arr: np.ndarray, N: int) -> np.ndarray:
        """Remove excluded parcel indices from a 1-D array."""
        if not self.exclude_parcels:
            return arr
        mask = np.ones(N, dtype=bool)
        for idx in self.exclude_parcels:
            if 0 <= idx < N:
                mask[idx] = False
        return arr[mask]
