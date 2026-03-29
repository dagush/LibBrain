"""
CHARMsc/simulation/bold_generator.py
--------------------------------------
Generates synthetic BOLD-like binary timeseries by simulating a stochastic
activation process over brain parcels, driven by a diffusion matrix P.

Implements Steps 2 and 3 of the FCmodel.m pipeline:

    Step 2 — Simulate brain trajectories:
        At each timestep t, for every currently active parcel i (state=1),
        independently fire each neighbour j with probability P[i,j].
        Repeat until at least one parcel fires (the MATLAB "while sum==0").
        This is NOT a single-walker random walk — multiple parcels can be
        active simultaneously, which is essential for FC to be well-defined.

    Step 3 — Compute FC from the simulated timeseries:
        FC = corrcoef(tssim, rowvar=True)  — Pearson correlation across time
        Uses the NeuroNumba FC observable.

Key insight (why single-walker fails)
--------------------------------------
With N=1000 parcels and T=1000 timesteps, a single-walker visits each
parcel on average only once. Most parcel timeseries are all-zeros —
constant vectors — whose Pearson correlation is undefined (0/0 → NaN).

The MATLAB loop fires EACH j independently: ``if rand < P[i,j]: fire j``.
With row sums of P equal to 1, the expected number of newly fired parcels
per active parcel per timestep is 1, but many can fire simultaneously.
This gives each parcel O(T/N × expected_visits) non-zero entries, enough
for meaningful correlations.

Reference:
    Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics
    reveal low-dimensional manifolds of critical brain dynamics.
    Physical Review E, 111(1). https://doi.org/10.1103/physreve.111.014410

    FCmodel.m — original MATLAB code by Gustavo Deco.

Two implementations
-------------------
1. _run_single_trial_loop()       — faithful MATLAB translation, O(T·N²)
2. _run_single_trial_vectorised() — NumPy equivalent, O(T·N), same statistics
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from neuronumba.observables.fc import FC as _FCObservable


class BOLDGenerator:
    """
    Generates synthetic FC matrices via stochastic parcel activation.

    Parameters
    ----------
    P : np.ndarray, shape (N, N)
        Row-stochastic diffusion matrix from a fitted HARM or CHARM_SC model.
        Obtained via ``geometry_model.diffusion_matrix``.
    n_timesteps : int
        Length of each simulated timeseries. Default: 1000 (paper value).
        With N=1000, use at least 1000 so each parcel fires multiple times.
    exclude_parcels : list of int or None
        0-indexed parcel indices never chosen as starting points.
        Default: [554, 907] (Schaefer1000 NaN parcels).
    random_state : int or None
        Random seed for reproducibility. Default: None.
    """

    def __init__(
        self,
        P:               np.ndarray,
        n_timesteps:     int = 1000,
        exclude_parcels: Optional[list[int]] = None,
        random_state:    Optional[int] = None,
    ):
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(f"P must be square (N×N), got {P.shape}")
        self.P             = P.astype(np.float64, copy=True)
        self.N             = P.shape[0]
        self.n_timesteps   = n_timesteps
        self.exclude_parcels = exclude_parcels if exclude_parcels is not None \
                               else [554, 907]
        self.rng           = np.random.default_rng(random_state)

        self._valid_parcels = np.array(
            [i for i in range(self.N) if i not in self.exclude_parcels],
            dtype=np.int32,
        )

    # -------------------------------------------------------------------------
    # Private: single trial — loop version (faithful MATLAB translation)
    # -------------------------------------------------------------------------

    def _run_single_trial_loop(self) -> np.ndarray:
        """
        One trial via explicit Python loops — exact MATLAB translation.

        MATLAB inner loop (FCmodel.m lines 112-135):
            tssim(:,1) = 0; tssim(ini,1) = 1;
            for tt = 2:T
                tssim(:,tt) = 0;
                while sum(tssim(:,tt)) == 0
                    for i = 1:N
                        if tssim(i,tt-1) == 1
                            for j = 1:N
                                if rand < P(i,j)
                                    tssim(j,tt) = 1;  % independent Bernoulli per j
                                end
                            end
                        end
                    end
                end
            end

        The critical point: every j is tested independently against P[i,j].
        Multiple j can fire simultaneously. Expected fires per step = 1
        (since sum_j P[i,j] = 1), but variance > 0 → multiple activations.

        Returns
        -------
        tssim : np.ndarray, shape (N, T), dtype float32
        """
        tssim = np.zeros((self.N, self.n_timesteps), dtype=np.float32)

        ini           = int(self.rng.choice(self._valid_parcels))
        tssim[ini, 0] = 1.0

        for tt in range(1, self.n_timesteps):
            # Retry until at least one parcel fires — matches MATLAB while loop
            while tssim[:, tt].sum() == 0:
                for i in range(self.N):
                    if tssim[i, tt - 1] == 1.0:
                        # Test every j independently — this is the key line
                        for j in range(self.N):
                            if self.rng.random() < self.P[i, j]:
                                tssim[j, tt] = 1.0

        return tssim

    # -------------------------------------------------------------------------
    # Private: single trial — vectorised version (fast, same statistics)
    # -------------------------------------------------------------------------

    def _run_single_trial_vectorised(self) -> np.ndarray:
        """
        One trial via NumPy vectorisation — same statistics as the loop.

        At each timestep, for every active parcel i, draw N uniform random
        numbers and compare against the row P[i,:] — firing all j where
        rand[j] < P[i,j]. This exactly replicates the MATLAB independent
        Bernoulli sampling without the Python for-loops.

        Complexity: O(T · N_active · N) per trial, where N_active is the
        number of parcels active at the previous timestep (typically ~1-3).
        In practice nearly identical to O(T · N) since N_active stays small.

        Returns
        -------
        tssim : np.ndarray, shape (N, T), dtype float32
        """
        tssim = np.zeros((self.N, self.n_timesteps), dtype=np.float32)

        ini           = int(self.rng.choice(self._valid_parcels))
        tssim[ini, 0] = 1.0

        for tt in range(1, self.n_timesteps):
            # Retry until at least one parcel fires
            while tssim[:, tt].sum() == 0:
                active = np.where(tssim[:, tt - 1] == 1.0)[0]
                for i in active:
                    # Draw N uniform randoms, fire j where rand < P[i,j]
                    # This is the vectorised equivalent of the MATLAB inner loop
                    fired = self.rng.random(self.N) < self.P[i, :]   # (N,) bool
                    tssim[fired, tt] = 1.0

        return tssim

    # -------------------------------------------------------------------------
    # Public: simulate n_trials and return average FC
    # -------------------------------------------------------------------------

    def simulate_trials(
        self,
        n_trials:       int  = 20,
        use_vectorised: bool = True,
    ) -> np.ndarray:
        """
        Simulate n_trials timeseries and return their nanmean FC.

        Parameters
        ----------
        n_trials : int
            Trials to average. Default: 20 (demo). Use 1000 for paper.
        use_vectorised : bool
            Use fast vectorised simulator. Default: True.

        Returns
        -------
        FC_avg : np.ndarray, shape (N, N)
            nanmean FC across trials (NaN-safe for robustness).
        """
        runner = (self._run_single_trial_vectorised if use_vectorised
                  else self._run_single_trial_loop)

        FC_stack = np.full((n_trials, self.N, self.N), np.nan)
        obs      = _FCObservable()
        obs.ignore_nans = True

        for trial in range(n_trials):
            tssim  = runner()               # (N, T)
            result = obs.from_fmri(tssim.T) # FC observable expects (T, N)
            FC_stack[trial] = result['FC']

        # nanmean matches MATLAB's nanmean(FCsim3) — ignores any residual NaNs
        return np.nanmean(FC_stack, axis=0)

    # -------------------------------------------------------------------------
    # Public: full two-level loop (FCmodel.m outer structure)
    # -------------------------------------------------------------------------

    def simulate_fc(
        self,
        n_trials:       int  = 20,
        n_repetitions:  int  = 10,
        use_vectorised: bool = True,
    ) -> np.ndarray:
        """
        Full two-level averaging loop from FCmodel.m.

        FCmodel.m:
            for tr = 1:NREP                           (200 repetitions)
                for sub = 1:NTRIALS                   (1000 trials per rep)
                    FCsim3(sub,:,:) = corrcoef(tssim')
                FCsim2(tr,:,:) = nanmean(FCsim3)
            FCsim = nanmean(FCsim2)

        Parameters
        ----------
        n_trials : int
            Trials per repetition. Default: 20. Paper: 1000.
        n_repetitions : int
            Number of repetitions. Default: 10. Paper: 200.
        use_vectorised : bool
            Default: True.

        Returns
        -------
        FC_reps : np.ndarray, shape (n_repetitions, N, N)
            Per-repetition FC (nanmean over trials).
            Use FC_reps.mean(axis=0) for the overall estimate.
        """
        FC_reps = np.full((n_repetitions, self.N, self.N), np.nan)

        for rep in range(n_repetitions):
            print(f"  Repetition {rep + 1}/{n_repetitions}...", end='\r')
            FC_reps[rep] = self.simulate_trials(
                n_trials=n_trials,
                use_vectorised=use_vectorised,
            )
        print()

        return FC_reps