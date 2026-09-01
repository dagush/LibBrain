# =========================================================================
#  main_demo_ring_hopf_G.py
#
#  Python translation of run_demo_ring_hopf_G_commented.m
#  Observables delegated to neuronumba.
#
#  PAPER CONTEXT:
#  Deco et al. (2021), Current Biology — "Revisiting the Global Workspace
#  orchestrating the hierarchical organization of the human brain"
#
#  WHAT THIS SCRIPT DOES (big picture):
#  ------------------------------------
#  The brain is modelled as a 1D *ring* of N=1000 coupled oscillators.
#  Two connectivity structures are compared:
#
#    (C)  SHORT-RANGE only: coupling decays exponentially with ring distance.
#         This represents a purely local, topographic brain architecture.
#
#    (C2) SMALL-WORLD: same short-range coupling BUT with a random ~5% of
#         node pairs additionally receiving a long-range "shortcut" connection
#         (weight 0.25). This mimics the rare long-range axons seen in cortex.
#
#  Each node is a Stuart-Landau (Hopf) oscillator — the normal-form model
#  of any system near a Hopf bifurcation. When coupled, these produce
#  travelling waves and spatiotemporal complexity.
#
#  For each coupling strength G (sweep 0 → 0.1):
#    - Run both networks (C and C2) for 100 subjects (noise realisations)
#    - Compute TURBULENCE: std of the spatial-temporal "enstrophy" field
#      (how heterogeneous is the local synchrony across space and time?)
#    - Compute INFORMATION TRANSFER: cross-scale temporal correlation of
#      enstrophy across spatial scales lambda (proxy for cascade dynamics)
#    - Compare FClarge: mean long-range functional connectivity
#
#  KEY HYPOTHESIS:
#  Long-range shortcuts increase turbulence, which enables richer
#  information transfer across spatial scales — even though global
#  synchrony (FClarge) may not change dramatically.
# =========================================================================

from __future__ import annotations

import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

import hopf as Hopf
from tools.connectivity_generators.SyntheticArchitectures import RingArchitecture, RingWithLongRangeConnectionsArchitecture

# Observable calculations are delegated to neuronumba.
from neuronumba.observables.fc import FC
from neuronumba.observables.turbulence2 import Information_cascade


NPARCELLS = 1000   # Number of oscillator nodes on the ring
Tmax = 1000        # Number of TR time points to record


def run():
    print("\n=======================================================")
    print("  Ring Hopf Turbulence Demo — Starting")
    print("  Paper: Deco et al. 2021, Current Biology")
    print("=======================================================\n")


    # =========================================================================
    #  SECTION 1: RING DISTANCE MATRIX
    #  =========================================================================
    #  We arrange N nodes on a ring (circular lattice). The "distance" between
    #  two nodes is the shortest arc length around the ring, divided by 10
    #  to put it in convenient units. Maximum possible distance = N/2/10 = 50.

    print(f"[SETUP] Building ring distance matrix for N={NPARCELLS} nodes...")

    # The matrix-construction code has been refactored into OO architecture classes.
    ring_arch = RingArchitecture(
        n_nodes=NPARCELLS,
        tmax=Tmax,
        distance_scale=10.0,
        spatial_decay=1.0,
    )

    rr = ring_arch.ring_distance_matrix()
    maxrr = np.max(rr)   # Maximum ring distance = NPARCELLS/2/10 = 50
    print(f"[SETUP] Max ring distance = {maxrr:.1f} units")


    # =========================================================================
    #  SECTION 2: SPATIAL SCALE PARAMETERS (LAMBDA)
    #  =========================================================================
    #  Lambda controls the spatial scale of the "enstrophy" (local order
    #  parameter). Large lambda = very local (tight Gaussian), small lambda =
    #  broad neighbourhood average.
    #
    #  Enstrophy is computed as a weighted circular mean of complex phases,
    #  where the weights are exp(-lambda * distance). This captures how
    #  synchronised each node's neighbourhood is at scale lambda.
    #
    #  The TRANSFER measure exploits the fact that enstrophy at one scale
    #  predicts enstrophy at a coarser scale one time step later — this
    #  temporal cross-scale correlation is the "information transfer".

    LAMBDA = np.array([4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.0312], dtype=float)
    NLAMBDA = len(LAMBDA)

    print("[SETUP] Spatial scales (lambda):", "  ".join(f"{x:.4f}" for x in LAMBDA))
    print("        (large lambda = fine scale / local; small = coarse / global)")

    # Pre-allocate the 3D weight matrix C1(scale, node_i, node_j)
    # C1(ilam, i, j) = exp(-lambda * rr(i,j))  — Gaussian spatial kernel
    # NOTE: diagonal is 1 here (self-weight included for enstrophy normalisation)
    #
    # Python note:
    # We do NOT allocate the full C1 tensor permanently because 8 x 1000 x 1000
    # doubles consume ~64 MB and each scale is used independently.  Instead, the
    # exact same kernel is generated on demand for each lambda.

    # Build short-range coupling matrix C at lambda=1 (used for the ring network)
    # NOTE: diagonal is 0 here (no self-coupling in the dynamical equations)
    lambda_ref = 1.0   # Spatial decay constant for the SHORT-RANGE coupling matrix C
    Nscale = int(np.flatnonzero(np.isclose(LAMBDA, lambda_ref))[0])
    # → used later to extract enstrophy at scale 1
    # MATLAB index was 1-based; Nscale here is 0-based.

    print(
        f"[SETUP] Reference scale for enstrophy output: "
        f"lambda={lambda_ref:.1f} (Python index {Nscale} in LAMBDA)"
    )

    C = ring_arch.generate()
    print("[SETUP] Coupling and enstrophy kernel matrices built.")


    # =========================================================================
    #  SECTION 3: HOPF OSCILLATOR PARAMETERS
    #  =========================================================================
    #  Each node follows a Stuart-Landau (Hopf) oscillator:
    #
    #    dz/dt = (a + i*omega)*z - |z|^2 * z + G * sum_j C_ij*(z_j - z_i) + noise
    #
    #  In Cartesian form (z = x + iy), with x = z(:,1) and y = z(:,2):
    #
    #    dx/dt =  a*x - omega*y - (x^2+y^2)*x + coupling_x + noise
    #    dy/dt =  a*y + omega*x - (x^2+y^2)*y + coupling_y + noise
    #
    #  The bifurcation parameter 'a' determines the regime:
    #    a > 0 → limit cycle (spontaneous oscillation)
    #    a < 0 → stable fixed point (damped, noise-driven oscillations)
    #    a = 0 → critical point (edge of bifurcation, maximal dynamic range)
    #
    #  The coupling term G * sum_j C_ij*(z_j - z_i) = G*(wC*z - sumC.*z)
    #  is a diffusive coupling: each node is driven toward the weighted average
    #  of its neighbours.

    TR = 1.0                               # Sampling interval (seconds, or arbitrary units)
    f_diff = 0.025 * np.ones(NPARCELLS)    # Natural frequency = 25 mHz for all nodes
    # (uniform → no frequency gradient, pure coupling effects)

    omega_scalar = 2 * np.pi * f_diff      # Angular frequency in rad/s

    # IMPORTANT ADAPTATION TO THE PROVIDED Hopf.py:
    # In MATLAB, omega(:,1) was explicitly negated because the integrator used
    #     zz.*omega
    # after swapping [x,y] -> [y,x].
    # Your Hopf.dfun() already writes the signs explicitly:
    #     dx ... - omega[:,0] * y
    #     dy ... + omega[:,1] * x
    # Therefore BOTH columns passed to Hopf.simulate must contain +omega.
    omega = np.column_stack((omega_scalar, omega_scalar))

    dt = 0.1 * TR / 2   # Integration time step (Euler-Maruyama): dt = 0.05 s
    # Small enough for numerical stability at these frequencies
    sig = 0.01           # Noise amplitude (additive Gaussian white noise)

    print("\n[PARAMS] Hopf oscillator settings:")
    print(f"         Tmax={Tmax} TRs, TR={TR:.1f}, dt={dt:.4f}")
    print(
        f"         Natural frequency f={f_diff[0]:.4f} Hz  "
        f"(omega={omega_scalar[0]:.4f} rad/s)"
    )
    print(f"         Noise amplitude sig={sig:.4f}")


    # =========================================================================
    #  SECTION 4: G SWEEP PARAMETERS
    #  =========================================================================
    # G is the global coupling strength — it scales all connection weights.
    # The values below reproduce the selected coupling conditions used
    # for the Figure 4 simulations.
    #
    # At each G, we run 100 independent noise realisations ("subjects").
    #
    # Low G:  nodes are independent → no turbulence, no information transfer
    # High G: nodes synchronise globally → also low turbulence (locked phase)
    # Optimal G: spatiotemporal complexity peaks → turbulence and transfer peak

    # G_range = np.arange(0.0, 0.1001, 0.001)  # 101 G values
    G_range = np.array([0.01, 0.65])            # 2 values for Figure 4
    beta = 0.05                                 # Probability that any given (i,j) pair gets a long-range shortcut
    # ~5% of all pairs → "small world" rare long-range exceptions
    shortcut_weight = 0.25
    NSUBJECTS = 100

    print(
        f"\n[PARAMS] G sweep: {G_range[0]:.3f} to {G_range[-1]:.3f} "
        f"({len(G_range)} values)"
    )
    print(f"[PARAMS] Long-range shortcut probability beta={beta:.2f}")


    # =========================================================================
    #  SECTION 5: RESULT ARRAYS
    #  =========================================================================
    nG = len(G_range)

    Turbulence = np.full((nG, NSUBJECTS), np.nan)
    Turbulence2 = np.full((nG, NSUBJECTS), np.nan)
    Transfer = np.full((nG, NSUBJECTS), np.nan)
    Transfer2 = np.full((nG, NSUBJECTS), np.nan)
    TransferInfo = np.full((nG, NSUBJECTS, NLAMBDA - 1), np.nan)
    TransferInfo2 = np.full((nG, NSUBJECTS, NLAMBDA - 1), np.nan)
    FClarge = np.full((nG, NSUBJECTS), np.nan)
    FClarge2 = np.full((nG, NSUBJECTS), np.nan)


    # =========================================================================
    #  SECTION 6: BUILD THE SMALL-WORLD MATRIX C2
    #  =========================================================================
    # C2 starts as a copy of the short-range matrix C. Each off-diagonal
    # node pair independently receives a long-range shortcut of weight 0.25
    # with probability beta.
    #
    # This implements the intended small-world perturbation while giving beta
    # its natural interpretation as the shortcut probability per node pair.
    #
    #  This is the "small-world" perturbation: rare long-distance connections
    #  drastically reduce path length (like Watts-Strogatz rewiring) while
    #  preserving local structure.

    print(f"\n[SETUP] Building small-world matrix C2 (shortcuts with beta={beta:.2f})...")

    # small_world_arch = SmallWorldArchitecture(
    #     n_nodes=NPARCELLS,
    #     tmax=Tmax,
    #     distance_scale=10.0,
    #     spatial_decay=1.0,
    #     shortcut_probability=beta,
    #     shortcut_weight=shortcut_weight,
    # )
    # C2 = small_world_arch.generate()
    ring_shortcuts = RingWithLongRangeConnectionsArchitecture(
        n_nodes=NPARCELLS,
        tmax=Tmax,
        spatial_decay=lambda_ref,
        shortcut_probability=beta,
        shortcut_weight=0.25,
    )
    C2 = ring_shortcuts.generate()

    # Count entries that differ from the baseline ring matrix.
    nshortcuts = int(np.count_nonzero(~np.isclose(C2, C)))
    print(f"[SETUP] Added {nshortcuts} long-range shortcuts to C2.")


    # =========================================================================
    #  SECTION 6b: NEURONUMBA OBSERVABLES
    #  =========================================================================

    # Information cascade across the hierarchy of spatial scales
    information_cascade_observable = Information_cascade(
        rr=rr,
        lambda_values=LAMBDA,
        use_absolute_correlation=False,  # reproduce original MATLAB behaviour
        ignore_nans=True,
    )
    information_cascade_observable.configure()

    # ============ compute multiscale observables (with neuronumba)
    def compute_neuronumba_multiscale_observables(
            ts: np.ndarray,
    ) -> tuple[float, np.ndarray, float]:
        """
        Compute turbulence and information cascade using neuronumba.

        Parameters
        ----------
        ts : ndarray
            Time series with shape (n_rois, n_time_samples).

        Returns
        -------
        turbulence : float
            Amplitude turbulence at lambda_ref.

        transfer_lambda : ndarray
            Information-cascade flow across adjacent spatial scales.

        information_cascade : float
            Mean information cascade across spatial scales.
        """

        result = information_cascade_observable.from_fmri(ts)

        turbulence = float(result[f"R_spa_time-{lambda_ref}"])
        transfer_lambda = result["TransferLambda"]
        information_cascade = float(result["InformationCascade"])

        return turbulence, transfer_lambda, information_cascade

    # Configure reusable neuronumba observables once.
    fc_observable = FC(ignore_nans=True)
    fc_observable.configure()


    # =========================================================================
    #  SECTION 7: MAIN LOOP — SWEEP OVER COUPLING STRENGTH G
    #  =========================================================================

    print("\n=========================================================")
    print(f"  STARTING MAIN SIMULATION LOOP ({nG} G values x {NSUBJECTS} subjects)")
    print("  Computational load: LARGE (1000 nodes x Euler-Maruyama)")
    print("=========================================================\n")

    long_range_mask = rr > (0.8 * maxrr)

    for ii, G in enumerate(G_range):
        print(f"--- G = {G:.4f} ({ii + 1}/{nG}) ---")
        tic = time.perf_counter()

        for sub in range(NSUBJECTS):
            if (sub + 1) % 20 == 0:
                print(f"    Subject {sub + 1}/{NSUBJECTS}...")

            # -----------------------------------------------------------------
            #  SIMULATION A: SHORT-RANGE NETWORK (matrix C)
            #  a = +0.1 → SUPERCRITICAL (above bifurcation: self-sustained oscillations)
            #  The network spontaneously oscillates even without coupling.
            #  Coupling synchronises neighbours, creating spatial wave patterns.
            # -----------------------------------------------------------------

            # Bifurcation parameter: a = +0.1 → limit cycle regime
            # All nodes oscillate spontaneously; coupling entrains neighbours
            #
            # The original MATLAB script explicitly constructed wC=G*C and sumC.
            # The provided Hopf.simulate() instead receives C and G separately,
            # and Hopf.dfun() constructs exactly the same diffusive term internally.
            xs, _debug = Hopf.simulate(
                SC=C,
                a=0.1,
                omega=omega,
                G=G,
                dt=dt,
                sigma=sig,
                Tmax=Tmax,
                TR=TR,
                burn_in=2000,
            )
            ts = xs.T   # ts: NPARCELLS x recorded_timepoints

            # NOTE ABOUT THE PROVIDED Hopf.simulate():
            # Its recording loop uses
            #     while t < (Tmax-1)*TR
            # rather than MATLAB's inclusive 0:dt:(Tmax-1)*TR.
            # Depending on its integer-time helper, this can yield Tmax-1 rather
            # than Tmax samples.  All analysis below therefore uses ts.shape[1]
            # rather than assuming exactly Tmax samples.

            # --- FUNCTIONAL CONNECTIVITY THROUGH NEURONUMBA ---
            # FC.from_fmri expects (n_time_samples, n_rois).
            FCmat = fc_observable.from_fmri(ts.T)["FC"]
            FClarge[ii, sub] = np.nanmean(FCmat[long_range_mask])


            # -----------------------------------------------------------------
            #  SIMULATION B: SMALL-WORLD NETWORK (matrix C2)
            #  a = -0.02 → SUBCRITICAL (below bifurcation: noise-driven oscillations)
            #  Without noise, nodes would settle to a fixed point.
            #  Noise + coupling produce irregular, complex spatiotemporal patterns.
            #
            #  NOTE: The two networks use different 'a' values!
            #  This is intentional — the small-world shortcuts provide enough
            #  extra input to maintain oscillations even subcritically, while
            #  the ring-only network needs to be supercritical to oscillate.
            # -----------------------------------------------------------------

            # Bifurcation parameter: a = -0.02 → subcritical (noise-driven)
            xs2, _debug2 = Hopf.simulate(
                SC=C2,
                a=-0.02,
                omega=omega,
                G=G,
                dt=dt,
                sigma=sig,
                Tmax=Tmax,
                TR=TR,
                burn_in=2000,
            )
            ts2 = xs2.T

            # --- FUNCTIONAL CONNECTIVITY THROUGH NEURONUMBA ---
            FCmat2 = fc_observable.from_fmri(ts2.T)["FC"]
            FClarge2[ii, sub] = np.nanmean(FCmat2[long_range_mask])

            # -----------------------------------------------------------------
            #  ENSTROPHY AND INFORMATION TRANSFER
            #
            #  ENSTROPHY (local order parameter at scale lambda):
            #    R_lambda(i,t) = |sum_j w_ij(lambda) * e^{i*phi_j(t)}|
            #                    / sum_j w_ij(lambda)
            #
            #  where w_ij = exp(-lambda * rr(i,j)).
            #
            #  This is the modulus of the weighted circular mean of complex phases
            #  in node i's neighbourhood at spatial scale 1/lambda.
            #    R → 1: all neighbours perfectly synchronised
            #    R → 0: phases uniformly distributed (incoherent)
            #
            #  TURBULENCE = std(R_lambda(i,t)) over all i and t
            #    High turbulence → spatiotemporally heterogeneous synchrony
            #    = complex, richly structured dynamics
            #
            #  INFORMATION TRANSFER (cross-scale temporal correlation):
            #    For adjacent scale pairs (lambda_k, lambda_{k-1}):
            #    corr( R_{lambda_k}(i, t+1),  R_{lambda_{k-1}}(i, t) )
            #    averaged over nodes i with significant correlations.
            #
            #  This measures whether fine-scale dynamics at time t predict
            #  coarse-scale dynamics at t+1 — a signature of an energy/information
            #  cascade from local to global (turbulence cascade, analogous to
            #  Kolmogorov turbulence in fluid dynamics).
            # -----------------------------------------------------------------

            # neuronumba performs the Hilbert transform, local Kuramoto order
            # parameter (enstrophy), turbulence, and cross-scale information
            # cascade calculation.
            (
                Turbulence[ii, sub],
                transfer_profile,
                Transfer[ii, sub],
            ) = compute_neuronumba_multiscale_observables(ts)

            (
                Turbulence2[ii, sub],
                transfer_profile2,
                Transfer2[ii, sub],
            ) = compute_neuronumba_multiscale_observables(ts2)

            TransferInfo[ii, sub, :] = transfer_profile[1:]
            TransferInfo2[ii, sub, :] = transfer_profile2[1:]

        t_elapsed = time.perf_counter() - tic
        print(
            f"    → Done. Time: {t_elapsed:.1f} s | "
            f"Turbulence(C)={np.nanmean(Turbulence[ii, :]):.4f}, "
            f"Turbulence(C2)={np.nanmean(Turbulence2[ii, :]):.4f} | "
            f"Transfer(C)={np.nanmean(Transfer[ii, :]):.4f}, "
            f"Transfer(C2)={np.nanmean(Transfer2[ii, :]):.4f}"
        )

    print("\n=======================================================")
    print("  SIMULATION COMPLETE — Saving results...")
    print("=======================================================")


    # =========================================================================
    #  SECTION 8: SAVE RESULTS
    #  =========================================================================
    savemat(
        "resultsring.mat",
        {
            "Turbulence": Turbulence,
            "Turbulence2": Turbulence2,
            "Transfer": Transfer,
            "Transfer2": Transfer2,
            "TransferInfo": TransferInfo,
            "TransferInfo2": TransferInfo2,
            "FClarge": FClarge,
            "FClarge2": FClarge2,
            "G_range": G_range,
        },
    )
    print("[SAVE] Results saved to resultsring.mat\n")


    # =========================================================================
    #  SECTION 9: FIGURE
    #  =========================================================================
    #  EXPECTED RESULT (from paper):
    #  - Turbulence2 (small-world, red) > Turbulence (short-range only, black)
    #    across intermediate G values
    #  - The peak turbulence G is lower for C2 than C (long-range shortcuts
    #    effectively amplify coupling, shifting the optimal point leftward)
    #  - The difference demonstrates that long-range connections don't just
    #    increase mean FC — they specifically boost SPATIOTEMPORAL HETEROGENEITY
    #    (turbulence), which is the substrate for information cascade/transfer

    print("[FIGURE] Plotting turbulence vs G...")

    plt.figure()
    plt.plot(G_range, np.nanmean(Turbulence, axis=1), "k", linewidth=2,
             label="Short-range only (C)")
    plt.plot(G_range, np.nanmean(Turbulence2, axis=1), "r", linewidth=2,
             label="Small-world (C2)")
    plt.xlabel("Global coupling strength G", fontsize=14)
    plt.ylabel("Turbulence (std of enstrophy)", fontsize=14)
    plt.title(
        "Ring Network: Turbulence vs Coupling Strength\n"
        "Black = short-range | Red = short-range + long-range shortcuts",
        fontsize=12,
    )
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("[DONE] Script finished successfully.")


if __name__ == '__main__':
    run()
