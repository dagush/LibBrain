# ==========================================================================
# ==========================================================================
# ==========================================================================
# Normal form of a supercritical Hopf bifurcation
#
# General neural mass model known as the normal form of a Hopf bifurcation
# (also known as Landau-Stuart Oscillators), which is the canonical model
# for studying the transition from noisy to oscillatory dynamics.
#
#     .. [Kuznetsov_2013] Kuznetsov, Y.A. "Elements of applied bifurcation theory", Springer Sci & Business
#     Media, 2013, vol. 112.
#
#     .. [Deco_2017]  Deco, G., Kringelbach, M.L., Jirsa, V.K. et al.
#     The dynamics of resting fluctuations in the brain: metastability and its
#     dynamical cortical core. Sci Rep 7, 3095 (2017).
#     https://doi.org/10.1038/s41598-017-03073-5
#
#
# The supHopf model describes the normal form of a supercritical Hopf bifurcation in Cartesian coordinates. This
# normal form has a supercritical bifurcation at $a = 0$ with a the bifurcation parameter in the model. So for
# $a < 0$, the local dynamics has a stable fixed point, and for $a > 0$, the local dynamics enters in a
# stable limit cycle.
#
# The dynamic equations were taken from [Deco_2017]:
#
#         \dot{x}_{i} &= (a_{i} - x_{i}^{2} - y_{i}^{2})x_{i} - omega{i}y_{i} \\
#         \dot{y}_{i} &= (a_{i} - x_{i}^{2} - y_{i}^{2})y_{i} + omega{i}x_{i}
#
#     where a is the local bifurcation parameter and omega the angular frequency.
#
# ==========================================================================
# ==========================================================================
import numpy as np

from Utils import numTricks as nT

from numpy.random import randn as randn  # normal randn, comment for debug
# from Utils.randn2 import randn2 as randn  # uncomment for debug


# @jit(nopython=True)
def dfun(z, a, omega, G, SC, I_ext):
    """
    Deterministic part of the supercritical Hopf model.

    Parameters
    ----------
    z : array (N, 2)
        State variables (x, y)
    a : float or array (N,)
        Bifurcation parameter
    omega : float or array (N,)
        Angular frequency
    G : float
        Global coupling strength
    SC : array (N, N)
        Structural connectivity matrix

    Returns
    -------
    dz : array (N, 2)
        Time derivative
    debug : dict
        Intermediate variables for inspection
    """

    # --------------------- From Gus' original code:
    # First, we need to compute the term (in pseudo-LaTeX notation):
    #       G Sum_i SC_ij (x_i - x_j) =
    #       G (Sum_i SC_ij x_i + Sum_i SC_ij x_j) =
    #       G ((Sum_i SC_ij x_i) + (Sum_i SC_ij) x_j)   <- adding some unnecessary parenthesis.
    # This is implemented in Gus' code as:
    #       wC = we * Cnew;  # <- 'we' is G in the paper, Cnew is SC -> wC = G * SC
    #       sumC = repmat(sum(wC, 2), 1, 2);  # <- for sum Cij * xj == sum(G*SC,2)
    # Thus, we have that:
    #       suma = wC*z - sumC.*z                 # this is sum(Cij*xi) - sum(Cij)*xj, all multiplied by G
    #            = G * SC * z - sum(G*SC,2) * z   # Careful, component 2 in Matlab is component 1 in Python...
    #            = G * (SC*z - sum(SC,2)*z)
    # And now the rest of it...
    # Remember that, in Gus' code,
    #       omega = repmat(2*pi*f_diff',1,2);
    #       omega(:,1) = -omega(:,1);
    # so here I will call omega(1)=-omega, and the other component as + omega
    #       zz = z(:,end:-1:1)  # <- flipped z, because (x.*x + y.*y)     # Thus, this zz vector is (y,x)
    #       dz = a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma               # original formula in the code, using complex numbers z instead of x and y...
    #          = zz * omega   +  z  * (a -  z.* z  - zz.* zz) + suma =    # I will be using vector notation here to simplify ASCII formulae... ;-)
    #          = (y)*(-omega) + (x) * (a - (x)*(x) - (y)*(y)) + suma      # here, (x)*(x) should actually be (x) * (x,y)
    #          =  x *(+omega)    y          y * y     x * x               #        y   y                     (y)
    # ---------------------
    # Calculate the input to nodes due to couplings
    # Coupling term:
    #   suma = G * (SC * x - sum(SC,2) * x)
    # same for y
    x = z[:, 0]
    y = z[:, 1]
    pC = I_ext + 0j  # to convert it to Complex in case it is not already

    sumSC = np.sum(SC, axis=1)  # Careful: component 2 in Matlab is component 1 in Python

    xcoup = SC @ x - sumSC * x
    ycoup = SC @ y - sumSC * y

    # --------------------- Hopf equations ---------------------
    # dx = (a - x^2 - y^2) * x - omega * y + G * xcoup
    # dy = (a - x^2 - y^2) * y + omega * x + G * ycoup
    r2 = x**2 + y**2
    dx = (a - r2) * x - omega[:,0] * y + G * xcoup + pC.real
    dy = (a - r2) * y + omega[:,1] * x + G * ycoup + pC.imag

    # --------------------- outputs ---------------------
    dz = np.stack((dx, dy), axis=1)

    # Debug info (very useful!)
    debug = {
        "r2": r2,
        "xcoup": xcoup,
        "ycoup": ycoup,
    }

    return dz, debug


# @jit(nopython=True)
def simulate(
    SC,
    a,
    omega,
    G,
    dt,
    sigma,
    Tmax,
    TR,
    I_ext=0.,
    burn_in=2000,
):
    """
    Minimal Hopf simulation with explicit integration.

    Returns
    -------
    xs : (Tmax, N)
    debug : dict
    """

    N = SC.shape[0]

    # Initial condition
    z = 0.1 * np.ones((N, 2))

    # -------- Burn-in --------
    t = 0.0
    while t < burn_in:
        dz, debug = dfun(z, a, omega, G, SC, I_ext)
        noise = randn(N, 2)
        z = z + dt * dz + np.sqrt(dt) * sigma * noise
        t += dt

    # -------- Main simulation --------
    xs = []

    debug = {
        "z_samples": [],
        "r2_samples": [],
    }

    t = 0.0
    while t < (Tmax-1) * TR:
        dz, dbg = dfun(z, a, omega, G, SC, I_ext)

        noise = np.sqrt(dt) * sigma * randn(N, 2)
        z = z + dt * dz + noise

        # Store debug samples (few only)
        if len(debug["z_samples"]) < 5:
            debug["z_samples"].append(z.copy())
            debug["r2_samples"].append(dbg["r2"].copy())

        # Sampling condition (same logic as MATLAB)
        if nT.isInt(t/TR):
            xs.append(z[:, 0].copy())  # x component

        t += dt

    xs = np.array(xs)

    return xs, debug