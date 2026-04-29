import numpy as np
from simulator.models.OrnsteinUhlenbeck import OrnsteinUhlenbeck

def test_hurwitz():
    # The Test Matrix: Let's use a 5×5 asymmetric SC with a
    # dominant hub node (node 0), a few reciprocal connections,
    # and some weak long-range links:
    # W = \begin{bmatrix}
    #      0 & 3.0 & 1.5 & 0.2 & 0.0 \\
    #      1.0 & 0 & 2.5 & 0.0 & 0.5 \\
    #      0.5 & 0.3 & 0 & 4.0 & 0.1 \\
    #      0.0 & 0.1 & 0.8 & 0 & 3.5 \\
    #      0.2 & 0.0 & 0.3 & 1.2 & 0
    # \end{bmatrix}
    # It has zero diagonal (realistic SC), is asymmetric (directed),
    # and its spectral radius is >> 1, making it unstable out of the box.

    model = OrnsteinUhlenbeck(g=1.0)

    # ------------------------------------------------------------------ #
    #  Non-trivial 5x5 test SC matrix                                     #
    #  - Asymmetric (directed), zero diagonal (no self-connections)       #
    #  - Has a dominant hub (node 0) with strong outgoing weights         #
    #  - Spectral radius ~ 3.57 --> clearly UNSTABLE out of the box      #
    # ------------------------------------------------------------------ #
    W_raw = np.array([
        [0.0, 3.0, 1.5, 0.2, 0.0],
        [1.0, 0.0, 2.5, 0.0, 0.5],
        [0.5, 0.3, 0.0, 4.0, 0.1],
        [0.0, 0.1, 0.8, 0.0, 3.5],
        [0.2, 0.0, 0.3, 1.2, 0.0],
    ])

    epsilon = 0.01

    def is_hurwitz(W, tol=1e-9):
        return np.all(np.real(np.linalg.eigvals(W)) < tol)

    def max_real_eig(W):
        return np.max(np.real(np.linalg.eigvals(W)))

    print("=" * 60)
    print("RAW MATRIX")
    print("=" * 60)
    print(f"  Spectral radius : {max_real_eig(W_raw):.4f}  (expected ~3.5727)")
    print(f"  Hurwitz stable  : {is_hurwitz(W_raw)}       (expected False)")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("1. normalize_spectral_radius")
    print("=" * 60)
    W1 = model.normalize_spectral_radius(W_raw)
    mr1 = max_real_eig(W1)
    print(f"  Max Re(lambda)  : {mr1:.6f}  (expected exactly 1.0)")
    print(f"  Hurwitz stable  : {is_hurwitz(W1)}      (expected False — needs g < 1/tau)")
    assert abs(mr1 - 1.0) < 1e-6, "FAIL: spectral radius should be normalized to 1.0"
    assert not is_hurwitz(W1), "FAIL: W1 should NOT be Hurwitz on its own"
    print("  --> PASSED")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("2. hurwitz_spectral_projection")
    print("=" * 60)
    W2 = model.hurwitz_spectral_projection(W_raw, epsilon=epsilon)
    mr2 = max_real_eig(W2)
    print(f"  Max Re(lambda)  : {mr2:.6f}  (expected <= -{epsilon}, ~-0.01)")
    print(f"  Hurwitz stable  : {is_hurwitz(W2)}       (expected True)")
    assert mr2 < 0, "FAIL: W2 should be Hurwitz"
    assert abs(mr2 - (-epsilon)) < 1e-4, "FAIL: largest eigenvalue should be ~-epsilon"
    print("  --> PASSED")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("3. hurwitz_diagonal_dominance")
    print("=" * 60)
    W3 = model.hurwitz_diagonal_dominance(W_raw, epsilon=epsilon)
    mr3 = max_real_eig(W3)
    expected_diag = np.array([-4.71, -4.01, -4.91, -4.41, -1.71])
    print(f"  Max Re(lambda)  : {mr3:.6f}  (expected < 0, ~ -0.01)")
    print(f"  Diagonal        : {np.round(np.diag(W3), 4)}")
    print(f"  Expected diag   : {expected_diag}")
    print(f"  Hurwitz stable  : {is_hurwitz(W3)}       (expected True)")
    assert mr3 < 0, "FAIL: W3 should be Hurwitz"
    assert np.allclose(np.diag(W3), expected_diag, atol=1e-2), \
        "FAIL: diagonal values are wrong"
    assert np.allclose(W3 - np.diag(np.diag(W3)),
                       W_raw - np.diag(np.diag(W_raw))), \
        "FAIL: off-diagonal structure should be unchanged"
    print("  --> PASSED")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("4. hurwitz_symmetrized_negative_definite")
    print("=" * 60)
    W4 = model.hurwitz_symmetrized_negative_definite(W_raw, epsilon=epsilon)
    mr4 = max_real_eig(W4)
    print(f"  Max Re(lambda)  : {mr4:.6f}  (expected exactly -{epsilon})")
    print(f"  Symmetric       : {np.allclose(W4, W4.T)}")
    print(f"  Hurwitz stable  : {is_hurwitz(W4)}       (expected True)")
    assert mr4 < 0, "FAIL: W4 should be Hurwitz"
    assert abs(mr4 - (-epsilon)) < 1e-6, "FAIL: largest eigenvalue should be exactly -epsilon"
    assert np.allclose(W4, W4.T, atol=1e-10), "FAIL: W4 should be symmetric"
    print("  --> PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)