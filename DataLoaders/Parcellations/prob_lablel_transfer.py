import numpy as np


def _scale_factor(source, target, atol=1e-6):
    """
    Return the isotropic voxel scaling factor between two atlases.

    The function assumes that both atlases have identical orientation and
    origin, and that their affine matrices differ only by a uniform scaling.

    Returns
    -------
    float
        Ratio of target voxel size to source voxel size.

    Raises
    ------
    ValueError
        If the affines are incompatible.
    """
    A = source.affine
    B = target.affine

    # Orientation must be identical
    Ra = A[:3, :3] / np.linalg.norm(A[:3, :3], axis=0)
    Rb = B[:3, :3] / np.linalg.norm(B[:3, :3], axis=0)

    if not np.allclose(Ra, Rb, atol=atol):
        raise ValueError("Atlases do not have the same orientation.")

    # Same origin
    if not np.allclose(A[:3, 3], B[:3, 3], atol=atol):
        raise ValueError("Atlases do not have the same origin.")

    va = np.linalg.norm(A[:3, :3], axis=0)
    vb = np.linalg.norm(B[:3, :3], axis=0)

    if not np.allclose(va, va[0], atol=atol):
        raise ValueError("Source voxels are not isotropic.")

    if not np.allclose(vb, vb[0], atol=atol):
        raise ValueError("Target voxels are not isotropic.")

    scale = vb[0] / va[0]

    if not np.allclose(vb, scale * va, atol=atol):
        raise ValueError("Affine difference is not a uniform scaling.")

    return scale


def _match_resolution(source_data, target_data, scale):
    """
    Match two label volumes with different isotropic resolutions.

    Rather than resampling the label images, this function expands the
    coarser atlas so that every fine voxel is paired with the label of the
    coarse voxel containing it. This preserves all label information from
    the higher-resolution atlas.

    Parameters
    ----------
    source_data : ndarray
        Source atlas labels.
    target_data : ndarray
        Target atlas labels.
    scale : float
        target_voxel_size / source_voxel_size.

    Returns
    -------
    source_labels : ndarray
    target_labels : ndarray

        Flattened arrays of identical length. Each position corresponds to
        the same physical voxel.
    """

    if np.isclose(scale, 1.0):
        return source_data.ravel(), target_data.ravel()

    # ---------------------------------------------------------------
    # Source is finer than target (e.g. 1 mm -> 2 mm)
    # ---------------------------------------------------------------
    if scale > 1:

        factor = int(round(scale))

        if not np.isclose(scale, factor):
            raise NotImplementedError(
                "Only integer isotropic scaling factors are supported."
            )

        target_expanded = (
            target_data
            .repeat(factor, axis=0)
            .repeat(factor, axis=1)
            .repeat(factor, axis=2)
        )

        if source_data.shape != target_expanded.shape:
            raise ValueError("Expanded target atlas does not match source dimensions.")

        return source_data.ravel(), target_expanded.ravel()

    # ---------------------------------------------------------------
    # Source is coarser than target (e.g. 2 mm -> 1 mm)
    # ---------------------------------------------------------------
    factor = int(round(1 / scale))

    if not np.isclose(1 / scale, factor):
        raise NotImplementedError(
            "Only integer isotropic scaling factors are supported."
        )

    source_expanded = (
        source_data
        .repeat(factor, axis=0)
        .repeat(factor, axis=1)
        .repeat(factor, axis=2)
    )

    if source_expanded.shape != target_data.shape:
        raise ValueError("Expanded source atlas does not match target dimensions.")

    return source_expanded.ravel(), target_data.ravel()


def probabilistic_label_transfer(source, target):
    """
    Compute probabilistic label transfer between two atlases.

    For each target label, returns the list of overlapping source labels
    ordered from most to least probable.

    Parameters
    ----------
    source : Atlas
    target : Atlas

    Returns
    -------
    dict[int, list[tuple[int, float]]]

        Example
        -------
        {
            17: [(42, 0.81), (41, 0.13), (38, 0.06)],
            18: [(39, 1.00)],
            ...
        }
    """

    scale = _scale_factor(source, target)

    source_labels, target_labels = _match_resolution(
        source.data,
        target.data,
        scale,
    )

    # Ignore background
    valid = (source_labels != 0) & (target_labels != 0)

    source_labels = source_labels[valid].astype(int)
    target_labels = target_labels[valid].astype(int)

    # ------------------------------------------------------------------
    # Contingency matrix
    # ------------------------------------------------------------------

    overlap = np.zeros(
        (source.max + 1, target.max + 1),
        dtype=np.int64,
    )

    np.add.at(overlap, (source_labels, target_labels), 1)

    # ------------------------------------------------------------------
    # Convert into probabilities
    # ------------------------------------------------------------------

    mapping = {}

    for target_label in range(1, target.max + 1):

        counts = overlap[:, target_label]

        idx = np.nonzero(counts)[0]

        if idx.size == 0:
            mapping[target_label] = []
            continue

        probs = counts[idx] / counts[idx].sum()

        order = np.argsort(probs)[::-1]

        mapping[target_label] = [
            (int(idx[i]), float(probs[i]))
            for i in order
        ]

    return mapping


# ==================================================================
# convenience functions
# ==================================================================
def all_label_transfer(source, target):
    """
    Compute probabilistic label transfer between two atlases.

    For each target label, returns the list of overlapping source labels
    ordered from most to least probable.

    Parameters
    ----------
    source : Atlas
    target : Atlas

    Returns
    -------
    dict[int, list[tuple[int, float]]]

        Example
        -------
        {
            17: [(42, 0.81), (41, 0.13), (38, 0.06)],
            18: [(39, 1.00)],
            ...
        }
    """
    res_probs = probabilistic_label_transfer(source, target)
    res = {l:[prob[0] for prob in res_probs[l]] for l in res_probs}
    return res


def label_transfer(source, target):
    labels = all_label_transfer(source, target)
    res = {l: labels[l][0] for l in labels}
    return res


# ==================================================================
# debug code: transfer data between parcellations
# ==================================================================
if __name__ == '__main__':
    import DataLoaders.Parcellations.atlas as atlas
    target = atlas.Atlas('Schaefer2018', N=100, normalization=2, RSN=7)
    source = atlas.Atlas('Glasser360')
    # ---- and now... ;-)
    res = probabilistic_label_transfer(source, target)
    labels = label_transfer(source, target)

    print(f'Done !!!')

# ======================================================
# ======================================================
# ======================================================EOF