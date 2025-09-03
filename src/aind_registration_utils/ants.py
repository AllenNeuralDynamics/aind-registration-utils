"""
Module for ANTs (Advanced Normalization Tools) registration utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import ants
import numpy as np

# import SimpleITK as sitk
import pandas as pd


def apply_ants_transforms_to_point_arr(arr, transform_list, **kwargs):
    dim = arr.shape[1]
    cols = list("xyz")[:dim]
    df = pd.DataFrame(arr, columns=cols)
    warped_df = ants.apply_transforms_to_points(dim, df, transform_list, **kwargs)
    warped_arr = warped_df[cols].to_numpy()
    return warped_arr


def apply_ants_transforms_to_point_dict(pts_dict, transform_list, **kwargs):
    """
    Apply ANTs spatial transforms to a dictionary of points.

    This function takes a dictionary of labeled points, applies a series of
    spatial transformations provided in `transform_list` using ANTs, and
    returns the transformed points in a new dictionary with the same labels.

    Parameters
    ----------
    pts_dict : dict
        Dictionary where the keys are point labels and the values are 3D points
        as (x, y, z) sequences. Example: {'pt1': [10, 20, 30], 'pt2': [40, 50,
        60]}
    transform_list : list
        List of spatial transformation filenames to apply to the points. The
        transformations are applied in the order they are provided.
    **kwargs :
        Additional keyword arguments passed to
        `ants.apply_transforms_to_points`.

    Returns
    -------
    dict
        A dictionary containing the transformed points, maintaining the
        original labels.

    Examples
    --------
    >>> points = {'pt1': [10, 20, 30], 'pt2': [40, 50, 60]}
    >>> transforms = ['path/to/transform1.nii.gz', 'path/to/transform2.nii.gz']
    >>> transformed_points = apply_ants_transforms_to_point_dict(points,
    ... transforms)
    """
    arr = np.vstack(list(pts_dict.values()))
    warped_arr = apply_ants_transforms_to_point_arr(arr, transform_list, **kwargs)
    warped_pts_dict = {k: warped_arr[i] for i, k in enumerate(pts_dict)}
    return warped_pts_dict


def _check_ants_prefix(prefix):
    """
    Checks and formats the given ANTs prefix.

    Parameters
    ----------
    prefix : str
        The prefix to check and format.

    Returns
    -------
    str
        The formatted prefix string. If the prefix is a directory, the
        directory path is concatenated with its anchor. Otherwise, the prefix
        is returned as a string.
    """
    prefix_path = Path(prefix)
    if prefix_path.is_dir():
        prefix_str = str(prefix_path) + prefix_path.anchor
    else:
        prefix_str = str(prefix_path)
    return prefix_str


def ants_register_syn(
    fixed_img,
    moving_img,
    rigid_kwargs=dict(),
    affine_kwargs=dict(),
    syn_kwargs=dict(),
    syn_save_prefix="",
    do_rigid=True,
    do_affine=True,
):
    """
    Perform SyN registration using ANTs with a two-stage initialization (rigid
    followed by affine).

    This function performs registration of the moving image to the fixed image
    using the Symmetric Normalization (SyN) method implemented in ANTs
    (Advanced Normalization Tools).  Before the SyN registration, it employs a
    two-stage initialization approach: first, it computes a rigid
    transformation, followed by an affine transformation. The final SyN
    registration is initialized with the affine transformation.

    Parameters
    ----------
    fixed_img : ants.ANTsImage
        Target image for the registration.
    moving_img : ants.ANTsImage
        Source image that will be aligned to the `fixed_img`.
    rigid_kwargs : dict, optional
        Keyword arguments for the rigid registration. Default is an empty
        dictionary.
    affine_kwargs : dict, optional
        Keyword arguments for the affine registration. Default is an empty
        dictionary.
    syn_kwargs : dict, optional
        Keyword arguments for the SyN registration. Default is an empty
        dictionary.
    syn_save_prefix : str, optional
        Prefix for the output files of the SyN registration. If not specified,
        no prefix is added.
    do_rigid : bool, optional
        Flag to perform rigid registration. Default is True.
    do_affine : bool, optional
        Flag to perform affine registration. Default is True.

    Returns
    -------
    dict
        A dictionary containing the results of the SyN registration, including
        forward and inverse transformations, warp fields, and other
        registration details.

    Examples
    --------
    >>> fixed = ants.image_read('path/to/fixed_image.nii.gz')
    >>> moving = ants.image_read('path/to/moving_image.nii.gz')
    >>> syn_results = ants_register_syn_cc(fixed, moving, 'output_prefix_')
    """
    syn_kwargs_def = dict(
        syn_metric="CC",
        syn_sampling=2,
        reg_iterations=(1000, 500, 500),
    )
    syn_comb_kwargs = {**syn_kwargs_def, **syn_kwargs}
    syn_save_prefix_str = _check_ants_prefix(syn_save_prefix)
    rigid_affine_kwargs_def = dict(aff_smoothing_sigmas=[3, 2, 1, 0])
    rigid_comb_kwargs = {**rigid_affine_kwargs_def, **rigid_kwargs}
    affine_comb_kwargs = {**rigid_affine_kwargs_def, **affine_kwargs}
    last_tx = None
    if do_rigid:
        tx_rigid = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform="Rigid",
            **rigid_comb_kwargs,
        )
        last_tx = tx_rigid["fwdtransforms"][0]
    if do_affine:
        tx_affine = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            initial_transform=last_tx,
            type_of_transform="Affine",
            **affine_comb_kwargs,
        )
        last_tx = tx_affine["fwdtransforms"][0]
    tx_syn = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        initial_transform=last_tx,
        outprefix=syn_save_prefix_str,
        type_of_transform="SyN",
        **syn_comb_kwargs,
    )
    return tx_syn


def combine_syn_txs(
    fixed_img,
    moving_img,
    tx_syn,
    fwd_prefix,
    rev_prefix,
):
    """
    Combine transformations for mouse-to-in vivo registrations.

    This function applies a series of spatial transformations to align the
    masked mouse image to the in vivo image. The combined transformations are
    returned for both alignments.

    Parameters
    ----------
    invivo_img : ants.ANTsImage
        The in vivo image used as a target or source for the registrations.
    mouse_img_masked : ants.ANTsImage
        The masked mouse image to be aligned.
    mouse_invivo_tx_syn : dict
        Dictionary containing the forward and inverse transformations between
        the mouse image and the in vivo image.
    mouse_invivo_prefix : str
        Prefix for the output files of the mouse-to-in vivo combined
        transformation.
    invivo_mouse_prefix : str
        Prefix for the output files of the in vivo-to-mouse combined
        transformation.

    Returns
    -------
    tuple
        A tuple containing the combined transformations for:
        - mouse-to-in vivo
        - in vivo-to-mouse

    Examples
    --------
    >>> invivo = ants.image_read('path/to/invivo_image.nii.gz')
    >>> mouse_masked = ants.image_read('path/to/mouse_masked_image.nii.gz')
    >>> mouse_tx = {
    ...     'fwdtransforms': ['path/to/fwd_transform.nii.gz'],
    ...     'invtransforms': ['path/to/inv_transform.nii.gz']
    ... }
    >>> combined_txs = combine_mouse_invivo_txs(
    ...     invivo, mouse_masked, mouse_tx, 'mouse_invivo_prefix_',
    ...     'invivo_mouse_prefix_'
    ... )
    """
    fwd_tx_cmb = ants.apply_transforms(
        fixed=fixed_img,
        moving=moving_img,
        transformlist=tx_syn["fwdtransforms"],
        compose=str(fwd_prefix),
    )
    rev_tx_cmb = ants.apply_transforms(
        fixed=moving_img,
        moving=fixed_img,
        transformlist=tx_syn["invtransforms"],
        whichtoinvert=[True, False],
        compose=str(rev_prefix),
    )
    return fwd_tx_cmb, rev_tx_cmb


def combine_syn_and_second_transform(
    fixed_img,
    moving_img,
    fwd_tx_syn,
    invivo_ccf_path,
    other_fwd_path,
    other_rev_path,
    combined_prefix,
):
    """
    Combine transformations for mouse-to-CCF (Common Coordinate Framework)
    registrations.

    This function applies a series of spatial transformations to align the
    masked mouse image with the CCF image. These transformations encompass
    mouse-to-in vivo and in vivo-to-CCF alignments. The combined
    transformations for both alignments are returned.

    Parameters
    ----------
    invivo_img : ants.ANTsImage
        The in vivo image used as an intermediate reference for the
        registrations.
    mouse_img_masked : ants.ANTsImage
        The masked mouse image to be aligned.
    mouse_invivo_tx_syn : dict
        Dictionary containing the forward and inverse transformations between
        the mouse image and the in vivo image.
    invivo_ccf_path : str or pathlib.Path
        Path to the transformation from in vivo to CCF image.
    ccf_invivo_path : str or pathlib.Path
        Path to the transformation from CCF to in vivo image.
    mouse_ccf_prefix : str or pathlib.Path
        Prefix for the output files of the mouse-to-CCF combined
        transformation.
    ccf_mouse_prefix : str or pathlib.Path
        Prefix for the output files of the CCF-to-mouse combined
        transformation.

    Returns
    -------
    tuple
        A tuple containing the combined transformations for:
        - mouse-to-CCF
        - CCF-to-mouse

    Examples
    --------
    >>> invivo = ants.image_read('path/to/invivo_image.nii.gz')
    >>> mouse_masked = ants.image_read('path/to/mouse_masked_image.nii.gz')
    >>> mouse_tx = {
    ...     'fwdtransforms': ['path/to/fwd_transform.nii.gz'],
    ...     'invtransforms': ['path/to/inv_transform.nii.gz']
    ... }
    >>> combined_txs = combine_mouse_invivo_and_invivo_ccf_txs(
    ...     invivo, mouse_masked, mouse_tx,
    ...     'path/to/invivo_ccf_tx.nii.gz', 'path/to/ccf_invivo_tx.nii.gz',
    ...     'mouse_ccf_prefix_', 'ccf_mouse_prefix_'
    ... )
    """
    fwd_tx_cmb = ants.apply_transforms(
        fixed=fixed_img,
        moving=moving_img,
        transformlist=[str(invivo_ccf_path)] + fwd_tx_syn["fwdtransforms"],
        compose=str(other_rev_path),
    )
    rev_tx_cmb = ants.apply_transforms(
        fixed=moving_img,
        moving=fixed_img,
        transformlist=fwd_tx_syn["invtransforms"] + [str(other_fwd_path)],
        whichtoinvert=[True, False, False],
        compose=str(combined_prefix),
    )

    return fwd_tx_cmb, rev_tx_cmb


def _surface_samples(size: Sequence[int], n: int = 2) -> List[Tuple[int, ...]]:  # noqa: C901
    """
    Generate sample index coordinates on an image's surface.

    This returns integer index coordinates on the outer surface of a 2D or 3D
    array. With `n=2`, this yields only the corners.  With `n>2`, it yields a
    sparse grid on each face (x-min/x-max, y-min/y-max, and for 3D,
    z-min/z-max).

    Parameters
    ----------
    size : Sequence[int]
        The array/image shape as a sequence of length 2 or 3 (e.g., `(nx, ny, nz)`).
    n : int, optional
        Number of sample points per edge (>= 2). `n=2` produces corners only.
        Larger values sample faces more densely. Default is 2.

    Returns
    -------
    list of tuple of float
        A list of index coordinates on the surface; each tuple has length equal
        to the array dimension (2 or 3). Values are floats to support later
        conversion to physical points without implicit rounding.

    Notes
    -----
    - Indices are **0-origin** and refer to voxel centers in index space.
    - This function does *not* include interior points; only the hull.

    Examples
    --------
    >>> _surface_samples((10, 8), n=2)  # 2D corners
    [(0.0, 0.0), (0.0, 7.0), (9.0, 0.0), (9.0, 7.0)]
    >>> len(_surface_samples((10, 8, 6), n=3))  # 3D sparse face grid
    6 * 3 * 3 - 12  # number of unique samples on 6 faces (no duplicates)
    """
    size = list(size)
    dims = len(size)
    if dims not in (2, 3):
        raise ValueError(f"`size` must have length 2 or 3, got {dims}")
    if n < 2:
        raise ValueError("`n` must be >= 2")

    ax = [np.linspace(0, s - 1, n, dtype=np.intp) for s in size]
    pts: List[Tuple[int, ...]] = []

    if dims == 2:
        xs, ys = ax
        # left/right edges
        for i in (0, n - 1):
            for j in range(n):
                pts.append((int(xs[i]), int(ys[j])))
        # top/bottom edges (avoid duplicates at corners)
        for j in (0, n - 1):
            for i in range(1, n - 1):
                pts.append((int(xs[i]), int(ys[j])))
    else:
        xs, ys, zs = ax
        # x-min/x-max faces
        for i in (0, n - 1):
            for j in range(n):
                for k in range(n):
                    pts.append((int(xs[i]), int(ys[j]), int(zs[k])))
        # y-min/y-max faces (skip edges already included along x faces)
        for j in (0, n - 1):
            for i in range(1, n - 1):
                for k in range(n):
                    pts.append((int(xs[i]), int(ys[j]), int(zs[k])))
        # z-min/z-max faces (skip edges already included)
        for k in (0, n - 1):
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    pts.append((int(xs[i]), int(ys[j]), int(zs[k])))

    return pts


def _to_continuous_index(
    phys_pts: Sequence[Sequence[float]],
    origin: Sequence[float],
    spacing: Sequence[float],
    direction: Sequence[float],
) -> np.ndarray:
    """
    Convert physical coordinates to continuous index coordinates.

    Uses the standard ANTs/SimpleITK geometry:
    ``p = o + D @ (s * i)``  =>  ``i = (D.T @ (p - o)) / s``.

    Parameters
    ----------
    phys_pts : Sequence[Sequence[float]]
        Iterable of physical coordinates with shape (N, dim).
    origin : Sequence[float]
        Origin vector ``o`` (length `dim`) in physical units.
    spacing : Sequence[float]
        Per-axis spacing vector ``s`` (length `dim`) in physical units.
    direction : Sequence[float]
        Flattened row-major direction matrix ``D`` of length `dim*dim`.

    Returns
    -------
    numpy.ndarray
        Continuous index coordinates with shape (N, dim), where each row
        represents the floating-point index coordinate corresponding to the
        input physical point on the target grid.

    Notes
    -----
    - This does *not* round indices; downstream callers can floor/ceil as needed
      to build bounding boxes on a voxel lattice.
    - `direction` is flattened row-major to match ANTs/SimpleITK conventions.

    Examples
    --------
    >>> pts = [(1.0, 2.0, 3.0)]
    >>> o = (0.0, 0.0, 0.0)
    >>> s = (1.0, 2.0, 3.0)
    >>> D = np.eye(3).reshape(-1).tolist()
    >>> _to_continuous_index(pts, o, s, D)
    array([[1. , 1. , 1. ]])
    """
    o = np.asarray(origin, dtype=float)
    s = np.asarray(spacing, dtype=float)
    dim = s.size
    D = np.asarray(direction, dtype=float).reshape((dim, dim))
    phys = np.asarray(phys_pts, dtype=float)
    if phys.ndim != 2 or phys.shape[1] != dim:
        raise ValueError(f"`phys_pts` must have shape (N, {dim})")
    return ((phys - o) @ D.T) / s


def apply_transforms_auto_bbox(
    moving: ants.ANTsImage,
    transformlist: Sequence[Any],
    whichtoinvert: Optional[Sequence[bool]] = None,
    fixed: Optional[ants.ANTsImage] = None,
    spacing: Optional[Sequence[float]] = None,
    direction: Optional[Sequence[float]] = None,
    origin: Optional[Sequence[float]] = None,
    samples_per_edge: int = 2,
    pad_voxels: int = 1,
    interpolator: str = "linear",
    default_value: float | int = 0,
    **kwargs,
) -> Tuple[ants.ANTsImage, ants.ANTsImage]:
    """
    Warp `moving` into fixed space using a reference grid that *auto-fits*
    the warped extent of `moving`.

    This utility builds a minimal ANTs reference image (spacing/direction/origin
    from `fixed` if provided, otherwise from arguments/defaults) that tightly
    bounds the warped moving image. It then calls `ants.apply_transforms`
    with that reference.

    Parameters
    ----------
    moving : ants.ANTsImage
        The moving image to be warped.
    transformlist : Sequence[Any]
        Forward transforms that map **moving → fixed** space. Typically the
        `fwdtransforms` from ANTsPy registration. These can be filenames
        (e.g., .mat, .h5) or in-memory transform specs accepted by ANTsPy.
    fixed : ants.ANTsImage, optional
        If provided, the output grid will align with this image's `spacing`,
        `direction`, and `origin` (but with size chosen to bound the warp).
        If omitted, you must supply (or accept defaults for) `spacing`,
        `direction`, and `origin`.
    spacing : Sequence[float], optional
        Output voxel spacing (length = image dimension). Defaults to
        `moving.spacing` when `fixed is None`.
    direction : Sequence[float], optional
        Flattened row-major direction matrix of length `dim*dim`. Defaults to
        identity when `fixed is None`.
    origin : Sequence[float], optional
        Output origin in physical units. If `fixed is None` and `origin` is
        omitted, defaults to all zeros. If `fixed` is provided, its origin
        is used.
    samples_per_edge : int, optional
        Number of samples per edge on the moving image surface when estimating
        the warped bounding box. `2` uses corners only; larger values add a sparse
        face grid to better capture nonlinear warps. Default is 2.
    pad_voxels : int, optional
        Safety padding (in voxels) added uniformly to the min/max index bounds.
        Default is 1.
    interpolator : str, optional
        Interpolator for `ants.apply_transforms` (e.g., "nearestNeighbor", "linear",
        "bspline"). Default is "linear".
    default_value : float or int, optional
        Fill value for voxels outside the pulled-back `moving` domain.
        Default is 0.

    Returns
    -------
    warped : ants.ANTsImage
        The warped moving image sampled on the auto-computed reference grid.
    ref : ants.ANTsImage
        The auto-constructed reference image whose extent tightly bounds the
        warped moving image.

    Notes
    -----
    - The function samples points on the **moving** image surface in index space,
      maps them to physical space, applies the provided forward transforms
      (moving→fixed) via `ants.apply_transforms_to_points`, converts those
      physical coordinates to continuous index space of the target grid, and
      constructs a minimal bounding box (with optional padding).
    - If `fixed` is provided, the resulting grid is guaranteed to lie on the
      **fixed voxel lattice** (origin is snapped using integer index offsets).
    - For highly nonlinear transforms, increase `samples_per_edge` (e.g., 5–9).

    See Also
    --------
    ants.apply_transforms : Resample an image through a transform chain.
    ants.apply_transforms_to_points : Apply transforms to point sets.

    Examples
    --------
    >>> # Suppose you have ANTs images `fixed`, `moving`, and transforms `xfms`
    >>> warped, ref = apply_transforms_auto_bbox(
    ...     moving=moving,
    ...     transformlist=xfms,     # moving→fixed transforms
    ...     fixed=fixed,            # or omit and specify spacing/direction/origin
    ...     samples_per_edge=5,
    ...     pad_voxels=2,
    ... )
    """
    dim = moving.dimension

    if fixed is not None:
        spacing = fixed.spacing
        direction = fixed.direction
        origin = fixed.origin
    else:
        if spacing is None:
            spacing = moving.spacing
        if direction is None:
            direction = tuple(np.eye(dim, dtype=float).reshape(-1).tolist())
        if origin is None:
            origin = tuple(float(0) for _ in range(dim))

    # 1) sample surface points in moving index space and convert to physical
    idx_samples = _surface_samples(moving.shape, n=samples_per_edge)
    phys_m = [
        ants.transform_index_to_physical_point(moving, idx) for idx in idx_samples
    ]

    # 2) map to fixed physical space
    cols = list("xyz")[:dim]
    df = pd.DataFrame(phys_m, columns=cols)
    warped_df = ants.apply_transforms_to_points(
        dim,
        df,
        transformlist,
        whichtoinvert=whichtoinvert,
        **kwargs,
    )
    phys_f = warped_df[cols].to_numpy()

    # 3) compute bounding box in index space of the target grid
    ci = _to_continuous_index(phys_f, origin, spacing, direction)
    idx_min = np.floor(ci.min(axis=0)).astype(int) - int(pad_voxels)
    idx_max = np.ceil(ci.max(axis=0)).astype(int) + int(pad_voxels)
    size = (idx_max - idx_min + 1).astype(int)

    # snap origin so that idx_min corresponds to the first voxel
    D = np.asarray(direction, dtype=float).reshape((dim, dim))
    s = np.asarray(spacing, dtype=float)
    o = np.asarray(origin, dtype=float)
    origin_out = o + D @ (s * idx_min)

    # 4) build reference image and 5) resample
    ref = ants.make_image(
        imagesize=tuple(map(int, size)),
        voxval=0,
        spacing=spacing,
        origin=origin_out.tolist(),
        direction=D,
        pixeltype=moving.pixeltype,  # optional: match moving’s pixel type
    )
    warped = ants.apply_transforms(
        fixed=ref,
        moving=moving,
        transformlist=list(transformlist),
        whichtoinvert=whichtoinvert,
        interpolator=interpolator,
        defaultvalue=default_value,
    )
    return warped, ref
