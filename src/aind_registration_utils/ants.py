"""
Module for ANTs (Advanced Normalization Tools) registration utilities.
"""

import ants
import numpy as np

# import SimpleITK as sitk
import pandas as pd


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
    pt_arr = np.vstack(list(pts_dict.values()))
    pt_df = pd.DataFrame(
        {
            "x": pt_arr[:, 0],
            "y": pt_arr[:, 1],
            "z": pt_arr[:, 2],
        }
    )
    tx_pt_df = ants.apply_transforms_to_points(
        3, pt_df, transform_list, **kwargs
    )
    tx_pts_dict = {
        k: tx_pt_df.iloc[i, :].values for i, k in enumerate(pts_dict.keys())
    }
    return tx_pts_dict


def ants_register_syn(
    fixed_img,
    moving_img,
    rigid_kwargs=dict(),
    affine_kwargs=dict(),
    syn_kwargs=dict(),
    syn_save_prefix="",
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
    syn_save_prefix : str, optional
        Prefix for the output files of the SyN registration. If not specified,
        no prefix is added.

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

    rigid_affine_kwargs_def = dict(aff_smoothing_sigmas=[3, 2, 1, 0])
    rigid_comb_kwargs = {**rigid_affine_kwargs_def, **rigid_kwargs}
    affine_comb_kwargs = {**rigid_affine_kwargs_def, **affine_kwargs}

    tx_rigid = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform="Rigid",
        **rigid_comb_kwargs,
    )
    tx_affine = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        initial_transform=tx_rigid["fwdtransforms"][0],
        type_of_transform="Affine",
        **affine_comb_kwargs,
    )
    print(f"syn kwargs is {syn_comb_kwargs}")
    tx_syn = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        initial_transform=tx_affine["fwdtransforms"][0],
        outprefix=str(syn_save_prefix),
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
