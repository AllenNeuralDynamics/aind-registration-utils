"""
Module for handling annotation images in ants
"""

import ants
import numpy as np


def map_annotations_safely(
    moving_annotations, fixed, transformlist, interpolator="nearestNeighbor", **kwargs
):
    """
    ANTs cannot map annotations with extremely large integer indicies.
    If you try, the result is slightly distorted values which can throw
    off e.g. atlas indexing.

    This fixes a bug with large indicies, mapping them first to smaller values,
    warping them, then mapping them back to the original index.

    Parameters
    ----------
    moving_annotations : ants.ANTsImage
        The source annotation image to be warped.
        (e.g. region IDs from the CCF atlas).
    fixed : ants.ANTsImage
        The reference image defining the target space for the warp.
    transformlist : list of str
        A list of transforms (or paths to transform files) to apply,
        in the same format expected by `ants.apply_transforms`.
    interpolator : str, optional
        Interpolation method for resampling. Defaults to 'nearestNeighbor',
        which is appropriate for label images/atlas annotations.
    **kwargs : dict, optional
        Additional keyword arguments passed to `ants.apply_transforms`.

    Returns
    -------
    warped_annotation : ants.ANTsImage
        The warped annotation image in the fixed space, with the original
        label values preserved.
    """
    # Remap annotations to an ANTs integer image.
    origional_index, index_mapping = np.unique(
        moving_annotations.view(), return_inverse=True
    )
    int_image = ants.from_numpy(index_mapping.astype("float"))
    int_image = int_image.astype("uint32")
    int_image = ants.copy_image_info(moving_annotations, int_image)
    # Check that conversion to ants didn't introduce errors.
    # Ensure dtype consistency and compare
    int_image_cast = int_image.view().astype(index_mapping.dtype)
    assert np.array_equal(int_image_cast, index_mapping), (
        "There appears to have been a rounding error during type conversion. Please check!"
    )

    # Apply the warp
    warped_int_annotations = ants.apply_transforms(
        fixed,
        int_image,
        transformlist=transformlist,
        interpolator=interpolator,
        **kwargs,
    )

    # Map indices back to original
    warped_numpy_annotations = origional_index[
        warped_int_annotations.view().astype(int)
    ]
    warped_annotation = ants.from_numpy(warped_numpy_annotations)
    warped_annotation = ants.copy_image_info(
        fixed,
        warped_annotation,
    )

    # Manually check that no labels changed. Raise an error if it did.
    unique_warped_labels = np.unique(warped_annotation.view())
    for x in unique_warped_labels:
        if x not in origional_index:
            raise ValueError(
                "A value exists in the warped annotations that was not in the starting annotations."
            )

    return warped_annotation
