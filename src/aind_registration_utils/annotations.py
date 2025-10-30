"""
Module for handling annotation images in ants
"""

from __future__ import annotations

from typing import Any

import ants
import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray


def _round_ants_image_to_uint32(img: ants.ANTsImage) -> ants.ANTsImage:
    """
    Round an ANTs image to uint32 type.

    Parameters
    ----------
    img : ants.ANTsImage
        Input ANTs image to be rounded and converted.

    Returns
    -------
    ants.ANTsImage
        New ANTs image with data rounded to nearest integer and
        converted to uint32 type.
    """
    arr = img.view()
    arr_rounded = np.round(arr).astype(np.uint32)
    img_uint32 = ants.from_numpy(arr_rounded)
    img_uint32 = ants.copy_image_info(img, img_uint32)
    return img_uint32


def map_annotations_safely(
    moving_annotations: ants.ANTsImage,
    fixed: ants.ANTsImage,
    transformlist: list[str],
    interpolator: str = "genericLabel",
    **kwargs: Any,
) -> ants.ANTsImage:
    """
    Safely warp annotation images with large integer indices using ANTs.

    ANTs cannot map annotations with extremely large integer indices without
    introducing rounding errors that distort label values. This function
    solves this by temporarily remapping labels to a compact range,
    applying the transformation, then restoring original values.

    Parameters
    ----------
    moving_annotations : ants.ANTsImage
        The source annotation image to be warped
        (e.g., region IDs from the CCF atlas).
    fixed : ants.ANTsImage
        The reference image defining the target space for the warp.
    transformlist : list of str
        List of transforms (or paths to transform files) to apply,
        in the format expected by ``ants.apply_transforms``.
    interpolator : str, optional
        Interpolation method for resampling. Default is 'genericLabel',
        which is appropriate for label images and atlas annotations.
    **kwargs : Any
        Additional keyword arguments passed to ``ants.apply_transforms``.

    Returns
    -------
    ants.ANTsImage
        The warped annotation image in the fixed space, with the original
        label values preserved.

    Raises
    ------
    ValueError
        If warped array contains values not present in the original
        annotations, indicating an interpolation error.

    See Also
    --------
    compact_labels_image : Compact label values to contiguous range.
    expand_compacted_image : Expand compacted labels back to originals.

    Notes
    -----
    The algorithm follows these steps:

    1. Extract unique labels and create compact mapping (0 to N-1)
    2. Convert to uint32 ANTs image
    3. Apply transforms with specified interpolator
    4. Map compacted indices back to original label values
    5. Validate no spurious labels were introduced

    Examples
    --------
    >>> import ants
    >>> # Load CCF annotation atlas and target image
    >>> ccf_anno = ants.image_read('ccf_annotation.nii.gz')
    >>> target_img = ants.image_read('subject_brain.nii.gz')
    >>> transforms = ['affine.mat', 'warp.nii.gz']
    >>> # Safely warp annotations
    >>> warped_anno = map_annotations_safely(
    ...     ccf_anno, target_img, transforms
    ... )
    """
    # Remap annotations to an ANTs integer image.
    annotation_arr = moving_annotations.view()
    if np.issubdtype(annotation_arr.dtype, np.integer):
        annotation_int_image = moving_annotations
    else:
        annotation_int_image = _round_ants_image_to_uint32(moving_annotations)

    # Warped image will have the same pixel type as the fixed image.
    # To avoid rounding issues, we convert fixed to integer type if needed.
    fixed_arr = fixed.view()
    if np.issubdtype(fixed_arr.dtype, np.integer):
        fixed_int_image = fixed
    else:
        fixed_int_image = _round_ants_image_to_uint32(fixed)

    # Apply the warp
    warped_img = ants.apply_transforms(
        fixed_int_image,
        annotation_int_image,
        transformlist=transformlist,
        interpolator=interpolator,
        **kwargs,
    )

    return warped_img


def _get_lateralization_regions(
    img: sitk.Image, midline: str = "left"
) -> tuple[list[int], list[int]]:
    """
    Calculate which region to negate for lateralization.

    Determines the start index and extent for the left hemisphere region
    that will be negated during lateralization, with control over how
    midline voxels are handled.

    Parameters
    ----------
    img : sitk.Image
        Input SimpleITK image. Must be approximately axis-aligned.
    midline : {"left", "right", "bilateral"}
        How to handle midline voxels (only relevant for odd dimensions).

        - "left": Midline voxel is treated as left hemisphere and will
          be negated. With odd dimensions (e.g., 11 voxels), left gets
          6 voxels including the midline.
        - "right": Midline voxel is treated as right hemisphere and will
          NOT be negated. Left gets 5 voxels, right gets 6.
        - "bilateral": Midline voxel represents a bilateral structure and
          will NOT be negated. Same spatial result as "right" but different
          semantic intent.

        For even dimensions, this parameter has no effect as there is no
        single midline voxel.

    Returns
    -------
    start : list of int
        Starting index for the region to negate (left hemisphere).
    extent : list of int
        Extent (size) of the region to negate.

    Raises
    ------
    ValueError
        If the image is not axis-aligned (direction matrix diagonal
        elements are not approximately ±1).

    Notes
    -----
    Uses the image direction matrix to identify the physical X-axis
    (in LPS, +X = LEFT) and determine whether increasing indices
    correspond to leftward or rightward directions.

    Examples
    --------
    With size=11 (indices 0-10, midpoint at index 5):

    - midline="left": Negate indices based on orientation
      (e.g., [5:11] if left_is_upper → 6 voxels negated)
    - midline="right" or "bilateral": Negate strict left only
      (e.g., [6:11] if left_is_upper → 5 voxels negated, midline preserved)

    With size=10 (indices 0-9, midpoint at index 5):

    - All modes behave identically: perfect 5/5 split, no midline voxel
    """
    dim = img.GetDimension()
    size = list(img.GetSize())
    D = np.array(img.GetDirection()).reshape((dim, dim))
    x_row = D[0]  # physical X axis row
    x_axis = max(range(dim), key=lambda k: abs(x_row[k]))

    if abs(x_row[x_axis]) < 0.9:
        raise ValueError("Not axis-aligned (direction not ~±1 on a single axis).")

    # In LPS, +X is LEFT. If D[0][x_axis] > 0, increasing index → LEFT.
    left_is_upper = x_row[x_axis] > 0
    mid = size[x_axis] // 2
    has_midline = size[x_axis] % 2 == 1

    start = [0] * dim
    extent = size[:]

    # Determine negation region based on midline mode
    if midline == "left" or not has_midline:
        # Include midline in left (default/current behavior)
        # Even: each side gets mid voxels
        # Odd: left gets mid+1 voxels (includes midline at index mid)
        if left_is_upper:
            start[x_axis] = mid
            extent[x_axis] = size[x_axis] - mid
        else:
            start[x_axis] = 0
            extent[x_axis] = mid + (1 if has_midline else 0)

    elif midline in ("right", "bilateral"):
        # Exclude midline from negation (strict left hemisphere only)
        # Only executes for odd dimensions (has_midline=True)
        # Even dimensions are handled by the first branch above
        if left_is_upper:
            start[x_axis] = mid + 1  # Skip midline voxel
            extent[x_axis] = size[x_axis] - mid - 1
        else:
            start[x_axis] = 0
            extent[x_axis] = mid  # Stop before midline

    return start, extent


def _sitk_roi_numpy_slices(start: list[int], extent: list[int]) -> tuple[slice, ...]:
    """
    Convert SITK ROI coordinates to NumPy array slices.

    SimpleITK uses (x, y, z) index ordering while NumPy arrays from
    ``sitk.GetArrayViewFromImage()`` use (z, y, x) ordering. This function
    performs the necessary axis reversal.

    Parameters
    ----------
    start : list of int
        Starting indices for the ROI in SITK order.
    extent : list of int
        Extent (size) of the ROI in SITK order.

    Returns
    -------
    tuple of slice
        Slice objects in NumPy array order, suitable for indexing with
        ``arr[slices]``.

    Examples
    --------
    >>> start = [10, 20, 30]  # x, y, z in SITK
    >>> extent = [50, 60, 70]
    >>> slices = _sitk_roi_numpy_slices(start, extent)
    >>> # slices = (slice(30, 100), slice(20, 80), slice(10, 60))
    >>> # Can be used as: arr[slices] for z, y, x indexing
    """
    # SITK order → slices
    sitk_slices = [slice(s, s + e) for s, e in zip(start, extent)]
    # NumPy array is reversed axis order: (z,y,x) ← reverse of (x,y,z)
    return tuple(sitk_slices[::-1])


def _compact_labels_np(
    arr: NDArray[np.integer], compact_dtype: type[np.unsignedinteger[Any]] = np.uint16
) -> tuple[NDArray[np.unsignedinteger], NDArray[np.integer]]:
    """
    Create a compacted label array from a NumPy array.

    Remaps annotation labels to a compact range [0, N-1] where N is the
    number of unique labels. The output array is uint16 type, supporting
    up to 65,535 unique labels.

    Parameters
    ----------
    arr : NDArray[np.integer]
        NumPy array containing annotation labels.
    compact_dtype : type, optional
        NumPy integer type for the compacted output array. Default is
        np.uint16.

    Returns
    -------
    compacted_arr : NDArray[np.uint16]
        NumPy array with labels remapped to [0, N-1], type uint16.
    unique_labels : NDArray[np.integer]
        Array of unique annotation values from the input, sorted in
        ascending order. Used to map compacted indices back to originals.

    Notes
    -----
    The mapping satisfies: ``original_arr = unique_labels[compacted_arr]``
    """
    mask = arr == 0
    if not mask.any():
        # No background, compact all labels
        unq_labels, inverse = np.unique(arr, return_inverse=True)
        K = int(unq_labels.size)
        if K >= np.iinfo(compact_dtype).max:
            raise ValueError(
                f"Too many unique labels ({K}) to compact into {compact_dtype} range."
            )
        compacted = inverse.astype(compact_dtype)
    elif mask.all():
        # All background
        unq_labels = np.array([0], dtype=arr.dtype)
        compacted = np.zeros_like(arr, dtype=compact_dtype)
    else:
        # Compact non-background labels
        other_vals = arr[~mask]
        unq_labels_non_bg, inverse = np.unique(other_vals, return_inverse=True)

        K = int(unq_labels_non_bg.size)
        if K > np.iinfo(compact_dtype).max:
            raise ValueError(
                f"Too many unique labels ({K + 1}) to compact "
                f"into {compact_dtype} range."
            )
        compacted = np.zeros_like(arr, dtype=compact_dtype)
        unq_inverse_compacted = inverse.astype(compact_dtype)
        unq_inverse_compacted += 1  # shift by 1 to reserve 0 for background
        compacted[~mask] = unq_inverse_compacted
        unq_labels = np.concatenate([np.array([0], dtype=arr.dtype), unq_labels_non_bg])
    return compacted, unq_labels


def _compact_labels_image_np_to_sitk(
    arr: NDArray[np.integer], image: sitk.Image
) -> tuple[sitk.Image, NDArray[np.integer]]:
    """
    Create a compacted label image from a NumPy array.

    Remaps annotation labels to a compact range [0, N-1] where N is the
    number of unique labels. The output image is uint16 type, supporting
    up to 65,535 unique labels.

    Parameters
    ----------
    arr : NDArray[np.integer]
        NumPy array containing annotation labels.
    image : sitk.Image
        Reference SimpleITK image providing spatial metadata (origin,
        spacing, direction) to copy to the output.

    Returns
    -------
    compacted_img : sitk.Image
        SimpleITK image with labels remapped to [0, N-1], type uint16.
    unique_labels : NDArray[np.integer]
        Array of unique annotation values from the input, sorted in
        ascending order. Used to map compacted indices back to originals.

    Notes
    -----
    The mapping satisfies: ``original_arr = unique_labels[compacted_arr]``
    """
    compacted, unq_labels = _compact_labels_np(arr)
    compacted_img = sitk.GetImageFromArray(compacted)
    compacted_img.CopyInformation(image)
    return compacted_img, unq_labels


def compact_labels_image(
    anno_img: sitk.Image,
) -> tuple[sitk.Image, NDArray[np.integer]]:
    """
    Compact annotation labels to a contiguous range.

    Remaps annotation values in the input image to a compact range [0, N-1]
    where N is the number of unique labels. Useful for processing with tools
    that expect labels in a contiguous range or struggle with large integers.

    Parameters
    ----------
    anno_img : sitk.Image
        The input annotation image with potentially sparse label values.

    Returns
    -------
    compacted_img : sitk.Image
        A new SimpleITK image with labels remapped to [0, N-1], type uint16.
    unique_labels : NDArray[np.integer]
        Array of unique annotation values from the input, sorted in
        ascending order. Maps compacted indices back to original values.

    See Also
    --------
    expand_compacted_image : Reverse operation to restore original labels.
    map_annotations_safely : Apply ANTs transforms to annotation images.

    Examples
    --------
    Compact an annotation image and recover original values:

    >>> import SimpleITK as sitk
    >>> # Compact the labels
    >>> compact_img, unique_labels = compact_labels_image(anno_img)
    >>> # Later, expand back to original values
    >>> arr = sitk.GetArrayViewFromImage(compact_img)
    >>> original_values = unique_labels[arr]
    >>> original_img = sitk.GetImageFromArray(original_values)
    >>> original_img.CopyInformation(compact_img)

    Compacting is useful before ANTs processing:

    >>> compact_img, labels = compact_labels_image(ccf_annotation)
    >>> # Convert to ANTs and process...
    >>> # Then expand back
    >>> expanded = expand_compacted_image(processed_img, labels)
    """
    arr = sitk.GetArrayViewFromImage(anno_img)
    return _compact_labels_image_np_to_sitk(arr, anno_img)


def lateralize_and_compact_ccf_image(
    ccf_anno_img: sitk.Image,
    midline: str = "left",
) -> tuple[sitk.Image, NDArray[np.integer]]:
    """
    Lateralize and compact CCF annotations for hemisphere-aware processing.

    Creates a lateralized version of the CCF annotation atlas by negating
    all left hemisphere annotation values (conforming to IBL convention),
    then compacts the labels to a contiguous range [0, N-1] suitable for
    ANTs processing. Original values can be recovered using the returned
    mapping array.

    Parameters
    ----------
    ccf_anno_img : sitk.Image
        The input CCF annotation image. Must be 3D and approximately
        axis-aligned in LPS orientation.
    midline : {"left", "right", "bilateral"}, optional
        How to handle midline voxels (only relevant for odd dimensions).
        Default is "left".

        - "left": Midline voxels are treated as left hemisphere and will
          be negated. This is the default behavior for backward
          compatibility. With odd dimensions (e.g., 11 voxels), left gets
          6 voxels including the midline.
        - "right": Midline voxels are treated as right hemisphere and will
          NOT be negated. Left gets 5 voxels, right gets 6 voxels.
        - "bilateral": Midline voxels represent bilateral structures and
          will NOT be lateralized (preserve original positive values).
          Same spatial behavior as "right" but semantically indicates
          structures that span both hemispheres.

        For even dimensions, this parameter has no effect as there is no
        single midline voxel (each hemisphere gets an equal number).

    Returns
    -------
    compacted_img : sitk.Image
        SimpleITK image with lateralized labels compacted to [0, N-1],
        type uint16. Left hemisphere regions have been negated before
        compaction (excluding midline if mode is "right" or "bilateral").
    unique_labels : NDArray[np.integer]
        Array of unique annotation values in the lateralized image
        (including negative values for left hemisphere). Maps compacted
        indices back to lateralized annotation values.

    Raises
    ------
    ValueError
        If input image is not 3D, is not axis-aligned, or if midline
        parameter is invalid.

    See Also
    --------
    expand_compacted_image : Restore lateralized annotation values.
    compact_labels_image : Compact without lateralization.

    Notes
    -----
    The IBL (International Brain Laboratory) convention uses negative
    annotation values for left hemisphere regions to distinguish them
    from the corresponding right hemisphere regions.

    For a CCF atlas with M unique regions, the lateralized atlas will
    have up to 2M+1 unique values: positive values for right hemisphere,
    negative values for left hemisphere, and potentially 0 for midline
    structures.

    **Midline handling example** (size=11 along X, indices 0-10):

    - midline="left": Negate [5:11] (6 voxels) → midline at index 5 is
      treated as left and negated
    - midline="right": Negate [6:11] (5 voxels) → midline at index 5
      remains positive (treated as right)
    - midline="bilateral": Negate [6:11] (5 voxels) → midline at index 5
      remains positive (treated as bilateral/unlateralized)

    The "bilateral" option is useful for preserving annotations of
    structures that truly lie on the midline (e.g., 3rd ventricle,
    longitudinal fissure, superior sagittal sinus).

    Examples
    --------
    Standard lateralization with default behavior:

    >>> import SimpleITK as sitk
    >>> ccf_anno = sitk.ReadImage('annotation_10.nrrd')
    >>> # Lateralize and compact (midline goes to left)
    >>> compact_img, unique_labels = lateralize_and_compact_ccf_image(
    ...     ccf_anno
    ... )
    >>> # After processing, expand back to lateralized values
    >>> arr = sitk.GetArrayViewFromImage(compact_img)
    >>> lateralized_values = unique_labels[arr]
    >>> lateralized_img = sitk.GetImageFromArray(lateralized_values)
    >>> lateralized_img.CopyInformation(compact_img)

    Preserve midline structures as bilateral:

    >>> # Lateralize with bilateral midline
    >>> compact_img, unique_labels = lateralize_and_compact_ccf_image(
    ...     ccf_anno, midline="bilateral"
    ... )
    >>> # Midline voxels now keep their original positive annotation values
    >>> # distinguishing them from left (negative) and right (positive)

    Typical workflow with ANTs registration:

    >>> # Lateralize and compact CCF
    >>> compact_ccf, labels = lateralize_and_compact_ccf_image(
    ...     ccf_anno, midline="bilateral"
    ... )
    >>> # Convert to ANTs and warp to subject space
    >>> ants_ccf = sitk_to_ants(compact_ccf)
    >>> warped = ants.apply_transforms(subject, ants_ccf, transforms)
    >>> # Convert back and expand
    >>> warped_sitk = ants_to_sitk(warped)
    >>> final_anno = expand_compacted_image(warped_sitk, labels)
    """
    d = ccf_anno_img.GetDimension()
    if d != 3:
        raise ValueError("Input image must be 3D.")

    if midline not in ("left", "right", "bilateral"):
        raise ValueError(
            f"midline must be 'left', 'right', or 'bilateral', got {midline!r}"
        )

    # Get region to negate based on midline mode
    start, extent = _get_lateralization_regions(ccf_anno_img, midline)
    sl = _sitk_roi_numpy_slices(start, extent)

    # Negate left hemisphere
    arr = sitk.GetArrayViewFromImage(ccf_anno_img).astype(np.int64)  # make a copy
    arr[sl] *= -1  # negate left half in-place

    return _compact_labels_image_np_to_sitk(arr, ccf_anno_img)


def expand_compacted_image(
    compact_img: sitk.Image, unq_annotation_nos: NDArray[np.integer]
) -> sitk.Image:
    """
    Recover original annotation values from a compacted image.

    Takes a compacted annotation image (with labels in range [0, N-1])
    and restores the original annotation values using the provided
    mapping array.

    Parameters
    ----------
    compact_img : sitk.Image
        The compacted annotation image with labels in range [0, N-1].
    unq_annotation_nos : NDArray[np.integer]
        Array of unique annotation values that maps compacted indices
        back to original values. This is the second return value from
        ``compact_labels_image`` or ``lateralize_and_compact_ccf_image``.

    Returns
    -------
    sitk.Image
        A new SimpleITK image with the original annotation values restored.
        Spatial metadata (origin, spacing, direction) is copied from the
        compacted image.

    See Also
    --------
    compact_labels_image : Create compacted image and mapping array.
    lateralize_and_compact_ccf_image : Lateralize and compact CCF atlas.

    Examples
    --------
    Basic usage with compact_labels_image:

    >>> import SimpleITK as sitk
    >>> # Compact labels
    >>> compact_img, unique_labels = compact_labels_image(anno_img)
    >>> # ... process the compacted image ...
    >>> # Expand back to original values
    >>> restored_img = expand_compacted_image(compact_img, unique_labels)

    With lateralized CCF annotations:

    >>> # Lateralize and compact
    >>> compact_ccf, labels = lateralize_and_compact_ccf_image(ccf_anno)
    >>> # ... warp to subject space ...
    >>> warped_compact = apply_transforms(compact_ccf, transforms)
    >>> # Expand to get lateralized annotations in subject space
    >>> subject_anno = expand_compacted_image(warped_compact, labels)
    >>> # Left hemisphere regions will have negative annotation values

    Notes
    -----
    The expansion is performed via array indexing:
    ``expanded_values = unq_annotation_nos[compacted_array]``

    This is the inverse operation of the compaction performed by
    ``np.unique(..., return_inverse=True)``.
    """
    arr = sitk.GetArrayViewFromImage(compact_img)
    original_values = unq_annotation_nos[arr]
    expanded_img = sitk.GetImageFromArray(original_values)
    expanded_img.CopyInformation(compact_img)
    return expanded_img
