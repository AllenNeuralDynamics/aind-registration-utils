import ants
import numpy as np
import scipy.ndimage as ni
from skimage.filters import threshold_li
from skimage.measure import label


def get_largest_cc(segmentation):
    """
    Return the largest connected component from a binary segmentation.


    Parameters
    ----------
    segmentation : numpy.ndarray
        Binary array (2D or 3D) where non-zero elements denote foreground.

    Returns
    -------
    numpy.ndarray
        A binary mask of the same shape as `segmentation` containing only the
        largest connected component.

    Raises
    ------
    AssertionError
        If `segmentation` contains no foreground (all zeros).
    """
    labels = label(segmentation)
    if labels.max() == 0:
        raise ValueError(
            "segmentation must contain at least one connected component"
        )
    # bincount of flattened labels, skip background count at index 0
    largest_cc_index = np.argmax(np.bincount(labels.flat)[1:]) + 1
    return labels == largest_cc_index


def perc_normalization(ants_img, percentiles=None):
    if percentiles is None:
        percentiles = [2, 98]
    """
    Apply percentile normalization to an ANTs image using the 2nd and 98th
    percentiles.

    Parameters
    ----------
    ants_img : ants.core.ants_image.ANTsImage
        The input image to normalize.
    percentiles : list or array (2,)
        Min and max percentile to use for normalization.

    Returns
    -------
    ants.core.ants_image.ANTsImage
        The percentile-normalized image with the same metadata (spacing,
        origin, direction).
    """
    img = ants_img.numpy()

    percentile_values = np.percentile(img, percentiles)

    img = (img - percentile_values[0]) / (
        percentile_values[1] - percentile_values[0]
    )

    # convert numpy array to ants image
    out = ants.from_numpy(
        img.astype("float32"),
        spacing=ants_img.spacing,
        origin=ants_img.origin,
        direction=ants_img.direction,
    )
    return out


def get_threshold_li(arr_img: np.ndarray):
    """
    Compute the optimal Li threshold for a 3D image array.

    Parameters
    ----------
    arr_img : numpy.ndarray
        The input image data as a NumPy array.

    Returns
    -------
    float
        The computed Li threshold value.
    """
    low_thresh = threshold_li(arr_img)
    return low_thresh


def cleanup_mask(arr_mask: np.ndarray):
    """
    Clean up a binary mask by filling holes, dilating, closing, and keeping the
    largest component.

    The following morphological operations are applied in sequence:
      1. Fill holes in the mask
      2. Dilate with a 3x3 structuring element (connectivity=2)
      3. Close the mask
      4. Retain only the largest connected component

    Parameters
    ----------
    arr_mask : numpy.ndarray
        A binary mask array to be cleaned (0s and 1s).

    Returns
    -------
    numpy.ndarray
        The cleaned binary mask.
    """
    # 3x3 structuring element with connectivity 2
    struct = ni.generate_binary_structure(3, 2)

    mask = ni.binary_fill_holes(arr_mask).astype(int)
    mask = ni.binary_dilation(mask, structure=struct).astype(int)
    mask = ni.binary_closing(mask).astype(int)
    mask = get_largest_cc(mask)

    return mask


def get_mask(ants_img):
    """
    Generate a cleaned binary mask for an ANTs image using Li thresholding and
    morphological cleanup.

    Parameters
    ----------
    ants_img : ants.core.ants_image.ANTsImage
        The input image from which to compute the mask.

    Returns
    -------
    ants.core.ants_image.ANTsImage
        The binary mask as an ANTs image, with the same spacing, origin, and
        direction.
    """
    # convert ants image to numpy array
    arr_img = ants_img.numpy()

    # get optimal threshold using Li thresholding
    low_thresh = get_threshold_li(arr_img)

    # thresholding
    arr_mask = arr_img > low_thresh

    # clean up
    arr_mask = cleanup_mask(arr_mask)

    # convert numpy array to ants image
    ants_img_mask = ants.from_numpy(
        arr_mask.astype("float32"),
        spacing=ants_img.spacing,
        origin=ants_img.origin,
        direction=ants_img.direction,
    )
    return ants_img_mask


def reflect_ants_image(image, axis=0):
    """
    Reflect (flip) an ANTs image along the specified axis.

    Parameters
    ----------
    image : ants.core.ants_image.ANTsImage
        The input image to reflect.
    axis : int, optional
        The axis along which to flip (0=LR, 1=AP, 2=SI). Defaults to 0.

    Returns
    -------
    ants.core.ants_image.ANTsImage
        The reflected image with original metadata preserved.
    """
    vol = image.view()
    vol = np.flip(vol, axis=axis)
    return ants.copy_image_info(image, ants.from_numpy(vol))


def fast_n4_preprocesses(
    image,
    resample_spacing=None,
    level=2,
    output_filename=None,
    flip_lr=False,
    flip_ap=False,
    spline_size=15,
):
    if resample_spacing is None:
        resample_spacing = [0.1, 0.1, 0.1]
    """
    Quickly preprocess a Zarr-based image for template building:
      - Load and optionally flip left-right or anterior-posterior
      - Downsample for N4 bias field correction
      - Compute and apply N4 bias field
      - Normalize intensities
      - Save result if an output path is given

    Parameters
    ----------
    input: ants.core.ants_image.ANTsImage
        Image to preprocess
    resample_spacing : list of float, optional
        Spacing to use for downsampling (in mm). Defaults to [0.1, 0.1, 0.1].
    level : int, optional
        Mipmap level to load from the Zarr dataset. Defaults to 2.
    output_filename : str or pathlib.Path, optional
        File path to write the preprocessed image. If None, no file is written.
        Defaults to None.
    flip_lr : bool, optional
        Whether to apply a left-right flip before processing. Defaults to
        False.
    flip_ap : bool, optional
        Whether to apply an anterior-posterior flip before processing. Defaults
        to False.
    spline_size: float, optional
        Spline size for smoothing, default is 15 mm
    Returns
    -------
    ants.core.ants_image.ANTsImage
        The final preprocessed image.

    Raises
    ------
    FileNotFoundError
        If the input Zarr file does not exist or cannot be read.
    """

    # Optional flips
    if flip_lr:
        image = reflect_ants_image(image, axis=0)
    if flip_ap:
        image = reflect_ants_image(image, axis=1)

    # Downsample for N4
    image_down = ants.resample_image(image, resample_spacing)

    # Compute mask
    mask = get_mask(image_down)

    # Compute N4 bias field
    n4_field_down = ants.n4_bias_field_correction(
        image_down, mask=mask, spline_param=spline_size, return_bias_field=True
    )

    # Upsample bias field
    n4_field = ants.resample_image_to_target(n4_field_down, image)
    n4_field.view()[n4_field.view() == 0] = 1

    # Apply bias field and normalize
    image_n4 = image / n4_field
    image_percn4 = perc_normalization(image_n4)

    # Save result
    if output_filename is not None:
        ants.image_write(image_percn4, output_filename)

    return image_percn4
