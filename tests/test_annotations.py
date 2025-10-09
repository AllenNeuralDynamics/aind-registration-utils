# tests/test_annotations.py
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest
import SimpleITK as sitk


# ---- Minimal fake ANTs shim -------------------------------------------------
class FakeAntsImage:
    """
    Tiny stand-in for ants.ANTsImage with just enough behavior for tests.

    - Holds a numpy array
    - Stores simple metadata (spacing/origin/direction)
    - .view() returns the ndarray (mirrors ANTsPy behavior)
    - .astype() returns a new FakeAntsImage with converted dtype
    """

    def __init__(self, array, spacing=None, origin=None, direction=None):
        arr = np.asarray(array)
        self._arr = arr
        self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
        self.origin = origin if origin is not None else (0.0, 0.0, 0.0)
        # ANTs typically stores a flattened direction cosine matrix
        if direction is None:
            eye = np.eye(arr.ndim, dtype=float).ravel()
            self.direction = tuple(eye)
        else:
            self.direction = tuple(direction)

    def view(self):
        return self._arr

    def astype(self, dtype) -> FakeAntsImage:
        """Mimic ants image astype by returning a new image with same metadata."""
        return FakeAntsImage(
            self._arr.astype(dtype),
            spacing=self.spacing,
            origin=self.origin,
            direction=self.direction,
        )

    def __repr__(self) -> str:
        return f"FakeAntsImage(shape={self._arr.shape}, dtype={self._arr.dtype})"


def fake_from_numpy(arr) -> FakeAntsImage:
    return FakeAntsImage(np.asarray(arr))


def fake_copy_image_info(src: FakeAntsImage, dest: FakeAntsImage) -> FakeAntsImage:
    """Copy spatial metadata FROM src TO dest and return dest."""
    dest.spacing = getattr(src, "spacing", (1.0, 1.0, 1.0))
    dest.origin = getattr(src, "origin", (0.0, 0.0, 0.0))
    ndim = dest.view().ndim
    default_dir = tuple(np.eye(ndim, dtype=float).ravel())
    dest.direction = getattr(src, "direction", default_dir)
    return dest


def fake_apply_transforms(
    fixed: FakeAntsImage,
    moving: FakeAntsImage,
    transformlist=None,
    interpolator: str = "nearestNeighbor",
    **_: object,
) -> FakeAntsImage:
    """
    For unit tests we don't need geometric warping; we only want to validate the
    label mapping round-trip. So we return the moving image with fixed's metadata.
    """
    # Ensure nearest for label images
    assert interpolator == "nearestNeighbor"
    moved = FakeAntsImage(moving.view())
    fake_copy_image_info(fixed, moved)
    return moved


def _install_fake_ants_in_sys_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> types.SimpleNamespace:
    fake_ants = types.SimpleNamespace(
        from_numpy=fake_from_numpy,
        copy_image_info=fake_copy_image_info,
        apply_transforms=fake_apply_transforms,
    )
    # Ensure any import of 'ants' grabs our shim
    monkeypatch.setitem(sys.modules, "ants", fake_ants)
    return fake_ants


# ---- Fixture (function-scoped to match monkeypatch) -------------------------
@pytest.fixture()
def annotations_module(monkeypatch: pytest.MonkeyPatch):
    """
    Import a fresh copy of aind_registration_utils.annotations with our fake
    'ants' module injected BEFORE import, so the module binds to the shim.
    """
    _install_fake_ants_in_sys_modules(monkeypatch)
    # Re-import module cleanly each test
    modname = "aind_registration_utils.annotations"
    if modname in sys.modules:
        del sys.modules[modname]
    mod = importlib.import_module(modname)
    return mod


# ---- Tests ------------------------------------------------------------------
def test_roundtrip_preserves_labels_small_ints(annotations_module) -> None:
    map_annotations_safely = annotations_module.map_annotations_safely

    # Small integer labels with repeats
    data = np.array([[0, 3, 3], [2, 2, 1]], dtype=np.int32)
    moving = FakeAntsImage(data, spacing=(2.0, 2.0, 2.0), origin=(1.0, 1.0, 1.0))
    fixed = FakeAntsImage(np.zeros_like(data))

    out = map_annotations_safely(moving, fixed, transformlist=[])

    # Values unchanged
    np.testing.assert_array_equal(out.view(), data)
    # Output should carry fixed's spatial metadata
    assert out.spacing == fixed.spacing
    assert out.origin == fixed.origin
    assert out.direction == fixed.direction
    # dtype preserved (important for downstream)
    assert out.view().dtype == data.dtype


def test_roundtrip_preserves_huge_integer_labels(annotations_module) -> None:
    map_annotations_safely = annotations_module.map_annotations_safely

    # Mix of small and very large labels; use unsigned to test > 2**31
    huge_vals = np.array(
        [0, 42, 2**40 + 123, 2**48 + 7, 999_999_999_999],
        dtype=np.uint64,
    )
    data = huge_vals.reshape(1, -1)
    moving = FakeAntsImage(data, spacing=(0.5, 0.5, 0.5))
    fixed = FakeAntsImage(np.zeros_like(data))

    out = map_annotations_safely(moving, fixed, transformlist=[])

    np.testing.assert_array_equal(out.view(), data)
    assert out.spacing == fixed.spacing
    # dtype preserved (should remain uint64 here)
    assert out.view().dtype == data.dtype


def test_inputs_not_mutated(annotations_module) -> None:
    map_annotations_safely = annotations_module.map_annotations_safely

    data = np.array([[10, 10], [20, 30]], dtype=np.int64)
    moving = FakeAntsImage(data.copy(), spacing=(3.0, 3.0, 3.0))
    fixed = FakeAntsImage(np.zeros_like(data), spacing=(1.0, 1.0, 1.0))

    # Keep original copies for comparison
    moving_before = moving.view().copy()
    fixed_before_spacing = fixed.spacing

    _ = map_annotations_safely(moving, fixed, transformlist=[])

    # Ensure original arrays and metadata unchanged
    np.testing.assert_array_equal(moving.view(), moving_before)
    assert fixed.spacing == fixed_before_spacing


def test_uses_nearest_neighbor_interpolator(
    annotations_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure nearest-neighbor is used when warping the temporary integer image.
    Our fake apply_transforms asserts this; here we also verify call was made.
    """
    calls = {"count": 0}

    def spying_apply_transforms(
        fixed: FakeAntsImage,
        moving: FakeAntsImage,
        transformlist=None,
        interpolator: str = "nearestNeighbor",
        **kwargs: object,
    ) -> FakeAntsImage:
        calls["count"] += 1
        return fake_apply_transforms(
            fixed,
            moving,
            transformlist=transformlist,
            interpolator=interpolator,
            **kwargs,
        )

    fake_ants = sys.modules["ants"]
    monkeypatch.setattr(
        fake_ants,
        "apply_transforms",
        spying_apply_transforms,
        raising=True,
    )

    map_annotations_safely = annotations_module.map_annotations_safely
    data = np.array([[1, 2], [3, 4]], dtype=np.int16)
    moving = FakeAntsImage(data)
    fixed = FakeAntsImage(np.zeros_like(data))

    _ = map_annotations_safely(moving, fixed, transformlist=["dummy"])

    assert calls["count"] == 1


# Note: Line 119 in annotations.py (ValueError for invalid labels) is defensive
# code that's unreachable in practice. Line 108 uses array indexing which ensures
# all warped values come from original_index. We accept 98% coverage for this module.


# ---- Fixtures for SimpleITK-based tests -------------------------------------


def make_direction_matrix(
    permutation: tuple[int, int, int] = (0, 1, 2),
    flips: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[float, ...]:
    """
    Generate direction matrix for permutation and/or flips.

    Parameters
    ----------
    permutation : tuple[int, int, int]
        Which input axis maps to each output axis. Default (0,1,2) is identity.
    flips : tuple[bool, bool, bool]
        Whether to flip each axis. Default (False, False, False) is no flip.

    Returns
    -------
    tuple[float, ...]
        Flattened 3x3 direction matrix.
    """
    D = np.zeros((3, 3))
    for i, axis in enumerate(permutation):
        D[axis, i] = -1.0 if flips[i] else 1.0
    return tuple(D.ravel())


@pytest.fixture
def synthetic_brain_3d() -> sitk.Image:
    """
    Create a simple 3D brain with clear left/right split.

    Left half (X > midpoint): label 100
    Right half (X <= midpoint): label 200
    """
    arr = np.zeros((10, 10, 10), dtype=np.int32)
    # In NumPy array (z, y, x), so x is the last axis
    arr[:, :, 5:] = 100  # Left half in LPS (+X = left)
    arr[:, :, :5] = 200  # Right half
    img = sitk.GetImageFromArray(arr)
    # LPS orientation: +X = Left, +Y = Posterior, +Z = Superior
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection(make_direction_matrix())  # Identity = LPS
    return img


@pytest.fixture
def ccf_like_labels() -> sitk.Image:
    """
    Create realistic CCF-like annotation with ~100 unique regions.
    """
    arr = np.random.randint(0, 100, size=(20, 20, 20), dtype=np.int32)
    # Ensure we have some repeated labels
    arr[0:5, :, :] = 42
    arr[5:10, :, :] = 1000
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.1, 0.1, 0.1))
    img.SetOrigin((-5.0, -5.0, -5.0))
    img.SetDirection(make_direction_matrix())
    return img


# ---- Tests for _sitk_roi_numpy_slices ----------------------------------------


def test_sitk_roi_numpy_slices_3d():
    """Test SITK (x,y,z) to NumPy (z,y,x) conversion."""
    from aind_registration_utils.annotations import (
        _sitk_roi_numpy_slices,
    )

    start = [10, 20, 30]  # SITK order: x, y, z
    extent = [5, 6, 7]
    slices = _sitk_roi_numpy_slices(start, extent)

    # Should be reversed: (z, y, x)
    assert slices[0] == slice(30, 37)  # z: start=30, extent=7
    assert slices[1] == slice(20, 26)  # y: start=20, extent=6
    assert slices[2] == slice(10, 15)  # x: start=10, extent=5


def test_sitk_roi_numpy_slices_2d():
    """Test 2D case."""
    from aind_registration_utils.annotations import (
        _sitk_roi_numpy_slices,
    )

    start = [5, 10]  # x, y
    extent = [3, 4]
    slices = _sitk_roi_numpy_slices(start, extent)

    # Reversed: (y, x)
    assert slices[0] == slice(10, 14)  # y
    assert slices[1] == slice(5, 8)  # x


def test_sitk_roi_numpy_slices_zero_extent():
    """Test with zero extent (degenerate ROI)."""
    from aind_registration_utils.annotations import (
        _sitk_roi_numpy_slices,
    )

    start = [0, 0, 0]
    extent = [0, 0, 0]
    slices = _sitk_roi_numpy_slices(start, extent)

    assert slices[0] == slice(0, 0)
    assert slices[1] == slice(0, 0)
    assert slices[2] == slice(0, 0)


# ---- Tests for _get_lateralization_regions ----------------------------------


def test_get_lateralization_regions_standard_lps():
    """Test standard LPS orientation (+X = LEFT) with default midline."""
    from aind_registration_utils.annotations import (
        _get_lateralization_regions,
    )

    arr = np.zeros((10, 12, 20), dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    # Size in SITK is (x=20, y=12, z=10)
    img.SetDirection(make_direction_matrix())  # Identity = LPS

    start, extent = _get_lateralization_regions(img, midline="left")

    # +X = LEFT, so left is upper half of X axis
    # X axis is at index 0, size=20, mid=10
    # Even dimensions: left from index 10 to 20
    assert start[0] == 10  # X start
    assert extent[0] == 10  # X extent
    assert start[1] == 0  # Y unchanged
    assert extent[1] == 12
    assert start[2] == 0  # Z unchanged
    assert extent[2] == 10


def test_get_lateralization_regions_flipped_x():
    """Test with X axis flipped (-X = LEFT)."""
    from aind_registration_utils.annotations import (
        _get_lateralization_regions,
    )

    arr = np.zeros((10, 12, 20), dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    # Flip X axis: -X = LEFT (RAS-like)
    img.SetDirection(make_direction_matrix(flips=(True, False, False)))

    start, extent = _get_lateralization_regions(img, midline="left")

    # -X = LEFT, so left is lower half of X axis
    # Left should be from index 0 to 10
    assert start[0] == 0
    assert extent[0] == 10
    assert start[1] == 0
    assert extent[1] == 12
    assert start[2] == 0
    assert extent[2] == 10


@pytest.mark.parametrize(
    "size,midline,expected_start,expected_extent",
    [
        # Even dimensions: all modes behave the same
        (10, "left", 5, 5),
        (10, "right", 5, 5),
        (10, "bilateral", 5, 5),
        # Odd dimensions: different behavior
        (11, "left", 5, 6),  # Include midline in left
        (11, "right", 6, 5),  # Exclude midline (strict left)
        (11, "bilateral", 6, 5),  # Exclude midline (preserve as bilateral)
    ],
)
def test_get_lateralization_regions_midline_modes(
    size, midline, expected_start, expected_extent
):
    """Test different midline handling modes with even/odd dimensions."""
    from aind_registration_utils.annotations import (
        _get_lateralization_regions,
    )

    # Create image with specified X dimension (left_is_upper orientation)
    arr = np.zeros((10, 10, size), dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    # SITK size is (size, 10, 10)
    img.SetDirection(make_direction_matrix())  # LPS: left_is_upper=True

    start, extent = _get_lateralization_regions(img, midline=midline)

    # Check X axis (index 0 in SITK space)
    assert start[0] == expected_start
    assert extent[0] == expected_extent


@pytest.mark.parametrize(
    "size,midline,left_is_upper,expected_start,expected_extent",
    [
        # Even, upper
        (10, "left", True, 5, 5),
        # Odd, upper, midline="left"
        (11, "left", True, 5, 6),
        # Odd, upper, midline="bilateral"
        (11, "bilateral", True, 6, 5),
        # Even, lower
        (10, "left", False, 0, 5),
        # Odd, lower, midline="left"
        (11, "left", False, 0, 6),
        # Odd, lower, midline="bilateral"
        (11, "bilateral", False, 0, 5),
    ],
)
def test_get_lateralization_regions_orientation_and_midline(
    size, midline, left_is_upper, expected_start, expected_extent
):
    """Test midline modes with different left_is_upper orientations."""
    from aind_registration_utils.annotations import (
        _get_lateralization_regions,
    )

    arr = np.zeros((10, 10, size), dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)

    # Set direction based on left_is_upper
    if left_is_upper:
        img.SetDirection(make_direction_matrix())  # +X = LEFT
    else:
        img.SetDirection(make_direction_matrix(flips=(True, False, False)))  # -X = LEFT

    start, extent = _get_lateralization_regions(img, midline=midline)

    assert start[0] == expected_start
    assert extent[0] == expected_extent


def test_get_lateralization_regions_non_axis_aligned_error():
    """Test error on non-axis-aligned image."""
    from aind_registration_utils.annotations import (
        _get_lateralization_regions,
    )

    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    # 45-degree rotation (not axis-aligned)
    D = np.array([[0.707, -0.707, 0], [0.707, 0.707, 0], [0, 0, 1]]).ravel()
    img.SetDirection(tuple(D))

    with pytest.raises(
        ValueError,
        match="Not axis-aligned",
    ):
        _get_lateralization_regions(img, midline="left")


@pytest.mark.parametrize("midline", ["right", "bilateral"])
def test_get_lateralization_regions_even_with_right_modes(midline):
    """Test even dimensions with right/bilateral modes and left_is_lower."""
    from aind_registration_utils.annotations import (
        _get_lateralization_regions,
    )

    # Even dimensions with left_is_lower orientation
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    # -X = LEFT (left_is_lower)
    img.SetDirection(make_direction_matrix(flips=(True, False, False)))

    start, extent = _get_lateralization_regions(img, midline=midline)

    # Even dimensions: should behave same as "left" mode
    assert start[0] == 0
    assert extent[0] == 5


# ---- Tests for compact_labels_image ------------------------------------------


def test_compact_labels_basic():
    """Test basic label compaction."""
    from aind_registration_utils.annotations import (
        compact_labels_image,
    )

    # Sparse labels
    arr = np.array([[[0, 100, 100], [500, 500, 1000]]], dtype=np.int32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))

    compact_img, unique_labels = compact_labels_image(img)

    # Check compacted values are in range [0, N-1]
    compact_arr = sitk.GetArrayViewFromImage(compact_img)
    assert compact_arr.min() >= 0
    assert compact_arr.max() < len(unique_labels)

    # Check unique labels are correct
    assert set(unique_labels) == {0, 100, 500, 1000}

    # Check dtype is uint16
    assert compact_arr.dtype == np.uint16

    # Check spatial metadata preserved
    assert compact_img.GetSpacing() == img.GetSpacing()
    assert compact_img.GetOrigin() == img.GetOrigin()
    assert compact_img.GetDirection() == img.GetDirection()


def test_compact_labels_single_label():
    """Test with single unique label."""
    from aind_registration_utils.annotations import (
        compact_labels_image,
    )

    arr = np.ones((5, 5, 5), dtype=np.int32) * 42
    img = sitk.GetImageFromArray(arr)

    compact_img, unique_labels = compact_labels_image(img)

    compact_arr = sitk.GetArrayViewFromImage(compact_img)
    assert len(unique_labels) == 1
    assert unique_labels[0] == 42
    assert np.all(compact_arr == 0)  # Only one label → index 0


def test_compact_labels_with_negatives():
    """Test compaction with negative values (for lateralization)."""
    from aind_registration_utils.annotations import (
        compact_labels_image,
    )

    arr = np.array([[[-100, -100, 0], [100, 100, 200]]], dtype=np.int64)
    img = sitk.GetImageFromArray(arr)

    compact_img, unique_labels = compact_labels_image(img)

    # Should handle negative values correctly
    assert -100 in unique_labels
    assert 100 in unique_labels

    # Verify round-trip
    compact_arr = sitk.GetArrayViewFromImage(compact_img)
    recovered = unique_labels[compact_arr]
    np.testing.assert_array_equal(recovered, arr)


# ---- Tests for expand_compacted_image ----------------------------------------


def test_expand_compacted_roundtrip():
    """Test compact → expand round-trip."""
    from aind_registration_utils.annotations import (
        compact_labels_image,
        expand_compacted_image,
    )

    arr = np.array([[[0, 50, 50], [100, 200, 300]]], dtype=np.int32)
    original_img = sitk.GetImageFromArray(arr)
    original_img.SetSpacing((0.5, 0.5, 0.5))
    original_img.SetOrigin((1.0, 2.0, 3.0))

    # Compact
    compact_img, unique_labels = compact_labels_image(original_img)

    # Expand
    expanded_img = expand_compacted_image(compact_img, unique_labels)

    # Verify values match original
    expanded_arr = sitk.GetArrayViewFromImage(expanded_img)
    np.testing.assert_array_equal(expanded_arr, arr)

    # Verify metadata preserved
    assert expanded_img.GetSpacing() == compact_img.GetSpacing()
    assert expanded_img.GetOrigin() == compact_img.GetOrigin()
    assert expanded_img.GetDirection() == compact_img.GetDirection()


def test_expand_compacted_with_lateralized_values():
    """Test expansion with negative (lateralized) values."""
    from aind_registration_utils.annotations import (
        compact_labels_image,
        expand_compacted_image,
    )

    arr = np.array([[[-500, -500, 0], [500, 500, 1000]]], dtype=np.int64)
    img = sitk.GetImageFromArray(arr)

    compact_img, unique_labels = compact_labels_image(img)
    expanded_img = expand_compacted_image(compact_img, unique_labels)

    expanded_arr = sitk.GetArrayViewFromImage(expanded_img)
    np.testing.assert_array_equal(expanded_arr, arr)

    # Verify negative values preserved
    assert expanded_arr.min() == -500
    assert expanded_arr.max() == 1000


# ---- Tests for lateralize_and_compact_ccf_image ------------------------------


def test_lateralize_and_compact_basic(synthetic_brain_3d):
    """Test basic lateralization: left hemisphere negated."""
    from aind_registration_utils.annotations import (
        lateralize_and_compact_ccf_image,
    )

    compact_img, unique_labels = lateralize_and_compact_ccf_image(synthetic_brain_3d)

    # Check we have both positive and negative values in mapping
    assert np.any(unique_labels < 0)  # Left hemisphere (negated)
    assert np.any(unique_labels > 0)  # Right hemisphere

    # Original had labels 100 (left) and 200 (right)
    # After lateralization: -100 (left) and 200 (right)
    assert -100 in unique_labels
    assert 200 in unique_labels

    # Verify compacted image is uint16
    compact_arr = sitk.GetArrayViewFromImage(compact_img)
    assert compact_arr.dtype == np.uint16


def test_lateralize_and_compact_roundtrip(synthetic_brain_3d):
    """Test lateralize → compact → expand round-trip."""
    from aind_registration_utils.annotations import (
        expand_compacted_image,
        lateralize_and_compact_ccf_image,
    )

    # Lateralize and compact
    compact_img, unique_labels = lateralize_and_compact_ccf_image(synthetic_brain_3d)

    # Expand
    lateralized_img = expand_compacted_image(compact_img, unique_labels)
    lateralized_arr = sitk.GetArrayViewFromImage(lateralized_img)

    # Left hemisphere should be negated
    # NumPy array is (z, y, x), left is x >= 5
    assert np.all(lateralized_arr[:, :, 5:] == -100)  # Left
    assert np.all(lateralized_arr[:, :, :5] == 200)  # Right


def test_lateralize_non_3d_error():
    """Test error on non-3D input."""
    from aind_registration_utils.annotations import (
        lateralize_and_compact_ccf_image,
    )

    # 2D image
    arr = np.ones((10, 10), dtype=np.int32)
    img = sitk.GetImageFromArray(arr)

    with pytest.raises(ValueError, match="must be 3D"):
        lateralize_and_compact_ccf_image(img)


def test_lateralize_different_orientations():
    """Test lateralization with flipped X axis (RAS-like)."""
    from aind_registration_utils.annotations import (
        lateralize_and_compact_ccf_image,
    )

    # Create image with flipped X (-X = LEFT, like RAS)
    arr = np.zeros((10, 10, 10), dtype=np.int32)
    # In NumPy (z,y,x): when X is flipped, left is x < midpoint
    arr[:, :, :5] = 100  # Left half
    arr[:, :, 5:] = 200  # Right half

    img = sitk.GetImageFromArray(arr)
    img.SetDirection(make_direction_matrix(flips=(True, False, False)))

    compact_img, unique_labels = lateralize_and_compact_ccf_image(img)

    # Left should be negated
    assert -100 in unique_labels
    assert 200 in unique_labels


def test_lateralize_odd_dimensions():
    """Test with odd dimensions for asymmetric hemispheres."""
    from aind_registration_utils.annotations import (
        lateralize_and_compact_ccf_image,
    )

    # Create 11x10x10 image (odd X dimension)
    arr = np.zeros((10, 10, 11), dtype=np.int32)
    # SITK size is (11, 10, 10) - X has 11 voxels
    # mid = 11 // 2 = 5, so left is [5:11] (6 voxels)
    arr[:, :, 5:] = 100  # Left (6 voxels)
    arr[:, :, :5] = 200  # Right (5 voxels)

    img = sitk.GetImageFromArray(arr)
    img.SetDirection(make_direction_matrix())

    compact_img, unique_labels = lateralize_and_compact_ccf_image(img)

    assert -100 in unique_labels
    assert 200 in unique_labels


def test_lateralize_preserves_metadata():
    """Test that spatial metadata is preserved."""
    from aind_registration_utils.annotations import (
        lateralize_and_compact_ccf_image,
    )

    arr = np.ones((10, 10, 10), dtype=np.int32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.25, 0.25, 0.25))
    img.SetOrigin((10.0, 20.0, 30.0))

    compact_img, _ = lateralize_and_compact_ccf_image(img)

    assert compact_img.GetSpacing() == (0.25, 0.25, 0.25)
    assert compact_img.GetOrigin() == (10.0, 20.0, 30.0)


def test_lateralize_invalid_midline_error():
    """Test error on invalid midline parameter."""
    from aind_registration_utils.annotations import (
        lateralize_and_compact_ccf_image,
    )

    arr = np.ones((10, 10, 10), dtype=np.int32)
    img = sitk.GetImageFromArray(arr)

    with pytest.raises(
        ValueError, match="midline must be 'left', 'right', or 'bilateral'"
    ):
        lateralize_and_compact_ccf_image(img, midline="invalid")


@pytest.mark.parametrize("midline", ["left", "right", "bilateral"])
def test_lateralize_midline_modes_even_dimensions(midline):
    """Test that even dimensions work identically for all midline modes."""
    from aind_registration_utils.annotations import (
        lateralize_and_compact_ccf_image,
    )

    # Even dimensions: no midline voxel
    arr = np.zeros((10, 10, 10), dtype=np.int32)
    arr[:, :, 5:] = 100  # Left half
    arr[:, :, :5] = 200  # Right half
    img = sitk.GetImageFromArray(arr)
    img.SetDirection(make_direction_matrix())

    compact_img, unique_labels = lateralize_and_compact_ccf_image(img, midline=midline)

    # All modes should produce same result for even dimensions
    assert -100 in unique_labels  # Left negated
    assert 200 in unique_labels  # Right unchanged
    # Should have exactly 2 unique values (plus any zero background)
    non_zero_labels = unique_labels[unique_labels != 0]
    assert len(non_zero_labels) == 2


def test_lateralize_midline_left_mode_odd_dimensions():
    """Test midline='left' mode with odd dimensions."""
    from aind_registration_utils.annotations import (
        expand_compacted_image,
        lateralize_and_compact_ccf_image,
    )

    # Odd dimensions: 11 voxels along X
    arr = np.zeros((10, 10, 11), dtype=np.int32)
    # Set specific values to track midline
    arr[:, :, :5] = 200  # Right (indices 0-4)
    arr[:, :, 5] = 150  # Midline (index 5)
    arr[:, :, 6:] = 100  # Left (indices 6-10)

    img = sitk.GetImageFromArray(arr)
    img.SetDirection(make_direction_matrix())  # LPS: left_is_upper

    compact_img, unique_labels = lateralize_and_compact_ccf_image(img, midline="left")

    # Expand to check values
    lateralized_img = expand_compacted_image(compact_img, unique_labels)
    lateralized_arr = sitk.GetArrayViewFromImage(lateralized_img)

    # midline="left": indices [5:11] should be negated (6 voxels)
    assert np.all(lateralized_arr[:, :, 5] == -150)  # Midline negated
    assert np.all(lateralized_arr[:, :, 6:] == -100)  # Left negated
    assert np.all(lateralized_arr[:, :, :5] == 200)  # Right unchanged


def test_lateralize_midline_right_mode_odd_dimensions():
    """Test midline='right' mode with odd dimensions."""
    from aind_registration_utils.annotations import (
        expand_compacted_image,
        lateralize_and_compact_ccf_image,
    )

    # Odd dimensions: 11 voxels along X
    arr = np.zeros((10, 10, 11), dtype=np.int32)
    arr[:, :, :5] = 200  # Right (indices 0-4)
    arr[:, :, 5] = 150  # Midline (index 5)
    arr[:, :, 6:] = 100  # Left (indices 6-10)

    img = sitk.GetImageFromArray(arr)
    img.SetDirection(make_direction_matrix())  # LPS: left_is_upper

    compact_img, unique_labels = lateralize_and_compact_ccf_image(img, midline="right")

    # Expand to check values
    lateralized_img = expand_compacted_image(compact_img, unique_labels)
    lateralized_arr = sitk.GetArrayViewFromImage(lateralized_img)

    # midline="right": indices [6:11] should be negated (5 voxels)
    assert np.all(lateralized_arr[:, :, 5] == 150)  # Midline NOT negated
    assert np.all(lateralized_arr[:, :, 6:] == -100)  # Left negated
    assert np.all(lateralized_arr[:, :, :5] == 200)  # Right unchanged


def test_lateralize_midline_bilateral_mode_odd_dimensions():
    """Test midline='bilateral' mode with odd dimensions."""
    from aind_registration_utils.annotations import (
        expand_compacted_image,
        lateralize_and_compact_ccf_image,
    )

    # Odd dimensions: 11 voxels along X
    arr = np.zeros((10, 10, 11), dtype=np.int32)
    arr[:, :, :5] = 200  # Right (indices 0-4)
    arr[:, :, 5] = 150  # Midline (index 5) - bilateral structure
    arr[:, :, 6:] = 100  # Left (indices 6-10)

    img = sitk.GetImageFromArray(arr)
    img.SetDirection(make_direction_matrix())  # LPS: left_is_upper

    compact_img, unique_labels = lateralize_and_compact_ccf_image(
        img, midline="bilateral"
    )

    # Expand to check values
    lateralized_img = expand_compacted_image(compact_img, unique_labels)
    lateralized_arr = sitk.GetArrayViewFromImage(lateralized_img)

    # midline="bilateral": indices [6:11] negated, midline preserved
    assert np.all(lateralized_arr[:, :, 5] == 150)  # Midline preserved (bilateral)
    assert np.all(lateralized_arr[:, :, 6:] == -100)  # Left negated
    assert np.all(lateralized_arr[:, :, :5] == 200)  # Right unchanged

    # Verify semantic distinction: midline value is positive (bilateral)
    # distinguishable from negative (left) and positive (right)
    assert 150 in unique_labels  # Bilateral midline value
    assert -100 in unique_labels  # Left hemisphere
    assert 200 in unique_labels  # Right hemisphere


# ---- Integration tests -------------------------------------------------------


def test_full_lateralization_pipeline(ccf_like_labels):
    """Test complete workflow: lateralize → compact → expand."""
    from aind_registration_utils.annotations import (
        expand_compacted_image,
        lateralize_and_compact_ccf_image,
    )

    # Original data
    original_arr = sitk.GetArrayViewFromImage(ccf_like_labels)

    # Lateralize and compact
    compact_img, unique_labels = lateralize_and_compact_ccf_image(ccf_like_labels)

    # Verify compacted values are valid indices
    compact_arr = sitk.GetArrayViewFromImage(compact_img)
    assert compact_arr.min() >= 0
    assert compact_arr.max() < len(unique_labels)

    # Expand back
    lateralized_img = expand_compacted_image(compact_img, unique_labels)
    lateralized_arr = sitk.GetArrayViewFromImage(lateralized_img)

    # Verify left hemisphere is negative
    # SITK size is (20, 20, 20), mid = 10
    # Left is x >= 10 in NumPy indexing
    left_values = lateralized_arr[:, :, 10:]
    right_values = lateralized_arr[:, :, :10]

    # All left values should be negative or zero
    assert np.all(left_values <= 0)
    # All right values should be >= 0
    assert np.all(right_values >= 0)

    # Check that negation relationship holds
    # If original[left] was X, lateralized[left] should be -X
    original_left = original_arr[:, :, 10:]
    for orig_val in np.unique(original_left):
        if orig_val != 0:  # Skip zero
            assert -orig_val in unique_labels


def test_lateralization_with_realistic_ccf_scale():
    """Test with CCF-like number of regions (~1000 → 2000 lateralized)."""
    from aind_registration_utils.annotations import (
        lateralize_and_compact_ccf_image,
    )

    # Create image with many unique labels
    arr = np.random.randint(0, 1300, size=(20, 20, 20), dtype=np.int32)
    img = sitk.GetImageFromArray(arr)
    img.SetDirection(make_direction_matrix())

    compact_img, unique_labels = lateralize_and_compact_ccf_image(img)

    # Should have up to ~2600 unique values (including negatives)
    # Well within uint16 limit of 65535
    assert len(unique_labels) < 65535

    # Verify compacted image uses valid indices
    compact_arr = sitk.GetArrayViewFromImage(compact_img)
    assert compact_arr.max() < len(unique_labels)
