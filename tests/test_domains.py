"""Tests for domains module."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from aind_registration_utils.domains import (
    ImageDomainAxisAligned,
    ImageHeader,
    _check_size_override,
    make_syn_sidecar,
)

# ---- Test Utilities ---------------------------------------------------------


def create_mock_ants_image(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: tuple[float, ...] | None = None,
) -> Mock:
    """Create mock ANTsImage for testing without real image data."""
    if direction is None:
        direction = tuple(np.eye(3).ravel())

    mock_img = Mock()
    mock_img.shape = shape
    mock_img.spacing = spacing
    mock_img.origin = origin
    mock_img.direction = direction
    return mock_img


def create_mock_sitk_image(
    size: tuple[int, int, int],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: tuple[float, ...] | None = None,
) -> Mock:
    """Create mock SimpleITK Image for testing."""
    if direction is None:
        direction = tuple(np.eye(3).ravel())

    mock_img = Mock()
    mock_img.GetSize = Mock(return_value=size)
    mock_img.GetSpacing = Mock(return_value=spacing)
    mock_img.GetOrigin = Mock(return_value=origin)
    mock_img.GetDirection = Mock(return_value=direction)
    return mock_img


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

    Examples
    --------
    >>> # Swap X and Z axes
    >>> make_direction_matrix(permutation=(2, 1, 0))

    >>> # Flip Y axis
    >>> make_direction_matrix(flips=(False, True, False))
    """
    D = np.zeros((3, 3))
    for i, axis in enumerate(permutation):
        D[axis, i] = -1.0 if flips[i] else 1.0
    return tuple(D.ravel())


# ---- Tests for _check_size_override -----------------------------------------


class TestCheckSizeOverride(unittest.TestCase):
    """Tests for _check_size_override function."""

    def test_check_size_override_none(self):
        """Returns None for None input."""
        result = _check_size_override(None)
        self.assertIsNone(result)

    def test_check_size_override_valid(self):
        """Returns valid tuple unchanged."""
        size = (10, 20, 30)
        result = _check_size_override(size)
        self.assertEqual(result, size)

    def test_check_size_override_wrong_length(self):
        """Raises ValueError for wrong length tuple."""
        with self.assertRaises(ValueError) as ctx:
            _check_size_override((10, 20))
        self.assertIn("must be a tuple of 3 integers", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            _check_size_override((10, 20, 30, 40))
        self.assertIn("must be a tuple of 3 integers", str(ctx.exception))

    def test_check_size_override_zero_value(self):
        """Raises ValueError for zero value."""
        with self.assertRaises(ValueError) as ctx:
            _check_size_override((10, 0, 30))
        self.assertIn("must be positive integers", str(ctx.exception))

    def test_check_size_override_negative_value(self):
        """Raises ValueError for negative value."""
        with self.assertRaises(ValueError) as ctx:
            _check_size_override((10, -5, 30))
        self.assertIn("must be positive integers", str(ctx.exception))


# ---- Tests for ImageHeader --------------------------------------------------


class TestImageHeaderFromAnts(unittest.TestCase):
    """Tests for ImageHeader.from_ants method."""

    def test_from_ants_basic(self):
        """Extract header from standard ANTs image with identity direction."""
        img = create_mock_ants_image(
            shape=(10, 20, 30),
            spacing=(0.5, 1.0, 2.0),
            origin=(1.0, 2.0, 3.0),
        )

        header = ImageHeader.from_ants(img)

        self.assertEqual(header.size, (10, 20, 30))
        self.assertEqual(header.spacing, (0.5, 1.0, 2.0))
        self.assertEqual(header.origin, (1.0, 2.0, 3.0))
        self.assertEqual(len(header.direction), 9)

    def test_from_ants_with_size_override(self):
        """Override size parameter works correctly."""
        img = create_mock_ants_image(shape=(10, 20, 30))

        header = ImageHeader.from_ants(img, size_override=(5, 10, 15))

        self.assertEqual(header.size, (5, 10, 15))

    def test_from_ants_nonzero_origin(self):
        """Non-zero origin preserved correctly."""
        img = create_mock_ants_image(shape=(10, 10, 10), origin=(10.5, -5.3, 100.0))

        header = ImageHeader.from_ants(img)

        self.assertEqual(header.origin, (10.5, -5.3, 100.0))

    def test_from_ants_nonunit_spacing(self):
        """Non-unit spacing preserved correctly."""
        img = create_mock_ants_image(shape=(10, 10, 10), spacing=(0.25, 0.5, 2.0))

        header = ImageHeader.from_ants(img)

        self.assertEqual(header.spacing, (0.25, 0.5, 2.0))

    def test_from_ants_invalid_size_override(self):
        """Raises ValueError with invalid size_override."""
        img = create_mock_ants_image(shape=(10, 20, 30))

        with self.assertRaises(ValueError):
            ImageHeader.from_ants(img, size_override=(10, 20))

        with self.assertRaises(ValueError):
            ImageHeader.from_ants(img, size_override=(10, 0, 30))


class TestImageHeaderFromSitk(unittest.TestCase):
    """Tests for ImageHeader.from_sitk method."""

    def test_from_sitk_basic(self):
        """Extract header from standard SimpleITK image."""
        img = create_mock_sitk_image(
            size=(10, 20, 30),
            spacing=(0.5, 1.0, 2.0),
            origin=(1.0, 2.0, 3.0),
        )

        header = ImageHeader.from_sitk(img)

        self.assertEqual(header.size, (10, 20, 30))
        self.assertEqual(header.spacing, (0.5, 1.0, 2.0))
        self.assertEqual(header.origin, (1.0, 2.0, 3.0))
        self.assertEqual(len(header.direction), 9)

    def test_from_sitk_with_size_override(self):
        """Override size parameter works correctly."""
        img = create_mock_sitk_image(size=(10, 20, 30))

        header = ImageHeader.from_sitk(img, size_override=(5, 10, 15))

        self.assertEqual(header.size, (5, 10, 15))

    def test_from_sitk_ants_equivalence(self):
        """Both methods produce equivalent results for same geometry."""
        ants_img = create_mock_ants_image(
            shape=(10, 20, 30),
            spacing=(0.5, 1.0, 2.0),
            origin=(1.0, 2.0, 3.0),
        )
        sitk_img = create_mock_sitk_image(
            size=(10, 20, 30),
            spacing=(0.5, 1.0, 2.0),
            origin=(1.0, 2.0, 3.0),
        )

        header_ants = ImageHeader.from_ants(ants_img)
        header_sitk = ImageHeader.from_sitk(sitk_img)

        self.assertEqual(header_ants.size, header_sitk.size)
        self.assertEqual(header_ants.spacing, header_sitk.spacing)
        self.assertEqual(header_ants.origin, header_sitk.origin)
        self.assertEqual(header_ants.direction, header_sitk.direction)


# ---- Tests for ImageDomainAxisAligned ---------------------------------------


class TestImageDomainAxisAlignedFromHeader(unittest.TestCase):
    """Tests for ImageDomainAxisAligned.from_header method."""

    def test_from_header_identity(self):
        """Identity direction at origin."""
        header = ImageHeader(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=tuple(np.eye(3).ravel()),
            size=(10, 20, 30),
        )

        domain = ImageDomainAxisAligned.from_header(header)

        # Spacing should match input
        self.assertEqual(domain.spacing_LPS, (1.0, 1.0, 1.0))

        # Shape should match input
        self.assertEqual(domain.shape_LPS, (10, 20, 30))

        # Bounding box: with origin at 0 and unit spacing,
        # voxel centers range from 0 to N-1
        self.assertEqual(domain.bbox_L, (0.0, 9.0))
        self.assertEqual(domain.bbox_P, (0.0, 19.0))
        self.assertEqual(domain.bbox_S, (0.0, 29.0))

    def test_from_header_flipped_axes(self):
        """Direction with negative values (axis flips)."""
        # Flip the second axis
        direction = make_direction_matrix(flips=(False, True, False))
        header = ImageHeader(
            origin=(0.0, 10.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=direction,
            size=(10, 20, 30),
        )

        domain = ImageDomainAxisAligned.from_header(header)

        # Spacing should be absolute value
        self.assertEqual(domain.spacing_LPS, (1.0, 1.0, 1.0))

        # Bounding box should be sorted correctly
        # Y axis flipped: origin at 10, goes to 10 - 19 = -9
        self.assertEqual(domain.bbox_L, (0.0, 9.0))
        self.assertEqual(domain.bbox_P, (-9.0, 10.0))
        self.assertEqual(domain.bbox_S, (0.0, 29.0))

    def test_from_header_permuted_axes(self):
        """Axes permutation (e.g., swap first and last)."""
        # Permute: Z, Y, X -> X, Y, Z becomes Z, Y, X
        direction = make_direction_matrix(permutation=(2, 1, 0))
        header = ImageHeader(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 2.0, 3.0),
            direction=direction,
            size=(10, 20, 30),
        )

        domain = ImageDomainAxisAligned.from_header(header)

        # Spacing should be reordered: original was (1, 2, 3) for (X, Y, Z)
        # After permutation (2,1,0), LPS order becomes (3, 2, 1)
        self.assertEqual(domain.spacing_LPS, (3.0, 2.0, 1.0))

        # Shape should be reordered: original (10, 20, 30)
        # becomes (30, 20, 10)
        self.assertEqual(domain.shape_LPS, (30, 20, 10))

    def test_from_header_nonunit_spacing(self):
        """Spacing affects bbox calculation."""
        header = ImageHeader(
            origin=(0.0, 0.0, 0.0),
            spacing=(0.5, 2.0, 0.25),
            direction=tuple(np.eye(3).ravel()),
            size=(10, 20, 30),
        )

        domain = ImageDomainAxisAligned.from_header(header)

        self.assertEqual(domain.spacing_LPS, (0.5, 2.0, 0.25))

        # Bbox should be scaled by spacing
        # X: 0 to (10-1)*0.5 = 4.5
        # Y: 0 to (20-1)*2.0 = 38.0
        # Z: 0 to (30-1)*0.25 = 7.25
        self.assertAlmostEqual(domain.bbox_L[0], 0.0)
        self.assertAlmostEqual(domain.bbox_L[1], 4.5)
        self.assertAlmostEqual(domain.bbox_P[0], 0.0)
        self.assertAlmostEqual(domain.bbox_P[1], 38.0)
        self.assertAlmostEqual(domain.bbox_S[0], 0.0)
        self.assertAlmostEqual(domain.bbox_S[1], 7.25)

    def test_from_header_nonzero_origin(self):
        """Origin offset affects bbox."""
        header = ImageHeader(
            origin=(10.0, 20.0, 30.0),
            spacing=(1.0, 1.0, 1.0),
            direction=tuple(np.eye(3).ravel()),
            size=(10, 10, 10),
        )

        domain = ImageDomainAxisAligned.from_header(header)

        # Bbox should be translated by origin
        self.assertEqual(domain.bbox_L, (10.0, 19.0))
        self.assertEqual(domain.bbox_P, (20.0, 29.0))
        self.assertEqual(domain.bbox_S, (30.0, 39.0))

    def test_from_header_single_voxel(self):
        """Size (1, 1, 1) edge case."""
        header = ImageHeader(
            origin=(5.0, 10.0, 15.0),
            spacing=(1.0, 1.0, 1.0),
            direction=tuple(np.eye(3).ravel()),
            size=(1, 1, 1),
        )

        domain = ImageDomainAxisAligned.from_header(header)

        # Single voxel: min equals max (single point)
        self.assertEqual(domain.bbox_L, (5.0, 5.0))
        self.assertEqual(domain.bbox_P, (10.0, 10.0))
        self.assertEqual(domain.bbox_S, (15.0, 15.0))

    def test_from_header_oblique_raises(self):
        """Rotated direction matrix raises ValueError."""
        # 45-degree rotation around Z axis
        angle = np.pi / 4
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        header = ImageHeader(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=tuple(rotation_matrix.ravel()),
            size=(10, 10, 10),
        )

        with self.assertRaises(ValueError) as ctx:
            ImageDomainAxisAligned.from_header(header)

        self.assertIn("Oblique", str(ctx.exception))

    def test_from_header_tolerance_parameter(self):
        """Tolerance controls validation."""
        # Nearly axis-aligned (0.999 instead of 1.0)
        nearly_identity = np.eye(3)
        nearly_identity[0, 0] = 0.999

        header = ImageHeader(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=tuple(nearly_identity.ravel()),
            size=(10, 10, 10),
        )

        # Should fail with strict tolerance
        with self.assertRaises(ValueError):
            ImageDomainAxisAligned.from_header(header, tol=1e-6)

        # Should pass with relaxed tolerance
        domain = ImageDomainAxisAligned.from_header(header, tol=0.01)
        self.assertIsNotNone(domain)


class TestImageDomainAxisAlignedToSidecar(unittest.TestCase):
    """Tests for ImageDomainAxisAligned.to_sidecar method."""

    def test_to_sidecar_complete(self):
        """Converts to Domain object correctly."""
        domain = ImageDomainAxisAligned(
            spacing_LPS=(1.0, 2.0, 3.0),
            bbox_L=(0.0, 10.0),
            bbox_P=(5.0, 15.0),
            bbox_S=(10.0, 20.0),
            shape_LPS=(11, 6, 4),
        )

        sidecar_domain = domain.to_sidecar()

        # Verify spacing transferred
        self.assertEqual(sidecar_domain.spacing_LPS, (1.0, 2.0, 3.0))

        # Verify BBox has L, P, S attributes
        self.assertEqual(sidecar_domain.bbox.L, (0.0, 10.0))
        self.assertEqual(sidecar_domain.bbox.P, (5.0, 15.0))
        self.assertEqual(sidecar_domain.bbox.S, (10.0, 20.0))

        # Verify shape_canonical matches shape_LPS
        self.assertEqual(sidecar_domain.shape_canonical, (11, 6, 4))


# ---- Tests for make_syn_sidecar ---------------------------------------------


class TestMakeSynSidecar(unittest.TestCase):
    """Tests for make_syn_sidecar function."""

    def test_make_syn_sidecar_basic(self):
        """Creates TransformSidecarV1 with valid inputs."""
        fixed_header = ImageHeader(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=tuple(np.eye(3).ravel()),
            size=(10, 10, 10),
        )
        moving_header = ImageHeader(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=tuple(np.eye(3).ravel()),
            size=(20, 20, 20),
        )

        sidecar = make_syn_sidecar(
            fixed_image_header=fixed_header,
            moving_image_header=moving_header,
            affine="path/to/affine.mat",
            warp="path/to/warp.nii.gz",
            inverse_warp="path/to/inverse_warp.nii.gz",
        )

        # Verify SynTriplet contains correct paths
        self.assertEqual(sidecar.transform.affine, "path/to/affine.mat")
        self.assertEqual(sidecar.transform.warp, "path/to/warp.nii.gz")
        self.assertEqual(sidecar.transform.inverse_warp, "path/to/inverse_warp.nii.gz")

        # Verify fixed_domain and moving_domain present
        self.assertIsNotNone(sidecar.fixed_domain)
        self.assertIsNotNone(sidecar.moving_domain)

    def test_make_syn_sidecar_pathlib(self):
        """Accepts Path objects and converts to strings."""
        fixed_header = ImageHeader(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=tuple(np.eye(3).ravel()),
            size=(10, 10, 10),
        )
        moving_header = ImageHeader(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=tuple(np.eye(3).ravel()),
            size=(20, 20, 20),
        )

        sidecar = make_syn_sidecar(
            fixed_image_header=fixed_header,
            moving_image_header=moving_header,
            affine=Path("path/to/affine.mat"),
            warp=Path("path/to/warp.nii.gz"),
            inverse_warp=Path("path/to/inverse_warp.nii.gz"),
        )

        # Paths should be converted to strings
        self.assertIsInstance(sidecar.transform.affine, str)
        self.assertIsInstance(sidecar.transform.warp, str)
        self.assertIsInstance(sidecar.transform.inverse_warp, str)

    def test_make_syn_sidecar_different_geometries(self):
        """Fixed and moving have different domains."""
        fixed_header = ImageHeader(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=tuple(np.eye(3).ravel()),
            size=(10, 10, 10),
        )
        moving_header = ImageHeader(
            origin=(5.0, 5.0, 5.0),
            spacing=(0.5, 0.5, 0.5),
            direction=tuple(np.eye(3).ravel()),
            size=(40, 40, 40),
        )

        sidecar = make_syn_sidecar(
            fixed_image_header=fixed_header,
            moving_image_header=moving_header,
            affine="affine.mat",
            warp="warp.nii.gz",
            inverse_warp="inverse_warp.nii.gz",
        )

        # Verify both domains computed independently
        self.assertEqual(sidecar.fixed_domain.spacing_LPS, (1.0, 1.0, 1.0))
        self.assertEqual(sidecar.moving_domain.spacing_LPS, (0.5, 0.5, 0.5))

        # Fixed bbox: 0 to 9
        self.assertEqual(sidecar.fixed_domain.bbox.L, (0.0, 9.0))

        # Moving bbox: 5 to 5 + (40-1)*0.5 = 24.5
        self.assertAlmostEqual(sidecar.moving_domain.bbox.L[0], 5.0)
        self.assertAlmostEqual(sidecar.moving_domain.bbox.L[1], 24.5)


# ---- Integration Tests ------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipelines."""

    def test_full_pipeline_ants(self):
        """ANTs image → Header → Domain → Sidecar."""
        # Create mock ANTs image
        ants_img = create_mock_ants_image(
            shape=(10, 20, 30),
            spacing=(0.5, 1.0, 2.0),
            origin=(1.0, 2.0, 3.0),
        )

        # Extract header
        header = ImageHeader.from_ants(ants_img)

        # Create domain
        domain = ImageDomainAxisAligned.from_header(header)

        # Convert to sidecar domain
        sidecar_domain = domain.to_sidecar()

        # Verify end-to-end
        self.assertEqual(sidecar_domain.spacing_LPS, (0.5, 1.0, 2.0))
        self.assertEqual(sidecar_domain.shape_canonical, (10, 20, 30))

    def test_full_pipeline_sitk(self):
        """SimpleITK image → Header → Domain → Sidecar."""
        # Create mock SimpleITK image
        sitk_img = create_mock_sitk_image(
            size=(10, 20, 30),
            spacing=(0.5, 1.0, 2.0),
            origin=(1.0, 2.0, 3.0),
        )

        # Extract header
        header = ImageHeader.from_sitk(sitk_img)

        # Create domain
        domain = ImageDomainAxisAligned.from_header(header)

        # Convert to sidecar domain
        sidecar_domain = domain.to_sidecar()

        # Verify end-to-end
        self.assertEqual(sidecar_domain.spacing_LPS, (0.5, 1.0, 2.0))
        self.assertEqual(sidecar_domain.shape_canonical, (10, 20, 30))

    def test_bbox_calculation_numerical(self):
        """Verify bbox math with known coordinates."""
        # Create a simple case we can compute by hand:
        # Image at origin (0,0,0), size (3,4,5), spacing (2,3,4)
        # Voxel centers: X: 0, 2, 4; Y: 0, 3, 6, 9; Z: 0, 4, 8, 12, 16
        # Last voxel centers: X=4, Y=9, Z=16

        header = ImageHeader(
            origin=(0.0, 0.0, 0.0),
            spacing=(2.0, 3.0, 4.0),
            direction=tuple(np.eye(3).ravel()),
            size=(3, 4, 5),
        )

        domain = ImageDomainAxisAligned.from_header(header)

        # Verify computed bbox matches hand calculation
        self.assertAlmostEqual(domain.bbox_L[0], 0.0)
        self.assertAlmostEqual(domain.bbox_L[1], 4.0)
        self.assertAlmostEqual(domain.bbox_P[0], 0.0)
        self.assertAlmostEqual(domain.bbox_P[1], 9.0)
        self.assertAlmostEqual(domain.bbox_S[0], 0.0)
        self.assertAlmostEqual(domain.bbox_S[1], 16.0)


if __name__ == "__main__":
    unittest.main()
