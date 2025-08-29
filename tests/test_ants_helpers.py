"""Tests for ANTs helper functions in ants.py module."""

import unittest

import numpy as np

from aind_registration_utils.ants import (
    _surface_samples,
    _to_continuous_index,
)


class TestSurfaceSamples(unittest.TestCase):
    """Test _surface_samples function."""

    def test_surface_samples_2d_corners(self):
        """Test 2D corner sampling (n=2)."""
        pts = _surface_samples((10, 8), n=2)
        expected = [(0.0, 0.0), (0.0, 7.0), (9.0, 0.0), (9.0, 7.0)]
        self.assertEqual(set(pts), set(expected))
        self.assertEqual(len(pts), 4)

    def test_surface_samples_2d_grid(self):
        """Test 2D edge grid sampling (n=3)."""
        pts = _surface_samples((4, 4), n=3)
        # Should have 3 points per edge, avoiding duplicates at corners
        # 4 edges × 3 points - 4 corner duplicates = 8 unique points
        self.assertEqual(len(pts), 8)
        # Check corner points are included
        corners = [(0.0, 0.0), (0.0, 3.0), (3.0, 0.0), (3.0, 3.0)]
        for corner in corners:
            self.assertIn(corner, pts)

    def test_surface_samples_3d_corners(self):
        """Test 3D corner sampling (n=2)."""
        pts = _surface_samples((4, 4, 4), n=2)
        # 8 corners of a cube
        expected_corners = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 3.0),
            (0.0, 3.0, 0.0),
            (0.0, 3.0, 3.0),
            (3.0, 0.0, 0.0),
            (3.0, 0.0, 3.0),
            (3.0, 3.0, 0.0),
            (3.0, 3.0, 3.0),
        ]
        self.assertEqual(set(pts), set(expected_corners))
        self.assertEqual(len(pts), 8)

    def test_surface_samples_3d_face_grid(self):
        """Test 3D face grid sampling."""
        pts = _surface_samples((4, 4, 4), n=3)
        # 6 faces with 3×3 grid each, minus edge/corner overlaps
        # Let's verify the actual count rather than theoretical
        self.assertGreater(len(pts), 8)  # More than just corners
        self.assertLess(len(pts), 54)  # Less than full 6×9 faces

    def test_surface_samples_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        with self.assertRaises(ValueError) as cm:
            _surface_samples((10,))  # 1D invalid
        self.assertIn("length 2 or 3", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _surface_samples((10, 8, 6, 4))  # 4D invalid
        self.assertIn("length 2 or 3", str(cm.exception))

    def test_surface_samples_invalid_n(self):
        """Test error handling for invalid n parameter."""
        with self.assertRaises(ValueError) as cm:
            _surface_samples((10, 8), n=1)  # n too small
        self.assertIn("`n` must be >= 2", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _surface_samples((10, 8), n=0)  # n too small
        self.assertIn("`n` must be >= 2", str(cm.exception))

    def test_surface_samples_single_voxel(self):
        """Test edge case with 1×1×1 image."""
        pts = _surface_samples((1, 1, 1), n=2)
        # All corners should be the same point
        self.assertEqual(len(set(pts)), 1)  # All points are identical
        self.assertEqual(pts[0], (0.0, 0.0, 0.0))


class TestToContinuousIndex(unittest.TestCase):
    """Test _to_continuous_index function."""

    def test_to_continuous_index_identity(self):
        """Test identity transform case."""
        pts = [(1.0, 2.0, 3.0)]
        origin = (0.0, 0.0, 0.0)
        spacing = (1.0, 1.0, 1.0)
        direction = list(np.eye(3).reshape(-1))

        result = _to_continuous_index(pts, origin, spacing, direction)
        expected = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(result, expected)

    def test_to_continuous_index_scaled_spacing(self):
        """Test with non-unit spacing."""
        pts = [(2.0, 4.0, 6.0)]
        origin = (0.0, 0.0, 0.0)
        spacing = (2.0, 2.0, 2.0)  # Should halve indices
        direction = list(np.eye(3).reshape(-1))

        result = _to_continuous_index(pts, origin, spacing, direction)
        expected = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(result, expected)

    def test_to_continuous_index_offset_origin(self):
        """Test with non-zero origin."""
        pts = [(3.0, 4.0, 5.0)]
        origin = (1.0, 2.0, 3.0)  # Offset origin
        spacing = (1.0, 1.0, 1.0)
        direction = list(np.eye(3).reshape(-1))

        result = _to_continuous_index(pts, origin, spacing, direction)
        expected = np.array([[2.0, 2.0, 2.0]])  # pts - origin
        np.testing.assert_allclose(result, expected)

    def test_to_continuous_index_flipped_direction(self):
        """Test with flipped direction matrix."""
        pts = [(1.0, 2.0, 3.0)]
        origin = (0.0, 0.0, 0.0)
        spacing = (1.0, 1.0, 1.0)
        # Flip x-axis direction
        direction_matrix = np.eye(3)
        direction_matrix[0, 0] = -1
        direction = list(direction_matrix.reshape(-1))

        result = _to_continuous_index(pts, origin, spacing, direction)
        expected = np.array([[-1.0, 2.0, 3.0]])  # x-coordinate flipped
        np.testing.assert_allclose(result, expected)

    def test_to_continuous_index_multiple_points(self):
        """Test with multiple points."""
        pts = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (2.0, 4.0, 6.0)]
        origin = (0.0, 0.0, 0.0)
        spacing = (1.0, 2.0, 3.0)
        direction = list(np.eye(3).reshape(-1))

        result = _to_continuous_index(pts, origin, spacing, direction)
        expected = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        np.testing.assert_allclose(result, expected)

    def test_to_continuous_index_2d(self):
        """Test with 2D coordinates."""
        pts = [(2.0, 4.0)]
        origin = (0.0, 0.0)
        spacing = (2.0, 2.0)
        direction = list(np.eye(2).reshape(-1))

        result = _to_continuous_index(pts, origin, spacing, direction)
        expected = np.array([[1.0, 2.0]])
        np.testing.assert_allclose(result, expected)

    def test_to_continuous_index_invalid_shape(self):
        """Test error handling for mismatched dimensions."""
        pts = [(1.0, 2.0)]  # 2D points
        origin = (0.0, 0.0, 0.0)  # 3D origin
        spacing = (1.0, 1.0, 1.0)  # 3D spacing
        direction = list(np.eye(3).reshape(-1))  # 3D direction

        with self.assertRaises(ValueError) as cm:
            _to_continuous_index(pts, origin, spacing, direction)
        self.assertIn("must have shape", str(cm.exception))

    def test_to_continuous_index_empty_points(self):
        """Test with empty point list."""
        pts = np.array([]).reshape(0, 3)  # Proper empty array shape
        origin = (0.0, 0.0, 0.0)
        spacing = (1.0, 1.0, 1.0)
        direction = list(np.eye(3).reshape(-1))

        result = _to_continuous_index(pts, origin, spacing, direction)
        self.assertEqual(result.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
