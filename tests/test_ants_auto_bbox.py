"""Tests for apply_transforms_auto_bbox function in ants.py module."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from aind_registration_utils.ants import apply_transforms_auto_bbox


def create_mock_ants_image(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Create synthetic ANTs image with controlled geometry."""
    mock_img = Mock()
    mock_img.dimension = len(shape)
    mock_img.shape = shape
    mock_img.spacing = spacing
    mock_img.origin = origin
    mock_img.direction = tuple(np.eye(len(shape)).reshape(-1).tolist())
    mock_img.pixeltype = "float32"
    return mock_img


class TestApplyTransformsAutoBbox(unittest.TestCase):
    """Test apply_transforms_auto_bbox function."""

    def setUp(self):
        """Set up common test fixtures."""
        self.mock_moving = create_mock_ants_image((10, 10, 10))
        self.mock_fixed = create_mock_ants_image((20, 20, 20), spacing=(0.5, 0.5, 0.5))
        self.mock_warped = Mock()
        self.mock_ref = Mock()

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_identity_transform_no_fixed(
        self, mock_idx2phys, mock_points, mock_make, mock_apply
    ):
        """Test identity transform without fixed reference."""
        # Setup mocks
        mock_idx2phys.side_effect = lambda img, idx: list(idx)  # index = physical
        mock_points.return_value = pd.DataFrame(
            {
                "x": [0.0, 0.0, 9.0, 9.0, 0.0, 0.0, 9.0, 9.0],
                "y": [0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0],
                "z": [0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0],
            }
        )
        mock_make.return_value = self.mock_ref
        mock_apply.return_value = self.mock_warped

        # Call function
        warped, ref = apply_transforms_auto_bbox(
            moving=self.mock_moving,
            transformlist=["identity.mat"],
            samples_per_edge=2,
            pad_voxels=1,
        )

        # Verify reference image created with correct bounds
        mock_make.assert_called_once()
        call_kwargs = mock_make.call_args[1]
        expected_size = (12, 12, 12)  # 10 + 2*pad_voxels
        self.assertEqual(call_kwargs["imagesize"], expected_size)
        self.assertEqual(call_kwargs["spacing"], (1.0, 1.0, 1.0))  # From moving

        # Verify apply_transforms called
        mock_apply.assert_called_once()
        apply_kwargs = mock_apply.call_args[1]
        self.assertEqual(apply_kwargs["fixed"], self.mock_ref)
        self.assertEqual(apply_kwargs["moving"], self.mock_moving)
        self.assertEqual(apply_kwargs["transformlist"], ["identity.mat"])

        self.assertEqual(warped, self.mock_warped)
        self.assertEqual(ref, self.mock_ref)

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_with_fixed_reference(
        self, mock_idx2phys, mock_points, mock_make, mock_apply
    ):
        """Test with fixed reference image provided."""
        # Setup mocks
        mock_idx2phys.side_effect = lambda img, idx: list(idx)
        mock_points.return_value = pd.DataFrame(
            {
                "x": [2.0, 2.0, 7.0, 7.0],
                "y": [1.0, 6.0, 1.0, 6.0],
                "z": [0.0, 0.0, 0.0, 5.0],
            }
        )
        mock_make.return_value = self.mock_ref
        mock_apply.return_value = self.mock_warped

        # Call function with fixed reference
        warped, ref = apply_transforms_auto_bbox(
            moving=self.mock_moving,
            transformlist=["transform.mat"],
            fixed=self.mock_fixed,
            pad_voxels=0,
        )

        # Verify uses fixed image's geometry
        call_kwargs = mock_make.call_args[1]
        self.assertEqual(call_kwargs["spacing"], (0.5, 0.5, 0.5))  # From fixed
        # Bounding box: x=[2,7], y=[1,6], z=[0,5] with spacing 0.5 and default pad=1
        # Continuous indices: x=[4,14], y=[2,12], z=[0,10] → floor/ceil with
        # pad → size=(11,11,11)
        expected_size = (11, 11, 11)  # Corrected based on actual calculation
        self.assertEqual(call_kwargs["imagesize"], expected_size)

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_multiple_transforms(
        self, mock_idx2phys, mock_points, mock_make, mock_apply
    ):
        """Test handling of multiple transforms in sequence."""
        mock_idx2phys.side_effect = lambda img, idx: list(idx)
        mock_points.return_value = pd.DataFrame(
            {"x": [0.0, 10.0], "y": [0.0, 10.0], "z": [0.0, 10.0]}
        )
        mock_make.return_value = self.mock_ref
        mock_apply.return_value = self.mock_warped

        transform_list = ["rigid.mat", "affine.mat", "warp.nii.gz"]
        warped, ref = apply_transforms_auto_bbox(
            moving=self.mock_moving, transformlist=transform_list
        )

        # Verify transform list passed correctly
        mock_points.assert_called_once()
        points_call_args = mock_points.call_args
        # transformlist is the 3rd positional argument (index 2)
        self.assertEqual(points_call_args[0][2], transform_list)

        apply_kwargs = mock_apply.call_args[1]
        self.assertEqual(apply_kwargs["transformlist"], transform_list)

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_whichtoinvert_parameter(
        self, mock_idx2phys, mock_points, mock_make, mock_apply
    ):
        """Test whichtoinvert parameter is correctly passed."""
        mock_idx2phys.side_effect = lambda img, idx: list(idx)
        mock_points.return_value = pd.DataFrame(
            {"x": [0.0, 9.0], "y": [0.0, 9.0], "z": [0.0, 9.0]}
        )
        mock_make.return_value = self.mock_ref
        mock_apply.return_value = self.mock_warped

        invert_flags = [True, False, True]
        warped, ref = apply_transforms_auto_bbox(
            moving=self.mock_moving,
            transformlist=["a.mat", "b.mat", "c.mat"],
            whichtoinvert=invert_flags,
        )

        # Verify whichtoinvert passed to both ANTs calls
        points_kwargs = mock_points.call_args[1]
        self.assertEqual(points_kwargs["whichtoinvert"], invert_flags)

        apply_kwargs = mock_apply.call_args[1]
        self.assertEqual(apply_kwargs["whichtoinvert"], invert_flags)

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_interpolator_and_default_value(
        self, mock_idx2phys, mock_points, mock_make, mock_apply
    ):
        """Test interpolator and default_value parameters."""
        mock_idx2phys.side_effect = lambda img, idx: list(idx)
        mock_points.return_value = pd.DataFrame(
            {"x": [0.0, 9.0], "y": [0.0, 9.0], "z": [0.0, 9.0]}
        )
        mock_make.return_value = self.mock_ref
        mock_apply.return_value = self.mock_warped

        warped, ref = apply_transforms_auto_bbox(
            moving=self.mock_moving,
            transformlist=["test.mat"],
            interpolator="nearestNeighbor",
            default_value=-1000,
        )

        # Verify parameters passed to apply_transforms
        apply_kwargs = mock_apply.call_args[1]
        self.assertEqual(apply_kwargs["interpolator"], "nearestNeighbor")
        self.assertEqual(apply_kwargs["defaultvalue"], -1000)

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_custom_spacing_direction_origin(
        self, mock_idx2phys, mock_points, mock_make, mock_apply
    ):
        """Test custom spacing, direction, and origin parameters."""
        mock_idx2phys.side_effect = lambda img, idx: list(idx)
        mock_points.return_value = pd.DataFrame(
            {"x": [0.0, 4.0], "y": [0.0, 4.0], "z": [0.0, 4.0]}
        )
        mock_make.return_value = self.mock_ref
        mock_apply.return_value = self.mock_warped

        custom_spacing = (2.0, 2.0, 2.0)
        custom_origin = (10.0, 20.0, 30.0)
        custom_direction = [-1, 0, 0, 0, 1, 0, 0, 0, 1]  # Flip x-axis

        warped, ref = apply_transforms_auto_bbox(
            moving=self.mock_moving,
            transformlist=["test.mat"],
            spacing=custom_spacing,
            origin=custom_origin,
            direction=custom_direction,
        )

        # Verify custom parameters used in make_image
        make_kwargs = mock_make.call_args[1]
        self.assertEqual(make_kwargs["spacing"], custom_spacing)
        self.assertEqual(make_kwargs["direction"], tuple(custom_direction))
        # Origin should be adjusted based on bounding box calculation

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_samples_per_edge_parameter(
        self, mock_idx2phys, mock_points, mock_make, mock_apply
    ):
        """Test samples_per_edge affects surface sampling."""
        mock_idx2phys.side_effect = lambda img, idx: list(idx)
        mock_points.return_value = pd.DataFrame(
            {"x": [0.0, 9.0], "y": [0.0, 9.0], "z": [0.0, 9.0]}
        )
        mock_make.return_value = self.mock_ref
        mock_apply.return_value = self.mock_warped

        # Test with different samples_per_edge values
        warped, ref = apply_transforms_auto_bbox(
            moving=self.mock_moving,
            transformlist=["test.mat"],
            samples_per_edge=5,
        )

        # Verify more points were generated and passed to
        # transform_index_to_physical_point
        # With samples_per_edge=5, should get more than 8 corner points
        self.assertGreater(mock_idx2phys.call_count, 8)

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_2d_image_handling(self, mock_idx2phys, mock_points, mock_make, mock_apply):
        """Test handling of 2D images."""
        mock_2d_moving = create_mock_ants_image(
            (100, 100), spacing=(1.0, 1.0), origin=(0.0, 0.0)
        )  # 2D image
        mock_idx2phys.side_effect = lambda img, idx: list(idx)
        mock_points.return_value = pd.DataFrame(
            {"x": [0.0, 99.0, 0.0, 99.0], "y": [0.0, 0.0, 99.0, 99.0]}
        )
        mock_make.return_value = self.mock_ref
        mock_apply.return_value = self.mock_warped

        warped, ref = apply_transforms_auto_bbox(
            moving=mock_2d_moving,
            transformlist=["test.mat"],
            samples_per_edge=2,
        )

        # Verify 2D DataFrame created correctly
        points_call_args = mock_points.call_args
        self.assertEqual(points_call_args[0][0], 2)  # dimension parameter
        df_columns = points_call_args[0][1].columns.tolist()
        self.assertEqual(df_columns, ["x", "y"])  # Only x,y for 2D


class TestApplyTransformsAutoBboxEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for apply_transforms_auto_bbox."""

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_extreme_transform_bounds(
        self, mock_idx2phys, mock_points, mock_make, mock_apply
    ):
        """Test handling of extreme transformation bounds."""
        mock_moving = create_mock_ants_image((10, 10, 10))
        mock_idx2phys.side_effect = lambda img, idx: list(idx)

        # Mock extreme transformation result
        extreme_points = pd.DataFrame(
            {
                "x": [-1000.0, 1000.0],
                "y": [-500.0, 500.0],
                "z": [-100.0, 100.0],
            }
        )
        mock_points.return_value = extreme_points
        mock_make.return_value = Mock()
        mock_apply.return_value = Mock()

        warped, ref = apply_transforms_auto_bbox(
            moving=mock_moving, transformlist=["extreme.mat"], pad_voxels=0
        )

        # Verify function handles extreme bounds gracefully
        make_kwargs = mock_make.call_args[1]
        image_size = make_kwargs["imagesize"]
        self.assertTrue(all(s > 0 for s in image_size))  # Positive dimensions
        self.assertTrue(all(isinstance(s, int) for s in image_size))  # Integer sizes

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_zero_pad_voxels(self, mock_idx2phys, mock_points, mock_make, mock_apply):
        """Test with zero padding."""
        mock_moving = create_mock_ants_image((5, 5, 5))
        mock_idx2phys.side_effect = lambda img, idx: list(idx)
        mock_points.return_value = pd.DataFrame(
            {"x": [0.0, 4.0], "y": [0.0, 4.0], "z": [0.0, 4.0]}
        )
        mock_make.return_value = Mock()
        mock_apply.return_value = Mock()

        warped, ref = apply_transforms_auto_bbox(
            moving=mock_moving,
            transformlist=["test.mat"],
            pad_voxels=0,  # No padding
        )

        # Verify exact bounding box without padding
        make_kwargs = mock_make.call_args[1]
        # Bounds: x=[0,4], y=[0,4], z=[0,4] → size=(5,5,5)
        self.assertEqual(make_kwargs["imagesize"], (5, 5, 5))

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_large_pad_voxels(self, mock_idx2phys, mock_points, mock_make, mock_apply):
        """Test with large padding."""
        mock_moving = create_mock_ants_image((5, 5, 5))
        mock_idx2phys.side_effect = lambda img, idx: list(idx)
        mock_points.return_value = pd.DataFrame(
            {
                "x": [2.0, 2.0],
                "y": [2.0, 2.0],
                "z": [2.0, 2.0],  # Single point
            }
        )
        mock_make.return_value = Mock()
        mock_apply.return_value = Mock()

        warped, ref = apply_transforms_auto_bbox(
            moving=mock_moving,
            transformlist=["test.mat"],
            pad_voxels=10,  # Large padding
        )

        make_kwargs = mock_make.call_args[1]
        # Single point at (2,2,2) with 10 padding → size=(21,21,21)
        self.assertEqual(make_kwargs["imagesize"], (21, 21, 21))

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_single_voxel_moving_image(
        self, mock_idx2phys, mock_points, mock_make, mock_apply
    ):
        """Test with 1×1×1 moving image."""
        mock_moving = create_mock_ants_image((1, 1, 1))
        mock_idx2phys.side_effect = lambda img, idx: list(idx)
        mock_points.return_value = pd.DataFrame(
            {
                "x": [5.0],
                "y": [10.0],
                "z": [15.0],  # Transformed single point
            }
        )
        mock_make.return_value = Mock()
        mock_apply.return_value = Mock()

        warped, ref = apply_transforms_auto_bbox(
            moving=mock_moving, transformlist=["test.mat"], pad_voxels=2
        )

        make_kwargs = mock_make.call_args[1]
        # Single point with padding → size=(5,5,5)
        self.assertEqual(make_kwargs["imagesize"], (5, 5, 5))

    @patch("aind_registration_utils.ants.ants.apply_transforms")
    @patch("aind_registration_utils.ants.ants.make_image")
    @patch("aind_registration_utils.ants.ants.apply_transforms_to_points")
    @patch("aind_registration_utils.ants.ants.transform_index_to_physical_point")
    def test_negative_transformed_coordinates(
        self, mock_idx2phys, mock_points, mock_make, mock_apply
    ):
        """Test handling of negative transformed coordinates."""
        mock_moving = create_mock_ants_image((10, 10, 10))
        mock_idx2phys.side_effect = lambda img, idx: list(idx)
        mock_points.return_value = pd.DataFrame(
            {"x": [-5.0, 5.0], "y": [-10.0, 10.0], "z": [-2.0, 2.0]}
        )
        mock_make.return_value = Mock()
        mock_apply.return_value = Mock()

        warped, ref = apply_transforms_auto_bbox(
            moving=mock_moving, transformlist=["test.mat"], pad_voxels=1
        )

        make_kwargs = mock_make.call_args[1]
        # Bounds: x=[-5,5], y=[-10,10], z=[-2,2] with pad=1 → size=(13,23,7)
        # floor([-5,-10,-2]) - 1 = [-6,-11,-3], ceil([5,10,2]) + 1 = [6,11,3]
        # size = [6-(-6)+1, 11-(-11)+1, 3-(-3)+1] = [13,23,7]
        self.assertEqual(make_kwargs["imagesize"], (13, 23, 7))

        # Origin should be adjusted for negative bounds
        origin = make_kwargs["origin"]
        self.assertTrue(all(isinstance(o, float) for o in origin))


if __name__ == "__main__":
    unittest.main()
