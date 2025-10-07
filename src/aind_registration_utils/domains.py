"""
Methods for calculating image domains.

This module provides utilities for working with medical image domains, including:
- Image header extraction from ANTs and SimpleITK images
- Axis-aligned domain calculation with LPS coordinate system
- Bounding box computation for voxel-centered image spaces
- Transform sidecar generation for SyN registration outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ants
import numpy as np
import SimpleITK as sitk
from aind_ants_transform_sidecar import BBox, Domain, SynTriplet, TransformSidecarV1


def _check_size_override(
    size_override: tuple[int, int, int] | None,
) -> tuple[int, int, int] | None:
    """
    Validate size_override parameter.

    Parameters
    ----------
    size_override : tuple[int, int, int] | None
        Optional tuple of 3 positive integers specifying image dimensions.

    Returns
    -------
    tuple[int, int, int] | None
        The validated size_override parameter.

    Raises
    ------
    ValueError
        If size_override is not a tuple of 3 integers or if any value is not positive.
    """
    if size_override is not None:
        if len(size_override) != 3:
            raise ValueError("size_override must be a tuple of 3 integers.")
        if any(s <= 0 for s in size_override):
            raise ValueError("size_override values must be positive integers.")
    return size_override


@dataclass(frozen=True, slots=True)
class ImageHeader:
    """
    Immutable container for medical image header information.

    Attributes
    ----------
    origin : tuple[float, float, float]
        Physical coordinates of the image origin in LPS space.
    spacing : tuple[float, float, float]
        Voxel spacing along each dimension in physical units.
    direction : tuple[float, float, float, float, float, float, float, float, float]
        Direction cosines as a flattened 3x3 matrix defining image orientation.
    size : tuple[int, int, int]
        Image dimensions in voxels (i,j,k).
    """

    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    direction: tuple[float, float, float, float, float, float, float, float, float]
    size: tuple[int, int, int]

    @classmethod
    def from_ants(
        cls, img: ants.ANTsImage, size_override: tuple[int, int, int] | None = None
    ) -> ImageHeader:
        """
        Create an ImageHeader from an ANTsImage.

        Parameters
        ----------
        img : ants.ANTsImage
            The ANTs image to extract header information from.
        size_override : tuple[int, int, int] | None, optional
            Override the image size with custom dimensions. If None, uses the
            actual image shape. Default is None.

        Returns
        -------
        ImageHeader
            Header information extracted from the ANTs image.

        Raises
        ------
        ValueError
            If size_override is provided but not a tuple of 3 positive integers.
        """
        size_override = _check_size_override(size_override)
        if size_override is not None:
            size = size_override
        else:
            size = img.shape
        assert len(size) == 3, "Image must be 3D."
        return cls(
            origin=tuple(img.origin),
            spacing=tuple(img.spacing),
            direction=tuple(img.direction),
            size=(int(size[0]), int(size[1]), int(size[2])),
        )

    @classmethod
    def from_sitk(
        cls, img: sitk.Image, size_override: tuple[int, int, int] | None = None
    ) -> ImageHeader:
        """
        Create an ImageHeader from a SimpleITK Image.

        Parameters
        ----------
        img : sitk.Image
            The SimpleITK image to extract header information from.
        size_override : tuple[int, int, int] | None, optional
            Override the image size with custom dimensions. If None, uses the
            actual image size. Default is None.

        Returns
        -------
        ImageHeader
            Header information extracted from the SimpleITK image.

        Raises
        ------
        ValueError
            If size_override is provided but not a tuple of 3 positive integers.
        """
        size_override = _check_size_override(size_override)
        if size_override is not None:
            size = size_override
        else:
            size = img.GetSize()
        return cls(
            origin=img.GetOrigin(),
            spacing=img.GetSpacing(),
            direction=img.GetDirection(),
            size=size,
        )


@dataclass(frozen=True, slots=True)
class ImageDomainAxisAligned:
    """
    Axis-aligned image domain representation in LPS coordinate system.

    This class represents a 3D image domain with axis-aligned bounding boxes
    in the LPS (Left-Posterior-Superior) anatomical coordinate system. It assumes
    the image direction matrix represents only permutations and flips of the
    canonical axes (no oblique orientations).

    Attributes
    ----------
    spacing_LPS : tuple[float, float, float]
        Voxel spacing in LPS order (Left, Posterior, Superior).
    bbox_L : tuple[float, float]
        Bounding box extent along the Left axis (min, max).
    bbox_P : tuple[float, float]
        Bounding box extent along the Posterior axis (min, max).
    bbox_S : tuple[float, float]
        Bounding box extent along the Superior axis (min, max).
    shape_LPS : tuple[int, int, int] | None, optional
        Image shape in LPS order. Default is None.
    """

    spacing_LPS: tuple[float, float, float]
    bbox_L: tuple[float, float]
    bbox_P: tuple[float, float]
    bbox_S: tuple[float, float]
    shape_LPS: tuple[int, int, int] | None = None

    @classmethod
    def from_header(
        cls,
        header: ImageHeader,
        tol: float = 1e-6,
    ) -> ImageDomainAxisAligned:
        """
        Compute axis-aligned domain from an ImageHeader.

        This method converts an image header into an axis-aligned domain
        representation in the LPS coordinate system. It validates that the
        direction matrix represents only axis permutations and flips
        (no oblique orientations), then computes the bounding box of voxel
        centers and spacing in canonical LPS order.

        Parameters
        ----------
        header : ImageHeader
            Image header containing origin, spacing, direction, and size.
        tol : float, optional
            Tolerance for validating axis alignment. Direction matrix columns
            must have one element with absolute value > (1 - tol) and all
            other elements with absolute value <= tol. Default is 1e-6.

        Returns
        -------
        ImageDomainAxisAligned
            Axis-aligned domain with LPS spacing, bounding boxes, and shape.

        Raises
        ------
        ValueError
            If the direction matrix is oblique (not a pure permutation/flip).
        """
        origin_LPS = header.origin
        spacing_native = header.spacing
        direction_3x3 = header.direction
        size_native = header.size
        origin = np.asarray(origin_LPS, float).reshape(3)
        D = np.asarray(direction_3x3, float).reshape(3, 3)
        s = np.asarray(spacing_native, float).reshape(3)
        N = np.asarray(size_native, int).reshape(3)

        # --- validate axis-aligned (columns are ± basis vectors) ---
        # each column must have a single +/-1, others ~0
        for j in range(3):
            col = D[:, j]
            i = np.argmax(np.abs(col))
            if not (abs(col[i]) > 1 - tol and np.all(np.abs(np.delete(col, i)) <= tol)):
                raise ValueError("Oblique direction: only pure permute/flip allowed.")

        # --- spacing and shape in L,P,S order (matrix form) ---
        spacing_LPS = np.abs(D) @ s  # (3,)
        shape_LPS = np.abs(D) @ N  # (3,)

        # --- voxel-center bbox (two-corner method) ---
        S = np.diag(s)
        delta = (N - 1).astype(float)  # center-to-center index span
        v = D @ (S @ delta)  # displacement from first to last voxel center in LPS
        p0 = origin
        p1 = origin + v
        Lmin, Lmax = sorted([p0[0], p1[0]])
        Pmin, Pmax = sorted([p0[1], p1[1]])
        Smin, Smax = sorted([p0[2], p1[2]])

        return cls(
            spacing_LPS=tuple(spacing_LPS),
            bbox_L=(Lmin, Lmax),
            bbox_P=(Pmin, Pmax),
            bbox_S=(Smin, Smax),
            shape_LPS=tuple(shape_LPS.astype(int)),
        )

    def to_sidecar(self) -> Domain:
        """
        Convert to a Domain object for transform sidecar metadata.

        Returns
        -------
        Domain
            Domain object with spacing, bounding box, and canonical shape.
        """
        return Domain(
            spacing_LPS=self.spacing_LPS,
            bbox=BBox(L=self.bbox_L, P=self.bbox_P, S=self.bbox_S),
            shape_canonical=self.shape_LPS,
        )


def make_syn_sidecar(
    fixed_image_header: ImageHeader,
    moving_image_header: ImageHeader,
    affine: str | Path,
    warp: str | Path,
    inverse_warp: str | Path,
) -> TransformSidecarV1:
    """
    Makes a TransformSidecarV1 for SyN registration outputs and the
    fixed/moving images.

    Parameters
    ----------
    imgheader : ImageHeader
        The image header containing origin, spacing, direction, and size.
    affine : str | Path
        Path to the affine transform file.
    warp : str | Path
        Path to the forward warp transform file.
    inverse_warp : str | Path
        Path to the inverse warp transform file.

    Returns
    -------
    TransformSidecarV1
        The resulting TransformSidecarV1 object containing the SynTriplet.
    """
    syn_triplet = SynTriplet(
        affine=str(affine),
        warp=str(warp),
        inverse_warp=str(inverse_warp),
    )
    fixed_domain = ImageDomainAxisAligned.from_header(fixed_image_header).to_sidecar()
    moving_domain = ImageDomainAxisAligned.from_header(moving_image_header).to_sidecar()
    sidecar = TransformSidecarV1(
        fixed_domain=fixed_domain, moving_domain=moving_domain, transform=syn_triplet
    )
    return sidecar
