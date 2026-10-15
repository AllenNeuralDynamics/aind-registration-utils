"""Apply ANTs SyN results with SimpleITK, so consumers need no ANTs install.

A :class:`SynTransform` is loaded from a transform sidecar. Directions name where
**points** go: ``"fixed_to_moving"`` maps fixed-space points into moving space,
which is also what resampling a moving image onto the fixed grid needs. Chains
across registrations are lists of ``(transform, direction)`` hops, applied in
order to points. For example, with a brain registered to a template (template
fixed) and the template registered to CCF (CCF fixed)::

    brain_pts_in_ccf = transform_points(
        brain_pts, [(brain_reg, "moving_to_fixed"), (tpl_to_ccf, "moving_to_fixed")]
    )
    brain_in_ccf = resample(
        brain, ccf, [(tpl_to_ccf, "fixed_to_moving"), (brain_reg, "fixed_to_moving")]
    )

Outside its grid a displacement field contributes nothing, so a point there
silently gets only the affine. :func:`transform_points` checks for this.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import SimpleITK as sitk
from aind_ants_transform_sidecar import TransformSidecarV1, load_package

from aind_registration_utils.domains import ImageDomainAxisAligned, ImageHeader

if TYPE_CHECKING:
    import numpy.typing as npt

    from aind_registration_utils.types import PathLike

Direction = Literal["fixed_to_moving", "moving_to_fixed"]
Outside = Literal["raise", "warn", "ignore"]
Hop = tuple["SynTransform", Direction]

# Sidecar bboxes and field headers both come from NIfTI's float32 geometry.
_DOMAIN_ATOL_MM = 1e-4


@dataclass(frozen=True)
class _FieldGrid:
    """Geometry of a displacement field, kept because the transform consumes it."""

    origin: npt.NDArray[np.float64]
    spacing: npt.NDArray[np.float64]
    inverse_direction: npt.NDArray[np.float64]
    size: npt.NDArray[np.int64]

    @classmethod
    def from_image(cls, image: sitk.Image) -> _FieldGrid:
        direction = np.asarray(image.GetDirection(), float).reshape(3, 3)
        return cls(
            origin=np.asarray(image.GetOrigin(), float),
            spacing=np.asarray(image.GetSpacing(), float),
            inverse_direction=np.linalg.inv(direction),
            size=np.asarray(image.GetSize(), np.int64),
        )

    def contains(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Whether each point is where ITK interpolates the field (±½ voxel)."""
        cidx = (points - self.origin) @ self.inverse_direction.T / self.spacing
        inside: npt.NDArray[np.bool_] = np.all(
            (cidx >= -0.5) & (cidx <= self.size - 0.5), axis=1
        )
        return inside


@dataclass(frozen=True)
class _Step:
    transform: sitk.Transform
    grid: _FieldGrid | None
    file: str


class SynTransform:
    """An ANTs SyN result (affine, warp, inverse warp) described by a sidecar.

    Parameters
    ----------
    sidecar : TransformSidecarV1
        Sidecar model; its file names are relative to ``root``.
    root : Path
        Directory the transform files are resolved against.

    Notes
    -----
    Fields are read lazily and kept, at 24 bytes per voxel each.
    """

    def __init__(self, sidecar: TransformSidecarV1, root: Path) -> None:
        self.sidecar = sidecar
        self.root = Path(root)
        self._cache: dict[tuple[str, bool], _Step] = {}

    @classmethod
    def from_sidecar(cls, path: PathLike) -> SynTransform:
        """Load from a sidecar JSON, resolving its file names beside it.

        Parameters
        ----------
        path : PathLike
            Sidecar JSON.

        Returns
        -------
        SynTransform
            Unloaded transform; files are read on first use.
        """
        path = Path(path)
        return cls(load_package(path.read_text()), path.parent)

    def _load(self, file: str, kind: str, invert: bool) -> _Step:
        key = (file, invert)
        if key not in self._cache:
            path = self.root / file
            if kind == "affine":
                transform = sitk.ReadTransform(str(path))
                if invert:
                    transform = transform.GetInverse()
                self._cache[key] = _Step(transform, None, file)
            else:
                field = sitk.ReadImage(str(path), sitk.sitkVectorFloat64)
                grid = _FieldGrid.from_image(field)
                self._check_field_grid(field, file)
                self._cache[key] = _Step(
                    sitk.DisplacementFieldTransform(field), grid, file
                )
        return self._cache[key]

    def _check_field_grid(self, field: sitk.Image, file: str) -> None:
        """Fields live on the fixed grid; a mismatch means the sidecar is wrong."""
        expected = self.sidecar.fixed_domain
        if expected is None:
            return
        found = ImageDomainAxisAligned.from_header(ImageHeader.from_sitk(field))
        bbox = expected.bbox
        ok = expected.shape_canonical in (None, found.shape_LPS) and np.allclose(
            [found.bbox_L, found.bbox_P, found.bbox_S],
            [bbox.L, bbox.P, bbox.S],
            atol=_DOMAIN_ATOL_MM,
        )
        if not ok:
            raise ValueError(
                f"{file} does not lie on the sidecar's fixed domain; "
                "the sidecar does not describe this registration"
            )

    def _steps(self, direction: Direction) -> list[_Step]:
        chain = (
            self.sidecar.forward_chain()
            if direction == "fixed_to_moving"
            else self.sidecar.inverse_chain()
        )
        return [
            self._load(s.file, s.kind, getattr(s, "invert", False)) for s in chain.steps
        ]

    def transform(self, direction: Direction) -> sitk.CompositeTransform:
        """Point transform for one direction.

        Parameters
        ----------
        direction : {"fixed_to_moving", "moving_to_fixed"}
            Where points go.

        Returns
        -------
        sitk.CompositeTransform
            Usable with ``TransformPoint`` or ``sitk.Resample``.
        """
        return _composite(self._steps(direction))

    def transform_points(
        self,
        points: npt.ArrayLike,
        direction: Direction,
        outside: Outside = "raise",
    ) -> npt.NDArray[np.float64]:
        """Map LPS mm points; see :func:`transform_points`."""
        return transform_points(points, [(self, direction)], outside)


def _composite(steps: Sequence[_Step]) -> sitk.CompositeTransform:
    # A composite applies the transform added last first.
    composite = sitk.CompositeTransform(3)
    for step in reversed(steps):
        composite.AddTransform(step.transform)
    return composite


def composite_transform(hops: Sequence[Hop]) -> sitk.CompositeTransform:
    """One point transform for a chain of hops, applied in list order.

    Parameters
    ----------
    hops : sequence of (SynTransform, direction)
        Hops in the order points pass through them.

    Returns
    -------
    sitk.CompositeTransform
        The whole chain.
    """
    return _composite([s for t, d in hops for s in t._steps(d)])


def transform_points(
    points: npt.ArrayLike,
    hops: Sequence[Hop],
    outside: Outside = "raise",
) -> npt.NDArray[np.float64]:
    """Map LPS mm points through a chain of hops.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Physical LPS coordinates in mm.
    hops : sequence of (SynTransform, direction)
        Hops in the order points pass through them.
    outside : {"raise", "warn", "ignore"}
        What to do when a point reaches a displacement field outside its grid.

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        Mapped points.

    Raises
    ------
    ValueError
        If ``outside="raise"`` and any point misses a field's grid.
    """
    current = np.atleast_2d(np.asarray(points, float))
    if current.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3); got {current.shape}")
    for synt, direction in hops:
        for step in synt._steps(direction):
            if step.grid is not None and outside != "ignore":
                missed = ~step.grid.contains(current)
                if missed.any():
                    msg = (
                        f"{int(missed.sum())} of {len(current)} points fall outside "
                        f"{step.file}, where only the affine applies"
                    )
                    if outside == "raise":
                        raise ValueError(msg)
                    warnings.warn(msg, stacklevel=2)
            current = np.array([step.transform.TransformPoint(p) for p in current])
    return current


def resample(
    image: sitk.Image,
    reference: sitk.Image,
    hops: Sequence[Hop],
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """Resample ``image`` onto ``reference``'s grid.

    Parameters
    ----------
    image : sitk.Image
        Image to resample.
    reference : sitk.Image
        Output grid.
    hops : sequence of (SynTransform, direction)
        Chain taking reference-space points to image-space points.
    interpolator : int
        SimpleITK interpolator; use ``sitk.sitkLabelGaussian`` (the counterpart
        of ANTs ``GenericLabel``) or ``sitk.sitkNearestNeighbor`` for labels.
    default_value : float
        Value outside ``image``.

    Returns
    -------
    sitk.Image
        Resampled image with ``image``'s pixel type.
    """
    return sitk.Resample(
        image,
        reference,
        composite_transform(hops),
        interpolator,
        default_value,
        image.GetPixelID(),
    )
