"""
Type aliases and definitions for aind-registration-utils.

This module defines common type aliases to ensure consistency across the package
while minimizing runtime import overhead through TYPE_CHECKING conditionals.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import ants
    import numpy as np
    import numpy.typing as npt

# Core types (no runtime import overhead)
PathLike: TypeAlias = str | Path

if TYPE_CHECKING:
    # Heavy numpy types (only for static analysis)
    Float32Array: TypeAlias = npt.NDArray[np.float32]
    Float64Array: TypeAlias = npt.NDArray[np.float64]
    FloatArray: TypeAlias = Float32Array | Float64Array
    IntArray: TypeAlias = npt.NDArray[np.integer[Any]]
    BoolArray: TypeAlias = npt.NDArray[np.bool_]

    # 3D coordinates
    Point3D: TypeAlias = tuple[float, float, float]
    Points3D: TypeAlias = Sequence[Point3D]
    PointDict: TypeAlias = dict[str, Sequence[float]]

    # ANTs-specific
    AntsImage: TypeAlias = ants.ANTsImage
    TransformList: TypeAlias = list[str]
    TransformDict: TypeAlias = dict[str, Any]

    # VTK-specific (for vtk.py)
    import vtk

    VtkTransform: TypeAlias = vtk.vtkThinPlateSplineTransform
    VtkPoints: TypeAlias = vtk.vtkPoints
