"""Serializable settings for a linear + SyN ``antsRegistration`` run.

Every field is emitted explicitly, so no ANTs or wrapper default decides the
result. The defaults match ``register_to_template.sh`` in mri-template-generation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SmoothingUnits = Literal["vox", "mm"]
Sampling = Literal["None", "Regular", "Random"]


class _Schedule(BaseModel):
    """Multi-resolution schedule shared by linear and SyN stages."""

    model_config = ConfigDict(extra="forbid")

    iterations: list[int]
    convergence_threshold: float = 1e-6
    convergence_window: int = 10
    shrink_factors: list[int]
    smoothing_sigmas: list[float]
    smoothing_units: SmoothingUnits = "vox"

    @model_validator(mode="after")
    def _levels_match(self) -> _Schedule:
        n = len(self.iterations)
        if len(self.shrink_factors) != n or len(self.smoothing_sigmas) != n:
            raise ValueError(
                "iterations, shrink_factors and smoothing_sigmas need one entry "
                f"per level; got {n}, {len(self.shrink_factors)} and "
                f"{len(self.smoothing_sigmas)}"
            )
        return self


class LinearConfig(_Schedule):
    """Linear stages, run in order with one shared metric and schedule."""

    stages: list[Literal["Rigid", "Similarity", "Affine"]] = Field(
        default=["Rigid", "Affine"], min_length=1
    )
    gradient_step: float = 0.1
    metric: Literal["MI", "Mattes"] = "MI"
    bins: int = 32
    sampling: Sampling = "Regular"
    sampling_pct: float = 0.25
    # Pydantic copies mutable defaults per instance.
    iterations: list[int] = [1000, 500, 250, 100]
    shrink_factors: list[int] = [6, 4, 2, 1]
    smoothing_sigmas: list[float] = [3.0, 2.0, 1.0, 0.0]


class SynConfig(_Schedule):
    """The SyN stage."""

    gradient_step: float = 0.2
    update_field_variance: float = 4.0
    total_field_variance: float = 0.0
    metric: Literal["CC", "MI", "Mattes", "MeanSquares"] = "CC"
    radius_or_bins: int = 4  # CC radius in voxels, or the MI/Mattes bin count
    weight: float = 1.0
    iterations: list[int] = [100, 100, 70, 40, 40]
    shrink_factors: list[int] = [8, 6, 4, 2, 1]
    smoothing_sigmas: list[float] = [4.0, 3.0, 2.0, 1.0, 0.0]


class ExtraMetric(BaseModel):
    """An additional SyN metric on another image pair, e.g. a second channel."""

    model_config = ConfigDict(extra="forbid")

    fixed: str
    moving: str
    metric: Literal["CC", "MI", "Mattes", "MeanSquares"] = "CC"
    weight: float = 1.0
    radius_or_bins: int = 4


class RegistrationConfig(BaseModel):
    """Linear stages followed by one SyN stage.

    Notes
    -----
    Masks given to the builder apply only to the stages named by
    ``mask_stages``. ANTs masks reject samples rather than extend the domain,
    so masks for SyN should be dilated beyond the tissue.
    """

    model_config = ConfigDict(extra="forbid")

    initialization: Literal["center-of-mass", "geometric", "origin", "none"] = (
        "center-of-mass"
    )
    linear: LinearConfig = LinearConfig()
    syn: SynConfig = SynConfig()
    mask_stages: Literal["syn", "all"] = "syn"
    winsorize: tuple[float, float] | None = (0.005, 0.995)
    histogram_matching: bool = False
    random_seed: int | None = None
    use_float: bool = True
    interpolation: str = "BSpline"
    write_warped: bool = True

    @classmethod
    def lightsheet_25um(cls) -> RegistrationConfig:
        """Starting settings for cleared light-sheet brains at 25 µm, to be tuned."""
        sigmas_mm = [0.2, 0.1, 0.05, 0.0]
        return cls(
            linear=LinearConfig(
                shrink_factors=[8, 4, 2, 1],
                smoothing_sigmas=sigmas_mm,
                smoothing_units="mm",
            ),
            syn=SynConfig(
                update_field_variance=3.0,
                iterations=[100, 70, 50, 20],
                shrink_factors=[8, 4, 2, 1],
                smoothing_sigmas=sigmas_mm,
                smoothing_units="mm",
            ),
            random_seed=1,
        )
