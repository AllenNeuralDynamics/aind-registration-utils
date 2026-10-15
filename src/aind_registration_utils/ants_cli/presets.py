"""Turn a :class:`RegistrationConfig` into a ``commandants`` registration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from commandants import (
    CC,
    MI,
    Affine,
    AntsRegistration,
    Convergence,
    Mattes,
    MeanSquares,
    Metric,
    Rigid,
    Similarity,
    SyN,
)

from aind_registration_utils.ants_cli.config import ExtraMetric, RegistrationConfig

if TYPE_CHECKING:
    from aind_registration_utils.types import PathLike

_LINEAR_TRANSFORMS = {"Rigid": Rigid, "Similarity": Similarity, "Affine": Affine}


def _num(x: float) -> float | int:
    """Integral floats as ints, so argv reads ``SyN[0.2,4,0]`` not ``4.0,0.0``."""
    return int(x) if float(x).is_integer() else x


def _nums(xs: Sequence[float]) -> list[float | int]:
    return [_num(x) for x in xs]


def _metric(
    kind: str,
    fixed: PathLike,
    moving: PathLike,
    weight: float,
    radius_or_bins: int,
    sampling: str | None = None,
    sampling_pct: float | None = None,
) -> Metric:
    """A commandants metric; ``sampling="None"`` is passed to ANTs as written."""
    if kind == "CC":
        return CC(
            fixed,
            moving,
            weight=_num(weight),
            radius=radius_or_bins,
            sampling=sampling,
            sampling_pct=sampling_pct,
        )
    if kind == "MeanSquares":
        return MeanSquares(
            fixed,
            moving,
            weight=_num(weight),
            sampling=sampling,
            sampling_pct=sampling_pct,
        )
    metric_cls = {"MI": MI, "Mattes": Mattes}[kind]
    return metric_cls(
        fixed,
        moving,
        weight=_num(weight),
        bins=radius_or_bins,
        sampling=sampling,
        sampling_pct=sampling_pct,
    )


def build_registration(
    fixed: PathLike,
    moving: PathLike,
    output_prefix: PathLike,
    config: RegistrationConfig | None = None,
    *,
    fixed_mask: PathLike | None = None,
    moving_mask: PathLike | None = None,
    extra_metrics: Sequence[ExtraMetric] = (),
    initial_transform: PathLike | None = None,
    ants_path: PathLike | None = None,
) -> AntsRegistration:
    """Linear stages then one SyN stage, with every setting stated explicitly.

    Parameters
    ----------
    fixed, moving : PathLike
        Images for the primary metric. The fixed image defines the warp grid.
    output_prefix : PathLike
        ANTs output prefix, e.g. ``out_dir / "reg_"``.
    config : RegistrationConfig or None
        Settings; ``None`` uses the defaults.
    fixed_mask, moving_mask : PathLike or None
        Metric masks, applied to the stages named by ``config.mask_stages``.
    extra_metrics : sequence of ExtraMetric
        Additional SyN metrics, e.g. a second channel.
    initial_transform : PathLike or None
        Transform file to start from; replaces ``config.initialization``.
    ants_path : PathLike or None
        Directory holding the ANTs binaries, ahead of ``$ANTSPATH`` and ``PATH``.

    Returns
    -------
    AntsRegistration
        Unrun builder; pass it to :func:`run_syn_registration`.
    """
    if config is None:
        config = RegistrationConfig()
    prefix = str(output_prefix)
    reg = AntsRegistration(
        dimensionality=3,
        output=prefix,
        warped_output=f"{prefix}Warped.nii.gz" if config.write_warped else None,
        inverse_warped_output=(
            f"{prefix}InverseWarped.nii.gz" if config.write_warped else None
        ),
        interpolation=config.interpolation,
        winsorize=_nums(config.winsorize) if config.winsorize else None,
        use_histogram_matching=config.histogram_matching,
        collapse_output_transforms=True,
        use_float=config.use_float,
        random_seed=config.random_seed,
        verbose=True,
        ants_path=ants_path,
    )
    if initial_transform is not None:
        reg.add_initial_moving_transform(initial_transform)
    elif config.initialization != "none":
        reg.initialize_from_images(fixed, moving, init=config.initialization)

    has_mask = fixed_mask is not None or moving_mask is not None
    linear_masks = (
        {"fixed_mask": fixed_mask, "moving_mask": moving_mask}
        if has_mask and config.mask_stages == "all"
        else {}
    )
    lin = config.linear
    for name in lin.stages:
        reg.add_stage(
            transform=_LINEAR_TRANSFORMS[name](gradient_step=_num(lin.gradient_step)),
            metrics=_metric(
                lin.metric,
                fixed,
                moving,
                weight=1.0,
                radius_or_bins=lin.bins,
                sampling=lin.sampling,
                sampling_pct=lin.sampling_pct,
            ),
            convergence=Convergence(
                lin.iterations, lin.convergence_threshold, lin.convergence_window
            ),
            shrink_factors=lin.shrink_factors,
            smoothing_sigmas=_nums(lin.smoothing_sigmas),
            smoothing_units=lin.smoothing_units,
            **linear_masks,
        )

    syn = config.syn
    metrics = [_metric(syn.metric, fixed, moving, syn.weight, syn.radius_or_bins)]
    metrics += [
        _metric(m.metric, m.fixed, m.moving, m.weight, m.radius_or_bins)
        for m in extra_metrics
    ]
    reg.add_stage(
        transform=SyN(
            gradient_step=_num(syn.gradient_step),
            update_field_variance=_num(syn.update_field_variance),
            total_field_variance=_num(syn.total_field_variance),
        ),
        metrics=metrics,
        convergence=Convergence(
            syn.iterations, syn.convergence_threshold, syn.convergence_window
        ),
        shrink_factors=syn.shrink_factors,
        smoothing_sigmas=_nums(syn.smoothing_sigmas),
        smoothing_units=syn.smoothing_units,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
    )
    return reg
