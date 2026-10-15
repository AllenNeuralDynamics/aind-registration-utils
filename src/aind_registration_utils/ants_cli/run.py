"""Run a linear + SyN registration with the ANTs binaries and record what ran."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import commandants
import numpy as np
import SimpleITK as sitk
from aind_ants_transform_sidecar import dump_package
from commandants import AntsRegistration, resolve_binary

from aind_registration_utils.domains import ImageHeader, make_syn_sidecar

if TYPE_CHECKING:
    from pydantic import BaseModel

    from aind_registration_utils.types import PathLike

logger = logging.getLogger(__name__)

MIN_ANTS_VERSION = (2, 5, 0)
# The sidecar declares mm; a mouse-brain voxel this large means a µm header.
MAX_PLAUSIBLE_SPACING_MM = 1.0

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


@dataclass(frozen=True)
class AntsInstall:
    """A resolved ``antsRegistration`` binary and its version."""

    binary: Path
    version: tuple[int, int, int]
    version_text: str


@dataclass(frozen=True)
class SynRegistrationResult:
    """Files written by :func:`run_syn_registration` and how they were made."""

    affine: Path
    warp: Path
    inverse_warp: Path
    sidecar: Path
    provenance: Path
    log: Path
    argv: list[str]
    ants_version: str
    duration_seconds: float


def parse_ants_version(text: str) -> tuple[int, int, int]:
    """First ``X.Y.Z`` in ``antsRegistration --version`` output.

    Parameters
    ----------
    text : str
        Version output, e.g. ``"ANTs Version: 2.6.5\\nCompiled: ..."``.

    Returns
    -------
    tuple of int
        Major, minor, patch.

    Raises
    ------
    ValueError
        If no version number is present.
    """
    match = _VERSION_RE.search(text)
    if match is None:
        raise ValueError(f"No ANTs version number in {text!r}")
    major, minor, patch = (int(g) for g in match.groups())
    return major, minor, patch


def find_ants(
    min_version: tuple[int, int, int] = MIN_ANTS_VERSION,
    ants_path: PathLike | None = None,
) -> AntsInstall:
    """Resolve ``antsRegistration`` and check its version.

    Parameters
    ----------
    min_version : tuple of int
        Oldest acceptable version.
    ants_path : PathLike or None
        Directory searched before ``$ANTSPATH`` and ``PATH``.

    Returns
    -------
    AntsInstall
        Binary path and version.

    Raises
    ------
    commandants.AntsNotFoundError
        If no binary is found; the message says how to install one.
    RuntimeError
        If the binary is older than ``min_version``.
    """
    binary = Path(resolve_binary("antsRegistration", ants_path))
    text = commandants.version("antsRegistration", ants_path)
    found = parse_ants_version(text)
    if found < min_version:
        raise RuntimeError(
            f"{binary} is ANTs {'.'.join(map(str, found))}; "
            f"need {'.'.join(map(str, min_version))} or newer"
        )
    return AntsInstall(binary=binary, version=found, version_text=text)


def _header(image: Any) -> ImageHeader:
    """Header of a path or in-memory SimpleITK image, without reading voxels."""
    if isinstance(image, sitk.Image):
        return ImageHeader.from_sitk(image)
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(image))
    reader.ReadImageInformation()
    size = reader.GetSize()
    return ImageHeader(
        origin=reader.GetOrigin(),
        spacing=reader.GetSpacing(),
        direction=reader.GetDirection(),
        size=(size[0], size[1], size[2]),
    )


def _check_mm(header: ImageHeader, role: str, max_spacing_mm: float) -> None:
    if np.any(np.asarray(header.spacing) >= max_spacing_mm):
        raise ValueError(
            f"{role} spacing {header.spacing} is implausible in mm; "
            "is the header in µm?"
        )


def _syn_outputs(reg: AntsRegistration) -> tuple[Path, Path, Path]:
    """Affine, warp and inverse warp paths, or an error for any other layout."""
    files = [Path(f) for f in reg.expected_transforms()["files"]]
    if len(files) != 3 or not files[0].name.endswith("0GenericAffine.mat"):
        raise ValueError(
            "run_syn_registration needs linear stages followed by one SyN stage; "
            f"this registration writes {[f.name for f in files]}"
        )
    return files[0], files[1], files[2]


def run_syn_registration(
    reg: AntsRegistration,
    *,
    threads: int | None = None,
    min_version: tuple[int, int, int] = MIN_ANTS_VERSION,
    log_path: PathLike | None = None,
    sidecar_name: str = "transform_information.json",
    provenance_name: str = "registration_provenance.json",
    config: BaseModel | None = None,
    max_spacing_mm: float = MAX_PLAUSIBLE_SPACING_MM,
) -> SynRegistrationResult:
    """Run a linear + SyN registration and write its sidecar and provenance.

    The SyN stage's first metric defines the fixed and moving images recorded in
    the sidecar, since its fixed image is the warp grid.

    Parameters
    ----------
    reg : AntsRegistration
        Builder, e.g. from :func:`build_registration`.
    threads : int or None
        ITK thread count. One thread with a fixed seed makes runs repeat exactly;
        ``None`` leaves ITK's default.
    min_version : tuple of int
        Oldest acceptable ANTs.
    log_path : PathLike or None
        ANTs output log; defaults to ``registration.log`` beside the transforms.
    sidecar_name, provenance_name : str
        File names written beside the transforms.
    config : pydantic.BaseModel or None
        Settings the builder came from, stored in the provenance.
    max_spacing_mm : float
        Reject inputs whose spacing reaches this, before spending time on ANTs.

    Returns
    -------
    SynRegistrationResult
        Output paths, the command and the ANTs version.

    Raises
    ------
    commandants.AntsRuntimeError
        If ANTs exits non-zero.
    """
    affine, warp, inverse_warp = _syn_outputs(reg)
    syn_metric = reg.stages[-1].metrics[0]
    fixed_header = _header(syn_metric.fixed)
    moving_header = _header(syn_metric.moving)
    _check_mm(fixed_header, "Fixed", max_spacing_mm)
    _check_mm(moving_header, "Moving", max_spacing_mm)
    install = find_ants(min_version, reg.ants_path)
    out_dir = affine.parent.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if threads is not None:
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    log = Path(log_path) if log_path is not None else out_dir / "registration.log"
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as fh:
        fh.write(f"# {install.version_text.splitlines()[0]}\n")
        # In-memory images show as placeholders; the provenance has their paths.
        preview = shlex.join(reg.build_command(resolve=True))
        fh.write(f"# {preview}\n")
        fh.flush()
        logger.info("Running %s", preview)
        completed = reg.run(log_file=fh, env=env)
    duration = time.perf_counter() - t0

    for path in (affine, warp, inverse_warp):
        if not path.is_file():
            raise FileNotFoundError(f"ANTs exited 0 but did not write {path}")

    sidecar_path = out_dir / sidecar_name
    sidecar = make_syn_sidecar(
        fixed_header,
        moving_header,
        affine=affine.name,
        warp=warp.name,
        inverse_warp=inverse_warp.name,
    )
    sidecar_path.write_text(dump_package(sidecar))

    provenance_path = out_dir / provenance_name
    provenance = {
        "argv": completed.argv,
        "command": shlex.join(completed.argv),
        "ants_binary": str(install.binary),
        "ants_version": install.version_text,
        "itk_threads": threads,
        "started_utc": started.isoformat(),
        "duration_seconds": duration,
        "config": config.model_dump(mode="json") if config is not None else None,
        "packages": {
            name: package_version(name)
            for name in ("aind-registration-utils", "commandants")
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    return SynRegistrationResult(
        affine=affine.resolve(),
        warp=warp.resolve(),
        inverse_warp=inverse_warp.resolve(),
        sidecar=sidecar_path,
        provenance=provenance_path,
        log=log,
        argv=completed.argv,
        ants_version=install.version_text,
        duration_seconds=duration,
    )
