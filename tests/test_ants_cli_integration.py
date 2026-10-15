"""Run a real ANTs registration and check the SimpleITK chains against ANTs."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import SimpleITK as sitk
from aind_ants_transform_sidecar import load_package
from commandants import AntsApplyTransforms, AntsApplyTransformsToPoints, is_available
from commandants.io.points import read_points, write_points

from aind_registration_utils.ants_cli import (
    LinearConfig,
    RegistrationConfig,
    SynConfig,
    build_registration,
    run_syn_registration,
)
from aind_registration_utils.chains import (
    SynTransform,
    composite_transform,
    resample,
    transform_points,
)

pytestmark = pytest.mark.skipif(
    not is_available("antsRegistration"), reason="ANTs binaries not found"
)

SHIFT_MM = np.array([0.3, -0.2, 0.1])
SPACING_MM = 0.1
N_VOX = 40
# CCF-style PIR index axes: P, I, R.
PIR_DIRECTION = (0, 0, -1, 1, 0, 0, 0, -1, 0)


def _phantom() -> sitk.Image:
    """Asymmetric blobs on an identity-direction grid."""
    coords = np.mgrid[0:N_VOX, 0:N_VOX, 0:N_VOX] * SPACING_MM
    centers = [(1.9, 1.9, 1.9), (1.2, 2.6, 1.6), (2.7, 1.3, 2.4), (1.8, 2.8, 2.9)]
    arr = sum(
        np.exp(
            -((coords[0] - z) ** 2 + (coords[1] - y) ** 2 + (coords[2] - x) ** 2) / 0.2
        )
        for x, y, z in centers
    )
    image = sitk.GetImageFromArray(arr.astype(np.float32))
    image.SetSpacing([SPACING_MM] * 3)
    return image


def _shifted_pir(fixed: sitk.Image) -> sitk.Image:
    """The phantom moved by SHIFT_MM, sampled on a PIR grid over the same box."""
    extent = (N_VOX - 1) * SPACING_MM
    reference = sitk.Image([N_VOX] * 3, sitk.sitkFloat32)
    reference.SetSpacing([SPACING_MM] * 3)
    reference.SetDirection(PIR_DIRECTION)
    reference.SetOrigin((extent, 0.0, extent))
    # moving(x) = fixed(x - shift), so fixed point p sits at moving point p + shift.
    return sitk.Resample(
        fixed,
        reference,
        sitk.TranslationTransform(3, tuple(-SHIFT_MM)),
        sitk.sitkLinear,
    )


@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    """One small registration shared by every test."""
    root = tmp_path_factory.mktemp("reg")
    fixed = root / "fixed.nii.gz"
    moving = root / "moving.nii.gz"
    phantom = _phantom()
    sitk.WriteImage(phantom, str(fixed))
    sitk.WriteImage(_shifted_pir(phantom), str(moving))
    config = RegistrationConfig(
        linear=LinearConfig(
            iterations=[200, 100], shrink_factors=[2, 1], smoothing_sigmas=[1, 0]
        ),
        syn=SynConfig(
            radius_or_bins=2,
            iterations=[20, 10],
            shrink_factors=[2, 1],
            smoothing_sigmas=[1, 0],
        ),
        random_seed=1,
    )
    reg = build_registration(fixed, moving, root / "out" / "reg_", config)
    result = run_syn_registration(reg, threads=1, config=config)
    return {"root": root, "fixed": fixed, "moving": moving, "result": result}


@pytest.fixture(scope="module")
def synt(registered):
    """The registration as a SimpleITK chain."""
    return SynTransform.from_sidecar(registered["result"].sidecar)


def _inside_points(n: int = 50) -> np.ndarray:
    """Points well inside both grids."""
    return np.random.default_rng(0).uniform(1.0, 2.9, size=(n, 3))


def _ants_points(points, transforms, tmp_path) -> np.ndarray:
    """``antsApplyTransformsToPoints`` with ``(file, invert)`` in -t order."""
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    write_points(src, points)
    cmd = AntsApplyTransformsToPoints(3, src, dst, precision=1)
    for path, invert in transforms:
        cmd.add_transform(path, invert=invert)
    cmd.run()
    return np.asarray(read_points(dst)[0], float)


class TestRun:
    """Outputs of run_syn_registration."""

    def test_files_and_records(self, registered):
        """Transforms, sidecar, provenance and log are written and consistent."""
        result = registered["result"]
        for path in (result.affine, result.warp, result.inverse_warp):
            assert path.is_file()
        sidecar = load_package(result.sidecar.read_text())
        assert sidecar.transform.affine == "reg_0GenericAffine.mat"
        assert sidecar.moving_domain.shape_canonical == (N_VOX,) * 3
        provenance = json.loads(result.provenance.read_text())
        assert provenance["argv"] == result.argv
        assert provenance["config"]["random_seed"] == 1
        assert provenance["itk_threads"] == 1
        header = result.log.read_text().splitlines()[:2]
        assert header[0].startswith("# ANTs Version")
        assert "antsRegistration" in header[1]

    def test_recovers_shift(self, synt):
        """A fixed point maps to itself plus the known shift."""
        center = np.array([[1.95, 1.95, 1.95]])
        moved = synt.transform_points(center, "fixed_to_moving")
        np.testing.assert_allclose(moved[0] - center[0], SHIFT_MM, atol=0.02)


class TestChainsMatchAnts:
    """SimpleITK chains against the ANTs binaries."""

    def test_fixed_to_moving_points(self, registered, synt, tmp_path):
        """Matches ``-t Warp -t Affine``."""
        r = registered["result"]
        pts = _inside_points()
        ants = _ants_points(pts, [(r.warp, False), (r.affine, False)], tmp_path)
        ours = synt.transform_points(pts, "fixed_to_moving")
        np.testing.assert_allclose(ours, ants, atol=1e-4)

    def test_moving_to_fixed_points(self, registered, synt, tmp_path):
        """Matches ``-t [Affine,1] -t InverseWarp``."""
        r = registered["result"]
        pts = _inside_points() + SHIFT_MM
        ants = _ants_points(pts, [(r.affine, True), (r.inverse_warp, False)], tmp_path)
        ours = synt.transform_points(pts, "moving_to_fixed")
        np.testing.assert_allclose(ours, ants, atol=1e-4)

    def test_resample_moving_onto_fixed(self, registered, synt, tmp_path):
        """Matches ``antsApplyTransforms`` with linear interpolation."""
        r = registered["result"]
        out = tmp_path / "warped.nii.gz"
        cmd = AntsApplyTransforms(
            3,
            registered["moving"],
            registered["fixed"],
            out,
            interpolation="Linear",
            use_float=False,
        )
        cmd.add_transform(r.warp).add_transform(r.affine)
        cmd.run()
        moving = sitk.ReadImage(str(registered["moving"]))
        fixed = sitk.ReadImage(str(registered["fixed"]))
        ours = sitk.GetArrayFromImage(
            resample(moving, fixed, [(synt, "fixed_to_moving")])
        )
        ants = sitk.GetArrayFromImage(sitk.ReadImage(str(out)))
        assert np.abs(ours - ants).max() < 1e-4 * np.abs(ants).max()


class TestChains:
    """Composition and domain checks."""

    def test_round_trip(self, synt):
        """Out and back returns within a tenth of a voxel."""
        pts = _inside_points()
        back = transform_points(
            pts, [(synt, "fixed_to_moving"), (synt, "moving_to_fixed")]
        )
        np.testing.assert_allclose(back, pts, atol=0.1 * SPACING_MM)

    def test_composite_matches_stepwise(self, synt):
        """The composite applies hops in list order, like transform_points."""
        hops = [(synt, "fixed_to_moving"), (synt, "moving_to_fixed")]
        composite = composite_transform(hops)
        pts = _inside_points(5)
        stepwise = transform_points(pts, hops)
        np.testing.assert_allclose(
            [composite.TransformPoint(p) for p in pts], stepwise, atol=1e-9
        )

    def test_outside_field(self, synt):
        """Points off the warp grid raise, warn, or pass as asked."""
        far = np.array([[50.0, 50.0, 50.0]])
        with pytest.raises(ValueError, match="outside"):
            synt.transform_points(far, "fixed_to_moving")
        with pytest.warns(UserWarning, match="outside"):
            synt.transform_points(far, "fixed_to_moving", outside="warn")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            synt.transform_points(far, "fixed_to_moving", outside="ignore")

    def test_mismatched_sidecar_rejected(self, registered, tmp_path):
        """A sidecar whose fixed domain is not the warp grid is caught on load."""
        r = registered["result"]
        data = json.loads(r.sidecar.read_text())
        data["fixed_domain"]["bbox"]["L"] = [0.5, 4.4]
        data["fixed_domain"].pop("spatial_signature", None)
        bad = SynTransform(load_package(data), r.sidecar.parent)
        with pytest.raises(ValueError, match="fixed domain"):
            bad.transform_points(_inside_points(1), "fixed_to_moving")
