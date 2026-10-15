"""Tests for the ANTs CLI layer that need no ANTs install."""

from __future__ import annotations

import pytest
import SimpleITK as sitk
from commandants import MI, AntsRegistration, Convergence, Rigid
from pydantic import ValidationError

from aind_registration_utils.ants_cli import (
    ExtraMetric,
    LinearConfig,
    RegistrationConfig,
    build_registration,
    parse_ants_version,
    run_syn_registration,
)
from aind_registration_utils.ants_cli import run as run_module


def _argv(reg: AntsRegistration) -> list[str]:
    return reg.build_command()[1:]


def _values(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, tok in enumerate(argv) if tok == flag]


class TestConfig:
    """Settings models."""

    def test_level_mismatch_rejected(self):
        """Each schedule list needs one entry per level."""
        with pytest.raises(ValidationError, match="one entry per level"):
            LinearConfig(
                iterations=[10, 10], shrink_factors=[2, 1], smoothing_sigmas=[0]
            )

    def test_unknown_field_rejected(self):
        """A misspelled setting is an error, not silently ignored."""
        with pytest.raises(ValidationError):
            RegistrationConfig(histogram_match=True)

    def test_json_round_trip(self):
        """A config survives JSON, so a capsule can record and replay it."""
        config = RegistrationConfig.lightsheet_25um()
        assert (
            RegistrationConfig.model_validate_json(config.model_dump_json()) == config
        )


class TestBuildRegistration:
    """Config to argv."""

    def test_defaults_are_explicit(self):
        """Settings ANTs would otherwise default are all stated."""
        argv = _argv(build_registration("f.nii.gz", "m.nii.gz", "out/reg_"))
        assert _values(argv, "--use-histogram-matching") == ["0"]
        assert _values(argv, "--winsorize-image-intensities") == ["[0.005,0.995]"]
        assert _values(argv, "--collapse-output-transforms") == ["1"]
        assert _values(argv, "--initial-moving-transform") == ["[f.nii.gz,m.nii.gz,1]"]
        assert _values(argv, "--transform") == [
            "Rigid[0.1]",
            "Affine[0.1]",
            "SyN[0.2,4,0]",
        ]
        assert _values(argv, "--output") == [
            "[out/reg_,out/reg_Warped.nii.gz,out/reg_InverseWarped.nii.gz]"
        ]
        assert "--masks" not in argv

    def test_masks_on_syn_only_by_default(self):
        """Linear stages run unmasked; the SyN stage gets both masks."""
        reg = build_registration(
            "f.nii.gz",
            "m.nii.gz",
            "reg_",
            fixed_mask="fm.nii.gz",
            moving_mask="mm.nii.gz",
        )
        assert _values(_argv(reg), "--masks") == [
            "[NA,NA]",
            "[NA,NA]",
            "[fm.nii.gz,mm.nii.gz]",
        ]

    def test_masks_on_all_stages(self):
        """``mask_stages="all"`` masks every stage."""
        reg = build_registration(
            "f.nii.gz",
            "m.nii.gz",
            "reg_",
            RegistrationConfig(mask_stages="all"),
            fixed_mask="fm.nii.gz",
        )
        assert _values(_argv(reg), "--masks") == ["[fm.nii.gz,NA]"] * 3

    def test_extra_metric_joins_syn_stage(self):
        """A second channel adds a metric to the SyN stage only."""
        reg = build_registration(
            "f.nii.gz",
            "m.nii.gz",
            "reg_",
            extra_metrics=[
                ExtraMetric(fixed="f2.nii.gz", moving="m2.nii.gz", weight=0.3)
            ],
        )
        metrics = _values(_argv(reg), "--metric")
        assert metrics[-2:] == [
            "CC[f.nii.gz,m.nii.gz,1,4]",
            "CC[f2.nii.gz,m2.nii.gz,0.3,4]",
        ]
        assert len(metrics) == 4

    def test_initial_transform_replaces_initialization(self):
        """A given transform file is the only initial transform."""
        reg = build_registration(
            "f.nii.gz", "m.nii.gz", "reg_", initial_transform="init.mat"
        )
        assert _values(_argv(reg), "--initial-moving-transform") == ["init.mat"]

    def test_mm_smoothing_and_seed(self):
        """The light-sheet preset smooths in mm and fixes the seed."""
        argv = _argv(
            build_registration(
                "f.nii.gz", "m.nii.gz", "reg_", RegistrationConfig.lightsheet_25um()
            )
        )
        assert set(_values(argv, "--smoothing-sigmas")) == {"0.2x0.1x0.05x0mm"}
        assert _values(argv, "--random-seed") == ["1"]


class TestVersion:
    """ANTs version parsing and checking."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("ANTs Version: 2.6.5\nCompiled: Jan 14 2026", (2, 6, 5)),
            ("ANTs Version: v2.5.1.post95-g2abc", (2, 5, 1)),
        ],
    )
    def test_parse(self, text, expected):
        """The first X.Y.Z wins, with or without a v prefix or suffix."""
        assert parse_ants_version(text) == expected

    def test_parse_garbage(self):
        """Output without a version number is an error."""
        with pytest.raises(ValueError, match="No ANTs version"):
            parse_ants_version("command not found")

    def test_too_old(self, monkeypatch):
        """An old binary is rejected with both versions named."""
        monkeypatch.setattr(
            run_module, "resolve_binary", lambda *a: "/opt/ants/antsRegistration"
        )
        monkeypatch.setattr(
            run_module.commandants, "version", lambda *a: "ANTs Version: 2.3.5"
        )
        with pytest.raises(RuntimeError, match=r"2\.3\.5.*need 2\.5\.0"):
            run_module.find_ants()


class TestRunChecks:
    """Checks that fail before ANTs is looked up."""

    def test_non_syn_layout_rejected(self):
        """A registration without a SyN stage has no triplet to record."""
        reg = AntsRegistration(3, "reg_")
        reg.add_stage(Rigid(), MI("f.nii.gz", "m.nii.gz"), Convergence([10]), [1], [0])
        with pytest.raises(ValueError, match="linear stages followed by one SyN"):
            run_syn_registration(reg)

    def test_micrometer_header_rejected(self, tmp_path):
        """A 25 µm image written as spacing 25 is caught before running."""
        image = sitk.Image([4, 4, 4], sitk.sitkFloat32)
        image.SetSpacing([25.0] * 3)
        path = tmp_path / "um.nii.gz"
        sitk.WriteImage(image, str(path))
        reg = build_registration(path, path, tmp_path / "reg_")
        with pytest.raises(ValueError, match="µm"):
            run_syn_registration(reg)
