# tests/test_annotations.py
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


# ---- Minimal fake ANTs shim -------------------------------------------------
class FakeAntsImage:
    """
    Tiny stand-in for ants.ANTsImage with just enough behavior for tests.

    - Holds a numpy array
    - Stores simple metadata (spacing/origin/direction)
    - .view() returns the ndarray (mirrors ANTsPy behavior)
    - .astype() returns a new FakeAntsImage with converted dtype
    """

    def __init__(self, array, spacing=None, origin=None, direction=None):
        arr = np.asarray(array)
        self._arr = arr
        self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
        self.origin = origin if origin is not None else (0.0, 0.0, 0.0)
        # ANTs typically stores a flattened direction cosine matrix
        if direction is None:
            eye = np.eye(arr.ndim, dtype=float).ravel()
            self.direction = tuple(eye)
        else:
            self.direction = tuple(direction)

    def view(self):
        return self._arr

    def astype(self, dtype) -> FakeAntsImage:
        """Mimic ants image astype by returning a new image with same metadata."""
        return FakeAntsImage(
            self._arr.astype(dtype),
            spacing=self.spacing,
            origin=self.origin,
            direction=self.direction,
        )

    def __repr__(self) -> str:
        return f"FakeAntsImage(shape={self._arr.shape}, dtype={self._arr.dtype})"


def fake_from_numpy(arr) -> FakeAntsImage:
    return FakeAntsImage(np.asarray(arr))


def fake_copy_image_info(src: FakeAntsImage, dest: FakeAntsImage) -> FakeAntsImage:
    """Copy spatial metadata FROM src TO dest and return dest."""
    dest.spacing = getattr(src, "spacing", (1.0, 1.0, 1.0))
    dest.origin = getattr(src, "origin", (0.0, 0.0, 0.0))
    ndim = dest.view().ndim
    default_dir = tuple(np.eye(ndim, dtype=float).ravel())
    dest.direction = getattr(src, "direction", default_dir)
    return dest


def fake_apply_transforms(
    fixed: FakeAntsImage,
    moving: FakeAntsImage,
    transformlist=None,
    interpolator: str = "nearestNeighbor",
    **_: object,
) -> FakeAntsImage:
    """
    For unit tests we don't need geometric warping; we only want to validate the
    label mapping round-trip. So we return the moving image with fixed's metadata.
    """
    # Ensure nearest for label images
    assert interpolator == "nearestNeighbor"
    moved = FakeAntsImage(moving.view())
    fake_copy_image_info(fixed, moved)
    return moved


def _install_fake_ants_in_sys_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> types.SimpleNamespace:
    fake_ants = types.SimpleNamespace(
        from_numpy=fake_from_numpy,
        copy_image_info=fake_copy_image_info,
        apply_transforms=fake_apply_transforms,
    )
    # Ensure any import of 'ants' grabs our shim
    monkeypatch.setitem(sys.modules, "ants", fake_ants)
    return fake_ants


# ---- Fixture (function-scoped to match monkeypatch) -------------------------
@pytest.fixture()
def annotations_module(monkeypatch: pytest.MonkeyPatch):
    """
    Import a fresh copy of aind_registration_utils.annotations with our fake
    'ants' module injected BEFORE import, so the module binds to the shim.
    """
    _install_fake_ants_in_sys_modules(monkeypatch)
    # Re-import module cleanly each test
    modname = "aind_registration_utils.annotations"
    if modname in sys.modules:
        del sys.modules[modname]
    mod = importlib.import_module(modname)
    return mod


# ---- Tests ------------------------------------------------------------------
def test_roundtrip_preserves_labels_small_ints(annotations_module) -> None:
    map_annotations_safely = annotations_module.map_annotations_safely

    # Small integer labels with repeats
    data = np.array([[0, 3, 3], [2, 2, 1]], dtype=np.int32)
    moving = FakeAntsImage(data, spacing=(2.0, 2.0, 2.0), origin=(1.0, 1.0, 1.0))
    fixed = FakeAntsImage(np.zeros_like(data))

    out = map_annotations_safely(moving, fixed, transformlist=[])

    # Values unchanged
    np.testing.assert_array_equal(out.view(), data)
    # Output should carry fixed's spatial metadata
    assert out.spacing == fixed.spacing
    assert out.origin == fixed.origin
    assert out.direction == fixed.direction
    # dtype preserved (important for downstream)
    assert out.view().dtype == data.dtype


def test_roundtrip_preserves_huge_integer_labels(annotations_module) -> None:
    map_annotations_safely = annotations_module.map_annotations_safely

    # Mix of small and very large labels; use unsigned to test > 2**31
    huge_vals = np.array(
        [0, 42, 2**40 + 123, 2**48 + 7, 999_999_999_999],
        dtype=np.uint64,
    )
    data = huge_vals.reshape(1, -1)
    moving = FakeAntsImage(data, spacing=(0.5, 0.5, 0.5))
    fixed = FakeAntsImage(np.zeros_like(data))

    out = map_annotations_safely(moving, fixed, transformlist=[])

    np.testing.assert_array_equal(out.view(), data)
    assert out.spacing == fixed.spacing
    # dtype preserved (should remain uint64 here)
    assert out.view().dtype == data.dtype


def test_inputs_not_mutated(annotations_module) -> None:
    map_annotations_safely = annotations_module.map_annotations_safely

    data = np.array([[10, 10], [20, 30]], dtype=np.int64)
    moving = FakeAntsImage(data.copy(), spacing=(3.0, 3.0, 3.0))
    fixed = FakeAntsImage(np.zeros_like(data), spacing=(1.0, 1.0, 1.0))

    # Keep original copies for comparison
    moving_before = moving.view().copy()
    fixed_before_spacing = fixed.spacing

    _ = map_annotations_safely(moving, fixed, transformlist=[])

    # Ensure original arrays and metadata unchanged
    np.testing.assert_array_equal(moving.view(), moving_before)
    assert fixed.spacing == fixed_before_spacing


def test_uses_nearest_neighbor_interpolator(
    annotations_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure nearest-neighbor is used when warping the temporary integer image.
    Our fake apply_transforms asserts this; here we also verify call was made.
    """
    calls = {"count": 0}

    def spying_apply_transforms(
        fixed: FakeAntsImage,
        moving: FakeAntsImage,
        transformlist=None,
        interpolator: str = "nearestNeighbor",
        **kwargs: object,
    ) -> FakeAntsImage:
        calls["count"] += 1
        return fake_apply_transforms(
            fixed,
            moving,
            transformlist=transformlist,
            interpolator=interpolator,
            **kwargs,
        )

    fake_ants = sys.modules["ants"]
    monkeypatch.setattr(
        fake_ants,
        "apply_transforms",
        spying_apply_transforms,
        raising=True,
    )

    map_annotations_safely = annotations_module.map_annotations_safely
    data = np.array([[1, 2], [3, 4]], dtype=np.int16)
    moving = FakeAntsImage(data)
    fixed = FakeAntsImage(np.zeros_like(data))

    _ = map_annotations_safely(moving, fixed, transformlist=["dummy"])

    assert calls["count"] == 1
