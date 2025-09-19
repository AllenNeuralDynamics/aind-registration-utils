"""
Utility functions
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aind_registration_utils.types import PathLike


def check_output_path(output_path: PathLike | None = None) -> Path:
    """
    Check if the provided output path is a valid directory.

    If no output path is provided, the current working directory is used.

    Parameters
    ----------
    output_path : str, optional
        The path to the output directory. Defaults to None.

    Returns
    -------
    pathlib.Path
        The output path as a `Path` object.

    Raises
    ------
    NotADirectoryError
        If the output path is not a directory.
    """
    if output_path is None:
        output_path = os.getcwd()
    if not os.path.isdir(output_path):
        raise NotADirectoryError(f"Output path {output_path} is not a directory")
    return Path(output_path)
