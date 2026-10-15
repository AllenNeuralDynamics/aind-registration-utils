"""Registration with the ANTs command-line binaries, via ``commandants``.

ANTs is not a Python dependency: install it separately (``commandants
install-ants``, a release archive, or conda-forge ``ants``) and expose it through
``PATH`` or ``$ANTSPATH``.
"""

from aind_registration_utils.ants_cli.config import (
    ExtraMetric,
    LinearConfig,
    RegistrationConfig,
    SynConfig,
)
from aind_registration_utils.ants_cli.presets import build_registration
from aind_registration_utils.ants_cli.run import (
    AntsInstall,
    SynRegistrationResult,
    find_ants,
    parse_ants_version,
    run_syn_registration,
)

__all__ = [
    "AntsInstall",
    "ExtraMetric",
    "LinearConfig",
    "RegistrationConfig",
    "SynConfig",
    "SynRegistrationResult",
    "build_registration",
    "find_ants",
    "parse_ants_version",
    "run_syn_registration",
]
