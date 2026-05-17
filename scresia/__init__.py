"""Compatibility namespace for repo-root imports.

This package supports notebooks that run from inside the repository directory:

    from scresia.supply_chain import MFSCSimulation

When the repository itself is cloned as a folder named ``scresia`` and imported
from its parent directory, the repository-root ``__init__.py`` handles the same
public namespace.
"""

from __future__ import annotations

import importlib
import sys


def _alias_loaded_package(package_name: str) -> None:
    package = importlib.import_module(package_name)
    sys.modules[f"{__name__}.{package_name}"] = package
    prefix = f"{package_name}."
    for module_name, module in list(sys.modules.items()):
        if module_name == package_name or module_name.startswith(prefix):
            sys.modules[f"{__name__}.{module_name}"] = module


_alias_loaded_package("supply_chain")
_alias_loaded_package("scripts")

from supply_chain import (  # noqa: E402,F401
    ExternalEnvSpec,
    MFSCGymEnv,
    MFSCGymEnvShifts,
    MFSCSimulation,
    get_shift_control_env_spec,
    get_track_b_env_spec,
    make_dkana_track_b_env,
    make_shift_control_env,
    make_track_b_env,
    resolve_hours_per_year,
    spec_to_dict,
)

__all__ = [
    "ExternalEnvSpec",
    "MFSCGymEnv",
    "MFSCGymEnvShifts",
    "MFSCSimulation",
    "get_shift_control_env_spec",
    "get_track_b_env_spec",
    "make_dkana_track_b_env",
    "make_shift_control_env",
    "make_track_b_env",
    "resolve_hours_per_year",
    "spec_to_dict",
]
