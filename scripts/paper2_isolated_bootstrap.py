#!/usr/bin/env python3
"""Run one Paper-2 runner without site initialization and attest its runtime.

This entrypoint is intentionally stdlib-only until it has verified ``-I -B -S``.
It adds exactly the repository root and the virtual environment's single
``site-packages`` directory; Python never processes ``.pth`` files or imports
``sitecustomize``/``usercustomize``.  The target runner is executed once by
``runpy`` with the normal ``__main__`` binding.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import runpy
import sys
import sysconfig
from typing import Any, Sequence


SCHEMA = "paper2_isolated_runtime_attestation_v2"
PACKAGES = ("numpy", "simpy", "gymnasium", "scipy", "pandas")
PACKAGE_MANIFEST_SCHEMA = "paper2_distribution_installed_files_v1"
PACKAGE_MANIFEST_EXCLUSIONS = {
    "cache": ("__pycache__", ".pyc", ".pyo"),
    "record_signatures": ("RECORD.jws", "RECORD.p7s"),
    "outside_site_packages": True,
}
NATIVE_SUFFIXES = {".so", ".dylib", ".pyd", ".dll", ".a", ".lib"}
THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class BootstrapError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _exclusive_write_json(path: Path, value: Any) -> None:
    """Create one custody artifact without following or replacing a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _venv_site_packages() -> Path:
    venv = Path(sys.executable).absolute().parent.parent
    candidates = (
        venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        venv / "Lib" / "site-packages",
    )
    matches = [candidate.resolve() for candidate in candidates if candidate.is_dir()]
    if len(matches) != 1:
        raise BootstrapError(
            f"expected exactly one virtual-environment site-packages directory, found {matches}"
        )
    return matches[0]


def _binary_record(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"present": False, "path": None, "resolved_path": None, "sha256": None}
    lexical = Path(path_value).absolute()
    resolved = lexical.resolve()
    return {
        "present": resolved.is_file(),
        "path": str(lexical),
        "resolved_path": str(resolved),
        "symlink_target": os.readlink(lexical) if lexical.is_symlink() else None,
        "sha256": _sha256_file(resolved) if resolved.is_file() else None,
    }


def _libpython_record() -> dict[str, Any]:
    libdir = sysconfig.get_config_var("LIBDIR")
    library = sysconfig.get_config_var("LDLIBRARY")
    candidate = Path(str(libdir)) / str(library) if libdir and library else None
    record = _binary_record(str(candidate) if candidate is not None else None)
    record.update(
        {
            "libdir": libdir,
            "ldlibrary": library,
            "py_enable_shared": sysconfig.get_config_var("Py_ENABLE_SHARED"),
        }
    )
    return record


def _module_records(repo_root: Path, site_packages: Path, runner: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    candidates: list[tuple[str, Any]] = list(sys.modules.items())
    candidates.append(("__paper2_runner__", type("RunnerFile", (), {"__file__": str(runner), "__spec__": None})()))
    for name, module in sorted(candidates, key=lambda row: row[0]):
        file_value = getattr(module, "__file__", None)
        if not isinstance(file_value, str):
            continue
        path = Path(file_value)
        if path.suffix in {".pyc", ".pyo"} and path.with_suffix(".py").is_file():
            path = path.with_suffix(".py")
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not resolved.is_file():
            continue
        key = (name, str(resolved))
        if key in seen:
            continue
        seen.add(key)
        spec = getattr(module, "__spec__", None)
        loader = getattr(spec, "loader", None)
        try:
            relative_repo = str(resolved.relative_to(repo_root))
        except ValueError:
            relative_repo = None
        try:
            relative_site = str(resolved.relative_to(site_packages))
        except ValueError:
            relative_site = None
        records.append(
            {
                "module": name,
                "origin": str(resolved),
                "relative_to_repo": relative_repo,
                "relative_to_site_packages": relative_site,
                "loader": (
                    f"{type(loader).__module__}.{type(loader).__qualname__}"
                    if loader is not None
                    else None
                ),
                "suffix": resolved.suffix,
                "sha256": _sha256_file(resolved),
            }
        )
    return records


def _native_extension_records(site_packages: Path) -> list[dict[str, Any]]:
    """Hash every installable native extension, including lazily imported code."""
    suffixes = {".so", ".dylib", ".pyd", ".dll"}
    return [
        {
            "path": str(path.resolve()),
            "relative_to_site_packages": str(path.resolve().relative_to(site_packages)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(site_packages.rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def _distribution_installed_file_manifests(
    site_packages: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Hash installed dependency files without treating platform code as portable.

    Every regular file reported by ``importlib.metadata`` below site-packages is
    hashed.  Only bytecode/cache files and detached RECORD signatures are
    excluded.  Pure Python plus package-data bytes form the cross-host portable
    digest; native/shared-library bytes receive a separate host digest.
    """
    manifests: dict[str, Any] = {}
    portable_summaries: dict[str, Any] = {}
    native_summaries: dict[str, Any] = {}
    for package in PACKAGES:
        try:
            distribution = metadata.distribution(package)
        except metadata.PackageNotFoundError as exc:
            raise BootstrapError(
                f"scientific dependency distribution is missing: {package}"
            ) from exc
        declared = distribution.files
        if declared is None:
            raise BootstrapError(
                f"scientific dependency has no installed-file manifest: {package}"
            )
        records: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []
        for declared_path in sorted(declared, key=lambda value: str(value)):
            relative_declared = str(declared_path)
            parts = Path(relative_declared).parts
            name = Path(relative_declared).name
            suffix = Path(relative_declared).suffix.lower()
            if "__pycache__" in parts or suffix in {".pyc", ".pyo"}:
                excluded.append(
                    {"declared_path": relative_declared, "reason": "mutable_cache"}
                )
                continue
            if name in {"RECORD.jws", "RECORD.p7s"}:
                excluded.append(
                    {
                        "declared_path": relative_declared,
                        "reason": "record_signature",
                    }
                )
                continue
            installed = Path(distribution.locate_file(declared_path))
            try:
                resolved = installed.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise BootstrapError(
                    f"installed dependency file is missing: {package}:{relative_declared}"
                ) from exc
            try:
                relative_site = resolved.relative_to(site_packages)
            except ValueError:
                excluded.append(
                    {
                        "declared_path": relative_declared,
                        "reason": "outside_site_packages",
                    }
                )
                continue
            if not resolved.is_file():
                raise BootstrapError(
                    f"installed dependency record is not a file: {package}:{relative_declared}"
                )
            is_metadata = any(part.endswith((".dist-info", ".egg-info")) for part in relative_site.parts)
            is_native = (
                suffix in NATIVE_SUFFIXES
                or any(part.endswith((".libs", ".dylibs")) for part in relative_site.parts)
                or re.search(r"\.(?:so|dylib|pyd|dll|a|lib)(?:\.|$)", name) is not None
            )
            classification = (
                "distribution_metadata"
                if is_metadata
                else "host_native"
                if is_native
                else "python_source"
                if suffix in {".py", ".pyi"}
                else "package_data"
            )
            try:
                size = resolved.stat().st_size
                digest = _sha256_file(resolved)
            except OSError as exc:
                raise BootstrapError(
                    f"installed dependency file is unreadable: {package}:{relative_declared}"
                ) from exc
            records.append(
                {
                    "declared_path": relative_declared,
                    "relative_to_site_packages": str(relative_site),
                    "bytes": size,
                    "sha256": digest,
                    "classification": classification,
                }
            )
        portable_records = [
            row
            for row in records
            if row["classification"] in {"python_source", "package_data"}
        ]
        native_records = [
            row for row in records if row["classification"] == "host_native"
        ]
        manifest_body = {
            "schema_version": PACKAGE_MANIFEST_SCHEMA,
            "package": package,
            "distribution_name": distribution.metadata["Name"],
            "version": distribution.version,
            "exclusion_schema": PACKAGE_MANIFEST_EXCLUSIONS,
            "files": records,
            "excluded": excluded,
        }
        manifests[package] = {
            **manifest_body,
            "manifest_sha256": _canonical_sha256(manifest_body),
        }
        portable_summaries[package] = {
            "file_count": len(portable_records),
            "files_sha256": _canonical_sha256(portable_records),
        }
        native_summaries[package] = {
            "file_count": len(native_records),
            "files_sha256": _canonical_sha256(native_records),
        }
    return manifests, portable_summaries, native_summaries


def _customizer_record(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    loader = getattr(spec, "loader", None) if spec is not None else None
    return {
        "name": name,
        "loaded": name in sys.modules,
        "discoverable": spec is not None,
        "origin": getattr(spec, "origin", None) if spec is not None else None,
        "loader": (
            f"{type(loader).__module__}.{type(loader).__qualname__}"
            if loader is not None
            else None
        ),
    }


def runtime_attestation(repo_root: Path, runner: Path, site_packages: Path) -> dict[str, Any]:
    requirements = {}
    for name in ("requirements.txt", "requirements-pinned.txt"):
        path = repo_root / name
        requirements[name] = _sha256_file(path) if path.is_file() else None
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    pth_records = [
        {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "processed": False,
            "contains_executable_line": any(
                line.lstrip().startswith(("import ", "import\t"))
                for line in path.read_text(errors="replace").splitlines()
            ),
        }
        for path in sorted(site_packages.glob("*.pth"))
        if path.is_file()
    ]
    customizers = [_customizer_record(name) for name in ("sitecustomize", "usercustomize")]
    forbidden_python_environment = sorted(
        key for key in os.environ if key.startswith("PYTHON")
    )
    flags = {
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": sys.flags.safe_path,
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
        "hash_randomization": sys.flags.hash_randomization,
    }
    (
        distribution_manifests,
        portable_distribution_files,
        host_native_distribution_files,
    ) = _distribution_installed_file_manifests(site_packages)
    checks = {
        "isolated": flags["isolated"] == 1,
        "no_site": flags["no_site"] == 1,
        "no_user_site": flags["no_user_site"] == 1,
        "safe_path": flags["safe_path"] is True,
        "dont_write_bytecode": flags["dont_write_bytecode"] is True,
        "site_module_not_loaded": "site" not in sys.modules,
        "customizers_absent_and_not_loaded": all(
            not row["discoverable"] and not row["loaded"] for row in customizers
        ),
        "pth_files_not_processed": all(row["processed"] is False for row in pth_records),
        "python_environment_absent": not forbidden_python_environment,
    }
    portable = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "packages": versions,
        "distribution_manifest_schema": PACKAGE_MANIFEST_SCHEMA,
        "distribution_manifest_exclusions": PACKAGE_MANIFEST_EXCLUSIONS,
        "portable_distribution_files": portable_distribution_files,
        "requirements_sha256": requirements,
        "runner_relative": str(runner.relative_to(repo_root)),
        "runner_sha256": _sha256_file(runner),
        "bootstrap_relative": str(Path(__file__).resolve().relative_to(repo_root)),
        "bootstrap_sha256": _sha256_file(Path(__file__).resolve()),
    }
    host = {
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "interpreter": _binary_record(sys.executable),
        "base_interpreter": _binary_record(getattr(sys, "_base_executable", None)),
        "libpython": _libpython_record(),
        "flags": flags,
        "sys_path": [str(Path(value).resolve()) if value else value for value in sys.path],
        "site_packages": str(site_packages),
        "pth_files": pth_records,
        "customizers": customizers,
        "forbidden_python_environment": forbidden_python_environment,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS},
        "modules": _module_records(repo_root, site_packages, runner),
        "native_extensions": _native_extension_records(site_packages),
        "distribution_installed_files": distribution_manifests,
        "host_native_distribution_files": host_native_distribution_files,
        "worker_boundary": (
            "multiprocessing descendants inherit the isolated parent flags and ordered "
            "sys.path; every site-packages native extension is hashed before main"
        ),
    }
    body = {
        "schema_version": SCHEMA,
        "portable": portable,
        "portable_sha256": _canonical_sha256(portable),
        "host": host,
        "isolation_checks": checks,
        "isolation_checks_passed": all(checks.values()),
    }
    return {**body, "runtime_sha256": _canonical_sha256(body)}


def _parse(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" in values:
        separator = values.index("--")
        bootstrap_values, runner_values = values[:separator], values[separator + 1 :]
    else:
        bootstrap_values, runner_values = values, []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--attestation-output", type=Path)
    parser.add_argument("--expected-runtime-sha256")
    parser.add_argument("--execution-nonce")
    parser.add_argument("--execution-role")
    parser.add_argument("--attest-only", action="store_true")
    return parser.parse_args(bootstrap_values), runner_values


def main(argv: Sequence[str] | None = None) -> int:
    args, runner_argv = _parse(argv)
    repo_root = args.repo_root.resolve(strict=True)
    runner = args.runner.resolve(strict=True)
    try:
        runner.relative_to(repo_root)
    except ValueError as exc:
        raise BootstrapError("runner escapes the repository root") from exc
    if not runner.is_file() or runner.suffix != ".py":
        raise BootstrapError("runner must be a Python source file")
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.safe_path and sys.dont_write_bytecode):
        raise BootstrapError("bootstrap requires Python -I -B -S")
    if "site" in sys.modules or "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise BootstrapError("site/customizer code loaded before bootstrap")
    if any(key.startswith("PYTHON") for key in os.environ):
        raise BootstrapError("scientific child environment contains PYTHON* variables")

    site_packages = _venv_site_packages()
    sys.path[:0] = [str(repo_root), str(site_packages)]
    attestation = runtime_attestation(repo_root, runner, site_packages)
    if not attestation["isolation_checks_passed"]:
        raise BootstrapError("runtime isolation attestation failed closed")
    if args.expected_runtime_sha256 is not None and (
        not re.fullmatch(r"[0-9a-f]{64}", args.expected_runtime_sha256)
        or args.expected_runtime_sha256 != attestation["runtime_sha256"]
    ):
        raise BootstrapError("host runtime attestation digest mismatch")
    if args.attestation_output is not None:
        _exclusive_write_json(args.attestation_output.resolve(), attestation)
    if args.attest_only:
        print(json.dumps(attestation, sort_keys=True))
        return 0
    if not isinstance(args.execution_nonce, str) or not re.fullmatch(
        r"[0-9a-f]{64}", args.execution_nonce
    ):
        raise BootstrapError("execution nonce is missing or malformed")
    if not isinstance(args.execution_role, str) or not args.execution_role.strip():
        raise BootstrapError("execution role is missing")
    os.environ["SCRES_EXECUTION_NONCE"] = args.execution_nonce
    os.environ["SCRES_EXECUTION_ROLE"] = args.execution_role
    os.environ["SCRES_HOST_RUNTIME_SHA256"] = attestation["runtime_sha256"]
    os.environ["SCRES_PORTABLE_RUNTIME_SHA256"] = attestation["portable_sha256"]
    previous_argv = sys.argv
    try:
        sys.argv = [str(runner), *runner_argv]
        try:
            runpy.run_path(str(runner), run_name="__main__")
            result = 0
        except SystemExit as exc:
            if exc.code is None:
                result = 0
            elif isinstance(exc.code, int):
                result = exc.code
            else:
                print(str(exc.code), file=sys.stderr)
                result = 1
    finally:
        sys.argv = previous_argv
    return int(result or 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
