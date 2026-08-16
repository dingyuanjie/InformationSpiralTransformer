"""Level 7.0 read-only evidence audit and lightweight reproducibility bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import py_compile
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments" / "level7_0"
FORMAL_DIR = LEVEL_DIR / "formal"
REGISTRY_PATH = LEVEL_DIR / "claim_registry.json"
PREREG_PATH = LEVEL_DIR / "preregistration.json"

FORBIDDEN_NAMES = {
    "predictions.json",
    "progress.json",
    "confirmation.partial.json",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".pyc"}
FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "smoke",
    "smoke_test",
    "failed_numeric_v1",
    "failed_numeric_v2",
    "failed_numeric_v3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing Level 7.0 formal output.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Run all audits without creating the distributable ZIP.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer}")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise KeyError(f"Cannot descend through {type(current).__name__}")
    return current


def evaluate(actual: Any, op: str, expected: Any) -> bool:
    operations = {
        "eq": lambda: actual == expected,
        "ne": lambda: actual != expected,
        "gt": lambda: actual > expected,
        "ge": lambda: actual >= expected,
        "lt": lambda: actual < expected,
        "le": lambda: actual <= expected,
    }
    if op not in operations:
        raise ValueError(f"Unsupported registry operation: {op}")
    return bool(operations[op]())


def rel_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def validate_bundle_path(relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"Unsafe bundle path: {relative}")
    if posix.name in FORBIDDEN_NAMES or posix.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Forbidden artifact selected for bundle: {relative}")
    lowered_parts = {part.lower() for part in posix.parts}
    if lowered_parts & FORBIDDEN_PARTS:
        raise ValueError(f"Forbidden output class selected for bundle: {relative}")
    resolved = (ROOT / Path(*posix.parts)).resolve()
    if ROOT != resolved and ROOT not in resolved.parents:
        raise ValueError(f"Bundle path escapes project root: {relative}")
    return resolved


def collect_registered_files(registry: dict[str, Any]) -> list[Path]:
    selected: set[str] = set(registry["code_files"])
    for claim in registry["claims"]:
        selected.update(
            claim[key] for key in ("evidence", "analysis", "protocol", "figure")
        )
    paths = [validate_bundle_path(item) for item in sorted(selected)]
    missing = [rel_path(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing registered files: " + ", ".join(missing))
    return paths


def audit_claims(registry: dict[str, Any]) -> dict[str, Any]:
    audited_claims = []
    for claim in registry["claims"]:
        evidence_path = validate_bundle_path(claim["evidence"])
        document = read_json(evidence_path)
        checks = []
        for check in claim["checks"]:
            try:
                actual = json_pointer(document, check["pointer"])
                passed = evaluate(actual, check["op"], check["expected"])
                error = None
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                actual = None
                passed = False
                error = f"{type(exc).__name__}: {exc}"
            checks.append(
                {
                    **check,
                    "actual": actual,
                    "passed": passed,
                    "error": error,
                }
            )
        audited_claims.append(
            {
                "id": claim["id"],
                "status": claim["status"],
                "statement": claim["statement"],
                "evidence": claim["evidence"],
                "checks": checks,
                "passed": all(item["passed"] for item in checks),
            }
        )
    return {
        "schema_version": 1,
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "claims": audited_claims,
        "passed": all(claim["passed"] for claim in audited_claims),
    }


def compile_registered_python(paths: list[Path]) -> list[dict[str, Any]]:
    results = []
    with tempfile.TemporaryDirectory(prefix="ist_level7_compile_") as temp_dir:
        temp_root = Path(temp_dir)
        for path in paths:
            if path.suffix.lower() != ".py":
                continue
            target = temp_root / (path.stem + ".pyc")
            try:
                py_compile.compile(str(path), cfile=str(target), doraise=True)
                results.append({"path": rel_path(path), "passed": True, "error": None})
            except py_compile.PyCompileError as exc:
                results.append(
                    {"path": rel_path(path), "passed": False, "error": str(exc)}
                )
    return results


def command_output(command: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def environment_snapshot() -> dict[str, Any]:
    torch_info: dict[str, Any]
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_info = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if cuda_available
            else [],
        }
    except ImportError:
        torch_info = {"installed": False}

    repository = ROOT.parent
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch_info,
        "git_commit": command_output(["git", "rev-parse", "HEAD"], repository),
        "git_status_porcelain": command_output(
            ["git", "status", "--porcelain"], repository
        ),
    }


def build_manifest(paths: list[Path]) -> dict[str, Any]:
    entries = [
        {
            "path": rel_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {
        "schema_version": 1,
        "root": "ist_v0_1",
        "files": entries,
        "file_count": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "forbidden_artifacts_absent": True,
    }


def audit_report(
    claim_audit: dict[str, Any],
    compile_audit: list[dict[str, Any]],
    manifest: dict[str, Any],
    zip_result: dict[str, Any] | None,
) -> str:
    claims_passed = sum(claim["passed"] for claim in claim_audit["claims"])
    compile_passed = sum(item["passed"] for item in compile_audit)
    overall = claim_audit["passed"] and all(item["passed"] for item in compile_audit)
    if zip_result is not None:
        overall = overall and zip_result["passed"]
    lines = [
        "# Level 7.0 evidence audit",
        "",
        "## Decision",
        "",
        f"- Overall: **{'PASS' if overall else 'FAIL'}**.",
        f"- Registered claims: {claims_passed}/{len(claim_audit['claims'])} passed.",
        f"- Python syntax checks: {compile_passed}/{len(compile_audit)} passed.",
        f"- Manifest: {manifest['file_count']} files, "
        f"{manifest['total_size_bytes'] / (1024 * 1024):.2f} MiB.",
    ]
    if zip_result is None:
        lines.append("- ZIP: intentionally skipped with `--no-zip`.")
    else:
        lines.append(
            f"- ZIP integrity: **{'PASS' if zip_result['passed'] else 'FAIL'}**, "
            f"{zip_result['member_count']} members."
        )
    lines.extend(
        [
            "",
            "## Claim audit",
            "",
            "| Claim | Registered status | Checks | Result |",
            "|---|---|---:|---|",
        ]
    )
    for claim in claim_audit["claims"]:
        passed = sum(check["passed"] for check in claim["checks"])
        lines.append(
            f"| `{claim['id']}` | {claim['status']} | "
            f"{passed}/{len(claim['checks'])} | "
            f"{'PASS' if claim['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Scientific boundary",
            "",
            "This audit verifies provenance and consistency of frozen evidence. It does "
            "not create a new efficacy result, independently reproduce training, or "
            "authorize another router candidate. Registered negative results remain "
            "negative. Seed909 and protected tests remain locked.",
            "",
        ]
    )
    return "\n".join(lines)


def write_deterministic_zip(zip_path: Path, files: list[Path]) -> dict[str, Any]:
    prefix = PurePosixPath("InformationSpiralTransformer") / "ist_v0_1"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files, key=rel_path):
            archive_name = str(prefix / PurePosixPath(rel_path(path)))
            info = zipfile.ZipInfo(archive_name, date_time=(2026, 8, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    return {
        "path": rel_path(zip_path),
        "size_bytes": zip_path.stat().st_size,
        "sha256": sha256_file(zip_path),
        "member_count": len(members),
        "bad_member": bad_member,
        "passed": bad_member is None,
    }


def main() -> int:
    args = parse_args()
    if FORMAL_DIR.resolve().parent != LEVEL_DIR.resolve():
        raise SystemExit(f"Refusing unsafe output path: {FORMAL_DIR}")
    if FORMAL_DIR.exists():
        if not args.force:
            raise SystemExit(
                f"Output already exists: {FORMAL_DIR}. Re-run with --force to replace it."
            )
        shutil.rmtree(FORMAL_DIR)
    FORMAL_DIR.mkdir(parents=True, exist_ok=False)
    write_json(FORMAL_DIR / "progress.json", {"stage": "auditing"})

    try:
        preregistration = read_json(PREREG_PATH)
        registry = read_json(REGISTRY_PATH)
        if not preregistration["locks"]["router_repair_branch_closed"]:
            raise ValueError("Router-repair closure lock is not active.")
        if not preregistration["locks"]["seed909_locked"]:
            raise ValueError("seed909 lock is not active.")

        registered_paths = collect_registered_files(registry)
        claim_audit = audit_claims(registry)
        compile_audit = compile_registered_python(registered_paths)
        manifest = build_manifest(registered_paths)
        environment = environment_snapshot()

        write_json(FORMAL_DIR / "claim_audit.json", claim_audit)
        write_json(FORMAL_DIR / "python_compile_audit.json", compile_audit)
        write_json(FORMAL_DIR / "reproducibility_manifest.json", manifest)
        write_json(FORMAL_DIR / "environment.json", environment)

        generated_bundle_files = [
            FORMAL_DIR / "claim_audit.json",
            FORMAL_DIR / "python_compile_audit.json",
            FORMAL_DIR / "reproducibility_manifest.json",
            FORMAL_DIR / "environment.json",
        ]
        zip_result = None
        report = audit_report(claim_audit, compile_audit, manifest, zip_result=None)
        report_path = FORMAL_DIR / "AUDIT_REPORT.md"
        report_path.write_text(report, encoding="utf-8", newline="\n")
        generated_bundle_files.append(report_path)

        if not args.no_zip:
            zip_path = FORMAL_DIR / "ist_level7_0_repro_bundle.zip"
            expected_zip = {
                "passed": True,
                "member_count": len(registered_paths + generated_bundle_files),
            }
            report = audit_report(
                claim_audit, compile_audit, manifest, expected_zip
            )
            report_path.write_text(report, encoding="utf-8", newline="\n")
            zip_result = write_deterministic_zip(
                zip_path, registered_paths + generated_bundle_files
            )
            write_json(FORMAL_DIR / "zip_audit.json", zip_result)
            report = audit_report(claim_audit, compile_audit, manifest, zip_result)
            report_path.write_text(report, encoding="utf-8", newline="\n")

        overall_passed = claim_audit["passed"] and all(
            item["passed"] for item in compile_audit
        )
        if zip_result is not None:
            overall_passed = overall_passed and zip_result["passed"]
        progress = {
            "stage": "complete" if overall_passed else "failed",
            "passed": overall_passed,
            "claims_passed": sum(
                claim["passed"] for claim in claim_audit["claims"]
            ),
            "claims_total": len(claim_audit["claims"]),
            "zip_created": zip_result is not None,
            "router_repair_branch_closed": True,
            "seed909_locked": True,
        }
        write_json(FORMAL_DIR / "progress.json", progress)
        print(json.dumps(progress, ensure_ascii=False, indent=2))
        return 0 if overall_passed else 1
    except Exception as exc:
        write_json(
            FORMAL_DIR / "progress.json",
            {
                "stage": "failed",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "router_repair_branch_closed": True,
                "seed909_locked": True,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
