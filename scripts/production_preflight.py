"""Fail-closed production-pretraining preflight for RAIQ.

A passed result is a readiness evidence record, not permission to make capability claims.
The command exits non-zero unless every configured production requirement is satisfied.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml


def _gate(passed: bool, **details: Any) -> dict[str, Any]:
    return {"passed": passed, **details}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _artifact_gate(path: Path, artifact_name: str) -> dict[str, Any]:
    payload = _load_json(path)
    return _gate(
        bool(payload and payload.get("passed") is True),
        artifact=artifact_name,
        path=str(path),
        found=payload is not None,
    )


def _production_manifest_gate(path: Path, repo_root: Path, requirements: dict[str, Any]) -> dict[str, Any]:
    payload = _load_json(path)
    if payload is None:
        return _gate(False, error=f"manifest not found or invalid: {path}")

    source_records = payload.get("source_records", [])
    source_record_complete = bool(source_records) and all(
        isinstance(record, dict)
        and all(record.get(field) for field in ("source_id", "provenance", "license_basis", "retrieved_at", "sha256", "allowed_use"))
        for record in source_records
    )
    report_fields = ("quality_report", "deduplication_report", "contamination_policy")
    report_paths = [payload.get(field) for field in report_fields]
    reports_exist = all(
        isinstance(report, str) and (repo_root / report).is_file()
        for report in report_paths
    )
    split_names = {split.get("name") for split in payload.get("splits", []) if isinstance(split, dict)}
    has_required_splits = {"train", "validation", "benchmark"}.issubset(split_names)
    approved = bool(payload.get("approval", {}).get("approved_by")) and bool(payload.get("approval", {}).get("approved_at"))
    checks = {
        "production_eligible": payload.get("production_eligible") is True,
        "license_approved": payload.get("license_status") == "approved",
        "approval_record": approved,
        "source_records": source_record_complete,
        "reports": reports_exist,
        "splits": has_required_splits,
    }
    required_checks = {
        "production_eligible": requirements.get("require_production_eligible_manifest", True),
        "license_approved": requirements.get("require_approved_license_status", True),
        "source_records": requirements.get("require_source_records", True),
        "reports": all(
            requirements.get(key, True)
            for key in ("require_quality_report", "require_deduplication_report", "require_contamination_policy")
        ),
    }
    passed = approved and has_required_splits and all(
        checks[name] for name, required in required_checks.items() if required
    )
    return _gate(passed, path=str(path), checks=checks, corpus_id=payload.get("corpus_id"))


def run_preflight(config_path: Path, evidence_dir: Path, repo_root: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "preflight" not in raw:
        raise ValueError("production configuration must define a preflight mapping")
    requirements = raw["preflight"]
    data = raw.get("data", {})
    training = raw.get("training", {})
    gates: dict[str, dict[str, Any]] = {}

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_memories = [
        round(torch.cuda.get_device_properties(index).total_memory / (1024**3), 2)
        for index in range(gpu_count)
    ]
    required_world_size = int(requirements["required_world_size"])
    min_gpu_memory = float(requirements["min_gpu_memory_gb_per_rank"])
    gates["accelerator"] = _gate(
        (not requirements.get("require_cuda", True) or cuda_available)
        and gpu_count >= required_world_size
        and all(memory >= min_gpu_memory for memory in gpu_memories[:required_world_size]),
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_memories,
        required_world_size=required_world_size,
        required_gpu_memory_gb_per_rank=min_gpu_memory,
    )

    free_disk_gb = shutil.disk_usage(repo_root).free / (1024**3)
    gates["storage"] = _gate(
        free_disk_gb >= float(requirements["min_free_disk_gb"]),
        free_disk_gb=round(free_disk_gb, 2),
        required_free_disk_gb=requirements["min_free_disk_gb"],
    )
    gates["configuration"] = _gate(
        training.get("device") == "cuda" and training.get("dtype") in {"bfloat16", "float16"},
        configured_device=training.get("device"),
        configured_dtype=training.get("dtype"),
    )
    manifest_path = repo_root / str(data.get("corpus_manifest_path", ""))
    gates["production_data"] = _production_manifest_gate(manifest_path, repo_root, requirements)
    gates["distributed_smoke"] = _artifact_gate(evidence_dir / "distributed_smoke.json", "distributed smoke test")
    gates["resume_smoke"] = _artifact_gate(evidence_dir / "resume_smoke.json", "checkpoint-resume smoke test")

    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "purpose": requirements.get("purpose"),
        "config": str(config_path),
        "evidence_dir": str(evidence_dir),
        "passed": passed,
        "gates": gates,
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "world_size_env": os.getenv("WORLD_SIZE", "1"),
        },
        "decision": "production_pretraining_ready" if passed else "not_ready_do_not_start_production_pretraining",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fail-closed RAIQ production-pretraining preflight")
    parser.add_argument("--config", default="configs/200m_production.yaml")
    parser.add_argument("--evidence-dir", default="artifacts/production_evidence")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result = run_preflight(Path(args.config), Path(args.evidence_dir), repo_root)
    output_path = Path(args.output) if args.output else Path(args.evidence_dir) / "production_preflight.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
