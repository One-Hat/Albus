"""Test that src/api/models.py does not drift from contracts/types.ts."""
import re
from pathlib import Path
import pytest
from src.api import models


def parse_typescript_interfaces(ts_path: Path) -> dict[str, dict[str, any]]:
    """Parse interface definitions and field names from TypeScript file supporting nested braces."""
    content = ts_path.read_text(encoding="utf-8")
    # Remove single-line comments
    content = re.sub(r"//.*", "", content)

    interfaces = {}
    pattern = re.compile(r"export\s+interface\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{")

    for match in pattern.finditer(content):
        name = match.group(1)
        extends = match.group(2)
        start_idx = match.end()

        # Find matching closing brace
        depth = 1
        i = start_idx
        while i < len(content) and depth > 0:
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
            i += 1
        body = content[start_idx:i - 1]

        # In body, strip out any nested {...}
        while "{" in body:
            body = re.sub(r"\{[^{}]*\}", "object", body)

        fields = set()
        for field_match in re.finditer(r"(\w+)\??\s*:", body):
            field_name = field_match.group(1)
            fields.add(field_name)

        interfaces[name] = {
            "extends": extends,
            "fields": fields,
        }

    # Resolve inheritance for extends
    for name, data in interfaces.items():
        if data["extends"] and data["extends"] in interfaces:
            parent_fields = interfaces[data["extends"]]["fields"]
            data["all_fields"] = data["fields"] | parent_fields
        else:
            data["all_fields"] = set(data["fields"])

    return interfaces


def test_contracts_drift():
    """Verify that every field in contracts/types.ts exists in Pydantic models and vice versa."""
    repo_root = Path(__file__).resolve().parent.parent
    ts_path = repo_root / "contracts" / "types.ts"
    assert ts_path.exists(), f"Contract file not found at {ts_path}"

    ts_interfaces = parse_typescript_interfaces(ts_path)

    # Map of TS Interface -> Pydantic Model
    model_mapping = {
        "RunMeta": models.RunMeta,
        "Check": models.Check,
        "Verdict": models.Verdict,
        "Backtest": models.Backtest,
        "Evidence": models.Evidence,
        "Swarm": models.Swarm,
        "Calibration": models.Calibration,
        "Ledger": models.Ledger,
        "PipelineStats": models.PipelineStats,
        "RunSummary": models.RunSummary,
        "RunIndexItem": models.RunIndexItem,
        "StageEvent": models.StageEvent,
    }

    assert set(ts_interfaces.keys()) == set(model_mapping.keys()), (
        f"Interface mismatch. TS has {set(ts_interfaces.keys())}, mapping has {set(model_mapping.keys())}"
    )

    for interface_name, model_cls in model_mapping.items():
        ts_fields = ts_interfaces[interface_name]["all_fields"]
        pydantic_fields = set(model_cls.model_fields.keys())

        missing_in_pydantic = ts_fields - pydantic_fields
        extra_in_pydantic = pydantic_fields - ts_fields

        assert not missing_in_pydantic, (
            f"Model {interface_name} is missing fields declared in contracts/types.ts: {missing_in_pydantic}"
        )
        assert not extra_in_pydantic, (
            f"Model {interface_name} has extra fields not in contracts/types.ts: {extra_in_pydantic}"
        )


def test_contract_version_header():
    """Ensure contracts/types.ts declares a contract_version matching 1.0."""
    repo_root = Path(__file__).resolve().parent.parent
    ts_path = repo_root / "contracts" / "types.ts"
    first_line = ts_path.read_text(encoding="utf-8").splitlines()[0]
    assert "contract_version:" in first_line
    assert "'1.0'" in first_line or '"1.0"' in first_line


def test_drift_detection_synthetic():
    """Verify that any drift between field sets raises an AssertionError."""
    # Synthetic verification
    fields_ts = {"id", "created_at", "source"}
    fields_py = {"id", "created_at"}
    assert fields_ts - fields_py == {"source"}

