"""
Tests for Phase 0: Setup and rules verification.
"""
from pathlib import Path
import subprocess
import sys

def test_license_exists():
    license_file = Path("LICENSE")
    assert license_file.exists(), "LICENSE file must exist in repo root"
    content = license_file.read_text(encoding="utf-8")
    assert "MIT License" in content

def test_agents_md_rules():
    agents_file = Path("AGENTS.md")
    assert agents_file.exists(), "AGENTS.md must exist in repo root"
    content = agents_file.read_text(encoding="utf-8")
    for rule_num in range(1, 11):
        assert f"{rule_num}." in content, f"Rule {rule_num} missing from AGENTS.md"

def test_gitignore_covers_secrets_and_venv():
    gitignore = Path(".gitignore")
    assert gitignore.exists()
    content = gitignore.read_text(encoding="utf-8")
    assert ".venv" in content
    assert ".env" in content

def test_env_example_has_no_secrets():
    env_example = Path(".env.example")
    assert env_example.exists()
    content = env_example.read_text(encoding="utf-8")
    assert "your_nebius_api_key_here" in content
    assert "sk-" not in content
    assert "neb-" not in content

def test_secret_scanner_clean():
    res = subprocess.run([sys.executable, "scripts/check_secrets.py"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "[OK]" in res.stdout
