#!/usr/bin/env python3
"""
Secret scanner script for pre-commit checks.
Ensures no API keys, tokens, or credentials are committed into git.
"""
import re
import sys
from pathlib import Path

# Common patterns for API keys and tokens
SECRET_PATTERNS = [
    (re.compile(r'(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*["\']([^"\']{16,})["\']'), "API key / Secret key assignment"),
    (re.compile(r'sk-[a-zA-Z0-9]{20,}'), "OpenAI / compatible secret key pattern"),
    (re.compile(r'neb-[a-zA-Z0-9]{20,}'), "Nebius secret key pattern"),
    (re.compile(r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----'), "Private key header"),
    (re.compile(r'(?i)ghp_[a-zA-Z0-9]{36}'), "GitHub personal access token"),
]

# Files/extensions that should be ignored or are safe
IGNORED_PATTERNS = [
    re.compile(r'\.git/'),
    re.compile(r'\.venv/'),
    re.compile(r'__pycache__/'),
    re.compile(r'\.cache/'),
    re.compile(r'\.env\.example$'),
]

def is_ignored(path: Path) -> bool:
    path_str = str(path).replace('\\', '/')
    return any(p.search(path_str) for p in IGNORED_PATTERNS)

def scan_file(file_path: Path) -> list[str]:
    violations = []
    if is_ignored(file_path):
        return violations

    if file_path.name == '.env':
        violations.append(f"{file_path}: .env files must never be committed!")
        return violations

    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        return [f"Could not read {file_path}: {e}"]

    for line_no, line in enumerate(content.splitlines(), start=1):
        for pattern, desc in SECRET_PATTERNS:
            if pattern.search(line):
                # Don't false-positive on placeholder text or comments mentioning variable names
                if "your_" in line.lower() or "example" in line.lower() or "placeholder" in line.lower():
                    continue
                violations.append(f"{file_path}:{line_no} - Detected potential {desc}: {line.strip()[:60]}...")
    return violations

def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    violations = []

    # If specific files passed via args (e.g. from pre-commit), check those; otherwise check all git files
    files_to_check = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else list(repo_root.rglob("*"))

    for p in files_to_check:
        if p.is_file() and not is_ignored(p):
            violations.extend(scan_file(p))

    if violations:
        print("[ERROR] Secret scanner found potential secrets or forbidden files:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    print("[OK] Secret scan passed: No secrets detected.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
