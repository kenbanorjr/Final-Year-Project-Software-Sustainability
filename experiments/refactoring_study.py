"""
Controlled Pre/Post LLM-Assisted Refactoring Study
===================================================

This script conducts a controlled study to evaluate whether LLM-assisted
refactoring can reduce maintainability risks in production code without
changing observable functionality.

Study Protocol:
1. Select small repositories from dataset
2. Record baseline static-analysis metrics (Sonar) at fixed commit
3. Apply LLM-guided refactoring under constraints:
   - No functionality removal
   - No rule suppression
   - No configuration changes affecting analysis
4. Validate functionality via build + tests
5. Re-run Sonar analysis post-refactoring
6. Compare maintainability metrics and log everything

Author: Final Year Project
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import hashlib
import shutil
import shlex
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Iterable
import difflib
import re

import pandas as pd
from openai import OpenAI

from pipeline.configs import config
from pipeline.sonar_runner import run_sonar_scan, fetch_file_metrics, project_key_for_repo
from pipeline.utils import detect_language, extract_code_from_response

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Study parameters
MAX_FILES_PER_REPO = 10          # Limit files to refactor per repo
MAX_NCLOC_PER_FILE = 300         # Only refactor small-medium files
MIN_CODE_SMELLS = 2              # Only files with some smells worth fixing
TARGET_LANGUAGES = {"java", "javascript", "python", "typescript"}
MAX_ATTEMPTS_PER_FILE = 3
FAST_VERIFY_TIMEOUT_SEC = 60
VERIFY_COMMAND_TIMEOUT_SEC = 600
VERIFY_OUTPUT_SNIPPET_CHARS = 1200
IMPORT_CONTEXT_MAX_LINES = 60
SYMBOL_CONTEXT_MAX_ITEMS = 20
CALLSITE_CONTEXT_MAX_LINES = 6

# LLM settings for refactoring
REFACTOR_MODEL = os.environ.get("REFACTOR_MODEL", config.LLM_MODEL)
REFACTOR_TEMPERATURE = 0.2       # Low temp for deterministic refactoring
REFACTOR_MAX_TOKENS = 8192       # Must accommodate files up to MAX_NCLOC_PER_FILE lines
TEST_CONTEXT_MAX_CHARS = 3000   # Max chars of test file content to include in prompt

# Directories
STUDY_DIR = config.RESULTS_DIR / "refactoring_study"
COMPARE_DIR = STUDY_DIR / "compare"

VERIFY_PROFILE_FILENAMES = (".refactor_verify.yaml", "verify.yaml")

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class RefactoringPrompt:
    """Stores the prompt used for refactoring."""
    system_prompt: str
    user_prompt: str
    model: str
    temperature: float
    max_tokens: int
    prompt_hash: str = ""
    
    def __post_init__(self):
        content = f"{self.system_prompt}{self.user_prompt}"
        self.prompt_hash = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class RefactoringResult:
    """Stores the result of a refactoring attempt."""
    file_path: str
    repo: str
    success: bool
    original_code: str
    refactored_code: Optional[str]
    diff: Optional[str]
    error_message: Optional[str]
    llm_response_raw: Optional[str]
    latency_ms: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class TestResult:
    """Stores build/test validation result."""
    repo: str
    build_success: bool
    test_success: bool
    build_output: str
    test_output: str
    build_command: str
    test_command: str
    duration_sec: float


@dataclass
class CommandRun:
    """Single verification command execution."""
    phase: str
    command: str
    exit_code: int
    success: bool
    duration_sec: float
    stdout_tail: str
    stderr_tail: str


@dataclass
class VerificationResult:
    """Result of fast checks + repo verification commands."""
    passed: bool
    commands: List[CommandRun]
    build_success: bool
    test_success: bool
    test_status: str  # passed|failed|missing
    failure_type: Optional[str] = None
    failure_command: Optional[str] = None
    failure_exit_code: Optional[int] = None
    failure_snippet: Optional[str] = None


@dataclass
class VerifyProfile:
    """Per-repo verification profile loaded from YAML."""
    source_path: Optional[str]
    install: List[str]
    build: List[str]
    test: List[str]


@dataclass  
class MetricComparison:
    """Stores before/after metric comparison for a file."""
    file_path: str
    repo: str
    
    # Before metrics
    pre_code_smells: Optional[int]
    pre_cognitive_complexity: Optional[float]
    pre_sqale_index: Optional[float]
    pre_duplicated_lines_density: Optional[float]
    pre_ncloc: Optional[int]
    
    # After metrics
    post_code_smells: Optional[int]
    post_cognitive_complexity: Optional[float]
    post_sqale_index: Optional[float]
    post_duplicated_lines_density: Optional[float]
    post_ncloc: Optional[int]
    
    # Deltas
    delta_code_smells: Optional[int] = None
    delta_cognitive_complexity: Optional[float] = None
    delta_sqale_index: Optional[float] = None
    delta_duplicated_lines_density: Optional[float] = None
    delta_ncloc: Optional[int] = None
    
    def __post_init__(self):
        # Calculate deltas (negative = improvement)
        if self.pre_code_smells is not None and self.post_code_smells is not None:
            self.delta_code_smells = self.post_code_smells - self.pre_code_smells
        if self.pre_cognitive_complexity is not None and self.post_cognitive_complexity is not None:
            self.delta_cognitive_complexity = self.post_cognitive_complexity - self.pre_cognitive_complexity
        if self.pre_sqale_index is not None and self.post_sqale_index is not None:
            self.delta_sqale_index = self.post_sqale_index - self.pre_sqale_index
        if self.pre_duplicated_lines_density is not None and self.post_duplicated_lines_density is not None:
            self.delta_duplicated_lines_density = self.post_duplicated_lines_density - self.pre_duplicated_lines_density
        if self.pre_ncloc is not None and self.post_ncloc is not None:
            self.delta_ncloc = self.post_ncloc - self.pre_ncloc


@dataclass
class StudySession:
    """Tracks an entire study session."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    repos_processed: List[str] = field(default_factory=list)
    files_refactored: int = 0
    files_failed: int = 0
    build_failures: int = 0
    test_failures: int = 0
    total_smells_reduced: int = 0
    total_debt_reduced_min: float = 0.0
    model_used: str = ""
    prompt_version: str = ""
    repo_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RegressionReport:
    """Baseline-vs-post regression analysis."""
    new_failures: List[Dict[str, Any]]
    same_failures: List[Dict[str, Any]]
    fixed_failures: List[Dict[str, Any]]
    safe_non_regression: bool
    baseline_command_count: int = 0
    post_command_count: int = 0


# ==============================================================================
# REFACTORING PROMPTS
# ==============================================================================

REFACTORING_SYSTEM_PROMPT = """You are an expert code refactoring assistant. Your task is to improve code maintainability while preserving exact functionality.

## STRICT CONSTRAINTS (MUST FOLLOW)
1. **NO functionality changes** - The refactored code must produce identical outputs for all inputs
2. **NO rule suppression** - Do not add @SuppressWarnings, # noqa, // eslint-disable, or similar
3. **NO configuration changes** - Do not modify build files, configs, or analysis settings
4. **PRESERVE all public APIs** - Method signatures, class names, and exports must stay the same
5. **PRESERVE all tests** - Do not modify test files

## REFACTORING GOALS (in priority order)
1. Reduce cognitive complexity (extract methods, simplify conditionals)
2. Eliminate code smells (long methods, deep nesting, duplicate code)
3. Improve naming (variables, methods, classes)
4. Add/improve documentation (but don't over-document trivial code)
5. Apply language-specific best practices

## OUTPUT FORMAT
Return ONLY the refactored code, no explanations. The code must be complete and compilable.
If the code cannot be improved without violating constraints, return it unchanged.

```{language}
<refactored code here>
```"""

REFACTORING_USER_PROMPT = """Refactor the following {language} code to improve maintainability.

Current metrics from SonarQube:
- Code Smells: {code_smells}
- Cognitive Complexity: {cognitive_complexity}
- Technical Debt: {sqale_index} minutes
- Lines of Code: {ncloc}
- Duplicated Lines: {duplicated_lines_density}%
- Rule/issue hints: {issue_hints}

Relative file path: {file_path}

Context pack:
- Import section:
{import_context}
- Public symbols:
{public_symbols}
- Type/base-class signatures:
{type_context}
- Nearby call sites:
{call_sites}
- Relevant test code (DO NOT modify test expectations):
{test_context}

```{language}
{code}
```

Retry feedback (if any):
{retry_feedback}

Return the refactored code that reduces these issues while preserving functionality.
If retry feedback is present, fix those failures while preserving API/behavior.
Do NOT disable tests, do NOT suppress rules, and do NOT change build configuration."""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def setup_directories():
    """Create top-level study directories. Session/repo dirs are created on demand."""
    for dir_path in (STUDY_DIR, COMPARE_DIR):
        dir_path.mkdir(parents=True, exist_ok=True)


def _safe_snippet(text: str, max_chars: int = VERIFY_OUTPUT_SNIPPET_CHARS) -> str:
    if not text:
        return ""
    cleaned = text.replace("\x00", "")
    return cleaned[-max_chars:]


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a JSON record to a JSONL file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_simple_yaml_list_block(lines: List[str], start_idx: int) -> tuple[list[str], int]:
    """Parse a simple YAML list block under a key (minimal parser for command profiles)."""
    values: list[str] = []
    idx = start_idx
    while idx < len(lines):
        raw = lines[idx]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            idx += 1
            continue
        if not raw.startswith(" ") and not raw.startswith("\t") and ":" in stripped and not stripped.startswith("-"):
            break
        if stripped.startswith("-"):
            item = stripped[1:].strip().strip("'").strip('"')
            if item:
                values.append(item)
        idx += 1
    return values, idx


def load_verify_profile(repo_path: Path, preferred_profile: Optional[str] = None) -> VerifyProfile:
    """
    Load verification profile from .refactor_verify.yaml or verify.yaml.
    Supports a minimal list-of-commands schema:
      install: [ ... ] or multiline list
      build:   [ ... ] or multiline list
      test:    [ ... ] or multiline list
    """
    profile_path: Optional[Path] = None
    candidates = (preferred_profile,) if preferred_profile else VERIFY_PROFILE_FILENAMES
    for candidate in candidates:
        if not candidate:
            continue
        path = repo_path / candidate
        if path.exists():
            profile_path = path
            break

    if not profile_path:
        return VerifyProfile(source_path=None, install=[], build=[], test=[])

    text = profile_path.read_text(encoding="utf-8", errors="replace")
    install: list[str] = []
    build: list[str] = []
    test: list[str] = []
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith("#"):
            idx += 1
            continue
        key = None
        if stripped.startswith("install:"):
            key = "install"
        elif stripped.startswith("build:"):
            key = "build"
        elif stripped.startswith("test:"):
            key = "test"
        if not key:
            idx += 1
            continue

        remainder = stripped.split(":", 1)[1].strip()
        if remainder.startswith("[") and remainder.endswith("]"):
            items = [part.strip().strip("'").strip('"') for part in remainder[1:-1].split(",")]
            parsed = [item for item in items if item]
            if key == "install":
                install.extend(parsed)
            elif key == "build":
                build.extend(parsed)
            else:
                test.extend(parsed)
            idx += 1
            continue

        parsed, idx = _parse_simple_yaml_list_block(lines, idx + 1)
        if key == "install":
            install.extend(parsed)
        elif key == "build":
            build.extend(parsed)
        else:
            test.extend(parsed)

    return VerifyProfile(
        source_path=str(profile_path),
        install=install,
        build=build,
        test=test,
    )


def _extract_import_context(code: str, language: str) -> str:
    lines = code.splitlines()
    collected: list[str] = []
    for line in lines[:IMPORT_CONTEXT_MAX_LINES]:
        stripped = line.strip()
        if not stripped:
            if collected:
                break
            continue
        if language in {"python"} and (stripped.startswith("import ") or stripped.startswith("from ")):
            collected.append(line)
            continue
        if language in {"javascript", "typescript"} and (
            stripped.startswith("import ") or stripped.startswith("const ") and "require(" in stripped
        ):
            collected.append(line)
            continue
        if language in {"java"} and (stripped.startswith("package ") or stripped.startswith("import ")):
            collected.append(line)
            continue
        if collected:
            break
    return "\n".join(collected[:IMPORT_CONTEXT_MAX_LINES]) or "(none)"


def _extract_public_symbols(code: str, language: str) -> list[str]:
    symbols: list[str] = []
    if language == "python":
        patterns = [r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\b"]
    elif language in {"javascript", "typescript"}:
        patterns = [
            r"^export\s+function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            r"^function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        ]
    elif language == "java":
        patterns = [r"^\s*public\s+(?:class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b", r"^\s*public\s+.*?\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("]
    else:
        patterns = [r"^func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("]

    for line in code.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                symbols.append(match.group(1))
                break
        if len(symbols) >= SYMBOL_CONTEXT_MAX_ITEMS:
            break
    return sorted(set(symbols))[:SYMBOL_CONTEXT_MAX_ITEMS]


def _extract_call_sites(repo_path: Path, file_rel_path: str, symbols: list[str]) -> str:
    if not symbols:
        return "(none)"
    primary = symbols[0]
    if not shutil.which("rg"):
        return "(ripgrep not available)"
    try:
        cmd = [
            "rg",
            "-n",
            "--no-heading",
            "--max-count",
            str(CALLSITE_CONTEXT_MAX_LINES),
            rf"\b{re.escape(primary)}\b",
            str(repo_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FAST_VERIFY_TIMEOUT_SEC,
            check=False,
        )
        lines = [line for line in result.stdout.splitlines() if file_rel_path.replace("\\", "/") not in line]
        return "\n".join(lines[:CALLSITE_CONTEXT_MAX_LINES]) or "(none)"
    except Exception:
        return "(unavailable)"


def _find_test_context(repo_path: Path, file_rel_path: str, symbols: list[str], language: str) -> str:
    """Find and return relevant test file excerpts that exercise the target file.

    Strategy: look for test files that import or reference public symbols from
    the target file.  We cap the returned text at TEST_CONTEXT_MAX_CHARS to
    keep the prompt manageable.
    """
    if not symbols:
        return "(no public symbols to search for)"

    # Common test directory patterns per language
    test_globs: list[str] = []
    stem = Path(file_rel_path).stem

    if language == "python":
        test_globs = [f"**/test_{stem}.py", f"**/{stem}_test.py", f"**/tests/test_{stem}.py", f"**/test/**/{stem}*.py"]
    elif language == "java":
        test_globs = [f"**/{stem}Test.java", f"**/Test{stem}.java", f"**/test/**/{stem}*.java"]
    elif language in {"javascript", "typescript"}:
        ext = "ts" if language == "typescript" else "js"
        test_globs = [f"**/{stem}.test.{ext}", f"**/{stem}.spec.{ext}", f"**/test/**/{stem}*.{ext}", f"**/__tests__/{stem}*.{ext}"]

    # Collect candidate test files
    candidates: list[Path] = []
    for glob_pat in test_globs:
        candidates.extend(repo_path.glob(glob_pat))
    candidates = sorted(set(candidates))[:5]  # cap number of test files

    if not candidates:
        # Fallback: use ripgrep to find files that reference the first symbol in test dirs
        if shutil.which("rg"):
            try:
                cmd = ["rg", "-l", "--max-count", "1", rf"\b{re.escape(symbols[0])}\b", str(repo_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=FAST_VERIFY_TIMEOUT_SEC, check=False)
                for line in result.stdout.splitlines()[:5]:
                    p = Path(line.strip())
                    lower = str(p).lower()
                    if ("test" in lower or "spec" in lower) and p.exists():
                        candidates.append(p)
            except Exception:
                pass

    if not candidates:
        return "(no test files found)"

    # Extract relevant snippets
    snippets: list[str] = []
    chars_remaining = TEST_CONTEXT_MAX_CHARS
    for test_file in candidates:
        if chars_remaining <= 0:
            break
        try:
            content = test_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # Only include if it actually references at least one symbol
        if not any(sym in content for sym in symbols[:5]):
            continue
        rel = test_file.relative_to(repo_path) if test_file.is_relative_to(repo_path) else test_file
        header = f"--- {rel} ---"
        # Truncate if needed
        excerpt = content[:chars_remaining]
        snippets.append(f"{header}\n{excerpt}")
        chars_remaining -= len(header) + len(excerpt) + 2

    return "\n".join(snippets) if snippets else "(no relevant test files found)"


def _extract_type_context(code: str, language: str) -> str:
    """Extract type signatures, base classes, decorators, and interface info.

    This gives the LLM visibility into contracts that must be preserved even
    though they might not appear in the simple public-symbol list.
    """
    lines = code.splitlines()
    context_lines: list[str] = []

    if language == "python":
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Decorators
            if stripped.startswith("@"):
                context_lines.append(stripped)
                continue
            # Class with base classes
            m = re.match(r"^class\s+([A-Za-z_]\w*)\s*\(([^)]+)\)\s*:", stripped)
            if m:
                context_lines.append(f"class {m.group(1)}({m.group(2)})")
                continue
            # Function signatures with type annotations
            m = re.match(r"^def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(?:->\s*(.+?))?\s*:", stripped)
            if m:
                sig = f"def {m.group(1)}({m.group(2)})"
                if m.group(3):
                    sig += f" -> {m.group(3)}"
                context_lines.append(sig)
                continue
            # Protocol / TypeVar / type aliases
            if any(kw in stripped for kw in ("TypeVar", "Protocol", "TypeAlias", "NewType")):
                context_lines.append(stripped)

    elif language == "java":
        for line in lines:
            stripped = line.strip()
            # Annotations
            if stripped.startswith("@") and not stripped.startswith("@Test"):
                context_lines.append(stripped)
                continue
            # Class/interface/enum declarations with extends/implements
            m = re.match(
                r"(public\s+(?:abstract\s+)?(?:class|interface|enum)\s+[A-Za-z_]\w*"
                r"(?:\s*<[^>]+>)?"
                r"(?:\s+extends\s+[A-Za-z_][\w.<>,\s]*?)?"
                r"(?:\s+implements\s+[A-Za-z_][\w.<>,\s]*?)?)\s*\{?",
                stripped,
            )
            if m:
                context_lines.append(m.group(1).strip())
                continue
            # Public method signatures (return type + generics)
            m = re.match(r"(public\s+(?:static\s+)?(?:final\s+)?(?:[\w<>\[\],\s]+?)\s+[A-Za-z_]\w*\s*\([^)]*\))", stripped)
            if m:
                context_lines.append(m.group(1))

    elif language in {"javascript", "typescript"}:
        for line in lines:
            stripped = line.strip()
            # TypeScript interfaces and type aliases
            m = re.match(r"(?:export\s+)?(?:interface|type)\s+([A-Za-z_]\w*(?:<[^>]+>)?)\s*(?:extends\s+([^{]+))?\s*[={]", stripped)
            if m:
                sig = f"interface/type {m.group(1)}"
                if m.group(2):
                    sig += f" extends {m.group(2).strip()}"
                context_lines.append(sig)
                continue
            # Class declarations with extends/implements
            m = re.match(
                r"(?:export\s+)?class\s+([A-Za-z_]\w*)"
                r"(?:\s+extends\s+([A-Za-z_][\w.]*))?",
                stripped,
            )
            if m:
                sig = f"class {m.group(1)}"
                if m.group(2):
                    sig += f" extends {m.group(2)}"
                context_lines.append(sig)
                continue
            # Decorators (experimental / Angular / NestJS)
            if stripped.startswith("@"):
                context_lines.append(stripped)

    # Cap output
    joined = "\n".join(context_lines[:30])
    return joined if joined else "(none)"


def _issue_hints_from_metrics(metrics: Dict[str, Any]) -> str:
    hints: list[str] = []
    for key, value in metrics.items():
        low = key.lower()
        if ("issue" in low or "rule" in low) and value not in (None, "", 0, 0.0):
            hints.append(f"{key}={value}")
    return "; ".join(hints[:8]) if hints else "not available"


def _extract_js_exports(code: str) -> set[str]:
    """Extract a conservative set of exported symbol names for JS/TS."""
    exports: set[str] = set()
    for line in code.splitlines():
        m = re.match(r"\s*export\s+(?:default\s+)?(?:class|function|const|let|var)\s+([A-Za-z_$][\w$]*)", line)
        if m:
            exports.add(m.group(1))
            continue
        m = re.match(r"\s*export\s+default\s+([A-Za-z_$][\w$]*)\s*;?\s*$", line)
        if m:
            exports.add(m.group(1))
            continue
        m = re.match(r"\s*export\s*\{([^}]*)\}", line)
        if m:
            for chunk in m.group(1).split(","):
                entry = chunk.strip()
                if not entry:
                    continue
                if " as " in entry:
                    left, _right = [part.strip() for part in entry.split(" as ", 1)]
                    if left:
                        exports.add(left)
                else:
                    exports.add(entry)
    for m in re.finditer(r"module\.exports(?:\.[A-Za-z_$][\w$]*)?\s*=\s*([A-Za-z_$][\w$]*)", code):
        exports.add(m.group(1))
    return exports


def _extract_js_class_static_methods(code: str) -> Dict[str, set[str]]:
    """Extract static methods by class for JS/TS source."""
    lines = code.splitlines()
    out: Dict[str, set[str]] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        class_match = re.match(r"\s*(?:export\s+default\s+|export\s+)?class\s+([A-Za-z_$][\w$]*)\b", line)
        if not class_match:
            idx += 1
            continue
        class_name = class_match.group(1)
        brace_depth = line.count("{") - line.count("}")
        idx += 1
        methods: set[str] = set()
        while idx < len(lines):
            current = lines[idx]
            static_match = re.match(r"\s*static\s+([A-Za-z_$][\w$]*)\s*\(", current)
            if static_match:
                methods.add(static_match.group(1))
            brace_depth += current.count("{") - current.count("}")
            if brace_depth <= 0:
                break
            idx += 1
        out[class_name] = methods
        idx += 1
    return out


def validate_refactor_invariants(original_code: str, refactored_code: str, language: str) -> Optional[str]:
    """
    Validate API/symbol invariants to reject risky refactors early.

    Enforced checks per language:
    - JS/TS: exported symbols and class static methods must remain present.
    - Python: top-level function/class names must remain present.
    - Java: public class/interface/enum and public method signatures must remain present.
    """
    # ---- JS / TS checks ----
    if language in {"javascript", "typescript"}:
        orig_exports = _extract_js_exports(original_code)
        new_exports = _extract_js_exports(refactored_code)
        missing_exports = sorted(orig_exports - new_exports)
        if missing_exports:
            return f"Invariant failed: missing exported symbols: {', '.join(missing_exports[:12])}"

        orig_static = _extract_js_class_static_methods(original_code)
        new_static = _extract_js_class_static_methods(refactored_code)
        missing_static: list[str] = []
        for cls, methods in orig_static.items():
            if not methods:
                continue
            current = new_static.get(cls, set())
            for method in sorted(methods - current):
                missing_static.append(f"{cls}.{method}")
        if missing_static:
            return f"Invariant failed: missing class static methods: {', '.join(missing_static[:12])}"
        return None

    # ---- Python checks ----
    if language == "python":
        orig_symbols = _extract_python_public_signatures(original_code)
        new_symbols = _extract_python_public_signatures(refactored_code)
        missing = sorted(set(orig_symbols) - set(new_symbols))
        if missing:
            return f"Invariant failed: missing Python public symbols: {', '.join(missing[:12])}"
        return None

    # ---- Java checks ----
    if language == "java":
        orig_sigs = _extract_java_public_signatures(original_code)
        new_sigs = _extract_java_public_signatures(refactored_code)
        missing = sorted(set(orig_sigs) - set(new_sigs))
        if missing:
            return f"Invariant failed: missing Java public signatures: {', '.join(missing[:12])}"
        return None

    return None


def _extract_python_public_signatures(code: str) -> list[str]:
    """Extract top-level public function and class names (with param lists) from Python code."""
    signatures: list[str] = []
    for line in code.splitlines():
        # Top-level def (no leading whitespace)
        m = re.match(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))", line)
        if m:
            signatures.append(f"def {m.group(1)}{m.group(2)}")
            continue
        # Top-level class
        m = re.match(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
        if m:
            signatures.append(f"class {m.group(1)}")
            continue
    return signatures


def _extract_java_public_signatures(code: str) -> list[str]:
    """Extract public class/interface/method signatures from Java code."""
    signatures: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        # Public class / interface / enum
        m = re.match(r"public\s+(?:abstract\s+)?(?:class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b", stripped)
        if m:
            signatures.append(m.group(0).split("{")[0].strip())
            continue
        # Public method
        m = re.match(r"public\s+(?:static\s+)?(?:final\s+)?(?:[\w<>\[\],\s]+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", stripped)
        if m:
            signatures.append(m.group(0).split("{")[0].strip())
    return signatures


def failure_signature(failure_type: str, command: Optional[str], error_snippet: Optional[str]) -> str:
    """Stable failure signature for dedup-stop in retry loops."""
    payload = f"{failure_type}|{command or ''}|{error_snippet or ''}"
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def get_git_commit(repo_path: Path) -> str:
    """Get the current git commit hash for a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def duplicate_repo(source_repo: Path, dest_dir: Path) -> Path:
    """
    Create a working copy of a repository for refactoring.

    Args:
        source_repo: Original repo path.
        dest_dir: Destination directory for the copy.

    Returns the path to the duplicated repo.
    """
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    shutil.copytree(
        source_repo,
        dest_dir,
        ignore=shutil.ignore_patterns('.git', '__pycache__', 'node_modules', '.pytest_cache', 'target', 'build'),
    )

    return dest_dir


def compute_diff(original: str, refactored: str, file_path: str) -> str:
    """Compute a unified diff between original and refactored code."""
    original_lines = original.splitlines(keepends=True)
    refactored_lines = refactored.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines, 
        refactored_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm=""
    )
    return "".join(diff)


def _artifact_name(file_path: str) -> str:
    """Build a stable filename for before/after file snapshots."""
    normalized = file_path.replace("\\", "/").strip("/")
    return normalized.replace("/", "_")


def save_before_after(
    before_dir: Path,
    after_dir: Path,
    file_path: str,
    original_code: str,
    refactored_code: str,
) -> tuple[Path, Path]:
    """Save only the changed file content before and after refactoring."""
    name = _artifact_name(file_path)
    before_path = before_dir / name
    after_path = after_dir / name
    before_path.write_text(original_code, encoding="utf-8")
    after_path.write_text(refactored_code, encoding="utf-8")
    return before_path, after_path


def create_working_copy(source_repo: Path, dest_dir: Path) -> Path:
    """
    Create a persisted working copy for refactoring.
    """
    return duplicate_repo(source_repo, dest_dir)


def repo_url_for_name(repo_name: str) -> str:
    for url in config.ALL_REPOSITORIES:
        if config.repo_dir_name(url) == repo_name:
            return url
    return ""


def init_git_snapshot(repo_path: Path, baseline_sha: str) -> None:
    """Initialize git repo in working copy and create baseline commit."""
    subprocess.run(["git", "init"], cwd=repo_path, check=False, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "refactor-bot"], cwd=repo_path, check=False, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "refactor-bot@example.local"], cwd=repo_path, check=False, capture_output=True, text=True)
    subprocess.run(["git", "add", "-A"], cwd=repo_path, check=False, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", f"baseline snapshot (source={baseline_sha})"],
        cwd=repo_path,
        check=False,
        capture_output=True,
        text=True,
    )


def commit_accepted_change(repo_path: Path, file_rel_path: str, attempt: int) -> None:
    """Commit accepted file change in working copy git snapshot."""
    subprocess.run(["git", "add", file_rel_path], cwd=repo_path, check=False, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", f"accept refactor: {file_rel_path} (attempt {attempt})"],
        cwd=repo_path,
        check=False,
        capture_output=True,
        text=True,
    )


def diff_stats(diff_text: str) -> Dict[str, int]:
    """Return lightweight unified-diff stats."""
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return {"added_lines": added, "removed_lines": removed}


# ==============================================================================
# CANDIDATE SELECTION
# ==============================================================================

def select_refactoring_candidates(
    sonar_df: pd.DataFrame,
    repo: str,
    max_files: int = MAX_FILES_PER_REPO
) -> pd.DataFrame:
    """
    Select files that are good candidates for refactoring study.
    
    Criteria:
    - Small to medium size (NCLOC <= threshold)
    - Has code smells worth fixing
    - Supported language
    - Not a test file
    """
    # Filter to this repo
    repo_df = sonar_df[sonar_df["repo"] == repo].copy()
    
    if repo_df.empty:
        return pd.DataFrame()
    
    # Filter criteria
    candidates = repo_df[
        (repo_df["sonar_ncloc"] <= MAX_NCLOC_PER_FILE) &
        (repo_df["sonar_code_smells"] >= MIN_CODE_SMELLS)
    ].copy()
    
    # Exclude test files
    test_patterns = ["test", "spec", "__test__", "_test.", ".test.", "tests/"]
    for pattern in test_patterns:
        candidates = candidates[~candidates["file_path"].str.lower().str.contains(pattern)]
    
    # Filter by language
    candidates["language"] = candidates["file_path"].apply(
        lambda p: detect_language(Path(p))
    )
    candidates = candidates[candidates["language"].isin(TARGET_LANGUAGES)]
    
    # Sort by code smells (prioritize files with more issues)
    candidates = candidates.sort_values("sonar_code_smells", ascending=False)
    
    # Take top N
    return candidates.head(max_files)


# ==============================================================================
# LLM REFACTORING
# ==============================================================================

def create_refactoring_prompt(
    code: str,
    file_path: str,
    language: str,
    metrics: Dict[str, Any],
    import_context: str,
    public_symbols: list[str],
    call_sites: str,
    test_context: str = "(none)",
    type_context: str = "(none)",
    retry_feedback: Optional[str] = None,
) -> RefactoringPrompt:
    """Create the refactoring prompt with file context."""
    system = REFACTORING_SYSTEM_PROMPT.format(language=language)
    retry_text = retry_feedback or "none"
    user = REFACTORING_USER_PROMPT.format(
        language=language,
        code=code,
        file_path=file_path,
        code_smells=metrics.get("code_smells", "N/A"),
        cognitive_complexity=metrics.get("cognitive_complexity", "N/A"),
        sqale_index=metrics.get("sqale_index", "N/A"),
        ncloc=metrics.get("ncloc", "N/A"),
        duplicated_lines_density=metrics.get("duplicated_lines_density", 0),
        issue_hints=_issue_hints_from_metrics(metrics),
        import_context=import_context,
        public_symbols=", ".join(public_symbols) if public_symbols else "(none)",
        type_context=type_context,
        call_sites=call_sites,
        test_context=test_context,
        retry_feedback=retry_text,
    )
    
    return RefactoringPrompt(
        system_prompt=system,
        user_prompt=user,
        model=REFACTOR_MODEL,
        temperature=REFACTOR_TEMPERATURE,
        max_tokens=REFACTOR_MAX_TOKENS,
    )


def _detect_line_ending(raw_bytes: bytes) -> str:
    """Detect the dominant line ending in raw file bytes. Returns '\\r\\n' or '\\n'."""
    crlf_count = raw_bytes.count(b"\r\n")
    lf_count = raw_bytes.count(b"\n") - crlf_count  # standalone LFs only
    return "\r\n" if crlf_count > lf_count else "\n"


def _normalize_line_endings(text: str, target_eol: str) -> str:
    """Normalize all line endings in *text* to *target_eol*."""
    # First unify to \n, then replace if needed
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    if target_eol == "\r\n":
        return unified.replace("\n", "\r\n")
    return unified


def apply_llm_refactoring(
    file_path: Path,
    repo_path: Path,
    repo: str,
    file_rel_path: str,
    metrics: Dict[str, Any],
    client: OpenAI,
    attempt: int = 1,
    retry_feedback: Optional[str] = None,
    previous_diff: Optional[str] = None,
) -> RefactoringResult:
    """Apply LLM-guided refactoring to a single file."""
    
    # Read original code – detect line endings from raw bytes first
    try:
        raw_bytes = file_path.read_bytes()
        original_eol = _detect_line_ending(raw_bytes)
        original_code = raw_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        return RefactoringResult(
            file_path=str(file_path),
            repo=repo,
            success=False,
            original_code="",
            refactored_code=None,
            diff=None,
            error_message=f"Failed to read file: {e}",
            llm_response_raw=None,
            latency_ms=0,
        )
    
    language = detect_language(file_path)
    import_context = _extract_import_context(original_code, language)
    public_symbols = _extract_public_symbols(original_code, language)
    call_sites = _extract_call_sites(repo_path, file_rel_path, public_symbols)
    test_context = _find_test_context(repo_path, file_rel_path, public_symbols, language)
    type_context = _extract_type_context(original_code, language)
    retry_context_parts: list[str] = []
    if attempt > 1:
        retry_context_parts.append(f"Attempt {attempt}.")
        if previous_diff:
            retry_context_parts.append("Previous diff (truncated):\n" + _safe_snippet(previous_diff, 1500))
        if retry_feedback:
            retry_context_parts.append("Verification failure feedback:\n" + _safe_snippet(retry_feedback, 1500))
    prompt = create_refactoring_prompt(
        original_code,
        file_rel_path,
        language,
        metrics,
        import_context=import_context,
        public_symbols=public_symbols,
        call_sites=call_sites,
        test_context=test_context,
        type_context=type_context,
        retry_feedback="\n\n".join(retry_context_parts) if retry_context_parts else None,
    )
    
    # Call LLM
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=prompt.model,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            temperature=prompt.temperature,
            max_tokens=prompt.max_tokens,
        )
        raw_response = response.choices[0].message.content or ""
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        
    except Exception as e:
        return RefactoringResult(
            file_path=str(file_path),
            repo=repo,
            success=False,
            original_code=original_code,
            refactored_code=None,
            diff=None,
            error_message=f"LLM call failed: {e}",
            llm_response_raw=None,
            latency_ms=int((time.perf_counter() - start_time) * 1000),
        )
    
    # Extract code from response
    refactored_code = extract_code_from_response(raw_response, language)
    
    if not refactored_code:
        return RefactoringResult(
            file_path=str(file_path),
            repo=repo,
            success=False,
            original_code=original_code,
            refactored_code=None,
            diff=None,
            error_message="Failed to extract code from LLM response",
            llm_response_raw=raw_response,
            latency_ms=latency_ms,
        )

    # Preserve original line endings
    refactored_code = _normalize_line_endings(refactored_code, original_eol)
    
    # Check for forbidden patterns (rule suppression)
    forbidden_patterns = [
        r"@SuppressWarnings",
        r"# noqa",
        r"// eslint-disable",
        r"// @ts-ignore",
        r"#pragma warning disable",
        r"// noinspection",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, refactored_code, re.IGNORECASE):
            return RefactoringResult(
                file_path=str(file_path),
                repo=repo,
                success=False,
                original_code=original_code,
                refactored_code=refactored_code,
                diff=None,
                error_message=f"Refactored code contains forbidden pattern: {pattern}",
                llm_response_raw=raw_response,
                latency_ms=latency_ms,
            )

    invariant_error = validate_refactor_invariants(original_code, refactored_code, language)
    if invariant_error:
        return RefactoringResult(
            file_path=str(file_path),
            repo=repo,
            success=False,
            original_code=original_code,
            refactored_code=refactored_code,
            diff=None,
            error_message=invariant_error,
            llm_response_raw=raw_response,
            latency_ms=latency_ms,
        )
    
    # Compute diff
    diff = compute_diff(original_code, refactored_code, file_rel_path)
    
    # Check if actually changed
    if not diff.strip():
        return RefactoringResult(
            file_path=str(file_path),
            repo=repo,
            success=True,  # Not a failure, just nothing to improve
            original_code=original_code,
            refactored_code=refactored_code,
            diff="",
            error_message="No changes made (code already optimal or cannot be improved)",
            llm_response_raw=raw_response,
            latency_ms=latency_ms,
        )
    
    return RefactoringResult(
        file_path=str(file_path),
        repo=repo,
        success=True,
        original_code=original_code,
        refactored_code=refactored_code,
        diff=diff,
        error_message=None,
        llm_response_raw=raw_response,
        latency_ms=latency_ms,
    )


# ==============================================================================
# BUILD & TEST VALIDATION
# ==============================================================================

def detect_build_system(repo_path: Path) -> Tuple[str, str]:
    """Detect build system and return fallback (build_cmd, test_cmd)."""
    # Maven (Java)
    if (repo_path / "pom.xml").exists():
        mvn = "mvn" if shutil.which("mvn") else "./mvnw"
        return f"{mvn} compile -q", f"{mvn} test -q"

    # Gradle (Java)
    if (repo_path / "build.gradle").exists() or (repo_path / "build.gradle.kts").exists():
        gradle = "gradle" if shutil.which("gradle") else "./gradlew"
        return f"{gradle} compileJava -q", f"{gradle} test -q"

    # Node.js / npm
    if (repo_path / "package.json").exists():
        pkg = json.loads((repo_path / "package.json").read_text())
        scripts = pkg.get("scripts", {})
        install_cmd = "npm ci" if (repo_path / "package-lock.json").exists() else "npm install"
        build_cmd = "npm run build" if "build" in scripts else install_cmd
        test_cmd = "npm test" if "test" in scripts else ""
        return build_cmd, test_cmd

    # Python
    if (repo_path / "setup.py").exists() or (repo_path / "pyproject.toml").exists() or (repo_path / "requirements.txt").exists():
        return "python -m compileall -q .", "pytest -q" if shutil.which("pytest") else ""

    # Go
    if (repo_path / "go.mod").exists():
        return "go build ./...", "go test ./..."

    return "echo 'Unknown build system'", ""


def _fallback_verify_profile(repo_path: Path) -> VerifyProfile:
    build_cmd, test_cmd = detect_build_system(repo_path)
    install_cmds: list[str] = []
    if (repo_path / "package.json").exists():
        pkg = json.loads((repo_path / "package.json").read_text(encoding="utf-8", errors="replace"))
        scripts = pkg.get("scripts", {})
        # Fallback install avoids lifecycle scripts (prepare/husky hooks),
        # which frequently fail in ephemeral working copies but are not needed
        # for build/test verification in this study pipeline.
        install_base = "npm ci --ignore-scripts" if (repo_path / "package-lock.json").exists() else "npm install --ignore-scripts"
        install_cmds = [install_base]
        if build_cmd in install_cmds:
            build_cmd = ""
        if "test:node" in scripts:
            test_cmd = "npm run test:node"
    return VerifyProfile(
        source_path=None,
        install=install_cmds,
        build=[build_cmd] if build_cmd else [],
        test=[test_cmd] if test_cmd else [],
    )


def _run_shell_command(
    command: str,
    cwd: Path,
    phase: str,
    timeout_sec: int,
) -> CommandRun:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        duration = time.perf_counter() - start
        return CommandRun(
            phase=phase,
            command=command,
            exit_code=completed.returncode,
            success=completed.returncode == 0,
            duration_sec=round(duration, 3),
            stdout_tail=_safe_snippet(completed.stdout),
            stderr_tail=_safe_snippet(completed.stderr),
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        return CommandRun(
            phase=phase,
            command=command,
            exit_code=124,
            success=False,
            duration_sec=round(duration, 3),
            stdout_tail=_safe_snippet(exc.stdout or ""),
            stderr_tail=_safe_snippet((exc.stderr or "") + "\n[TIMEOUT]"),
        )
    except Exception as exc:
        duration = time.perf_counter() - start
        return CommandRun(
            phase=phase,
            command=command,
            exit_code=1,
            success=False,
            duration_sec=round(duration, 3),
            stdout_tail="",
            stderr_tail=_safe_snippet(str(exc)),
        )


def run_repo_verification(
    repo_path: Path,
    profile: VerifyProfile,
    include_install: bool = True,
    run_test: bool = True,
) -> VerificationResult:
    """Run install/build/test commands and return detailed verification output."""
    commands: list[CommandRun] = []

    def fail_result(reason: str, cmd: CommandRun, build_success: bool, test_success: bool, test_status: str) -> VerificationResult:
        snippet = (cmd.stderr_tail or "") + ("\n" + cmd.stdout_tail if cmd.stdout_tail else "")
        return VerificationResult(
            passed=False,
            commands=commands,
            build_success=build_success,
            test_success=test_success,
            test_status=test_status,
            failure_type=reason,
            failure_command=cmd.command,
            failure_exit_code=cmd.exit_code,
            failure_snippet=_safe_snippet(snippet),
        )

    if include_install:
        for command in profile.install:
            run = _run_shell_command(command, repo_path, phase="install", timeout_sec=VERIFY_COMMAND_TIMEOUT_SEC)
            commands.append(run)
            if not run.success:
                return fail_result("install_failed", run, build_success=False, test_success=False, test_status="missing")

    build_success = True
    for command in profile.build:
        run = _run_shell_command(command, repo_path, phase="build", timeout_sec=VERIFY_COMMAND_TIMEOUT_SEC)
        commands.append(run)
        if not run.success:
            build_success = False
            return fail_result("build_failed", run, build_success=False, test_success=False, test_status="missing")

    if not run_test:
        return VerificationResult(
            passed=build_success,
            commands=commands,
            build_success=build_success,
            test_success=False,
            test_status="skipped",
        )

    if not profile.test:
        return VerificationResult(
            passed=build_success,
            commands=commands,
            build_success=build_success,
            test_success=False,
            test_status="missing",
        )

    for command in profile.test:
        run = _run_shell_command(command, repo_path, phase="test", timeout_sec=VERIFY_COMMAND_TIMEOUT_SEC)
        commands.append(run)
        if not run.success:
            return fail_result("test_failed", run, build_success=build_success, test_success=False, test_status="failed")

    return VerificationResult(
        passed=build_success,
        commands=commands,
        build_success=build_success,
        test_success=True,
        test_status="passed",
    )


def run_fast_file_checks(file_path: Path, language: str) -> VerificationResult:
    """
    Run quick per-file checks before expensive repo-level verification.
    Keeps checks intentionally cheap and bounded.
    """
    commands: list[CommandRun] = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        cmd = CommandRun(
            phase="fast_check",
            command="read_file",
            exit_code=1,
            success=False,
            duration_sec=0.0,
            stdout_tail="",
            stderr_tail=_safe_snippet(str(exc)),
        )
        return VerificationResult(
            passed=False,
            commands=[cmd],
            build_success=False,
            test_success=False,
            test_status="missing",
            failure_type="fast_check_failed",
            failure_command=cmd.command,
            failure_exit_code=cmd.exit_code,
            failure_snippet=cmd.stderr_tail,
        )

    non_empty = bool(content.strip())
    empty_cmd = CommandRun(
        phase="fast_check",
        command="non_empty_file",
        exit_code=0 if non_empty else 1,
        success=non_empty,
        duration_sec=0.0,
        stdout_tail="",
        stderr_tail="" if non_empty else "File became empty after refactor.",
    )
    commands.append(empty_cmd)
    if not non_empty:
        return VerificationResult(
            passed=False,
            commands=commands,
            build_success=False,
            test_success=False,
            test_status="missing",
            failure_type="fast_check_failed",
            failure_command=empty_cmd.command,
            failure_exit_code=empty_cmd.exit_code,
            failure_snippet=empty_cmd.stderr_tail,
        )

    if language == "python":
        compile_cmd = _run_shell_command(
            command=f"python -m py_compile {shlex.quote(str(file_path))}",
            cwd=file_path.parent,
            phase="fast_check",
            timeout_sec=FAST_VERIFY_TIMEOUT_SEC,
        )
        commands.append(compile_cmd)
        if not compile_cmd.success:
            snippet = (compile_cmd.stderr_tail or "") + ("\n" + compile_cmd.stdout_tail if compile_cmd.stdout_tail else "")
            return VerificationResult(
                passed=False,
                commands=commands,
                build_success=False,
                test_success=False,
                test_status="missing",
                failure_type="fast_check_failed",
                failure_command=compile_cmd.command,
                failure_exit_code=compile_cmd.exit_code,
                failure_snippet=_safe_snippet(snippet),
            )

    return VerificationResult(
        passed=True,
        commands=commands,
        build_success=True,
        test_success=False,
        test_status="missing",
    )


def run_build_and_tests(
    repo_path: Path,
    preferred_profile: Optional[str] = None,
    include_install: bool = False,
) -> TestResult:
    """
    Compatibility wrapper for repo-level reporting.
    Uses verification profile if present, otherwise fallback heuristics.
    """
    profile = load_verify_profile(repo_path, preferred_profile)
    if not (profile.install or profile.build or profile.test):
        profile = _fallback_verify_profile(repo_path)

    start_time = time.perf_counter()
    verify = run_repo_verification(repo_path, profile, include_install=include_install, run_test=True)
    duration = time.perf_counter() - start_time
    build_cmd = " && ".join(profile.build) if profile.build else "(none)"
    test_cmd = " && ".join(profile.test) if profile.test else "(missing)"
    build_output = "\n".join(
        [f"[{c.phase}] {c.command}\n{c.stdout_tail}\n{c.stderr_tail}" for c in verify.commands if c.phase in {"install", "build"}]
    )
    test_output = "\n".join(
        [f"[{c.phase}] {c.command}\n{c.stdout_tail}\n{c.stderr_tail}" for c in verify.commands if c.phase == "test"]
    ) or ("No tests configured" if verify.test_status == "missing" else "")
    effective_test_success = verify.test_success or verify.test_status == "missing"
    return TestResult(
        repo=repo_path.name,
        build_success=verify.build_success,
        test_success=effective_test_success,
        build_output=_safe_snippet(build_output, 2000),
        test_output=_safe_snippet(f"[test_status:{verify.test_status}] {test_output}", 2000),
        build_command=build_cmd,
        test_command=test_cmd,
        duration_sec=round(duration, 2),
    )


# ==============================================================================
# VERIFICATION SERIALIZATION & REGRESSION
# ==============================================================================

def serialize_verification(verify: VerificationResult) -> List[Dict[str, Any]]:
    """Convert VerificationResult commands to a plain list of dicts for JSON output."""
    return [
        {
            "phase": cmd.phase,
            "command": cmd.command,
            "exit_code": cmd.exit_code,
            "success": cmd.success,
            "duration_sec": cmd.duration_sec,
            "stdout_tail": cmd.stdout_tail,
            "stderr_tail": cmd.stderr_tail,
        }
        for cmd in verify.commands
    ]


def compute_regression(
    baseline_cmds: List[Dict[str, Any]],
    post_cmds: List[Dict[str, Any]],
) -> RegressionReport:
    """
    Compare baseline vs post verification command outcomes.

    Rules:
    - new_failure: baseline success (or command absent from baseline) + post fail.
    - same_failure: baseline fail + post fail, same (phase, command).
    - fixed_failure: baseline fail + post success.
    - Commands that passed in both are ignored (normal).
    - Commands present in baseline but absent in post count as new failures
      (conservative: we cannot prove they still pass).
    - safe_non_regression = len(new_failures) == 0.
    """
    baseline_by_key: Dict[Tuple[str, str], bool] = {}
    for cmd in baseline_cmds:
        key = (cmd["phase"], cmd["command"])
        baseline_by_key[key] = cmd["success"]

    post_by_key: Dict[Tuple[str, str], bool] = {}
    for cmd in post_cmds:
        key = (cmd["phase"], cmd["command"])
        post_by_key[key] = cmd["success"]

    new_failures: List[Dict[str, Any]] = []
    same_failures: List[Dict[str, Any]] = []
    fixed_failures: List[Dict[str, Any]] = []

    # Walk post commands
    for cmd in post_cmds:
        key = (cmd["phase"], cmd["command"])
        baseline_success = baseline_by_key.get(key)
        if cmd["success"]:
            # Post passed
            if baseline_success is False:
                fixed_failures.append({"phase": cmd["phase"], "command": cmd["command"]})
        else:
            # Post failed
            if baseline_success is True or baseline_success is None:
                # Was passing or is a new command that failed
                new_failures.append({"phase": cmd["phase"], "command": cmd["command"],
                                     "exit_code": cmd["exit_code"]})
            elif baseline_success is False:
                same_failures.append({"phase": cmd["phase"], "command": cmd["command"],
                                      "exit_code": cmd["exit_code"]})

    # Baseline commands not present in post → conservative new failure
    for key, success in baseline_by_key.items():
        if key not in post_by_key and success:
            new_failures.append({"phase": key[0], "command": key[1],
                                 "exit_code": None, "note": "not_run_post"})

    return RegressionReport(
        new_failures=new_failures,
        same_failures=same_failures,
        fixed_failures=fixed_failures,
        safe_non_regression=len(new_failures) == 0,
        baseline_command_count=len(baseline_cmds),
        post_command_count=len(post_cmds),
    )


# ==============================================================================
# MAIN STUDY WORKFLOW
# ==============================================================================

def run_refactoring_study(
    repos: Optional[List[str]] = None,
    max_repos: int = 5,
    dry_run: bool = False,
    max_files_per_repo: int = MAX_FILES_PER_REPO,
    max_attempts_per_file: int = MAX_ATTEMPTS_PER_FILE,
    verify_profile_path: Optional[str] = None,
    enable_git_commits: bool = False,
) -> StudySession:
    """
    Run the complete refactoring study.
    
    Args:
        repos: List of repo names to process. If None, auto-select from Sonar data.
        max_repos: Maximum number of repos to process.
        dry_run: If True, don't actually modify files.
    
    Returns:
        StudySession with complete results.
    """
    setup_directories()
    session_id = generate_session_id()
    session_dir = COMPARE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("LLM-ASSISTED REFACTORING STUDY")
    print("=" * 70)
    print(f"Session ID: {session_id}")
    print(f"Model: {REFACTOR_MODEL}")
    print(f"Dry run: {dry_run}")
    print(f"Max attempts/file: {max_attempts_per_file}")
    print(f"Verification profile: {verify_profile_path or 'auto (.refactor_verify.yaml|verify.yaml)'}")
    print(f"Git snapshots: {enable_git_commits}")
    print(f"Output dir: {session_dir}")
    print()
    
    # Initialize session
    session = StudySession(
        session_id=session_id,
        start_time=datetime.now(timezone.utc).isoformat(),
        model_used=REFACTOR_MODEL,
        prompt_version=hashlib.sha256(REFACTORING_SYSTEM_PROMPT.encode()).hexdigest()[:12],
    )
    
    # Load Sonar metrics
    sonar_path = config.sonar_metrics_path()
    if not sonar_path.exists():
        print(f"ERROR: Sonar metrics not found at {sonar_path}")
        print("Please run sonar_runner.py first.")
        return session
    
    sonar_df = pd.read_csv(sonar_path)
    print(f"Loaded {len(sonar_df)} file metrics from Sonar")
    
    # Select repos
    if repos is None:
        # Auto-select repos with moderate code smells
        repo_smells = sonar_df.groupby("repo")["sonar_code_smells"].sum().sort_values()
        repos = repo_smells[repo_smells > 10].head(max_repos).index.tolist()
    
    print(f"Selected repos: {repos}")
    print()
    
    # Initialize LLM client
    client = OpenAI(
        api_key=config.LLM_API_KEY or "not-needed",
        base_url=f"{config.LLM_BASE_URL}/v1" if "localhost" in config.LLM_BASE_URL else config.LLM_BASE_URL,
    )
    
    # Storage for per-session summaries
    repo_summaries: List[Dict[str, Any]] = []
    
    # Process each repo
    for repo in repos:
        print("-" * 70)
        print(f"PROCESSING: {repo}")
        print("-" * 70)

        original_repo_path = config.RAW_REPOS_DIR / repo
        if not original_repo_path.exists():
            print(f"  Repo not found at {original_repo_path}, skipping")
            continue

        session.repos_processed.append(repo)

        baseline_commit = get_git_commit(original_repo_path)
        repo_url = repo_url_for_name(repo)
        print(f"  Baseline commit: {baseline_commit}")

        # ---- Compare-package directory layout ----
        repo_compare_dir = session_dir / repo
        repo_before_dir = repo_compare_dir / "before_files"
        repo_after_dir = repo_compare_dir / "after_files"
        repo_refactored_dir = repo_compare_dir / "refactored_repo"
        for d in (repo_compare_dir, repo_before_dir, repo_after_dir):
            d.mkdir(parents=True, exist_ok=True)

        if dry_run:
            working_repo_path = original_repo_path
            print(f"  Working on: {working_repo_path} (dry-run, no copy)")
        else:
            working_repo_path = create_working_copy(original_repo_path, repo_refactored_dir)
            print(f"  Created working copy: {working_repo_path}")

        session.repo_metadata[repo] = {
            "repo_url": repo_url,
            "baseline_commit_sha": baseline_commit,
            "source_repo_path": str(original_repo_path),
            "working_repo_path": str(working_repo_path),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        if enable_git_commits and not dry_run:
            init_git_snapshot(working_repo_path, baseline_commit)
            print("  Initialized git snapshot in working copy")

        verify_profile = load_verify_profile(working_repo_path, verify_profile_path)
        if not (verify_profile.install or verify_profile.build or verify_profile.test):
            verify_profile = _fallback_verify_profile(working_repo_path)
        print(f"  Verification profile: {verify_profile.source_path or 'fallback heuristics'}")

        candidates = select_refactoring_candidates(sonar_df, repo, max_files=max_files_per_repo)
        print(f"  Refactoring candidates: {len(candidates)} files")
        if candidates.empty:
            print("  No suitable candidates found")
            session.repo_metadata[repo]["ended_at"] = datetime.now(timezone.utc).isoformat()
            # Still emit a minimal manifest
            manifest = {
                "session_id": session_id, "repo": repo, "repo_url": repo_url,
                "baseline_commit_sha": baseline_commit,
                "model_used": REFACTOR_MODEL,
                "prompt_version": session.prompt_version,
                "started_at": session.repo_metadata[repo]["started_at"],
                "ended_at": session.repo_metadata[repo]["ended_at"],
                "accepted_files": [], "files_attempted": 0, "files_accepted": 0, "files_failed": 0,
                "safe_non_regression": None, "new_failures_count": 0,
                "same_failures_count": 0, "fixed_failures_count": 0,
                "sonar_post_ran": False,
                "total_smells_reduced": 0, "total_debt_reduced_min": 0.0,
            }
            (repo_compare_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            continue

        repo_changed_files = 0
        accepted_file_paths: set[str] = set()
        install_completed = not bool(verify_profile.install)

        # Run install once per repo; abort this repo early on install failure.
        if not dry_run and verify_profile.install:
            print("  Running install preflight once for repo...")
            install_only_profile = VerifyProfile(
                source_path=verify_profile.source_path,
                install=verify_profile.install,
                build=[],
                test=[],
            )
            install_result = run_repo_verification(
                working_repo_path,
                install_only_profile,
                include_install=True,
                run_test=False,
            )
            if not install_result.passed:
                print("      ✗ Install preflight FAILED; skipping repo")
                session.build_failures += 1
                session.repo_metadata[repo]["ended_at"] = datetime.now(timezone.utc).isoformat()
                manifest = {
                    "session_id": session_id, "repo": repo, "repo_url": repo_url,
                    "baseline_commit_sha": baseline_commit,
                    "model_used": REFACTOR_MODEL,
                    "prompt_version": session.prompt_version,
                    "started_at": session.repo_metadata[repo]["started_at"],
                    "ended_at": session.repo_metadata[repo]["ended_at"],
                    "accepted_files": [], "files_attempted": 0, "files_accepted": 0, "files_failed": 0,
                    "safe_non_regression": None, "new_failures_count": 0,
                    "same_failures_count": 0, "fixed_failures_count": 0,
                    "sonar_post_ran": False,
                    "install_failed": True,
                    "total_smells_reduced": 0, "total_debt_reduced_min": 0.0,
                }
                (repo_compare_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                continue
            install_completed = True
            print("      ✓ Install preflight passed")

        # ---- Baseline full verification (build + test) ----
        baseline_verify: Optional[VerificationResult] = None
        baseline_cmds: List[Dict[str, Any]] = []
        if not dry_run:
            print("  Running baseline verification (build + test) before edits...")
            baseline_verify = run_repo_verification(
                working_repo_path,
                verify_profile,
                include_install=False,
                run_test=True,
            )
            baseline_cmds = serialize_verification(baseline_verify)
            (repo_compare_dir / "verification_baseline.json").write_text(
                json.dumps(baseline_cmds, indent=2), encoding="utf-8"
            )
            pass_count = sum(1 for c in baseline_cmds if c["success"])
            fail_count = len(baseline_cmds) - pass_count
            print(f"      Baseline: {pass_count} passed, {fail_count} failed ({baseline_verify.test_status} tests)")

        for _, row in candidates.iterrows():
            file_rel_path = str(row["file_path"])
            file_abs_path = working_repo_path / file_rel_path
            if not file_abs_path.exists():
                print(f"    ✗ File not found: {file_rel_path}")
                session.files_failed += 1
                continue

            print(f"\n  [{session.files_refactored + session.files_failed + 1}] {file_rel_path}")
            print(f"      Pre-metrics: smells={row['sonar_code_smells']}, complexity={row.get('sonar_cognitive_complexity', 'N/A')}")

            file_metrics = {
                "code_smells": row.get("sonar_code_smells"),
                "cognitive_complexity": row.get("sonar_cognitive_complexity"),
                "sqale_index": row.get("sonar_sqale_index"),
                "ncloc": row.get("sonar_ncloc"),
                "duplicated_lines_density": row.get("sonar_duplicated_lines_density", 0),
            }
            for key, value in row.items():
                if ("issue" in str(key).lower() or "rule" in str(key).lower()) and value not in (None, "", 0, 0.0):
                    file_metrics[str(key)] = value

            accepted_this_file = False
            last_feedback: Optional[str] = None
            last_diff: Optional[str] = None
            final_result: Optional[RefactoringResult] = None
            previous_failure_sig: Optional[str] = None

            for attempt in range(1, max_attempts_per_file + 1):
                result = apply_llm_refactoring(
                    file_path=file_abs_path,
                    repo_path=working_repo_path,
                    repo=repo,
                    file_rel_path=file_rel_path,
                    metrics=file_metrics,
                    client=client,
                    attempt=attempt,
                    retry_feedback=last_feedback,
                    previous_diff=last_diff,
                )
                final_result = result

                if not result.success:
                    sig = failure_signature("llm_refactor_failed", None, _safe_snippet(result.error_message or ""))
                    last_feedback = result.error_message or "LLM refactoring failed."
                    last_diff = result.diff or ""
                    if previous_failure_sig and sig == previous_failure_sig:
                        print("      ✗ Repeated failure signature detected; stopping retries early")
                        break
                    previous_failure_sig = sig
                    print(f"      ✗ Attempt {attempt}/{max_attempts_per_file} failed: {result.error_message}")
                    continue

                if not result.diff:
                    session.files_refactored += 1
                    accepted_this_file = True
                    print("      ○ No changes needed")
                    break

                if dry_run:
                    session.files_refactored += 1
                    repo_changed_files += 1
                    accepted_this_file = True
                    print(f"      ✓ Would refactor (dry run), diff lines={len((result.diff or '').splitlines())}")
                    break

                original_disk = file_abs_path.read_text(encoding="utf-8", errors="replace")
                try:
                    file_abs_path.write_text(result.refactored_code or "", encoding="utf-8")
                except Exception as exc:
                    sig = failure_signature("write_failed", "write_file", _safe_snippet(str(exc)))
                    last_feedback = f"Failed to write file: {exc}"
                    last_diff = result.diff or ""
                    if previous_failure_sig and sig == previous_failure_sig:
                        print("      ✗ Repeated failure signature detected; stopping retries early")
                        break
                    previous_failure_sig = sig
                    continue

                fast_verify = run_fast_file_checks(file_abs_path, detect_language(file_abs_path))
                verify_commands: list[CommandRun] = list(fast_verify.commands)
                verify_result: Optional[VerificationResult] = None

                if fast_verify.passed:
                    verify_result = run_repo_verification(
                        working_repo_path,
                        verify_profile,
                        include_install=False,
                        run_test=True,
                    )
                    verify_commands.extend(verify_result.commands)

                passed = fast_verify.passed and (verify_result.passed if verify_result else False)

                if not passed:
                    file_abs_path.write_text(original_disk, encoding="utf-8")
                    if not fast_verify.passed:
                        fail_type = fast_verify.failure_type or "fast_check_failed"
                        fail_command = fast_verify.failure_command
                        fail_snippet = fast_verify.failure_snippet or "fast check failed"
                    else:
                        fail_type = verify_result.failure_type if verify_result else "verification_failed"
                        fail_command = verify_result.failure_command if verify_result else None
                        fail_snippet = verify_result.failure_snippet if verify_result else "verification failed"
                    sig = failure_signature(fail_type, fail_command, _safe_snippet(fail_snippet or ""))
                    last_feedback = (
                        f"Verification failed on command: {(fail_command or '')}\n"
                        f"{_safe_snippet(fail_snippet or '')}"
                    )
                    last_diff = result.diff or ""
                    print(f"      ✗ Attempt {attempt}/{max_attempts_per_file} failed verification ({fail_type}); rolled back")
                    if previous_failure_sig and sig == previous_failure_sig:
                        print("      ✗ Repeated failure signature detected; stopping retries early")
                        break
                    previous_failure_sig = sig
                    continue

                # Accepted: save before/after snapshots
                save_before_after(
                    before_dir=repo_before_dir,
                    after_dir=repo_after_dir,
                    file_path=file_rel_path,
                    original_code=result.original_code,
                    refactored_code=result.refactored_code or "",
                )

                if enable_git_commits:
                    commit_accepted_change(working_repo_path, file_rel_path, attempt)

                session.files_refactored += 1
                repo_changed_files += 1
                accepted_file_paths.add(file_rel_path)
                accepted_this_file = True
                print(f"      ✓ Accepted on attempt {attempt} ({result.latency_ms}ms)")
                break

            if not accepted_this_file:
                session.files_failed += 1
                print(f"      ✗ Exhausted retries ({max_attempts_per_file}) for {file_rel_path}")

        # ---- Post verification + regression analysis ----
        regression: Optional[RegressionReport] = None
        safe_non_regression: Optional[bool] = None
        sonar_post_ran = False
        repo_smells_reduced = 0
        repo_debt_reduced = 0.0

        if not dry_run and repo_changed_files > 0:
            print("\n  Running post verification on refactored copy...")
            post_verify = run_repo_verification(
                working_repo_path,
                verify_profile,
                include_install=False,
                run_test=True,
            )
            post_cmds = serialize_verification(post_verify)
            (repo_compare_dir / "verification_post.json").write_text(
                json.dumps(post_cmds, indent=2), encoding="utf-8"
            )

            # Compute regression
            regression = compute_regression(baseline_cmds, post_cmds)
            safe_non_regression = regression.safe_non_regression
            (repo_compare_dir / "verification_regression.json").write_text(
                json.dumps(asdict(regression), indent=2), encoding="utf-8"
            )

            pass_count = sum(1 for c in post_cmds if c["success"])
            fail_count = len(post_cmds) - pass_count
            print(f"      Post: {pass_count} passed, {fail_count} failed ({post_verify.test_status} tests)")
            print(f"      Regression: new_failures={len(regression.new_failures)}, "
                  f"same_failures={len(regression.same_failures)}, "
                  f"fixed_failures={len(regression.fixed_failures)}")
            print(f"      safe_non_regression={safe_non_regression}")

            if not safe_non_regression:
                session.test_failures += 1
                print(f"      (Original repo preserved at {original_repo_path})")

            # ---- Sonar post-scan gated on safe_non_regression ----
            if safe_non_regression:
                print("\n  Re-running Sonar analysis on refactored copy...")
                try:
                    project_key = f"{project_key_for_repo(working_repo_path)}_refactored"
                    run_sonar_scan(working_repo_path, project_key)
                    time.sleep(5)

                    post_metrics = fetch_file_metrics(project_key)
                    post_df = pd.DataFrame(post_metrics)

                    # Build Sonar subsets scoped to analyzed (accepted) files
                    pre_rows = []
                    post_rows = []
                    delta_rows = []
                    for _, cand_row in candidates.iterrows():
                        fp = str(cand_row["file_path"])
                        if fp not in accepted_file_paths:
                            continue
                        pre_entry = {
                            "file_path": fp,
                            "repo": repo,
                            "sonar_code_smells": cand_row.get("sonar_code_smells"),
                            "sonar_cognitive_complexity": cand_row.get("sonar_cognitive_complexity"),
                            "sonar_sqale_index": cand_row.get("sonar_sqale_index"),
                            "sonar_duplicated_lines_density": cand_row.get("sonar_duplicated_lines_density"),
                            "sonar_ncloc": cand_row.get("sonar_ncloc"),
                        }
                        pre_rows.append(pre_entry)

                        post_match = post_df[post_df["file_path"] == fp]
                        if post_match.empty:
                            continue

                        post_entry = {
                            "file_path": fp,
                            "repo": repo,
                            "sonar_code_smells": post_match["sonar_code_smells"].values[0] if "sonar_code_smells" in post_match else None,
                            "sonar_cognitive_complexity": post_match["sonar_cognitive_complexity"].values[0] if "sonar_cognitive_complexity" in post_match else None,
                            "sonar_sqale_index": post_match["sonar_sqale_index"].values[0] if "sonar_sqale_index" in post_match else None,
                            "sonar_duplicated_lines_density": post_match["sonar_duplicated_lines_density"].values[0] if "sonar_duplicated_lines_density" in post_match else None,
                            "sonar_ncloc": post_match["sonar_ncloc"].values[0] if "sonar_ncloc" in post_match else None,
                        }
                        post_rows.append(post_entry)

                        # delta
                        delta_entry = {"file_path": fp, "repo": repo}
                        for metric_key in ("sonar_code_smells", "sonar_cognitive_complexity",
                                           "sonar_sqale_index", "sonar_duplicated_lines_density", "sonar_ncloc"):
                            pre_val = pre_entry.get(metric_key)
                            post_val = post_entry.get(metric_key)
                            delta_entry[f"pre_{metric_key}"] = pre_val
                            delta_entry[f"post_{metric_key}"] = post_val
                            if pre_val is not None and post_val is not None:
                                delta_entry[f"delta_{metric_key}"] = post_val - pre_val
                            else:
                                delta_entry[f"delta_{metric_key}"] = None
                        delta_rows.append(delta_entry)

                        # Accumulate session-level totals
                        d_smells = delta_entry.get("delta_sonar_code_smells")
                        if d_smells is not None and d_smells < 0:
                            repo_smells_reduced += abs(int(d_smells))
                        d_debt = delta_entry.get("delta_sonar_sqale_index")
                        if d_debt is not None and d_debt < 0:
                            repo_debt_reduced += abs(float(d_debt))

                    # Write Sonar CSVs
                    if pre_rows:
                        pd.DataFrame(pre_rows).to_csv(repo_compare_dir / "sonar_pre_subset.csv", index=False)
                    if post_rows:
                        pd.DataFrame(post_rows).to_csv(repo_compare_dir / "sonar_post_subset.csv", index=False)
                    if delta_rows:
                        pd.DataFrame(delta_rows).to_csv(repo_compare_dir / "sonar_delta.csv", index=False)

                    sonar_post_ran = True
                    session.total_smells_reduced += repo_smells_reduced
                    session.total_debt_reduced_min += repo_debt_reduced
                    print(f"      ✓ Sonar comparison written ({len(delta_rows)} files)")
                except Exception as e:
                    print(f"      ✗ Sonar re-analysis failed: {e}")
            else:
                print("  Skipping post-refactor Sonar scan (safe_non_regression=false)")

        elif not dry_run and repo_changed_files == 0:
            # No changes accepted — still emit baseline verification if we had one
            safe_non_regression = True  # trivially safe, nothing changed

        # ---- Emit manifest.json ----
        session.repo_metadata[repo]["ended_at"] = datetime.now(timezone.utc).isoformat()
        manifest = {
            "session_id": session_id,
            "repo": repo,
            "repo_url": repo_url,
            "baseline_commit_sha": baseline_commit,
            "model_used": REFACTOR_MODEL,
            "prompt_version": session.prompt_version,
            "started_at": session.repo_metadata[repo]["started_at"],
            "ended_at": session.repo_metadata[repo]["ended_at"],
            "accepted_files": sorted(accepted_file_paths),
            "files_attempted": len(candidates),
            "files_accepted": len(accepted_file_paths),
            "files_failed": session.files_failed,
            "safe_non_regression": safe_non_regression,
            "new_failures_count": len(regression.new_failures) if regression else 0,
            "same_failures_count": len(regression.same_failures) if regression else 0,
            "fixed_failures_count": len(regression.fixed_failures) if regression else 0,
            "sonar_post_ran": sonar_post_ran,
            "total_smells_reduced": repo_smells_reduced,
            "total_debt_reduced_min": round(repo_debt_reduced, 1),
            "artifact_paths": {
                "before_files": "before_files/",
                "after_files": "after_files/",
                "refactored_repo": "refactored_repo/",
                "verification_baseline": "verification_baseline.json" if baseline_cmds else None,
                "verification_post": "verification_post.json" if regression else None,
                "verification_regression": "verification_regression.json" if regression else None,
                "sonar_pre_subset": "sonar_pre_subset.csv" if sonar_post_ran else None,
                "sonar_post_subset": "sonar_post_subset.csv" if sonar_post_ran else None,
                "sonar_delta": "sonar_delta.csv" if sonar_post_ran else None,
            },
        }
        (repo_compare_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        repo_summaries.append({
            "repo": repo,
            "safe_non_regression": safe_non_regression,
            "new_failures": len(regression.new_failures) if regression else 0,
            "sonar_post": "ran" if sonar_post_ran else "skipped",
            "files_accepted": len(accepted_file_paths),
            "smells_reduced": repo_smells_reduced,
        })

    # ======================================================================
    # Finalize session
    # ======================================================================
    session.end_time = datetime.now(timezone.utc).isoformat()

    # Save thin session-level JSON
    session_json_path = session_dir / "session.json"
    session.repo_metadata["_compare_dir"] = str(session_dir)
    session_dict = asdict(session)
    with open(session_json_path, "w", encoding="utf-8") as f:
        json.dump(session_dict, f, indent=2)

    # Print comparison-centric summary
    print("\n" + "=" * 70)
    print("STUDY SUMMARY")
    print("=" * 70)
    print(f"  Session ID:       {session.session_id}")
    print(f"  Model:            {session.model_used}")
    print(f"  Duration:         {session.start_time} → {session.end_time}")
    print(f"  Repos processed:  {len(session.repos_processed)}")
    print(f"  Files accepted:   {session.files_refactored}")
    print(f"  Files failed:     {session.files_failed}")
    print()
    for rs in repo_summaries:
        print(f"  {rs['repo']}: safe_non_regression={rs['safe_non_regression']}, "
              f"new_failures={rs['new_failures']}, sonar_post={rs['sonar_post']}, "
              f"files_accepted={rs['files_accepted']}, smells_reduced={rs['smells_reduced']}")
    print()
    print(f"  Output dir: {session_dir}")
    print()
    
    return session


# ==============================================================================
# CLI
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM-Assisted Refactoring Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on auto-selected repos
  python -m experiments.refactoring_study --dry-run
  
  # Run on specific repos
  python -m experiments.refactoring_study --repos axios,express
  
  # Full study with 3 repos
  python -m experiments.refactoring_study --max-repos 3
        """
    )
    
    parser.add_argument(
        "--repos",
        type=str,
        default=None,
        help="Comma-separated list of repo names to process"
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=5,
        help="Maximum number of repos to auto-select (default: 5)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without actually modifying files"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=MAX_FILES_PER_REPO,
        help=f"Maximum files to refactor per repo (default: {MAX_FILES_PER_REPO})"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_ATTEMPTS_PER_FILE,
        help=f"Maximum repair attempts per file (default: {MAX_ATTEMPTS_PER_FILE})",
    )
    parser.add_argument(
        "--verify-profile",
        type=str,
        default=None,
        help="Profile filename at repo root (e.g., .refactor_verify.yaml or verify.yaml)",
    )
    parser.add_argument(
        "--enable-git-commits",
        action="store_true",
        help="Initialize git in working copy and commit accepted changes",
    )
    
    args = parser.parse_args()
    
    # Parse repos
    repos = None
    if args.repos:
        repos = [r.strip() for r in args.repos.split(",")]
    
    # Run study
    session = run_refactoring_study(
        repos=repos,
        max_repos=args.max_repos,
        dry_run=args.dry_run,
        max_files_per_repo=args.max_files,
        max_attempts_per_file=max(1, int(args.max_attempts)),
        verify_profile_path=args.verify_profile,
        enable_git_commits=args.enable_git_commits,
    )
    
    # Exit with error if there were test failures
    if session.test_failures > 0:
        print("\n⚠ WARNING: Some tests failed after refactoring!")
        exit(1)


if __name__ == "__main__":
    main()
