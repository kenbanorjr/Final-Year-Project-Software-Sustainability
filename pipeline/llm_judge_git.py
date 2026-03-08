"""
Git-Augmented LLM-based Code Quality Assessor.

This module extends the LLM judge approach from ``llm_judge.py`` by also
feeding the LLM each file's git history metrics (churn, bus-factor, recency,
contributor concentration, etc.).  The model can then factor code-evolution
context into its quality assessment alongside the source code itself.

Key differences from ``llm_judge.py``:
- Input file list comes from ``git_metrics.csv`` (only files with git data).
- Each prompt includes a structured block of git history metrics.
- System prompt contains additional guidance on interpreting git signals.
- Output written to ``llm_git_metrics_<model>_runNNN.csv``.
- A separate parse-failure log is used (``llm_git_parse_failures.jsonl``).

The output CSV schema is identical to the code-only judge (all ``llm_*`` prefixed
columns) plus an ``llm_git_augmented`` flag, ensuring merge compatibility with
the rest of the pipeline.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.configs import config
from pipeline.configs.general_repo_filter import included_extensions, is_excluded_path, is_included_path
from pipeline.utils import detect_language, LLMClient, LLMError

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# Path to store raw responses for parse failures (append-only JSONL)
PARSE_FAILURE_LOG = config.llm_git_parse_failures_path()

# ============================================================================
# CONSTANTS – same budget as the code-only judge
# ============================================================================

MAX_CODE_CHARS = 20000
MAX_RESPONSE_TOKENS = 1000
PROMPT_TOKENS_ESTIMATE = 600

HEAD_CHARS = 16000
TAIL_CHARS = 4000

MAX_FILE_SIZE_KB = 500
MIN_FILE_SIZE_BYTES = 10

ANALYZABLE_EXTENSIONS = included_extensions()

# Git metric columns to inject into the prompt (order matters for readability).
GIT_PROMPT_FIELDS: list[tuple[str, str]] = [
    ("churn_12m", "Code churn (lines added + deleted, 12 months)"),
    ("added_lines_12m", "Lines added (12 months)"),
    ("deleted_lines_12m", "Lines deleted (12 months)"),
    ("unique_authors_12m", "Unique authors (12 months)"),
    ("dominant_author", "Dominant author"),
    ("dominant_author_share", "Dominant author share (0-1)"),
    ("single_contributor_12m", "Single contributor (12 months)"),
    ("knowledge_concentration_flag_75", "Knowledge concentration flag (≥75 %)"),
    ("bus_factor_estimate", "Bus factor estimate"),
    ("commit_count_12m", "Commits (12 months)"),
    ("recency_days", "Days since last commit"),
    ("file_age_days", "File age (days)"),
    ("git_metrics_status", "Git metrics status"),
]


# ============================================================================
# SYSTEM PROMPT – GIT-AUGMENTED ASSESSMENT RUBRIC
# ============================================================================

SYSTEM_PROMPT = """You are a code quality assessor for a research study comparing LLM-based evaluation against SonarQube static analysis. Your task is to analyze source code *together with git history metrics* and produce structured metrics that can be empirically compared with SonarQube outputs.

CRITICAL INSTRUCTIONS:
1. Analyze ONLY the provided source code and git metrics - do not assume external context
2. Use the git history metrics to *inform* your assessment (see GIT METRICS GUIDANCE below)
3. Provide numeric scores following the exact rubrics below
4. Return ONLY valid JSON - no markdown, no explanations outside the JSON structure
5. Be consistent and calibrated - similar code quality should yield similar scores

OUTPUT SCHEMA (return exactly this structure):
{
  "cyclomatic_complexity": <integer, estimated total for file>,
  "cognitive_complexity": <integer, estimated cognitive load>,
  "lines_of_code": <integer, non-blank non-comment lines>,
  "comment_density": <float 0.0-100.0, percentage of comment lines>,
  "code_smells_count": <integer, number of code smell instances>,
  "code_smells_list": ["<smell_type>: <brief description>", ...],
  "duplication_estimate": <float 0.0-100.0, estimated % duplicated logic>,
  "maintainability_rating": <integer 1-5, where 1=best, 5=worst>,
  "reliability_issues": <integer, count of potential bugs/errors>,
  "security_issues": <integer, count of security concerns>,
  "technical_debt_minutes": <integer, estimated fix time>,
  "readability_score": <integer 1-10, where 10=excellent>,
  "assessment_confidence": <float 0.0-1.0, your confidence in this assessment>,
  "primary_concerns": ["<issue 1>", "<issue 2>", ...],
  "summary": "<1-2 sentence overall assessment>"
}

SCORING RUBRICS:

## Cyclomatic Complexity (estimated)
Count decision points: if, else, elif/else if, for, while, case/switch cases, catch, &&, ||, ternary operators, null coalescing.
- Simple utility functions: 1-5 per function
- Moderate logic: 6-15 per function
- Complex functions: 16-30 per function
- Highly complex: 30+ per function
Sum all functions in the file.

## Cognitive Complexity (SonarQube-aligned)
Beyond cyclomatic, penalize:
- Nesting depth (each level adds to inner constructs)
- Recursion
- Non-linear flow (goto, break to label, continue)
- Cognitive jumps (interrupting sequences)
Apply multiplier based on nesting level.

## Code Smells (count and categorize)
Identify instances of:
- Long Method (>30 lines of logic)
- Large Class (>300 lines)
- Long Parameter List (>5 parameters)
- Duplicate Code blocks
- Dead Code (unreachable/unused)
- Magic Numbers/Strings
- Deep Nesting (>4 levels)
- God Class/Object (does too much)
- Feature Envy (method uses other class more than its own)
- Primitive Obsession
- Inappropriate Intimacy
- Refused Bequest
- Comments describing "what" not "why"

## Maintainability Rating (SonarQube A-E mapped to 1-5)
1 (A): Technical debt ratio <5%, clean and modular
2 (B): Technical debt ratio 5-10%, minor issues
3 (C): Technical debt ratio 10-20%, notable debt
4 (D): Technical debt ratio 20-50%, significant refactoring needed
5 (E): Technical debt ratio >50%, major rewrite may be needed

When git metrics show high churn combined with knowledge concentration or a low bus factor, consider nudging the maintainability rating upward (worse) because the code is harder to maintain *in practice* even if the code structure itself is clean.

## Technical Debt Minutes
Estimate time to fix all identified issues:
- Simple smell: 5-15 minutes
- Medium smell: 15-45 minutes
- Complex smell: 45-120 minutes
- Architectural issue: 120-480 minutes
Sum all estimated fix times.

## Reliability Issues
Count potential runtime problems:
- Null/undefined access risks
- Unhandled exceptions
- Resource leaks
- Race conditions
- Type mismatches
- Out-of-bounds access

## Security Issues
Count security concerns:
- Hardcoded credentials
- SQL injection risks
- XSS vulnerabilities
- Path traversal
- Insecure deserialization
- Sensitive data exposure

## Readability Score (1-10)
1-2: Nearly unreadable - cryptic names, no structure
3-4: Poor - minimal documentation, unclear flow
5-6: Acceptable - some documentation, follows basic conventions
7-8: Good - clear naming, proper structure, helpful comments
9-10: Excellent - self-documenting, exemplary clarity

## Assessment Confidence
0.0-0.3: Limited visibility (truncated, unfamiliar language)
0.4-0.6: Partial confidence (some uncertainty in estimates)
0.7-0.8: Good confidence (clear patterns, standard code)
0.9-1.0: High confidence (straightforward, complete visibility)

If git metrics are unavailable (status "missing"), reduce your confidence slightly since you lack the code-evolution context.

CALIBRATION NOTES:
- For truncated files, extrapolate proportionally but reduce confidence
- Empty or near-empty files: minimal complexity, high maintainability
- Configuration files: typically low complexity unless procedural
- Be conservative with security/reliability issues - only flag clear patterns

GIT METRICS GUIDANCE:
You will receive a block of git history metrics for the file being analyzed.
Use them as *contextual signals* that complement your code-level analysis:
- High churn (many lines added/deleted) may indicate unstable or frequently patched code – look more closely for complexity and debt.
- A single contributor or high knowledge concentration increases maintenance risk (bus factor concern).
- Low bus factor (≤2) suggests the file's knowledge is concentrated in very few people.
- High recency_days (>180) may mean the code is stale and could hide latent issues no one has addressed.
- A young file (low file_age_days) with many commits may still be in active development – be lenient on structure.
- If git_metrics_status is "missing", ignore git signals and rely solely on the code.
These signals should influence your *summary*, *primary_concerns*, *assessment_confidence*, and – where relevant – *maintainability_rating* and *technical_debt_minutes*.  They should NOT change raw code metrics such as cyclomatic_complexity or lines_of_code, which must be derived from the source code alone.
""".strip()

# Compute prompt hash for reproducibility tracking
PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
PROMPT_VERSION = f"v1.0-git-sha256:{PROMPT_HASH[:12]}"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FileAnalysis:
    """Container for file metadata and analysis preparation."""
    repo: str
    file_path: str
    absolute_path: Path
    language: str = ""
    file_size_bytes: int = 0
    is_truncated: bool = False
    original_chars: int = 0
    analyzed_chars: int = 0
    skip_reason: str = ""

    @property
    def should_analyze(self) -> bool:
        return not self.skip_reason


@dataclass
class LLMResult:
    """Container for LLM assessment results."""
    repo: str
    file_path: str

    # SonarQube-comparable metrics
    cyclomatic_complexity: int | None = None
    cognitive_complexity: int | None = None
    lines_of_code: int | None = None
    comment_density: float | None = None
    code_smells_count: int | None = None
    code_smells_list: list[str] = field(default_factory=list)
    duplication_estimate: float | None = None
    maintainability_rating: int | None = None
    reliability_issues: int | None = None
    security_issues: int | None = None
    technical_debt_minutes: int | None = None

    # LLM-specific metrics
    readability_score: int | None = None
    assessment_confidence: float | None = None
    primary_concerns: list[str] = field(default_factory=list)
    summary: str = ""

    # Metadata
    llm_model: str = ""
    llm_provider: str = ""
    is_truncated: bool = False
    original_chars: int = 0
    analyzed_chars: int = 0
    success: bool = False
    error: str = ""
    raw_response: str = ""
    prompt_version: str = PROMPT_VERSION
    analysis_date_utc: str = ""
    repo_head_sha: str | None = None
    git_augmented: bool = True


# ============================================================================
# FILE HANDLING
# ============================================================================

def should_analyze_file(file_path: Path, rel_path: str | None = None) -> tuple[bool, str]:
    """
    Determine if a file should be analyzed.

    Returns:
        (should_analyze, skip_reason) - skip_reason empty if should analyze
    """
    if file_path.suffix.lower() not in ANALYZABLE_EXTENSIONS:
        return False, f"extension '{file_path.suffix}' not analyzable"

    filter_path = rel_path or str(file_path)
    if not is_included_path(filter_path):
        return False, "not included by Sonar filters"
    if is_excluded_path(filter_path):
        return False, "matches Sonar exclusions"

    if not file_path.exists():
        return False, "file does not exist"

    try:
        size = file_path.stat().st_size
    except OSError as e:
        return False, f"cannot stat file: {e}"

    if size == 0:
        return False, "empty file"

    return True, ""


def read_and_truncate_code(file_path: Path) -> tuple[str, bool, int, int]:
    """
    Read source code and truncate if necessary to fit context window.

    Returns:
        (code, is_truncated, original_chars, analyzed_chars)
    """
    try:
        code = file_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return "", False, 0, 0

    original_chars = len(code)

    if MAX_CODE_CHARS is None:
        return code, False, original_chars, original_chars

    if original_chars <= MAX_CODE_CHARS:
        return code, False, original_chars, original_chars

    head = code[:HEAD_CHARS]
    tail = code[-TAIL_CHARS:] if TAIL_CHARS > 0 else ""

    if tail:
        truncated = f"{head}\n\n/* ... [{original_chars - HEAD_CHARS - TAIL_CHARS} characters truncated] ... */\n\n{tail}"
    else:
        truncated = head

    return truncated, True, original_chars, len(truncated)


def prepare_file_analysis(repo: str, file_path: str, base_dir: Path) -> FileAnalysis:
    """Prepare a file for analysis, checking eligibility and reading content."""
    rel_path = file_path.replace('\\', '/')
    if Path(rel_path).is_absolute():
        abs_path = Path(rel_path)
    else:
        abs_path = (base_dir / repo / rel_path).resolve()

    analysis = FileAnalysis(
        repo=repo,
        file_path=rel_path,
        absolute_path=abs_path,
        language=detect_language(abs_path),
    )

    should_analyze, skip_reason = should_analyze_file(abs_path, rel_path)
    if not should_analyze:
        analysis.skip_reason = skip_reason
        return analysis

    analysis.file_size_bytes = abs_path.stat().st_size
    return analysis


# ============================================================================
# GIT METRICS LOADING
# ============================================================================

def load_git_metrics(git_csv: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Load git_metrics.csv into a lookup dict keyed by ``(repo, file_path)``.

    Only the columns listed in :data:`GIT_PROMPT_FIELDS` (plus ``repo`` and
    ``file_path``) are retained.  Missing values are kept as ``None``.
    """
    if not git_csv.exists():
        logger.warning(f"Git metrics file not found: {git_csv}")
        return {}

    try:
        df = pd.read_csv(git_csv)
    except Exception as e:
        logger.error(f"Failed to read git metrics CSV: {e}")
        return {}

    if df.empty:
        return {}

    required = {"repo", "file_path"}
    if not required.issubset(df.columns):
        logger.error("Git metrics CSV missing 'repo' and/or 'file_path' columns")
        return {}

    df["file_path"] = df["file_path"].astype(str).str.replace("\\", "/", regex=False)

    keep_cols = list(required) + [col for col, _ in GIT_PROMPT_FIELDS if col in df.columns]
    df = df[keep_cols]

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for record in df.to_dict("records"):
        key = (record["repo"], record["file_path"])
        # Convert NaN → None for cleaner handling
        clean = {}
        for col, _ in GIT_PROMPT_FIELDS:
            val = record.get(col)
            if pd.isna(val) if not isinstance(val, str) else False:
                clean[col] = None
            else:
                clean[col] = val
        lookup[key] = clean

    logger.info(f"Loaded git metrics for {len(lookup)} files")
    return lookup


def format_git_block(git_data: dict[str, Any] | None) -> str:
    """
    Format git metrics into a human-readable block for the LLM prompt.

    Returns an empty string when there is no usable data.
    """
    if git_data is None:
        return "\n\nGIT HISTORY METRICS:\nGit metrics are unavailable for this file."

    status = git_data.get("git_metrics_status")
    if status and str(status).lower() != "ok":
        return "\n\nGIT HISTORY METRICS:\nGit metrics are unavailable for this file (status: {}).".format(status)

    lines = ["\n\nGIT HISTORY METRICS:"]
    for col, label in GIT_PROMPT_FIELDS:
        val = git_data.get(col)
        display = "N/A" if val is None else val
        # Format dominant_author_share as percentage
        if col == "dominant_author_share" and val is not None:
            try:
                display = f"{float(val):.1%}"
            except (ValueError, TypeError):
                pass
        lines.append(f"- {label}: {display}")

    return "\n".join(lines)


# ============================================================================
# RESPONSE PARSING AND VALIDATION
# ============================================================================

def clean_json_response(content: str) -> str:
    """Remove markdown fences and extract JSON from LLM response."""
    content = content.strip()

    if content.startswith("```"):
        lines = content.split('\n')
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = '\n'.join(lines).strip()

    if content.lower().startswith("json"):
        content = content[4:].strip()

    return content


def _try_parse_json(content: str) -> dict[str, Any] | None:
    if not content:
        return None
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for i, char in enumerate(content):
        if char == "{":
            try:
                decoder = json.JSONDecoder()
                parsed, _ = decoder.raw_decode(content[i:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


def _try_parse_literal(content: str) -> dict[str, Any] | None:
    if not content:
        return None
    try:
        parsed = ast.literal_eval(content)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return content[start : end + 1]


def repair_json_response(content: str) -> str:
    """Attempt minimal repairs for near-JSON output."""
    extracted = _extract_json_object(content)
    if not extracted:
        return ""
    cleaned = re.sub(r",\s*([}\]])", r"\1", extracted)
    return cleaned


def parse_llm_response(content: str) -> tuple[dict[str, Any] | None, str]:
    """
    Parse and validate LLM JSON response.

    Returns:
        (parsed_dict, error_message) - error_message empty on success
    """
    content = clean_json_response(content)

    parsed = _try_parse_json(content)
    if parsed:
        return parsed, ""

    parsed = _try_parse_literal(content)
    if parsed:
        return parsed, ""

    repaired = repair_json_response(content)
    parsed = _try_parse_json(repaired)
    if parsed:
        return parsed, ""

    parsed = _try_parse_literal(repaired)
    if parsed:
        return parsed, ""

    return None, "Could not parse JSON from response"


def validate_and_normalize(parsed: dict[str, Any]) -> tuple[LLMResult, list[str]]:
    """
    Validate parsed response and normalize to LLMResult.

    Returns:
        (result, warnings) - result has fields set, warnings list issues
    """
    result = LLMResult(repo="", file_path="")
    warnings = []

    int_fields = [
        ("cyclomatic_complexity", 0, 10000),
        ("cognitive_complexity", 0, 10000),
        ("lines_of_code", 0, 100000),
        ("code_smells_count", 0, 1000),
        ("maintainability_rating", 1, 5),
        ("reliability_issues", 0, 1000),
        ("security_issues", 0, 1000),
        ("technical_debt_minutes", 0, 100000),
        ("readability_score", 1, 10),
    ]

    for field_name, min_val, max_val in int_fields:
        value = parsed.get(field_name)
        if value is None:
            warnings.append(f"missing {field_name}")
            continue
        try:
            int_val = int(value)
            if min_val <= int_val <= max_val:
                setattr(result, field_name, int_val)
            else:
                warnings.append(f"{field_name}={value} out of range [{min_val}, {max_val}]")
        except (ValueError, TypeError):
            warnings.append(f"{field_name}={value} not a valid integer")

    float_fields = [
        ("comment_density", 0.0, 100.0),
        ("duplication_estimate", 0.0, 100.0),
        ("assessment_confidence", 0.0, 1.0),
    ]

    for field_name, min_val, max_val in float_fields:
        value = parsed.get(field_name)
        if value is None:
            warnings.append(f"missing {field_name}")
            continue
        try:
            float_val = float(value)
            if min_val <= float_val <= max_val:
                setattr(result, field_name, round(float_val, 2))
            else:
                warnings.append(f"{field_name}={value} out of range [{min_val}, {max_val}]")
        except (ValueError, TypeError):
            warnings.append(f"{field_name}={value} not a valid float")

    code_smells = parsed.get("code_smells_list", [])
    if isinstance(code_smells, list):
        result.code_smells_list = [str(s) for s in code_smells if s]

    concerns = parsed.get("primary_concerns", [])
    if isinstance(concerns, list):
        result.primary_concerns = [str(c) for c in concerns if c]

    summary = parsed.get("summary", "")
    if isinstance(summary, str):
        result.summary = summary.strip()

    return result, warnings


# ============================================================================
# CORE ANALYSIS FUNCTION
# ============================================================================

def analyze_file(
    client: LLMClient,
    file_analysis: FileAnalysis,
    model: str,
    git_data: dict[str, Any] | None = None,
    max_retries: int = 2,
) -> LLMResult:
    """
    Analyze a single file using the LLM, augmented with git metrics.

    Args:
        client: Configured LLM client
        file_analysis: Prepared file analysis metadata
        model: Model name to use
        git_data: Dict of git history metrics for this file, or None
        max_retries: Number of retries on parse failures

    Returns:
        LLMResult with assessment data
    """
    result = LLMResult(
        repo=file_analysis.repo,
        file_path=file_analysis.file_path,
        llm_model=model,
        llm_provider=config.LLM_PROVIDER,
        prompt_version=PROMPT_VERSION,
        analysis_date_utc=config.now_utc_iso(),
    )

    if not file_analysis.should_analyze:
        result.error = f"Skipped: {file_analysis.skip_reason}"
        logger.debug(f"Skipping {file_analysis.file_path}: {file_analysis.skip_reason}")
        return result

    code, is_truncated, original_chars, analyzed_chars = read_and_truncate_code(
        file_analysis.absolute_path
    )

    if not code:
        result.error = "Failed to read file"
        return result

    result.is_truncated = is_truncated
    result.original_chars = original_chars
    result.analyzed_chars = analyzed_chars

    # ---- Build user message with code + git context ----
    truncation_note = ""
    if is_truncated:
        truncation_note = (
            f"\n\nNOTE: This file has been truncated from {original_chars} to "
            f"{analyzed_chars} characters. Adjust confidence accordingly and "
            f"extrapolate metrics proportionally where appropriate."
        )

    git_block = format_git_block(git_data)

    user_message = (
        f"Analyze the following {file_analysis.language} source code and return the JSON assessment."
        f"{truncation_note}\n\n"
        f"```{file_analysis.language}\n{code}\n```"
        f"{git_block}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # ---- Call LLM with retries ----
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = client.complete(model=model, messages=messages)
            raw_content = client.extract_content(response)
            result.raw_response = raw_content

            parsed, parse_error = parse_llm_response(raw_content)
            if parse_error:
                try:
                    with PARSE_FAILURE_LOG.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "repo": file_analysis.repo,
                                    "file_path": file_analysis.file_path,
                                    "error": parse_error,
                                    "raw_response": raw_content,
                                    "model": model,
                                    "analysis_date_utc": result.analysis_date_utc,
                                }
                            )
                            + "\n"
                        )
                except OSError:
                    pass
                last_error = parse_error
                logger.warning(
                    f"Parse attempt {attempt + 1} failed for {file_analysis.file_path}: {parse_error}"
                )
                continue

            validated, warnings = validate_and_normalize(parsed)

            for fld in [
                'cyclomatic_complexity', 'cognitive_complexity', 'lines_of_code',
                'comment_density', 'code_smells_count', 'code_smells_list',
                'duplication_estimate', 'maintainability_rating', 'reliability_issues',
                'security_issues', 'technical_debt_minutes', 'readability_score',
                'assessment_confidence', 'primary_concerns', 'summary'
            ]:
                setattr(result, fld, getattr(validated, fld))

            if warnings:
                logger.debug(f"Validation warnings for {file_analysis.file_path}: {warnings}")

            critical_fields = ['cyclomatic_complexity', 'maintainability_rating', 'readability_score']
            missing_critical = [f for f in critical_fields if getattr(result, f) is None]

            if missing_critical:
                last_error = f"Missing critical fields: {missing_critical}"
                if attempt < max_retries:
                    logger.warning(f"Retry {attempt + 1} for {file_analysis.file_path}: {last_error}")
                    continue

            result.success = True
            return result

        except LLMError as e:
            last_error = str(e)
            logger.warning(f"LLM error for {file_analysis.file_path}: {e}")
            if attempt < max_retries:
                continue
            break
        except Exception as e:
            last_error = str(e)
            logger.error(f"Unexpected error for {file_analysis.file_path}: {e}")
            break

    result.error = last_error
    return result


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def result_to_row(result: LLMResult) -> dict[str, Any]:
    """Convert LLMResult to a dictionary suitable for CSV output."""
    return {
        # Identifiers
        "repo": result.repo,
        "file_path": result.file_path,

        # SonarQube-comparable metrics (prefixed for clarity in merged dataset)
        "llm_cyclomatic_complexity": result.cyclomatic_complexity,
        "llm_cognitive_complexity": result.cognitive_complexity,
        "llm_ncloc": result.lines_of_code,
        "llm_comment_density": result.comment_density,
        "llm_code_smells": result.code_smells_count,
        "llm_duplicated_lines_density": result.duplication_estimate,
        "llm_maintainability_rating": result.maintainability_rating,
        "llm_reliability_issues": result.reliability_issues,
        "llm_security_issues": result.security_issues,
        "llm_technical_debt_minutes": result.technical_debt_minutes,

        # LLM-specific metrics
        "llm_readability_score": result.readability_score,
        "llm_assessment_confidence": result.assessment_confidence,
        "llm_code_smells_list": json.dumps(result.code_smells_list) if result.code_smells_list else "",
        "llm_primary_concerns": json.dumps(result.primary_concerns) if result.primary_concerns else "",
        "llm_summary": result.summary,

        # Metadata
        "llm_model": result.llm_model,
        "llm_provider": result.llm_provider,
        "llm_is_truncated": result.is_truncated,
        "llm_original_chars": result.original_chars,
        "llm_analyzed_chars": result.analyzed_chars,
        "llm_success": result.success,
        "llm_error": result.error,
        "llm_prompt_version": result.prompt_version,
        "llm_git_augmented": result.git_augmented,
        "analysis_date_utc": result.analysis_date_utc,
        "repo_head_sha": result.repo_head_sha,
    }


def load_existing_results(output_path: Path, model: str) -> set[tuple[str, str]]:
    """Load set of (repo, file_path) already processed for a model."""
    if not output_path.exists():
        return set()

    try:
        df = pd.read_csv(output_path)
    except (pd.errors.EmptyDataError, Exception):
        return set()

    if df.empty:
        return set()

    mask = df["llm_success"].astype(str).str.lower().isin({"true", "1"})
    if "llm_model" in df.columns and model:
        mask &= df["llm_model"] == model

    df = df[mask]

    if not {"repo", "file_path"}.issubset(df.columns):
        return set()

    return set(zip(df["repo"], df["file_path"]))


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_llm_judge_git(
    input_csv: Path | None = None,
    output_path: Path | None = None,
    resume: bool = True,
) -> Path:
    """
    Run the git-augmented LLM code assessment.

    This variant reads the file list from ``git_metrics.csv`` (so only files
    that have git data are evaluated) and injects the git history metrics into
    each LLM prompt alongside the source code.

    Args:
        input_csv: Path to git_metrics.csv (default: ``config.git_metrics_path()``)
        output_path: Custom output path
            (default: ``data/results/llm/llm_git_metrics_<model>_runNNN.csv``)
        resume: If True, skip files already successfully analyzed

    Returns:
        Path to the output CSV file
    """
    # Default input: git_metrics.csv
    if input_csv is None:
        input_csv = config.git_metrics_path()

    logger.info(f"Loading file list from {input_csv}")
    try:
        input_df = pd.read_csv(input_csv)
    except Exception as e:
        raise RuntimeError(f"Failed to read input CSV: {e}")

    if "repo" not in input_df.columns or "file_path" not in input_df.columns:
        raise ValueError("Input CSV must contain 'repo' and 'file_path' columns")

    # Deduplicate and normalise paths
    input_df = input_df.drop_duplicates(subset=["repo", "file_path"])
    input_df["file_path"] = input_df["file_path"].astype(str).str.replace("\\", "/", regex=False)

    # Build git-metrics lookup from the same CSV
    git_lookup = load_git_metrics(input_csv)

    # Model configuration
    model = config.LLM_MODEL
    if not model:
        raise RuntimeError("LLM_MODEL must be configured")

    # Resolve output path
    if output_path is None:
        output_path = config.llm_git_metrics_path(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume handling
    existing: set[tuple[str, str]] = set()
    if resume and config.LLM_RESUME:
        existing = load_existing_results(output_path, model)
        if existing:
            logger.info(f"Resuming: {len(existing)} files already processed")

    file_records = input_df.to_dict("records")
    to_process = [
        r for r in file_records
        if (r["repo"], r["file_path"]) not in existing
    ]

    logger.info(f"Processing {len(to_process)} files with model {model} (git-augmented)")

    if not to_process:
        logger.info("No files to process")
        return output_path

    # Initialize LLM client
    client = LLMClient(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_PROVIDER)

    # SHA cache
    sha_cache: dict[str, str | None] = {}

    def get_repo_sha(repo: str) -> str | None:
        if repo not in sha_cache:
            sha_cache[repo] = config.git_head_sha(config.RAW_REPOS_DIR / repo)
        return sha_cache[repo]

    # Process files
    results: list[dict] = []
    write_every = max(config.LLM_WRITE_EVERY, 10)
    header_written = output_path.exists() and output_path.stat().st_size > 0

    for idx, record in enumerate(to_process, 1):
        repo = record["repo"]
        file_path = record["file_path"]

        file_analysis = prepare_file_analysis(repo, file_path, config.RAW_REPOS_DIR)

        # Look up git metrics for this file
        git_data = git_lookup.get((repo, file_path))

        result = analyze_file(
            client=client,
            file_analysis=file_analysis,
            model=model,
            git_data=git_data,
            max_retries=config.LLM_MAX_RETRIES,
        )
        result.repo_head_sha = get_repo_sha(repo)

        status = "✓" if result.success else "✗"
        if result.success:
            logger.info(f"[{idx}/{len(to_process)}] {status} {repo}/{file_path}")
        else:
            logger.warning(f"[{idx}/{len(to_process)}] {status} {repo}/{file_path}: {result.error}")

        if result.success:
            results.append(result_to_row(result))

        # Periodic flush
        if len(results) >= write_every:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False, mode="a", header=not header_written)
            header_written = True
            logger.debug(f"Flushed {len(results)} results to {output_path}")
            results.clear()

    # Write remaining results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, mode="a", header=not header_written)

    # Sort output
    if config.LLM_SORT_OUTPUT and output_path.exists():
        try:
            final_df = pd.read_csv(output_path)
            if not final_df.empty and {"repo", "file_path"}.issubset(final_df.columns):
                final_df.sort_values(["repo", "file_path"], inplace=True)
                final_df.to_csv(output_path, index=False)
        except Exception as e:
            logger.warning(f"Could not sort output: {e}")

    logger.info(f"Wrote git-augmented LLM metrics to {output_path}")
    return output_path


def main() -> None:
    """Command-line entry point."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(asctime)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    git_metrics_csv = config.git_metrics_path()

    if not git_metrics_csv.exists():
        logger.error(f"Input file not found: {git_metrics_csv}")
        logger.info("Run miner.py first to generate git_metrics.csv")
        return

    run_llm_judge_git(git_metrics_csv)


if __name__ == "__main__":
    main()
