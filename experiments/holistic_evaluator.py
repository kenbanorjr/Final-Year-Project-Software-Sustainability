"""
Holistic Software Sustainability Evaluator

Combines static analysis metrics (SonarQube) with project evolution and
social/process metrics (Git) to produce structured sustainability assessments.

This evaluator synthesizes technical debt signals with social fragility indicators
to provide a comprehensive view of long-term code sustainability.
"""

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from pipeline.configs import config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = config.LLM_MODEL  # Use configured model (local or OpenAI)
TEMPERATURE = 0.2  # Low temp for consistency
MAX_TOKENS = 2000
MAX_RETRIES = 3
RETRY_DELAY = 2.0
# Store holistic outputs under data/results/holistic.
OUTPUT_DIR = config.RESULTS_HOLISTIC_DIR

# Code excerpt controls (token-safe, configurable via env)
CODE_EXCERPT_MAX_CHARS = int(os.environ.get("CODE_EXCERPT_MAX_CHARS", "1800"))
CODE_EXCERPT_HEAD_LINES = int(os.environ.get("CODE_EXCERPT_HEAD_LINES", "120"))
CODE_EXCERPT_TAIL_LINES = int(os.environ.get("CODE_EXCERPT_TAIL_LINES", "0"))
CODE_EXCERPT_MAX_BYTES = int(os.environ.get("CODE_EXCERPT_MAX_BYTES", "200000"))

# ---------------------------------------------------------------------------
# Numeric Thresholds for One-Hot Flags (deterministic, not LLM-derived)
# ---------------------------------------------------------------------------

THRESHOLD_LARGE_FILE_NCLOC = 500
THRESHOLD_HIGH_COGNITIVE_COMPLEXITY = 15
THRESHOLD_HIGH_RATING = 4  # Rating >= 4 means D or E (poor)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a "Holistic Software Sustainability Evaluator".

Goal:
Given (a) optional code excerpt, (b) static analysis metrics (Sonar-like), and (c) project evolution + social/process metrics (Git/miner), produce a structured sustainability assessment that synthesizes technical and social risks.

You must:

* Use the provided metrics explicitly (do not invent numbers).
* Explain which signals drive your conclusions ("top_drivers").
* Be conservative: use "Unknown" only when required inputs for that judgment are genuinely missing/null.
* Output ONLY valid JSON that matches the schema below. No extra text, no markdown fences.

Definitions:

* maintainability_risk: likelihood the code becomes hard/costly to change (Low/Med/High/Unknown).
* sustainability_risk: likelihood the code/project becomes difficult to sustain long-term due to a combination of technical debt + process/social fragility (Low/Med/High/Unknown).
* technical_flags: short phrases about technical risks (e.g., "high complexity", "low comment density", "duplication present", "high debt").
* social_sustainability_flags: short phrases about social/process risks (e.g., "high churn", "many contributors", "knowledge concentration", "bus factor risk", "inactive/low recency").
* top_drivers: the 3–6 strongest drivers from the provided inputs (must cite metric names, and if possible values).
* recommended_actions: 3–8 concrete actions, prioritized, that connect directly to the flags/drivers.
* confidence: integer 1–5 (1 = very uncertain, 5 = very confident).

=== MANDATORY NUMERIC THRESHOLDS (DO NOT DEVIATE) ===

Comment density classification (based on comment_lines_density):
- comment_lines_density < 10  -> "low comment density"
- 10 <= comment_lines_density <= 25 -> "moderate comment density"
- comment_lines_density > 25 -> "good comment density" (NEVER label as "low")

Contributor classification (based on unique_authors_12m):
- unique_authors_12m == 1 -> "single contributor"
- unique_authors_12m == 2 or 3 -> "few contributors"
- unique_authors_12m >= 4 and <= 7 -> "moderate contributors"
- unique_authors_12m >= 8 -> "many contributors" (NEVER use for < 8)

Recency classification (based on recency_days):
- recency_days is null/missing -> "unknown recency" ONLY (do NOT infer inactivity)
- recency_days <= 90 -> "active recency" or "healthy recency"
- recency_days > 90 and <= 180 -> "stale recency"
- recency_days > 180 -> "inactive/low recency"

Bus factor classification (based on bus_factor_estimate):
- bus_factor_estimate is null/missing -> "unknown bus factor" ONLY (NEVER output "bus factor risk")
- bus_factor_estimate <= 2 -> "bus factor risk"
- bus_factor_estimate >= 3 -> no bus factor flag needed

CRITICAL: Your flags MUST NOT contradict the numeric values. If comment_lines_density=41, you MUST NOT say "low comment density".

=== END THRESHOLDS ===

Missingness rules:

* If a metric is null/NA, treat as Unknown (do NOT assume 0) unless the input explicitly says "absence implies 0".
* If duplication is null and the input states "Sonar reports duplication only when present", treat null as "no duplication detected" BUT note this assumption in "notes".
* If git_metrics_status is "missing", all git-related flags should use "unknown" variants.

Maintainability decision rule (strict):

* Core maintainability inputs are:
  - ncloc (input_ncloc)
  - code_smells (input_code_smells)
  - complexity (input_complexity)
  - cognitive_complexity (input_cognitive_complexity)
  - sqale_index (input_technical_debt_minutes)
  - sqale_rating (input_maintainability_rating)
* If ANY core maintainability input above is present (non-null), you MUST output maintainability_risk as Low, Med, or High (NOT Unknown).
* maintainability_risk may be Unknown ONLY when ALL core maintainability inputs are missing/null.
* Do NOT output maintainability_risk="Unknown" when maintainability metrics are provided.

Robustness:
You are one run of k=3 repeats. Do not reference other runs. Just give your best single-run assessment.

JSON schema (output exactly this structure, no markdown fences):
{
  "repo_id": string,
  "file_path": string,
  "language": string | null,
  "commit_sha": string | null,

  "maintainability_risk": "Low" | "Med" | "High" | "Unknown",
  "sustainability_risk": "Low" | "Med" | "High" | "Unknown",
  "confidence": 1 | 2 | 3 | 4 | 5,

  "technical_flags": [string, ...],
  "social_sustainability_flags": [string, ...],

  "top_drivers": [
    {
      "driver": string,
      "evidence": [string, ...]
    }
  ],

  "recommended_actions": [
    {
      "priority": 1 | 2 | 3 | 4 | 5,
      "action": string,
      "rationale": string
    }
  ],

  "notes": [string, ...]
}

Now produce the JSON assessment for the given INPUT."""


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SonarMetrics:
    """Static analysis metrics from SonarQube."""
    complexity: float | None = None
    cognitive_complexity: float | None = None
    ncloc: float | None = None
    comment_lines_density: float | None = None
    sqale_index: float | None = None  # Technical debt in minutes
    sqale_rating: int | None = None   # 1=A to 5=E
    code_smells: int | None = None
    bugs: int | None = None
    vulnerabilities: int | None = None
    duplicated_lines_density: float | None = None
    duplicated_blocks: int | None = None
    reliability_rating: int | None = None  # 1=A to 5=E
    security_rating: int | None = None     # 1=A to 5=E
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, preserving None as null for JSON."""
        return {k: v for k, v in asdict(self).items()}


@dataclass
class GitMetrics:
    """Social/process metrics from Git history."""
    churn_12m: int | None = None           # Lines added + deleted
    unique_authors_12m: int | None = None  # Distinct contributors
    dominant_author_share: float | None = None  # 0-1, share of top author
    single_contributor_flag: bool | None = None
    commit_count_12m: int | None = None
    recency_days: int | None = None        # Days since last commit
    bus_factor_estimate: int | None = None # Approx. bus factor
    file_age_days: int | None = None       # Days since file creation
    git_metrics_status: str | None = None  # "ok" or "missing"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, preserving None as null for JSON."""
        return {k: v for k, v in asdict(self).items()}


@dataclass
class EvaluationInput:
    """Complete input for holistic evaluation."""
    repo_id: str
    file_path: str
    language: str | None = None
    commit_sha: str | None = None
    code_excerpt: str | None = None
    sonar_metrics: SonarMetrics | None = None
    git_metrics: GitMetrics | None = None
    
    def to_prompt(self) -> str:
        """Format as INPUT block for the LLM."""
        lines = [
            "INPUT:",
            f"* repo_id: {self.repo_id}",
            f"* file_path: {self.file_path}",
            f"* language: {self.language or 'unknown'}",
            f"* commit_sha: {self.commit_sha or 'unknown'}",
        ]
        
        if self.code_excerpt:
            # Truncate if too long
            excerpt = self.code_excerpt[:2000] + "..." if len(self.code_excerpt) > 2000 else self.code_excerpt
            lines.append(f"* code_excerpt:\n```\n{excerpt}\n```")
        else:
            lines.append("* code_excerpt: (not provided)")
        
        if self.sonar_metrics:
            lines.append(f"* sonar_metrics: {json.dumps(self.sonar_metrics.to_dict())}")
            lines.append("  (Note: Sonar reports duplication only when present; null duplication = no duplication detected)")
        else:
            lines.append("* sonar_metrics: (not provided)")
        
        if self.git_metrics:
            lines.append(f"* git_metrics: {json.dumps(self.git_metrics.to_dict())}")
        else:
            lines.append("* git_metrics: (not provided)")
        
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation Result Schema
# ---------------------------------------------------------------------------

@dataclass
class TopDriver:
    driver: str
    evidence: list[str]


@dataclass
class RecommendedAction:
    priority: int
    action: str
    rationale: str


@dataclass
class HolisticAssessment:
    """Structured output from the evaluator."""
    repo_id: str
    file_path: str
    language: str | None
    commit_sha: str | None
    
    maintainability_risk: str  # Low/Med/High/Unknown
    sustainability_risk: str   # Low/Med/High/Unknown
    confidence: int            # 1-5
    
    technical_flags: list[str]
    social_sustainability_flags: list[str]
    
    top_drivers: list[TopDriver]
    recommended_actions: list[RecommendedAction]
    
    notes: list[str]
    
    # Metadata
    model: str = ""
    latency_ms: int = 0
    raw_response: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "repo_id": self.repo_id,
            "file_path": self.file_path,
            "language": self.language,
            "commit_sha": self.commit_sha,
            "maintainability_risk": self.maintainability_risk,
            "sustainability_risk": self.sustainability_risk,
            "confidence": self.confidence,
            "technical_flags": self.technical_flags,
            "social_sustainability_flags": self.social_sustainability_flags,
            "top_drivers": [{"driver": d.driver, "evidence": d.evidence} for d in self.top_drivers],
            "recommended_actions": [
                {"priority": a.priority, "action": a.action, "rationale": a.rationale}
                for a in self.recommended_actions
            ],
            "notes": self.notes,
            "model": self.model,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Robust JSON Extraction
# ---------------------------------------------------------------------------

def _extract_json_robustly(raw: str) -> str:
    """
    Extract JSON from LLM response, handling:
    - Markdown code fences (```json ... ```)
    - Leading/trailing text
    - Whitespace issues
    
    Returns the extracted JSON string (may still be invalid JSON).
    """
    text = raw.strip()
    
    # Try to extract from markdown code fences first
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    
    # Find the first '{' and last '}' to extract the JSON object
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]
    
    return text


def _create_error_assessment(
    input_data: 'EvaluationInput',
    error_message: str,
    model: str,
    raw_response: str = "",
) -> 'HolisticAssessment':
    """Create a standardized error assessment when parsing/evaluation fails."""
    return HolisticAssessment(
        repo_id=input_data.repo_id,
        file_path=input_data.file_path,
        language=input_data.language,
        commit_sha=input_data.commit_sha,
        maintainability_risk="Unknown",
        sustainability_risk="Unknown",
        confidence=1,
        technical_flags=[],  # No flags for errors
        social_sustainability_flags=[],  # No flags for errors
        top_drivers=[],
        recommended_actions=[],
        notes=[f"Evaluation error: {error_message}"],
        model=model,
        latency_ms=0,  # Will be set to None in result_row
        raw_response=raw_response,
    )


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class HolisticEvaluator:
    """Evaluates code sustainability using LLM with combined metrics."""
    
    def __init__(self, model: str = MODEL, api_key: str | None = None):
        self.model = model
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        # Support both OpenAI and local (Ollama) providers
        if config.LLM_PROVIDER == "openai":
            self.client = OpenAI(api_key=api_key or config.LLM_API_KEY)
        else:
            # Local Ollama or compatible endpoint
            self.client = OpenAI(
                api_key=api_key or config.LLM_API_KEY or "ollama",
                base_url=f"{config.LLM_BASE_URL}/v1",
            )
    
    def evaluate(self, input_data: EvaluationInput) -> HolisticAssessment:
        """Run holistic sustainability evaluation."""
        prompt = input_data.to_prompt()
        last_raw = ""
        last_error = ""
        
        for attempt in range(MAX_RETRIES):
            raw = ""
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency_ms = int((time.time() - start) * 1000)
                
                raw = response.choices[0].message.content.strip()
                assessment = self._parse_response(raw, input_data, latency_ms)
                return assessment
                
            except Exception as e:
                if raw:
                    last_raw = raw
                last_error = _summarize_error_message(e)
                if attempt < MAX_RETRIES - 1:
                    print(f"  Attempt {attempt + 1} failed: {last_error}. Retrying...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    # Return clean error assessment (no evaluation_error in flags)
                    return _create_error_assessment(
                        input_data=input_data,
                        error_message=f"Failed after {MAX_RETRIES} attempts: {last_error}",
                        model=self.model,
                        raw_response=last_raw,
                    )
    
    def _parse_response(
        self, raw: str, input_data: EvaluationInput, latency_ms: int
    ) -> HolisticAssessment:
        """Parse LLM JSON response into structured assessment."""
        json_str = _extract_json_robustly(raw)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
        
        # Parse top_drivers
        top_drivers = []
        for d in data.get("top_drivers", []):
            if isinstance(d, dict):
                top_drivers.append(TopDriver(
                    driver=d.get("driver", ""),
                    evidence=d.get("evidence", [])
                ))
        
        # Parse recommended_actions
        actions = []
        for a in data.get("recommended_actions", []):
            if isinstance(a, dict):
                actions.append(RecommendedAction(
                    priority=a.get("priority", 5),
                    action=a.get("action", ""),
                    rationale=a.get("rationale", "")
                ))
        
        return HolisticAssessment(
            repo_id=data.get("repo_id", input_data.repo_id),
            file_path=data.get("file_path", input_data.file_path),
            language=data.get("language", input_data.language),
            commit_sha=data.get("commit_sha", input_data.commit_sha),
            maintainability_risk=data.get("maintainability_risk", "Unknown"),
            sustainability_risk=data.get("sustainability_risk", "Unknown"),
            confidence=data.get("confidence", 1),
            technical_flags=data.get("technical_flags", []),
            social_sustainability_flags=data.get("social_sustainability_flags", []),
            top_drivers=top_drivers,
            recommended_actions=actions,
            notes=data.get("notes", []),
            model=self.model,
            latency_ms=latency_ms,
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# Dataset Processing
# ---------------------------------------------------------------------------

def _normalize_merge_path(path: str) -> str:
    """Normalize file path for merge: forward slashes, strip leading ./"""
    if not path:
        return ""
    # Replace backslashes with forward slashes
    normalized = path.replace("\\", "/")
    # Strip leading ./
    while normalized.startswith("./"):
        normalized = normalized[2:]
    # Strip leading /
    normalized = normalized.lstrip("/")
    return normalized


def _load_existing_results(csv_path: Path) -> pd.DataFrame:
    """Load existing holistic CSV safely."""
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        existing_df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()
    except Exception as exc:
        print(f"⚠ Could not read existing holistic CSV ({csv_path}): {exc}")
        return pd.DataFrame()
    if existing_df is None:
        return pd.DataFrame()
    return existing_df


def _select_resume_run_id(existing_df: pd.DataFrame, resume_run_id: str | None) -> str | None:
    """Pick a run_id to resume from existing CSV rows."""
    if resume_run_id:
        return str(resume_run_id)
    if existing_df.empty or "run_id" not in existing_df.columns:
        return None
    run_ids = existing_df["run_id"].dropna().astype(str)
    if run_ids.empty:
        return None
    return run_ids.iloc[-1]


def _index_existing_files(existing_df: pd.DataFrame, run_id: str) -> set[tuple[str, str]]:
    """Index existing rows by (repo, file_path) for a given run_id."""
    if existing_df.empty:
        return set()
    if "run_id" not in existing_df.columns or "repo" not in existing_df.columns or "file_path" not in existing_df.columns:
        return set()
    run_rows = existing_df[existing_df["run_id"].astype(str) == str(run_id)]
    keys: set[tuple[str, str]] = set()
    for _, row in run_rows.iterrows():
        repo = row.get("repo")
        file_path = row.get("file_path")
        if pd.isna(repo) or pd.isna(file_path):
            continue
        keys.add((str(repo), str(file_path)))
    return keys


def _resolve_source_path(repo_id: str, file_path: str) -> Path | None:
    """Resolve a file_path against the raw repo directory, handling common path variants."""
    if not repo_id or not file_path:
        return None

    repo_dir = config.RAW_REPOS_DIR / repo_id
    if not repo_dir.exists():
        return None

    raw_path = Path(str(file_path))
    candidates: list[Path] = []

    if raw_path.is_absolute():
        if repo_dir in raw_path.parents or str(raw_path).startswith(str(repo_dir)):
            candidates.append(raw_path)

    normalized = _normalize_merge_path(str(file_path))
    if normalized.lower().startswith(repo_id.lower() + "/"):
        normalized = normalized[len(repo_id) + 1 :]

    candidates.extend([
        repo_dir / str(file_path),
        repo_dir / normalized,
    ])

    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue

    return None


def _is_binary_file(path: Path) -> bool:
    """Best-effort binary detection to avoid dumping non-text into prompts."""
    try:
        with path.open("rb") as handle:
            chunk = handle.read(2048)
        return b"\x00" in chunk
    except OSError:
        return True


def _load_code_excerpt(repo_id: str, file_path: str) -> str | None:
    """Load a small, token-safe code excerpt for the LLM prompt."""
    if CODE_EXCERPT_MAX_CHARS <= 0 or CODE_EXCERPT_HEAD_LINES <= 0:
        return None

    source_path = _resolve_source_path(repo_id, file_path)
    if not source_path or not source_path.exists():
        return None

    if _is_binary_file(source_path):
        return None

    try:
        if source_path.stat().st_size <= 0:
            return None
    except OSError:
        return None

    excerpt_lines: list[str] = []
    total_chars = 0

    try:
        with source_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(CODE_EXCERPT_HEAD_LINES):
                line = handle.readline()
                if not line:
                    break
                cleaned = line.rstrip("\n")
                excerpt_lines.append(cleaned)
                total_chars += len(cleaned) + 1
                if total_chars >= CODE_EXCERPT_MAX_CHARS:
                    break

            if (
                CODE_EXCERPT_TAIL_LINES > 0
                and total_chars < CODE_EXCERPT_MAX_CHARS
                and source_path.stat().st_size <= CODE_EXCERPT_MAX_BYTES
            ):
                from collections import deque

                tail = deque(maxlen=CODE_EXCERPT_TAIL_LINES)
                for line in handle:
                    tail.append(line.rstrip("\n"))

                if tail:
                    excerpt_lines.append("...")
                    for line in tail:
                        if total_chars >= CODE_EXCERPT_MAX_CHARS:
                            break
                        excerpt_lines.append(line)
                        total_chars += len(line) + 1
    except OSError:
        return None

    excerpt = "\n".join(excerpt_lines).strip()
    return excerpt or None


def _fill_git_fallbacks(df: pd.DataFrame) -> pd.DataFrame:
    """Fill minimal git metric fallbacks to reduce 'unknown' social signals."""
    if "git_metrics_status" not in df.columns:
        return df

    ok = df["git_metrics_status"] == "ok"

    # Recency fallback: if missing but file_age_days exists, use file_age_days.
    if "recency_days" in df.columns and "file_age_days" in df.columns:
        recency_missing = df["recency_days"].isna() & df["file_age_days"].notna()
        df.loc[ok & recency_missing, "recency_days"] = df.loc[ok & recency_missing, "file_age_days"]

    # Bus factor fallback: if missing and no recent activity, use 0 (no active contributors).
    if "bus_factor_estimate" in df.columns:
        bus_missing = df["bus_factor_estimate"].isna()
        no_commits = (
            df["commit_count_12m"].fillna(0) == 0
            if "commit_count_12m" in df.columns
            else False
        )
        no_authors = (
            df["unique_authors_12m"].fillna(0) == 0
            if "unique_authors_12m" in df.columns
            else False
        )
        df.loc[ok & bus_missing & (no_commits | no_authors), "bus_factor_estimate"] = 0

    # Contributor fallback: if missing and no commits, set to 0.
    if "unique_authors_12m" in df.columns and "commit_count_12m" in df.columns:
        authors_missing = df["unique_authors_12m"].isna() & (df["commit_count_12m"].fillna(0) == 0)
        df.loc[ok & authors_missing, "unique_authors_12m"] = 0

    return df


def _enforce_git_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce git missingness rules:
    - If no git signals exist, git_metrics_status must be "missing".
    - If git_metrics_status is "missing", set all Git fields to None.
    """
    if "git_metrics_status" not in df.columns:
        return df

    git_fields = [
        "churn_12m",
        "unique_authors_12m",
        "dominant_author_share",
        "single_contributor_12m",
        "commit_count_12m",
        "recency_days",
        "bus_factor_estimate",
        "file_age_days",
    ]
    existing_fields = [field for field in git_fields if field in df.columns]
    if existing_fields:
        has_any = df[existing_fields].notna().any(axis=1)
    else:
        has_any = pd.Series([False] * len(df), index=df.index)

    # If any git signals exist but status is missing/blank, mark ok
    needs_ok = has_any & df["git_metrics_status"].isna()
    df.loc[needs_ok, "git_metrics_status"] = "ok"

    # If no git signals exist, force missing
    df.loc[~has_any, "git_metrics_status"] = "missing"

    # If git_metrics_status is "missing", null out all git fields
    is_missing = df["git_metrics_status"] == "missing"
    for field in existing_fields:
        df.loc[is_missing, field] = None

    return df


def load_merged_dataset() -> pd.DataFrame:
    """Load and merge Sonar + Git metrics with LEFT JOIN (keep all Sonar rows)."""
    sonar_path = config.sonar_metrics_path()
    git_path = config.git_metrics_path()
    
    if not sonar_path.exists():
        raise FileNotFoundError(f"Sonar metrics not found: {sonar_path}")
    
    sonar_df = pd.read_csv(sonar_path)
    
    # Check if git metrics exist
    if git_path.exists():
        git_df = pd.read_csv(git_path)
        
        # Normalize file paths for merging
        sonar_df["merge_key"] = sonar_df["repo"] + "/" + sonar_df["file_path"].apply(_normalize_merge_path)
        git_df["merge_key"] = git_df["repo"] + "/" + git_df["file_path"].apply(_normalize_merge_path)
        
        # LEFT JOIN: keep all Sonar rows, add Git where available
        merged = pd.merge(
            sonar_df,
            git_df,
            on="merge_key",
            how="left",
            suffixes=("_sonar", "_git")
        )
        
        # For rows with no git match, set git_metrics_status to "missing"
        if "git_metrics_status" not in merged.columns:
            merged["git_metrics_status"] = None
        merged.loc[merged["git_metrics_status"].isna(), "git_metrics_status"] = "missing"
        
        print(f"Loaded {len(sonar_df)} Sonar records, {len(git_df)} Git records")
        print(f"Merged dataset: {len(merged)} records (LEFT JOIN, keeping all Sonar rows)")
        
        # Fill fallbacks before enforcing missingness
        merged = _fill_git_fallbacks(merged)
        merged = _enforce_git_missingness(merged)
        
        missing_git = (merged["git_metrics_status"] == "missing").sum()
        print(f"  - {missing_git} rows have git_metrics_status='missing'")
    else:
        print(f"Warning: Git metrics not found at {git_path}, proceeding with Sonar only")
        merged = sonar_df.copy()
        merged["git_metrics_status"] = "missing"
    
    return merged


def row_to_evaluation_input(row: pd.Series) -> EvaluationInput:
    """Convert a DataFrame row to EvaluationInput."""
    # Determine repo and file_path
    repo = row.get("repo_sonar") or row.get("repo") or row.get("repo_git", "unknown")
    file_path = row.get("file_path_sonar") or row.get("file_path") or row.get("file_path_git", "unknown")
    
    # Build SonarMetrics
    sonar = SonarMetrics(
        complexity=_safe_float(row.get("sonar_complexity")),
        cognitive_complexity=_safe_float(row.get("sonar_cognitive_complexity")),
        ncloc=_safe_float(row.get("sonar_ncloc")),
        comment_lines_density=_safe_float(row.get("sonar_comment_lines_density")),
        sqale_index=_safe_float(row.get("sonar_sqale_index")),
        sqale_rating=_safe_int(row.get("sonar_sqale_rating")),
        code_smells=_safe_int(row.get("sonar_code_smells")),
        bugs=_safe_int(row.get("sonar_bugs")),
        vulnerabilities=_safe_int(row.get("sonar_vulnerabilities")),
        duplicated_lines_density=_safe_float(row.get("sonar_duplicated_lines_density")),
        duplicated_blocks=_safe_int(row.get("sonar_duplicated_blocks")),
        reliability_rating=_safe_int(row.get("sonar_reliability_rating")),
        security_rating=_safe_int(row.get("sonar_security_rating")),
    )
    
    # Build GitMetrics
    git = GitMetrics(
        churn_12m=_safe_int(row.get("churn_12m")),
        unique_authors_12m=_safe_int(row.get("unique_authors_12m")),
        dominant_author_share=_safe_float(row.get("dominant_author_share")),
        single_contributor_flag=_safe_bool(row.get("single_contributor_12m")),
        commit_count_12m=_safe_int(row.get("commit_count_12m")),
        recency_days=_safe_int(row.get("recency_days")),
        bus_factor_estimate=_safe_int(row.get("bus_factor_estimate")),
        file_age_days=_safe_int(row.get("file_age_days")),
        git_metrics_status=row.get("git_metrics_status"),
    )
    
    # Detect language from file extension
    language = _detect_language(file_path)
    
    return EvaluationInput(
        repo_id=repo,
        file_path=file_path,
        language=language,
        commit_sha=row.get("commit_sha"),
        code_excerpt=_load_code_excerpt(repo, file_path),
        sonar_metrics=sonar,
        git_metrics=git,
    )


def _safe_float(val) -> float | None:
    """Convert to float, returning None for NaN/None."""
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    """Convert to int, returning None for NaN/None."""
    if val is None or pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_bool(val) -> bool | None:
    """Convert to bool, returning None for NaN/None."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return None


def _detect_language(file_path: str) -> str | None:
    """Detect language from file extension."""
    ext_map = {
        ".py": "Python",
        ".java": "Java",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".go": "Go",
        ".rs": "Rust",
        ".cpp": "C++",
        ".c": "C",
        ".cs": "C#",
        ".rb": "Ruby",
        ".php": "PHP",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".scala": "Scala",
    }
    ext = Path(file_path).suffix.lower()
    return ext_map.get(ext)


# ---------------------------------------------------------------------------
# Provenance & Reproducibility Helpers
# ---------------------------------------------------------------------------

def _compute_prompt_hash() -> str:
    """Compute hash of system prompt for version tracking."""
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]


def _normalize_flag(flag: str) -> str:
    """Normalize a flag string to snake_case for matching."""
    # Remove common prefixes and normalize
    flag = flag.lower().strip()
    flag = re.sub(r"[^a-z0-9]+", "_", flag)
    flag = flag.strip("_")
    return flag


def _summarize_error_message(err: Exception) -> str:
    """Return a single-line error summary safe for CSV/logging."""
    msg = str(err).strip()
    if not msg:
        return err.__class__.__name__
    return msg.splitlines()[0]


def _sanitize_csv_field(text: str | None) -> str:
    """Collapse whitespace to keep CSV rows single-line."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _compute_numeric_flags(sonar: SonarMetrics | None, git: GitMetrics | None) -> dict[str, int | None]:
    """
    Compute one-hot flags from NUMERIC THRESHOLDS (deterministic, not LLM-derived).
    
    Returns dict with flag columns. Values are 1, 0, or None (if input metric is missing).
    
    THRESHOLD DEFINITIONS:
    - Comment density: < 10 = low, 10-25 = moderate, > 25 = good
    - Complexity: > 15 cognitive = high, <= 5 = low
    - Recency: <= 90 = active, 91-180 = stale, > 180 = inactive
    - Bus factor: <= 2 = risk, None = unknown
    - Contributors: 1 = single, 2-3 = few, 4-7 = moderate, >= 8 = many
    """
    flags = {}
    
    # Technical flags from Sonar metrics
    if sonar:
        # flag_large_file = (ncloc >= 500)
        if sonar.ncloc is not None:
            flags["flag_large_file"] = 1 if sonar.ncloc >= THRESHOLD_LARGE_FILE_NCLOC else 0
        else:
            flags["flag_large_file"] = None
        
        # flag_high_complexity = (cognitive_complexity > 15)
        if sonar.cognitive_complexity is not None:
            flags["flag_high_complexity"] = 1 if sonar.cognitive_complexity > THRESHOLD_HIGH_COGNITIVE_COMPLEXITY else 0
        else:
            flags["flag_high_complexity"] = None
        
        # flag_low_complexity = (cognitive_complexity <= 5)
        if sonar.cognitive_complexity is not None:
            flags["flag_low_complexity"] = 1 if sonar.cognitive_complexity <= 5 else 0
        else:
            flags["flag_low_complexity"] = None
        
        # Comment density flags (mutually exclusive)
        if sonar.comment_lines_density is not None:
            flags["flag_low_comment_density"] = 1 if sonar.comment_lines_density < 10 else 0
            flags["flag_good_comment_density"] = 1 if sonar.comment_lines_density > 25 else 0
        else:
            flags["flag_low_comment_density"] = None
            flags["flag_good_comment_density"] = None
        
        # flag_duplication_present = (duplicated_lines_density > 0)
        if sonar.duplicated_lines_density is not None:
            flags["flag_duplication_present"] = 1 if sonar.duplicated_lines_density > 0 else 0
        else:
            flags["flag_duplication_present"] = None
        
        # flag_security_issues = (vulnerabilities > 0) OR (security_rating >= 4)
        vuln_issue = sonar.vulnerabilities is not None and sonar.vulnerabilities > 0
        sec_rating_issue = sonar.security_rating is not None and sonar.security_rating >= THRESHOLD_HIGH_RATING
        if sonar.vulnerabilities is not None or sonar.security_rating is not None:
            flags["flag_security_issues"] = 1 if (vuln_issue or sec_rating_issue) else 0
        else:
            flags["flag_security_issues"] = None
        
        # flag_reliability_issues = (bugs > 0) OR (reliability_rating >= 4)
        bugs_issue = sonar.bugs is not None and sonar.bugs > 0
        rel_rating_issue = sonar.reliability_rating is not None and sonar.reliability_rating >= THRESHOLD_HIGH_RATING
        if sonar.bugs is not None or sonar.reliability_rating is not None:
            flags["flag_reliability_issues"] = 1 if (bugs_issue or rel_rating_issue) else 0
        else:
            flags["flag_reliability_issues"] = None
    else:
        flags["flag_large_file"] = None
        flags["flag_high_complexity"] = None
        flags["flag_low_complexity"] = None
        flags["flag_low_comment_density"] = None
        flags["flag_good_comment_density"] = None
        flags["flag_duplication_present"] = None
        flags["flag_security_issues"] = None
        flags["flag_reliability_issues"] = None
    
    # Social/Process flags from Git metrics
    if git:
        bus_factor = git.bus_factor_estimate
        dominant_share = git.dominant_author_share
        unique_authors = git.unique_authors_12m
        recency = git.recency_days
        git_status = git.git_metrics_status
        
        # Recency flags (mutually exclusive)
        if recency is not None:
            flags["flag_active_recency"] = 1 if recency <= 90 else 0
            flags["flag_stale_recency"] = 1 if 90 < recency <= 180 else 0
            flags["flag_inactive_recency"] = 1 if recency > 180 else 0
            flags["flag_unknown_recency"] = 0
        elif git_status == "ok":
            # Git status OK but recency null = should not happen after miner fix
            flags["flag_active_recency"] = 0
            flags["flag_stale_recency"] = 0
            flags["flag_inactive_recency"] = 0
            flags["flag_unknown_recency"] = 1
        else:
            flags["flag_active_recency"] = None
            flags["flag_stale_recency"] = None
            flags["flag_inactive_recency"] = None
            flags["flag_unknown_recency"] = 1 if git_status == "missing" else None
        
        # Bus factor flags
        flags["flag_bus_factor_unknown"] = 1 if bus_factor is None else 0
        
        if bus_factor is not None:
            flags["flag_bus_factor_risk"] = 1 if bus_factor <= 2 else 0
        else:
            # Bus factor unknown - risk flag should be 0 (unknown, not risky)
            flags["flag_bus_factor_risk"] = 0
        
        # Contributor count flags (mutually exclusive)
        if unique_authors is not None:
            flags["flag_single_contributor"] = 1 if unique_authors == 1 else 0
            flags["flag_few_contributors"] = 1 if unique_authors in (2, 3) else 0
            flags["flag_many_contributors"] = 1 if unique_authors >= 8 else 0
        else:
            flags["flag_single_contributor"] = None
            flags["flag_few_contributors"] = None
            flags["flag_many_contributors"] = None
    else:
        flags["flag_active_recency"] = None
        flags["flag_stale_recency"] = None
        flags["flag_inactive_recency"] = None
        flags["flag_unknown_recency"] = None
        flags["flag_bus_factor_unknown"] = None
        flags["flag_bus_factor_risk"] = None
        flags["flag_single_contributor"] = None
        flags["flag_few_contributors"] = None
        flags["flag_many_contributors"] = None
    
    return flags


def _validate_llm_flags(
    assessment: 'HolisticAssessment',
    sonar: SonarMetrics | None,
    git: GitMetrics | None,
) -> list[str]:
    """
    Validate LLM-generated flags against numeric thresholds.
    Returns list of violation descriptions (empty if valid).
    """
    violations = []
    
    tech_flags_lower = [f.lower() for f in assessment.technical_flags]
    social_flags_lower = [f.lower() for f in assessment.social_sustainability_flags]
    
    # Check 1: comment_lines_density > 25 but "low comment density" in flags
    if sonar and sonar.comment_lines_density is not None and sonar.comment_lines_density > 25:
        for flag in tech_flags_lower:
            if "low comment" in flag:
                violations.append(
                    f"comment_density={sonar.comment_lines_density:.1f} but LLM said '{flag}'"
                )
    
    # Check 2: cognitive_complexity <= 5 but "high complexity" in flags
    if sonar and sonar.cognitive_complexity is not None and sonar.cognitive_complexity <= 5:
        for flag in tech_flags_lower:
            if "high complex" in flag:
                violations.append(
                    f"cognitive_complexity={sonar.cognitive_complexity} but LLM said '{flag}'"
                )
    
    # Check 3: git_metrics_status == "ok" but "unknown recency/bus factor" in flags
    if git and git.git_metrics_status == "ok":
        if git.recency_days is not None:
            for flag in social_flags_lower:
                if "unknown recency" in flag:
                    violations.append(
                        f"git_status=ok, recency_days={git.recency_days} but LLM said '{flag}'"
                    )
        if git.bus_factor_estimate is not None:
            for flag in social_flags_lower:
                if "unknown bus factor" in flag:
                    violations.append(
                        f"git_status=ok, bus_factor={git.bus_factor_estimate} but LLM said '{flag}'"
                    )
    
    # Check 4: unique_authors < 8 but "many contributors" in flags
    if git and git.unique_authors_12m is not None and git.unique_authors_12m < 8:
        for flag in social_flags_lower:
            if "many contributor" in flag:
                violations.append(
                    f"unique_authors={git.unique_authors_12m} but LLM said '{flag}'"
                )
    
    return violations


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run_holistic_evaluation(
    sample_size: int | None = None,
    repos: list[str] | None = None,
    n_repeats: int = 1,
    resume: bool = False,
    resume_run_id: str | None = None,
) -> pd.DataFrame:
    """
    Run holistic evaluation on the merged dataset.
    
    Args:
        sample_size: If set, evaluate only this many files (random sample)
        repos: If set, filter to only these repositories
        n_repeats: Number of repeated evaluations per file (for stability analysis)
    
    Returns:
        DataFrame with evaluation results
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate run metadata for reproducibility
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    prompt_version = _compute_prompt_hash()

    # Load data
    df = load_merged_dataset()

    # Filter by repos if specified
    if repos:
        repo_col = "repo_sonar" if "repo_sonar" in df.columns else "repo"
        df = df[df[repo_col].isin(repos)]
        print(f"Filtered to {len(df)} records in repos: {repos}")

    # Sample if specified
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} records")

    # Initialize evaluator
    evaluator = HolisticEvaluator()

    # Output CSV path (per model + run index).
    if resume:
        csv_path = config.holistic_assessments_path(model=evaluator.model, prefer_existing=True)
    else:
        csv_path = config.holistic_assessments_path(model=evaluator.model, prefer_existing=False)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing_df = _load_existing_results(csv_path) if resume else pd.DataFrame()
    existing_keys: set[tuple[str, str]] = set()

    if resume:
        if existing_df.empty:
            print("⚠ Resume requested but no existing holistic CSV rows found; starting a new run.")
            resume = False
            csv_path = config.holistic_assessments_path(model=evaluator.model, prefer_existing=False)
            existing_df = pd.DataFrame()
        else:
            selected_run_id = _select_resume_run_id(existing_df, resume_run_id)
            if selected_run_id:
                run_id = selected_run_id
                existing_keys = _index_existing_files(existing_df, run_id)
                run_rows = existing_df[existing_df["run_id"].astype(str) == str(run_id)] if "run_id" in existing_df.columns else pd.DataFrame()
                if not run_rows.empty and "prompt_version" in run_rows.columns:
                    prompt_values = run_rows["prompt_version"].dropna().astype(str)
                    if not prompt_values.empty:
                        last_prompt_version = prompt_values.iloc[-1]
                        if last_prompt_version != prompt_version:
                            print(f"⚠ Resume: prompt_version changed ({last_prompt_version} -> {prompt_version})")
            else:
                print("⚠ Resume requested but no run_id found in CSV; starting a new run.")
                resume = False

    print(f"\nRun ID: {run_id}")
    print(f"Model: {evaluator.model}")
    print(f"Prompt version: {prompt_version}")
    print(f"Repeats per file: {n_repeats}")
    print(f"Output CSV: {csv_path}")
    print(f"\nEvaluating {len(df)} files...")
    print("=" * 60)

    header_written = csv_path.exists() and csv_path.stat().st_size > 0
    new_rows: list[dict[str, Any]] = []
    skipped = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        input_data = row_to_evaluation_input(row)

        key = (input_data.repo_id, input_data.file_path)
        if resume and key in existing_keys:
            skipped += 1
            continue

        print(f"[{i+1}/{len(df)}] {input_data.repo_id}/{input_data.file_path[:50]}...")

        # Run n_repeats evaluations for stability analysis
        repeat_assessments: list[HolisticAssessment] = []

        for rep in range(n_repeats):
            assessment = evaluator.evaluate(input_data)
            repeat_assessments.append(assessment)

        if not repeat_assessments:
            continue

        # Use first assessment for main results (or aggregate if repeats > 1)
        assessment = repeat_assessments[0]

        # Determine evaluation status by checking notes for error message
        is_error = any("Evaluation error:" in note for note in assessment.notes)
        evaluation_status = "error" if is_error else "ok"
        error_message = assessment.notes[0] if is_error and assessment.notes else None

        # Extract key input metrics for join/analysis
        sonar = input_data.sonar_metrics
        git = input_data.git_metrics

        # Validate LLM flags against numeric thresholds (for QA tracking)
        llm_violations = _validate_llm_flags(assessment, sonar, git) if not is_error else []

        # Compute agreement if multiple repeats
        risk_variance = None
        agreement_rate = None
        if n_repeats > 1 and len(repeat_assessments) == n_repeats:
            sustainability_votes = [a.sustainability_risk for a in repeat_assessments]
            # Agreement = fraction that match the mode
            from collections import Counter
            sust_mode = Counter(sustainability_votes).most_common(1)[0][0]
            agreement_rate = sum(1 for v in sustainability_votes if v == sust_mode) / n_repeats
            # Variance proxy: count unique values
            risk_variance = len(set(sustainability_votes))

        # Compute numeric threshold-based flags (deterministic)
        numeric_flags = _compute_numeric_flags(sonar, git)

        # For errors, flags should be empty; for successful evals, use LLM flags directly
        if is_error:
            clean_tech_flags = []
            clean_social_flags = []
        else:
            clean_tech_flags = assessment.technical_flags
            clean_social_flags = assessment.social_sustainability_flags
        
        # Flatten to CSV-friendly row
        result_row = {
            # Provenance columns
            "run_id": run_id,
            "model_name": evaluator.model,
            "prompt_version": prompt_version,
            "temperature": evaluator.temperature,
            "max_tokens": evaluator.max_tokens,
            "n_repeats": n_repeats,
            "agreement_rate": agreement_rate,
            "risk_variance": risk_variance,
            
            # Identity
            "repo": assessment.repo_id,
            "file_path": assessment.file_path,
            "language": assessment.language,
            
            # Status columns
            "evaluation_status": evaluation_status,
            "error_message": error_message,
            "git_metrics_status": git.git_metrics_status if git else None,
            "n_llm_violations": len(llm_violations),
            "llm_violations": _sanitize_csv_field("; ".join(llm_violations)) if llm_violations else None,
            
            # LLM outputs
            "maintainability_risk": assessment.maintainability_risk,
            "sustainability_risk": assessment.sustainability_risk,
            "confidence": assessment.confidence,
            
            # Structured counts
            "n_technical_flags": len(clean_tech_flags),
            "n_social_flags": len(clean_social_flags),
            "n_drivers": len(assessment.top_drivers),
            "n_actions": len(assessment.recommended_actions),
            
            # String versions (for reference) - LLM-derived flags kept here
            "technical_flags": _sanitize_csv_field("; ".join(clean_tech_flags)),
            "social_flags": _sanitize_csv_field("; ".join(clean_social_flags)),
            "top_drivers": _sanitize_csv_field("; ".join([d.driver for d in assessment.top_drivers])),
            "notes": _sanitize_csv_field("; ".join(assessment.notes) if assessment.notes else ""),
            
            # Key input metrics (self-contained for analysis)
            "input_ncloc": sonar.ncloc if sonar else None,
            "input_complexity": sonar.complexity if sonar else None,
            "input_cognitive_complexity": sonar.cognitive_complexity if sonar else None,
            "input_comment_lines_density": sonar.comment_lines_density if sonar else None,
            "input_sqale_index": sonar.sqale_index if sonar else None,
            "input_sqale_rating": sonar.sqale_rating if sonar else None,
            "input_code_smells": sonar.code_smells if sonar else None,
            "input_bugs": sonar.bugs if sonar else None,
            "input_vulnerabilities": sonar.vulnerabilities if sonar else None,
            "input_duplicated_density": sonar.duplicated_lines_density if sonar else None,
            "input_reliability_rating": sonar.reliability_rating if sonar else None,
            "input_security_rating": sonar.security_rating if sonar else None,
            "input_churn_12m": git.churn_12m if git else None,
            "input_unique_authors": git.unique_authors_12m if git else None,
            "input_dominant_author_share": git.dominant_author_share if git else None,
            "input_single_contributor": git.single_contributor_flag if git else None,
            "input_recency_days": git.recency_days if git else None,
            "input_file_age_days": git.file_age_days if git else None,
            "input_commit_count_12m": git.commit_count_12m if git else None,
            "input_bus_factor_estimate": git.bus_factor_estimate if git else None,
            
            # Latency (null for errors)
            "latency_ms": assessment.latency_ms if evaluation_status == "ok" else None,
        }

        # Add numeric threshold-based flag columns
        result_row.update(numeric_flags)

        # Incremental write: append each evaluated row immediately.
        pd.DataFrame([result_row]).to_csv(
            csv_path,
            index=False,
            mode="a",
            header=not header_written,
        )
        header_written = True
        new_rows.append(result_row)
        existing_keys.add(key)

        status_indicator = "✓" if evaluation_status == "ok" else "✗"
        print(f"  {status_indicator} Sustainability: {assessment.sustainability_risk} "
              f"(confidence={assessment.confidence})")

        # Rate limiting
        time.sleep(0.5)

    # Load run results for summary/validation.
    results_df = _load_existing_results(csv_path)
    if results_df.empty:
        results_df = pd.DataFrame(new_rows)
    if "run_id" in results_df.columns:
        run_results_df = results_df[results_df["run_id"].astype(str) == str(run_id)].copy()
    else:
        run_results_df = results_df.copy()

    # Print validation summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nRun ID: {run_id}")
    print(f"Total files in run: {len(run_results_df)}")
    if resume and skipped:
        print(f"Skipped (already complete): {skipped}")
    if run_results_df.empty:
        print("No results available for summary.")
        print(f"\nResults saved to:\n  - {csv_path}")
        return run_results_df

    # Evaluation status counts
    print("\nEvaluation Status:")
    print(run_results_df["evaluation_status"].value_counts())

    # Git metrics status counts
    if "git_metrics_status" in run_results_df.columns:
        print("\nGit Metrics Status:")
        print(run_results_df["git_metrics_status"].value_counts(dropna=False))

    print("\nSustainability Risk Distribution:")
    print(run_results_df["sustainability_risk"].value_counts())

    print("\nMaintainability Risk Distribution:")
    print(run_results_df["maintainability_risk"].value_counts())

    print("\nConfidence Distribution:")
    print(run_results_df["confidence"].value_counts().sort_index())

    # Missingness report for new fields
    print("\n--- Missingness Report (new fields) ---")
    new_fields = [
        "input_recency_days", "input_file_age_days", "input_commit_count_12m",
        "input_bus_factor_estimate", "git_metrics_status"
    ]
    for field in new_fields:
        if field in run_results_df.columns:
            null_count = run_results_df[field].isna().sum()
            denom = len(run_results_df)
            null_pct = 100 * null_count / denom if denom > 0 else 0
            print(f"  {field}: {null_count}/{denom} missing ({null_pct:.1f}%)")

    if n_repeats > 1 and "agreement_rate" in run_results_df.columns and not run_results_df.empty:
        print(f"\nAgreement Rate (mean): {run_results_df['agreement_rate'].mean():.2%}")

    # Data Quality Validation Checks
    print("\n" + "=" * 60)
    print("DATA QUALITY VALIDATION CHECKS")
    print("=" * 60)

    # Check 1: No rows where recency_days is null AND git_metrics_status == "ok"
    if "input_recency_days" in run_results_df.columns and "git_metrics_status" in run_results_df.columns:
        bad_recency = (run_results_df["input_recency_days"].isna()) & (run_results_df["git_metrics_status"] == "ok")
        count_bad_recency = bad_recency.sum()
        if count_bad_recency > 0:
            print(f"⚠ CHECK 1 FAILED: {count_bad_recency} rows have recency_days=null but git_metrics_status='ok'")
        else:
            print("✓ CHECK 1 PASSED: No rows with recency_days=null AND git_metrics_status='ok'")

    # Check 2: Error rows have evaluation_status=="error" AND flags are empty AND latency_ms is null
    error_rows = run_results_df[run_results_df["evaluation_status"] == "error"]
    if len(error_rows) > 0:
        flags_not_empty = (error_rows["n_technical_flags"] > 0) | (error_rows["n_social_flags"] > 0)
        latency_not_null = error_rows["latency_ms"].notna()
        bad_errors = flags_not_empty | latency_not_null
        count_bad_errors = bad_errors.sum()
        if count_bad_errors > 0:
            print(f"⚠ CHECK 2 FAILED: {count_bad_errors} error rows have non-empty flags or non-null latency_ms")
        else:
            print(f"✓ CHECK 2 PASSED: All {len(error_rows)} error rows have empty flags and null latency_ms")
    else:
        print("✓ CHECK 2 PASSED: No error rows to validate")
    
    # Check 3: LLM flag consistency - using the pre-computed llm_violations column
    if "n_llm_violations" in run_results_df.columns:
        total_violations = run_results_df["n_llm_violations"].sum()
        rows_with_violations = (run_results_df["n_llm_violations"] > 0).sum()
        
        if total_violations > 0:
            print(f"⚠ CHECK 3 FAILED: {total_violations} LLM flag violations across {rows_with_violations} rows")
            print("  LLM generated flags that contradict numeric thresholds:")
            
            # Sample violations for visibility
            violation_rows = run_results_df[run_results_df["n_llm_violations"] > 0][
                ["repo", "file_path", "n_llm_violations", "llm_violations"]
            ].head(10)
            for _, row in violation_rows.iterrows():
                print(f"    - {row['repo']}/{row['file_path']}: {row['llm_violations']}")
            
            if rows_with_violations > 10:
                print(f"    ... and {rows_with_violations - 10} more rows with violations")
        else:
            print("✓ CHECK 3 PASSED: No LLM flag violations detected")
    
    # Check 4: Deterministic flags were computed correctly
    flag_cols = [c for c in run_results_df.columns if c.startswith("flag_")]
    if flag_cols:
        print(f"ℹ CHECK 4: {len(flag_cols)} deterministic flag columns computed: {', '.join(flag_cols[:5])}...")

    # Check 5 (report-only): maintainability Unknown should only occur when core inputs are absent.
    core_cols = [
        "input_ncloc",
        "input_code_smells",
        "input_complexity",
        "input_cognitive_complexity",
        "input_sqale_index",
        "input_sqale_rating",
    ]
    existing_core_cols = [c for c in core_cols if c in run_results_df.columns]
    if "maintainability_risk" in run_results_df.columns and existing_core_cols:
        unknown_mask = run_results_df["maintainability_risk"].astype(str).str.lower() == "unknown"
        has_any_core_input = run_results_df[existing_core_cols].notna().any(axis=1)
        bad_unknown = unknown_mask & has_any_core_input
        bad_unknown_count = int(bad_unknown.sum())
        if bad_unknown_count > 0:
            print(
                f"⚠ CHECK 5 WARNING: {bad_unknown_count} rows have maintainability_risk='Unknown' "
                "despite non-null maintainability inputs"
            )
        else:
            print(
                "✓ CHECK 5 PASSED: maintainability_risk='Unknown' appears only when core "
                "maintainability inputs are missing"
            )

    # Overall QA result
    print("\n" + "-" * 60)
    qa_passed = True
    failure_reasons = []

    if "n_llm_violations" in run_results_df.columns and run_results_df["n_llm_violations"].sum() > 0:
        qa_passed = False
        failure_reasons.append(f"{run_results_df['n_llm_violations'].sum()} LLM flag violations")

    if qa_passed:
        print("✅ QA PASSED: All validation checks passed")
    else:
        print("❌ QA FAILED: " + "; ".join(failure_reasons))
        print("   Review the llm_violations column in the output CSV for details.")
        # Don't raise an exception - just warn. The violations are tracked in the output.

    print(f"\nResults saved to:")
    print(f"  - {csv_path}")

    return run_results_df


def analyze_single_file(
    repo_id: str,
    file_path: str,
    sonar_metrics: dict | None = None,
    git_metrics: dict | None = None,
    code_excerpt: str | None = None,
    verbose: bool = True,
) -> HolisticAssessment:
    """
    Analyze a single file with provided or looked-up metrics.
    
    Useful for testing or ad-hoc analysis.
    """
    # Build metrics objects
    sonar = SonarMetrics(**sonar_metrics) if sonar_metrics else None
    git = GitMetrics(**git_metrics) if git_metrics else None
    
    input_data = EvaluationInput(
        repo_id=repo_id,
        file_path=file_path,
        language=_detect_language(file_path),
        code_excerpt=code_excerpt,
        sonar_metrics=sonar,
        git_metrics=git,
    )
    
    if verbose:
        print("INPUT:")
        print(input_data.to_prompt())
        print("\n" + "=" * 60)
    
    evaluator = HolisticEvaluator()
    assessment = evaluator.evaluate(input_data)
    
    if verbose:
        print("\nASSESSMENT:")
        print(json.dumps(assessment.to_dict(), indent=2))
    
    return assessment


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Holistic Software Sustainability Evaluator"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Number of files to sample (default: all)"
    )
    parser.add_argument(
        "--repos", nargs="+", default=None,
        help="Filter to specific repositories"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run a single test evaluation"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest holistic CSV for this model"
    )
    parser.add_argument(
        "--resume-run-id", type=str, default=None,
        help="Run ID to resume (defaults to last run_id in existing CSV)"
    )
    
    args = parser.parse_args()

    # Env overrides (PowerShell-friendly)
    if not args.resume:
        resume_env = os.environ.get("HOLISTIC_RESUME", "false").strip().lower()
        if resume_env in {"1", "true", "yes"}:
            args.resume = True
    if args.resume and not args.resume_run_id:
        env_run_id = os.environ.get("HOLISTIC_RESUME_RUN_ID", "").strip()
        if env_run_id:
            args.resume_run_id = env_run_id
    
    if args.test:
        # Run a test with sample data
        print("Running test evaluation with sample data...")
        assessment = analyze_single_file(
            repo_id="test-repo",
            file_path="src/main/java/Example.java",
            sonar_metrics={
                "complexity": 15,
                "cognitive_complexity": 22,
                "ncloc": 150,
                "comment_lines_density": 8.5,
                "sqale_index": 45,
                "sqale_rating": 2,
                "code_smells": 5,
                "bugs": 1,
                "vulnerabilities": 0,
                "duplicated_lines_density": None,  # Will test null handling
                "reliability_rating": 2,
                "security_rating": 1,
            },
            git_metrics={
                "churn_12m": 450,
                "unique_authors_12m": 3,
                "dominant_author_share": 0.65,
                "single_contributor_flag": False,
                "commit_count_12m": 12,
                "recency_days": 14,
                "bus_factor_estimate": 2,
            },
            verbose=True,
        )
    else:
        # Run full evaluation
        results = run_holistic_evaluation(
            sample_size=args.sample,
            repos=args.repos,
            resume=args.resume,
            resume_run_id=args.resume_run_id,
        )
