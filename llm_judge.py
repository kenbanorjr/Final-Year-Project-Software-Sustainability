"""
LLM-based "Augmented Reviewer" for representative files.

Reads git_metrics.csv to locate candidate files, optionally samples them for
faster evaluation, submits source code plus lightweight context to the LLM, and
stores structured JSON results in llm_metrics.csv.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

import config

SYSTEM_PROMPT = """
You are evaluating code maintainability and sustainability risk for a research study.
Use BOTH the source code and the provided project context signals (activity, churn, authorship).
Return ONLY a single JSON object (no extra text, no markdown) with this schema:
{
  "readability_score": <integer 1-10>,
  "maintainability_risk": "Low" | "Med" | "High",
  "sustainability_risk": "Low" | "Med" | "High" | null,
  "architectural_issues": ["list of concise issue statements"],
  "sustainability_flags": ["list of short strings"],
  "reasoning": "one paragraph explaining the assessment"
}
If a field is unknown, use null for ratings and [] for lists.
""".strip()
PROMPT_VERSION = "v1"
CODE_MAX_LINES = 400
CODE_MAX_CHARS = 20000


README_CACHE: dict[str, tuple[str, str]] = {}


@dataclass
class CostTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add_openai_usage(self, usage: object) -> None:
        """Accumulate token usage from OpenAI responses."""
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        total = getattr(usage, "total_tokens", None)

        if prompt is not None:
            self.prompt_tokens += int(prompt)
        if completion is not None:
            self.completion_tokens += int(completion)
        if prompt is None and completion is None and total is not None:
            # Fall back to total when sub-components are unavailable.
            self.prompt_tokens += int(total)

    def add_gemini_usage(self, usage: object) -> None:
        """Accumulate token usage from Gemini usage_metadata."""
        if not usage:
            return
        getter = usage.get if isinstance(usage, dict) else getattr
        prompt = getter("prompt_token_count", None)
        completion = getter("candidates_token_count", None)
        if prompt is not None:
            self.prompt_tokens += int(prompt)
        if completion is not None:
            self.completion_tokens += int(completion)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _load_git_metrics_df(git_metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(git_metrics_csv)
    required_cols = {"repo", "file_path", "absolute_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"git_metrics.csv is missing columns: {', '.join(sorted(missing))}")
    numeric_cols = [
        "lines_of_code",
        "unique_authors_12m",
        "repo_unique_authors_12m",
        "repo_commits_12m",
        "repo_commits_per_month_12m",
        "churn_12m",
        "added_lines_12m",
        "deleted_lines_12m",
        "dominant_author_share",
        "last_12m_observed_commits",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("bus_factor_single_dev", "bus_factor_75_dominant_author"):
        if col in df.columns:
            df[col] = df[col].apply(_normalize_bool)
    return df


def _parse_json_response(content: str) -> Dict:
    """Parse JSON while stripping stray whitespace and markdown fences."""
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return False
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _format_stat_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _truncate_code(code: str) -> tuple[str, bool, int, int]:
    original_length = len(code)
    truncated = code
    truncated_flag = False

    if CODE_MAX_LINES and code:
        lines = code.splitlines()
        if len(lines) > CODE_MAX_LINES:
            truncated = "\n".join(lines[:CODE_MAX_LINES])
            truncated_flag = True

    if CODE_MAX_CHARS and len(truncated) > CODE_MAX_CHARS:
        truncated = truncated[:CODE_MAX_CHARS]
        truncated_flag = True

    truncated_length = len(truncated)
    return truncated, truncated_flag, truncated_length, original_length


def _normalize_risk_label(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip().lower()
    mapping = {"low": "Low", "med": "Med", "medium": "Med", "high": "High"}
    return mapping.get(cleaned)


def _normalize_readability_score(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    score: int | None = None
    if isinstance(value, int):
        score = value
    elif isinstance(value, float) and value.is_integer():
        score = int(value)
    elif isinstance(value, str):
        try:
            score = int(value.strip())
        except ValueError:
            score = None
    if score is None:
        return None
    if 1 <= score <= 10:
        return score
    return None


def _validate_llm_response(parsed: object) -> tuple[bool, str | None, dict]:
    normalized = {
        "readability_score": None,
        "maintainability_risk": None,
        "sustainability_risk": None,
        "architectural_issues": [],
        "sustainability_flags": [],
        "reasoning": "",
    }
    if not isinstance(parsed, dict):
        return False, "Invalid JSON: expected object", normalized

    errors: list[str] = []

    score = _normalize_readability_score(parsed.get("readability_score"))
    if score is None:
        errors.append("readability_score")
    normalized["readability_score"] = score

    maintainability = _normalize_risk_label(parsed.get("maintainability_risk"))
    if maintainability is None:
        errors.append("maintainability_risk")
    normalized["maintainability_risk"] = maintainability

    architectural_issues = parsed.get("architectural_issues")
    if not isinstance(architectural_issues, list):
        errors.append("architectural_issues")
        architectural_issues = []
    normalized["architectural_issues"] = [str(item) for item in architectural_issues]

    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        errors.append("reasoning")
        reasoning = ""
    normalized["reasoning"] = reasoning.strip() if isinstance(reasoning, str) else ""

    sustainability_risk = _normalize_risk_label(parsed.get("sustainability_risk"))
    normalized["sustainability_risk"] = sustainability_risk

    sustainability_flags = parsed.get("sustainability_flags")
    if sustainability_flags is None:
        sustainability_flags = []
    if not isinstance(sustainability_flags, list):
        sustainability_flags = []
    normalized["sustainability_flags"] = [str(item) for item in sustainability_flags]

    if errors:
        return False, f"Missing/invalid field: {', '.join(errors)}", normalized
    return True, None, normalized


def _format_project_context(file_record: dict) -> str:
    seed_category = file_record.get("seed_category")
    if seed_category is None:
        seed_category = file_record.get("category")
    fields = [
        ("seed_category", seed_category),
        ("activity_label", file_record.get("activity_label")),
        ("churn_12m", file_record.get("churn_12m")),
        ("added_lines_12m", file_record.get("added_lines_12m")),
        ("deleted_lines_12m", file_record.get("deleted_lines_12m")),
        ("unique_authors_12m", file_record.get("unique_authors_12m")),
        ("bus_factor_single_dev", file_record.get("bus_factor_single_dev")),
        ("dominant_author_share", file_record.get("dominant_author_share")),
        ("bus_factor_75_dominant_author", file_record.get("bus_factor_75_dominant_author")),
        ("repo_commits_12m", file_record.get("repo_commits_12m")),
        ("repo_unique_authors_12m", file_record.get("repo_unique_authors_12m")),
    ]
    return "\n".join(f"{label}: {_format_stat_value(value)}" for label, value in fields)


def _read_readme_snippet(repo_path: Path, max_chars: int = 1200, max_lines: int = 40) -> tuple[str, str]:
    if not repo_path.exists():
        return "", ""
    candidates = [
        path
        for path in repo_path.iterdir()
        if path.is_file() and path.name.lower().startswith("readme")
    ]
    if not candidates:
        return "", ""

    def sort_key(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if name == "readme.md":
            return (0, name)
        if name.startswith("readme.md"):
            return (1, name)
        if name.endswith(".md"):
            return (2, name)
        if name.endswith(".rst"):
            return (3, name)
        return (4, name)

    selected = sorted(candidates, key=sort_key)[0]
    try:
        content = selected.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "", ""

    lines = content.splitlines()
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
    snippet = "\n".join(lines).strip()
    if max_chars and len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "..."
    return selected.name, snippet


def _get_readme_snippet(repo_name: str) -> tuple[str, str]:
    if not repo_name:
        return "", ""
    if repo_name in README_CACHE:
        return README_CACHE[repo_name]
    repo_path = config.RAW_REPOS_DIR / repo_name
    snippet = _read_readme_snippet(repo_path)
    README_CACHE[repo_name] = snippet
    return snippet


def _build_context_block(file_record: dict) -> str:
    repo = file_record.get("repo", "")
    file_path = file_record.get("file_path", "")
    context_block = _format_project_context(file_record)
    readme_name, readme_snippet = _get_readme_snippet(repo)

    parts = [
        "Project Context:",
        context_block,
        "",
        f"Repository: {repo}",
        f"File: {file_path}",
    ]
    if readme_snippet:
        parts.append(f"README snippet ({readme_name}):\n{readme_snippet}")
    parts.append("Produce the JSON response for this file only.")
    return "\n".join(parts)


def _loc_band(value: object) -> str:
    try:
        loc = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(loc):
        return "unknown"
    if loc <= 150:
        return "50-150"
    if loc <= 275:
        return "151-275"
    return "276-400"


def _allocate_strata_targets(counts: dict[str, int], total: int) -> dict[str, int]:
    total_count = sum(counts.values())
    if total_count <= 0:
        return {key: 0 for key in counts}

    exact = {key: counts[key] / total_count * total for key in counts}
    targets = {key: int(math.floor(exact[key])) for key in counts}
    remainder = total - sum(targets.values())
    if remainder <= 0:
        return targets

    by_fraction = sorted(counts.keys(), key=lambda key: exact[key] - targets[key], reverse=True)
    for key in by_fraction[:remainder]:
        targets[key] += 1
    return targets


def _allocate_repo_targets(repos: list[str], total: int, min_per_repo: int) -> dict[str, int]:
    if not repos:
        return {}
    if total <= 0:
        return {repo: 0 for repo in repos}

    if min_per_repo > 0 and total < min_per_repo * len(repos):
        print(
            "[llm] LLM_SAMPLE_SIZE is smaller than LLM_SAMPLE_MIN_PER_REPO * repo_count; "
            "using equal allocation."
        )
        min_per_repo = 0

    if min_per_repo > 0:
        remaining = total - min_per_repo * len(repos)
        base = remaining // len(repos)
        remainder = remaining % len(repos)
        return {
            repo: min_per_repo + base + (1 if idx < remainder else 0)
            for idx, repo in enumerate(repos)
        }

    base = total // len(repos)
    remainder = total % len(repos)
    return {repo: base + (1 if idx < remainder else 0) for idx, repo in enumerate(repos)}


def _sample_repo_by_loc_band(repo_df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    repo_df = repo_df.copy()
    repo_df["loc_band"] = repo_df["lines_of_code"].apply(_loc_band)
    band_counts = repo_df["loc_band"].value_counts().to_dict()
    if len(band_counts) <= 1:
        return repo_df.sample(n=target, random_state=seed)

    targets = _allocate_strata_targets(band_counts, target)
    parts: list[pd.DataFrame] = []
    for idx, band in enumerate(sorted(targets.keys())):
        take = targets[band]
        if take <= 0:
            continue
        band_df = repo_df[repo_df["loc_band"] == band]
        if len(band_df) <= take:
            parts.append(band_df)
        else:
            parts.append(band_df.sample(n=take, random_state=seed + idx))

    if not parts:
        return repo_df.sample(n=target, random_state=seed)

    sampled = pd.concat(parts)
    if len(sampled) < target:
        remaining = repo_df.drop(sampled.index)
        extra = remaining.sample(n=target - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, extra])
    elif len(sampled) > target:
        sampled = sampled.sample(n=target, random_state=seed)
    return sampled


def _sample_repo_risk_stratified(repo_df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    if target <= 0:
        return repo_df.head(0)
    if len(repo_df) <= target:
        return repo_df

    risk_fraction = max(0.0, min(1.0, config.LLM_SAMPLE_RISK_FRACTION))
    risk_target = int(round(target * risk_fraction))
    if risk_fraction > 0 and target > 1 and risk_target == 0:
        risk_target = 1
    risk_target = min(risk_target, target)

    churn = repo_df.get("churn_12m", pd.Series([0] * len(repo_df), index=repo_df.index)).fillna(0)
    churn_threshold = None
    if not churn.empty:
        churn_quantile = max(0.0, min(1.0, config.LLM_SAMPLE_RISK_CHURN_QUANTILE))
        churn_threshold = churn.quantile(churn_quantile)
    churn_high = churn >= churn_threshold if churn_threshold and churn_threshold > 0 else False

    dominant_share = repo_df.get(
        "dominant_author_share", pd.Series([0] * len(repo_df), index=repo_df.index)
    ).fillna(0)
    single_dev = repo_df.get(
        "bus_factor_single_dev", pd.Series([False] * len(repo_df), index=repo_df.index)
    ).fillna(False)
    high_risk_mask = (single_dev == True) | (dominant_share >= 0.75)
    if isinstance(churn_high, pd.Series):
        high_risk_mask = high_risk_mask | churn_high

    high_risk_df = repo_df[high_risk_mask]
    risk_sample = (
        high_risk_df.sample(n=min(risk_target, len(high_risk_df)), random_state=seed)
        if risk_target > 0 and not high_risk_df.empty
        else repo_df.head(0)
    )

    remaining_target = target - len(risk_sample)
    if remaining_target <= 0:
        return risk_sample

    remaining_pool = repo_df.drop(risk_sample.index)
    if remaining_pool.empty:
        return risk_sample

    stratified = _sample_repo_by_loc_band(remaining_pool, remaining_target, seed + 1)
    return pd.concat([risk_sample, stratified])


def _stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    repos = sorted(df["repo"].dropna().unique().tolist())
    if not repos:
        return df.sample(n=sample_size, random_state=seed)

    targets = _allocate_repo_targets(repos, sample_size, config.LLM_SAMPLE_MIN_PER_REPO)
    parts: list[pd.DataFrame] = []

    for idx, repo in enumerate(repos):
        repo_df = df[df["repo"] == repo]
        target = targets.get(repo, 0)
        if target <= 0:
            continue
        if len(repo_df) <= target:
            parts.append(repo_df)
            continue
        parts.append(_sample_repo_by_loc_band(repo_df, target, seed + idx))

    if not parts:
        return df.sample(n=sample_size, random_state=seed)

    sampled = pd.concat(parts)
    if len(sampled) < sample_size:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(n=sample_size - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, extra])
    elif len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed)

    return sampled


def _risk_stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    repos = sorted(df["repo"].dropna().unique().tolist())
    if not repos:
        return df.sample(n=sample_size, random_state=seed)

    targets = _allocate_repo_targets(repos, sample_size, config.LLM_SAMPLE_MIN_PER_REPO)
    parts: list[pd.DataFrame] = []

    for idx, repo in enumerate(repos):
        repo_df = df[df["repo"] == repo]
        target = targets.get(repo, 0)
        if target <= 0:
            continue
        parts.append(_sample_repo_risk_stratified(repo_df, target, seed + idx))

    if not parts:
        return df.sample(n=sample_size, random_state=seed)

    sampled = pd.concat(parts)
    if len(sampled) < sample_size:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(n=sample_size - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, extra])
    elif len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed)

    return sampled


def _apply_sampling(df: pd.DataFrame) -> pd.DataFrame:
    sample_size = config.LLM_SAMPLE_SIZE
    if sample_size <= 0 or len(df) <= sample_size:
        return df
    strategy = config.LLM_SAMPLE_STRATEGY
    if strategy == "stratified":
        sampled = _stratified_sample(df, sample_size, config.LLM_SAMPLE_SEED)
    elif strategy == "risk_stratified":
        sampled = _risk_stratified_sample(df, sample_size, config.LLM_SAMPLE_SEED)
    elif strategy == "random":
        sampled = df.sample(n=sample_size, random_state=config.LLM_SAMPLE_SEED)
    else:
        print(f"[llm] Unknown LLM_SAMPLE_STRATEGY '{strategy}', falling back to random sampling.")
        sampled = df.sample(n=sample_size, random_state=config.LLM_SAMPLE_SEED)

    manifest = config.LLM_SAMPLE_MANIFEST
    if manifest:
        manifest_path = Path(manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "repo",
            "file_path",
            "absolute_path",
            "lines_of_code",
            "churn_12m",
            "unique_authors_12m",
            "dominant_author_share",
            "bus_factor_single_dev",
            "bus_factor_75_dominant_author",
        ]
        existing_cols = [col for col in cols if col in sampled.columns]
        sampled[existing_cols].to_csv(manifest_path, index=False)
        print(f"[llm] Wrote sample manifest to {manifest_path}")

    return sampled


def _load_existing_success(output_path: Path, model_name: str) -> set[tuple[str, str]]:
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_csv(output_path)
    except pd.errors.EmptyDataError:
        return set()
    if existing.empty:
        return set()
    if "llm_success" in existing.columns:
        success_mask = existing["llm_success"].astype(str).str.lower().isin({"true", "1", "yes"})
        existing = existing[success_mask]
    if "llm_model" in existing.columns and model_name:
        existing = existing[existing["llm_model"] == model_name]
    if not {"repo", "file_path"}.issubset(existing.columns):
        return set()
    return set(zip(existing["repo"], existing["file_path"]))


def _append_results(output_path: Path, rows: list[dict], header_written: bool) -> bool:
    if not rows:
        return header_written
    df = pd.DataFrame(rows)
    if df.empty:
        return header_written
    mode = "a" if header_written else "w"
    df.to_csv(output_path, index=False, mode=mode, header=not header_written)
    return True


def _maybe_sort_output(output_path: Path) -> None:
    if not output_path.exists():
        return
    try:
        df = pd.read_csv(output_path)
    except pd.errors.EmptyDataError:
        return
    if df.empty:
        return
    if {"repo", "file_path"}.issubset(df.columns):
        df.sort_values(["repo", "file_path"], inplace=True)
        df.to_csv(output_path, index=False)


def judge_file_openai(client, file_record: dict, tracker: CostTracker, model_name: str) -> dict:
    """Send a single file to an OpenAI-compatible API and return the structured result."""
    abs_path = Path(file_record["absolute_path"])
    try:
        code = abs_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return {
            "repo": file_record["repo"],
            "file_path": file_record["file_path"],
            "llm_model": model_name,
            "readability_score": None,
            "maintainability_risk": None,
            "sustainability_risk": None,
            "architectural_issues": [],
            "sustainability_flags": [],
            "reasoning": "",
            "code_truncated": False,
            "original_length": 0,
            "truncated_length": 0,
            "llm_success": False,
            "llm_error": f"File read failed: {exc}",
            "llm_response": "",
        }

    code, code_truncated, truncated_length, original_length = _truncate_code(code)
    context_block = _build_context_block(file_record)
    trunc_note = ""
    if code_truncated:
        trunc_note = f"NOTE: Source code truncated to {truncated_length} of {original_length} chars."
    user_content = f"{context_block}\n"
    if trunc_note:
        user_content += f"{trunc_note}\n"
    user_content += f"Source code:\n```{code}```"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_content,
        },
    ]

    raw_content = ""
    try:
        before_prompt = tracker.prompt_tokens
        before_completion = tracker.completion_tokens

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=0.1,
        )
        tracker.add_openai_usage(getattr(response, "usage", None))
        raw_content = response.choices[0].message.content or ""
        prompt_used = tracker.prompt_tokens - before_prompt
        completion_used = tracker.completion_tokens - before_completion
        try:
            parsed = _parse_json_response(raw_content)
        except Exception as exc:
            return {
                "repo": file_record["repo"],
                "file_path": file_record["file_path"],
                "llm_model": model_name,
                "readability_score": None,
                "maintainability_risk": None,
                "sustainability_risk": None,
                "architectural_issues": [],
                "sustainability_flags": [],
                "reasoning": "",
                "code_truncated": code_truncated,
                "original_length": original_length,
                "truncated_length": truncated_length,
                "llm_prompt_tokens": prompt_used,
                "llm_completion_tokens": completion_used,
                "llm_success": False,
                "llm_error": f"JSON parse failed: {exc}",
                "llm_response": raw_content,
            }

        valid, error, normalized = _validate_llm_response(parsed)
        if not valid:
            return {
                "repo": file_record["repo"],
                "file_path": file_record["file_path"],
                "readability_score": normalized["readability_score"],
                "maintainability_risk": normalized["maintainability_risk"],
                "sustainability_risk": normalized["sustainability_risk"],
                "architectural_issues": normalized["architectural_issues"],
                "sustainability_flags": normalized["sustainability_flags"],
                "reasoning": normalized["reasoning"],
                "llm_model": model_name,
                "code_truncated": code_truncated,
                "original_length": original_length,
                "truncated_length": truncated_length,
                "llm_prompt_tokens": prompt_used,
                "llm_completion_tokens": completion_used,
                "llm_success": False,
                "llm_error": error or "Missing/invalid field",
                "llm_response": raw_content,
            }

        return {
            "repo": file_record["repo"],
            "file_path": file_record["file_path"],
            "readability_score": normalized["readability_score"],
            "maintainability_risk": normalized["maintainability_risk"],
            "sustainability_risk": normalized["sustainability_risk"],
            "architectural_issues": normalized["architectural_issues"],
            "sustainability_flags": normalized["sustainability_flags"],
            "reasoning": normalized["reasoning"],
            "llm_model": model_name,
            "code_truncated": code_truncated,
            "original_length": original_length,
            "truncated_length": truncated_length,
            "llm_prompt_tokens": prompt_used,
            "llm_completion_tokens": completion_used,
            "llm_success": True,
            "llm_error": "",
            "llm_response": raw_content,
        }
    except Exception as exc:
        return {
            "repo": file_record["repo"],
            "file_path": file_record["file_path"],
            "llm_model": model_name,
            "readability_score": None,
            "maintainability_risk": None,
            "sustainability_risk": None,
            "architectural_issues": [],
            "sustainability_flags": [],
            "reasoning": "",
            "code_truncated": code_truncated,
            "original_length": original_length,
            "truncated_length": truncated_length,
            "llm_success": False,
            "llm_error": str(exc),
            "llm_response": raw_content,
        }


def judge_file_gemini(model, file_record: dict, tracker: CostTracker, model_name: str) -> dict:
    """Send a single file to the Gemini API and return the structured result."""
    abs_path = Path(file_record["absolute_path"])
    try:
        code = abs_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return {
            "repo": file_record["repo"],
            "file_path": file_record["file_path"],
            "llm_model": model_name,
            "readability_score": None,
            "maintainability_risk": None,
            "sustainability_risk": None,
            "architectural_issues": [],
            "sustainability_flags": [],
            "reasoning": "",
            "code_truncated": False,
            "original_length": 0,
            "truncated_length": 0,
            "llm_success": False,
            "llm_error": f"File read failed: {exc}",
            "llm_response": "",
        }

    context_block = _build_context_block(file_record)
    code, code_truncated, truncated_length, original_length = _truncate_code(code)
    trunc_note = ""
    if code_truncated:
        trunc_note = f"NOTE: Source code truncated to {truncated_length} of {original_length} chars."
    prompt = f"{SYSTEM_PROMPT}\n\n{context_block}\n"
    if trunc_note:
        prompt += f"{trunc_note}\n"
    prompt += f"Source code:\n```{code}```"

    raw_content = ""
    try:
        before_prompt = tracker.prompt_tokens
        before_completion = tracker.completion_tokens

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": config.LLM_MAX_TOKENS,
                "response_mime_type": "application/json",
            },
        )
        tracker.add_gemini_usage(getattr(response, "usage_metadata", None))
        raw_content = response.text or ""
        prompt_used = tracker.prompt_tokens - before_prompt
        completion_used = tracker.completion_tokens - before_completion
        try:
            parsed = _parse_json_response(raw_content)
        except Exception as exc:
            return {
                "repo": file_record["repo"],
                "file_path": file_record["file_path"],
                "llm_model": model_name,
                "readability_score": None,
                "maintainability_risk": None,
                "sustainability_risk": None,
                "architectural_issues": [],
                "sustainability_flags": [],
                "reasoning": "",
                "code_truncated": code_truncated,
                "original_length": original_length,
                "truncated_length": truncated_length,
                "llm_prompt_tokens": prompt_used,
                "llm_completion_tokens": completion_used,
                "llm_success": False,
                "llm_error": f"JSON parse failed: {exc}",
                "llm_response": raw_content,
            }

        valid, error, normalized = _validate_llm_response(parsed)
        if not valid:
            return {
                "repo": file_record["repo"],
                "file_path": file_record["file_path"],
                "readability_score": normalized["readability_score"],
                "maintainability_risk": normalized["maintainability_risk"],
                "sustainability_risk": normalized["sustainability_risk"],
                "architectural_issues": normalized["architectural_issues"],
                "sustainability_flags": normalized["sustainability_flags"],
                "reasoning": normalized["reasoning"],
                "llm_model": model_name,
                "code_truncated": code_truncated,
                "original_length": original_length,
                "truncated_length": truncated_length,
                "llm_prompt_tokens": prompt_used,
                "llm_completion_tokens": completion_used,
                "llm_success": False,
                "llm_error": error or "Missing/invalid field",
                "llm_response": raw_content,
            }

        return {
            "repo": file_record["repo"],
            "file_path": file_record["file_path"],
            "readability_score": normalized["readability_score"],
            "maintainability_risk": normalized["maintainability_risk"],
            "sustainability_risk": normalized["sustainability_risk"],
            "architectural_issues": normalized["architectural_issues"],
            "sustainability_flags": normalized["sustainability_flags"],
            "reasoning": normalized["reasoning"],
            "llm_model": model_name,
            "code_truncated": code_truncated,
            "original_length": original_length,
            "truncated_length": truncated_length,
            "llm_prompt_tokens": prompt_used,
            "llm_completion_tokens": completion_used,
            "llm_success": True,
            "llm_error": "",
            "llm_response": raw_content,
        }
    except Exception as exc:
        return {
            "repo": file_record["repo"],
            "file_path": file_record["file_path"],
            "llm_model": model_name,
            "readability_score": None,
            "maintainability_risk": None,
            "sustainability_risk": None,
            "architectural_issues": [],
            "sustainability_flags": [],
            "reasoning": "",
            "code_truncated": code_truncated,
            "original_length": original_length,
            "truncated_length": truncated_length,
            "llm_success": False,
            "llm_error": str(exc),
            "llm_response": raw_content,
        }


def run_llm_judge(git_metrics_csv: Path, output_path: Path | None = None) -> Path:
    """
    Iterate over representative files and score them with the LLM.

    Returns the path to the generated llm_metrics CSV.
    """
    output = output_path or (config.RESULTS_DIR / "llm_metrics.csv")
    tracker = CostTracker()
    if output.exists() and not config.LLM_RESUME:
        output.unlink()

    analysis_date_utc = config.now_utc_iso()
    llm_provider = config.LLM_PROVIDER
    repo_sha_cache: dict[str, str | None] = {}

    def _repo_head_sha(repo_name: str) -> str | None:
        if repo_name in repo_sha_cache:
            return repo_sha_cache[repo_name]
        if not repo_name:
            repo_sha_cache[repo_name] = None
            return None
        sha = config.git_head_sha(config.RAW_REPOS_DIR / repo_name)
        repo_sha_cache[repo_name] = sha
        return sha

    files_df = _load_git_metrics_df(git_metrics_csv)
    sampled_df = _apply_sampling(files_df)
    if len(sampled_df) < len(files_df):
        print(
            "[llm] Sampling "
            f"{len(sampled_df)}/{len(files_df)} files "
            f"(strategy={config.LLM_SAMPLE_STRATEGY}, seed={config.LLM_SAMPLE_SEED}, "
            f"min_per_repo={config.LLM_SAMPLE_MIN_PER_REPO})"
        )
    files = sampled_df.to_dict(orient="records")
    print(f"[llm] Scoring {len(files)} representative files")

    results: List[dict] = []
    header_written = output.exists() and output.stat().st_size > 0
    write_every = max(config.LLM_WRITE_EVERY, 0)

    if config.LLM_PROVIDER == "gemini":
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY must be set before running the LLM judge with Gemini.")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("google-generativeai must be installed to use Gemini.") from exc

        # Force v1 so 1.5 models resolve; default is v1beta which 404s for 1.5.
        try:
            # Newer SDKs support api_version; fall back silently if not available.
            genai.configure(api_key=config.GEMINI_API_KEY, api_version="v1")
        except TypeError:
            genai.configure(api_key=config.GEMINI_API_KEY)

        preferred_model = config.GEMINI_MODEL or "gemini-pro"

        # Build a small fallback list to handle model availability and naming variants.
        candidates = []
        seen: set[str] = set()
        base_candidates = [
            preferred_model,
            f"{preferred_model}-latest"
            if not preferred_model.endswith("-latest")
            and (preferred_model.endswith("-flash") or preferred_model.endswith("-pro"))
            else None,
            "gemini-1.0-pro",
            "gemini-pro",
            "gemini-pro-latest",
            "gemini-flash-latest",
        ]
        for name in base_candidates:
            if name and name not in seen:
                seen.add(name)
                candidates.append(name)
            # Also try the fully-qualified "models/..." variant that v1beta expects.
            fq = f"models/{name}" if name else None
            if fq and fq not in seen:
                seen.add(fq)
                candidates.append(fq)

        model = None
        model_name = preferred_model
        last_exc: Exception | None = None
        for name in candidates:
            try:
                model = genai.GenerativeModel(name)
                model_name = name
                if name != preferred_model:
                    print(f"[llm] Gemini model '{preferred_model}' unavailable; fell back to '{name}'.")
                break
            except Exception as exc:
                last_exc = exc
                continue

        if model is None:
            raise RuntimeError(
                f"Failed to initialize Gemini model. Tried: {', '.join(candidates)}; last error: {last_exc}"
            )

        if config.LLM_RESUME:
            existing = _load_existing_success(output, model_name)
            if existing:
                before = len(files)
                files = [record for record in files if (record["repo"], record["file_path"]) not in existing]
                skipped = before - len(files)
                if skipped:
                    print(f"[llm] Skipping {skipped} files already scored for model {model_name}.")

        for record in files:
            result = judge_file_gemini(model, record, tracker, model_name)
            repo_name = record.get("repo", "")
            result["analysis_date_utc"] = analysis_date_utc
            result["repo_head_sha"] = _repo_head_sha(repo_name)
            result["llm_provider"] = llm_provider
            result["prompt_version"] = PROMPT_VERSION
            if result.get("llm_success"):
                print(f"[llm] OK: {record['repo']}/{record['file_path']}")
            else:
                print(f"[llm] FAIL: {record['repo']}/{record['file_path']} ({result.get('llm_error')})")
            results.append(result)
            if write_every and len(results) >= write_every:
                header_written = _append_results(output, results, header_written)
                results.clear()

    else:
        if config.LLM_PROVIDER not in {"openai", "ollama"}:
            raise RuntimeError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai must be installed to use the OpenAI provider.") from exc
        if config.LLM_PROVIDER == "ollama":
            base_url = config.OLLAMA_BASE_URL or config.OLLAMA_HOST
            if base_url:
                base_url = base_url.rstrip("/")
                if not base_url.endswith("/v1"):
                    base_url = f"{base_url}/v1"
            if not base_url:
                raise RuntimeError("OLLAMA_HOST or OLLAMA_BASE_URL must be set for the ollama provider.")
            api_key = config.OLLAMA_API_KEY or "ollama"
            model_name = config.OLLAMA_MODEL or config.OPENAI_MODEL
        else:
            if not config.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY must be set before running the LLM judge.")
            base_url = config.OPENAI_BASE_URL
            api_key = config.OPENAI_API_KEY
            model_name = config.OPENAI_MODEL

        if not model_name:
            raise RuntimeError("LLM model name must be set before running the LLM judge.")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        if config.LLM_RESUME:
            existing = _load_existing_success(output, model_name)
            if existing:
                before = len(files)
                files = [record for record in files if (record["repo"], record["file_path"]) not in existing]
                skipped = before - len(files)
                if skipped:
                    print(f"[llm] Skipping {skipped} files already scored for model {model_name}.")

        for record in files:
            result = judge_file_openai(client, record, tracker, model_name)
            repo_name = record.get("repo", "")
            result["analysis_date_utc"] = analysis_date_utc
            result["repo_head_sha"] = _repo_head_sha(repo_name)
            result["llm_provider"] = llm_provider
            result["prompt_version"] = PROMPT_VERSION
            if result.get("llm_success"):
                print(f"[llm] OK: {record['repo']}/{record['file_path']}")
            else:
                print(f"[llm] FAIL: {record['repo']}/{record['file_path']} ({result.get('llm_error')})")
            results.append(result)
            if write_every and len(results) >= write_every:
                header_written = _append_results(output, results, header_written)
                results.clear()

    header_written = _append_results(output, results, header_written)
    if config.LLM_SORT_OUTPUT:
        _maybe_sort_output(output)
    print(
        f"[llm] Wrote LLM metrics to {output} "
        f"(prompt_tokens={tracker.prompt_tokens}, completion_tokens={tracker.completion_tokens})"
    )
    return output


def main() -> None:
    git_metrics_csv = config.RESULTS_DIR / "git_metrics.csv"
    run_llm_judge(git_metrics_csv)


if __name__ == "__main__":
    main()
