"""
Generate refactored file proposals from Sonar hotspots.

This script does NOT modify repositories and does NOT run verification.
It only asks the LLM for refactored code and writes the generated files to
an output folder for manual review/testing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

import pandas as pd
from openai import OpenAI

from pipeline.configs import config


TARGET_LANGUAGES = {"java", "javascript", "python", "typescript", "go"}
DEFAULT_MAX_FILES_PER_REPO = 10
DEFAULT_MIN_CODE_SMELLS = 5
DEFAULT_MAX_NCLOC = 400
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4000


SYSTEM_PROMPT = """You are an expert code refactoring assistant.

Task: improve maintainability while preserving exact behavior.

Hard constraints:
1. Preserve public API and functionality.
2. Do not remove features.
3. Do not suppress warnings/rules.
4. Do not modify build/test/config files or imports that change behavior.

Output:
- Return only the complete refactored file content in a single fenced code block.
"""


USER_PROMPT_TEMPLATE = """Refactor this file to reduce maintainability issues.

Relative path: {file_path}
Language: {language}

Sonar metrics:
- Code Smells: {code_smells}
- Cognitive Complexity: {cognitive_complexity}
- Technical Debt (min): {sqale_index}
- NCLOC: {ncloc}
- Duplicated Lines Density: {duplicated_lines_density}

```{language}
{code}
```
"""


def detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".py":
        return "python"
    if ext == ".java":
        return "java"
    if ext in {".js", ".jsx"}:
        return "javascript"
    if ext in {".ts", ".tsx"}:
        return "typescript"
    if ext == ".go":
        return "go"
    return "unknown"


def extract_code_from_response(response: str, language: str) -> str | None:
    patterns = [
        rf"```{language}\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    stripped = response.strip()
    if stripped:
        return stripped
    return None


def normalize_rel_path(path: str) -> str | None:
    normalized = str(path).replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    if not normalized or ".." in Path(normalized).parts:
        return None
    return normalized


def select_candidates(
    sonar_df: pd.DataFrame,
    repos: list[str],
    max_files_per_repo: int,
    min_code_smells: int,
    max_ncloc: int,
) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    for repo in repos:
        repo_df = sonar_df[sonar_df["repo"] == repo].copy()
        if repo_df.empty:
            continue

        repo_df["language"] = repo_df["file_path"].astype(str).map(detect_language)
        filtered = repo_df[
            (repo_df["language"].isin(TARGET_LANGUAGES))
            & (repo_df["sonar_code_smells"].fillna(0) >= min_code_smells)
            & (repo_df["sonar_ncloc"].fillna(0) <= max_ncloc)
        ].copy()

        if filtered.empty:
            continue

        filtered["priority_score"] = (
            filtered["sonar_code_smells"].fillna(0) * 1000
            + filtered["sonar_cognitive_complexity"].fillna(0) * 10
            + filtered["sonar_sqale_index"].fillna(0)
        )
        filtered = filtered.sort_values(
            ["priority_score", "sonar_code_smells", "sonar_cognitive_complexity"],
            ascending=False,
        )
        selected_parts.append(filtered.head(max_files_per_repo))

    if not selected_parts:
        return pd.DataFrame()
    return pd.concat(selected_parts, ignore_index=True)


def build_client() -> OpenAI:
    base_url = f"{config.LLM_BASE_URL}/v1" if "localhost" in config.LLM_BASE_URL else config.LLM_BASE_URL
    return OpenAI(api_key=config.LLM_API_KEY or "not-needed", base_url=base_url)


def generate_one_file(
    client: OpenAI,
    model: str,
    repo: str,
    rel_path: str,
    row: pd.Series,
    output_root: Path,
    temperature: float,
    max_tokens: int,
) -> Path | None:
    source_path = config.RAW_REPOS_DIR / repo / rel_path
    if not source_path.exists():
        print(f"  ✗ Missing source file: {repo}/{rel_path}")
        return None

    code = source_path.read_text(encoding="utf-8", errors="replace")
    language = detect_language(rel_path)
    prompt = USER_PROMPT_TEMPLATE.format(
        file_path=rel_path,
        language=language,
        code_smells=row.get("sonar_code_smells", "N/A"),
        cognitive_complexity=row.get("sonar_cognitive_complexity", "N/A"),
        sqale_index=row.get("sonar_sqale_index", "N/A"),
        ncloc=row.get("sonar_ncloc", "N/A"),
        duplicated_lines_density=row.get("sonar_duplicated_lines_density", "N/A"),
        code=code,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        print(f"  ✗ LLM call failed: {repo}/{rel_path}: {exc}")
        return None

    raw = response.choices[0].message.content or ""
    refactored = extract_code_from_response(raw, language)
    if not refactored:
        print(f"  ✗ Could not extract code: {repo}/{rel_path}")
        return None

    out_path = output_root / repo / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(refactored, encoding="utf-8")
    print(f"  ✓ Wrote: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate refactored file proposals from Sonar hotspots")
    parser.add_argument("--repos", type=str, default=None, help="Comma-separated repos (default: auto from Sonar)")
    parser.add_argument("--max-repos", type=int, default=3, help="Auto-selected repo count if --repos omitted")
    parser.add_argument("--max-files-per-repo", type=int, default=DEFAULT_MAX_FILES_PER_REPO)
    parser.add_argument("--min-code-smells", type=int, default=DEFAULT_MIN_CODE_SMELLS)
    parser.add_argument("--max-ncloc", type=int, default=DEFAULT_MAX_NCLOC)
    parser.add_argument("--model", type=str, default=config.LLM_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output root directory (default: data/results/refactored_file_proposals/<session_id>)",
    )
    args = parser.parse_args()

    sonar_path = config.sonar_metrics_path()
    if not sonar_path.exists():
        raise FileNotFoundError(f"Sonar metrics not found: {sonar_path}")
    sonar_df = pd.read_csv(sonar_path)
    if "repo" not in sonar_df.columns or "file_path" not in sonar_df.columns:
        raise ValueError("sonar_metrics.csv must contain repo and file_path columns")

    if args.repos:
        repos = [item.strip() for item in args.repos.split(",") if item.strip()]
    else:
        repo_rank = (
            sonar_df.groupby("repo")["sonar_code_smells"]
            .sum()
            .sort_values(ascending=False)
        )
        repos = repo_rank.head(args.max_repos).index.tolist()

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) if args.output_dir else (config.RESULTS_DIR / "refactored_file_proposals" / session_id)
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = select_candidates(
        sonar_df=sonar_df,
        repos=repos,
        max_files_per_repo=max(1, args.max_files_per_repo),
        min_code_smells=max(0, args.min_code_smells),
        max_ncloc=max(1, args.max_ncloc),
    )

    print("=" * 70)
    print("REFACTOR FILE GENERATOR")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Repos: {repos}")
    print(f"Candidates: {len(candidates)}")
    print(f"Output dir: {output_root}")

    if candidates.empty:
        print("No candidate files found with current filters.")
        return

    client = build_client()
    written = 0
    generated_locations: list[dict[str, str]] = []
    for _, row in candidates.iterrows():
        repo = str(row["repo"])
        rel_path = normalize_rel_path(str(row["file_path"]))
        if not rel_path:
            continue
        print(f"\n[{written + 1}/{len(candidates)}] {repo}/{rel_path}")
        out_path = generate_one_file(
            client=client,
            model=args.model,
            repo=repo,
            rel_path=rel_path,
            row=row,
            output_root=output_root,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if out_path is not None:
            written += 1
            generated_locations.append(
                {
                    "repo": repo,
                    "file_path": rel_path,
                    "output_path": str(out_path),
                }
            )

    if generated_locations:
        manifest_path = output_root / "refactored_file_locations.csv"
        pd.DataFrame(generated_locations).to_csv(manifest_path, index=False)
    else:
        manifest_path = output_root / "refactored_file_locations.csv"

    print("\n" + "=" * 70)
    print(f"Generated refactored files: {written}")
    print(f"Output dir: {output_root}")
    print(f"Locations manifest: {manifest_path}")
    if generated_locations:
        print("Refactored file locations:")
        for item in generated_locations:
            print(f"  - {item['output_path']}")


if __name__ == "__main__":
    main()
