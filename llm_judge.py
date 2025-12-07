"""
LLM-based "Augmented Reviewer" for representative files.

Reads git_metrics.csv to locate candidate files, submits source code to the LLM,
and stores structured JSON results in llm_metrics.csv.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

import config

SYSTEM_PROMPT = """
You are evaluating code maintainability for a research study.
Return ONLY a single JSON object with the following exact schema and no extra text:
{
  "readability_score": <integer 1-10>,
  "maintainability_risk": "Low" | "Med" | "High",
  "architectural_issues": ["list of concise issue statements"],
  "reasoning": "one paragraph explaining the assessment"
}
The JSON must be valid and self-contained. Avoid markdown.
""".strip()


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


def _load_git_metrics(git_metrics_csv: Path) -> List[dict]:
    df = pd.read_csv(git_metrics_csv)
    required_cols = {"repo", "file_path", "absolute_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"git_metrics.csv is missing columns: {', '.join(sorted(missing))}")
    return df.to_dict(orient="records")


def _parse_json_response(content: str) -> Dict:
    """Parse JSON while stripping stray whitespace and markdown fences."""
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def judge_file_openai(client, file_record: dict, tracker: CostTracker) -> dict:
    """Send a single file to the OpenAI API and return the structured result."""
    abs_path = Path(file_record["absolute_path"])
    try:
        code = abs_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return {
            "repo": file_record["repo"],
            "file_path": file_record["file_path"],
            "llm_success": False,
            "llm_error": f"File not found: {abs_path}",
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Repository: {file_record['repo']}\n"
                f"File: {file_record['file_path']}\n"
                "Produce the JSON response for this file only.\n"
                f"Source code:\n```{code}```"
            ),
        },
    ]

    try:
        before_prompt = tracker.prompt_tokens
        before_completion = tracker.completion_tokens

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=0.1,
        )
        tracker.add_openai_usage(getattr(response, "usage", None))
        raw_content = response.choices[0].message.content
        parsed = _parse_json_response(raw_content)

        prompt_used = tracker.prompt_tokens - before_prompt
        completion_used = tracker.completion_tokens - before_completion

        return {
            "repo": file_record["repo"],
            "file_path": file_record["file_path"],
            "readability_score": parsed.get("readability_score"),
            "maintainability_risk": parsed.get("maintainability_risk"),
            "architectural_issues": parsed.get("architectural_issues"),
            "reasoning": parsed.get("reasoning"),
            "llm_model": config.OPENAI_MODEL,
            "llm_prompt_tokens": prompt_used,
            "llm_completion_tokens": completion_used,
            "llm_success": True,
            "llm_error": "",
        }
    except Exception as exc:
        return {
            "repo": file_record["repo"],
            "file_path": file_record["file_path"],
            "llm_success": False,
            "llm_error": str(exc),
        }


def judge_file_gemini(model, file_record: dict, tracker: CostTracker, model_name: str) -> dict:
    """Send a single file to the Gemini API and return the structured result."""
    abs_path = Path(file_record["absolute_path"])
    try:
        code = abs_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return {
            "repo": file_record["repo"],
            "file_path": file_record["file_path"],
            "llm_success": False,
            "llm_error": f"File not found: {abs_path}",
        }

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Repository: {file_record['repo']}\n"
        f"File: {file_record['file_path']}\n"
        "Produce the JSON response for this file only.\n"
        f"Source code:\n```{code}```"
    )

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
        raw_content = response.text
        parsed = _parse_json_response(raw_content)

        prompt_used = tracker.prompt_tokens - before_prompt
        completion_used = tracker.completion_tokens - before_completion

        return {
            "repo": file_record["repo"],
            "file_path": file_record["file_path"],
            "readability_score": parsed.get("readability_score"),
            "maintainability_risk": parsed.get("maintainability_risk"),
            "architectural_issues": parsed.get("architectural_issues"),
            "reasoning": parsed.get("reasoning"),
            "llm_model": model_name,
            "llm_prompt_tokens": prompt_used,
            "llm_completion_tokens": completion_used,
            "llm_success": True,
            "llm_error": "",
        }
    except Exception as exc:
        return {
            "repo": file_record["repo"],
            "file_path": file_record["file_path"],
            "llm_success": False,
            "llm_error": str(exc),
        }


def run_llm_judge(git_metrics_csv: Path, output_path: Path | None = None) -> Path:
    """
    Iterate over representative files and score them with the LLM.

    Returns the path to the generated llm_metrics CSV.
    """
    output = output_path or (config.RESULTS_DIR / "llm_metrics.csv")
    tracker = CostTracker()

    files = _load_git_metrics(git_metrics_csv)
    print(f"[llm] Scoring {len(files)} representative files")

    results: List[dict] = []

    if config.LLM_PROVIDER == "gemini":
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY must be set before running the LLM judge with Gemini.")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("google-generativeai must be installed to use Gemini.") from exc

        genai.configure(api_key=config.GEMINI_API_KEY)

        preferred_model = config.GEMINI_MODEL or "gemini-pro"
        candidates = [preferred_model]
        if preferred_model != "gemini-pro":
            candidates.append("gemini-pro")

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

        for record in files:
            result = judge_file_gemini(model, record, tracker, model_name)
            if result.get("llm_success"):
                print(f"[llm] OK: {record['repo']}/{record['file_path']}")
            else:
                print(f"[llm] FAIL: {record['repo']}/{record['file_path']} ({result.get('llm_error')})")
            results.append(result)

    else:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY must be set before running the LLM judge.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai must be installed to use the OpenAI provider.") from exc

        client = OpenAI(api_key=config.OPENAI_API_KEY)

        for record in files:
            result = judge_file_openai(client, record, tracker)
            if result.get("llm_success"):
                print(f"[llm] OK: {record['repo']}/{record['file_path']}")
            else:
                print(f"[llm] FAIL: {record['repo']}/{record['file_path']} ({result.get('llm_error')})")
            results.append(result)

    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values(["repo", "file_path"], inplace=True)
    df.to_csv(output, index=False)
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
