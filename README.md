# Software Sustainability Evaluation Pipeline

This project automates a comparative study of open-source repositories across three dimensions:

- Git mining (churn, bus-factor)
- SonarQube baseline metrics (complexity, technical debt, violations)
- LLM-based “augmented reviewer” scoring per file

## Prerequisites

- Python 3.10+
- SonarQube running at `http://localhost:9000` (default; override with `SONAR_HOST_URL`)
- `sonar-scanner` available on PATH (or set `SONAR_SCANNER`)
- LLM API access (Gemini, OpenAI, or Ollama local server)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
python main.py
python visualize_metrics.py
python visualize_sonar_metrics.py
```

## Configuration

All configuration lives in `config.py` and environment variables.

Required environment variables:

- SonarQube: `SONAR_HOST_URL`, `SONAR_TOKEN`
- OpenAI: `OPENAI_API_KEY` (if using OpenAI)
- Gemini: `GEMINI_API_KEY` (if using Gemini)

Other key env vars:

- `LLM_PROVIDER` (`gemini`, `openai`, or `ollama`; default `openai`)
- `GEMINI_MODEL` (if using Gemini; default model `gemini-pro` for widest availability)
- `OPENAI_MODEL` (if using OpenAI)
- `OPENAI_BASE_URL` (optional; use for OpenAI-compatible local servers such as vLLM)
- `OLLAMA_HOST` / `OLLAMA_MODEL` (if using Ollama; default host `http://localhost:11434`)
- `OLLAMA_API_KEY` / `OLLAMA_BASE_URL` (optional; advanced overrides)
- `LLM_SAMPLE_SIZE` (optional; set > 0 to sample files for LLM evaluation)
- `LLM_SAMPLE_SEED` (optional; default `42`)
- `LLM_SAMPLE_STRATEGY` (`stratified`, `risk_stratified`, or `random`; default `stratified`)
- `LLM_SAMPLE_MIN_PER_REPO` (optional; ensure each repo gets at least N files)
- `LLM_SAMPLE_RISK_FRACTION` (optional; fraction of sample reserved for high-risk files)
- `LLM_SAMPLE_RISK_CHURN_QUANTILE` (optional; churn percentile for risk selection)
- `LLM_SAMPLE_MANIFEST` (optional; write selected sample rows to a CSV)
- `LLM_RESUME` (optional; default `true`, skip files already scored in llm_metrics.csv)
- `LLM_WRITE_EVERY` (optional; flush results to CSV every N files)
- `LLM_SORT_OUTPUT` (optional; default `true`, sort output by repo/file_path)
- Optional: `LLM_MAX_TOKENS`, `SONAR_SCANNER`

Repository lists (ACTIVE/STAGNANT) are defined in `config.py`. These user-provided
labels are stored as `seed_category` and are not used as ground truth. The pipeline
computes `activity_label` from mined activity (repo commits and unique authors) and
uses it in analysis for reproducibility.

### Using Ollama locally (example: qwen2.5-coder:7b)

If you want to use a local model like `qwen2.5-coder:7b` via Ollama, run the model
and point the pipeline at Ollama's OpenAI-compatible endpoint:

```bash
ollama run qwen2.5-coder:7b
export LLM_PROVIDER=ollama
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="qwen2.5-coder:7b"
```

## Running the pipeline

From the repo root:

```bash
export SONAR_HOST_URL="http://localhost:9000"
export SONAR_TOKEN="your_sonar_token"
export LLM_PROVIDER=gemini
export GEMINI_API_KEY="your_gemini_key"
# or for OpenAI:
# export OPENAI_API_KEY="your_openai_key"

python main.py
```

Pipeline steps (executed by `main.py`):

1. Clone repositories into `data/raw_repos/`.
2. Git mining (`miner.py`): representative files (.py/.java, 50–400 LOC), churn (12 months), added/deleted lines, dominant-author share (75% bus factor), single-dev bus factor, repo commit frequency, unique authors.
3. SonarQube scan (`sonar_runner.py`): collect complexity, sqale_index, violations, code_smells, bugs, vulnerabilities, duplication (blocks and density), test success density per file.
4. LLM judge (`llm_judge.py`): structured JSON assessment per representative file. The prompt includes
   contributor stats from git mining and a README snippet (if present) to add social/context signals.
5. Merge CSVs into `data/results/final_dataset.csv`.

Optional: `llm_repo_summary.py` produces repo-level LLM summaries using aggregated file-level
LLM results plus git/Sonar summaries. Outputs `data/results/llm_repo_summary.csv`.

Intermediate outputs:

- `data/results/git_metrics.csv`
- `data/results/sonar_metrics.csv`
- `data/results/llm_metrics.csv`

### LLM sampling (recommended for large datasets)

Set `LLM_SAMPLE_SIZE` to evaluate a reproducible subset of files while keeping git and Sonar
metrics for the full dataset. The default `stratified` strategy balances samples across repositories
and LOC bands. `risk_stratified` additionally reserves a fraction of each repo's sample for
high-risk files (bus factor, dominant author share, or high churn). Use `LLM_SAMPLE_SEED`
to make the sample repeatable.
