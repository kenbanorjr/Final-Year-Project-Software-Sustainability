# Software Sustainability Evaluation Pipeline

This project automates a comparative study of open-source repositories across three dimensions:

- Git mining (churn, bus-factor)
- SonarQube baseline metrics (complexity, technical debt, violations)
- LLM-based “augmented reviewer” scoring per file

## Prerequisites

- Python 3.10+
- SonarQube running at `http://localhost:9000` (default; override with `SONAR_HOST_URL`)
- `sonar-scanner` available on PATH (or set `SONAR_SCANNER`)
- LLM API access (Gemini or OpenAI)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All configuration lives in `config.py` and environment variables. Key env vars:

- `SONAR_HOST_URL` (default `http://localhost:9000`)
- `SONAR_TOKEN` (SonarQube token)
- `LLM_PROVIDER` (`gemini` or `openai`; default `openai`)
- `GEMINI_API_KEY` / `GEMINI_MODEL` (if using Gemini; default model `gemini-pro` for widest availability)
- `OPENAI_API_KEY` / `OPENAI_MODEL` (if using OpenAI)
- Optional: `LLM_MAX_TOKENS`, `SONAR_SCANNER`

Repository lists (ACTIVE/STAGNANT) are defined in `config.py`.

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
4. LLM judge (`llm_judge.py`): structured JSON assessment per representative file.
5. Merge CSVs into `data/results/final_dataset.csv`.

Intermediate outputs:

- `data/results/git_metrics.csv`
- `data/results/sonar_metrics.csv`
- `data/results/llm_metrics.csv`
