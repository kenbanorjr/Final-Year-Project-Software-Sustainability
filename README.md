# Software Sustainability Evaluation Pipeline

This repository runs and analyzes a comparative sustainability study across three dimensions:

- Git mining (churn, contributor concentration, bus-factor proxies)
- SonarQube file-level static metrics
- LLM file-level assessment

## Canonical Core Run

From repo root:

```bash
python -m pipeline.main
```

This executes the core runtime pipeline and writes:

- `data/results/git/git_metrics.csv`
- `data/results/sonar/sonar_metrics.csv`
- `data/results/llm/llm_metrics_<model>_runNNN.csv`
- `data/results/merged/final_dataset.csv`

## Project Structure

- `pipeline/`: core runtime implementation (`main`, miner, sonar, llm judge, validation, config code)
- `experiments/`: experimental workflows (`holistic_evaluator`, `refactoring_study`)
- `analysis/`: RQ analysis scripts, figures, and tables
- `ui/`: optional web interface
- `data/raw_repos/`: cloned repositories (generated)
- `data/results/`: pipeline outputs (generated)

## Environment Setup

Use the provided environment scripts (real files):

- PowerShell: `configs/env.ps1`
- Bash: `configs/env.sh`

Required variables:

- SonarQube: `SONAR_HOST_URL`, `SONAR_TOKEN`
- LLM: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL` (and `LLM_API_KEY` for hosted APIs)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Core Pipeline Steps

`main.py` runs:

1. Clone repositories
2. SonarQube metrics collection
3. Git mining
4. LLM judging
5. Dataset merge
6. Optional repo summary (`--with-repo-summary`)
7. Optional validation (`VALIDATE_OUTPUTS=true`)

## Research Extras

These are not required for the core pipeline:

- Holistic evaluator: `python -m experiments.holistic_evaluator`
- Refactoring study: `python -m experiments.refactoring_study`
- Process sustainability analysis: `python -m analysis.rq_process_sustainability`
- Process visualizations: `python -m analysis.rq_visualizations`
- RQ1 analysis: `python -m analysis.rq1_analysis.scripts.rq1_full_analysis`

Holistic evaluator output:
- `data/results/holistic/holistic_assessments_<model>_runNNN.csv`
- resume supported with `--resume` or `HOLISTIC_RESUME=true`

## Documentation

- `REQUIREMENTS`: hardware/software requirements
- `INSTALL`: installation and smoke test
- `REPLICATION_GUIDE`: end-to-end reproduction instructions
- `LICENSE`: license terms
