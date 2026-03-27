# Signal 38

> **The 38th parallel.** Real-time North Korean military activity risk analysis — fine-tuned LFM2-350M, running entirely in the browser via WebGPU.

[![GitHub Pages](https://img.shields.io/badge/demo-live-brightgreen?logo=github)](https://signal38.github.io)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?logo=python)](https://www.python.org)
[![Model: LFM2-350M](https://img.shields.io/badge/model-LFM2--350M-purple)](https://huggingface.co/LiquidAI/LFM2-350M)
[![Data: GDELT](https://img.shields.io/badge/data-GDELT%20v2-orange)](https://www.gdeltproject.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What it does

Signal 38 ingests weekly clusters of GDELT v2 events involving North Korean military actors and produces structured risk assessments: escalation level (1–5), situation summary, key actors, watch indicators, and projected trajectories.

The model runs fully client-side via WebGPU — no server, no API key, no latency cost.

## How it works

```
GDELT v2 events (11 years, NK military CAMEO codes)
  → weekly clusters (event counts, Goldstein scale, tone, actor codes)
  → Claude-labeled risk assessments (knowledge distillation)
  → three models evaluated:
      1. Naive baseline    — Goldstein-scale threshold rule
      2. Classical ML      — XGBoost on GDELT features
      3. LFM2-350M QLoRA  — fine-tuned on 463 labeled examples
  → LoRA adapter merged → ONNX int4 export → browser inference via WebGPU
```

## Models

| Model | Approach | Escalation MAE | Valid JSON |
|-------|----------|---------------|-----------|
| Naive baseline | Goldstein threshold rule | TBD | n/a |
| XGBoost | GDELT feature vector | TBD | n/a |
| LFM2-350M (fine-tuned) | QLoRA knowledge distillation | TBD | TBD |

*Results populated by `03_evaluate.ipynb` — see `data/outputs/results.json`.*

## Notebooks

Run in order. Notebooks 02–04 require a T4 GPU runtime. All publish their artifacts back to this repo automatically.

| Notebook | What it does | Runtime | |
|----------|-------------|---------|---|
| [`00_acled_labels.ipynb`](notebooks/00_acled_labels.ipynb) | ACLED ground truth labels (optional) | CPU, ~2 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/signal38/signal38.github.io/blob/main/notebooks/00_acled_labels.ipynb) |
| [`01_baseline.ipynb`](notebooks/01_baseline.ipynb) | Naive rule + XGBoost baseline | CPU, ~5 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/signal38/signal38.github.io/blob/main/notebooks/01_baseline.ipynb) |
| [`02_finetune.ipynb`](notebooks/02_finetune.ipynb) | LFM2-350M QLoRA fine-tuning | T4 GPU, ~20 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/signal38/signal38.github.io/blob/main/notebooks/02_finetune.ipynb) |
| [`03_evaluate.ipynb`](notebooks/03_evaluate.ipynb) | Three-model evaluation + results export | T4 GPU, ~10 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/signal38/signal38.github.io/blob/main/notebooks/03_evaluate.ipynb) |
| [`04_export_onnx.ipynb`](notebooks/04_export_onnx.ipynb) | Merge adapter → ONNX int4 → HuggingFace Hub | T4 GPU, ~15 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/signal38/signal38.github.io/blob/main/notebooks/04_export_onnx.ipynb) |

## Repo structure

```
signal38.github.io/
├── notebooks/          # Colab-ready pipeline notebooks
├── scripts/            # Shared helpers (colab_utils, features, metrics)
├── src/                # App source (WebGPU inference, UI)
├── docs/               # GitHub Pages site
├── data/
│   ├── clusters/       # Weekly GDELT event clusters
│   ├── labeled/        # Claude-generated risk assessments
│   ├── training/       # Train / val / test splits
│   └── outputs/        # Evaluation results (published by notebooks)
└── models/             # LoRA adapter weights (published by notebook 02)
```

## Setup

```bash
pip install -r requirements.txt
```

Colab notebooks are self-contained. Open any notebook via the badge above, connect a T4 runtime, and run all cells.

## Colab setup

The notebooks share common setup via `scripts/colab_utils.py`. Notebooks that publish artifacts back to this repo require a `GITHUB_TOKEN` Colab secret. Notebook 00 additionally requires `ACLED_EMAIL` and `ACLED_KEY`.

**Setting up the GitHub token (one-time):**

Create a [fine-grained personal access token](https://github.com/settings/tokens?type=beta) scoped to the `signal38` organization with **Contents: Read and Write** permission. Then add it to Colab: open the key icon in the left sidebar → **Secrets** → **Add new secret**, name it `GITHUB_TOKEN`, paste the token, and enable notebook access.

<img src="docs/assets/colab-secrets.png" width="420" alt="Colab Secrets panel showing GITHUB_TOKEN" />

For ACLED credentials (notebook 00), register at [acleddata.com](https://acleddata.com/register/) and add `ACLED_EMAIL` and `ACLED_KEY` as Colab secrets.

For HuggingFace Hub (notebook 04), create a write token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and add it as `HF_TOKEN`.

## Team

- **Diya Mirji** — [@dvm14](https://github.com/dvm14)
- **Jonas Neves** — [@jonasneves](https://github.com/jonasneves)
- **Mike Saju** — [@Michaelsaju1](https://github.com/Michaelsaju1)

Built for **AIPI 540.01 — Deep Learning**, Spring 2026, Duke University AIPI Program.

## License

MIT
