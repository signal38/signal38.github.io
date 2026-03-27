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

All notebooks run on a free Colab T4 GPU and publish their artifacts back to this repo automatically.

| Notebook | What it does | Runtime | |
|----------|-------------|---------|---|
| [`01_baseline.ipynb`](notebooks/01_baseline.ipynb) | Naive rule + XGBoost baseline | CPU, ~5 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/signal38/signal38.github.io/blob/main/notebooks/01_baseline.ipynb) |
| [`02_finetune.ipynb`](notebooks/02_finetune.ipynb) | LFM2-350M QLoRA fine-tuning | T4 GPU, ~20 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/signal38/signal38.github.io/blob/main/notebooks/02_finetune.ipynb) |
| [`03_evaluate.ipynb`](notebooks/03_evaluate.ipynb) | Three-model evaluation + results export | T4 GPU, ~10 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/signal38/signal38.github.io/blob/main/notebooks/03_evaluate.ipynb) |

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

## Team

- **Diya Mirji** — [@dvm14](https://github.com/dvm14)
- **Jonas Neves** — [@jonasneves](https://github.com/jonasneves)
- **Mike Saju** — [@Michaelsaju1](https://github.com/Michaelsaju1)

## License

MIT
