# AoF: Boosting MLLM-based Image Quality Assessment via Visual Location Reasoning


[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/PLACEHOLDER/AoF-200K)

This repository provides the project code and data construction pipeline for **AoF**, proposed in the paper **Boosting MLLM-based Image Quality Assessment via Visual Location Reasoning**.

AoF introduces spatially grounded reasoning into multimodal image quality assessment. In addition to global image understanding, the framework builds fine-grained semantic-location supervision and uses structured reasoning over localized regions to improve image quality perception and interpretability.

## Highlights

- A large-scale automated annotation pipeline for constructing `AoF-200K`
- Fine-grained semantic-location reasoning for image quality assessment
- Majority expert voting across multiple MLLMs for reliable pseudo labels
- Reject sampling to retain high-quality `⟨image, mos, thinking, location⟩` tuples
- A modular codebase for dataset construction and model-side integration

## AoF-200K

`AoF-200K` is a large-scale dataset built for spatially grounded image quality assessment. Each sample is organized as a quadruple:

- `image`
- `mos`
- `thinking`
- `location`

The dataset construction process follows four stages:

1. `Data Collection`
2. `Data Annotation`
3. `Majority Expert Voting`
4. `Reject Sampling`

## Data Construction Pipeline

### 1. Data Collection and Coarse Filtering

Raw images are collected from curated sources and filtered before annotation. In the current codebase, this stage is represented by a quality-based pre-screening module that removes samples with extremely low or extremely high quality.

### 2. Data Annotation

For each input image, the pipeline first extracts coarse global semantics. The image is then recursively divided into sub-images, and each region is assessed for:

- local semantics
- local quality description
- pseudo MOS

To reduce hallucinated local concepts, local semantic candidates are constrained by the global semantic context.

### 3. Majority Expert Voting

Multiple expert models independently annotate the same image. Their pseudo scores are merged through a majority voting procedure that searches for the largest consensus subset under a score-distance threshold. The averaged consensus score is used as the final `mos`.

### 4. Reject Sampling

A reasoning model produces multiple candidate reasoning traces for each image. A separate judge model evaluates whether the reasoning is logically consistent with the consensus `mos`. Only accepted samples are kept in the final dataset.

## Repository Overview

```text
aof_data_pipeline_repro/
├── configs/
│   └── default.yaml
├── src/
│   └── aof_pipeline/
│       ├── cli.py
│       ├── pipeline.py
│       ├── providers.py
│       └── types.py
├── pyproject.toml
├── setup.py
└── README.md
```

## Installation

```bash
cd /Users/bytedance/Desktop/code/aof_data_pipeline_repro
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

## Quick Start

### Generate demo data

```bash
aof-pipeline demo-data --output-dir ./data/raw/demo --count 12
```

### Run the data construction pipeline

```bash
aof-pipeline run --config ./configs/default.yaml
```

The pipeline writes two output files:

- `accepted.jsonl`
- `rejected.jsonl`

## Output Format

Each accepted sample is stored as a JSON object similar to the following:

```json
{
  "image_id": "000_street_blur_noise",
  "image_path": ".../000_street_blur_noise.png",
  "mos": 2.731,
  "thinking": "Step 1 ... <loc0>(...)...</loc0> ...",
  "location": {
    "bright": [0, 0, 256, 256],
    "textured": [128, 0, 512, 512]
  },
  "global_semantics": ["bright", "street", "textured"],
  "accepted_experts": ["claude_mock", "gemini_mock", "o1_mock"],
  "debug_votes": {
    "claude_mock": 2.68,
    "gemini_mock": 2.74,
    "o1_mock": 2.77
  }
}
```

## Model Backends

The current implementation includes:

- `mock` providers for local debugging and pipeline validation
- `openai_compatible` providers for integrating external multimodal APIs

To connect a real model backend, update `configs/default.yaml` with your model endpoint, model name, and API key environment variable.

Example:

```yaml
- type: openai_compatible
  name: expert_model
  model: your-vision-model-name
  api_base: https://your-openai-compatible-endpoint/v1
  api_key_env: YOUR_API_KEY_ENV
  temperature: 0.2
  max_tokens: 1200
```

## Dataset Release Plan

The Hugging Face release of **AoF-200K** will be **fully open-sourced after manual cleaning**.

The manual cleaning stage includes sample inspection, annotation quality control, duplicate removal, and additional filtering for abnormal or non-compliant content. After this process is completed, the full AoF-200K dataset and its Hugging Face release will be made publicly available.

## Notes

This repository currently focuses on the AoF data construction pipeline and its engineering workflow. It is designed to make the annotation process modular, extensible, and straightforward to adapt to real model services and real-world data sources.
