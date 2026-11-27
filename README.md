# Data for LLM: End-to-End Data Preparation Pipeline

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)
[![uv](https://img.shields.io/badge/package%20manager-uv-black)](https://github.com/astral-sh/uv)

A professional, reproducible, and scalable data preparation pipeline for training Large Language Models (LLMs).  
From raw text to an optimized binary dataset with full control over language ratios, data sources, and token budgets.

## Project Goal

To achieve **complete control over training data** for LLMs. The pipeline enables assembling a unified, cleaned dataset from open or local sources, training a custom tokenizer tailored to it, and building the final flat dataset with strict adherence to token quotas.

The entire workflow is reproducible, well-documented, and backed by hash-verified manifests.

> **Philosophy:** At constrained model scales (e.g., 126M–1B parameters), data quality matters far more than sheer quantity. This pipeline follows a "thought → code" principle—built from first principles using industry best practices rather than blind copy-pasting.

---

## Key Features

- **Idempotency** — Every stage verifies hashes of input data and configurations. If nothing has changed, the stage is skipped.
- **Memory Efficiency** — Streaming processing powered by PyArrow and Parquet keeps memory usage strictly bounded by batch size.
- **Precise Ratio Control** — Weighted Round-Robin + ShuffleBuffer ensure the final dataset strictly matches target proportions.
- **Strict Budget Enforcement** — Dataset assembly stops precisely upon reaching `target_tokens`.
- **Detailed Telemetry** — Each stage generates a `manifest.json` containing comprehensive statistics and metadata.
- **Cross-Platform Support** — Reliable UTF-8 handling across operating systems (including Windows).

---

## Pipeline Architecture

The pipeline consists of three independent yet interconnected stages:

```mermaid
graph TD
    subgraph S1["Stage 1: RAW → CLEAN"]
        A1["Sources: JSONL / Parquet"] --> A2("Load with limits & seed")
        A2 --> A3{"Cleaning Profile"}
        A3 -->|"Prose"| A4["Normalization, quality heuristics"]
        A3 -->|"Code"| A5["Code-specific cleaning rules"]
        A4 --> A6["Global hash-based deduplication"]
        A5 --> A6
        A6 --> A7[("CLEAN: Parquet shards")]
        A7 -.-> M1["manifest_clean.json"]
    end

    subgraph S2["Stage 2: CLEAN → TOKENIZER"]
        A7 --> B1["Stratified data slice"]
        B1 --> B2("Train BPE tokenizer")
        B2 --> B3{"Evaluation: round-trip, fertility, compression"}
        B3 --> B4[("Tokenizer: model.json")]
        B4 -.-> M2["manifest_tokenizer.json"]
    end

    subgraph S3["Stage 3: CLEAN + TOKENIZER → FINAL DATASET"]
        A7 --> C1["Streaming document tokenization"]
        B4 --> C1
        C1 --> C2{"Weighted Round-Robin + ShuffleBuffer"}
        C2 --> C3["Build flat binary: data.bin + index.parquet"]
        C3 --> C4["Hash-based train / val split by ID"]
        C4 --> C5[("FINAL: Ready dataset")]
        C5 -.-> M3["manifest_dataset.json"]
    end

    M1 -. "Idempotency: skip on matching hashes" .-> S1
    M2 -. "Idempotency" .-> S2
    M3 -. "Idempotency" .-> S3

    classDef artifact fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef manifest fill:#fff3cd,stroke:#ffc107,stroke-width:2px,stroke-dasharray: 4 2;
    class A7,B4,C5 artifact;
    class M1,M2,M3 manifest;
```

---

# Project Structure
```
data_for_llm/
├── configs/                # YAML configurations (sources, ratios, limits)
├── scripts/                # CLI entry points
│   ├── pipeline.py         # Main orchestrator
│   ├── tokenize_cleaned.py
│   └── build_final_dataset.py
├── src/
│   ├── common/             # Utilities (config, logger, paths)
│   ├── dataset/            # Core data processing
│   │   ├── io/
│   │   ├── cleaning/
│   │   └── sampling/
│   └── tokenizer/          # Tokenizer training and validation
├── tests/
├── .env.example
├── pyproject.toml
└── uv.lock
```

---

# Installation
```
git clone https://github.com/Gigilocha/data_for_llm.git
cd data_for_llm

uv sync
cp .env.example .env
```

### Edit the files in configs/ to match your data sources, proportions, and token budgets.

---

# Usage

### Full Pipeline
```
uv run scripts/pipeline.py --stage all
```

### Individual Stages
```
# Stage 1: Cleaning and deduplication
uv run scripts/pipeline.py --stage cleaned

# Stage 2: Tokenizer training
uv run scripts/pipeline.py --stage tokenizer

# Stage 3: Tokenization and final dataset assembly
uv run scripts/pipeline.py --stage final
```

---

# Outputs
```
output/
├── cleaned/          # Cleaned texts (parquet)
├── tokenizer/        # tokenizer.json + manifest.json
├── tokenized/        # Tokenized texts (parquet)
└── dataset/
    ├── train/
    │   ├── data.bin
    │   └── index.parquet
    └── val/
        └── ...
```

---

# Key Concepts

| Concept | Concept |
| --- | --- |
| Character Budget | During the cleaning stage, text is collected based on character count (target_tokens × share × chars_per_token × overshoot)|
| Token Budget | During the assembly stage, exactly target_tokens × share tokens are collected |
| Weighted Round-Robin | Selects the next document from whichever language/source is furthest behind its target budget |
| ShuffleBuffer | Accumulates documents, shuffles them, and yields batches without splitting individual documents|
| BOS / EOS | Every document is wrapped in <|bos|> and <|eos|> tokens |

---

# Testing
```
# Fast unit tests
pytest -m unit

# All tests
pytest
```

---

# License

MIT

---

# Author

This project is developed as part of an independent deep dive into ML/LLM Engineering. Every component is implemented with a clear understanding of the "why" behind each design choice—without blind copy-pasting.

Telegram: t.me/Gigilocha