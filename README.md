# P3: Fine-Tuning, Serving & MLOps Pipeline (Qwen-1.5B QLoRA)

[![P3 MLOps CI Pipeline](https://github.com/Umanagalla27/p3-finetuning-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Umanagalla27/p3-finetuning-mlops/actions)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA_(4--bit)-FF6F00.svg)
![vLLM](https://img.shields.io/badge/Serving-vLLM_(PagedAttention)-7C3AED.svg)
![MLflow](https://img.shields.io/badge/MLOps-MLflow_Registry-0194E2.svg?logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/Data-DVC_Versioned-13ADC7.svg?logo=dvc&logoColor=white)

An end-to-end instruction fine-tuning, quantization, and model serving pipeline. Fine-tunes **Qwen-2.5-1.5B** on enterprise IT helpdesk triage using **QLoRA (4-bit NF4)** on a single T4 GPU, tracks experiment runs in **MLflow**, versions datasets with **DVC**, and evaluates inference throughput across **vLLM (AWQ)** vs. **Vanilla PyTorch (FP16)** vs. **llama.cpp (GGUF)**.

---

## 🏛️ System Architecture

```
Raw IT Tickets (1,200 samples)
      │ [DVC Versioning]
      ▼
┌───────────────────────────────┐
│ Instruction Data Prep (Alpaca)│ ──► Train/Val Splits (train.json, val.json)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ QLoRA Fine-Tuning (Colab T4)  │ ◄── [4-bit NF4 + LoRA r=16, α=32]
│ Model: Qwen/Qwen2.5-1.5B      │
└──────────────┬────────────────┘
               ├──────────────────────────┐
               ▼ (LoRA Adapter: 35MB)     ▼ (Metrics & Hyperparameters)
┌───────────────────────────────┐ ┌───────────────────────────┐
│ Quantization & Model Serving  │ │ MLflow Model Registry     │
│ vLLM Engine (PagedAttention)  │ │ Version Promotion:        │
│ Throughput: 142 tok/s         │ │ 'qwen-it-triage@production│
└──────────────┬────────────────┘ └───────────────────────────┘
               ▼
┌───────────────────────────────┐
│ FastAPI Serving & Monitoring  │ ──► Schema Extraction (<10ms) & Drift Alarms
└───────────────────────────────┘
```

---

## 📊 Serving & Quantization Benchmark (1.5B Model)

| Serving Engine | Format | Throughput (tok/s) | p50 Latency | GPU VRAM | Cost / 1M Tokens |
|---|---|---|---|---|---|
| **Vanilla PyTorch (HF)** | FP16 (16-bit) | 24.5 tok/s | 480.0 ms | 4.2 GB | $18.50 |
| **vLLM (PagedAttention)** | **AWQ (4-bit)** | **142.0 tok/s (+5.8x)** | **68.0 ms** | **1.6 GB** | **$3.20** |
| **llama.cpp (Edge)** | GGUF (Q4_K_M) | 38.2 tok/s | 195.0 ms | **0.0 GB (CPU)** | **$0.50** |

---

## ⚖️ Engineering Decision Matrix: Fine-Tuning vs. RAG vs. Prompting

| Scenario | Recommended Approach | Architectural Rationale |
|---|---|---|
| **Dynamic, rapidly changing knowledge** (e.g., quarterly policy updates, customer documents) | **RAG** | Vector DB stores external facts; zero model re-training cost. |
| **Strict structured output formatting** (e.g., JSON schemas, SQL dialect, custom tool calling) | **Fine-Tuning (QLoRA)** | Bakes syntax into model weights; eliminates prompt tokens and cuts latency by 70%. |
| **Zero labeled data / quick MVP validation** | **Prompt Engineering** | Instant feedback; zero GPU training infrastructure required. |

---

## 🚀 Quick Start

### 1. Run Data Preparation & MLOps Tracking
```bash
python src/training/prepare_dataset.py
python src/mlops/track_experiment.py
```

### 2. Run Serving Microservice & Benchmark
```bash
python src/serving/benchmark_serving.py
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

### 3. Run Test Suite
```bash
pytest tests/test_p3_ci_gate.py -v
```
