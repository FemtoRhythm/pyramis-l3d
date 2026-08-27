# Pyramis-L3D

[**中文文档**](README.zh-CN.md) · **English**

![License](https://img.shields.io/badge/License-Apache%202.0-blue) ![Params](https://img.shields.io/badge/Params-4.47M-orange) ![Attention](https://img.shields.io/badge/Attention-Hierarchical%20Sparse-9cf) ![Framework](https://img.shields.io/badge/Framework-Transformers-yellow)

Pyramis-L3D is a **hierarchical / sparse attention** research prototype. Inspired by CPU multi-level caches (L1/L2/L3) and the TLB (translation lookaside buffer), it models attention as a three-tier pyramidal storage system — targeting **sub-linear KV-cache storage** and **O(k) query-time sparse routing**.

> ⚠️ **Disclaimer:** This is a **research prototype**, not production-ready. It explores hierarchical / sparse attention mechanisms and is intended for further development by researchers.

## Highlights

| 💾 Sub-linear storage | ⚡ O(k) attention | 🔺 3-tier pyramid | 🔬 Open & reproducible |
|---|---|---|---|
| KV-cache rows stay bounded by `M` | Only the `top_k` hit rows are activated | L1/L2/L3 cache hierarchy | Weights + code fully open |

## Introduction

- 🔴 **Pain point:** Conventional Transformers grow the KV cache **linearly** with sequence length, making long contexts and high-dimensional data (e.g. 3D point clouds) increasingly expensive.
- 🔵 **Solution:** A hierarchical pyramid cache (L1/L2/L3) plus Latent-TLB sparse routing organizes KV into three tiers, attending only to the `top_k` hit rows per query for **O(k)** complexity.

## Core Mechanism

| Tier | Mechanism | Description |
|---|---|---|
| L1 | Dense GQA | Full-rank attention over the nearest `L1_window` tokens |
| L2 | CNN-pooled KV | 1D-CNN local pooling, ~`stride`× compression, preserving local syntax |
| L3 | Addressable dictionary (codebook) | Distant tokens enter a learnable `M`-row dictionary; KV rows stay constant at `M` (sub-linear) |
| Latent-TLB | Query-time sparse routing | Attention only over the `top_k` hit rows |
| distinctness | Recent per-row latent | The last `W_distinct` tokens keep per-row latent, fixing long-range needle loss |
| CNN local gate | 1D-CNN gate | Depthwise conv gating after attention, strengthening short-range dependencies |

## Tiny Dev Config

```text
d_model    = 256            n_layers   = 4
n_heads    = 8              head_dim   = 32
vocab_size = 121            intermediate_size = 1024
L1_window  = 128            L2_window  = 512
L2_stride  = 2              L2_conv_kernel    = 5
L3_codebook_size (M) = 1024        L3_latent_dim = 64
W_distinct = 64             top_k      = 16
max_position_embeddings = 1024
Parameters ≈ 4.47M
```

## Quick Start

Model weights are hosted on Hugging Face; this repository contains only the code.

```bash
pip install transformers torch safetensors
```

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "FemtoRhythm/pyramis-l3d", trust_remote_code=True
)
```

Local development from source:

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(".")          # reads config.json via auto_map
model = AutoModelForCausalLM.from_config(config)
```

## Training & Evaluation

```bash
# Train (character-level tokenizer, reads cached IMDB parquet)
python train.py --nsamples 2000 --steps 200 --ctx 256 --batch 16 --lr 1e-3 --out ./checkpoint

# Evaluate (PPL / sub-linear row-pool scaling / routing health)
python eval.py --checkpoint ./checkpoint --ctx 256
```

Briefly trained on character-level IMDB, held-out **PPL = 1.027**. The row pool stays constant across context length (sub-linear storage):

| ctx | Routable row pool | Growth |
|---|---|---|
| 256 | 1344 | — |
| 512 | 1472 | saturated |
| 1024 | 1472 | constant |
| 2048 | 1472 | constant |

## Benchmark (vs MLA)

Standalone `uv` environment comparing Pyramis-L3D against DeepSeek-V2-style MLA (only the attention module differs):

```bash
cd benchmark
uv run python bench_compare.py --nsamples 500 --steps 300 --ctx 256 --batch 16
```

| Metric | Pyramis-L3D | MLA |
|---|---|---|
| Parameters | 566,722 | 536,448 |
| PPL (eval) | **1.355** | 8.376 |
| KV cache @ctx | Bounded (14,336) | Linear (16,384) |

## Project Status

- [x] L1 dense GQA
- [x] L2 CNN-pooled KV
- [x] L3 addressable dictionary (codebook)
- [x] Latent-TLB sparse routing
- [x] distinctness protection
- [x] CNN local gating
- [x] Training script + safetensors checkpoint
- [x] Benchmark evaluation (vs MLA)
- [ ] Paging mechanism
- [ ] Daemon thread
- [ ] PD dual-path separation

## Limitations

- **Research prototype**: no long-sequence / production validation; system-level claims (paging, daemon thread, PD dual-path) are not implemented.
- **Low codebook utilization**: the VQ codebook collapses under short training (~0.2% utilization), a known phenomenon mitigated by longer training or better initialization.
- **Character-level tokenizer**: tiny vocabulary, unsuitable for general text modeling.
- Weights are briefly trained on 2000 samples, with no generalization guarantee.

## Citation

```bibtex
@misc{pyramis-l3d,
  author       = {FemtoRhythm},
  title        = {Pyramis-L3D: A Hierarchical Sparse Attention Research Prototype},
  year         = {2026},
  howpublished = {\url{https://github.com/FemtoRhythm/pyramis-l3d}},
  note         = {Research prototype, not formally published}
}
```

## License

[Apache 2.0](LICENSE)
