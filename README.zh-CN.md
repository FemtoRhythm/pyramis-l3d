# Pyramis-L3D

[**English**](README.md) · **中文文档**

![License](https://img.shields.io/badge/License-Apache%202.0-blue) ![Params](https://img.shields.io/badge/Params-4.47M-orange) ![Attention](https://img.shields.io/badge/Attention-Hierarchical%20Sparse-9cf) ![Framework](https://img.shields.io/badge/Framework-Transformers-yellow)

Pyramis-L3D 是一个**分层 / 稀疏注意力**研究原型。它借鉴 CPU 多级缓存（L1/L2/L3）与 TLB（快表）的思路，把注意力建模为三层金字塔分级存储系统，目标是实现**亚线性 KV-cache 存储**与 **O(k) 查询时稀疏路由**。

> ⚠️ **声明：** 本模型为**研究原型**，非生产可用。目标是探索分层 / 稀疏注意力机制的有效性，鼓励科研人员二次开发。

## 亮点

| 💾 亚线性存储 | ⚡ O(k) 注意力 | 🔺 3级金字塔 | 🔬 开源可复现 |
|---|---|---|---|
| KV-cache 行数恒定为 `M` | 查询时仅激活 `top_k` 命中行 | L1/L2/L3 分层缓存 | 权重 + 代码全部开源 |

## 简介

- 🔴 **痛点：** 传统 Transformer 的 KV Cache 随序列长度**线性增长**，长上下文与高维数据（如 3D 点云）的注意力成本急剧上升。
- 🔵 **方案：** 分层金字塔缓存（L1/L2/L3）+ Latent-TLB 稀疏路由，将 KV 组织为三级缓存，查询时仅对 `top_k` 命中行做注意力，实现 **O(k)** 复杂度。

## 核心机制

| 温区 | 机制 | 说明 |
|---|---|---|
| L1 | 稠密 GQA | 近端 `L1_window` 个 token 全秩注意力 |
| L2 | CNN 池化 KV | 1D-CNN 局部池化，约 `stride` 倍压缩，保留局部句法 |
| L3 | 可寻址字典 (codebook) | 远端 token 入 `M` 行可学习字典，KV 行数恒为 `M`（亚线性） |
| Latent-TLB | 查询时稀疏路由 | 仅对 `top_k` 命中行做 attention |
| distinctness | 近端逐行 latent | 最近 `W_distinct` 个 token 保留逐行 latent，修复长程丢针 |
| CNN 局部门控 | 1D-CNN gate | 注意力后 depthwise conv 门控，增强短距离依赖 |

## 开发期 Tiny 配置

```text
d_model    = 256            n_layers   = 4
n_heads    = 8              head_dim   = 32
vocab_size = 121            intermediate_size = 1024
L1_window  = 128            L2_window  = 512
L2_stride  = 2              L2_conv_kernel    = 5
L3_codebook_size (M) = 1024        L3_latent_dim = 64
W_distinct = 64             top_k      = 16
max_position_embeddings = 1024
参数量     ≈ 4.47M
```

## 快速开始

模型权重托管在 Hugging Face 上；本仓库仅包含代码。

```bash
pip install transformers torch safetensors
```

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "FemtoRhythm/pyramis-l3d", trust_remote_code=True
)
```

从源码本地开发：

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(".")          # 读取 config.json, 走 auto_map
model = AutoModelForCausalLM.from_config(config)
```

## 训练 / 评测

```bash
# 训练 (字符级 tokenizer, 默认读 IMDB 本地缓存 parquet)
python train.py --nsamples 2000 --steps 200 --ctx 256 --batch 16 --lr 1e-3 --out ./checkpoint

# 评测 (PPL / 亚线性行池 scaling / 路由健康)
python eval.py --checkpoint ./checkpoint --ctx 256
```

在 IMDB 字符级数据上短时训练，held-out 集 **PPL = 1.027**。行池随上下文长度保持恒定（亚线性存储）：

| ctx | 可路由行池 | 是否增长 |
|---|---|---|
| 256 | 1344 | — |
| 512 | 1472 | 饱和 |
| 1024 | 1472 | 恒定 |
| 2048 | 1472 | 恒定 |

## Benchmark (vs 传统 MLA)

独立 `uv` 环境，对比 Pyramis-L3D 与 DeepSeek-V2 风格 MLA（仅注意力模块不同）：

```bash
cd benchmark
uv run python bench_compare.py --nsamples 500 --steps 300 --ctx 256 --batch 16
```

| 指标 | Pyramis-L3D | MLA |
|---|---|---|
| 参数量 | 566,722 | 536,448 |
| PPL (eval) | **1.355** | 8.376 |
| KV cache @ctx | 有界 (14,336) | 线性 (16,384) |

## 项目状态

- [x] L1 稠密 GQA
- [x] L2 CNN 池化 KV
- [x] L3 可寻址字典 (codebook)
- [x] Latent-TLB 稀疏路由
- [x] distinctness 保护
- [x] CNN 局部门控
- [x] 训练脚本 + safetensors checkpoint
- [x] 评测 benchmark（vs MLA）
- [ ] 换页机制 (paging)
- [ ] 守护线程 (daemon thread)
- [ ] PD 双路分离

## 局限

- **研究原型**：未做长序列 / 生产级验证；系统级宣称（换页 / 守护线程 / PD 双路分离）均未实现。
- **codebook 利用率低**：短时训练下 VQ codebook 出现坍塌（约 0.2% 利用率），属已知现象，需更长训练或更好初始化缓解。
- **字符级 tokenizer**：词汇极小，不适用于通用文本建模。
- 权重仅在 2000 条样本上短时训练，无泛化保证。

## 引用 BibTeX

```bibtex
@misc{pyramis-l3d,
  author       = {FemtoRhythm},
  title        = {Pyramis-L3D: A Hierarchical Sparse Attention Research Prototype},
  year         = {2026},
  howpublished = {\url{https://github.com/FemtoRhythm/pyramis-l3d}},
  note         = {研究原型, 未正式发表}
}
```

## License

[Apache 2.0](LICENSE)
