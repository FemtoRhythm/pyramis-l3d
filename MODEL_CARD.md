---
language:
  - zh
  - en
license: apache-2.0
tags:
  - hierarchical-attention
  - sparse-attention
  - research
library_name: transformers
---

![pyramis-l3d](./logo-light.png)

<div style="display:flex; gap:8px; margin:12px 0 24px 0;">
  <a href="#zh-full" style="padding:6px 16px; background-color:rgba(74,92,201,0.08); color:#4c64eeff; border-radius:16px; text-decoration:none; font-weight:500; font-size:14px;">🇨🇳 中文</a>
  <a href="#en-full" style="padding:6px 16px; background-color:rgba(74,92,201,0.08); color:#4c64eeff; border-radius:16px; text-decoration:none; font-weight:500; font-size:14px;">🌐 English</a>
</div>

<a id="zh-full"></a>

# Pyramis-L3D

![license](https://img.shields.io/badge/License-Apache%202.0-blue) ![params](https://img.shields.io/badge/Params-4.47M-orange) ![attention](https://img.shields.io/badge/Attention-Hierarchical%20Sparse-9cf) ![framework](https://img.shields.io/badge/Framework-Transformers-yellow)

<div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 24px 0;">
  <div style="flex:1; min-width:140px; padding:12px; background-color:rgba(74,92,201,0.06); border-radius:8px; border:1px solid rgba(74,92,201,0.15);">
    <div style="font-size:18px; margin-bottom:4px;">💾</div>
    <div style="font-size:13px; font-weight:600; color:#4c64eeff;">亚线性存储</div>
    <div style="font-size:12px; color:#6B7280; margin-top:2px;">KV Cache 行数恒定为 M</div>
  </div>
  <div style="flex:1; min-width:140px; padding:12px; background-color:rgba(74,92,201,0.06); border-radius:8px; border:1px solid rgba(74,92,201,0.15);">
    <div style="font-size:18px; margin-bottom:4px;">⚡</div>
    <div style="font-size:13px; font-weight:600; color:#4c64eeff;">O(k) 注意力</div>
    <div style="font-size:12px; color:#6B7280; margin-top:2px;">查询时仅激活 Top-K 行</div>
  </div>
  <div style="flex:1; min-width:140px; padding:12px; background-color:rgba(74,92,201,0.06); border-radius:8px; border:1px solid rgba(74,92,201,0.15);">
    <div style="font-size:18px; margin-bottom:4px;">🔺</div>
    <div style="font-size:13px; font-weight:600; color:#4c64eeff;">3级金字塔</div>
    <div style="font-size:12px; color:#6B7280; margin-top:2px;">L1/L2/L3 分层缓存</div>
  </div>
  <div style="flex:1; min-width:140px; padding:12px; background-color:rgba(74,92,201,0.06); border-radius:8px; border:1px solid rgba(74,92,201,0.15);">
    <div style="font-size:18px; margin-bottom:4px;">🔬</div>
    <div style="font-size:13px; font-weight:600; color:#4c64eeff;">开源可复现</div>
    <div style="font-size:12px; color:#6B7280; margin-top:2px;">权重+代码全部开源</div>
  </div>
</div>

<div style="margin:16px 0 24px 0; padding:12px 16px; background-color:#F3F4F6; border-radius:8px; border-left:4px solid #9CA3AF;">
  <p style="margin:0; font-size:13px; color:#4B5563; line-height:1.6;">
    ⚠️ <strong>Disclaimer:</strong> 本模型为<strong>研究原型</strong>，非生产可用。目标是探索分层/稀疏注意力机制的有效性，鼓励科研人员二次开发。
  </p>
</div>

## 模型简介

<div style="margin:12px 0; padding:12px 16px; background-color:#FEF2F2; border-radius:8px; border-left:4px solid #DC2626;">
  <p style="margin:0; font-size:13px; color:#7F1D1D; line-height:1.6;">
    🔴 <strong>痛点：</strong>传统 Transformer 的 KV Cache 随序列长度<strong>线性增长</strong>，长上下文与高维数据（如 3D 点云）的注意力成本急剧上升。
  </p>
</div>

<div style="margin:12px 0; padding:12px 16px; background-color:#EEF2FF; border-radius:8px; border-left:4px solid #4c64eeff;">
  <p style="margin:0; font-size:13px; color:#1E3A8A; line-height:1.6;">
    🔵 <strong>方案：</strong>分层金字塔缓存（L1/L2/L3）+ Latent-TLB 稀疏路由，将 KV 组织为三级缓存，查询时仅对 <strong>top_k</strong> 命中行做注意力，实现 <strong>O(k)</strong> 复杂度。
  </p>
</div>


Pyramis-L3D 是一个**分层 / 稀疏注意力**的研究原型。它借鉴 CPU 多级缓存（L1/L2/L3）与 TLB（快表）的思路，把注意力建模为三层金字塔分级存储系统，目标是让 **KV cache 不再随序列长度线性增长**，收敛到**有界行池**，并通过查询时稀疏路由实现 **O(k)** 的注意力计算。

## 核心机制详解

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup>
<col style="width: 18%;">
<col style="width: 30%;">
<col style="width: 52%;">
</colgroup>
<tr>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">温区</th>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">机制</th>
<th style="padding: 8px 12px; text-align: left; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">说明</th>
</tr>
<tr style="background-color: #D6DAFC;">
<td colspan="3" style="padding: 8px 12px; font-weight: 600;">📦 分层存储</td>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center;">L1</td>
<td style="padding: 8px 12px; text-align: center;">稠密 GQA</td>
<td style="padding: 8px 12px; text-align: left;">近端 <code>L1_window</code> 个 token 全秩注意力</td>
</tr>
<tr style="background-color: rgba(74, 92, 201, 0.06);">
<td style="padding: 8px 12px; text-align: center;">L2</td>
<td style="padding: 8px 12px; text-align: center;">CNN 池化 KV</td>
<td style="padding: 8px 12px; text-align: left;">1D-CNN 局部池化，约 <code>stride</code> 倍压缩，保留局部句法</td>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center; font-weight: 600; color: #4c64eeff;">L3</td>
<td style="padding: 8px 12px; text-align: center; font-weight: 600; color: #4c64eeff;">可寻址字典 (codebook)</td>
<td style="padding: 8px 12px; text-align: left; font-weight: 600; color: #4c64eeff;">远端 token 入 <code>M</code> 行可学习字典，KV 行数恒为 <code>M</code>（亚线性）</td>
</tr>
<tr style="background-color: #D6DAFC;">
<td colspan="3" style="padding: 8px 12px; font-weight: 600;">🎯 稀疏路由</td>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center; font-weight: 600; color: #4c64eeff;">Latent-TLB</td>
<td style="padding: 8px 12px; text-align: center; font-weight: 600; color: #4c64eeff;">查询时稀疏路由</td>
<td style="padding: 8px 12px; text-align: left; font-weight: 600; color: #4c64eeff;">仅对 <code>top_k</code> 命中行做 attention</td>
</tr>
<tr style="background-color: rgba(74, 92, 201, 0.06);">
<td style="padding: 8px 12px; text-align: center;">distinctness</td>
<td style="padding: 8px 12px; text-align: center;">近端逐行 latent</td>
<td style="padding: 8px 12px; text-align: left;">最近 <code>W_distinct</code> 个 token 保留逐行 latent，修复长程丢针</td>
</tr>
<tr style="background-color: #D6DAFC;">
<td colspan="3" style="padding: 8px 12px; font-weight: 600;">🧠 局部门控</td>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center;">CNN 局部门控</td>
<td style="padding: 8px 12px; text-align: center;">1D-CNN gate</td>
<td style="padding: 8px 12px; text-align: left;">注意力后 depthwise conv 门控，增强短距离依赖</td>
</tr>
</table>

## 开发期 Tiny 配置

<pre style="background-color:#F9FAFB; padding:12px 16px; border-radius:8px; border:1px solid #E5E7EB; overflow-x:auto; font-size:13px; line-height:1.7;">
d_model    = 256          n_layers   = 4
n_heads    = 8            head_dim   = 32
vocab_size = 121          intermediate_size = 1024
L1_window  = 128          L2_window  = 512
L2_stride  = 2            L2_conv_kernel    = 5
L3_codebook_size (M) = 1024      W_distinct = 64
top_k      = 16           max_position_embeddings = 1024
参数量     ≈ 4.47M
</pre>

## 快速开始

```bash
pip install transformers torch safetensors
```

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "FemtoRhythm/pyramis-l3d", trust_remote_code=True
)

from tokenizer import CharTokenizer

tok = CharTokenizer.load("vocab.json")
ids = tok.encode("this movie is")
```

## 训练与评测

在 IMDB 字符级数据上短时训练，held-out 集 **PPL = 1.027**。行池随上下文长度保持恒定（亚线性存储）：

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup>
<col style="width: 5%;">
<col style="width: 10%;">
<col style="width: 10%;">
</colgroup>
<tr>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">ctx</th>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">可路由行池</th>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">是否增长</th>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center;">256</td>
<td style="padding: 8px 12px; text-align: center;">1344</td>
<td style="padding: 8px 12px; text-align: center;">—</td>
</tr>
<tr style="background-color: rgba(74, 92, 201, 0.06);">
<td style="padding: 8px 12px; text-align: center;">512</td>
<td style="padding: 8px 12px; text-align: center;">1472</td>
<td style="padding: 8px 12px; text-align: center;">饱和</td>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center;">1024</td>
<td style="padding: 8px 12px; text-align: center;">1472</td>
<td style="padding: 8px 12px; text-align: center;">恒定</td>
</tr>
<tr style="background-color: rgba(74, 92, 201, 0.06);">
<td style="padding: 8px 12px; text-align: center;">2048</td>
<td style="padding: 8px 12px; text-align: center;">1472</td>
<td style="padding: 8px 12px; text-align: center;">恒定</td>
</tr>
</table>

与 DeepSeek-V2 风格 MLA 的对比（仅注意力模块不同，其余结构一致）：

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup>
<col style="width: 5%;">
<col style="width: 10%;">
<col style="width: 10%;">
</colgroup>
<tr>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">指标</th>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">Pyramis-L3D</th>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">MLA</th>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center;">参数量</td>
<td style="padding: 8px 12px; text-align: center;">566,722</td>
<td style="padding: 8px 12px; text-align: center;">536,448</td>
</tr>
<tr style="background-color: rgba(74, 92, 201, 0.06);">
<td style="padding: 8px 12px; text-align: center;">PPL (eval)</td>
<td style="padding: 8px 12px; text-align: center; background-color: rgba(74, 92, 201, 0.08);"><strong>1.355</strong></td>
<td style="padding: 8px 12px; text-align: center;">8.376</td>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center;">KV cache @ctx</td>
<td style="padding: 8px 12px; text-align: center;">有界 (14,336)</td>
<td style="padding: 8px 12px; text-align: center;">线性 (16,384)</td>
</tr>
</table>

## 训练状态

<div style="margin:12px 0; font-size:14px; line-height:1.9;">
  <div><span style="color:#16A34A;">✔</span> L1 稠密 GQA</div>
  <div><span style="color:#16A34A;">✔</span> L2 CNN 池化 KV</div>
  <div><span style="color:#16A34A;">✔</span> L3 可寻址字典 (codebook)</div>
  <div><span style="color:#16A34A;">✔</span> Latent-TLB 稀疏路由</div>
  <div><span style="color:#16A34A;">✔</span> distinctness 保护</div>
  <div><span style="color:#16A34A;">✔</span> CNN 局部门控</div>
  <div><span style="color:#16A34A;">✔</span> 训练 + safetensors checkpoint</div>
  <div><span style="color:#16A34A;">✔</span> 评测 benchmark（vs MLA）</div>
  <div><span style="color:#9CA3AF;">○</span> 换页机制 (paging)</div>
  <div><span style="color:#9CA3AF;">○</span> 守护线程 (daemon thread)</div>
  <div><span style="color:#9CA3AF;">○</span> PD 双路分离</div>
</div>

## 局限性

- **研究原型**：未做长序列/生产级验证；系统级宣称（换页/守护线程/PD 双路分离）均未实现。
- **codebook 利用率低**：短时训练下 VQ codebook 出现坍塌（约 0.2% 利用率），属已知现象，需更长训练或更好初始化缓解。
- **字符级 tokenizer**：词汇极小，不适用于通用文本建模。
- 权重仅在 2000 条样本上短时训练，无泛化保证。

## 引用 BibTeX

```bibtex
@misc{pyramis-l3d,
  author       = {FemtoRhythm},
  title        = {Pyramis-L3D: A Hierarchical Sparse Attention Research Prototype},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/FemtoRhythm/pyramis-l3d}},
  note         = {研究原型, 未正式发表}
}
```

***

<a id="en-full"></a>

# Pyramis-L3D

![license](https://img.shields.io/badge/License-Apache%202.0-blue) ![params](https://img.shields.io/badge/Params-4.47M-orange) ![attention](https://img.shields.io/badge/Attention-Hierarchical%20Sparse-9cf) ![framework](https://img.shields.io/badge/Framework-Transformers-yellow)

<div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 24px 0;">
  <div style="flex:1; min-width:140px; padding:12px; background-color:rgba(74,92,201,0.06); border-radius:8px; border:1px solid rgba(74,92,201,0.15);">
    <div style="font-size:18px; margin-bottom:4px;">💾</div>
    <div style="font-size:13px; font-weight:600; color:#4c64eeff;">Sub-linear storage</div>
    <div style="font-size:12px; color:#6B7280; margin-top:2px;">KV cache rows bounded by M</div>
  </div>
  <div style="flex:1; min-width:140px; padding:12px; background-color:rgba(74,92,201,0.06); border-radius:8px; border:1px solid rgba(74,92,201,0.15);">
    <div style="font-size:18px; margin-bottom:4px;">⚡</div>
    <div style="font-size:13px; font-weight:600; color:#4c64eeff;">O(k) attention</div>
    <div style="font-size:12px; color:#6B7280; margin-top:2px;">Only Top-K rows activated</div>
  </div>
  <div style="flex:1; min-width:140px; padding:12px; background-color:rgba(74,92,201,0.06); border-radius:8px; border:1px solid rgba(74,92,201,0.15);">
    <div style="font-size:18px; margin-bottom:4px;">🔺</div>
    <div style="font-size:13px; font-weight:600; color:#4c64eeff;">3-tier pyramid</div>
    <div style="font-size:12px; color:#6B7280; margin-top:2px;">L1/L2/L3 cache hierarchy</div>
  </div>
  <div style="flex:1; min-width:140px; padding:12px; background-color:rgba(74,92,201,0.06); border-radius:8px; border:1px solid rgba(74,92,201,0.15);">
    <div style="font-size:18px; margin-bottom:4px;">🔬</div>
    <div style="font-size:13px; font-weight:600; color:#4c64eeff;">Open & reproducible</div>
    <div style="font-size:12px; color:#6B7280; margin-top:2px;">Weights + code fully open</div>
  </div>
</div>

<div style="margin:16px 0 24px 0; padding:12px 16px; background-color:#F3F4F6; border-radius:8px; border-left:4px solid #9CA3AF;">
  <p style="margin:0; font-size:13px; color:#4B5563; line-height:1.6;">
    ⚠️ <strong>Disclaimer:</strong> This model is a <strong>research prototype</strong>, not production-ready. It explores the effectiveness of hierarchical / sparse attention mechanisms and encourages further development by researchers.
  </p>
</div>

## Introduction

<div style="margin:12px 0; padding:12px 16px; background-color:#FEF2F2; border-radius:8px; border-left:4px solid #DC2626;">
  <p style="margin:0; font-size:13px; color:#7F1D1D; line-height:1.6;">
    🔴 <strong>Pain point:</strong> Conventional Transformers grow KV cache <strong>linearly</strong> with sequence length, making long contexts and high-dimensional data (e.g. 3D point clouds) increasingly expensive.
  </p>
</div>

<div style="margin:12px 0; padding:12px 16px; background-color:#EEF2FF; border-radius:8px; border-left:4px solid #4c64eeff;">
  <p style="margin:0; font-size:13px; color:#1E3A8A; line-height:1.6;">
    🔵 <strong>Solution:</strong> A hierarchical pyramid cache (L1/L2/L3) plus Latent-TLB sparse routing organizes KV into three cache tiers, attending only to the <strong>top_k</strong> hit rows per query for <strong>O(k)</strong> complexity.
  </p>
</div>


Pyramis-L3D is a **hierarchical / sparse attention** research prototype. Inspired by CPU multi-level caches (L1/L2/L3) and the TLB (translation lookaside buffer), it models attention as a three-tier pyramidal storage system, aiming to make the **KV cache stop growing linearly with sequence length**, converging to a **bounded row pool** with **O(k)** attention via query-time sparse routing.

## Core Mechanism

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup>
<col style="width: 18%;">
<col style="width: 30%;">
<col style="width: 52%;">
</colgroup>
<tr>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">Tier</th>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">Mechanism</th>
<th style="padding: 8px 12px; text-align: left; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">Description</th>
</tr>
<tr style="background-color: #D6DAFC;">
<td colspan="3" style="padding: 8px 12px; font-weight: 600;">📦 Hierarchical storage</td>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center;">L1</td>
<td style="padding: 8px 12px; text-align: center;">Dense GQA</td>
<td style="padding: 8px 12px; text-align: left;">Full-rank attention over the nearest <code>L1_window</code> tokens</td>
</tr>
<tr style="background-color: rgba(74, 92, 201, 0.06);">
<td style="padding: 8px 12px; text-align: center;">L2</td>
<td style="padding: 8px 12px; text-align: center;">CNN-pooled KV</td>
<td style="padding: 8px 12px; text-align: left;">1D-CNN local pooling, ~<code>stride</code>× compression, preserving local syntax</td>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center; font-weight: 600; color: #4c64eeff;">L3</td>
<td style="padding: 8px 12px; text-align: center; font-weight: 600; color: #4c64eeff;">Addressable dictionary (codebook)</td>
<td style="padding: 8px 12px; text-align: left; font-weight: 600; color: #4c64eeff;">Distant tokens enter a learnable <code>M</code>-row dictionary; KV rows stay constant at <code>M</code> (sub-linear)</td>
</tr>
<tr style="background-color: #D6DAFC;">
<td colspan="3" style="padding: 8px 12px; font-weight: 600;">🎯 Sparse routing</td>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center; font-weight: 600; color: #4c64eeff;">Latent-TLB</td>
<td style="padding: 8px 12px; text-align: center; font-weight: 600; color: #4c64eeff;">Query-time sparse routing</td>
<td style="padding: 8px 12px; text-align: left; font-weight: 600; color: #4c64eeff;">Attention only over the <code>top_k</code> hit rows</td>
</tr>
<tr style="background-color: rgba(74, 92, 201, 0.06);">
<td style="padding: 8px 12px; text-align: center;">distinctness</td>
<td style="padding: 8px 12px; text-align: center;">Recent per-row latent</td>
<td style="padding: 8px 12px; text-align: left;">The last <code>W_distinct</code> tokens keep per-row latent, fixing long-range needle loss</td>
</tr>
<tr style="background-color: #D6DAFC;">
<td colspan="3" style="padding: 8px 12px; font-weight: 600;">🧠 Local gating</td>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center;">CNN local gate</td>
<td style="padding: 8px 12px; text-align: center;">1D-CNN gate</td>
<td style="padding: 8px 12px; text-align: left;">Depthwise conv gating after attention, strengthening short-range dependencies</td>
</tr>
</table>

## Tiny Dev Config

<pre style="background-color:#F9FAFB; padding:12px 16px; border-radius:8px; border:1px solid #E5E7EB; overflow-x:auto; font-size:13px; line-height:1.7;">
d_model    = 256          n_layers   = 4
n_heads    = 8            head_dim   = 32
vocab_size = 121          intermediate_size = 1024
L1_window  = 128          L2_window  = 512
L2_stride  = 2            L2_conv_kernel    = 5
L3_codebook_size (M) = 1024      W_distinct = 64
top_k      = 16           max_position_embeddings = 1024
Parameters ≈ 4.47M
</pre>

## Quick Start

```bash
pip install transformers torch safetensors
```

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "FemtoRhythm/pyramis-l3d", trust_remote_code=True
)

from tokenizer import CharTokenizer

tok = CharTokenizer.load("vocab.json")
ids = tok.encode("this movie is")
```

## Training & Evaluation

Briefly trained on character-level IMDB data, held-out **PPL = 1.027**. The row pool stays constant across context length (sub-linear storage):

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup>
<col style="width: 5%;">
<col style="width: 10%;">
<col style="width: 10%;">
</colgroup>
<tr>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">ctx</th>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">Routable row pool</th>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">Growth</th>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center;">256</td>
<td style="padding: 8px 12px; text-align: center;">1344</td>
<td style="padding: 8px 12px; text-align: center;">—</td>
</tr>
<tr style="background-color: rgba(74, 92, 201, 0.06);">
<td style="padding: 8px 12px; text-align: center;">512</td>
<td style="padding: 8px 12px; text-align: center;">1472</td>
<td style="padding: 8px 12px; text-align: center;">saturated</td>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center;">1024</td>
<td style="padding: 8px 12px; text-align: center;">1472</td>
<td style="padding: 8px 12px; text-align: center;">constant</td>
</tr>
<tr style="background-color: rgba(74, 92, 201, 0.06);">
<td style="padding: 8px 12px; text-align: center;">2048</td>
<td style="padding: 8px 12px; text-align: center;">1472</td>
<td style="padding: 8px 12px; text-align: center;">constant</td>
</tr>
</table>

Comparison against DeepSeek-V2-style MLA (only the attention module differs, all else identical):

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup>
<col style="width: 5%;">
<col style="width: 10%;">
<col style="width: 10%;">
</colgroup>
<tr>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">Metric</th>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">Pyramis-L3D</th>
<th style="padding: 8px 12px; text-align: center; background-color: #4c64eeff; color: #FFFFFF; border-bottom: 2px solid #4c64eeff;">MLA</th>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center;">Parameters</td>
<td style="padding: 8px 12px; text-align: center;">566,722</td>
<td style="padding: 8px 12px; text-align: center;">536,448</td>
</tr>
<tr style="background-color: rgba(74, 92, 201, 0.06);">
<td style="padding: 8px 12px; text-align: center;">PPL (eval)</td>
<td style="padding: 8px 12px; text-align: center; background-color: rgba(74, 92, 201, 0.08);"><strong>1.355</strong></td>
<td style="padding: 8px 12px; text-align: center;">8.376</td>
</tr>
<tr>
<td style="padding: 8px 12px; text-align: center;">KV cache @ctx</td>
<td style="padding: 8px 12px; text-align: center;">Bounded (14,336)</td>
<td style="padding: 8px 12px; text-align: center;">Linear (16,384)</td>
</tr>
</table>

## Training Status

<div style="margin:12px 0; font-size:14px; line-height:1.9;">
  <div><span style="color:#16A34A;">✔</span> L1 dense GQA</div>
  <div><span style="color:#16A34A;">✔</span> L2 CNN-pooled KV</div>
  <div><span style="color:#16A34A;">✔</span> L3 addressable dictionary (codebook)</div>
  <div><span style="color:#16A34A;">✔</span> Latent-TLB sparse routing</div>
  <div><span style="color:#16A34A;">✔</span> distinctness protection</div>
  <div><span style="color:#16A34A;">✔</span> CNN local gating</div>
  <div><span style="color:#16A34A;">✔</span> Training + safetensors checkpoint</div>
  <div><span style="color:#16A34A;">✔</span> Benchmark evaluation (vs MLA)</div>
  <div><span style="color:#9CA3AF;">○</span> Paging mechanism</div>
  <div><span style="color:#9CA3AF;">○</span> Daemon thread</div>
  <div><span style="color:#9CA3AF;">○</span> PD dual-path separation</div>
</div>

## Limitations

- **Research prototype**: no long-sequence / production validation; system-level claims (paging, daemon thread, PD dual-path) are not implemented.
- **Low codebook utilization**: VQ codebook collapses under short training (~0.2% utilization), a known phenomenon, mitigated by longer training or better initialization.
- **Character-level tokenizer**: tiny vocabulary, unsuitable for general text modeling.
- Weights are briefly trained on 2000 samples, with no generalization guarantee.

## BibTeX Citation

```bibtex
@misc{pyramis-l3d,
  author       = {FemtoRhythm},
  title        = {Pyramis-L3D: A Hierarchical Sparse Attention Research Prototype},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/FemtoRhythm/pyramis-l3d}},
  note         = {Research prototype, not formally published}
}
```
