# Pyramis-L3D Benchmark

对比 Pyramis-L3D 与传统 MLA (Multi-head Latent Attention, DeepSeek-V2 风格) 的基准。

## 对比项

- **参数量**: 相同 d_model/n_layers/MLP/RoPE, 仅注意力机制不同
- **PPL**: 字符级 IMDB 评测
- **训练吞吐** (tok/s)
- **KV cache**: 有界(L3D) vs 线性(MLA)

## 运行

需要 `uv` 与本地 IMDB 缓存(离线可用):

```bash
cd benchmark
uv sync
uv run python bench_compare.py --nsamples 500 --steps 200 --ctx 256 --batch 16
```

## 结果 (d_model=128, n_layers=2, ctx=256)

| 指标 | Pyramis-L3D | MLA |
|---|---|---|
| 参数量 | 566,722 | 536,448 |
| PPL (eval) | 1.355 | 8.376 |
| KV cache @ctx | 有界(14,336) | 线性(16,384) |

KV cache 节省随 ctx 增长: 256→12.5%, 512→56.2%, 1024→78.1%, 2048→89.1%, 4096→94.5%。

## 文件

- `bench_compare.py`: 对比主脚本
- `modeling_mla.py`: 自包含 MLA baseline (低秩潜空间 KV, 全稠密 attention)
- `pyproject.toml` / `uv.lock`: uv 环境 (CPU torch)
