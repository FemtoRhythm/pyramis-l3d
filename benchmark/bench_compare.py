"""Pyramis-L3D vs 传统 MLA 对比基准。

- 开源数据集: IMDB 电影评论 (Apache-2.0, 从本地 HF 缓存读取, 无需联网)
- 字符级 tokenizer (极小 vocab, 满足"少参数")
- 相同 d_model/n_layers/MLP/RoPE, 只对比注意力机制
- 指标: 参数量 / PPL / 训练吞吐 / KV cache 内存

用法:
    uv run python bench_compare.py [--nsamples 500] [--steps 300] [--ctx 256] [--batch 16]
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

# 引入父目录中的 Pyramis-L3D 模型
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration_pyramis_l3d import PyramisL3DConfig  # noqa: E402
from modeling_pyramis_l3d import PyramisL3DForCausalLM  # noqa: E402
from modeling_mla import MLAModel  # noqa: E402


def load_texts(n_samples):
    # 从本地 HF 缓存读取 IMDB 训练集 (Apache-2.0 开源数据集), 全程离线
    import pyarrow.parquet as pq
    path = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "datasets--imdb", "snapshots",
        "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
        "plain_text", "train-00000-of-00001.parquet",
    )
    table = pq.read_table(path)
    texts = [str(t) for t in table.column("text").to_pylist() if t]
    return texts[:n_samples]


def build_tokenizer(texts):
    chars = sorted(set("".join(texts)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return stoi, itos


def encode(texts, stoi):
    flat = []
    for t in texts:
        flat.extend(stoi[c] for c in t)
    return torch.tensor(flat, dtype=torch.long)


def get_batch(tokens, batch, ctx, device):
    ix = torch.randint(0, len(tokens) - ctx - 1, (batch,))
    x = torch.stack([tokens[i:i + ctx] for i in ix])
    y = torch.stack([tokens[i + 1:i + ctx + 1] for i in ix])
    return x.to(device), y.to(device)


def logits_of(model, x, is_l3d):
    if is_l3d:
        return model(x).logits
    return model(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(model, is_l3d, train_tokens, eval_tokens, device, args):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    ce_history = []
    t0 = time.time()
    for step in range(args.steps):
        x, y = get_batch(train_tokens, args.batch, args.ctx, device)
        opt.zero_grad()
        if is_l3d:
            # L3D 用自带 loss (CE + balance + sparsity + commitment), 即其训练目标
            out = model(x, labels=y)
            loss = out.loss
            logits = out.logits
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        pure_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        loss.backward()
        opt.step()
        ce_history.append(pure_ce)
        if (step + 1) % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(f"  step {step + 1:4d}/{args.steps}  train_CE={pure_ce:.4f}  "
                  f"ppl={math.exp(pure_ce):.2f}")

    elapsed = time.time() - t0
    tokens_per_sec = args.steps * args.batch * args.ctx / elapsed

    # ---- 评估 PPL (纯 CE) ----
    model.eval()
    total_ce = 0.0
    total_n = 0
    with torch.no_grad():
        for _ in range(args.eval_iters):
            x, y = get_batch(eval_tokens, args.batch, args.ctx, device)
            logits = logits_of(model, x, is_l3d)
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_ce += ce.item() * y.numel()
            total_n += y.numel()
    ppl = math.exp(total_ce / total_n)

    return {"ppl": ppl, "tokens_per_sec": tokens_per_sec, "elapsed": elapsed,
            "final_ce": ce_history[-1], "first_ce": ce_history[0]}


def kv_cache_floats(args, n_kv_heads, head_dim):
    """返回 ctx 长度下的 KV cache 元素数 (float32)。"""
    mla = args.ctx * args.latent_dim  # 每 token 只存 latent
    l2_rows = args.L2_window // args.L2_stride
    l3d = (args.L1_window + l2_rows + args.L3_codebook_size) * n_kv_heads * head_dim * 2
    return {"mla": mla, "l3d": l3d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=500)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 共享结构超参 (少参数)
    d_model = 128
    n_layers = 2
    n_heads = 4
    head_dim = 32          # d_model = n_heads * head_dim
    intermediate_size = 512
    args.L1_window = 32
    args.L2_window = 128
    args.L2_stride = 2
    args.L3_codebook_size = 128
    args.latent_dim = 64  # MLA latent 压缩维度

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    print(f"== 加载 IMDB 训练集 (train[:{args.nsamples}], 本地缓存) ==")
    texts = load_texts(args.nsamples)
    stoi, itos = build_tokenizer(texts)
    vocab_size = len(stoi)
    print(f"  vocab_size={vocab_size}, 总字符~{sum(len(t) for t in texts)}")

    # train / eval 切分 (后 10% 作为 held-out)
    split = int(len(texts) * 0.9)
    train_tokens = encode(texts[:split], stoi)
    eval_tokens = encode(texts[split:], stoi)
    print(f"  train tokens={len(train_tokens)}, eval tokens={len(eval_tokens)}")

    print("\n== 构建模型 ==")
    l3d_cfg = PyramisL3DConfig(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, head_dim=head_dim, intermediate_size=intermediate_size,
        L1_window=args.L1_window, L2_window=args.L2_window,
        L2_conv_kernel=5, L2_stride=args.L2_stride,
        L3_codebook_size=args.L3_codebook_size, L3_latent_dim=32,
        W_distinct=16, top_k=8, tlb_mode="ste",
        max_position_embeddings=args.ctx,
    )
    l3d = PyramisL3DForCausalLM(l3d_cfg).to(device)
    mla = MLAModel(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, head_dim=head_dim, latent_dim=args.latent_dim,
        intermediate_size=intermediate_size, max_position_embeddings=args.ctx,
    ).to(device)

    n_kv_heads = n_heads // 4  # Pyramis-L3D 硬编码 GQA 4:1

    print(f"  Pyramis-L3D params: {count_params(l3d):,}")
    print(f"  MLA          params: {count_params(mla):,}")

    print("\n== 训练 Pyramis-L3D ==")
    r_l3d = train_and_eval(l3d, True, train_tokens, eval_tokens, device, args)

    print("\n== 训练 MLA ==")
    r_mla = train_and_eval(mla, False, train_tokens, eval_tokens, device, args)

    kv = kv_cache_floats(args, n_kv_heads, head_dim)

    print("\n================ 对比结果 ================")
    print(f"{'指标':<24}{'Pyramis-L3D':>16}{'MLA':>16}")
    print("-" * 56)
    print(f"{'参数量':<24}{count_params(l3d):>16,}{count_params(mla):>16,}")
    print(f"{'PPL (eval)':<24}{r_l3d['ppl']:>16.3f}{r_mla['ppl']:>16.3f}")
    print(f"{'训练吞吐 (tok/s)':<24}{r_l3d['tokens_per_sec']:>16.1f}{r_mla['tokens_per_sec']:>16.1f}")
    print(f"{'训练耗时 (s)':<24}{r_l3d['elapsed']:>16.1f}{r_mla['elapsed']:>16.1f}")
    print(f"{'KV cache @ctx (float32)':<24}{kv['l3d']:>16,}{kv['mla']:>16,}")
    print(f"{'KV cache 增长方式':<24}{'有界(常数)':>16}{'线性(ctx)':>16}")
    print("-" * 56)
    print(f"\n说明: L3D 的 KV cache 由 L1({args.L1_window})+L2({args.L2_window//args.L2_stride})+"
          f"L3({args.L3_codebook_size}) 组成, 不随 ctx 增长; MLA 每 token 存 {args.latent_dim} 维 latent, 随 ctx 线性增长。")

    # KV cache 随上下文长度的缩放对比 (L3D 核心卖点: 有界 vs 线性)
    print("\n== KV cache 缩放 (float32 元素数) ==")
    print(f"{'ctx':>8}{'Pyramis-L3D':>16}{'MLA':>16}{'L3D 节省':>14}")
    print("-" * 54)
    for ctx in (256, 512, 1024, 2048, 4096):
        mla_n = ctx * args.latent_dim
        ratio = (1 - kv["l3d"] / mla_n) * 100
        print(f"{ctx:>8}{kv['l3d']:>16,}{mla_n:>16,}{ratio:>13.1f}%")


if __name__ == "__main__":
    main()
