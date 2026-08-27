"""Pyramis-L3D 评测脚本 (研究原型)。

覆盖: PPL (held-out) / 亚线性行池 scaling / 路由健康 (字典利用率 + 命中行数)。

用法:
    python eval.py [--checkpoint ./checkpoint] [--ctx 512]
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import CharTokenizer  # noqa: E402

IMDB_TEST_PARQUET = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "datasets--imdb", "snapshots",
    "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
    "plain_text", "test-00000-of-00001.parquet",
)


def load_test_texts(n_samples):
    import pyarrow.parquet as pq
    table = pq.read_table(IMDB_TEST_PARQUET)
    texts = [str(t) for t in table.column("text").to_pylist() if t]
    return texts[:n_samples]


def get_batch(tokens, batch, ctx, device):
    ix = torch.randint(0, len(tokens) - ctx - 1, (batch,))
    x = torch.stack([tokens[i:i + ctx] for i in ix])
    y = torch.stack([tokens[i + 1:i + ctx + 1] for i in ix])
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="./checkpoint")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # 模型 + tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True
    ).to(device)
    model.eval()
    tok = CharTokenizer.load(os.path.join(args.checkpoint, "vocab.json"))

    # 数据 (跳过 vocab 外的字符, 保证与训练字符集一致)
    texts = load_test_texts(1000)
    known = set(tok.stoi)
    data = torch.tensor(
        [tok.stoi[c] for t in texts for c in t if c in known], dtype=torch.long
    )

    # ---- PPL ----
    total_ce = 0.0
    total_n = 0
    with torch.no_grad():
        for _ in range(args.eval_iters):
            x, y = get_batch(data, args.batch, args.ctx, device)
            logits = model(x).logits
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_ce += ce.item() * y.numel()
            total_n += y.numel()
    ppl = math.exp(total_ce / total_n)
    print(f"PPL (held-out) = {ppl:.3f}")

    # ---- 亚线性行池 scaling ----
    print("\n== 可路由行池 vs ctx (应恒定) ==")
    attn = model.model.blocks[0].self_attn
    n_kv, hd = attn.n_kv_heads, attn.head_dim
    for ctx in (256, 512, 1024, 2048):
        x = torch.randint(0, tok.vocab_size, (1, ctx))
        model(x)
        print(f"  ctx={ctx:>5} -> n_cand={attn._last_n_cand}")

    # ---- 路由健康: 字典利用率 ----
    print("\n== 路由健康 ==")
    x = torch.randint(0, tok.vocab_size, (1, 2048))
    model(x)
    utils = [b.self_attn.l3_dict._last_utilization for b in model.model.blocks]
    print(f"  codebook 利用率: {[f'{u:.3f}' for u in utils]}")
    cb_norms = [b.self_attn.l3_dict.codebook.norm().item() for b in model.model.blocks]
    fp_norms = [b.self_attn.l3_dict.f_proj.weight.norm().item() for b in model.model.blocks]
    ku_norms = [b.self_attn.l3_dict.k_up.weight.norm().item() for b in model.model.blocks]
    print(f"  codebook norm: {[f'{v:.4f}' for v in cb_norms]}")
    print(f"  f_proj norm:   {[f'{v:.4f}' for v in fp_norms]}")
    print(f"  k_up norm:     {[f'{v:.4f}' for v in ku_norms]}")
    top_k = model.config.top_k
    hard_sums = [b.self_attn._last_hard.sum(-1).mean().item() for b in model.model.blocks]
    print(f"  每 query 命中行数(应==top_k={top_k}): {[f'{h:.1f}' for h in hard_sums]}")


if __name__ == "__main__":
    main()
