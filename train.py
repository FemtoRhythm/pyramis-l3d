"""Pyramis-L3D 训练脚本 (研究原型)。

用字符级 tokenizer 在开源数据集(默认 IMDB 本地缓存)上训练,
产出 .safetensors checkpoint + config + tokenizer vocab。

用法:
    python train.py --steps 200 --batch 16 --ctx 256 --lr 1e-3 --out ./checkpoint
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import CharTokenizer  # noqa: E402

IMDB_PARQUET = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "datasets--imdb", "snapshots",
    "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
    "plain_text", "train-00000-of-00001.parquet",
)


def load_texts(n_samples):
    import pyarrow.parquet as pq
    table = pq.read_table(IMDB_PARQUET)
    texts = [str(t) for t in table.column("text").to_pylist() if t]
    return texts[:n_samples]


def get_batch(tokens, batch, ctx, device):
    ix = torch.randint(0, len(tokens) - ctx - 1, (batch,))
    x = torch.stack([tokens[i:i + ctx] for i in ix])
    y = torch.stack([tokens[i + 1:i + ctx + 1] for i in ix])
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="./checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    print(f"== 加载数据 (IMDB train[:{args.nsamples}]) ==")
    texts = load_texts(args.nsamples)
    tok = CharTokenizer.build_from_texts(texts)
    vocab_size = tok.vocab_size
    print(f"  vocab_size={vocab_size}, 总字符~{sum(len(t) for t in texts)}")

    data = torch.tensor([i for t in texts for i in tok.encode(t)], dtype=torch.long)
    split = int(len(data) * 0.9)
    train_tokens, eval_tokens = data[:split], data[split:]

    print("== 构建模型 ==")
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    config = AutoConfig.from_pretrained(repo_dir, trust_remote_code=True)
    config.vocab_size = vocab_size
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        x, y = get_batch(train_tokens, args.batch, args.ctx, device)
        opt.zero_grad()
        out = model(x, labels=y)
        out.loss.backward()
        opt.step()
        if (step + 1) % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            ce = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)), y.view(-1)).item()
            print(f"  step {step + 1:4d}/{args.steps}  loss={out.loss.item():.4f}  "
                  f"CE={ce:.4f}  ppl={math.exp(ce):.2f}")

    elapsed = time.time() - t0
    print(f"  训练耗时 {elapsed:.1f}s")

    # 评估 PPL
    model.eval()
    total_ce = 0.0
    total_n = 0
    with torch.no_grad():
        for _ in range(10):
            x, y = get_batch(eval_tokens, args.batch, args.ctx, device)
            logits = model(x).logits
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_ce += ce.item() * y.numel()
            total_n += y.numel()
    ppl = math.exp(total_ce / total_n)
    print(f"  eval PPL = {ppl:.3f}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save(os.path.join(args.out, "vocab.json"))
    print(f"== 已保存到 {args.out} ==")


if __name__ == "__main__":
    main()
