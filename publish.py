"""Pyramis-L3D 打包并发布到 Hugging Face。

用法:
    python publish.py [--repo FemtoRhythm/pyramis-l3d] [--token <HF_TOKEN>] [--dry-run]

步骤:
  1. 组装 release/ 目录 (模型权重 + 代码 + benchmark + tests + model card)
  2. 在 HF 创建/复用 model repo
  3. 上传 release/ 到 repo
"""

import argparse
import os
import shutil
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.join(REPO_DIR, "release")
CHECKPOINT_DIR = os.path.join(REPO_DIR, "checkpoint")

# (源路径, 目标相对路径) —— 源路径相对 REPO_DIR
FILES = [
    ("checkpoint/config.json", "config.json"),
    ("checkpoint/model.safetensors", "model.safetensors"),
    ("checkpoint/vocab.json", "vocab.json"),
    ("checkpoint/generation_config.json", "generation_config.json"),
    ("configuration_pyramis_l3d.py", "configuration_pyramis_l3d.py"),
    ("modeling_pyramis_l3d.py", "modeling_pyramis_l3d.py"),
    ("tokenizer.py", "tokenizer.py"),
    ("train.py", "train.py"),
    ("eval.py", "eval.py"),
    ("LICENSE", "LICENSE"),
    (".gitattributes", ".gitattributes"),
    ("MODEL_CARD.md", "README.md"),
    ("logo-light.png", "logo-light.png"),
]

# 目录 (源目录, 目标目录, 忽略的 glob 列表)
DIRS = [
    ("benchmark", "benchmark", ["*.pyc", "__pycache__", ".venv", ".lock", "CACHEDIR.TAG"]),
]


def build_release():
    if os.path.isdir(RELEASE_DIR):
        shutil.rmtree(RELEASE_DIR)
    os.makedirs(RELEASE_DIR)

    for src, dst in FILES:
        s = os.path.join(REPO_DIR, src)
        d = os.path.join(RELEASE_DIR, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(s, d)
        print(f"  copy {src} -> {dst}")

    for src_dir, dst_dir, ignore in DIRS:
        s = os.path.join(REPO_DIR, src_dir)
        d = os.path.join(RELEASE_DIR, dst_dir)
        shutil.copytree(s, d, ignore=shutil.ignore_patterns(*ignore))
        print(f"  copy {src_dir}/ -> {dst_dir}/")

    print(f"release/ 组装完成 -> {RELEASE_DIR}")


def upload(repo_id: str, token: str):
    from huggingface_hub import HfApi

    api = HfApi(token=token, endpoint="https://huggingface.co")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"repo 就绪: {repo_id}")

    api.upload_folder(
        folder_path=RELEASE_DIR,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Release Pyramis-L3D research prototype (model + code + benchmark)",
        delete_patterns=["tests/**", "conftest.py"],
    )
    print(f"上传完成: https://huggingface.co/{repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default="FemtoRhythm/pyramis-l3d")
    ap.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    build_release()

    if args.dry_run:
        print("[dry-run] 跳过上传")
        return

    if not args.token:
        print("缺少 HF token (传 --token 或设 HF_TOKEN 环境变量)")
        sys.exit(1)

    upload(args.repo, args.token)


if __name__ == "__main__":
    main()
