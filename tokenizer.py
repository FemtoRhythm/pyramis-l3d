"""极简字符级 tokenizer (自包含, 无需联网)。

研究原型用: 把文本按字符切分, 建立 char <-> id 映射。
适用于小规模实验, 词汇表极小(约几百), 满足"少参数"目标。
"""

import json


class CharTokenizer:
    def __init__(self, stoi=None):
        self.stoi = stoi if stoi is not None else {}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    @classmethod
    def build_from_texts(cls, texts):
        chars = sorted(set("".join(texts)))
        stoi = {c: i for i, c in enumerate(chars)}
        return cls(stoi)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            stoi = json.load(f)
        return cls(stoi)
