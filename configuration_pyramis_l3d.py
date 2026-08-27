"""Pyramis-L3D 配置类。

Pyramis-L3D 是一个分层 / 稀疏注意力的研究原型, 核心思想:
- L1: 近端稠密 GQA(窗口内全秩)
- L2: CNN 池化压缩 KV(局部句法保留, 约 2x 压缩)
- L3: 可寻址字典(codebook)全局 KV(亚线性存储, 行数恒为 M)
- Latent-TLB: 查询时稀疏路由(仅对 top_k 命中行做 attention)
- distinctness 保护(W_distinct 近端逐行 latent, 不入字典)

本配置类继承 HF PretrainedConfig, 通过 config.json 的 auto_map 注册,
支持 AutoConfig / AutoModel / AutoModelForCausalLM.from_pretrained 直接加载。
"""

from transformers import PretrainedConfig


class PyramisL3DConfig(PretrainedConfig):
    model_type = "pyramis_l3d"

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        head_dim: int = 32,
        L1_window: int = 128,
        L2_window: int = 512,
        L2_conv_kernel: int = 5,
        L2_stride: int = 2,
        L3_codebook_size: int = 1024,
        L3_latent_dim: int = 64,
        L3_ema: float = 0.99,
        L3_temperature: float = 1.0,
        L3_commitment_weight: float = 0.25,
        L3_entropy_weight: float = 0.01,
        L3_dead_threshold: int = 100,
        W_distinct: int = 64,
        top_k: int = 16,
        tlb_mode: str = "ste",
        tlb_balance_weight: float = 0.01,
        tlb_sparsity_weight: float = 0.01,
        vocab_size: int = 32000,
        max_position_embeddings: int = 1024,
        intermediate_size: int = 1024,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        # 先注册自定义字段, 再调用父类初始化
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.L1_window = L1_window
        self.L2_window = L2_window
        self.L2_conv_kernel = L2_conv_kernel
        self.L2_stride = L2_stride
        self.L3_codebook_size = L3_codebook_size
        self.L3_latent_dim = L3_latent_dim
        self.L3_ema = L3_ema
        self.L3_temperature = L3_temperature
        self.L3_commitment_weight = L3_commitment_weight
        self.L3_entropy_weight = L3_entropy_weight
        self.L3_dead_threshold = L3_dead_threshold
        self.W_distinct = W_distinct
        self.top_k = top_k
        self.tlb_mode = tlb_mode
        self.tlb_balance_weight = tlb_balance_weight
        self.tlb_sparsity_weight = tlb_sparsity_weight
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta

        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            initializer_range=initializer_range,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
