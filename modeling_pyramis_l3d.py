"""Pyramis-L3D 模型实现 (Step 6)。

当前进度:
- [x] Step 0: 骨架可加载
- [x] Step 1: L1 稠密 GQA 注意力 + 完整残差 block
- [x] Step 2: L2 CNN 池化 KV 压缩
- [x] Step 3: Latent-TLB 查询时稀疏路由 + STE
- [x] Step 4: L3 可寻址字典 (codebook)
- [x] Step 5: distinctness 保护(W_distinct 逐行 latent)
- [x] Step 6: CNN 局部门控

后续步骤:
- Step 7: MoD 早退(可选)

HF 热启动友好: 权重命名对齐常见 GQA 模型
(q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj /
 embed_tokens / lm_head), 便于 Step 8 从开源小模型加载对齐权重。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from configuration_pyramis_l3d import PyramisL3DConfig


# ---------------------------------------------------------------------------
# RoPE 工具
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    """由绝对位置 [..., T] 计算 cos/sin 表 [..., T, head_dim]。"""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(positions.reshape(-1).float(), inv_freq)  # [N, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [N, head_dim]
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos.view(*positions.shape, head_dim), sin.view(*positions.shape, head_dim)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, heads, T, head_dim], cos/sin: [B, T, head_dim]
    return x * cos.unsqueeze(1) + _rotate_half(x) * sin.unsqueeze(1)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: 将 kv head 广播到 q head 数。x: [B, n_kv, T, head_dim]"""
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, T, hd).reshape(B, n_kv * n_rep, T, hd)


# ---------------------------------------------------------------------------
# L2 CNN 池化压缩器
# ---------------------------------------------------------------------------

class L2KVCompressor(nn.Module):
    """对最近 L2_window 个 token 的 K/V 做 depthwise 1D conv 池化压缩。

    输入: K, V [B, n_kv_heads, T, head_dim], positions [B, T]
    输出: K2, V2 [B, n_kv_heads, ~T/stride, head_dim] 及每行的代表位置 rep_pos [B, T2]
    门控: 1x1 depthwise conv + sigmoid 控制池化强度。
    """

    def __init__(self, config: PyramisL3DConfig):
        super().__init__()
        self.kernel = config.L2_conv_kernel
        self.stride = config.L2_stride
        self.head_dim = config.head_dim

        # depthwise 1D conv over 序列维 (groups=head_dim, 每个 channel 独立)
        self.k_conv = nn.Conv1d(
            self.head_dim, self.head_dim, kernel_size=self.kernel,
            stride=self.stride, padding=self.kernel // 2, groups=self.head_dim,
        )
        self.v_conv = nn.Conv1d(
            self.head_dim, self.head_dim, kernel_size=self.kernel,
            stride=self.stride, padding=self.kernel // 2, groups=self.head_dim,
        )
        # 可学习门控
        self.k_gate = nn.Conv1d(self.head_dim, self.head_dim, kernel_size=1, groups=self.head_dim)
        self.v_gate = nn.Conv1d(self.head_dim, self.head_dim, kernel_size=1, groups=self.head_dim)

    def forward(self, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor):
        # k, v: [B, n_kv, T, head_dim]
        B, n_kv, T, hd = k.shape
        k = k.permute(0, 1, 3, 2).reshape(B * n_kv, hd, T)
        v = v.permute(0, 1, 3, 2).reshape(B * n_kv, hd, T)

        k2 = self.k_conv(k)
        v2 = self.v_conv(v)
        k2 = k2 * torch.sigmoid(self.k_gate(k2))
        v2 = v2 * torch.sigmoid(self.v_gate(v2))

        T2 = k2.shape[-1]
        k2 = k2.reshape(B, n_kv, hd, T2).permute(0, 1, 3, 2)
        v2 = v2.reshape(B, n_kv, hd, T2).permute(0, 1, 3, 2)

        # 每行的代表位置 = 其感受野中心源位置 (o * stride)
        idx = (torch.arange(T2, device=k.device) * self.stride).clamp(max=T - 1)
        rep_pos = positions[:, idx]  # [B, T2]
        return k2, v2, rep_pos


# ---------------------------------------------------------------------------
# Latent-TLB 路由器
# ---------------------------------------------------------------------------

class TLBRouter(nn.Module):
    """轻量路由器: 把 query 翻译成跨所有候选行的打分 scores。

    内容式路由: scores = (W_q · q_route) · row_keys^T, 带可学习温度。
    这里 row_keys 取候选 K 的 kv-head 均值, q_route 取 q 的 head 均值。
    """

    def __init__(self, config: PyramisL3DConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.head_dim, config.head_dim, bias=False)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, q_route: torch.Tensor, row_keys: torch.Tensor) -> torch.Tensor:
        q_route = self.q_proj(q_route)
        scores = torch.matmul(q_route, row_keys.transpose(-2, -1)) * self.scale
        return scores * self.temperature


# ---------------------------------------------------------------------------
# L3 可寻址字典 (codebook)
# ---------------------------------------------------------------------------

class L3Dictionary(nn.Module):
    """可学习 codebook C ∈ R^{M × latent_dim}。

    远端 token(超出 L2_window) 的 K 经 CNN-pool 得 feature f ∈ R^{latent_dim},
    再分配到最近的 codebook entry。L3 的 K3/V3 = 上投影后的 codebook 本身,
    行数恒为 M, 与远端 token 数量无关(亚线性存储关键)。

    - 分配: 训练用 soft assignment(softmax 相似度加权, 可微); 推理/commitment 用 hard 最近邻。
    - 更新: EMA 更新 C ← (1-ema)C + ema*(分配到该 entry 的 feature 均值)。
    - commitment loss = ||f - C[idx]||² (梯度只回传到 f, C 由 EMA 驱动)。
    """

    def __init__(self, config: PyramisL3DConfig):
        super().__init__()
        self.M = config.L3_codebook_size
        self.latent_dim = config.L3_latent_dim
        self.head_dim = config.head_dim
        self.ema = config.L3_ema
        self.temperature = config.L3_temperature
        self.dead_threshold = config.L3_dead_threshold

        # 可学习 codebook
        self.codebook = nn.Parameter(torch.empty(self.M, self.latent_dim))
        nn.init.normal_(self.codebook, std=0.02)

        # 上投影: codebook -> K3/V3 (head_dim)
        self.k_up = nn.Linear(self.latent_dim, self.head_dim, bias=False)
        self.v_up = nn.Linear(self.latent_dim, self.head_dim, bias=False)
        # 特征提取: 远端 [K; V] -> latent_dim。
        # 正交初始化保持随机初始的等向性, 避免 feature 退化到少数方向导致 codebook 利用率过低。
        self.f_proj = nn.Linear(2 * self.head_dim, self.latent_dim, bias=False)
        nn.init.orthogonal_(self.f_proj.weight)
        # CNN 局部门控池化 (depthwise); 初始化为恒等(中心 delta), 随机初始时不破坏等向性
        self.cnn_pool = nn.Conv1d(
            self.latent_dim, self.latent_dim, kernel_size=5,
            stride=1, padding=2, groups=self.latent_dim, bias=False,
        )
        nn.init.zeros_(self.cnn_pool.weight)
        with torch.no_grad():
            self.cnn_pool.weight[:, 0, self.cnn_pool.kernel_size[0] // 2] = 1.0

        # 诊断(供测试 hook 读取)
        self._last_hard_idx = None
        self._last_utilization = 0.0

    def forward(self, k_remote: torch.Tensor, v_remote: torch.Tensor):
        # k_remote, v_remote: [B, n_kv, T_remote, head_dim]
        B, n_kv, T_remote, _ = k_remote.shape

        # K3/V3 = codebook 上投影 (去 detach: 接 attention/CE 梯度, 维持非零模长)
        k3 = self.k_up(self.codebook)  # [M, head_dim]
        v3 = self.v_up(self.codebook)  # [M, head_dim]

        commitment = torch.zeros((), device=k_remote.device)
        entropy_loss = torch.zeros((), device=k_remote.device)
        hard_idx = None

        if T_remote > 0:
            f_in = torch.cat([k_remote, v_remote], dim=-1)  # [B, n_kv, T_remote, 2*head_dim]
            f = self.f_proj(f_in)  # [B, n_kv, T_remote, latent_dim]

            # CNN over 序列维
            f = f.permute(0, 1, 3, 2).reshape(B * n_kv, self.latent_dim, T_remote)
            f = self.cnn_pool(f)  # [B*n_kv, latent_dim, T_remote]
            f = f.reshape(B, n_kv, self.latent_dim, T_remote).permute(0, 1, 3, 2)

            # 分配: 余弦相似度 -> hard 最近邻 + soft 可微分配
            f_n = F.normalize(f, dim=-1)
            c_n = F.normalize(self.codebook, dim=-1)  # [M, latent_dim] (去 detach: 熵/分配梯度回传)
            sim = torch.matmul(f_n, c_n.t())  # [B, n_kv, T_remote, M]
            hard_idx = sim.argmax(dim=-1)  # [B, n_kv, T_remote]
            soft_assign = F.softmax(sim / self.temperature, dim=-1)

            # commitment loss = ||f - C[idx]||² (去 detach: 双向回传, codebook 向 f 对齐)
            c_idx = self.codebook[hard_idx]  # [B, n_kv, T_remote, latent_dim]
            commitment = ((f - c_idx) ** 2).sum(dim=-1).mean()

            # usage entropy 损失: 鼓励 codebook 均匀使用 (对抗坍塌)
            p_j = soft_assign.mean(dim=(0, 1, 2))  # [M] 平均使用概率, 和为 1
            usage_entropy = -(p_j * torch.log(p_j + 1e-8)).sum()
            entropy_loss = -usage_entropy  # 最小化 -> 最大化使用熵(均匀)

            # EMA 更新 codebook (no_grad, 硬分配)
            if self.training and self.ema < 1.0:
                self._ema_update(f.detach(), hard_idx)

        # 诊断
        if hard_idx is not None:
            flat_idx = hard_idx.detach().reshape(-1)
            self._last_hard_idx = flat_idx
            used = torch.unique(flat_idx).numel()
            self._last_utilization = used / self.M
        else:
            self._last_hard_idx = None
            self._last_utilization = 0.0

        return k3, v3, commitment, entropy_loss

    def _ema_update(self, f: torch.Tensor, hard_idx: torch.Tensor):
        # f: [B, n_kv, T_pool, latent_dim], hard_idx: [B, n_kv, T_pool]
        flat_f = f.reshape(-1, self.latent_dim)  # [N, latent_dim]
        flat_idx = hard_idx.reshape(-1)  # [N]
        onehot = torch.zeros(flat_idx.numel(), self.M, device=f.device, dtype=f.dtype)
        onehot.scatter_(1, flat_idx.unsqueeze(1), 1.0)  # [N, M]
        new = torch.matmul(onehot.t(), flat_f)  # [M, latent_dim]
        count = onehot.sum(dim=0)  # [M] 硬分配命中数
        mean = new / (count.unsqueeze(-1) + 1e-8)
        mask = (count > 1e-8).unsqueeze(-1)

        updated = (1.0 - self.ema) * self.codebook.data + self.ema * mean
        self.codebook.data.copy_(torch.where(mask, updated, self.codebook.data))

        # 死码字复活: 长期未被硬分配的 entry 用当前 batch 随机 feature 重新初始化
        # (参考 VQ-VAE-2 / SoundStream)
        if self.dead_threshold is not None and self.dead_threshold > 0:
            dead = count < self.dead_threshold  # [M]
            n_dead = int(dead.sum().item())
            if n_dead > 0:
                sample_idx = torch.randint(0, flat_f.shape[0], (n_dead,), device=f.device)
                self.codebook.data[dead] = flat_f[sample_idx].detach()


# ---------------------------------------------------------------------------
# L1 稠密 GQA + L2 CNN 池化 层次注意力 (带 TLB 稀疏路由)
# ---------------------------------------------------------------------------

class PyramisL3DAttention(nn.Module):
    """层次注意力 + TLB 稀疏路由。

    候选行 = L1(最近 L1_window 稠密 key) + L2(最近 L2_window CNN 池化压缩)。
    路由器对候选行打分, 选择 top_k 行做 attention (STE / soft 两种模式)。

    - tlb_mode="ste": 前向 hard top-k 掩码, 反向 softmax(scores) 梯度回传 (STE)
    - tlb_mode="soft": softmax(scores) 加权全行 soft attention + 稀疏损失
    """

    def __init__(self, config: PyramisL3DConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_kv_heads = config.n_heads // 4  # GQA 分组
        self.n_rep = self.n_heads // self.n_kv_heads
        self.L1_window = config.L1_window
        self.L2_window = config.L2_window
        self.M = config.L3_codebook_size
        self.W_distinct = config.W_distinct
        self.latent_dim = config.L3_latent_dim
        self.top_k = config.top_k
        self.tlb_mode = config.tlb_mode
        self.rope_theta = config.rope_theta
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        self.l2_compressor = L2KVCompressor(config)
        self.l3_dict = L3Dictionary(config)
        self.router = TLBRouter(config)

        # distinctness 保护: 最近 W_distinct 个 token 保留逐行 latent(不入字典)
        # 低秩 [K;V] -> latent_dim -> 上投影回 head_dim
        self.distinct_down = nn.Linear(2 * self.head_dim, self.latent_dim, bias=False)
        self.distinct_k_up = nn.Linear(self.latent_dim, self.head_dim, bias=False)
        self.distinct_v_up = nn.Linear(self.latent_dim, self.head_dim, bias=False)

        # 诊断(供测试 hook 读取)
        self._last_hard = None
        self._last_soft = None
        self._last_commitment = None
        self._last_n_cand = None

    def _distinct_rows(self, k_dist: torch.Tensor, v_dist: torch.Tensor):
        # k_dist, v_dist: [B, n_kv, W, head_dim] -> 逐行 latent 再上投影
        f = torch.cat([k_dist, v_dist], dim=-1)  # [B, n_kv, W, 2*head_dim]
        z = self.distinct_down(f)  # [B, n_kv, W, latent_dim]
        k4 = self.distinct_k_up(z)  # [B, n_kv, W, head_dim]
        v4 = self.distinct_v_up(z)  # [B, n_kv, W, head_dim]
        return k4, v4

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value=None,
        use_cache: bool = False,
    ):
        B, Tq, _ = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device

        q = self.q_proj(hidden_states).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, Tq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, Tq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            k_cache, v_cache, pos_cache = past_key_value
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            key_pos = torch.cat([pos_cache, position_ids], dim=1)
        else:
            key_pos = position_ids

        # RoPE
        cos_q, sin_q = _rope_cos_sin(position_ids, self.head_dim, self.rope_theta, dtype)
        cos_k, sin_k = _rope_cos_sin(key_pos, self.head_dim, self.rope_theta, dtype)
        q = _apply_rope(q, cos_q, sin_q)
        k = _apply_rope(k, cos_k, sin_k)

        Tk = k.shape[2]

        # ---- L1 稠密: 最近 L1_window 个 key (有界, 不随 seq 增长) ----
        l1_len = min(Tk, self.L1_window)
        k1 = k[:, :, -l1_len:, :]
        v1 = v[:, :, -l1_len:, :]
        pos1 = key_pos[:, -l1_len:]

        # ---- L2 CNN 池化 (最近 L2_window 个 key) ----
        k_l2 = k[:, :, -self.L2_window:, :]
        v_l2 = v[:, :, -self.L2_window:, :]
        pos_l2 = key_pos[:, -self.L2_window:]
        k2, v2, rep_pos2 = self.l2_compressor(k_l2, v_l2, pos_l2)

        # ---- W_distinct 逐行 latent (最近 W_distinct 个 token, 不入字典) ----
        w_len = min(Tk, self.W_distinct)
        k4, v4 = self._distinct_rows(k[:, :, -w_len:, :], v[:, :, -w_len:, :])
        pos4 = key_pos[:, -w_len:]

        # ---- L3 远端字典 (超出 L2_window 的 token 入 codebook) ----
        remote_len = max(0, Tk - self.L2_window)
        k3, v3, commitment, l3_entropy_loss = self.l3_dict(
            k[:, :, :remote_len, :], v[:, :, :remote_len, :]
        )
        # codebook 共享于所有 kv head / batch: [M, head_dim] -> [B, n_kv, M, head_dim]
        k3 = k3.unsqueeze(0).unsqueeze(0).expand(B, self.n_kv_heads, self.M, self.head_dim)
        v3 = v3.unsqueeze(0).unsqueeze(0).expand(B, self.n_kv_heads, self.M, self.head_dim)

        # ---- 候选行: L1(有界) + L2(池化) + W(逐行 latent) + L3(M 行 codebook) ----
        K_cand = torch.cat([k1, k2, k4, k3], dim=2)  # [B, n_kv, l1_len+T2+w_len+M, head_dim]
        V_cand = torch.cat([v1, v2, v4, v3], dim=2)
        l3_pos = torch.zeros(B, self.M, device=device, dtype=key_pos.dtype)
        P_cand = torch.cat([pos1, rep_pos2, pos4, l3_pos], dim=1)  # [B, n_cand]
        n_cand = K_cand.shape[2]

        # ---- TLB 路由打分 ----
        q_route = q.mean(dim=1)  # [B, Tq, head_dim]
        row_keys = K_cand.mean(dim=1)  # [B, n_cand, head_dim]
        scores = self.router(q_route, row_keys)  # [B, Tq, n_cand]

        # 合法性掩码: L1 需窗口 + causal; L2/W 只需 causal; L3 恒合法(全局字典)
        q_pos = position_ids[:, :, None]  # [B, Tq, 1]
        p_cand = P_cand[:, None, :]  # [B, 1, n_cand]
        arange = torch.arange(n_cand, device=device)
        i_l1 = l1_len
        i_l2 = i_l1 + k2.shape[2]
        i_w = i_l2 + w_len
        is_l1 = arange < i_l1
        is_l2 = (arange >= i_l1) & (arange < i_l2)
        is_w = (arange >= i_l2) & (arange < i_w)
        l1_valid = (p_cand <= q_pos) & ((q_pos - p_cand) < self.L1_window)
        l2_valid = p_cand <= q_pos
        w_valid = p_cand <= q_pos
        l3_valid = torch.ones_like(l2_valid)
        valid = torch.where(
            is_l1.view(1, 1, -1), l1_valid,
            torch.where(
                is_l2.view(1, 1, -1), l2_valid,
                torch.where(is_w.view(1, 1, -1), w_valid, l3_valid),
            ),
        )  # [B, Tq, n_cand]

        scores = scores.masked_fill(~valid, float("-inf"))
        soft = F.softmax(scores, dim=-1)

        # top-k 硬选择 (诊断用; ste 模式也用它)
        k_actual = min(self.top_k, n_cand)
        _topk_vals, topk_idx = scores.topk(k_actual, dim=-1)
        hard = torch.zeros_like(scores).scatter(-1, topk_idx, 1.0)

        # gate: ste -> hard(前向)/soft(反向); soft -> soft
        if self.tlb_mode == "ste":
            gate = hard + soft - soft.detach()  # STE
        else:
            gate = soft

        # ---- 稀疏注意力 ----
        K_rep = _repeat_kv(K_cand, self.n_rep)
        V_rep = _repeat_kv(V_cand, self.n_rep)
        attn_scores = torch.matmul(q, K_rep.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn_scores, dim=-1)
        attn_gated = attn * gate.unsqueeze(1)
        attn_gated = attn_gated / (attn_gated.sum(dim=-1, keepdim=True) + 1e-8)
        out = torch.matmul(attn_gated, V_rep)

        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        out = self.o_proj(out)

        # ---- 辅助损失 ----
        balance_loss, sparsity_loss = self._aux_losses(soft, hard, n_cand)

        past_key_value = None
        if use_cache:
            new_k = k[:, :, -self.L1_window:, :]
            new_v = v[:, :, -self.L1_window:, :]
            new_pos = key_pos[:, -self.L1_window:]
            past_key_value = (new_k, new_v, new_pos)

        # 诊断
        self._last_hard = hard.detach()
        self._last_soft = soft.detach()
        self._last_commitment = commitment.detach()
        self._last_n_cand = n_cand

        return out, past_key_value, balance_loss, sparsity_loss, commitment, l3_entropy_loss

    def _aux_losses(self, soft: torch.Tensor, hard: torch.Tensor, n_cand: int):
        """负载均衡损失 + 稀疏(低熵)损失。"""
        p_j = soft.mean(dim=(0, 1))  # [n_cand]
        f_j = hard.float().mean(dim=(0, 1))  # [n_cand]
        balance = n_cand * (f_j * p_j).sum()

        log_soft = torch.log(soft + 1e-8)
        entropy = -(soft * log_soft).sum(dim=-1).mean()
        sparsity = entropy  # 最小化熵 -> 软选择更尖(稀疏)

        return balance, sparsity


# ---------------------------------------------------------------------------
# MLP (SwiGLU)
# ---------------------------------------------------------------------------

class PyramisL3DMLP(nn.Module):
    def __init__(self, config: PyramisL3DConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# CNN 局部门控 (Step 6)
# ---------------------------------------------------------------------------

class CNNLocalGate(nn.Module):
    """1D-CNN 局部门控: 增强短距离依赖, 修复稀疏/潜空间注意力的局部语法崩坏。

    depthwise conv(kernel=7, padding=3) + GELU + 1x1 conv + sigmoid 门控,
    out = x * gate + x (残差)。插在 attention 之后、MLP 之前。
    """

    def __init__(self, config: PyramisL3DConfig):
        super().__init__()
        d = config.d_model
        self.dw_conv = nn.Conv1d(d, d, kernel_size=7, padding=3, groups=d)
        self.pointwise = nn.Conv1d(d, d, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        g = x.transpose(1, 2)  # [B, d_model, T]
        g = self.dw_conv(g)
        g = self.act(g)
        g = self.pointwise(g)
        gate = torch.sigmoid(g).transpose(1, 2)  # [B, T, d_model]
        return x * gate + x


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class PyramisL3DBlock(nn.Module):
    """RMSNorm -> PyramisL3DAttention(L1+L2+W+L3+TLB) -> CNN 局部门控 -> SwiGLU MLP -> 残差。"""

    def __init__(self, config: PyramisL3DConfig):
        super().__init__()
        self.config = config
        self.input_layernorm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.self_attn = PyramisL3DAttention(config)
        self.cnn_gate = CNNLocalGate(config)
        self.post_attention_layernorm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = PyramisL3DMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value=None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out, present_kv, balance_loss, sparsity_loss, commitment_loss, l3_entropy_loss = self.self_attn(
            hidden_states, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + attn_out

        # Step 6: CNN 局部门控 (attention 之后、MLP 之前)
        hidden_states = self.cnn_gate(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_kv, balance_loss, sparsity_loss, commitment_loss, l3_entropy_loss


# ---------------------------------------------------------------------------
# 主干模型 + CausalLM
# ---------------------------------------------------------------------------

class PyramisL3DModel(PreTrainedModel):
    config_class = PyramisL3DConfig

    def __init__(self, config: PyramisL3DConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [PyramisL3DBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        B, T = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(B, T)

        hidden_states = self.embed_tokens(input_ids)

        past_key_values = [] if use_cache else None
        total_balance = torch.zeros((), device=input_ids.device)
        total_sparsity = torch.zeros((), device=input_ids.device)
        total_commitment = torch.zeros((), device=input_ids.device)
        total_l3_entropy = torch.zeros((), device=input_ids.device)
        for block in self.blocks:
            hidden_states, present_kv, balance_loss, sparsity_loss, commitment_loss, l3_entropy_loss = block(
                hidden_states, position_ids, None, use_cache
            )
            if use_cache:
                past_key_values.append(present_kv)
            total_balance = total_balance + balance_loss
            total_sparsity = total_sparsity + sparsity_loss
            total_commitment = total_commitment + commitment_loss
            total_l3_entropy = total_l3_entropy + l3_entropy_loss

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values, total_balance, total_sparsity, total_commitment, total_l3_entropy


class PyramisL3DForCausalLM(PreTrainedModel):
    config_class = PyramisL3DConfig

    def __init__(self, config: PyramisL3DConfig):
        super().__init__(config)
        self.model = PyramisL3DModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        hidden_states, past_key_values, total_balance, total_sparsity, total_commitment, total_l3_entropy = self.model(
            input_ids, position_ids=position_ids, use_cache=use_cache
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            loss = (
                ce_loss
                + self.config.tlb_balance_weight * total_balance
                + self.config.tlb_sparsity_weight * total_sparsity
                + self.config.L3_commitment_weight * total_commitment
                + self.config.L3_entropy_weight * total_l3_entropy
            )

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=past_key_values
        )
