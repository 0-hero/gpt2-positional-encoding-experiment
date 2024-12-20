# gpt2-model-positional-encodings.py

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import necessary modules for different positional encodings
import numpy as np
import scipy.special
import scipy.signal

from packaging import version

# Check if scaled_dot_product_attention is available and supports flash attention
use_flash_attn = 'scaled_dot_product_attention' in dir(F) and version.parse(torch.__version__) >= version.parse('2.0.0')
if use_flash_attn:
    print("Flash Attention v2 is available and will be used where possible.")
else:
    print("Flash Attention v2 is not available. Using standard attention.")

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

def get_positional_encoding(position, d_model, method, max_len=5000):
    """
    Generate positional encodings based on the specified method.
    """
    if method == 'default':
        return None  # Handled by nn.Embedding in the model
    elif method == 'learned':
        return None  # Handled by nn.Embedding in the model
    elif method == 'sinusoidal':
        pe = torch.zeros(max_len, d_model)
        position_enc = position.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position_enc * div_term)
        pe[:, 1::2] = torch.cos(position_enc * div_term)
        return pe
    elif method == 'exponential':
        pe = torch.exp(-position.float() / max_len).unsqueeze(1).repeat(1, d_model)
        return pe
    elif method == 'polynomial_legendre':
        pe = torch.zeros(max_len, d_model)
        x = (position / max_len * 2) - 1  # Scale positions to [-1,1]
        for i in range(d_model):
            pe[:, i] = scipy.special.eval_legendre(i, x)
        return pe
    elif method == 'polynomial_chebyshev':
        pe = torch.zeros(max_len, d_model)
        x = (position / max_len * 2) - 1  # Scale positions to [-1,1]
        for i in range(d_model):
            pe[:, i] = scipy.special.eval_chebyt(i, x)
        return pe
    elif method == 'gaussian':
        pe = torch.zeros(max_len, d_model)
        positions = position.float()
        means = torch.linspace(0, max_len, d_model)
        std = max_len / d_model
        for i in range(d_model):
            pe[:, i] = torch.exp(- ((positions - means[i]) **2) / (2 * std **2))
        return pe
    elif method == 'random_fourier':
        B = torch.randn(d_model, 1)
        x = position.float() / max_len
        x = x @ B.T * 2 * math.pi
        pe = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        return pe[:, :d_model]
    elif method == 'wavelet':
        pe = torch.zeros(max_len, d_model)
        scales = torch.arange(1, d_model+1)
        x = position.float()
        for i in range(d_model):
            wavelet = scipy.signal.ricker(points=max_len, a=scales[i])
            pe[:, i] = torch.from_numpy(wavelet[position])
        return pe
    elif method == 'bessel':
        pe = torch.zeros(max_len, d_model)
        x = position.float()
        for i in range(d_model):
            pe[:, i] = scipy.special.jv(i, x)
        return pe
    elif method == 'alternative':
        pe = torch.zeros(max_len, d_model)
        position_enc = position.float()
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.tan(position_enc * div_term)
        pe[:, 1::2] = torch.sin(position_enc * div_term + math.pi / 4)
        return pe
    elif method == 'none':
        return torch.zeros(max_len, d_model)
    else:
        raise ValueError(f"Unknown positional encoding method: {method}")

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = self.n_embd // self.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Implement attention-level positional encodings
        if config.attention_type == 'rope':
            self.rotary_dim = self.n_embd // self.n_head
            if self.rotary_dim % 2 != 0:
                self.rotary_dim -= self.rotary_dim % 2  # Ensure even dimension
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
            self.register_buffer('inv_freq', inv_freq)
        elif config.attention_type == 'alibi':
            slopes = self.get_alibi_slopes(self.n_head)
            self.register_buffer('alibi_slopes', slopes)
        elif config.attention_type == 'relative':
            num_rel_dis = 2 * config.block_size - 1
            self.relative_positions = nn.Embedding(num_rel_dis, self.n_head)
        # else: default attention (nothing extra to define)

    def get_alibi_slopes(self, n_heads):
        def get_slopes(n):
            import math
            def get_slopes_power_of_2(n):
                start = 2 ** (-2 ** -(math.log2(n) - 3))
                ratio = start
                return [start * (ratio ** i) for i in range(n)]
            if math.log2(n).is_integer():
                return torch.Tensor(get_slopes_power_of_2(n))
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                slopes = get_slopes_power_of_2(closest_power_of_2)
                extra_slopes = get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
                return torch.Tensor(slopes + extra_slopes)
        slopes = get_slopes(n_heads)
        return slopes.view(n_heads, 1, 1)

    def apply_rope(self, x):
        # x: (B, n_head, T, head_dim)
        seq_len = x.size(-2)
        device = x.device
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)  # (T, rotary_dim)
        emb = emb[None, None, :, :]  # (1, 1, T, rotary_dim)
        x1 = x[..., :self.rotary_dim]
        x2 = x[..., self.rotary_dim:]
        x1_rot = x1 * emb + torch.flip(x1, dims=[-1]) * torch.flip(emb, dims=[-1])
        x = torch.cat((x1_rot, x2), dim=-1)
        return x

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        qkv = self.c_attn(x).view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_head, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, n_head, T, head_dim)
        
        if self.config.attention_type == 'rope':
            q = self.apply_rope(q)
            k = self.apply_rope(k)

        # Decide whether to use Flash Attention based on training/evaluation mode and tracking flags
        if use_flash_attn and self.config.attention_type in ['default', 'rope'] and not (self.config.track_attention_patterns and not self.training):
            # Use PyTorch's scaled_dot_product_attention which leverages Flash Attention 2
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Standard attention mechanism
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if self.config.attention_type == 'alibi':
                position_ids = torch.arange(T, device=x.device).unsqueeze(0).unsqueeze(0)
                alibi = self.alibi_slopes.to(x.device) * position_ids  # (n_head, 1, T)
                attn_scores = attn_scores + alibi

            elif self.config.attention_type == 'relative':
                positions = torch.arange(-T+1, T, device=x.device)
                rel_pos = self.relative_positions(positions + T -1)
                attn_scores = attn_scores + rel_pos

            # Apply causal mask
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

            # Collect attention patterns if required
            if self.config.track_attention_patterns and not self.training:
                self.attn_weights = attn_weights.detach().cpu()
            y = torch.matmul(attn_weights, v)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    embedding_type: str = 'default'  # Default uses learned positional embeddings
    attention_type: str = 'default'  # Default attention without any modifications
    track_activations: bool = False
    track_attention_patterns: bool = False

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict()
        self.transformer['wte'] = nn.Embedding(config.vocab_size, config.n_embd)

        if config.embedding_type in ['learned', 'default']:
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
            self.pos_emb = None
        elif config.embedding_type == 'none':
            self.transformer['wpe'] = None
            self.pos_emb = None
        else:
            self.transformer['wpe'] = None
            position = torch.arange(0, config.block_size)
            pe = get_positional_encoding(position, config.n_embd, config.embedding_type, config.block_size)
            self.register_buffer('pos_emb', pe)

        self.transformer['drop'] = nn.Dropout(config.dropout)
        self.transformer['h'] = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.transformer['ln_f'] = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer['wte'].weight = self.lm_head.weight  # Weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Initialize activations and attention patterns
        self.activations = []
        self.attention_patterns = []

        print("Number of parameters: {:.2f}M".format(self.get_num_params() / 1e6))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer['wpe'] is not None:
            n_params -= self.transformer['wpe'].weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        tok_emb = self.transformer['wte'](idx)  # token embeddings

        if self.config.embedding_type in ['learned', 'default']:
            pos_emb = self.transformer['wpe'](pos)
            x = tok_emb + pos_emb
        elif self.config.embedding_type == 'none':
            x = tok_emb
        else:
            pos_emb = self.pos_emb[:t, :].to(device)
            x = tok_emb + pos_emb.unsqueeze(0)

        x = self.transformer['drop'](x)

        # Reset activations and attention patterns if tracking
        if self.config.track_activations and not self.training:
            self.activations = []
        if self.config.track_attention_patterns and not self.training:
            self.attention_patterns = []

        for block in self.transformer['h']:
            x = block(x)
            if self.config.track_activations and not self.training:
                self.activations.append(x.detach().cpu())
            if self.config.track_attention_patterns and not self.training:
                if hasattr(block.attn, 'attn_weights'):
                    self.attention_patterns.append(block.attn.attn_weights)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Start with all candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU)"""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate sequences of tokens from the model"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
