import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        # dim: the head dimension (not the total model dimension)
        
        inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        # x shape: [batch, heads, seq_len, head_dim]
        if seq_len is None:
            seq_len = x.shape[2]
            
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    """
    Apply rotary embeddings to input x.
    x:   [batch, heads, seq_len, head_dim]
    cos: [1,     1,     seq_len, head_dim]
    cos: [1,     1,     seq_len, head_dim]
    """
    return (x * cos) + (rotate_half(x) * sin)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.rotary_emb = RotaryPositionalEmbedding(dim_head, seq_len)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        _, n, _ = x.size() # Batch, N (SeqLen), Channels (EmbedDim)

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        cos, sin = self.rotary_emb(v, seq_len=n)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots.masked_fill(self.causal[:,:,:self.seq_len,:self.seq_len] == 0, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, seq_len, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(seq_len, dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class GPT(nn.Module):
    def __init__(self, *, seq_len, vocab_size, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.wte = nn.Embedding(vocab_size, dim)

        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(seq_len, dim, depth, heads, dim_head, mlp_dim, dropout)

        self.norm = nn.LayerNorm(dim)
        
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        device = x.device
        b, n = x.shape
        pos = torch.arange(0, n, dtype=torch.long, device=device)
        
        tok_emb = self.wte(x)
        
        x = self.dropout(tok_emb)

        x = self.transformer(x)

        x = self.norm(x)

        return self.output_head(x)

def rope_tiny(seq_len, vocab_size):
    return GPT(seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0.) 

def rope_mini(seq_len, vocab_size):
    return GPT(seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0.)

def rope_small(seq_len, vocab_size):
    return GPT(seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0.)
 
def rope_base(seq_len, vocab_size):
    return GPT(seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0.) 

def rope_large(seq_len, vocab_size):
    return GPT(seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0.) 
