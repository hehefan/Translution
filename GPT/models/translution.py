import math
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class SharedParameter(nn.Module):
    def __init__(self, length, in_dim, out_dim):
        super().__init__()
        num_weights = 2*length - 1 
        self.unique_params = nn.Parameter(torch.empty(num_weights, in_dim, out_dim))
        torch.nn.init.kaiming_uniform_(self.unique_params, a=math.sqrt(5))

        index_map = []
        for i in range(length):
            tmp = []
            for j in range(length):
                d = i - j + length - 1 
                tmp.append(d)
            index_map.append(tmp)
        self.index_map = torch.tensor(index_map)

    def forward(self):
        weight = self.unique_params[self.index_map]
        return weight

class CausalSharedParameter1(nn.Module):
    def __init__(self, length, in_dim, out_dim):
        super().__init__()
        # causual
        self.unique_params = nn.Parameter(torch.empty(length, in_dim, out_dim))
        torch.nn.init.kaiming_uniform_(self.unique_params, a=math.sqrt(5))

        index_map = []
        for i in range(length):
            tmp = []
            for j in range(length):
                # causual
                d = max(0, i - j) 
                tmp.append(d)
            index_map.append(tmp)
        self.index_map = torch.tensor(index_map)

    def forward(self):
        weight = self.unique_params[self.index_map]
        return weight

class CausalSharedParameter2(nn.Module):
    def __init__(self, length, in_dim, out_dim):
        super().__init__()
        # causual
        self.unique_params = nn.Parameter(torch.empty(length, in_dim, out_dim))
        torch.nn.init.kaiming_uniform_(self.unique_params, a=math.sqrt(5))

        index_map = []
        for i in range(length):
            tmp = []
            for j in range(length):
                # causual
                d = min(0, i - j) + length - 1
                tmp.append(d)
            index_map.append(tmp)
        self.index_map = torch.tensor(index_map)

    def forward(self):
        weight = self.unique_params[self.index_map]
        return weight

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

class Translution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_q = CausalSharedParameter1(self.seq_len, dim, inner_dim)
        self.to_k = CausalSharedParameter2(self.seq_len, dim, inner_dim)
        self.to_v = CausalSharedParameter1(self.seq_len, dim, inner_dim)
    
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = x.unsqueeze(1).unsqueeze(3)                             # b 1 n 1   dim

        # query
        w_q = self.to_q().unsqueeze(0)                              # 1 n n dim inner_dim
        q = torch.matmul(x, w_q).squeeze(3)                         # b n n inner_dim
        q = rearrange(q, 'b n m (h d) -> b h n m d', h = self.heads)# b h n n d
        
        # key
        w_k = self.to_k().unsqueeze(0)                              # 1 n n dim inner_dim
        k = torch.matmul(x, w_k).squeeze(3)                         # b n n inner_dim
        k = rearrange(k, 'b n m (h d) -> b h m n d', h = self.heads)# b h n n d
        
        # value
        w_v = self.to_v().unsqueeze(0)                              # 1 n n dim inner_dim
        v = torch.matmul(x, w_v).squeeze(3)                         # b n n inner_dim
        v = rearrange(v, 'b n m (h d) -> b h n m d', h = self.heads)# b h n n d
        
        # attention
        dots = torch.sum(q*k, dim=4, keepdim=False) * self.scale
        # causual
        dots = dots.masked_fill(self.causal[:,:,:self.seq_len,:self.seq_len] == 0, float('-inf'))
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # sum
        out = attn.unsqueeze(-1) * v                                # b h n n d
        out = torch.sum(out, dim=3, keepdim=False)                  # b h n d

        # output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TNN(nn.Module):
    def __init__(self, seq_len, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Translution(seq_len, dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for translution, ff in self.layers:
            x = translution(x) + x
            x = ff(x) + x

        return self.norm(x)

class GPT(nn.Module):
    def __init__(self, *, seq_len, vocab_size, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., pos_embedding = False):
        super().__init__()

        if pos_embedding:
            self.wpe = nn.Embedding(seq_len, dim)
        else:
            self.wpe = False
        self.wte = nn.Embedding(vocab_size, dim)

        self.dropout = nn.Dropout(emb_dropout)
        
        self.translution = TNN(seq_len, dim, depth, heads, dim_head, mlp_dim, dropout)

        self.norm = nn.LayerNorm(dim)
        
        self.output_head = nn.Linear(dim, vocab_size, bias=False)


    def forward(self, x):
        b, n = x.shape
        
        x = self.wte(x)
        if self.wpe:
            pos = torch.arange(0, n, dtype=torch.long, device=device)
            x += self.wpe(pos)
        
        x = self.dropout(x)

        x = self.translution(x)

        x = self.norm(x)

        return self.output_head(x)

def lution_gpt_tiny(seq_len, vocab_size):
    return GPT(seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0.) 

def lution_gpt_mini(seq_len, vocab_size):
    return GPT(seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0.) 

def lution_gpt_small(seq_len, vocab_size):
    return GPT(seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0.)
 
def lution_gpt_base(seq_len, vocab_size):
    return GPT(seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0.) 

def lution_gpt_large(seq_len, vocab_size):
    return GPT(seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0.) 
