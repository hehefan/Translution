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

class CausalSharedParameter(nn.Module):
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
    
class QKVTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_q = CausalSharedParameter2(self.seq_len, dim, inner_dim)
        self.to_k = CausalSharedParameter(self.seq_len, dim, inner_dim)
        self.to_v = CausalSharedParameter(self.seq_len, dim, inner_dim)
    
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = x.unsqueeze(1).unsqueeze(3)                             # b 1 n 1 dim

        # query
        w_q = self.to_q().unsqueeze(0)                              # 1 n n dim inner_dim
        q = torch.matmul(x, w_q).squeeze(3)                         # b n n inner_dim
        q = rearrange(q, 'b n m (h d) -> b h m n d', h = self.heads)# b h n n d

        # key
        w_k = self.to_k().unsqueeze(0)                              # 1 n n dim inner_dim
        k = torch.matmul(x, w_k).squeeze(3)                         # b n n inner_dim
        k = rearrange(k, 'b n m (h d) -> b h n m d', h = self.heads)# b h n n d
        
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

        # weighted sum
        out = attn.unsqueeze(-1) * v                                # b h n n d
        out = torch.sum(out, dim=3, keepdim=False)                  # b h n d

        # output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class QKTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_q = CausalSharedParameter2(self.seq_len, dim, inner_dim)
        self.to_k = CausalSharedParameter(self.seq_len, dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
    
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # value
        v = self.to_v(x)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

        x = x.unsqueeze(1).unsqueeze(3)                             # b 1 n 1 dim

        # query
        w_q = self.to_q().unsqueeze(0)                              # 1 n n dim inner_dim
        q = torch.matmul(x, w_q).squeeze(3)                         # b n n inner_dim
        q = rearrange(q, 'b n m (h d) -> b h m n d', h = self.heads)# b h n n d

        # key
        w_k = self.to_k().unsqueeze(0)                              # 1 n n dim inner_dim
        k = torch.matmul(x, w_k).squeeze(3)                         # b n n inner_dim
        k = rearrange(k, 'b n m (h d) -> b h n m d', h = self.heads)# b h n n d
        
        # attention
        dots = torch.sum(q*k, dim=4, keepdim=False) * self.scale
        # causual
        dots = dots.masked_fill(self.causal[:,:,:self.seq_len,:self.seq_len] == 0, float('-inf'))
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # weighted sum
        out = torch.matmul(attn, v)

        # output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class QVTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_q = CausalSharedParameter2(self.seq_len, dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = CausalSharedParameter(self.seq_len, dim, inner_dim)
    
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # key
        k = self.to_k(x)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        k = k.unsqueeze(2)                                          # b h 1 n d

        x = x.unsqueeze(1).unsqueeze(3)                             # b 1 n 1 dim

        # query
        w_q = self.to_q().unsqueeze(0)                              # 1 n n dim inner_dim
        q = torch.matmul(x, w_q).squeeze(3)                         # b n n inner_dim
        q = rearrange(q, 'b n m (h d) -> b h m n d', h = self.heads)# b h n n d
        
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

        # weighted sum
        out = attn.unsqueeze(-1) * v                                # b h n n d
        out = torch.sum(out, dim=3, keepdim=False)                  # b h n d

        # output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class KVTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = CausalSharedParameter(self.seq_len, dim, inner_dim)
        self.to_v = CausalSharedParameter(self.seq_len, dim, inner_dim)
    
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # query
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q.unsqueeze(3)                                          # b h n 1 d

        x = x.unsqueeze(1).unsqueeze(3)                             # b 1 n 1 dim
        
        # key
        w_k = self.to_k().unsqueeze(0)                              # 1 n n dim inner_dim
        k = torch.matmul(x, w_k).squeeze(3)                         # b n n inner_dim
        k = rearrange(k, 'b n m (h d) -> b h n m d', h = self.heads)# b h n n d
        
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

        # weighted sum
        out = attn.unsqueeze(-1) * v                                # b h n n d
        out = torch.sum(out, dim=3, keepdim=False)                  # b h n d

        # output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class QTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_q = CausalSharedParameter2(self.seq_len, dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
    
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # key and value
        kv = self.to_kv(x).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        k = k.unsqueeze(2)                                          # b h 1 n d

        x = x.unsqueeze(1).unsqueeze(3)                             # b 1 n 1 dim

        # query
        w_q = self.to_q().unsqueeze(0)                              # 1 n n dim inner_dim
        q = torch.matmul(x, w_q).squeeze(3)                         # b n n inner_dim
        q = rearrange(q, 'b n m (h d) -> b h m n d', h = self.heads)# b h n n d
        
        # attention
        dots = torch.sum(q*k, dim=4, keepdim=False) * self.scale
        # causual
        dots = dots.masked_fill(self.causal[:,:,:self.seq_len,:self.seq_len] == 0, float('-inf'))
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # weighted sum
        out = torch.matmul(attn, v)

        # output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class KTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_k = CausalSharedParameter(self.seq_len, dim, inner_dim)
    
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # query and value
        qv = self.to_qv(x).chunk(2, dim = -1)
        q, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qv)
        q = q.unsqueeze(3)                                          # b h n 1 d     

        x = x.unsqueeze(1).unsqueeze(3)                             # b 1 n 1 dim
        
        # key
        w_k = self.to_k().unsqueeze(0)                              # 1 n n dim inner_dim
        k = torch.matmul(x, w_k).squeeze(3)                         # b n n inner_dim
        k = rearrange(k, 'b n m (h d) -> b h n m d', h = self.heads)# b h n n d
       
        # attention
        dots = torch.sum(q*k, dim=4, keepdim=False) * self.scale
        # causual
        dots = dots.masked_fill(self.causal[:,:,:self.seq_len,:self.seq_len] == 0, float('-inf'))
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # weighted sum
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # output
        return self.to_out(out)

class VTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qk = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_v = CausalSharedParameter(self.seq_len, dim, inner_dim)
    
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # query and key
        qk = self.to_qk(x).chunk(2, dim = -1)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qk)

        x = x.unsqueeze(1).unsqueeze(3)                             # b 1 n 1   dim
        
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

        # weighted sum
        out = attn.unsqueeze(-1) * v                                # b h n n d
        out = torch.sum(out, dim=3, keepdim=False)                  # b h n d

        # output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TNN(nn.Module):
    def __init__(self, tnn_type, seq_len, dim, depth, heads, dim_head, mlp_dim, dropout = 0., tln_num = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        if tln_num is not None: # hybrid
            assert tln_num <= depth, (
                f"Number of Translution layers must be smaller than the network depth: "
                f"tln_num={tln_num}, depth={depth}"
            )
            for _ in range(tln_num):
                self.layers.append(nn.ModuleList([
                    tnn_type(seq_len, dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
            for _ in range(depth - tln_num):
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    tnn_type(seq_len, dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for translution, ff in self.layers:
            x = translution(x) + x
            x = ff(x) + x

        return self.norm(x)

class GPT(nn.Module):
    def __init__(self, *, tnn_type, seq_len, vocab_size, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., tln_num = None, pos_embedding = False):
        super().__init__()

        if pos_embedding:
            self.wpe = nn.Embedding(seq_len, dim)
        else:
            self.wpe = False
        self.wte = nn.Embedding(vocab_size, dim)

        self.dropout = nn.Dropout(emb_dropout)
        
        self.translution = TNN(tnn_type, seq_len, dim, depth, heads, dim_head, mlp_dim, dropout, tln_num)

        self.norm = nn.LayerNorm(dim)
        
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        b, n = x.shape
        
        x = self.wte(x)
        if self.wpe:
            pos = torch.arange(0, n, dtype=torch.long, device=x.device)
            x += self.wpe(pos)
        
        x = self.dropout(x)

        x = self.translution(x)

        x = self.norm(x)

        return self.output_head(x)

def qkv_tiny(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QKVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def qkv_mini(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QKVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def qkv_small(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QKVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def qkv_base(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QKVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def qkv_large(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QKVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def qk_tiny(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QKTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def qk_mini(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QKTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def qk_small(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QKTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def qk_base(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QKTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def qk_large(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QKTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def qv_tiny(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def qv_mini(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def qv_small(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def qv_base(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def qv_large(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def kv_tiny(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = KVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def kv_mini(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = KVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def kv_small(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = KVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def kv_base(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = KVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def kv_large(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = KVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def q_tiny(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def q_mini(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def q_small(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def q_base(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def q_large(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = QTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def k_tiny(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = KTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def k_mini(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = KTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def k_small(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = KTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def k_base(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = KTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def k_large(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = KTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def v_tiny(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = VTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def v_mini(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = VTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def v_small(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = VTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def v_base(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = VTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def v_large(seq_len, vocab_size, tln_num = None):
    return GPT(tnn_type = VTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
