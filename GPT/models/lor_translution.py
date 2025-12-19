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

class LoR_QKVTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dim_relenc = 16, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)


        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_q1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_q2 = CausalSharedParameter2(self.seq_len, dim_relenc * heads, dim_relenc * heads)
        
        self.to_k1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_k2 = CausalSharedParameter(self.seq_len, dim_relenc * heads, dim_relenc * heads)

        self.to_v1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_v2 = CausalSharedParameter(self.seq_len, dim_relenc * heads, dim_relenc * heads)
        self.to_v3 = nn.Linear(dim_relenc * heads, inner_dim, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # query1, key1, value1
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # query2
        q2 = self.to_q1(x)
        q2 = q2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_q = self.to_q2().unsqueeze(0)                                 # 1 n n dim C
        q2 = torch.matmul(q2, w_q).squeeze(3)                           # b n n C
        q2 = rearrange(q2, 'b n m (h d) -> b h m n d', h = self.heads)  # b h n n d

        # key2
        k2 = self.to_k1(x)
        k2 = k2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_k = self.to_k2().unsqueeze(0)                                 # 1 n n dim C
        k2 = torch.matmul(k2, w_k).squeeze(3)                           # b n n C
        k2 = rearrange(k2, 'b n m (h d) -> b h n m d', h = self.heads)  # b h n n d

        # value2
        v2 = self.to_v1(x)
        v2 = v2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_v = self.to_v2().unsqueeze(0)                                 # 1 n n dim C
        v2 = torch.matmul(v2, w_v).squeeze(3)                           # b n n C
        v2 = rearrange(v2, 'b n m (h d) -> b h n m d', h = self.heads)  # b h n n d

        # attention
        dots1 = torch.matmul(q1, k1.transpose(-1, -2))
        dots2 = torch.sum(q2*k2, dim=4, keepdim=False)
        dots = (dots1 + dots2) / 2 * self.scale
        # causual
        dots = dots.masked_fill(self.causal[:,:,:self.seq_len,:self.seq_len] == 0, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # weighted sum 1
        out1 = torch.matmul(attn, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        # weighted sum 2
        out2 = attn.unsqueeze(-1) * v2                                  # b h n n d
        out2 = torch.sum(out2, dim=3, keepdim=False)                    # b h n d
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out2 = self.to_v3(out2)

        # output
        return self.to_out((out1 + out2)/2)

class LoR_QKTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dim_relenc = 16, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)


        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_q1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_q2 = CausalSharedParameter2(self.seq_len, dim_relenc * heads, dim_relenc * heads)
        
        self.to_k1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_k2 = CausalSharedParameter(self.seq_len, dim_relenc * heads, dim_relenc * heads)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # query1, key1, value1
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q1, k1, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # query2
        q2 = self.to_q1(x)
        q2 = q2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_q = self.to_q2().unsqueeze(0)                                 # 1 n n dim C
        q2 = torch.matmul(q2, w_q).squeeze(3)                           # b n n C
        q2 = rearrange(q2, 'b n m (h d) -> b h m n d', h = self.heads)  # b h n n d

        # key2
        k2 = self.to_k1(x)
        k2 = k2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_k = self.to_k2().unsqueeze(0)                                 # 1 n n dim C
        k2 = torch.matmul(k2, w_k).squeeze(3)                           # b n n C
        k2 = rearrange(k2, 'b n m (h d) -> b h n m d', h = self.heads)  # b h n n d

        # value2
        v2 = self.to_v1(x)
        v2 = v2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_v = self.to_v2().unsqueeze(0)                                 # 1 n n dim C
        v2 = torch.matmul(v2, w_v).squeeze(3)                           # b n n C
        v2 = rearrange(v2, 'b n m (h d) -> b h n m d', h = self.heads)  # b h n n d

        # attention
        dots1 = torch.matmul(q1, k1.transpose(-1, -2))
        dots2 = torch.sum(q2*k2, dim=4, keepdim=False)
        dots = (dots1 + dots2) / 2 * self.scale
        # causual
        dots = dots.masked_fill(self.causal[:,:,:self.seq_len,:self.seq_len] == 0, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # weighted sum 
        out = torch.matmul(attn, v) 
        out = rearrange(out, 'b h n d -> b n (h d)')

        # output
        return self.to_out(out)
    

class LoR_QVTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dim_relenc = 16, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)


        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_q1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_q2 = CausalSharedParameter2(self.seq_len, dim_relenc * heads, dim_relenc * heads)
        
        self.to_k1 = nn.Linear(dim, dim_relenc * heads, bias = False)

        self.to_v1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_v2 = CausalSharedParameter(self.seq_len, dim_relenc * heads, dim_relenc * heads)
        self.to_v3 = nn.Linear(dim_relenc * heads, inner_dim, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # query1, key1, value1
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # query2
        q2 = self.to_q1(x)
        q2 = q2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_q = self.to_q2().unsqueeze(0)                                 # 1 n n dim C
        q2 = torch.matmul(q2, w_q).squeeze(3)                           # b n n C
        q2 = rearrange(q2, 'b n m (h d) -> b h m n d', h = self.heads)  # b h n n d

        # key2
        k2 = self.to_k1(x)                                              # b n dim
        k2 = rearrange(k2, 'b n (h d) -> b h n d', h = self.heads)      # b h n d
        k2 = k2.unsqueeze(2)                                            # b h 1 n d

        # value2
        v2 = self.to_v1(x)
        v2 = v2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_v = self.to_v2().unsqueeze(0)                                 # 1 n n dim C
        v2 = torch.matmul(v2, w_v).squeeze(3)                           # b n n C
        v2 = rearrange(v2, 'b n m (h d) -> b h n m d', h = self.heads)  # b h n n d

        # attention
        dots1 = torch.matmul(q1, k1.transpose(-1, -2))
        dots2 = torch.sum(q2*k2, dim=4, keepdim=False)
        dots = (dots1 + dots2) / 2 * self.scale
        # causual
        dots = dots.masked_fill(self.causal[:,:,:self.seq_len,:self.seq_len] == 0, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # weighted sum 1
        out1 = torch.matmul(attn, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        # weighted sum 2
        out2 = attn.unsqueeze(-1) * v2                                  # b h n n d
        out2 = torch.sum(out2, dim=3, keepdim=False)                    # b h n d
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out2 = self.to_v3(out2)

        # output
        return self.to_out((out1 + out2)/2)

class LoR_KVTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dim_relenc = 16, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)


        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_q1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        
        self.to_k1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_k2 = CausalSharedParameter(self.seq_len, dim_relenc * heads, dim_relenc * heads)

        self.to_v1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_v2 = CausalSharedParameter(self.seq_len, dim_relenc * heads, dim_relenc * heads)
        self.to_v3 = nn.Linear(dim_relenc * heads, inner_dim, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # query1, key1, value1
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # query2
        q2 = self.to_q1(x)                                              # b n dim
        q2 = rearrange(q2, 'b n (h d) -> b h n d', h = self.heads)      # b h n d
        q2 = q2.unsqueeze(3)                                            # b h n 1 d

        # key2
        k2 = self.to_k1(x)
        k2 = k2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_k = self.to_k2().unsqueeze(0)                                 # 1 n n dim C
        k2 = torch.matmul(k2, w_k).squeeze(3)                           # b n n C
        k2 = rearrange(k2, 'b n m (h d) -> b h n m d', h = self.heads)  # b h n n d

        # value2
        v2 = self.to_v1(x)
        v2 = v2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_v = self.to_v2().unsqueeze(0)                                 # 1 n n dim C
        v2 = torch.matmul(v2, w_v).squeeze(3)                           # b n n C
        v2 = rearrange(v2, 'b n m (h d) -> b h n m d', h = self.heads)  # b h n n d

        # attention
        dots1 = torch.matmul(q1, k1.transpose(-1, -2))
        dots2 = torch.sum(q2*k2, dim=4, keepdim=False)
        dots = (dots1 + dots2) / 2 * self.scale
        # causual
        dots = dots.masked_fill(self.causal[:,:,:self.seq_len,:self.seq_len] == 0, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # weighted sum 1
        out1 = torch.matmul(attn, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        # weighted sum 2
        out2 = attn.unsqueeze(-1) * v2                                  # b h n n d
        out2 = torch.sum(out2, dim=3, keepdim=False)                    # b h n d
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out2 = self.to_v3(out2)

        # output
        return self.to_out((out1 + out2)/2)

class LoR_QTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dim_relenc = 16, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_q1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_q2 = CausalSharedParameter2(self.seq_len, dim_relenc * heads, dim_relenc * heads)
        
        self.to_k1 = nn.Linear(dim, dim_relenc * heads, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # query1, key1, value1
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q1, k1, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # query2
        q2 = self.to_q1(x)
        q2 = q2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_q = self.to_q2().unsqueeze(0)                                 # 1 n n dim C
        q2 = torch.matmul(q2, w_q).squeeze(3)                           # b n n C
        q2 = rearrange(q2, 'b n m (h d) -> b h m n d', h = self.heads)  # b h n n d

        # key2
        k2 = self.to_k1(x)                                              # b n dim
        k2 = rearrange(k2, 'b n (h d) -> b h n d', h = self.heads)      # b h n d
        k2 = k2.unsqueeze(2)                                            # b h 1 n d

        # attention
        dots1 = torch.matmul(q1, k1.transpose(-1, -2))
        dots2 = torch.sum(q2*k2, dim=4, keepdim=False)
        dots = (dots1 + dots2) / 2 * self.scale
        # causual
        dots = dots.masked_fill(self.causal[:,:,:self.seq_len,:self.seq_len] == 0, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # weighted sum
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # output
        return self.to_out(out)

class LoR_KTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dim_relenc = 16, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_q1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        
        self.to_k1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_k2 = CausalSharedParameter(self.seq_len, dim_relenc * heads, dim_relenc * heads)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # query1, key1, value1
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q1, k1, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # query2
        q2 = self.to_q1(x)                                              # b n dim
        q2 = rearrange(q2, 'b n (h d) -> b h n d', h = self.heads)      # b h n d
        q2 = q2.unsqueeze(3)                                            # b h n 1 d

        # key2
        k2 = self.to_k1(x)
        k2 = k2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_k = self.to_k2().unsqueeze(0)                                 # 1 n n dim C
        k2 = torch.matmul(k2, w_k).squeeze(3)                           # b n n C
        k2 = rearrange(k2, 'b n m (h d) -> b h n m d', h = self.heads)  # b h n n d

        # attention
        dots1 = torch.matmul(q1, k1.transpose(-1, -2))
        dots2 = torch.sum(q2*k2, dim=4, keepdim=False)
        dots = (dots1 + dots2) / 2 * self.scale
        # causual
        dots = dots.masked_fill(self.causal[:,:,:self.seq_len,:self.seq_len] == 0, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # weighted sum
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # output
        return self.to_out(out)

class LoR_VTranslution(nn.Module):
    def __init__(self, seq_len, dim, heads = 8, dim_head = 64, dim_relenc = 16, dropout = 0.):
        super().__init__()
        self.seq_len = seq_len
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_v1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        #self.to_v2 = SharedParameter(self.seq_len, dim_relenc * heads, dim_relenc * heads)
        self.to_v2 = CausalSharedParameter(self.seq_len, dim_relenc * heads, dim_relenc * heads)
        self.to_v3 = nn.Linear(dim_relenc * heads, inner_dim, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # query1, key1, value1 
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # value2
        v2 = self.to_v1(x)
        v2 = v2.unsqueeze(1).unsqueeze(3)                                 # b 1 n 1   dim
        w_v = self.to_v2().unsqueeze(0)                                   # 1 n n dim C
        v2 = torch.matmul(v2, w_v).squeeze(3)                             # b n n C
        v2 = rearrange(v2, 'b n m (h d) -> b h n m d', h = self.heads)    # b h n n d

        # attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # causual
        dots = dots.masked_fill(self.causal[:,:,:self.seq_len,:self.seq_len] == 0, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # weighted sum 1
        out1 = torch.matmul(attn, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        # weighted sum 2
        out2 = attn.unsqueeze(-1) * v2                                  # b h n n d
        out2 = torch.sum(out2, dim=3, keepdim=False)                    # b h n d
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out2 = self.to_v3(out2)

        # output
        return self.to_out((out1 + out2)/2)

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
    
class LoR_TNN(nn.Module):
    def __init__(self, tnn_type, seq_len, dim, depth, heads, dim_head, mlp_dim, dim_relenc = 16, dropout = 0., tln_num = None):
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
                    tnn_type(seq_len, dim, heads = heads, dim_head = dim_head, dim_relenc = dim_relenc, dropout = dropout),
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
                    tnn_type(seq_len, dim, heads = heads, dim_head = dim_head, dim_relenc = dim_relenc, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for translution, ff in self.layers:
            x = translution(x) + x
            x = ff(x) + x

        return self.norm(x)

class GPT(nn.Module):
    def __init__(self, *, tnn_type, seq_len, vocab_size, dim, depth, heads, mlp_dim, dim_head = 64, dim_relenc = 16, dropout = 0., emb_dropout = 0., tln_num = None, pos_embedding = False):
        super().__init__()

        if pos_embedding:
            self.wpe = nn.Embedding(seq_len, dim)
        else:
            self.wpe = False
        self.wte = nn.Embedding(vocab_size, dim)

        self.dropout = nn.Dropout(emb_dropout)
        
        self.translution = LoR_TNN(tnn_type, seq_len, dim, depth, heads, dim_head, mlp_dim, dim_relenc, dropout, tln_num)

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


def lor_qkv_tiny(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QKVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_qkv_mini(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QKVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_qkv_small(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QKVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def lor_qkv_base(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QKVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_qkv_large(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QKVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_qk_tiny(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QKTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_qk_mini(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QKTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_qk_small(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QKTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def lor_qk_base(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QKTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_qk_large(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QKTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_qv_tiny(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_qv_mini(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_qv_small(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def lor_qv_base(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_qv_large(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_kv_tiny(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_KVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_kv_mini(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_KVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_kv_small(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_KVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def lor_kv_base(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_KVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_kv_large(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_KVTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_q_tiny(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_q_mini(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_q_small(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def lor_q_base(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_q_large(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_QTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_k_tiny(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_KTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_k_mini(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_KTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_k_small(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_KTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def lor_k_base(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_KTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_k_large(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_KTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_v_tiny(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_VTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_v_mini(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_VTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 192,
               depth = 12,
               heads = 3,
               mlp_dim = 768,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_v_small(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_VTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 384,
               depth = 12,
               heads = 6,
               mlp_dim = 1536,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
 
def lor_v_base(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_VTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 768,
               depth = 12,
               heads = 12,
               mlp_dim = 3072,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 

def lor_v_large(seq_len, vocab_size, dim_relenc=16, tln_num = None):
    return GPT(tnn_type = LoR_VTranslution,
               seq_len = seq_len,
               vocab_size = vocab_size,
               dim = 1024,
               depth = 24,
               heads = 16,
               mlp_dim = 4096,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0., 
               tln_num = tln_num) 
