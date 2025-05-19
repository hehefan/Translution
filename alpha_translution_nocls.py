import math
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class SharedParameter(nn.Module):
    def __init__(self, h, w, dim):
        super().__init__()
        num_tokens = (2*h-1) * (2*w-1)
        self.unique_params = nn.Parameter(torch.empty(num_tokens, dim))
        torch.nn.init.kaiming_uniform_(self.unique_params, a=math.sqrt(5))
        
        index_map = []
        for x in range(h):
            for y in range(w):
                tmp = []
                for i in range(h):
                    for j in range(w):
                        dx = x - i + h - 1
                        dy = y - j + w - 1
                        tmp.append(dx*(2*w-1) + dy)
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

class Attention(nn.Module):
    def __init__(self, hw_size, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        h, w = hw_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.r_pos = SharedParameter(h, w, inner_dim)
                
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
       
        # attention 
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)                                   # b h n n

        # value v: [b n inner_dim]
        v = v.unsqueeze(1)                                          # b 1 n inner_dim
        r_pos = self.r_pos().unsqueeze(0)                           # 1 n n inner_dim
        v = v + r_pos                                               # b n n inner_dim
        v = rearrange(v, 'b n m (h d) -> b h n m d', h = self.heads)# b h n n d
        
        # sum
        out = attn.unsqueeze(-1) * v                                # b h n n d
        out = torch.sum(out, dim=3, keepdim=False)                  # b h n d
 
        # output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Translution(nn.Module):
    def __init__(self, hw_size, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(hw_size, dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.translution = Translution((image_height // patch_height, image_width // patch_width), dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):                 # b, c, H, W
        x = self.to_patch_embedding(img)    # b, h*w, dim
        b, n, _ = x.shape

        x = self.dropout(x)                 # b, 1 + h*w, dim
        x = self.translution(x)

        x = x.mean(dim = 1) 

        x = self.to_latent(x)
        return self.mlp_head(x)
