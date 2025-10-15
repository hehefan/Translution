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
    def __init__(self, height, width, in_dim, out_dim):
        super().__init__()
        num_img_tokens = (2*height-1) * (2*width-1)
        self.unique_params = nn.Parameter(torch.empty(num_img_tokens, in_dim, out_dim))
        torch.nn.init.kaiming_uniform_(self.unique_params, a=math.sqrt(5))

        index_map = []
        for x in range(height):
            for y in range(width):
                tmp = []
                for i in range(height):
                    for j in range(width):
                        dx = x - i + height - 1
                        dy = y - j + width - 1
                        tmp.append(dx*(2*width-1) + dy)
                index_map.append(tmp)
        self.index_map = torch.tensor(index_map)

    def forward(self):
        weight = self.unique_params[self.index_map]
        return weight
        
# relative cls token 
class SharedParameterRelCls(nn.Module):
    def __init__(self, height, width, in_dim, out_dim):
        super().__init__()
        num_img_tokens = (2*height-1) * (2*width-1)
        idx_cls = num_img_tokens
        idx_cls_in = num_img_tokens + 1 
        idx_cls_out = num_img_tokens + 2 
        self.unique_params = nn.Parameter(torch.empty(num_img_tokens + 3, in_dim, out_dim))
        torch.nn.init.kaiming_uniform_(self.unique_params, a=math.sqrt(5))

        index_map = [[idx_cls] + [idx_cls_in for _ in range(height*width)]]
        for x in range(height):
            for y in range(width):
                tmp = [idx_cls_out]
                for i in range(height):
                    for j in range(width):
                        dx = x - i + height - 1 
                        dy = y - j + width - 1 
                        tmp.append(dx*(2*width-1) + dy) 
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
    def __init__(self, hw_size, dim, heads = 8, dim_head = 64, dim_relenc = 16, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        height, width = hw_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_q1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_q2 = SharedParameterRelCls(height, width, dim_relenc * heads, dim_relenc * heads)
        
        self.to_k1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_k2 = SharedParameterRelCls(height, width, dim_relenc * heads, dim_relenc * heads)

        self.to_v1 = nn.Linear(dim, dim_relenc * heads, bias = False)
        self.to_v2 = SharedParameterRelCls(height, width, dim_relenc * heads, dim_relenc * heads)
        self.to_v3 = nn.Linear(dim_relenc * heads, inner_dim, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

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
        q2 = rearrange(q2, 'b n m (h d) -> b h n m d', h = self.heads)  # b h n n d

        # key2
        k2 = self.to_k1(x)
        k2 = k2.unsqueeze(1).unsqueeze(3)                               # b 1 n 1   dim
        w_k = self.to_k2().unsqueeze(0)                                 # 1 n n dim C
        k2 = torch.matmul(k2, w_k).squeeze(3)                           # b n n C
        k2 = rearrange(k2, 'b n m (h d) -> b h m n d', h = self.heads)  # b h n n d

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

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # output1
        out1 = torch.matmul(attn, v1) 
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        # output2
        out2 = attn.unsqueeze(-1) * v2                                  # b h n n d
        out2 = torch.sum(out2, dim=3, keepdim=False)                    # b h n d
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out2 = self.to_v3(out2)

        # output
        return self.to_out((out1 + out2)/2)

class TNN(nn.Module):
    def __init__(self, hw_size, dim, depth, heads, dim_head, mlp_dim, dim_relenc = 16, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Translution(hw_size, dim, heads = heads, dim_head = dim_head, dim_relenc = dim_relenc, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for translution, ff in self.layers:
            x = translution(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dim_relenc = 16, dropout = 0., emb_dropout = 0., pos_embedding = False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        if pos_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        else:
            self.pos_embedding = False
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.translution = TNN((image_height // patch_height, image_width // patch_width), dim, depth, heads, dim_head, mlp_dim, dim_relenc, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):                 # b, c, H, W
        x = self.to_patch_embedding(img)    # b, h*w, dim
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embedding:
            x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)                 # b, 1 + h*w, dim

        x = self.translution(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

def alution_vit_tiny(image_size = 224, patch_size = 16, dim_relenc = 16, num_classes = 1000):
    return ViT(image_size = image_size,
               patch_size = patch_size,
               num_classes = num_classes,
               dim = 192,
               depth = 6,
               heads = 3,
               mlp_dim = 768,
               channels = 3,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0.) 

def alution_vit_mini(image_size = 224, patch_size = 16, dim_relenc = 16, num_classes = 1000):
    return ViT(image_size = image_size,
               patch_size = patch_size,
               num_classes = num_classes,
               dim = 192,
               depth = 12, 
               heads = 3,
               mlp_dim = 768,
               channels = 3,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0.) 

def alution_vit_small(image_size = 224, patch_size = 16, dim_relenc = 16, num_classes = 1000):
    return ViT(image_size = image_size,
               patch_size = patch_size,
               num_classes = num_classes,
               dim = 384,
               depth = 12, 
               heads = 6,
               mlp_dim = 1536,
               channels = 3,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0.) 

def alution_vit_base(image_size = 224, patch_size = 16, dim_relenc = 16, num_classes = 1000):
    return ViT(image_size = image_size,
               patch_size = patch_size,
               num_classes = num_classes,
               dim = 768,
               depth = 12, 
               heads = 12, 
               mlp_dim = 3072,
               channels = 3,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0.) 

def alution_vit_large(image_size = 224, patch_size = 16, dim_relenc = 16, num_classes = 1000):
    return ViT(image_size = image_size,
               patch_size = patch_size,
               num_classes = num_classes,
               dim = 1024,
               depth = 24, 
               heads = 16, 
               mlp_dim = 4096,
               channels = 3,
               dim_head = 64, 
               dim_relenc = dim_relenc, 
               dropout = 0., 
               emb_dropout = 0.) 
