import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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

def para(in_dim, out_dim):
    w = nn.init.kaiming_uniform_(torch.empty(in_dim, out_dim))
    return nn.Parameter(w)

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

        self.to_qk = nn.Linear(dim, inner_dim * 2, bias = False)
        
        shared_Ws = []
        for i in range(2*h - 1):
            tmp = []
            for j in range(2*w - 1):
                W = para(dim, inner_dim)
                tmp.append(W)
            shared_Ws.append(tmp)
        # for cls token
        tmp = []
        for _ in range(1 + h*w):
            W = para(dim, inner_dim)
            tmp.append(W)
        Ws = [torch.stack(tmp, dim=0)]
        # for img patch
        for x in range(h):
            for y in range(w):
                cls_W = para(dim, inner_dim)
                tmp = [cls_W]
                for i in range(h):
                    for j in range(w):
                        dx = x - i + h - 1
                        dy = y - j + w - 1
                        tmp.append(shared_Ws[dx][dy])
                Ws.append(torch.stack(tmp, dim=0))
        self.to_v = torch.stack(Ws, dim=0)
                
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
       
        # attention 
        qk = self.to_qk(x).chunk(2, dim = -1)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)                                   # b h n n

        # value
        x = x.unsqueeze(2).unsqueeze(2)                             # b n 1 1   dim
        w = self.to_v.unsqueeze(0).to(x.device)                     # 1 n n dim inner_dim
        v = torch.matmul(x, w).squeeze(3)                           # b n n inner_dim
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
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
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

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.translution = Translution((image_height // patch_height, image_width // patch_width), dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):                 # b, c, H, W
        x = self.to_patch_embedding(img)    # b, h*w, dim
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)                 # b, 1 + h*w, dim

        x = self.translution(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        return F.log_softmax(x, dim=1)
