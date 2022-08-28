from turtle import forward
from requests import patch
import torch
import torch.nn as nn


def get_pos_emb(patches,device,temperature=1e4,dtype=torch.float):
    _,h,w,dim = patches.shape
    y,x = torch.arange(h,device=device),torch.arange(w,device=device)
    y,x = torch.meshgrid(y,x)
    
    omega = torch.arange(dim//4,device=device)/(dim/4-1)
    omega = 1. / temperature ** omega
    x = x.flatten()[:,None] * omega[None,:]
    y = y.flatten()[:,None] * omega[None,:]

    pe = torch.cat((x.cos(),x.sin(),y.cos(),y.sin()),dim=1)
    return pe.to(dtype)

class PreNorm(nn.Module):
    def __init__(self,dim,fn) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self,x):
        x = self.norm(x)
        x = self.fn(x)
        return x


class Mlp(nn.Module):
    def __init__(self,dim,mlp_dim,drop_prob) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,mlp_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(mlp_dim,dim),
            nn.Dropout(drop_prob)
        )

    def forward(self,input):
        return self.net(input)

class Attention(nn.Module):
    def __init__(self,dim,num_heads,drop_prob) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        assert(type(self.head_dim) == int)

        self.layernorm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim,3*dim)
        self.attn_drop = nn.Dropout(drop_prob)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(drop_prob)

    def forward(self,input):
        batch_size,num_seq = input.shape[0],input.shape[1]
        assert(self.dim == input.shape[2])
        x = input
        qkv = self.qkv(x).reshape(batch_size,num_seq,3,self.num_heads,self.head_dim)
        q,k,v = qkv.permute(2,0,3,1,4)
        attn = q @ k.transpose(-1,-2) * self.scale
        attn = nn.functional.softmax(attn,dim = -1)
        attn = self.attn_drop(attn)
        output = (attn @ v).reshape(batch_size,num_seq,self.dim)
        output = self.proj_drop(self.proj(output))
        return output

class Transformer(nn.Module):
    def  __init__(self,dim,mlp_dim,depth,num_heads,drop_prob) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim,Attention(dim,num_heads,drop_prob)),
                    PreNorm(dim,Mlp(dim,mlp_dim,drop_prob))
                ])
            )
    def forward(self,x):
        for attn,ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x

class Vit(nn.Module):
    def __init__(self,img_shape,patch_shape,in_channels,dim,mlp_dim,depth,num_heads,drop_prob) -> None:
        super().__init__()
        self.img_shape,self.patch_shape,self.in_channels = img_shape,patch_shape,in_channels
        self.dim,self.mlp_dim,self.depth = dim,mlp_dim,depth
        self.num_heads,self.drop_prob = num_heads,drop_prob


        self.h,self.w = img_shape[0] // patch_shape[0],img_shape[1]//patch_shape[1]
        self.pos_embed = nn.Parameter(torch.randn(1,self.h * self.w,dim))
        self.cls_token = nn.Parameter(torch.rand(1,1,dim))

        self.img_embed = nn.Linear(patch_shape[0] * patch_shape[1] * in_channels,dim)

        self.transformer = Transformer(dim,mlp_dim,depth,num_heads,drop_prob)

        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,1)
        )

    def forward(self,input):
        batch_size = input.shape[0]
        x = input.reshape(batch_size,self.in_channels,self.h,self.patch_shape[0],self.w,self.patch_shape[1])
        x = x.permute(0,2,4,1,3,5).reshape(batch_size,self.h,self.w,self.in_channels * self.patch_shape[0] * self.patch_shape[1])
        
        x = self.img_embed(x)
        x = x.reshape(batch_size,self.h * self.w,self.dim) + self.pos_embed

        cls_tokens = self.cls_token.repeat(batch_size,1,1)
        x = torch.cat([cls_tokens,x],dim=1)

        x = self.transformer(x)
        x = x.view(batch_size,self.h * self.w + 1,self.dim)
        x = x[:,0]
        x = self.linear_head(x)
        return x
