import torch
import torch.nn as nn

from einops import rearrange

def get_1d_pos_emb(size, dim, device="cuda", temperature=1e4):
    pos = torch.arange(0, size, device=device,dtype=torch.float)

    omega = torch.arange(dim//2, device=device)/(dim/2-1)
    omega = 1. / temperature ** omega
    pos_embed = pos[:, None] * omega[None, :]

    pos_embed = torch.cat((pos_embed.cos(), pos_embed.sin()), dim=1)
    return pos_embed


def get_2d_pos_emb(patch_num,dim,device="cuda"):
    height_embed = get_1d_pos_emb(patch_num[0],dim//2,device)
    height_embed = rearrange(height_embed,"h dim -> h 1 dim").repeat(1,patch_num[1],1)
    width_embed = get_1d_pos_emb(patch_num[1],dim//2,device)
    width_embed = rearrange(width_embed,"w dim -> 1 w dim").repeat(patch_num[0],1,1)
    return torch.cat([height_embed,width_embed],dim=-1)


class Multi_Head(nn.Module):
    def __init__(self, dim, head_num, drop_prob) -> None:
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scale = self.head_dim ** (-0.5)
        assert(type(self.head_dim) == int)

        self.qkv = nn.Linear(dim, 3*dim)
        self.attn_drop = nn.Dropout(drop_prob)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_prob)

    def forward(self, input):
        # input (batch_size,patch_num,dim)
        batch_size, patch_num = input.shape[0], input.shape[1]
        x = input
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv,self.dim,dim=2)
        attn = q @ k.transpose(-1, -2) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        #reshape 能不能去掉
        output = attn @ v
        output = self.proj_drop(self.proj(output))
        return output


class Transformer_Block(nn.Module):
    def __init__(self,dim,head) -> None:
        super().__init__()
        self.multi_head = Multi_Head(dim=dim, head_num=head, drop_prob=0)
        self.layer_norm0 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.ReLU(),
            nn.Linear(4*dim,dim)
        )
        self.layer_norm1 = nn.LayerNorm(dim)

    def forward(self,input):
        x = self.multi_head(input)
        x += input
        x0 = self.layer_norm0(x)

        x = self.mlp(x0)
        x += x0
        x = self.layer_norm1(x)
        return x


class Vision_Transformer(nn.Module):
    def __init__(self,image_shape,patch_shape,input_steps,num_features,dim,head_num,block_num) -> None:
        super().__init__()
        self.image_shape,self.patch_shape = image_shape,patch_shape
        self.input_steps = input_steps
        self.dim,self.head_num = dim,head_num
        self.block_num = block_num
        self.patch_embed = nn.Linear(patch_shape[0]*patch_shape[1]*num_features*input_steps,dim)
        self.pos_embed = get_2d_pos_emb([image_shape[0]//patch_shape[0],image_shape[1]//patch_shape[1]],dim)

        self.blocks = nn.ModuleList()
        for i in range(block_num):
            self.blocks.append(Transformer_Block(dim,head_num))

        self.patch_decoder = nn.Linear(dim,patch_shape[0]*patch_shape[1]*input_steps)


    def forward(self,x):
        # x (batch_size,input_steps,features,image_shape)
        batch_size,input_steps = x.shape[0],x.shape[1]
        patch_num = [self.image_shape[0]//self.patch_shape[0],self.image_shape[1]//self.patch_shape[1]]

        # x (batch_size,patch_num[0],patch_num[1],patch_shape[0] * patch_shape[1] * num_features * input_steps)
        x = rearrange(x,"bs ws fn (pn0 ps0) (pn1 ps1) -> bs pn0 pn1 (ps0 ps1 fn ws)",ps0=self.patch_shape[0],ps1=self.patch_shape[1])

        # x (batch_size,patch_num[0],patch_num[1],dim)
        x = self.patch_embed(x)

        # x (batch_size,*patch_num,dim)
        x = rearrange(x,"bs pn0 pn1 dim -> bs (pn0 pn1) dim")

        x = x + rearrange(self.pos_embed,"pn0 pn1 d -> 1 (pn0 pn1) d").repeat(batch_size,1,1)

        for i in range(self.block_num):
            x = self.blocks[i](x)

        # x (batch_size,patch_num[0] * patch_num[1],patch_shape[0] * patch_shape[1] * input_steps)
        x = self.patch_decoder(x)

        x = rearrange(x,"bs (pn0 pn1) (ps0 ps1 ws) -> bs ws 1 (pn0 ps0) (pn1 ps1)",ps0=self.patch_shape[0],ps1=self.patch_shape[1],pn0=patch_num[0])
        

        return x
        

        





