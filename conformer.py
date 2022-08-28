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

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,drop_prob) -> None:
        super().__init__()
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.relu0 = nn.ReLU()
        self.conv0 = nn.Conv2d(in_channels,out_channels,3,padding="same")
        self.dropout0 = nn.Dropout(drop_prob)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(out_channels,out_channels,3,padding="same")
        self.dropout1 = nn.Dropout(drop_prob)

        self.attn_bn0 = nn.BatchNorm2d(in_channels)
        self.attn_relu0 = nn.ReLU()
        self.attn_conv0 = nn.Conv2d(in_channels,out_channels,3,padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        x = input

        x = self.bn0(x)
        x = self.relu0(x)
        x = self.conv0(x)

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        attn = self.attn_conv0(self.attn_relu0(self.attn_bn0(input)))
        attn = self.sigmoid(attn)

        x = x * attn

        return x + input

class ResNet(nn.Module):
    def __init__(self,img_shape,depth,out_channels,drop_prob) -> None:
        super().__init__()
        self.img_shape = img_shape #因为要池化
        self.out_shape = [i // 2 for i in img_shape] #因为要池化
        self.depth = depth
        self.out_channels = out_channels

        self.batchnorm0 = nn.BatchNorm2d(3)
        self.relu0 = nn.ReLU()
        self.conv0 = nn.Conv2d(3,out_channels,5,padding="same")
        self.maxpool0 = nn.MaxPool2d(2,2)

        self.resblocks = nn.ModuleList([ResBlock(out_channels,out_channels,drop_prob) for i in range(self.depth)])

    def forward(self,input):
        x = input

        x = self.conv0(self.relu0(self.batchnorm0(x)))
        x = self.maxpool0(x)

        for resblock in self.resblocks:
            x = resblock(x)

        return x

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
    def __init__(self,image_shape,patch_shape,channels,dim,mlp_dim,depth,num_heads,drop_prob) -> None:
        super().__init__()
        self.image_shape,self.patch_shape,self.channels = image_shape,patch_shape,channels
        self.dim,self.mlp_dim,self.depth = dim,mlp_dim,depth
        self.num_heads,self.drop_prob = num_heads,drop_prob


        self.h,self.w = image_shape[0] // patch_shape[0],image_shape[1]//patch_shape[1]
        self.pos_embed = nn.Parameter(torch.randn(1,self.h * self.w,dim))
        # self.cls_token = nn.Parameter(torch.rand(1,1,dim))

        self.img_embed = nn.Linear(patch_shape[0] * patch_shape[1] * channels,dim)

        self.transformer = Transformer(dim,mlp_dim,depth,num_heads,drop_prob)

        # self.linear_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim,1)
        # )

    def forward(self,input):
        batch_size = input.shape[0]
        x = input.reshape(batch_size,self.channels,self.h,self.patch_shape[0],self.w,self.patch_shape[1])
        x = x.permute(0,2,4,1,3,5).reshape(batch_size,self.h,self.w,self.channels * self.patch_shape[0] * self.patch_shape[1])
        
        x = self.img_embed(x)
        x = x.reshape(batch_size,self.h * self.w,self.dim) + self.pos_embed

        # cls_tokens = self.cls_token.repeat(batch_size,1,1)
        # x = torch.cat([cls_tokens,x],dim=1)

        x = self.transformer(x)
        x = x.view(batch_size,self.h,self.w,self.dim)
        # x = self.linear_head(x)
        # x = torch.sigmoid(x) * self.scale
        return x

class FCU(nn.Module):
    def __init__(self,in_shape,out_shape,in_channels,out_channels) -> None:
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layernorm = nn.LayerNorm(in_channels)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_channels,out_channels)

    def forward(self,input):
        x = self.linear(self.relu(self.layernorm(input)))
        x = x.permute(0,3,1,2)
        x = nn.functional.interpolate(x,self.out_shape)
        return x

class Conformer(nn.Module):
    def __init__(self,res_depth,image_shape,patch_shape,in_channels,res_out_channels,dim,mlp_dim,vit_depth,num_heads,drop_prob) -> None:
        super().__init__()
        self.resnet = ResNet(img_shape=image_shape,depth=res_depth,out_channels=res_out_channels,drop_prob=drop_prob)
        self.vit = Vit(image_shape,patch_shape,in_channels,dim,mlp_dim,vit_depth,num_heads,drop_prob)
        self.fcu = FCU((self.vit.h,self.vit.w),self.resnet.out_shape,self.vit.dim,self.resnet.out_channels)

        self.batchnorm0 = nn.BatchNorm2d(res_out_channels * 2)
        self.relu0 = nn.ReLU()
        self.conv0 = nn.Conv2d(res_out_channels * 2,1,3,padding="same")
        self.maxpool = nn.MaxPool2d(5,5)
        self.linear0 = nn.Linear(3*2,1)
        

    def forward(self,input):
        resnet_output = self.resnet(input)
        vit_output = self.vit(input)
        fcu_output = self.fcu(vit_output)

        merge_output = torch.cat([resnet_output,fcu_output],dim=1)
        x = self.maxpool(self.conv0(self.relu0(self.batchnorm0(merge_output))))
        x = x.reshape(x.shape[0],3*2)
        x = self.linear0(x)
        
        return x

