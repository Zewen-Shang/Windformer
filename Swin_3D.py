import torch
from torch import nn
from einops import rearrange

# L = H * W * D
# N = Wh * Ww * Wd


class PatchEmbed(nn.Module):
    def __init__(self,image_size,patch_size,input_steps,num_features,dim) -> None:
        """
        input : (B, input_steps, num_features, *image_size)
        output : (B, L, C)
        """
        super().__init__()
        self.image_size,self.patch_size = image_size,patch_size
        self.input_steps,self.num_features = input_steps,num_features
        self.dim = dim

        self.proj = nn.Conv2d(num_features,dim,patch_size,patch_size)

    def forward(self,x):
        # B -- batch_size, C -- dim
        # x(B,input_steps,num_features,*image_size)
        input_resolution = [self.image_size[0]//self.patch_size[0],self.image_size[1]//self.patch_size[1],self.image_size[2]//self.patch_size[2]]

        # x(B,input_steps * num_features,*image_size)
        x = rearrange(x,"B Id fn Ih Iw -> B fn Id Ih Iw")

        # x(B,C,*window_size)
        x = self.proj(x)

        # x (batch_size,window_size[0] * window_size[1],dim)
        x = rearrange(x, "B C D H W -> B (D H W) C")
        return x


class PatchMerge(nn.Module):
    def __init__(self,input_resolution,dim) -> None:
        """
        input : (B, L, C)
        output : (B, L, C)
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        self.conv = nn.Conv2d(dim,dim,3,padding="same")
        self.norm = nn.LayerNorm(dim)

    def forward(self,x):
        # H,W 指的是patch的分辨率
        D,H,W = self.input_resolution
        # L = H * W
        B,L,C = x.shape

        # x (C, B, H, W)
        x = rearrange(x,"B (D H W) C -> B C D H W",H=H,W=W)

        x = self.conv(x)

        x = rearrange(x,"B C D H W -> B (D H W) C")
        x = self.norm(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self,window_size,dim,num_heads,attn_drop,proj_drop) -> None:
        """
        input x : (B_, N, C)
        input mask : mask (nW,N,N)
        output : (B_, N, C)
        """
        super().__init__()
        self.window_size,self.dim,self.num_heads = window_size,dim,num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_table = nn.Parameter(torch.zeros(((2*window_size[0]-1)*(2*window_size[1]-1)*(2*window_size[2]-1),num_heads)))

        coords_d = torch.arange(0,window_size[0])
        coords_h = torch.arange(0,window_size[1])
        coords_w = torch.arange(0,window_size[2])

        # coords (3, Wd, Wh, Ww)
        coords = torch.stack(torch.meshgrid([coords_h,coords_w]))
        # coords (3, N)
        coords = coords.flatten(1)

        # relative_coords (N, N, 3)
        relative_coords = (coords[:,:,None] - coords[:,None,:]).permute(1,2,0)
        
        relative_coords[:,:,0] += (window_size[0]-1) 
        relative_coords[:,:,1] += (window_size[1]-1)
        relative_coords[:,:,2] += (window_size[2]-1)

        relative_coords[:,:,0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:,:,1] *= (2 * window_size[2] - 1)
        # relative_position_index  (N, N)
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index",relative_position_index)

        self.qkv = nn.Linear(dim,dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_table,std=0.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x,mask):
        # mask (nW,N,N)

        # B_:batch_size * num_windows
        # N : window 中的patch数  N = Wh * Ww * Wd
        B_,N,C = x.shape
        Wd,Wh,Ww = self.window_size

        # nH : num_heads
        # hD : head_dim
        # q,k,v (B_, nH, N, hd)
        q,k,v = rearrange(self.qkv(x),"Bs N (nH hD qkv) -> qkv Bs nH N hD",qkv=3,nH=self.num_heads)
        
        # attn (B_, nH, N, N)
        attn = (q @ k.transpose(-1,-2)) * self.scale
        # relative_position_bias (nH, N, N)
        relative_position_bias = rearrange(self.relative_position_table[self.relative_position_index.view(-1)],
        "(Wd0 Wh0 Ww0 Wd1 Wh1 Ww1) nH -> nH (Wd0 Wh0 Ww0) (Wd1 Wh1 Ww1)",Wd0=Wd,Wd1=Wd,Wh0=Wh,Wh1=Wh,Ww0=Ww,Ww1=Ww)

        attn = attn + relative_position_bias.unsqueeze(0)

        if(mask != None):
            nW = mask.shape[0]
            attn = rearrange(attn,"(B nW) nH N0 N1 -> B nW nH N0 N1",nW=nW)
            attn = attn + rearrange(mask,"nW N0 N1 -> 1 nW 1 N0 N1")
            attn = rearrange(attn,"B nW nH N0 N1 -> (B nW) nH N0 N1")
        
        attn = self.softmax(attn)
        # x (B_,N,C)
        x = rearrange(attn @ v,"Bs nH N hd -> Bs N (nH hd)")
        return x




def window_partition(x,window_size):
    B,H,W,C = x.shape
    Wd,Wh,Ww = window_size
    x = rearrange(x,"B (nWd Wd) (nWh Wh) (nWw Ww) C -> (B nWd nWh nWw) (Wd Wh Ww) C",Wd=Wd,Wh=Wh,Ww=Ww)
    return x

def window_reverse(x,window_size,D,H,W):
    B_,N,C = x.shape
    Wd,Wh,Ww = window_size
    x = rearrange(x,"(B nWd nWh nWw) (Wd Wh Ww) C -> B (nWd Wd) (nWh Wh) (nWw Ww) C",nWd=D//Wd,nWh=H//Wh,nWw=W//Ww,Wd=Wd,Wh=Wh,Ww=Ww)
    return x


class SwinBlock(nn.Module):

    def __init__(self,dim,input_resolution,num_heads,window_size,shift_size,mlp_ratio,attn_drop,proj_drop) -> None:
        """
        input : (B, L, C)
        output : (B, L, C)
        """
        super().__init__()
        self.dim,self.input_resolution,self.num_heads = dim,input_resolution,num_heads
        self.window_size,self.shift_size = window_size,shift_size
        self.mlp_ratio,self.attn_drop,self.proj_drop = mlp_ratio,attn_drop,proj_drop

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(window_size,dim,num_heads,attn_drop,proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim,mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim,dim),
            nn.ReLU()
        )

        D,H,W = self.input_resolution

        if(self.shift_size[0]*self.shift_size[1]*self.shift_size[2] != 0):

            mask = torch.zeros((1,D,H,W,1))

            d_slices = [
                slice(0,-window_size[0]),
                slice(-window_size[0],-shift_size[0]),
                slice(-shift_size[0])
            ]

            h_slices = [
                slice(0,-window_size[1]),
                slice(-window_size[1],-shift_size[1]),
                slice(-shift_size[1])
            ]
            w_slices = [
                slice(0,-window_size[2]),
                slice(-window_size[2],-shift_size[2]),
                slice(-shift_size[2])
            ]
            
            cnt = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        mask[:,d,h,w] = cnt
                        cnt = cnt + 1
            # mask_window (nW, N, C)
            mask_window = window_partition(mask,window_size)
            # mask_window (nW, Wh*Ww)
            mask_window = mask_window.reshape(mask_window.shape[0],mask_window.shape[1])
            # attn_window (nW,Wh*Ww,Wh*Ww)
            attn_mask = mask_window[:,:,None] - mask_window[:,None,:]
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask",attn_mask)
        
    def forward(self,x):
        B,L,C = x.shape
        H,W = self.input_resolution


        short_cut = x
        x = self.norm1(x)
        x = x.reshape(B,H,W,C)

        if(self.shift_size[0]*self.shift_size[1] != 0):
            shifted_x = torch.roll(x,[-self.shift_size[0],-self.shift_size[1]],[1,2])
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x,self.window_size)
        attn_windows = self.attn(x_windows,self.attn_mask)
        # x (B,H,W,C)
        shifted_x = window_reverse(x_windows,self.window_size,H,W)

        if(self.shift_size[0]*self.shift_size[1] != 0):
            x = torch.roll(shifted_x,self.shift_size,[1,2])
        else:
            x = shifted_x

        x = rearrange(x,"B H W C -> B (H W) C") + short_cut
        
        short_cut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + short_cut

        return x

class BasicLayer(nn.Module):
    def __init__(self,dim,input_resolution,num_heads,window_size,depth,mlp_ratio=4,attn_drop=0,proj_drop=0) -> None:
        """
        input : (B, L, C)
        output : (B, L, C)
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [SwinBlock(dim,input_resolution,num_heads,window_size,
            [window_size[0]//2,window_size[1]//2,window_size[2]//2] if(i % 2) else [0,0],mlp_ratio,attn_drop,proj_drop)
            for i in range(depth)]
        )
        self.patch_merge = PatchMerge(input_resolution,dim)


    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        x = self.patch_merge(x)
        return x

class Swin(nn.Module):
    def __init__(self,image_size,patch_size,input_steps,predict_steps,num_features,
    dim,num_heads,window_size,depth,mlp_ratio=4,attn_drop=0,proj_drop=0) -> None:
        """
        input x : (B,input_steps,features,image_size)
        output : (B, image_size)
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(image_size,patch_size,input_steps,num_features,dim)
        D,H,W = image_size[0]//patch_size[0],image_size[1]//patch_size[1],image_size[2]//patch_size[2]
        self.input_resolution = [D,H,W]

        self.layer = BasicLayer(dim,self.input_resolution,num_heads,window_size,depth,mlp_ratio,attn_drop,proj_drop)

        self.output_head = nn.Linear(dim,patch_size[0] * patch_size[1] * patch_size[2])
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight,std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1)

    def forward(self,x):

        D,H,W = self.input_resolution
        Pd,Ph,Pw = self.patch_size
        # x (B, L, C)
        x = self.patch_embed(x)
        x = self.layer(x)
        # x (B, L, Ph * Pw)
        x = self.output_head(x)

        x = rearrange(x,"B (D H W) (Pd Ph Pw) -> B (D Pd) (H Ph) (W Pw)",D=D,H=H,W=W,Pd=Pd,Ph=Ph,Pw=Pw)
        return x

    