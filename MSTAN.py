from turtle import forward
import torch
import torch.nn as nn

class CNN_Block(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding="same")
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self,input):
        x = self.conv(input)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class CNN(nn.Module):
    def __init__(self,window_size,feature_num,img_shape,depth) -> None:
        super().__init__()
        self.window_size,self.feature_num,self.img_shape = window_size,feature_num,img_shape
        self.conv0 = nn.Conv2d(feature_num,32,7,padding="same")
        self.blocks = nn.ModuleList(
            [CNN_Block(32,32) for i in range(depth)]
        )
        self.conv1 = nn.Conv2d(32,1,7,padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        # input (batch_size,window_size,feature_num,img_height,img_weight)
        batch_size = input.shape[0]
        x = input.reshape(batch_size * self.window_size,self.feature_num,self.img_shape[0],self.img_shape[1])
        x = self.conv0(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x.reshape(batch_size,self.window_size,self.img_shape[0] * self.img_shape[1])
        x = nn.functional.softmax(x,dim=-1)
        x = x.reshape(batch_size,self.window_size,self.img_shape[0],self.img_shape[1])
        # output (batch_size,window_size,30,20)
        return x

class LSTM_Attn(nn.Module):
    def __init__(self,window_size,feature_num,img_shape,hidden_size,device="cuda") -> None:
        super().__init__()
        self.feature_num = feature_num
        self.img_shape = img_shape
        self.drive_num = img_shape[0] * img_shape[1]
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.device = device

        self.encode_attn = nn.Linear(feature_num * window_size + 2*hidden_size,1)
        self.softmax = nn.Softmax(-1)
        # ??
        self.lstm = nn.LSTM(input_size=self.drive_num * feature_num,hidden_size=hidden_size,num_layers=1,batch_first=True)

    def forward(self,input):
        # input (batch_size,window_size,feature_num,img_height,img_weight)
        batch_size = input.shape[0]
        # input (batch_size,window_size,feature_num,drive_num)
        input = input.reshape(batch_size,self.window_size,self.feature_num,self.img_shape[0] * self.img_shape[1])
        # input (batch_size,drive_num,feature_num,window_size)
        input = input.permute(0,3,2,1)
        self.hidden_states = torch.zeros((batch_size,self.window_size+1,self.hidden_size),device=self.device)
        self.cell_states = torch.zeros((batch_size,self.window_size+1,self.hidden_size),device=self.device)

        
        for t in range(self.window_size):
            # h_0,s_0 (1,batch_size,hiden_size)
            h_0,s_0 = self.hidden_states[:,t:t+1].permute(1,0,2),self.cell_states[:,t:t+1].permute(1,0,2)
            # hidden_state (batch_size,drive_num,hidden_state)
            hidden_state = self.hidden_states[:,t:t+1].repeat((1,self.drive_num,1))
            cell_state = self.cell_states[:,t:t+1].repeat((1,self.drive_num,1))
            # x (batch_size,drive_num,feature_num * window_size)
            x = input.reshape(batch_size,self.drive_num,self.feature_num*self.window_size)
            attn_in = torch.cat((hidden_state,cell_state,x),dim=2)
            # e,a (batch_size,drive_num)
            e = self.encode_attn(attn_in).squeeze()
            a = self.softmax(e)
            # x_hat (batch_size,drive_num * feature_num)
            x_hat = (a.unsqueeze(2) * input[:,:,:,t]).reshape(batch_size,self.drive_num * self.feature_num)
            _,(h_0,s_0) = self.lstm(x_hat.unsqueeze(1),(h_0.contiguous(),s_0.contiguous()))
            self.hidden_states[:,t+1],self.cell_states[:,t+1] = h_0.squeeze(),s_0.squeeze()

        # output (batch_size,window_size,hidden_size)
        return self.hidden_states[:,1:]

class Spatial_Attn(nn.Module):
    def __init__(self,window_size,feature_num,img_shape,patch_shape,dim,head_num,drop_prob=0) -> None:
        super().__init__()
        self.window_size,self.feature_num = window_size,feature_num
        self.img_shape,self.drive_num = img_shape,img_shape[0] // patch_shape[0] * img_shape[1] // patch_shape[1]
        self.patch_shape = patch_shape
        
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scale = self.head_dim ** (-0.5)
        assert(type(self.head_dim) == int)

        self.avgpool = nn.AvgPool2d(2,2)
        self.layernorm = nn.LayerNorm(self.feature_num)
        self.qkv = nn.Linear(self.feature_num,3*dim)
        self.attn_drop = nn.Dropout(drop_prob)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(drop_prob)

        self.linear = nn.Linear(dim,1)
        

    def forward(self,input):
        # input (batch_size,window_size,feature_num,img_height,img_weight)
        batch_size = input.shape[0]
        
        x = self.avgpool(input.reshape(batch_size * self.window_size,self.feature_num,self.img_shape[0],self.img_shape[1]))
        # x (batch_size,window_size,feature_num,15 * 10)
        x = x.reshape(batch_size,self.window_size,self.feature_num,self.drive_num)
        # x (batch_size,window_size,15 * 10,feature_num)
        x = x.permute(0,1,3,2)
        qkv = self.qkv(x).reshape(batch_size,self.window_size,self.drive_num,3,self.head_num,self.head_dim)
        q,k,v = qkv.permute(3,0,1,4,2,5)
        attn = q @ k.transpose(-1,-2) * self.scale
        attn = nn.functional.softmax(attn,dim = -1)
        attn = self.attn_drop(attn)
        x = (attn @ v).reshape(batch_size,self.window_size,self.drive_num,self.dim)
        # x (batch_size,window_size,drive_num,dim)
        x = self.proj_drop(self.proj(x))
        # x (batch_size,window_size,drive_num)
        x = self.linear(x).squeeze()
        x = nn.functional.softmax(x,dim=-1)
        # x (batch_size,window_size,15,10)
        x = x.reshape(batch_size,self.window_size,self.img_shape[0] // self.patch_shape[0],self.img_shape[1] // self.patch_shape[1])
        # x (batch_size,window_size,30,20)
        x = nn.functional.interpolate(x,scale_factor=2,mode='nearest')
        # x (batch_size,window_size,30 * 20)
        x = x.reshape(batch_size,self.window_size,self.img_shape[0] * self.img_shape[1])
        output = input.reshape(batch_size,self.window_size,self.feature_num,self.img_shape[0] * self.img_shape[1]).permute(0,1,3,2)
        output = output * x.unsqueeze(3)
        output = output.sum(dim=-2)
        return output

class LSTM(nn.Module):
    def __init__(self,feature_num,window_size,hidden_size) -> None:
        super().__init__()
        self.feature_num = feature_num
        self.window_size = window_size
        self.hidden_size = hidden_size
    
        self.lstm = nn.LSTM(input_size=feature_num,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.gate = nn.GLU()
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self,input):
        #input (batch_size,window_size,feature_num)
        batch_size = input.shape[0]
        # x (batch_size,window_size,hidden_size)
        x,(h_n,c_n) = self.lstm(input)
        
        gate_x = self.gate(x.repeat((1,1,2)))
        x = x + gate_x
        x = self.layernorm(x)
        # x (batch_size,window_size,hidden_size)
        return x

def get_pos_emb(window_size,dim,device="cuda",temperature=1e4,dtype=torch.float):
    pos = torch.arange(0,window_size+1,device=device)
    
    omega = torch.arange(dim//2,device=device)/(dim/2-1)
    omega = 1. / temperature ** omega
    pos_embed = pos[:,None] * omega[None,:]
    

    pos_embed = torch.cat((pos_embed.cos(),pos_embed.sin()),dim=1)
    return pos_embed.to(dtype)


class SelfAttention(nn.Module):
    def __init__(self,dim,head_num,drop_prob) -> None:
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scale = self.head_dim ** (-0.5)
        assert(type(self.head_dim) == int)

        self.layernorm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim,3*dim)
        self.attn_drop = nn.Dropout(drop_prob)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(drop_prob)

    def forward(self,input):
        # input (batch_size,window_size+1,hidden_state)
        batch_size,num_seq = input.shape[0],input.shape[1]
        assert(self.dim == input.shape[2])
        x = input
        qkv = self.qkv(x).reshape(batch_size,num_seq,3,self.head_num,self.head_dim)
        q,k,v = qkv.permute(2,0,3,1,4)
        attn = q @ k.transpose(-1,-2) * self.scale
        attn = nn.functional.softmax(attn,dim = -1)
        attn = self.attn_drop(attn)
        output = (attn @ v).reshape(batch_size,num_seq,self.dim)
        output = self.proj_drop(self.proj(output))
        return output


class MSTAN(nn.Module):
    def __init__(self,feature_num,img_shape,window_size,patch_size,hidden_size,head_num) -> None:
        super().__init__()
        drive_num = 1
        self.feature_num,self.window_size,self.drive_num,self.hidden_size = feature_num,window_size,drive_num,hidden_size
        self.head_num = head_num
        self.img_shape = img_shape

        self.spatial_attn = Spatial_Attn(window_size,feature_num,img_shape,patch_size,hidden_size,head_num,0)
        self.lstm_attn = LSTM_Attn(window_size,feature_num,img_shape,hidden_size)
        self.cnn = CNN(window_size,feature_num,img_shape,2)

        self.lstm = LSTM(feature_num=feature_num,window_size=window_size,hidden_size=hidden_size)
        
        self.cls = torch.rand((1,1,hidden_size),device="cuda")
        # pos_embed (window_size + 1,hidden_state)
        self.pos_embed = get_pos_emb(window_size=window_size,dim=hidden_size)
        self.self_attn = SelfAttention(dim=hidden_size,head_num=head_num,drop_prob=0.1)

        self.linear = nn.Linear(hidden_size,1)

    def forward(self,input):
        # input (batch_size,feature_num,img_height,img_weight,window_size)
        batch_size = input.shape[0]
        # x (batch_size,window_size,feature_num,img_height,img_weight)
        x = input.permute(0,4,1,2,3)

        # # x (batch_size,window_size,hidden_size)
        # x = self.lstm_attn(x)

        x = self.spatial_attn(x)

        # # area_attn (batch_size,window_size,1,img_height,img_weight)
        # area_attn = self.cnn(x).unsqueeze(2)
        # x = x * area_attn
        # # x (batch_size,window_size,feature_num)
        # x = x.reshape(batch_size,self.window_size,self.feature_num,self.img_shape[0] * self.img_shape[1]).sum(dim=-1)

        # x (batch_size,window_size,hidden_size)
        x = self.lstm(x)
        # x (batch_size,window_size + 1,hidden_size)
        x = torch.cat((x,self.cls.repeat((batch_size,1,1))),dim=1)
        x = x + self.pos_embed.reshape(1,self.window_size+1,self.hidden_size).repeat((batch_size,1,1))

        x = self.self_attn(x)
        # x (batch_size,1,hidden_size)
        x = x[:,0]

        # x = nn.functional.relu(x)
        x = self.linear(x)

        return x