import torch
import torch.nn as nn

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

        self.pos_embed = torch.rand((1,1,self.drive_num,feature_num),device="cuda")

        assert(type(self.head_dim) == int)

        self.avgpool = nn.AvgPool2d(2,2)

        self.test = nn.Linear(feature_num,dim)

        self.layernorm = nn.LayerNorm(self.feature_num)
        self.qkv = nn.Linear(self.feature_num,3*dim)
        self.attn_drop = nn.Dropout(drop_prob)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(drop_prob)
        

    def forward(self,input):
        # input (batch_size,window_size,feature_num,img_height,img_weight)
        batch_size = input.shape[0]
        
        x = self.avgpool(input.reshape(batch_size * self.window_size,self.feature_num,self.img_shape[0],self.img_shape[1]))
        # x (batch_size,window_size,feature_num,15 * 10)
        x = x.reshape(batch_size,self.window_size,self.feature_num,self.drive_num)
        # x (batch_size,window_size,15 * 10,feature_num)
        x = x.permute(0,1,3,2)

        # x = self.test(x)

        x = x + self.pos_embed.repeat((batch_size,self.window_size,1,1))
        
        x = self.layernorm(x)
        qkv = self.qkv(x).reshape(batch_size,self.window_size,self.drive_num,3,self.head_num,self.head_dim)
        q,k,v = qkv.permute(3,0,1,4,2,5)
        attn = q @ k.transpose(-1,-2) * self.scale
        attn = nn.functional.softmax(attn,dim = -1)
        attn = self.attn_drop(attn)
        x = (attn @ v).reshape(batch_size,self.window_size,self.drive_num,self.dim)
        # x (batch_size,window_size,drive_num,spatial_dim)
        x = self.proj_drop(self.proj(x))

        return x


class LSTM_Attn(nn.Module):
    def __init__(self,window_size,input_size,drive_num,hidden_size,device="cuda") -> None:
        super().__init__()
        self.input_size = input_size
        self.drive_num = drive_num
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.device = device

        self.encode_attn = nn.Linear(input_size * window_size + 2*hidden_size,1)
        self.softmax = nn.Softmax(-1)
        # ??
        self.lstm = nn.LSTM(input_size=self.drive_num * input_size,hidden_size=hidden_size,num_layers=1,batch_first=True)

    def forward(self,input):
        # input (batch_size,window_size,drive_num,input_size)
        batch_size = input.shape[0]
        # input (batch_size,drive_num,input_size,window_size)
        input = input.permute(0,2,3,1)
        self.hidden_states = torch.zeros((batch_size,self.window_size+1,self.hidden_size),device=self.device)
        self.cell_states = torch.zeros((batch_size,self.window_size+1,self.hidden_size),device=self.device)
        
        for t in range(self.window_size):
            # h_0,s_0 (1,batch_size,hiden_size)
            h_0,s_0 = self.hidden_states[:,t:t+1].permute(1,0,2),self.cell_states[:,t:t+1].permute(1,0,2)
            # hidden_state (batch_size,drive_num,hidden_state)
            hidden_state = self.hidden_states[:,t:t+1].repeat((1,self.drive_num,1))
            cell_state = self.cell_states[:,t:t+1].repeat((1,self.drive_num,1))
            # x (batch_size,drive_num,feature_num * window_size)
            x = input.reshape(batch_size,self.drive_num,self.input_size*self.window_size)
            attn_in = torch.cat((hidden_state,cell_state,x),dim=2)
            # e,a (batch_size,drive_num)
            e = self.encode_attn(attn_in).squeeze()
            a = self.softmax(e)
            # x_hat (batch_size,drive_num * feature_num)
            x_hat = (a.unsqueeze(2) * input[:,:,:,t]).reshape(batch_size,self.drive_num * self.input_size)
            _,(h_0,s_0) = self.lstm(x_hat.unsqueeze(1),(h_0.contiguous(),s_0.contiguous()))
            self.hidden_states[:,t+1],self.cell_states[:,t+1] = h_0.squeeze(),s_0.squeeze()

        # output (batch_size,window_size,hidden_size)
        return self.hidden_states[:,1:]


class LSTM(nn.Module):
    def __init__(self,input_size,window_size,hidden_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.hidden_size = hidden_size
    
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=1,batch_first=True)
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


class Temporal_Attn(nn.Module):
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


class Dual_Trans(nn.Module):
    def __init__(self,feature_num,img_shape,window_size,spatial_dim,spatial_head,patch_shape,hidden_size,temporal_dim,temporal_head) -> None:
        super().__init__()
        
        self.feature_num,self.window_size,self.hidden_size = feature_num,window_size,hidden_size
        self.spatial_head,self.temporal_head = spatial_head,temporal_head
        self.spatial_dim = spatial_dim
        self.img_shape = img_shape

        self.spatial_attn = Spatial_Attn(window_size,feature_num,img_shape,patch_shape,spatial_dim,spatial_head)
        self.lstm_attn = LSTM_Attn(window_size=window_size,input_size=spatial_dim,
        drive_num=img_shape[0] * img_shape[1] // (patch_shape[0] * patch_shape[1]),hidden_size=hidden_size)

        self.lstm = LSTM(input_size=hidden_size,window_size=window_size,hidden_size=hidden_size)
        
        self.cls = torch.rand((1,1,hidden_size),device="cuda")
        # pos_embed (window_size + 1,hidden_state)
        self.pos_embed = get_pos_emb(window_size=window_size,dim=hidden_size)
        self.temporal_attn = Temporal_Attn(dim=hidden_size,head_num=temporal_head,drop_prob=0.1)

        self.linear = nn.Linear(hidden_size,1)

    def forward(self,input):
        # input (batch_size,feature_num,img_height,img_weight,window_size)
        batch_size = input.shape[0]
        # x (batch_size,window_size,feature_num,img_height,img_weight)
        x = input.permute(0,4,1,2,3)

        # x (batch_size,window_size,drive_num,spatial_dim)
        x = self.spatial_attn(x)

        # x (batch_size,window_size,hidden_size)
        x = self.lstm_attn(x)

        # x (batch_size,window_size,hidden_size)
        x = self.lstm(x)
        # x (batch_size,window_size + 1,hidden_size)
        x = torch.cat((x,self.cls.repeat((batch_size,1,1))),dim=1)
        x = x + self.pos_embed.reshape(1,self.window_size+1,self.hidden_size).repeat((batch_size,1,1))

        x = self.temporal_attn(x)
        # x (batch_size,1,hidden_size)
        x = x[:,0]

        # x = nn.functional.relu(x)
        x = self.linear(x)

        return x