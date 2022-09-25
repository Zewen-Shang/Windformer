from turtle import forward
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,feature_num,window_size,drive_num,hidden_size) -> None:
        super().__init__()
        self.feature_num = feature_num
        self.window_size = window_size
        self.drive_num = drive_num
        self.hidden_size = hidden_size
    
        self.lstm = nn.LSTM(input_size=feature_num * drive_num,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.gate = nn.GLU()
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self,input):
        #input (batch_size,feature_num,drive_num,window_size)
        batch_size = input.shape[0]
        x = input.permute(0,2,1,3).reshape(batch_size,self.window_size,self.feature_num * self.drive_num)
        # x (batch_size,window_size,hidden_size)
        x,(h_n,c_n) = self.lstm(x)
        
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

class MSTAN_Seq(nn.Module):
    def __init__(self,feature_num,window_size,drive_num,hidden_size,head_num) -> None:
        super().__init__()
        drive_num = 1
        self.feature_num,self.window_size,self.drive_num,self.hidden_size = feature_num,window_size,drive_num,hidden_size
        self.head_num = head_num
        self.lstm = LSTM(feature_num=feature_num,window_size=window_size,drive_num=drive_num,hidden_size=hidden_size)
        
        self.cls = torch.rand((1,1,hidden_size),device="cuda")
        # pos_embed (window_size + 1,hidden_state)
        self.pos_embed = get_pos_emb(window_size=window_size,dim=hidden_size)
        self.self_attn = SelfAttention(dim=hidden_size,head_num=head_num,drop_prob=0.1)

        self.linear = nn.Linear(hidden_size,1)

    def forward(self,input):
        #input (batch_size,feature_num,drive_num,window_size)
        batch_size = input.shape[0]
        input = input[:,:,50:51,:]
        # x (batch_size,window_size,hidden_size)
        x = self.lstm(input)
        # x (batch_size,window_size + 1,hidden_size)
        x = torch.cat((x,self.cls.repeat((batch_size,1,1))),dim=1)
        x = x + self.pos_embed.reshape(1,self.window_size+1,self.hidden_size).repeat((batch_size,1,1))

        x = self.self_attn(x)
        # x (batch_size,1,hidden_size)
        x = x[:,0]

        # x = nn.functional.relu(x)
        x = self.linear(x)

        return x