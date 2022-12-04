import torch
import torch.nn as nn
from einops import rearrange

class Conv_LstmCell(nn.Module):
    def __init__(self,in_channels,hidden_channels,in2hi_kernel,hi2hi_kernel) -> None:
        super().__init__()
        self.in_channels,self.hidden_channels = in_channels,hidden_channels
        self.in2hi_kernel,self.hi2hi_kernel = in2hi_kernel,hi2hi_kernel

        self.W_x = nn.Conv2d(in_channels,hidden_channels*4,in2hi_kernel,padding="same")
        self.W_h = nn.Conv2d(hidden_channels,hidden_channels*4,hi2hi_kernel,padding="same")

    def forward(self,x,h_pre,c_pre):
        #x (batch_size,in_channels,image_shape)

        xi,xf,xc,xo = torch.split(self.W_x(x),self.hidden_channels,dim=1)
        hi,hf,hc,ho = torch.split(self.W_h(h_pre),self.hidden_channels,dim=1)

        i = torch.sigmoid(xi+hi)
        f = torch.sigmoid(xf+hf)
        o = torch.sigmoid(xo+ho)
        c_new = f * c_pre + i * torch.tanh(xc+hc)
        h = o * torch.tanh(c_new)

        # h,c_next (batch_size,hidden_channels,images_shape)
        return h,c_new

class Conv_LstmModule(nn.Module):
    def __init__(self,in_channels,hidden_channels,in2hi_kernel,hi2hi_kernel,layers_num) -> None:
        super().__init__()
        self.in_channels,self.hidden_channels = in_channels,hidden_channels
        self.in2hi_kernel,self.hi2hi_kernel = in2hi_kernel,hi2hi_kernel
        self.layers_num = layers_num
        self.cell_list = nn.ModuleList()
        for i in range(layers_num):
            cell = Conv_LstmCell(in_channels if i == 0 else hidden_channels,hidden_channels,in2hi_kernel,hi2hi_kernel)
            self.cell_list.append(cell)
        
    def forward(self,x,h_0=None,c_0=None):
        
        # x (batch_size,input_steps,hidden_channels,image_shape)
        
        batch_size,input_steps,image_shape = x.shape[0],x.shape[1],(x.shape[3],x.shape[4])
        
        # h_0,c_0 (batch_size,hidden_channels,image_shape,layers_num)
        if(h_0 == None):
            h_0 = torch.zeros((batch_size,self.hidden_channels,*image_shape,self.layers_num),device=x.device)
        if(c_0 == None):
            c_0 = torch.zeros((batch_size,self.hidden_channels,*image_shape,self.layers_num),device=x.device)

        # output (batch_size,input_steps,hidden_channels,image_shape)
        output = x
        hidden,cell = [],[]
        for layer in range(self.layers_num):
            # h,c (batch_size,hidden_channels,image_shape)
            h,c = h_0[:,:,:,:,layer],c_0[:,:,:,:,layer]
            out = []
            for i in range(input_steps):
                h,c = self.cell_list[layer](output[:,i],h,c)
                out.append(h)
            output = torch.stack(out,dim=1)
            hidden.append(h)
            cell.append(c)
        
        #hidden cell  (batch_size,input_steps,hidden_channels,image_shape)
        hidden = torch.stack(hidden,dim=-1)
        cell = torch.stack(cell,dim=-1)
        return output,hidden,cell

class Conv_Lstm(nn.Module):
    def __init__(self,in_channels,hidden_channels,in2hi_kernel,hi2hi_kernel,layers_num,predict_steps) -> None:
        super().__init__()

        self.predict_steps = predict_steps

        self.encoder = Conv_LstmModule(in_channels,hidden_channels,in2hi_kernel,hi2hi_kernel,layers_num)
        self.decoder = Conv_LstmModule(1,hidden_channels,in2hi_kernel,hi2hi_kernel,layers_num)

        self.fc = nn.Conv2d(hidden_channels,1,1)

    def forward(self,x):
        # x (batch_size,input_steps,in_channels,image_shape)

        (batch_size,input_steps,in_channels),image_shape = x.shape[0:3],x.shape[3:5]
        
        _,hidden,cell = self.encoder(x)

        # decoder_input (batch_size, 1(input_steps), 1(in_channels), *image_shape)
        decoder_input = torch.zeros((batch_size,1,1,*image_shape),device=x.device)
        
        out = []
        for i in range(self.predict_steps):
            decoder_output,hidden,cell = self.decoder(decoder_input,hidden,cell)
            fc_output = self.fc(rearrange(decoder_output,"bz 1 hc h w -> (bz 1) hc h w"))
            out.append(fc_output)
            decoder_input = rearrange(fc_output,"(bz 1) 1 h w -> bz 1 1 h w")
        # output (batch_size, input_steps, 1, h, w)
        out = torch.stack(out,dim=1)
        out = rearrange(out,"bz ws 1 h w -> bz ws h w")
        return out

