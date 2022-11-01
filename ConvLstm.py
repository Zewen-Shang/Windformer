import torch
import torch.nn as nn

class ConvLstmCell(nn.Module):
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

        i = nn.functional.sigmoid(xi+hi)
        f = nn.functional.sigmoid(xf+hf)
        o = nn.functional.sigmoid(xo+ho)
        c_new = f * c_pre + i * nn.functional.tanh(xc+hc)
        h = o * nn.functional.tanh(c_new)

        # h,c_next (batch_size,hidden_channels,images_shape)
        return h,c_new

class ConvLstmModule(nn.Module):
    def __init__(self,in_channels,hidden_channels,in2hi_kernel,hi2hi_kernel,layers_num) -> None:
        super().__init__()
        self.in_channels,self.hidden_channels = in_channels,hidden_channels
        self.in2hi_kernel,self.hi2hi_kernel = in2hi_kernel,hi2hi_kernel
        self.layers_num = layers_num
        self.cell_list = nn.ModuleList()
        for i in range(layers_num):
            cell = ConvLstmCell(in_channels if i == 0 else hidden_channels,hidden_channels,in2hi_kernel,hi2hi_kernel)
            self.cell_list.append(cell)
        
    def forward(self,x,h_0=None,c_0=None):
        
        # x (batch_size,window_size,hidden_channels,image_shape)
        
        batch_size,window_size,image_shape = x.shape[0],x.shape[1],(x.shape[3],x.shape[4])
        
        # h_0,c_0 (batch_size,hidden_channels,image_shape,layers_num)
        if(h_0 == None):
            h_0 = torch.tensor(batch_size,self.hidden_channels,image_shape[0],image_shape[1],self.layers_num).to(x.device)
        if(c_0 == None):
            c_0 = torch.tensor(batch_size,self.hidden_channels,image_shape[0],image_shape[1],self.layers_num).to(x.device)

        # output (batch_size,window_size,hidden_channels,image_shape)
        output = x
        hidden,cell = [],[]
        for layer in range(self.layers_num):
            # h,c (batch_size,hidden_channels,image_shape)
            h,c = h_0[:,:,:,:,layer],c_0[:,:,:,:,layer]
            out = []
            for i in range(window_size):
                h,c = self.cell_list[layer](output[:,i],h,c)
                out.append(h)
            output = torch.stack(out,dim=1)
            hidden.append(h)
            cell.append(c)
        
        #hidden cell  (batch_size,window_size,hidden_channels,image_shape)
        hidden = torch.stack(hidden,dim=-1)
        cell = torch.stack(cell,dim=-1)
        return output,hidden,cell

class ConvLstm(nn.Module):
    def __init__(self,in_channels,hidden_channels,in2hi_kernel,hi2hi_kernel,layers_num,pred_len) -> None:
        super().__init__()

        self.pred_len = pred_len

        self.encoder = ConvLstmModule(in_channels,hidden_channels,in2hi_kernel,hi2hi_kernel,layers_num)
        self.decoder = ConvLstmModule(in_channels,hidden_channels,in2hi_kernel,hi2hi_kernel,layers_num)

        self.fc = nn.Conv2d(hidden_channels,1,1)

    def forward(self,x):
        # x (batch_size,window_size,hidden_channels,image_shape)

        (batch_size,window_size,hidden_channels),image_shape = x.shape[0:3],x.shape[3:5]
        
        _,hidden,cell = self.encoder(x)

        
        decoder_input = torch.zeros((batch_size,window_size,hidden_channels,image_shape))
        
        output = []
        for i in range(self.pred_len):
            output,hidden,cell = self.decoder(decoder_input,hidden,cell)
            decoder_input = self.fc(output)
            output.append(decoder_input)

        output = torch.stack(output,dim=1)
        return output

