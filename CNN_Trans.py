from turtle import forward
import torch
import torch.nn as nn

class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding="same")
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv(input)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_steps, num_features, img_shape, hidden_channels, depth) -> None:
        super().__init__()
        self.input_steps, self.num_features, self.img_shape = input_steps, num_features, img_shape
        self.hidden_channels, self.depth = hidden_channels, depth
        self.conv0 = nn.Conv2d(num_features, hidden_channels, 1, padding="same")
        self.blocks = nn.ModuleList(
            [CNN_Block(hidden_channels, hidden_channels) for i in range(depth)]
        )
        self.conv1 = nn.Conv2d(hidden_channels, num_features, 1, padding="same")

    def forward(self, input):
        # input (batch_size,input_steps,num_features,img_height,img_weight)
        batch_size = input.shape[0]
        x = input.reshape(batch_size * self.input_steps,
                          self.num_features, self.img_shape[0], self.img_shape[1])
        x = self.conv0(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x.reshape(batch_size, self.input_steps, self.num_features,
                      self.img_shape[0] * self.img_shape[1])
        x = x.permute(0, 1, 3, 2)
        return x


class Spatial_Attn(nn.Module):
    def __init__(self, input_steps, num_features, drive_num, hidden_size, device="cuda") -> None:
        super().__init__()
        self.num_features = num_features
        self.drive_num = drive_num
        self.input_steps = input_steps
        self.hidden_size = hidden_size
        self.device = device

        self.encode_attn = nn.Linear(
            num_features * input_steps + 2*hidden_size, 1)
        self.softmax = nn.Softmax(-1)
        # ??
        self.lstm = nn.LSTM(input_size=self.drive_num * num_features,
                            hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, input):
        # input (batch_size,input_steps,drive_num,num_features)
        batch_size = input.shape[0]
        # input (batch_size,drive_num,num_features,input_steps)
        input = input.permute(0, 2, 3, 1)
        self.hidden_states = torch.zeros(
            (batch_size, self.input_steps+1, self.hidden_size), device=self.device)
        self.cell_states = torch.zeros(
            (batch_size, self.input_steps+1, self.hidden_size), device=self.device)

        for t in range(self.input_steps):
            # h_0,s_0 (1,batch_size,hiden_size)
            h_0, s_0 = self.hidden_states[:, t:t+1].permute(
                1, 0, 2), self.cell_states[:, t:t+1].permute(1, 0, 2)
            # hidden_state (batch_size,drive_num,hidden_state)
            hidden_state = self.hidden_states[:,
                                              t:t+1].repeat((1, self.drive_num, 1))
            cell_state = self.cell_states[:,
                                          t:t+1].repeat((1, self.drive_num, 1))
            # x (batch_size,drive_num,num_features * input_steps)
            x = input.reshape(batch_size, self.drive_num,
                              self.num_features*self.input_steps)
            attn_in = torch.cat((hidden_state, cell_state, x), dim=2)
            # e,a (batch_size,drive_num)
            e = self.encode_attn(attn_in).reshape(batch_size,self.drive_num)
            a = self.softmax(e)
            # x_hat (batch_size,drive_num * num_features)
            x_hat = (a.unsqueeze(
                2) * input[:, :, :, t]).reshape(batch_size, self.drive_num * self.num_features)
            _, (h_0, s_0) = self.lstm(x_hat.unsqueeze(
                1), (h_0.contiguous(), s_0.contiguous()))
            self.hidden_states[:, t+1], self.cell_states[:,
                                                         t+1] = h_0.reshape(batch_size,self.hidden_size), s_0.reshape(batch_size,self.hidden_size)

        # output (batch_size,input_steps,hidden_size)
        return self.hidden_states[:, 1:]


class LSTM(nn.Module):
    def __init__(self, input_size, input_steps, hidden_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.input_steps = input_steps
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.gate = nn.GLU()
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, input):
        #input (batch_size,input_steps,num_features)
        batch_size = input.shape[0]
        # x (batch_size,input_steps,hidden_size)
        x, (h_n, c_n) = self.lstm(input)

        gate_x = self.gate(x.repeat((1, 1, 2)))
        x = x + gate_x
        x = self.layernorm(x)
        # x (batch_size,input_steps,hidden_size)
        return x


def get_pos_emb(input_steps, dim, device="cuda", temperature=1e4, dtype=torch.float):
    pos = torch.arange(0, input_steps, device=device)

    omega = torch.arange(dim//2, device=device)/(dim/2-1)
    omega = 1. / temperature ** omega
    pos_embed = pos[:, None] * omega[None, :]

    pos_embed = torch.cat((pos_embed.cos(), pos_embed.sin()), dim=1)
    return pos_embed.to(dtype)


class Multi_Head(nn.Module):
    def __init__(self, dim, head_num, drop_prob) -> None:
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scale = self.head_dim ** (-0.5)
        assert(type(self.head_dim) == int)

        self.layernorm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3*dim)
        self.attn_drop = nn.Dropout(drop_prob)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_prob)

    def forward(self, input):
        # input (batch_size,input_steps+1,hidden_state)
        batch_size, num_seq = input.shape[0], input.shape[1]
        assert(self.dim == input.shape[2])
        x = input
        qkv = self.qkv(x).reshape(batch_size, num_seq,
                                  3, self.head_num, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = q @ k.transpose(-1, -2) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        output = (attn @ v).reshape(batch_size, num_seq, self.dim)
        output = self.proj_drop(self.proj(output))
        return output


class Trans_Block(nn.Module):
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

class CNN_Trans(nn.Module):
    def __init__(self, num_features, img_shape, input_steps, hidden_channels, depth, hidden_size, temporal_dim, temporal_head,trans_num) -> None:
        super().__init__()

        self.num_features, self.input_steps, self.hidden_size = num_features, input_steps, hidden_size
        self.hidden_channels, self.depth = hidden_channels, depth
        self.temporal_head = temporal_head
        self.temporal_dim = temporal_dim
        self.img_shape = img_shape

        # self.spatial_attn = Spatial_Attn(input_steps,num_features,img_shape,patch_shape,spatial_dim,spatial_head)
        self.cnn = CNN(input_steps, num_features, img_shape,
                       hidden_channels=hidden_channels, depth=depth)
        self.spatial_attn = Spatial_Attn(input_steps=input_steps, num_features=num_features,
                                   drive_num=img_shape[0] * img_shape[1], hidden_size=hidden_size)

        self.lstm = LSTM(input_size=hidden_size,
                         input_steps=input_steps, hidden_size=hidden_size)

        self.linear0 = nn.Linear(hidden_size,temporal_dim)
        self.cls = torch.rand((1, 1, temporal_dim), device="cuda")
        # pos_embed (input_steps + 1,hidden_state)
        self.pos_embed = get_pos_emb(input_steps=input_steps, dim=temporal_dim)

        self.trans_blocks = nn.ModuleList(
            [Trans_Block(dim=temporal_dim,head=temporal_head) for i in range(trans_num)]
        )

        self.linear1 = nn.Linear(temporal_dim, 1)

    def forward(self, input):
        # input (batch_size,num_features,img_height,img_weight,input_steps)
        batch_size = input.shape[0]
        # x (batch_size,input_steps,num_features,img_height,img_weight)
        x = input.permute(0, 4, 1, 2, 3)

        # x (batch_size,input_steps,drive_num,hidden_channels)
        x = self.cnn(x)

        # x (batch_size,input_steps,hidden_size)
        x = self.spatial_attn(x)

        # x (batch_size,input_steps,hidden_size)
        x = self.lstm(x)

        # x (batch_size,input_steps,temporal_dim)
        x = self.linear0(x)

        # x (batch_size,input_steps,temporal_dim)
        # x = torch.cat((x, self.cls.repeat((batch_size, 1, 1))), dim=1)
        x = x + self.pos_embed.reshape(1, self.input_steps,
                                       self.temporal_dim).repeat((batch_size, 1, 1))

        for block in self.trans_blocks:
            x = block(x)
        # x (batch_size,1,hidden_size)
        x = x[:, -1]

        # x = nn.functional.relu(x)
        x = self.linear1(x)

        return x
