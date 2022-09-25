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
    def __init__(self, window_size, feature_num, img_shape, hidden_channels, depth) -> None:
        super().__init__()
        self.window_size, self.feature_num, self.img_shape = window_size, feature_num, img_shape
        self.hidden_channels, self.depth = hidden_channels, depth
        self.conv0 = nn.Conv2d(feature_num, hidden_channels, 1, padding="same")
        self.blocks = nn.ModuleList(
            [CNN_Block(hidden_channels, hidden_channels) for i in range(depth)]
        )
        self.conv1 = nn.Conv2d(hidden_channels, feature_num, 1, padding="same")

    def forward(self, input):
        # input (batch_size,window_size,feature_num,img_height,img_weight)
        batch_size = input.shape[0]
        x = input.reshape(batch_size * self.window_size,
                          self.feature_num, self.img_shape[0], self.img_shape[1])
        x = self.conv0(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x.reshape(batch_size, self.window_size, self.feature_num,
                      self.img_shape[0] * self.img_shape[1])
        x = x.permute(0, 1, 3, 2)
        return x


class Spatial_Attn(nn.Module):
    def __init__(self, window_size, feature_num, drive_num, hidden_size, device="cuda") -> None:
        super().__init__()
        self.feature_num = feature_num
        self.drive_num = drive_num
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.device = device

        self.encode_attn = nn.Linear(
            feature_num * window_size + 2*hidden_size, 1)
        self.softmax = nn.Softmax(-1)
        # ??
        self.lstm = nn.LSTM(input_size=self.drive_num * feature_num,
                            hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, input):
        # input (batch_size,window_size,drive_num,feature_num)
        batch_size = input.shape[0]
        # input (batch_size,drive_num,feature_num,window_size)
        input = input.permute(0, 2, 3, 1)
        self.hidden_states = torch.zeros(
            (batch_size, self.window_size+1, self.hidden_size), device=self.device)
        self.cell_states = torch.zeros(
            (batch_size, self.window_size+1, self.hidden_size), device=self.device)

        for t in range(self.window_size):
            # h_0,s_0 (1,batch_size,hiden_size)
            h_0, s_0 = self.hidden_states[:, t:t+1].permute(
                1, 0, 2), self.cell_states[:, t:t+1].permute(1, 0, 2)
            # hidden_state (batch_size,drive_num,hidden_state)
            hidden_state = self.hidden_states[:,
                                              t:t+1].repeat((1, self.drive_num, 1))
            cell_state = self.cell_states[:,
                                          t:t+1].repeat((1, self.drive_num, 1))
            # x (batch_size,drive_num,feature_num * window_size)
            x = input.reshape(batch_size, self.drive_num,
                              self.feature_num*self.window_size)
            attn_in = torch.cat((hidden_state, cell_state, x), dim=2)
            # e,a (batch_size,drive_num)
            e = self.encode_attn(attn_in).squeeze()
            a = self.softmax(e)
            # x_hat (batch_size,drive_num * feature_num)
            x_hat = (a.unsqueeze(
                2) * input[:, :, :, t]).reshape(batch_size, self.drive_num * self.feature_num)
            _, (h_0, s_0) = self.lstm(x_hat.unsqueeze(
                1), (h_0.contiguous(), s_0.contiguous()))
            self.hidden_states[:, t+1], self.cell_states[:,
                                                         t+1] = h_0.squeeze(), s_0.squeeze()

        # output (batch_size,window_size,hidden_size)
        return self.hidden_states[:, 1:]



class Spatial_Mlp(nn.Module):
    def __init__(self, feature_num, img_shape, window_size, hidden_channels, depth, hidden_size) -> None:
        super().__init__()

        self.feature_num, self.window_size, self.hidden_size = feature_num, window_size, hidden_size
        self.hidden_channels, self.depth = hidden_channels, depth
        self.img_shape = img_shape

        # self.spatial_attn = Spatial_Attn(window_size,feature_num,img_shape,patch_shape,spatial_dim,spatial_head)
        self.cnn = CNN(window_size, feature_num, img_shape,
                       hidden_channels=hidden_channels, depth=depth)
        self.spatial_attn = Spatial_Attn(window_size=window_size, feature_num=feature_num,
                                   drive_num=img_shape[0] * img_shape[1], hidden_size=hidden_size)

        self.linear = nn.Linear(hidden_size,1)

    def forward(self, input):
        # input (batch_size,feature_num,img_height,img_weight,window_size)
        batch_size = input.shape[0]
        # x (batch_size,window_size,feature_num,img_height,img_weight)
        x = input.permute(0, 4, 1, 2, 3)

        # x (batch_size,window_size,drive_num,hidden_channels)
        x = self.cnn(x)

        # x (batch_size,window_size,hidden_size)
        x = self.spatial_attn(x)

        # x (batch_size,1)
        x = self.linear(x[:,-1,:])

        return x
