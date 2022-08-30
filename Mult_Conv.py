from curses import window
import torch
from torch import nn
from einops import rearrange

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=3,
            padding="same",
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels * kernels_per_layer, out_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultidimConv(nn.Module):
    def __init__(self, height,width, window_size, kernels_per_layer=1):
        super(MultidimConv, self).__init__()
        self.height = DepthwiseSeparableConv(
            height, height, kernels_per_layer=kernels_per_layer
        )
        self.width = DepthwiseSeparableConv(
            width, width, kernels_per_layer=kernels_per_layer
        )
        self.window = DepthwiseSeparableConv(
            window_size, window_size, kernels_per_layer=kernels_per_layer
        )
        self.bn_height = nn.BatchNorm2d(height)
        self.bn_width = nn.BatchNorm2d(width)
        self.bn_window = nn.BatchNorm2d(window_size)

    def forward(self, input):
        # input (batch_size,height,width,window_size)
        x_height = self.height(input)
        x_width = self.width(rearrange(input,"b h w win -> b w h win"))
        x_window = self.window(rearrange(input,"b h w win -> b win h w"))
        
        x_height = nn.functional.relu(self.bn_height(x_height))
        x_width = nn.functional.relu(self.bn_width(x_width))
        x_window = nn.functional.relu(self.bn_window(x_window))
        
        output = torch.cat(
            [
                rearrange(x_height, "b h w win -> b (h w win)"),
                rearrange(x_width, "b w h win -> b (h w win)"),
                rearrange(x_window, "b win h w -> b (h w win)"),
            ],
            dim=1,
        )
        return output


class Mult_Conv(nn.Module):
    def __init__(self, feature_num, img_shape, window_size,hidden_neurons, kernels_per_layer=1, ):
        super(Mult_Conv, self).__init__()

        assert(feature_num == 1)

        self.img_shape = img_shape
        self.window_size = window_size

        self.multidim = MultidimConv(
            img_shape[0],img_shape[1], window_size, kernels_per_layer=kernels_per_layer,
        )

        self.merge = nn.Linear(
            3 * img_shape[0] * img_shape[1] * window_size, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)

    def forward(self, input):
        # input (batch_size,feature_num,img_height,img_weight,window_size)
        batch_size = input.shape[0]
        # x (batch_size,height,width,window_size)
        x = input.reshape(batch_size,
                          self.img_shape[0],self.img_shape[1], self.window_size)
        output = self.multidim(x)
        output = nn.functional.relu(self.merge(output))
        output = self.output(output)
        return output
