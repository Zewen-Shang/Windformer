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
    def __init__(self, feature_num, img_size, window_size, kernels_per_layer=16):
        super(MultidimConv, self).__init__()
        self.feature = DepthwiseSeparableConv(
            feature_num, feature_num, kernels_per_layer=kernels_per_layer
        )
        self.img = DepthwiseSeparableConv(
            img_size, img_size, kernels_per_layer=kernels_per_layer
        )
        self.window = DepthwiseSeparableConv(
            window_size, window_size, kernels_per_layer=kernels_per_layer
        )
        self.bn_feature = nn.BatchNorm2d(feature_num)
        self.bn_img = nn.BatchNorm2d(img_size)
        self.bn_window = nn.BatchNorm2d(window_size)

    def forward(self, input):
        # input (batch_size,feature_num,img_size,window_size)
        x_feature = self.feature(input)
        x_img = self.img(rearrange(input,"b f i w -> b i f w"))
        x_window = self.window(rearrange(input,"b f i w -> b w f i"))
        
        x_feature = nn.functional.relu(self.bn_feature(x_feature))
        x_img = nn.functional.relu(self.bn_img(x_img))
        x_window = nn.functional.relu(self.bn_window(x_window))
        
        output = torch.cat(
            [
                rearrange(x_feature, "b f i w -> b (f i w)"),
                rearrange(x_img, "b i f w -> b (f i w)"),
                rearrange(x_window, "b w f i -> b (f i w)"),
            ],
            dim=1,
        )
        return output


class Mult_Conv(nn.Module):
    def __init__(self, feature_num, img_shape, window_size, kernels_per_layer=1, hidden_neurons=128):
        super(Mult_Conv, self).__init__()

        self.feature_num = feature_num
        self.img_shape = img_shape
        self.window_size = window_size

        self.multidim = MultidimConv(
            feature_num, img_shape[0] * img_shape[1], window_size, kernels_per_layer=kernels_per_layer,
        )

        self.merge = nn.Linear(
            3 * feature_num * img_shape[0] * img_shape[1] * window_size, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)

    def forward(self, input):
        # input (batch_size,feature_num,img_height,img_weight,window_size)
        batch_size = input.shape[0]
        # x (batch_size,feature_num,img_size,window_size)
        x = input.reshape(batch_size, self.feature_num,
                          self.img_shape[0] * self.img_shape[1], self.window_size,)
        output = self.multidim(x)
        output = nn.functional.relu(self.merge(output))
        output = self.output(output)
        return output
