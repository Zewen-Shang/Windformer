from Conv_Lstm import Conv_Lstm
from Persist import Persist
from Vision_Transformer import Vision_Transformer
from Swin import Swin
from Conv_3D import Conv_3D

import torch

configs = {
    "Conv_Lstm":{
        "in_channels":1,#num_features
        "hidden_channels":3,
        "in2hi_kernel":3,
        "hi2hi_kernel":3,
        "layers_num":3,
        "predict_steps":6
    },
    "Persist":{
        "predict_steps":6
    },
    "Vision_Transformer":{
        "image_shape":[30,20],
        "patch_shape":[5,5],
        "input_steps":6,
        "num_features":1,
        "dim":128,
        "head_num":32,
        "block_num":3
    },
    "Swin":{
        "epochs":100,
        "lr":1e-3,
        "lr_decay":[
            [30,1]
        ],
        "optimizer":torch.optim.AdamW,
        "weight_decay":0,
        "args":{
            "image_size":[30,20],
            "patch_size":[2,2],
            "input_steps":6,
            "predict_steps":6,
            "num_features":5,
            "dim":32,
            "num_heads":4,
            "window_size":[5,5],
            "depth":3
        }

    },
    "Conv_3D":{
        "input_steps":6,
        "predict_steps":6,
        "num_features":5,
        "dim":32,
        "depth":0
    }
}

models = {
    "Conv_Lstm":Conv_Lstm,
    "Persist":Persist,
    "Vision_Transformer":Vision_Transformer,
    "Swin":Swin,
    "Conv_3D":Conv_3D
}