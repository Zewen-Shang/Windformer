from CNN import CNN
from CNN_Trans import CNN_Trans
from Dual_Trans import Dual_Trans
from Mult_Conv import Mult_Conv
from conformer import Conformer
from lstm import LSTM
# from lstm_Seq import LSTM_Seq
# from resnet import ResNet
# from resnet_MF import ResNet_Fig_MF
# from vit import Vit
# from vit_Seq import Vit_Seq
from dualattn import DualAttn
from dualattn_Seq import DualAttn_Seq
from MSTAN_Seq import MSTAN_Seq
from MSTAN import MSTAN

args = {
    "window_size":6,
    "predict_steps":6,
    "ResNet":{
        "depth":2,
        "drop_prob":0.1
    },
    "ResNet_Fig_MF":{
        "feature_num":6,
        "window_size":6,
        "img_shape":[30,20],
        "depth":2,
        "drop_prob":0.2
    },
    "Conformer":{
        "res_depth":16,
        "image_shape":[30,20],
        "patch_shape":[5,5],
        "in_channels":3,
        "res_out_channels":16,
        "dim":32,
        "mlp_dim":64,
        "vit_depth":24,
        "num_heads":8,
        "drop_prob":0.1          
    },
    "Vit":{
        "img_shape":[30,20],
        "patch_shape":[5,5],
        "in_channels":3,
        "dim":32,
        "mlp_dim":64,
        "depth":24,
        "num_heads":8,
        "drop_prob":0.1
    },
    "Vit_Seq":{
        "feature_num":6,
        "window_size":6,
        "drive_num":106,
        "dim":32,
        "mlp_dim":64,
        "depth":24,
        "num_heads":8,
        "drop_prob":0.1
    },
    "DualAttn":{
        "drive_num":600,
        "window_size":3,
        "hidden_size":128,
        "target_pos":[23,15]
    },
    "DualAttn_Seq":{
        "feature_num":6,
        "window_size":6,
        "hidden_size":16,
        "drive_num":106,
        "target_pos":50
    },
    "LSTM":{
        "feature_num":1,
        "window_size":6,
        "img_shape":[30,20],
        "hidden_size":4,
        "target_pos":[15,10]
    },
    "LSTM_Seq":{
        "feature_num":6,
        "window_size":6,
        "drive_num":106,
        "hidden_size":16
    },
    "MSTAN":{
        "feature_num":1,
        "img_shape":[30,20],
        "window_size":6,
        "spatial_dim":8,
        "spatial_head":1,
        "patch_size":[2,2],
        "hidden_size":32,
        "temporal_dim":32,#==hidden_size
        "temporal_head":4,
    },
    "MSTAN_Seq":{
        "feature_num":6,
        "window_size":6,
        "drive_num":106,
        "hidden_size":32,
        "head_num":8
    },
    "Dual_Trans":{
        "feature_num":1,
        "img_shape":[30,20],
        "window_size":6,
        "spatial_dim":128,
        "spatial_head":1,
        "patch_shape":[2,2],
        "hidden_size":64,
        "temporal_dim":64,#==hidden_size
        "temporal_head":8,
    },
    "CNN_Trans":{
        "feature_num":1,
        "img_shape":[30,20],
        "window_size":6,
        "hidden_channels":2,
        "depth":1,
        "hidden_size":32,
        "temporal_dim":32,#==hidden_size
        "temporal_head":4,
    },
    "Mult_Conv":{
        "feature_num":1,
        "img_shape":[30,20],
        "window_size":6,
        "hidden_neurons":16
    },
    "CNN":{
        "feature_num":1,
        "img_shape":[30,20],
        "window_size":6,
        "hidden_neurons":16
    },
}

models = {
    "Conformer":Conformer,
    # "ResNet":ResNet,
    # "ResNet_Fig_MF":ResNet_Fig_MF,
    # "Vit":Vit,
    # "Vit_Seq":Vit_Seq,
    "DualAttn":DualAttn,
    "DualAttn_Seq":DualAttn_Seq,
    "LSTM":LSTM,
    # "LSTM_Seq":LSTM_Seq,
    "MSTAN":MSTAN,
    "MSTAN_Seq":MSTAN_Seq,
    "Dual_Trans":Dual_Trans,
    "CNN_Trans":CNN_Trans,
    "Mult_Conv":Mult_Conv,
    "CNN":CNN
}