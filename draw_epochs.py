import os
import torch
import matplotlib.pyplot as plt
import numpy as np 


from tensorboard.backend.event_processing import event_accumulator
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

# font = font_manager.FontProperties(fname="./SIMSUN.TTF")





def draw_epochs(filenames,target_map,y_lim):
    plt.figure(figsize=(16,8))
    linewidth = 2
    font_label = FontProperties(fname="./TIMES.TTF",size=33)
    font_tick = FontProperties(fname="./TIMES.TTF",size=28)
    font_legend = FontProperties(fname="./TIMES.TTF",size=28)

    for filename,label,color in zip(filenames,["30min","40min","50min"],["r","g","b"]):
        ea = event_accumulator.EventAccumulator(filename)
        ea.Reload()
        

        result = [item.value for item in ea.scalars.Items("test_MSE")]
        plt.plot(result,linewidth=linewidth,label=label,color=color)

    plt.xlabel("epochs",fontproperties=font_label)
    plt.ylabel("MSE",fontproperties=font_label)

    
    plt.gca().set_ylim(y_lim[0],y_lim[1])
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels(["%.1f"%y for y in y_ticks],fontproperties=font_tick)

    x_ticks = plt.gca().get_xticks()
    plt.gca().set_xticklabels(np.int32(x_ticks),fontproperties=font_tick)

    plt.legend(loc="best",shadow=True,prop=font_legend)
    plt.savefig("./line/epoch_%d.png"%target_map,dpi=300,bbox_inches='tight',pad_inches=0.0)
    plt.show()


def draw_contrast(name_lists,y_lims,mins):

    linewidth = 2
    font_label = FontProperties(fname="./TIMES.TTF",size=50)
    font_tick = FontProperties(fname="./TIMES.TTF",size=40)
    font_legend = FontProperties(fname="./TIMES.TTF",size=35)

    figure = plt.figure(figsize=(36,10))
    ax_indexs = [131,132,133]

    for name_list,y_lim,min,ax_index in zip(name_lists,y_lims,mins,ax_indexs):

        ax = figure.add_subplot(ax_index)
        for filename,label,color in zip(name_list,["[1]","[2]","[1]+[2]","MFDAWSP"],["r","y","g","b"]):
            ea = event_accumulator.EventAccumulator(filename)
            ea.Reload()
            

            result = [item.value for item in ea.scalars.Items("test_MSE")]
            ax.plot(result,linewidth=linewidth,label=label,color=color)

        ax.set_xlabel("epochs",fontproperties=font_label)
        ax.set_ylabel("MSE",fontproperties=font_label)

        ax.set_ylim(y_lim[0],y_lim[1])
        y_ticks = ax.get_yticks()
        ax.set_yticklabels(["%.1f"%y for y in y_ticks],fontproperties=font_tick)

        x_ticks = ax.get_xticks()
        ax.set_xticklabels(np.int32(x_ticks),fontproperties=font_tick)

        ax.legend(loc="best",prop=font_legend)
    plt.savefig("./line/epoch_contrast.png",dpi=300,bbox_inches='tight',pad_inches=0.0)
    plt.show()



draw_epochs(["./tf_dir/CNN_Trans_1_-1_6_AdamW_block1_nodrop/events.out.tfevents.1664275041.tju.899816.0",
"./tf_dir/CNN_Trans_1_-1_8_AdamW_block1_nodrop/events.out.tfevents.1664284376.tju.899816.1",
"./tf_dir/CNN_Trans_1_-1_10_AdamW_block1_nodrop/events.out.tfevents.1664593681.tju.1479826.0"],
1,[0.5,4.0])

draw_epochs(["./tf_dir/CNN_Trans_2_-1_6_AdamW_block1_nodrop/events.out.tfevents.1664289672.tju.908591.0",
"./tf_dir/CNN_Trans_2_-1_8_AdamW_block1_nodrop/events.out.tfevents.1664305454.tju.908591.1",
"./tf_dir/CNN_Trans_2_-1_10_AdamW_block1_nodrop/events.out.tfevents.1664318626.tju.908591.2"],
2,[0.1,1.1])

draw_contrast([[
"./tf_dir/None_Trans_1_-1_6_AdamW_block1_nodrop/events.out.tfevents.1664258382.tju.888606.0",
"./tf_dir/Spatial_Mlp_1_-1_6_AdamW_nodrop/events.out.tfevents.1664363842.tju.946170.0",
"./tf_dir/None_Mlp_1_-1_6_AdamW/events.out.tfevents.1664631246.tju.1488110.0",
"./tf_dir/CNN_Trans_1_-1_6_AdamW_block1_nodrop/events.out.tfevents.1664275041.tju.899816.0"
],
[
"./tf_dir/None_Trans_1_-1_8_AdamW_block1_nodrop/events.out.tfevents.1664344575.tju.934551.0",
"./tf_dir/Spatial_Mlp_1_-1_8_AdamW_nodrop/events.out.tfevents.1664371755.tju.946170.1",
"./tf_dir/None_Mlp_1_-1_8_AdamW/events.out.tfevents.1664640811.tju.1488110.1",
"./tf_dir/CNN_Trans_1_-1_8_AdamW_block1_nodrop/events.out.tfevents.1664284376.tju.899816.1"
],
[
"./tf_dir/None_Trans_1_-1_10_AdamW_block1_nodrop/events.out.tfevents.1664351692.tju.934551.1",
"./tf_dir/Spatial_Mlp_1_-1_10_AdamW_nodrop/events.out.tfevents.1664379941.tju.946170.2",
"./tf_dir/None_Mlp_1_-1_10_AdamW/events.out.tfevents.1664650526.tju.1488110.2",
"./tf_dir/CNN_Trans_1_-1_10_AdamW_block1_nodrop/events.out.tfevents.1664593681.tju.1479826.0"]],
[[0.7,2.0],[1.0,3.0],[1.3,4.0]],[30,40,50])