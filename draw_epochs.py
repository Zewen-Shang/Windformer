import os
import torch
import matplotlib.pyplot as plt
import numpy as np 


from tensorboard.backend.event_processing import event_accumulator
from matplotlib import font_manager

font = font_manager.FontProperties(fname="./SIMSUN.TTF")

label_size = 10
tick_size = 20
linewidth = 1



def draw_epochs(filenames,target_map,y_lim):
    plt.cla()
    for filename,label,color in zip(filenames,["30min","40min","50min"],["r","g","b"]):
        ea = event_accumulator.EventAccumulator(filename)
        ea.Reload()
        

        result = [item.value for item in ea.scalars.Items("test_MSE")]
        plt.plot(result,linewidth=linewidth,label=label,color=color)

        plt.xlabel("epochs",fontproperties=font,fontsize=tick_size)
        plt.ylabel("MSE",fontproperties=font,fontsize=tick_size)

        
        # plt.gca().set_xticks(np.linspace(0,end_pos-start_pos,11))
        # plt.gca().set_xticklabels(np.linspace(start_pos*5,end_pos*5,11).astype(np.int))
    plt.gca().set_ylim(y_lim[0],y_lim[1])
    plt.legend(loc="best",shadow=True,fontsize=label_size)
    plt.savefig("./line/epoch_%d.jpg"%target_map,dpi=500)
    plt.show()




draw_epochs(["./tf_dir/CNN_Trans_1_-1_6/events.out.tfevents.1662095802.tju.303260.0",
"./tf_dir/CNN_Trans_1_-1_8/events.out.tfevents.1662105067.tju.310970.0",
"./tf_dir/CNN_Trans_1_-1_10/events.out.tfevents.1662114605.tju.318694.0"],
1,[0.5,4.0])

draw_epochs(["./tf_dir/CNN_Trans_2_-1_6/events.out.tfevents.1662389066.tju.559561.0",
"./tf_dir/CNN_Trans_2_-1_8/events.out.tfevents.1662433531.tju.587798.0",
"./tf_dir/CNN_Trans_2_-1_10/events.out.tfevents.1662454990.tju.587798.5"],
2,[0.3,2])