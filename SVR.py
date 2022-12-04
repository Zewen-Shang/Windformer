import numpy as np 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

import time

from dataset import *


@torch.no_grad()
def test_model(model,dataloader,device):
    total_MSE,total_MAE = 0.,0.
    total_num = 0
    for imgs,targets in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device).squeeze()
        output = model(imgs).squeeze()
        mse = nn.MSELoss()(output,targets)
        
        mae = MAE(targets,output)
        total_MSE += mse * len(imgs)
        total_MAE += mae * len(imgs)
        total_num += len(imgs)
    return total_MSE / total_num,total_MAE / total_num


target_map = 1
input_steps,predict_steps = 6,6

dataset = get_dataset(target_map,input_steps,predict_steps,debug=False)
cut_pos = int(0.75 * len(dataset))
dataset_train,dataset_test = dataset[:cut_pos],dataset[cut_pos:]
dataset_train,dataset_test = dataset_norm(dataset_train,dataset_test)

image_size = [30,20]
knn_models = [[0,]*20]*30
mse_mat,mae_mat = np.zeros(image_size),np.zeros(image_size)

start = time.time()
for h in range(image_size[0]):
    for w in range(image_size[1]):
        X_train,Y_train = [item[0][:,:,h,w].squeeze() for item in dataset_train],[item[1][:,h,w].squeeze() for item in dataset_train]
        X_test,Y_test = [item[0][:,:,h,w].squeeze() for item in dataset_test],[item[1][:,h,w].squeeze() for item in dataset_test]

        model = SVR(kernel="linear")
        model.fit(X_train,Y_train)
        knn_models[h][w] = model
        Y_predict = model.predict(X_test)
        mse_mat[h,w],mae_mat[h,w] = MSE_np(Y_predict,Y_test),MAE_np(Y_predict,Y_test)
        print(h,w)



end = time.time()
print("%f s. "%(end-start))

# import pickle
# with open('./model_new/SVR_%d_-1_%d.pkl'%(target_map,predict_steps), 'wb') as f:
#     pickle.dump(model, f)




# np.save("./result/SVR_%d_-1_%d"%(target_map,predict_steps),Y_predict)