import os
import torch
import glob
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataloader2 import concat_dataset
from Temporal_Pretrain_Model import feature_extractor3
from Temporal_Path_Loss import Temporal_Trend_Identification_Task_Loss
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


SC_list = glob.glob('/DataCommon2/wypark/smj_data//Preprocessed_SleepEDF_expanded_remove_long_wake/**')
print(len(SC_list))
sleepedf_list = []
for i in range(len(SC_list)):
  sleepedf_list.extend(glob.glob(SC_list[i]+'/**'))
print(len(sleepedf_list))

SS1_list = glob.glob('/DataCommon2/wypark/smj_data/Preprocessed_MASS1_EDF/**')
print(len(SS1_list))

MASS_list = []
for i in range(len(SS1_list)):
    MASS_list.extend(glob.glob(SS1_list[i]+'/**'))
print(len(MASS_list))

data_dic = {'MASS' : MASS_list, 'Sleep_edf': sleepedf_list}
train_dataset, val_dataset, test_dataset = concat_dataset(data_dic, seq_len = 2000).call() # sequence_length = 2000(10s)

# batch size
batch_size = 180
learning_rate = 0.0005
epochs = 3

trainLoader = DataLoader(train_dataset, num_workers=4, batch_size = batch_size, shuffle=True,pin_memory=True)
valLoader = DataLoader(train_dataset, num_workers=4, batch_size = batch_size, shuffle=True, pin_memory=True)
testLoader = DataLoader(test_dataset, num_workers=4, batch_size = batch_size, shuffle=True, pin_memory=True)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

model = feature_extractor3(2000).to(device)

criterion = Temporal_Trend_Identification_Task_Loss(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



print('----learning---')
loss_tr = []
loss_val = []
acc_tr = []
acc_val = []
loss_mini_tr = []
loss_mini_val = []
acc_mini_tr = []
acc_mini_val = []
print(len(trainLoader))
for epoch in range(epochs):
    loss_ep = 0  # add batch loss in epoch
    acc_ep = 0
    for batch_idx, batch in enumerate(trainLoader):
        # print(batch.shape)
        optimizer.zero_grad()
        loss_batch, acc_batch = criterion.forward(batch, model, train=True)
        # print(batch_idx)
        optimizer.step()
        loss_ep += loss_batch.item()
        acc_ep += acc_batch
        batch_idx += 1
        if batch_idx%100 == 0:
            print('epoch_percent:',batch_idx/len(trainLoader),'loss:',loss_ep/batch_idx, 'acc:',acc_ep/batch_idx)
            loss_mini_tr.append(loss_ep/batch_idx)
            acc_mini_tr.append(acc_ep/batch_idx)

    loss_tr.append((loss_ep) / len(trainLoader))
    acc_tr.append((acc_ep) / len(trainLoader))

    loss_ep_val = 0
    acc_ep_val = 0
    model.eval()
    for batch_idx, batch in enumerate(valLoader):
        loss_batch_val, acc_batch_val = criterion.forward(batch, model, train=False)
        loss_ep_val += loss_batch_val.item()
        acc_ep_val += acc_batch_val
        batch_idx += 1
        if batch_idx % 100 == 0:
            print('loss:', loss_ep_val / batch_idx, 'acc:', acc_ep_val / batch_idx)
            loss_mini_val.append(loss_ep_val / batch_idx)
            acc_mini_val.append(acc_ep_val / batch_idx)

    loss_val.append((loss_ep_val) / len(valLoader))
    acc_val.append((acc_ep_val) / len(valLoader))

    print("epoch : ", epoch, "  train loss : ", loss_tr[epoch], 'train acc : ', acc_tr[epoch], "    val loss : ", loss_val[epoch], 'val acc : ', acc_val[epoch])
    torch.save(model, '/home/wypark/smj/Temporal_pathway/model/10s/'+'Temporal_10s_ep' + str(epoch) + '_.pt')
    model.train()

col = ['loss_tr','loss_val','acc_tr','acc_val']
data = np.array([loss_tr,
                 loss_val,
                 acc_tr,
                 acc_val])
print(data.shape)
data = np.transpose(data)
df = pd.DataFrame(data = data, columns= col)
df.to_excel('/home/wypark/smj/Temporal_pathway/result/10s/'+'Sleepedf+MASS_10s.xlsx', index = False)



plt.plot(range(epochs), loss_tr, color='red')
plt.plot(range(epochs), loss_val, color='blue')
plt.title('Model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('/home/wypark/smj/Temporal_pathway/result/10s/'+'Sleepedf+MASS_loss_10s',bbox_inches = 'tight')
plt.show()

# plt.figure(figsize =(15, 10))
plt.plot(range(epochs), acc_tr, color='red')
plt.plot(range(epochs), acc_val, color='blue')
plt.title('Model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig('/home/wypark/smj/Temporal_pathway/result/10s/'+'Sleepedf+MASS_accuracy_10s',bbox_inches = 'tight')
plt.show()


loss_ep_test = 0
acc_ep_test = 0
for batch_idx, batch in enumerate(testLoader):
    loss_batch_test, acc_batch_test = criterion.forward(batch, model, train=False)
    loss_ep_test += loss_batch_test.item()
    acc_ep_test += acc_batch_test

acc_ep_test = acc_ep_test / len(testLoader)
loss_ep_test = loss_ep_test / len(testLoader)

print('test_acc:',acc_ep_test)
print('test_loss:',loss_ep_test)