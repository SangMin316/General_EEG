import os
import torch
import glob
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataloader.dataloader4 import concat_dataset
from Spectral_Path_Loss import StoppedBandPredTaskLoss
# from Spectral_Pathway.Spectral_Path_Model import StoppedBandPathway
import TIPNet
import matplotlib.pyplot as plt


#
SC_list = sorted(glob.glob('/DataCommon2/wypark/smj_data/Preprocessed_SleepEDF20_EEG/**'))
print(len(SC_list))
sleepedf_list = []
for i in range(len(SC_list)): # 20 subjects
  sleepedf_list.extend(glob.glob(SC_list[i]+'/**'))
print(len(sleepedf_list))

CHB_mit = glob.glob('/DataCommon2/wypark/smj_data/same_ratio_CHB_MIT_10s/**')
print(len(CHB_mit))

CHB_list = []
for i in range(len(CHB_mit)):
    CHB_list.extend(glob.glob(CHB_mit[i]+'/**'))
print(len(CHB_list))

CHB_LIST = []
for i in range(len(CHB_list)):
    CHB_LIST.extend(glob.glob(CHB_list[i]+'/**'))


seq_len = 400


data_dic = {'CHB-MIT' : CHB_LIST, 'Sleep_edf': sleepedf_list }
train_dataset, val_dataset, test_dataset = concat_dataset(data_dic, seq_len = seq_len).call() # sequence_length = 2000(10s)

# batch size
batch_size = 1500
learning_rate = 0.0005 #
epochs = 20

# trainLoader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True,pin_memory=True)
# valLoader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, pin_memory=True)
# testLoader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True, pin_memory=True)

trainLoader = DataLoader(train_dataset, num_workers= 4 ,batch_size = batch_size, shuffle=True,pin_memory=True)
valLoader = DataLoader(train_dataset, num_workers= 4, batch_size = batch_size, shuffle=True, pin_memory=True)
testLoader = DataLoader(test_dataset, num_workers= 4, batch_size = batch_size, shuffle=True, pin_memory=True)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

BANDS = [(0.5,4), (4,8), (8,15), (15,30), (30,49.9)]
LABEL = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]

encode_info = [(10, 16, 30),(16, 302, 15),(32, 64, 5)]
sfreq = 200


model = TIPNet.StoppedBandPathway(sfreq).to(device)

criterion = StoppedBandPredTaskLoss(BANDS, LABEL, device=device)

# use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


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

loss_tr = []
loss_val = []
acc_tr = []
acc_val = []

for epoch in range(epochs):
    loss_ep = 0  # add batch loss in epoch
    acc_ep = 0
    # concatdata.getTrain()
    model.train()
    for batch_idx, batch in enumerate(trainLoader):
        optimizer.zero_grad()
        # print(batch.shape)
        loss_batch, acc_batch = criterion.forward(batch, model, sfreq, train=True)
        optimizer.step()
        loss_ep += loss_batch.item()
        acc_ep += acc_batch
        batch_idx += 1
        if batch_idx % 1000 == 0:
            print('epoch_percent:', batch_idx / len(trainLoader), 'loss:', loss_ep / batch_idx, 'acc:',
                  acc_ep / batch_idx)
            loss_mini_tr.append(loss_ep / batch_idx)
            acc_mini_tr.append(acc_ep / batch_idx)
        # print('acc',acc_batch)
        # print('loss:',loss_batch.item())
    loss_tr.append((loss_ep) / len(trainLoader))
    acc_tr.append((acc_ep) / len(trainLoader))


    loss_ep_val = 0
    acc_ep_val = 0
    model.eval()
    with torch.no_grad():
        loss_v = 0
        acc_v = 0
        for batch_idx, batch in enumerate(valLoader):
            loss_batch, acc_batch = criterion.forward(batch, model, sfreq, train=False)
            loss_ep_val += loss_batch.item()
            acc_ep_val += acc_batch
            batch_idx += 1
            if batch_idx % 1000 == 0:
                print('epoch_percent:', batch_idx / len(valLoader), 'loss:', loss_ep / batch_idx, 'acc:',
                      acc_ep / batch_idx)
                loss_mini_val.append(loss_ep / batch_idx)
                acc_mini_val.append(acc_ep / batch_idx)

        loss_val.append((loss_ep_val) / len(valLoader))
        acc_val.append((acc_ep_val) / len(valLoader))
        print("epoch : ", epoch, "  train loss : ", loss_tr[epoch], 'train acc : ', acc_tr[epoch], "    val loss : ",
              loss_val[epoch], 'val acc : ', acc_val[epoch])
        torch.save(model, '/home/wypark/smj/Spectral_Pathway/model/CHBMIT10_SleepEDF20/2s/'+'Spectral_2s_ep' +str(acc_val[epoch])[:5]+ str(epoch) + '.pt')


col = ['loss_tr','loss_val','acc_tr','acc_val']
data = np.array([loss_mini_tr,
                 loss_mini_val,
                 acc_mini_tr,
                 acc_mini_val])
print(data.shape)
data = np.transpose(data)
df = pd.DataFrame(data = data, columns= col)
df.to_excel('/home/wypark/smj/Spectral_Pathway/result/CHBMIT_SleepEDF20/EEG2s/'+'Sleepedf+MASS_2s.xlsx', index = False)




plt.plot(range(epochs), loss_tr, color='red')
plt.plot(range(epochs), loss_val, color='blue')
plt.title('Model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('/home/wypark/smj/Spectral_Pathway/result/CHBMIT10_SleepEDF20/2s/'+'Sleepedf+MASS_loss_2s',bbox_inches = 'tight')
# plt.show()

# plt.figure(figsize =(15, 10))
plt.plot(range(epochs), acc_tr, color='red')
plt.plot(range(epochs), acc_val, color='blue')
plt.title('Model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig('/home/wypark/smj/Spectral_Pathway/result/CHBMIT10_SleepEDF20/2s/'+'Sleepedf+MASS_accuracy_2s',bbox_inches = 'tight')
# plt.show()

model.eval()
with torch.no_grad():
    loss_ep_test = 0
    acc_ep_test = 0
    for batch_idx, batch in enumerate(testLoader):
        loss_batch_test, acc_batch_test = criterion.forward(batch, model, sfreq, train=False)
        loss_ep_test += loss_batch_test.item()
        acc_ep_test += acc_batch_test

acc_ep_test = acc_ep_test / len(testLoader)
loss_ep_test = loss_ep_test / len(testLoader)

print('test_acc:',acc_ep_test)
print('test_loss:',loss_ep_test)