import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from dataloader.dataloader3 import Sleepedf_dataset
from DownstreamTask.SleepEDF_WCE import WCE_weight

import Spectral_Pathway.Spectral_Path_Model
import TIPNet



# load a data adress
SleepEDF_file_list = glob.glob('/DataCommon2/wypark/smj_data/Preprocessed_SleepEDF20/**')
print(len(SleepEDF_file_list))
SleepEDF_list = []
for i in range(len(SleepEDF_file_list)):
    SleepEDF_list.extend(glob.glob(SleepEDF_file_list[i]+'/**'))
print(len(SleepEDF_list))

#dataloader
train, test = train_test_split(SleepEDF_list, test_size=0.2, random_state= 77)
train, val = train_test_split(train, test_size= 0.25, random_state= 77)

train_dataset = Sleepedf_dataset(train,3000,SSL = False)
val_dataset = Sleepedf_dataset(val,3000,SSL = False)
test_dataset = Sleepedf_dataset(test,3000,SSL = False)

#deivce
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())


#learning parameter
epochs = 20
learning_rate = 0.0005
batch_size = 70

trainLoader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 4)
valLoader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers = 4)
testLoader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True, num_workers = 4)

#load pretrain model
# specPath = torch.load('/home/wypark/smj/SSC_SleepEDF/model/Dense/2s/69/0.716sepcPath_ep69.pt')
# tempPath = torch.load('/home/wypark/smj/SSC_SleepEDF/model/Dense/2s/69/0.716tempPath_ep69.pt')
# model = torch.load('/home/wypark/smj/SSC_SleepEDF/model/Dense/2s/69/0.716SleepEDF_DIP_ep69.pt')

specPath = torch.load('/home/wypark/smj/SSC_SleepEDF20/model/DIP_Conv2/2s/0.928_56/sepcPath_ep56.pt') #82
tempPath = torch.load('/home/wypark/smj/SSC_SleepEDF20/model/DIP_Conv2/2s/0.928_56/tempPath_ep56.pt')
model = torch.load('/home/wypark/smj/SSC_SleepEDF20/model/DIP_Conv2/2s/0.928_56/SleepEDF_DIP_ep56.pt')


# model
ep_n = 57
epochs = epochs + ep_n

# loss, optimizer
normedWeights = WCE_weight('soft').to(device)
CrossEL = torch.nn.CrossEntropyLoss(normedWeights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)


print('-----learning-------')
loss_tr = []
loss_val = []
acc_tr = []
acc_val = []
for epoch in range(ep_n,epochs):
    specPath.train()
    tempPath.train()
    model.train()

    loss_ep = 0  # add batch loss in epoch
    acc_ep = 0
    for batch_idx, batch in enumerate(trainLoader):
        'batch[x].shape  = [batch_size, channels, length]'
        optimizer.zero_grad()

        b, c, t = batch['x'].shape
        data = torch.reshape(batch['x'], (b * c, 1, t))  # for Conv1d
        data = torch.Tensor(data).type(torch.float).to(device)

        x1 = specPath.getRep(data)
        x2 = tempPath.embedding(data)

        bc, f, t = x1.shape
        pred = model.forward(x1, x2, b, c, f, t)

        CrossEL = torch.nn.CrossEntropyLoss()
        label = batch['y'].type(torch.float64).to(device)

        loss = CrossEL(pred, label)
        loss.backward(retain_graph=True)
        optimizer.step()

        _, label = torch.max(label, 1)
        _, predicted = torch.max(pred, 1)
        acc = (predicted == label).sum().item()

        acc = acc / batch['x'].shape[0]  # acc/batch
        loss_ep += loss.item()
        acc_ep += acc

    loss_tr.append((loss_ep) / len(trainLoader))
    acc_tr.append((acc_ep) / len(trainLoader))

    loss_ep_val = 0
    acc_ep_val = 0


    with torch.no_grad():
        specPath.eval()
        tempPath.eval()
        model.eval()
        for batch_idx, batch in enumerate(valLoader):
            b, c, t = batch['x'].shape
            data = torch.reshape(batch['x'], (b * c, 1, t))  # for Conv1d
            data = torch.Tensor(data).type(torch.float).to(device)

            x1 = specPath.getRep(data)
            x2 = tempPath.embedding(data)
            bc, f, t = x1.shape
            pred = model.forward(x1, x2, b, c, f, t)

            CrossEL = torch.nn.CrossEntropyLoss()
            label = batch['y'].type(torch.float64).to(device)

            loss = CrossEL(pred, label)

            _, label = torch.max(label, 1)
            _, predicted = torch.max(pred, 1)
            acc = (predicted == label).sum().item()
            acc = acc / batch['x'].shape[0]

            loss_ep_val += loss.item()
            acc_ep_val += acc

        loss_val.append((loss_ep_val) / len(valLoader))
        acc_val.append((acc_ep_val) / len(valLoader))
    print("epoch : ", epoch, "  train loss : ", loss_tr[epoch - ep_n], 'train acc : ', acc_tr[epoch - ep_n], "    val loss : ",
          loss_val[epoch - ep_n], 'val acc : ', acc_val[epoch - ep_n])

    ad = '/home/wypark/smj/SSC_SleepEDF20/model/DIP_Conv2/2s/' + str(acc_val[epoch])[:5] + '_' + str(epoch)
    if not os.path.exists(ad):
        os.makedirs(ad)
    torch.save(model, ad + '/SleepEDF_DIP_ep' + str(epoch) + '.pt')
    torch.save(specPath, ad + '/sepcPath_ep' + str(epoch) + '.pt')
    torch.save(tempPath, ad + '/tempPath_ep' + str(epoch) + '.pt')

#save result
col = ['loss_tr','loss_val','acc_tr','acc_val']
data = np.array([loss_tr,
                 loss_val,
                 acc_tr,
                 acc_val])
print(data.shape)
data = np.transpose(data)
df = pd.DataFrame(data = data, columns= col)
df.to_excel('/home/wypark/smj/SSC_SleepEDF/result/Dense/'+'SleepEDF.xlsx', index = False)



plt.plot(range(epochs), loss_tr, color='red')
plt.plot(range(epochs), loss_val, color='blue')
plt.title('Model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('/home/wypark/smj/SSC_SleepEDF/result/Dense/'+'SleepEDF_loss',bbox_inches = 'tight')
plt.show()

# plt.figure(figsize =(15, 10))
plt.plot(range(epochs), acc_tr, color='red')
plt.plot(range(epochs), acc_val, color='blue')
plt.title('Model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig('/home/wypark/smj/SSC_SleepEDF/result/Dense/'+'SleepEDF_accuracy',bbox_inches = 'tight')
plt.show()

