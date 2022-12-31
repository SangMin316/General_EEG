import glob
from dataloader.dataloader4 import CHB_MIT_dataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
#
# CHB_mit = glob.glob('/DataCommon2/wypark/smj_data/Preprocessed_CHB_MIT_10s/**')
# print(len(CHB_mit))
#
# CHB_list = []
# for i in range(len(CHB_mit)):
#     CHB_list.extend(glob.glob(CHB_mit[i]+'/**'))
# print(len(CHB_list))
#
# CHB_LIST = []
# for i in range(len(CHB_list)):
#     CHB_LIST.extend(glob.glob(CHB_list[i]+'/**'))
#
# # #dataloader
# train_dataset = CHB_MIT_dataset(CHB_LIST,200,SSL = False)
#
# batch_size = 512
#
# trainLoader = DataLoader(train_dataset, batch_size = batch_size)
#
# a = torch.zeros([1, 2])
# for batch_idx, batch in enumerate(trainLoader):
#     a = a + torch.sum(batch['y'],0)
#     print(batch['x'].shape)
#
# print(a)

# without long
# tensor([[40183., 21522., 69132., 13039., 25835.]])
# tensor([[8846., 4423.]])



### result ###
def WCE_weight(mode):
    # Nsamples = [65729., 1040.]
    Nsamples = [8846., 4423.]
    Nsamples = [3200., 400.]

    if mode == 'soft':
        weights = [1 - (x / sum(Nsamples)) for x in Nsamples]
    elif mode == 'hard':
        weights = [(sum(Nsamples)/(10*x) ) for x in Nsamples]
    print('weights of WCE:',weights)
    return torch.FloatTensor(weights)

#
# # soft
# x = [1,2,3,4,5]
# y = WCE_weight('soft')
# plt.bar(x,y)
# plt.xticks(x, labels=['wake', 'N1', 'N2', 'N3(N4)', 'REM'])
# plt.xlabel('sleep stage')
# plt.ylabel('weight')
# plt.title('weight of soft WCE')
# plt.savefig('SleepEDF_soft_WCE.png') # your path
#
#
# # hard
# x = [1,2,3,4,5]
# y = WCE_weight('hard')
# plt.bar(x,y)
# plt.xticks(x, labels=['wake', 'N1', 'N2', 'N3(N4)', 'REM'])
# plt.xlabel('sleep stage')
# plt.ylabel('weight')
# plt.title('weight of hard WCE')
# plt.savefig('SleepEDF_hard_WCE.png')
