import glob
from dataloader.dataloader3 import Sleepedf_dataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np


# SleepEDF_file_list = glob.glob('/DataCommon2/wypark/Preprocessed_SleepEDFx/smj_data/**')
# print(len(SleepEDF_file_list))
# SleepEDF_list = []
# for i in range(len(SleepEDF_file_list)):
#     SleepEDF_list.extend(glob.glob(SleepEDF_file_list[i]+'/**'))
# print(len(SleepEDF_list))
#
# #dataloader
# train_dataset = Sleepedf_dataset(SleepEDF_list,3000,SSL = False)
#
# batch_size = 128
#
# trainLoader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 4)
#
# a = torch.zeros([1, 5])
# for batch_idx, batch in enumerate(trainLoader):
#     a = a + torch.sum(batch['y'],0)
#
# print(a)

# without long
# tensor([[40183., 21522., 69132., 13039., 25835.]])



### result ###
def WCE_weight(mode):
    # Nsamples = [40183., 21522., 69132., 13039., 25835.] # without long wake stage

    Nsamples = [285561., 21522., 69132., 13039., 25835.]
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
