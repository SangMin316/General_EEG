import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CHB_MIT_dataset(Dataset):
    def __init__(self, files, seq_len, SSL):
        self.files = files
        self.sequence_length = seq_len
        self.SSL = SSL
        # sample을 split해줬을 때 몇개로 split되는지 누적해서 저장, i번째 data를 찾을 때 data_adress 각 값이 기준이 됨
        data_adress = [0]
        ad = 0
        max_value = 0.
        min_value = 0.
        L = seq_len
        for i in range(len(self.files)):
            sample = np.load(files[i])['x']
            c, t = sample.shape
            a = L * int(t / L)

            if t == a:
                t = int(c * t / self.sequence_length)
            else:
                t = int(c * a / self.sequence_length )

            # dataloader2: c --> c*t
            ad += t
            data_adress.append(ad)
            temp_max = sample.max()
            temp_min = sample.min()
            max_value = np.max([max_value, temp_max])
            min_value = np.min([min_value, temp_min])

        self.data_adress = data_adress
        self.max_value = max_value
        self.min_value = min_value

    def preprocessing(self, data):
        if self.SSL:
            data_max = np.max(data, axis=2, keepdims=True)  # max value of each channels
            data_min = np.min(data, axis=2, keepdims=True)  # shape = b,c
            b, c, t = data.shape

            return data / data_max * np.ones((b, c, t)) - (data_max - data_min) * np.ones((b, c, t)) / (
                        self.max_value - self.min_value)
        else:
            data_max = np.max(data, axis=1, keepdims=True)  # max value of each channels
            data_min = np.min(data, axis=1, keepdims=True)  # shape = c,t
            c, t = data.shape

            return data / data_max * np.ones((c, t)) - (data_max - data_min) * np.ones((c, t)) / (
                        self.max_value - self.min_value)

        # load the data shape [batch, 1, L]

    def split_data(self, data):
        L = self.sequence_length
        channels, length = data.shape
        a = L * int(length / L)

        if length == a:
            data = np.reshape(data, (int(length / L * channels), 1, L))

        else:
            data = data[:, :a]
            data = np.reshape(data, (int(a / L * channels), 1, L))
        return data

    def one_hot_encoding(self,y):
        if y == 0.:
          y = np.array([1,0])
        elif y == 1.:
          y = np.array([0,1])
        return y

    def __getitem__(self, index):
        if self.SSL:
            for i in range(len(self.data_adress)):
                if index < self.data_adress[i]:
                    break

            sample = np.load(self.files[i - 1])['x']
            sample = self.split_data(sample)
            sample = self.preprocessing(sample)
            # print(index, i)
            # print(sample.shape)
            # print(index - self.data_adress[i - 1])
            # print(sample[index - self.data_adress[i - 1], :, :].shape)
            return sample[index - self.data_adress[i - 1], :, :]

        else:
            sample = np.load(self.files[index])
            y = self.one_hot_encoding(sample['y'])
            sample = self.preprocessing(sample['x'])
            return {'x': torch.tensor(sample),
                    'y': torch.tensor(y)
                    }

    def __len__(self):
        if self.SSL:
            return self.data_adress[-1]
        else:
            return len(self.files)



class Sleepedf_dataset(Dataset):
    def __init__(self, files, seq_len, SSL):
        self.files = files
        self.sequence_length = seq_len
        self.SSL = SSL
        # sample을 split해줬을 때 몇개로 split되는지 누적해서 저장, i번째 data를 찾을 때 data_adress 각 값이 기준이 됨
        data_adress = [0]
        ad = 0
        max_value = 0.
        min_value = 0.
        L = seq_len

        for i in range(len(self.files)):
            sample = np.load(files[i])['x']
            c, t = sample.shape
            a = L * int(t / L)

            if t == a:
                t = int(c * t / self.sequence_length)
            else:
                t = int(c * a / self.sequence_length)

            # dataloader2: c --> c*t
            t = int(c*t / self.sequence_length)
            ad += t
            data_adress.append(ad)
            temp_max = sample.max()
            temp_min = sample.min()
            max_value = np.max([max_value, temp_max])
            min_value = np.min([min_value, temp_min])

        self.data_adress = data_adress
        self.max_value = max_value
        self.min_value = min_value

    def preprocessing(self, data):
        if self.SSL:
            data_max = np.max(data, axis=2, keepdims=True)  # max value of each channels
            data_min = np.min(data, axis=2, keepdims=True)  # shape = b,c
            b, c, t = data.shape

            return data / data_max * np.ones((b, c, t)) - (data_max - data_min) * np.ones((b, c, t)) / (self.max_value - self.min_value)
        else:
            data_max = np.max(data, axis=1, keepdims=True)  # max value of each channels
            data_min = np.min(data, axis=1, keepdims=True)  # shape = c,t
            c, t = data.shape

            return data / data_max * np.ones((c, t)) - (data_max - data_min) * np.ones((c, t)) / (self.max_value - self.min_value)

    # load the data shape [batch, 1, L]
    def split_data(self, data):
        L = self.sequence_length
        channels, length = data.shape
        a = L * int(length / L)

        if length == a:
            data = np.reshape(data, (int(length/L*channels), 1, L))

        else:
            data = data[:, :a]
            data = np.reshape(data, (int(a/L*channels),1 , L))
        return data

    # unify stage 3,4
    def one_hot_encoding(self,y):
        if y == '1':
          y = np.array([0,1,0,0,0])
        elif y == '2':
          y = np.array([0,0,1,0,0])
        elif y == '3':
          y = np.array([0,0,0,1,0])
        elif y == '4':
          y = np.array([0,0,0,1,0])
        elif y == 'R':
          y = np.array([0,0,0,0,1])
        else: # wake, move
          y = np.array([1,0,0,0,0])
        return y


    def __getitem__(self, index):
        if self.SSL:
            for i in range(len(self.data_adress)):
                if index < self.data_adress[i]:
                    break

            sample = np.load(self.files[i - 1])
            y = self.one_hot_encoding(sample['y'])
            sample = self.split_data(sample['x'])
            sample = self.preprocessing(sample)
            return sample[index - self.data_adress[i - 1], :, :]

        else:
            sample = np.load(self.files[index])
            y = self.one_hot_encoding(sample['y'])
            sample = self.preprocessing(sample['x'])
            return {'x': torch.tensor(sample),
                    'y': torch.tensor(y)
                    }

    def __len__(self):
        if self.SSL:
            return self.data_adress[-1]
        else:
            return len(self.files)


class MASS_dataset(Dataset):
    def __init__(self, files, seq_len, SSL):
        self.files = files
        self.sequence_length = seq_len
        self.SSL = SSL
        # sample을 split해줬을 때 몇개로 split되는지 누적해서 저장, i번째 data를 찾을 때 data_adress 각 값이 기준이 됨
        data_adress = [0]
        ad = 0
        max_value = 0.
        min_value = 0.

        for i in range(len(self.files)):
            sample = np.load(files[i])['x']
            c, t = sample.shape
            t = int(c*t / self.sequence_length)
            ad += t
            data_adress.append(ad)
            temp_max = sample.max()
            temp_min = sample.min()
            max_value = np.max([max_value, temp_max])
            min_value = np.min([min_value, temp_min])

        self.data_adress = data_adress
        self.max_value = max_value
        self.min_value = min_value

    def preprocessing(self, data):
        if self.SSL:
            data_max = np.max(data, axis=2, keepdims=True)  # max value of each channels
            data_min = np.min(data, axis=2, keepdims=True)  # shape = b,c
            b, c, t = data.shape

            return data / data_max * np.ones((b, c, t)) - (data_max - data_min) * np.ones((b, c, t)) / (
                        self.max_value - self.min_value)
        else:
            data_max = np.max(data, axis=1, keepdims=True)  # max value of each channels
            data_min = np.min(data, axis=1, keepdims=True)  # shape = c,t
            c, t = data.shape

            return data / data_max * np.ones((c, t)) - (data_max - data_min) * np.ones((c, t)) / (
                        self.max_value - self.min_value)

    def split_data(self, data):
        L = self.sequence_length
        channels, length = data.shape
        a = L * int(length / L)

        if length == a:
            data = np.reshape(data, (int(length / L * channels), 1, L))

        else:
            data = data[:, :a]
            data = np.reshape(data, (int(a / L * channels), 1, L))
        return data

    def one_hot_encoding(self, y):
        if y == 'Sleep stage 1':
            y = np.array([0, 1, 0, 0, 0])
        elif y == 'Sleep stage 2':
            y = np.array([0, 0, 1, 0, 0])
        elif y == 'Sleep stage 3':
            y = np.array([0, 0, 0, 1, 0])
        elif y == 'Sleep stage R':
            y = np.array([0, 0, 0, 0, 1])
        else: # wake
            y = np.array([1, 0, 0, 0, 0])
        return y

    def __getitem__(self, index):
        if self.SSL:
            for i in range(len(self.data_adress)):
                if index < self.data_adress[i]:
                    break

            sample = np.load(self.files[i - 1])
            y = self.one_hot_encoding(sample['y'])
            sample = self.split_data(sample['x'])
            sample = self.preprocessing(sample)
            return sample[index - self.data_adress[i - 1], :, :]

        else:
            sample = np.load(self.files[index])
            y = self.one_hot_encoding(sample['y'])
            sample = self.preprocessing(sample['x'])
            return {'x': torch.tensor(sample),
                    'y': torch.tensor(y)
                    }

    def __len__(self):
        if self.SSL:
            return self.data_adress[-1]
        else:
            return len(self.files)



class concat_dataset():
    def __init__(self, data_dic, seq_len):
        self.data_dic = data_dic  # data_dic : {'dataset1_name : [dataset1_adress],,,datasetN_name : [datasetN_adress]}
        self.seq_len = seq_len

    def tr_val_te_split(self, data_list):
        train, test = train_test_split(data_list, test_size=0.2,random_state=7)  # , shuffle=True, random_state=34), #stratify=target
        train, val = train_test_split(train, test_size=0.25, random_state= 7)  # , shuffle=True, random_state=34)
        del data_list
        print('split done')
        return train, val, test

    def call(self):
        for name, data_list in self.data_dic.items():
            print(name)
            tr, val, te = self.tr_val_te_split(data_list)

            if name == 'Sleep_edf':
                sleepedf_train_data = Sleepedf_dataset(tr, self.seq_len, SSL=True)
                print('sleep train done')
                sleepedf_val_data = Sleepedf_dataset(val, self.seq_len, SSL=True)
                print('sleep val done')
                sleepedf_test_data = Sleepedf_dataset(te, self.seq_len, SSL=True)
                print('sleep test done')

            # elif name == 'MASS':
            #     MASS_train_data = Sleepedf_dataset(tr, self.seq_len, SSL=True)
            #     print('MASS train done')
            #     MASS_val_data = Sleepedf_dataset(val, self.seq_len, SSL=True)
            #     print('MASS val done')
            #     MASS_test_data = Sleepedf_dataset(te, self.seq_len, SSL=True)
            #     print('MASS test done')

            elif name == "CHB-MIT":
                CHB_MIT_train_data = CHB_MIT_dataset(tr, self.seq_len, SSL=True)
                print('CHB-MIT train done')
                CHB_MIT_val_data = CHB_MIT_dataset(val, self.seq_len, SSL=True)
                print('CHB-MIT val done')
                CHB_MIT_test_data = CHB_MIT_dataset(te, self.seq_len, SSL=True)
                print('CHB-MIT test done')

        # del train_data,val_data, test_data

        # train_dataset = torch.utils.data.ConcatDataset([sleepedf_train_data, MASS_train_data,CHB_MIT_train_data])
        # val_dataset = torch.utils.data.ConcatDataset([sleepedf_test_data, MASS_val_data, CHB_MIT_val_data])
        # test_dataset = torch.utils.data.ConcatDataset([sleepedf_val_data, MASS_test_data, CHB_MIT_val_data])

        train_dataset = torch.utils.data.ConcatDataset([sleepedf_train_data,CHB_MIT_train_data])
        val_dataset = torch.utils.data.ConcatDataset([sleepedf_val_data, CHB_MIT_val_data])
        test_dataset = torch.utils.data.ConcatDataset([sleepedf_test_data, CHB_MIT_test_data])

        return train_dataset, val_dataset, test_dataset