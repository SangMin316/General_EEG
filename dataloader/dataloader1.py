import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


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

        for i in range(len(self.files)):
            sample = np.load(files[i])['x']
            c, t = sample.shape
            t = int(t / self.sequence_length) # Calculation of how many sequences a sample splits into.
            # ex) smple length = 6000, sequence_length = 1000, 6000/1000 = 6. when load a data, load to the same sample for the first, second, and sixth data
            ad += t
            data_adress.append(ad)  # refer to data_adress, we search a sample file without having to load it into memory.
            temp_max = sample.max()
            temp_min = sample.min()
            max_value = np.max([max_value, temp_max]) # max value in all samples
            min_value = np.min([min_value, temp_min]) # min value in all samples

        self.data_adress = data_adress
        self.max_value = max_value
        self.min_value = min_value

    def preprocessing(self, data):
        # When SSL, since the data is split into several sequences, the data shape is [n,c,t].
        if self.SSL:
            data_max = np.max(data, axis=2, keepdims=True)  # max value of each channels
            data_min = np.min(data, axis=2, keepdims=True)  # data_min.shape = [n,c]
            n, c, t = data.shape

            # use scaling used by BENDR
            return data / data_max * np.ones((n, c, t)) - (data_max - data_min) * np.ones((n, c, t)) / (self.max_value - self.min_value)
        else:
            data_max = np.max(data, axis=1, keepdims=True)  # max value of each channels
            data_min = np.min(data, axis=1, keepdims=True)  # data_min.shape = [c]
            c, t = data.shape

            return data / data_max * np.ones((c, t)) - (data_max - data_min) * np.ones((c, t)) / (self.max_value - self.min_value)

    def split_data(self, data):
        L = self.sequence_length
        channels, length = data.shape
        a = L * int(length / L)


        if length == a:
            data = np.reshape(data, (int(length / L), channels, L))

        # When the data is not accurately divided into regular sequences, we just cut front part.
        else:
            data = data[:, :a]
            data = np.reshape(data, (int(a / L), channels, L))
        return data

    def one_hot_encoding(self,y):
        if y == '1':
          y = np.array([0,1,0,0,0,0])
        elif y == '2':
          y = np.array([0,0,1,0,0,0])
        elif y == '3':
          y = np.array([0,0,0,1,0,0])
        elif y == '4':
          y = np.array([0,0,0,0,1,0])
        elif y == 'R':
          y = np.array([0,0,0,0,0,1])
        else: # wake, move
          y = np.array([1,0,0,0,0,0])
        return y


    def __getitem__(self, index):
        if self.SSL:
            #search which file should be selected when loading the i-th data
            for i in range(len(self.data_adress)):
                if index < self.data_adress[i]:
                    break

            sample = np.load(self.files[i - 1])
            y = self.one_hot_encoding(sample['y'])
            sample = self.split_data(sample['x'])
            sample = self.preprocessing(sample)
            return sample[index - self.data_adress[i - 1], :, :] # ex, index = 1020, i = 5, data_adress[i-1] = 1000, --> samle[20,:,:]

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
            t = int(t / self.sequence_length)
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
            data_min = np.min(data, axis=2, keepdims=True)  # shape = n,c
            n, c, t = data.shape

            return data / data_max * np.ones((n, c, t)) - (data_max - data_min) * np.ones((n, c, t)) / (
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
            data = np.reshape(data, (int(length / L), channels, L))

        else:
            data = data[:, :a]
            data = np.reshape(data, (int(a / L), channels, L))
        return data

    def one_hot_encoding(self, y):
        if y == 'Sleep stage W':
            y = np.array([1, 0, 0, 0, 0])
        elif y == 'Sleep stage 1':
            y = np.array([0, 1, 0, 0, 0])
        elif y == 'Sleep stage 2':
            y = np.array([0, 0, 1, 0, 0])
        elif y == 'Sleep stage 3':
            y = np.array([0, 0, 0, 1, 0])
        elif y == 'Sleep stage R':
            y = np.array([0, 0, 0, 0, 1])
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
        self.data_dic = data_dic  # data_dic : {'dataset1_name : [dataset1_adress,,,],,,, datasetN_name : [datasetN_adress]}
        self.seq_len = seq_len

    def tr_val_te_split(self, data_list):
        train, test = train_test_split(data_list, test_size=0.2)
        train, val = train_test_split(train, test_size=0.25)  # train : val : test  = 6 : 2 : 2
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

            elif name == 'MASS':
                MASS_train_data = Sleepedf_dataset(tr, self.seq_len, SSL=True)
                print('MASS train done')
                MASS_val_data = Sleepedf_dataset(val, self.seq_len, SSL=True)
                print('MASS val done')
                MASS_test_data = Sleepedf_dataset(te, self.seq_len, SSL=True)
                print('MASS test done')


        # del train_data,val_data, test_data

        train_dataset = torch.utils.data.ConcatDataset([sleepedf_train_data, MASS_train_data])
        val_dataset = torch.utils.data.ConcatDataset([sleepedf_test_data, MASS_val_data])
        test_dataset = torch.utils.data.ConcatDataset([sleepedf_val_data, MASS_test_data])

        return train_dataset, val_dataset, test_dataset