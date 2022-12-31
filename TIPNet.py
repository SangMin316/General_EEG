import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
StoppedBandPathway classes
'''

class Encoder(nn.Module):
    def __init__(self, fs):
        super(Encoder, self).__init__()
        # spectral layer means spectral convolution
        # self.bac_layer is consist of several SeparableConv2d, which plays the role of temporal separable convolution
        # convolution layer are initiated by xavier_uniform initization
        # Input are Normalized by self.bn(=torch.nn.BatchNorm2d)
        # [batch, electrode, length] -> [batch, electrode, Feature]

        self.fs = fs
        self.activation = nn.LeakyReLU()

        self.spectral_layer = nn.Conv1d(1, 10, int(self.fs / 2), padding="same")

        self.conv1t = nn.Conv1d(10, 16, 30, padding='same')
        self.conv2t = nn.Conv1d(16, 32, 15, padding='same')
        self.conv3t = nn.Conv1d(32, 64, 5, padding='same')

        torch.nn.init.xavier_uniform_(self.spectral_layer.weight)
        torch.nn.init.xavier_uniform_(self.conv1t.weight)
        torch.nn.init.xavier_uniform_(self.conv2t.weight)
        torch.nn.init.xavier_uniform_(self.conv3t.weight)

    def forward(self, x):
        x = self.activation(self.spectral_layer(x))
        x = self.activation(self.conv1t(x))
        x = self.activation(self.conv2t(x))
        x = self.activation(self.conv3t(x))
        return x


# Linear layer for SSL classification
class Head_NN(nn.Module):
    def __init__(self):
        super(Head_NN, self).__init__()

        self.linear = nn.Linear(64, 5)
        self.dropout = nn.Dropout(0.5)

        self.softmax = torch.nn.Softmax()
        torch.nn.init.xavier_uniform_(self.linear.weight)

        self.bn = nn.BatchNorm1d(64)
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = torch.mean(x, axis=2)  # Global average pooling into temporal dimension
        # x = self.flatten(x)
        x = self.linear(self.dropout(x))
        x = self.softmax(x)
        return x


class StoppedBandPathway(nn.Module):
    def __init__(self, fs):
        super(StoppedBandPathway, self).__init__()
        self.encoder = Encoder(fs)
        self.pretrain = Head_NN()

    def forward(self, x):
        x = self.encoder(x)
        x = self.pretrain(x)
        return x

    def getRep(self, x):
        x = self.encoder(x)
        return x



'''
Temporal Dynamics Pathway classe
'''
class feature_extractor3(nn.Module):
    def __init__(self, seq_len):
        super(feature_extractor3, self).__init__()
        self.channels = 1  # we use only single channel

        # Activation functions
        self.activation = nn.LeakyReLU()
        # self.bn = nn.BatchNorm1d(1)

        # self.conv2t = nns.SeparableConv1d(16,32,10,padding ='same') (in_channels, out_channels, kernel size,,,)

        self.softmax = nn.Softmax()
        self.conv1t = nn.Conv1d(1, 10, 30, padding='same')  # in_channels, out_channels, kernel_size,
        self.conv1s = nn.Conv1d(10, 10, self.channels)
        self.conv2t = nn.Conv1d(10, 20, 15, padding='same')
        self.conv2s = nn.Conv1d(20, 20, self.channels)
        self.conv3t = nn.Conv1d(20, 34, 5, padding='same')
        self.conv3s = nn.Conv1d(34, 34, self.channels)

        # Flatteninig
        self.flatten = nn.Flatten()

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Decision making
        self.Linear = nn.Linear(int(64 * seq_len), 4)  # feature = 64, target = 4

        # xavier initialization
        torch.nn.init.xavier_uniform_(self.conv1t.weight)
        torch.nn.init.xavier_uniform_(self.conv2t.weight)
        torch.nn.init.xavier_uniform_(self.conv3t.weight)
        torch.nn.init.xavier_uniform_(self.conv1s.weight)
        torch.nn.init.xavier_uniform_(self.conv2s.weight)
        torch.nn.init.xavier_uniform_(self.conv3s.weight)

    def embedding(self, x):
        x = self.activation(self.conv1t(x))
        f1 = self.activation(self.conv1s(x))

        x = self.activation(self.conv2t(x))
        f2 = self.activation(self.conv2s(x))

        x = self.activation(self.conv3t(x))
        f3 = self.activation(self.conv3s(x))

        # multi-scale feature representation by exploiting intermediate features
        feature = torch.cat([f1, f2, f3], dim=1)

        return feature

    def classifier(self, feature):
        # Flattening, dropout, mapping into the decision nodes
        feature = self.flatten(feature)
        feature = self.dropout(feature)
        y_hat = self.softmax(self.Linear(feature))
        return y_hat

    def forward(self, x):
        feature = self.embedding(x)
        y_hat = self.classifier(feature)
        return y_hat




#filter 수 up
class feature_extractor2(nn.Module):
    def __init__(self, seq_len):
        super(feature_extractor2, self).__init__()
        self.channels = 1  # we use only single channel

        # Activation functions
        self.activation = nn.LeakyReLU()

        self.softmax = nn.Softmax()
        self.conv1t = nn.Conv1d(1, 20, 30, padding='same')  # in_channels, out_channels, kernel_size,
        self.conv1s = nn.Conv1d(20, 20, self.channels)
        self.conv2t = nn.Conv1d(20, 40, 15, padding='same')
        self.conv2s = nn.Conv1d(40, 40, self.channels)
        self.conv3t = nn.Conv1d(40, 68, 5, padding='same')
        self.conv3s = nn.Conv1d(68, 68, self.channels)

        # Flatteninig
        self.flatten = nn.Flatten()

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Decision making
        self.Linear = nn.Linear(int(128 * seq_len), 4)  # feature = 64, target = 4

        # xavier initialization
        torch.nn.init.xavier_uniform_(self.conv1t.weight)
        torch.nn.init.xavier_uniform_(self.conv2t.weight)
        torch.nn.init.xavier_uniform_(self.conv3t.weight)
        torch.nn.init.xavier_uniform_(self.conv1s.weight)
        torch.nn.init.xavier_uniform_(self.conv2s.weight)
        torch.nn.init.xavier_uniform_(self.conv3s.weight)

    def embedding(self, x):
        x = self.activation(self.conv1t(x))
        f1 = self.activation(self.conv1s(x))

        x = self.activation(self.conv2t(x))
        f2 = self.activation(self.conv2s(x))

        x = self.activation(self.conv3t(x))
        f3 = self.activation(self.conv3s(x))

        # multi-scale feature representation by exploiting intermediate features
        feature = torch.cat([f1, f2, f3], dim=1)

        return feature

    def classifier(self, feature):
        # Flattening, dropout, mapping into the decision nodes
        feature = self.flatten(feature)
        feature = self.dropout(feature)
        y_hat = self.softmax(self.Linear(feature))
        return y_hat

    def forward(self, x):
        feature = self.embedding(x)
        y_hat = self.classifier(feature)
        return y_hat


'''
Feature Encoder & StatisticianModule
 - Using Feature
 - only 1 Pathway
  1) spectral
  2) spatial
  3) temporal
 - have 2 Pathway
  1) spectral & spatial
  2) spatial & temporal
  3) temporal & spectral
 - have all Pathway
'''


# filter*2, conv + dense
class StatisticianPipe1(nn.Module):
    def __init__(self, dense,channels,classes):
        super(StatisticianPipe1, self).__init__()
        self.classes = classes

        self.conv1 = nn.Conv2d(64, 128, (channels, 1))  # spatial convolution
        self.conv2 = nn.Conv2d(64, 128, (channels, 1))

        self.dropout = nn.Dropout(0.5)
        # self.dropout = nn.Dropout(0.4) sleepEDF20

        self.activation = nn.LeakyReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.c_dense = nn.Linear(dense, 128 * 2 * 2)  # 64*2*2

        self.gap_pwconv = nn.Conv1d(128*2, dense, 1)
        self.gvp_pwconv = nn.Conv1d(128*2, dense, 1)

        self.layer1 = nn.Linear(int(dense/2), int(dense/2))
        self.layer2 = nn.Linear(int(dense/2), int(dense/2))
        self.layer3 = nn.Linear(int(dense/2), self.classes)


        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.c_dense.weight)
        torch.nn.init.xavier_uniform_(self.gap_pwconv.weight)
        torch.nn.init.xavier_uniform_(self.gvp_pwconv.weight)
        # torch.nn.init.xavier_uniform_(self.fullconnect.weight)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)



    def GAPGVP(self, x1, x2, b, c, f, t):
        x1 = torch.reshape(x1, (b, c, f, t))
        x2 = torch.reshape(x2, (b, c, f, t))
        x1 = torch.permute(x1, (0,2,1,3))
        x2 = torch.permute(x2, (0,2,1,3))

        x1 = self.conv1(x1)
        x2 = self.conv2(x2) #x2.shape = [b,f,1,t]

        f_GAP = torch.cat((F.adaptive_avg_pool2d(x1, (1, 1)).squeeze(), F.adaptive_avg_pool2d(x2, (1, 1)).squeeze()),
                          axis=1)
        # f_GAP.shape = [b,2*f],  F.adaptive_avg_pool2d(x1, (1, 1)).squeeze() <--- [b,f]

        f_GVP = torch.cat((torch.var(x1.view(x1.size(0), x1.size(1), -1), dim=2),
                           torch.var(x2.view(x2.size(0), x2.size(1), -1), dim=2)), axis=1)
        # x1.view(x1.size(0), x1.size(1), -1) <-- [batch, feature, channel*time]
        # f_GAP.shape = [b,2*f]

        del x1
        del x2
        return f_GAP, f_GVP

    def forward(self, x1, x2, b, c, f, t):
        f_GAP, f_GVP = self.GAPGVP(x1, x2, b, c, f, t)
        #f_GAP.shape = f_GVP.shape = [b,2*f]
        c = self.softmax(self.c_dense(torch.cat((f_GAP, f_GVP), axis=1)))
        # c.shape = [b,2*2*f]  2*2*f = dense

        # [batch, gap, 1] -> [batch, 1, dense] -> [batch, dense]
        f_GAP_d = self.gap_pwconv(f_GAP.unsqueeze(dim=-1)).squeeze()
        f_GVP_d = self.gvp_pwconv(f_GVP.unsqueeze(dim=-1)).squeeze()
        #f_GAP_d.shape = [b,dense]

        f_GAP_dd = torch.sum(c * f_GAP_d, dim=1)
        f_GVP_dd = torch.sum(c * f_GVP_d, dim=1)
        #f_GAP_dd.shape = [b]

        ALN = torch.div(torch.sub(f_GAP.T, f_GAP_dd), f_GVP_dd).T
        #ALN.shape = [b, 2 * f]
        y_hat = self.activation(self.layer1(ALN))
        y_hat = self.dropout(y_hat)
        y_hat = self.activation(self.layer2(y_hat))
        y_hat = self.dropout(y_hat)
        y_hat = self.softmax(self.layer3(y_hat))
        # y_hat = self.sigmoid(self.layer3(y_hat))


        return y_hat

