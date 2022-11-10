
import torch
import numpy as np
import utils

class Temporal_Trend_Identification_Task_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, batch, encoder, train):
        acc = 0
        x, y = utils.temporal_augmentation(batch, window=200, a=0.6, b=0, c=0.6, d=1.5) # augmentation
        c, t = x.shape
        x = np.reshape(x, (c, 1, t))            # changed the shape to put it in Conv1d().
        CrossEL = torch.nn.CrossEntropyLoss()
        pred = encoder.forward(torch.Tensor(x).type(torch.float32).to(self.device))
        y = torch.Tensor(y).to(self.device)
        loss = CrossEL(pred, y)

        _, y = torch.max(y, 1) # return maxvalue position

        if train:
            loss.backward(retain_graph=True)

        _, predicted = torch.max(pred, 1) # return maxvalue position

        acc = (predicted == y).sum().item() #
        acc = acc / c  # acc/(batch*channels*4(augmentation)

        del x
        del y
        return loss, acc