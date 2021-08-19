import torch.nn as nn
from .binarized_modules import  BinarizeLinear

__all__ = ['generic_bnn']

class GenericBNN(nn.Module):
    def __init__(self, num_classes=10):
        super(GenericBNN, self).__init__()
        self.fc1 = BinarizeLinear(784, 100)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = BinarizeLinear(100, 100)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, num_classes)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        return self.logsoftmax(x)


def generic_bnn(**kwargs):
  num_classes = kwargs.get('num_classes', 10)
  return GenericBNN(num_classes=num_classes)
