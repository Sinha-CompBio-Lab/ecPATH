import torch
from torch import nn


class MLP_regression(nn.Module):
    def __init__(self, n_inputs, n_hiddens, n_outputs, dropout, bias_init):
        super(MLP_regression, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.Dropout(dropout),
        )
        self.layer1 = nn.Linear(n_hiddens, n_outputs)
        if bias_init is not None:
            self.layer1.bias = bias_init

    ##-------------------------------------
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = torch.mean(x, dim=0)
        return x
