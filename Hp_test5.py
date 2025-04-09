

import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class CNN(nn.Module):
    def __init__(
        self,
        n_in_channel,
        activation="relu",
        conv_dropout=0,
        kernel_size= [5] * 10,
        padding= [2] * 10,
        stride= [1] * 10,
        nb_filters = [16, 16, 32, 32, 64, 64, 128, 128, 256, 256],
        pooling= [(1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (1, 2), (1, 2)],
        normalization="batch",
        **transformer_kwargs
    ):
        """
            Initialization of CNN network s

        Args:
            n_in_channel: int, number of input channel
            activation: str, activation function
            conv_dropout: float, dropout
            kernel_size: kernel size
            padding: padding
            stride: list, stride
            nb_filters: number of filters
            pooling: list of tuples, time and frequency pooling
            normalization: choose between "batch" for BatchNormalization and "layer" for LayerNormalization.
        """
        super(CNN, self).__init__()

        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module(
                "conv{0}".format(i),
                nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]),
            )
            if normalization == "batch":
                cnn.add_module(
                    "batchnorm{0}".format(i),
                    nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
                )
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, nOut))

            if activ.lower() == "leakyrelu":
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(nOut))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        # 128x862x64
        for i in range(len(nb_filters)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module(
                "pooling{0}".format(i), nn.AvgPool2d(pooling[i])
            )  # bs x tframe x mels

        self.cnn = cnn

    def forward(self, x):
        """
        Forward step of the CNN module

        Args:
            x (Tensor): input batch of size (batch_size, n_channels, n_frames, n_freq)

        Returns:
            Tensor: batch embedded
        """
        # conv features
        x = self.cnn(x)
        return x
    

# Event based metrics (onset-offset)
# ========================================
#   Evaluated length                  : 10459.12 sec
#   Evaluated files                   : 1168 
#   Evaluate onset                    : True 
#   Evaluate offset                   : True 
#   T collar                          : 200.00 ms
#   Offset (length)                   : 20.00 %

#   Overall metrics (micro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 43.46 %
#     Precision                       : 49.40 %
#     Recall                          : 38.80 %
#   Error rate
#     Error rate (ER)                 : 0.99 
#     Substitution rate               : 0.02 
#     Deletion rate                   : 0.59 
#     Insertion rate                  : 0.37 

#   Class-wise average metrics (macro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 42.26 %
#     Precision                       : 44.09 %
#     Recall                          : 42.70 %
#   Error rate
#     Error rate (ER)                 : 1.14 
#     Deletion rate                   : 0.57 
#     Insertion rate                  : 0.57 
  

#   Class-wise metrics
#   ======================================
#     Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
#     ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
#     Blender      | 94      146   | 40.8%    33.6%    52.1%  | 1.51     0.48     1.03   |
#     Dog          | 570     318   | 24.8%    34.6%    19.3%  | 1.17     0.81     0.36   |
#     Dishes       | 559     338   | 26.5%    35.2%    21.3%  | 1.18     0.79     0.39   |
#     Frying       | 94      146   | 36.7%    30.1%    46.8%  | 1.62     0.53     1.09   |
#     Alarm_bell.. | 420     307   | 48.7%    57.7%    42.1%  | 0.89     0.58     0.31   |
#     Cat          | 341     297   | 43.9%    47.1%    41.1%  | 1.05     0.59     0.46   |
#     Vacuum_cle.. | 92      112   | 63.7%    58.0%    70.7%  | 0.80     0.29     0.51   |
#     Running_wa.. | 237     195   | 39.8%    44.1%    36.3%  | 1.10     0.64     0.46   |
#     Electric_s.. | 65      80    | 45.5%    41.2%    50.8%  | 1.22     0.49     0.72   |
#     Speech       | 1752    1379  | 52.1%    59.2%    46.6%  | 0.86     0.53     0.32   |


