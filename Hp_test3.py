

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
        kernel_size= [5] * 7,
        padding= [2] * 7,
        stride= [1] * 7,
        nb_filters = [16, 16, 32, 32, 64, 64, 128],
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
    

# Overall metrics (micro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 43.36 %
#     Precision                       : 49.63 %
#     Recall                          : 38.49 %
#   Error rate
#     Error rate (ER)                 : 0.99 
#     Substitution rate               : 0.02 
#     Deletion rate                   : 0.59 
#     Insertion rate                  : 0.37 

#   Class-wise average metrics (macro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 40.88 %
#     Precision                       : 42.88 %
#     Recall                          : 41.79 %
#   Error rate
#     Error rate (ER)                 : 1.16 
#     Deletion rate                   : 0.58 
#     Insertion rate                  : 0.58 
  

#   Class-wise metrics
#   ======================================
#     Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
#     ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
#     Frying       | 94      137   | 38.1%    32.1%    46.8%  | 1.52     0.53     0.99   |
#     Speech       | 1752    1413  | 54.6%    61.1%    49.3%  | 0.82     0.51     0.31   |
#     Vacuum_cle.. | 92      116   | 51.9%    46.6%    58.7%  | 1.09     0.41     0.67   |
#     Alarm_bell.. | 420     283   | 43.2%    53.7%    36.2%  | 0.95     0.64     0.31   |
#     Dishes       | 559     319   | 25.3%    34.8%    19.9%  | 1.17     0.80     0.37   |
#     Blender      | 94      136   | 39.1%    33.1%    47.9%  | 1.49     0.52     0.97   |
#     Running_wa.. | 237     169   | 36.5%    43.8%    31.2%  | 1.09     0.69     0.40   |
#     Cat          | 341     278   | 43.3%    48.2%    39.3%  | 1.03     0.61     0.42   |
#     Electric_s.. | 65      106   | 53.8%    43.4%    70.8%  | 1.22     0.29     0.92   |
#     Dog          | 570     319   | 22.9%    32.0%    17.9%  | 1.20     0.82     0.38   |

