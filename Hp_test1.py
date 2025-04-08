# 시간 영역은 최대한 건들지 않도록 함(RNN에서 처리하기 위해)
# Conv Layer 구조 강화 (깊이 증가 & 채널 수 조절)
# 노이즈를 학습시킬 수도 있지만, kernel 영역을 키워서 더 많은 패턴을 학습하게끔 함
# layer의 개수에 맞게 pooling를 서서히 진행하도록 설정

# layer 3 -> 7 
# kernel_size 3 -> 5
# nb_filter 서서히 감소
# pooling 천천히 진행 및 시간 축도 pooling 약간 pooling 해봄 

# 목표
# 전반적인 모델 성능 향상
# 과적합 방지 (nb_filter 점차 축소)


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
        nb_filters = [64, 64, 64, 64, 32, 32, 16],
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
    


#  Overall metrics (micro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 42.61 %
#     Precision                       : 49.12 %
#     Recall                          : 37.62 %
#   Error rate
#     Error rate (ER)                 : 0.99 
#     Substitution rate               : 0.02 
#     Deletion rate                   : 0.60 
#     Insertion rate                  : 0.37 

#   Class-wise average metrics (macro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 43.50 %
#     Precision                       : 46.34 %
#     Recall                          : 43.36 %
#   Error rate
#     Error rate (ER)                 : 1.08 
#     Deletion rate                   : 0.57 
#     Insertion rate                  : 0.52 
  

#   Class-wise metrics
#   ======================================
#     Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
#     ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
#     Blender      | 94      91    | 46.5%    47.3%    45.7%  | 1.05     0.54     0.51   |
#     Vacuum_cle.. | 92      121   | 64.8%    57.0%    75.0%  | 0.82     0.25     0.57   |
#     Cat          | 341     291   | 44.3%    48.1%    41.1%  | 1.03     0.59     0.44   |
#     Electric_s.. | 65      72    | 58.4%    55.6%    61.5%  | 0.88     0.38     0.49   |
#     Frying       | 94      170   | 37.1%    28.8%    52.1%  | 1.77     0.48     1.29   |
#     Dishes       | 559     230   | 25.3%    43.5%    17.9%  | 1.05     0.82     0.23   |
#     Dog          | 570     384   | 25.8%    32.0%    21.6%  | 1.24     0.78     0.46   |
#     Running_wa.. | 237     204   | 38.1%    41.2%    35.4%  | 1.15     0.65     0.51   |
#     Speech       | 1752    1364  | 49.9%    57.0%    44.4%  | 0.89     0.56     0.33   |
#     Alarm_bell.. | 420     308   | 44.8%    52.9%    38.8%  | 0.96     0.61     0.35   |

