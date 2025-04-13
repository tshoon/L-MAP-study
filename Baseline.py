# CNN 펼쳐보기

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
        sig = self.sigmoid(lin)
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

'''
net:
  dropout: 0.5
  rnn_layers: 2
  n_in_channel: 1
  nclass: 10
  attention: True
  n_RNN_cell: 128
  activation: glu
  rnn_type: BGRU
  kernel_size: [3, 3, 3, 3, 3, 3, 3]
  padding: [1, 1, 1, 1, 1, 1, 1]
  stride: [1, 1, 1, 1, 1, 1, 1]
  nb_filters: [ 16, 32, 64, 128, 128, 128, 128 ]
  pooling: [ [ 2, 2 ], [ 2, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ] ]
  dropout_recurrent: 0
  use_embeddings: False
'''

class CNN(nn.Module):
    def __init__(
        self,
        n_in_channel=1,
        activation="glu",
        conv_dropout=0.5,
        kernel_size=[3] * 7,
        padding=[1] * 7,
        stride=[1] * 7,
        nb_filters=[16, 32, 64, 128, 128, 128, 128],
        pooling=[[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
        normalization="batch",
        **transformer_kwargs
    ):
        super(CNN, self).__init__()

        self.nb_filters = nb_filters

        self.layer1 = nn.Sequential(
            nn.Conv2d(n_in_channel, nb_filters[0], kernel_size[0], stride[0], padding[0]),
            nn.BatchNorm2d(nb_filters[0]) if normalization == "batch" else nn.GroupNorm(1, nb_filters[0]),
            self.get_activation_func(activation, nb_filters[0]),
            nn.Dropout(conv_dropout),
            nn.AvgPool2d(pooling[0])
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filters[0], nb_filters[1], kernel_size[1], stride[1], padding[1]),
            nn.BatchNorm2d(nb_filters[1]) if normalization == "batch" else nn.GroupNorm(1, nb_filters[1]),
            self.get_activation_func(activation, nb_filters[1]),
            nn.Dropout(conv_dropout),
            nn.AvgPool2d(pooling[1])
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(nb_filters[1], nb_filters[2], kernel_size[2], stride[2], padding[2]),
            nn.BatchNorm2d(nb_filters[2]) if normalization == "batch" else nn.GroupNorm(1, nb_filters[2]),
            self.get_activation_func(activation, nb_filters[2]),
            nn.Dropout(conv_dropout),
            nn.AvgPool2d(pooling[2])
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(nb_filters[2], nb_filters[3], kernel_size[3], stride[3], padding[3]),
            nn.BatchNorm2d(nb_filters[3]) if normalization == "batch" else nn.GroupNorm(1, nb_filters[3]),
            self.get_activation_func(activation, nb_filters[3]),
            nn.Dropout(conv_dropout),
            nn.AvgPool2d(pooling[3])
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(nb_filters[3], nb_filters[4], kernel_size[4], stride[4], padding[4]),
            nn.BatchNorm2d(nb_filters[4]) if normalization == "batch" else nn.GroupNorm(1, nb_filters[4]),
            self.get_activation_func(activation, nb_filters[4]),
            nn.Dropout(conv_dropout),
            nn.AvgPool2d(pooling[4])
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(nb_filters[4], nb_filters[5], kernel_size[5], stride[5], padding[5]),
            nn.BatchNorm2d(nb_filters[5]) if normalization == "batch" else nn.GroupNorm(1, nb_filters[5]),
            self.get_activation_func(activation, nb_filters[5]),
            nn.Dropout(conv_dropout),
            nn.AvgPool2d(pooling[5])
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(nb_filters[5], nb_filters[6], kernel_size[6], stride[6], padding[6]),
            nn.BatchNorm2d(nb_filters[6]) if normalization == "batch" else nn.GroupNorm(1, nb_filters[6]),
            self.get_activation_func(activation, nb_filters[6]),
            nn.Dropout(conv_dropout),
            nn.AvgPool2d(pooling[6])
        )

    def get_activation_func(self, name, nOut):
        if name.lower() == "leakyrelu":
            return nn.LeakyReLU(0.2)
        elif name.lower() == "relu":
            return nn.ReLU()
        elif name.lower() == "glu":
            return GLU(nOut)
        elif name.lower() == "cg":
            return ContextGating(nOut)
        else:
            raise NotImplementedError(f"Activation {name} not implemented")
            


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x

    


if __name__ == "__main__":
    model = CNN()  # 입력 채널 1로 가정
    x = torch.randn(48, 1, 626, 128)  # (batch_size, channel, height, width)

    print("Input shape:", x.shape)

    x = model(x)  

    print(f"Output shape: {x.shape}")
    assert x.size(-1) == 128, f"GRU input mismatch: got {x.size(-1)}, expected 128"


#   Overall metrics (micro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 40.97 %
#     Precision                       : 46.22 %
#     Recall                          : 36.79 %
#   Error rate
#     Error rate (ER)                 : 1.03 
#     Substitution rate               : 0.03 
#     Deletion rate                   : 0.60 
#     Insertion rate                  : 0.40 

#   Class-wise average metrics (macro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 38.64 %
#     Precision                       : 39.95 %
#     Recall                          : 40.07 %
#   Error rate
#     Error rate (ER)                 : 1.26 
#     Deletion rate                   : 0.60 
#     Insertion rate                  : 0.66 
  

#   Class-wise metrics
#   ======================================
#     Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
#     ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
#     Frying       | 94      178   | 33.8%    25.8%    48.9%  | 1.91     0.51     1.40   |
#     Cat          | 341     292   | 45.2%    49.0%    41.9%  | 1.02     0.58     0.44   |
#     Running_wa.. | 237     180   | 35.5%    41.1%    31.2%  | 1.14     0.69     0.45   |
#     Speech       | 1752    1348  | 51.5%    59.2%    45.5%  | 0.86     0.54     0.31   |
#     Electric_s.. | 65      100   | 38.8%    32.0%    49.2%  | 1.55     0.51     1.05   |
#     Vacuum_cle.. | 92      115   | 58.0%    52.2%    65.2%  | 0.95     0.35     0.60   |
#     Blender      | 94      146   | 35.8%    29.5%    45.7%  | 1.64     0.54     1.10   |
#     Dog          | 570     335   | 20.3%    27.5%    16.1%  | 1.26     0.84     0.43   |
#     Dishes       | 559     382   | 23.6%    29.1%    19.9%  | 1.29     0.80     0.48   |
#     Alarm_bell.. | 420     286   | 43.9%    54.2%    36.9%  | 0.94     0.63     0.31   |


# student

## scenrio1
### PDSD: 0.33356 / 0.34200
## scenrio2
### PDSD: 0.52908 / 0.54591