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


class CNN(nn.Module):
    def __init__(
        self,
        n_in_channel,
        activation="relu",
        conv_dropout=0,
        kernel_size=[3, 3, 3],
        padding=[1, 1, 1],
        stride=[1, 1, 1],
        nb_filters=[64, 64, 64],
        pooling=[(1, 4), (1, 4), (1, 4)],
        normalization="batch",
        **transformer_kwargs
    ):

        super(CNN, self).__init__()



        self.layer1 = nn.Sequential(
            nn.Conv2d(n_in_channel, nb_filters[0], kernel_size[0], stride[0], padding[0]),
            nn.BatchNorm2d(self.nb_filters[0]) if normalization == "batch" else nn.GroupNorm(1, nb_filters[0]),
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
            


    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
    
    def get_activation_func(self, activ, nOut):
        if activ.lower() == "leakyrelu":
            return nn.LeakyReLU(0.2)
        elif activ.lower() == "relu":
            return nn.ReLU()
        elif activ.lower() == "glu":
            return GLU(nOut)
        elif activ.lower() == "cg":
            return ContextGating(nOut)



if __name__ == "__main__":
    model = CNN(n_in_channel=1)  # 입력 채널 1로 가정
    x = torch.randn(1, 1, 128, 862)  # (batch_size, channel, height, width)

    print("Input shape:", x.shape)

    x = model(x)  

    assert x.size(-1) == 128, f"GRU input mismatch: got {x.size(-1)}, expected 128"


#  Overall metrics (micro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 43.26 %
#     Precision                       : 48.94 %
#     Recall                          : 38.75 %
#   Error rate
#     Error rate (ER)                 : 0.99 
#     Substitution rate               : 0.02 
#     Deletion rate                   : 0.59 
#     Insertion rate                  : 0.38 

#   Class-wise average metrics (macro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 40.99 %
#     Precision                       : 43.26 %
#     Recall                          : 41.29 %
#   Error rate
#     Error rate (ER)                 : 1.17 
#     Deletion rate                   : 0.59 
#     Insertion rate                  : 0.58 
  

#   Class-wise metrics
#   ======================================
#     Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
#     ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
#     Blender      | 94      110   | 40.2%    37.3%    43.6%  | 1.30     0.56     0.73   |
#     Vacuum_cle.. | 92      119   | 53.1%    47.1%    60.9%  | 1.08     0.39     0.68   |
#     Cat          | 341     286   | 44.0%    48.3%    40.5%  | 1.03     0.60     0.43   |
#     Alarm_bell.. | 420     279   | 49.8%    62.4%    41.4%  | 0.84     0.59     0.25   |
#     Frying       | 94      181   | 32.7%    24.9%    47.9%  | 1.97     0.52     1.45   |
#     Dishes       | 559     270   | 22.2%    34.1%    16.5%  | 1.15     0.84     0.32   |
#     Electric_s.. | 65      81    | 50.7%    45.7%    56.9%  | 1.11     0.43     0.68   |
#     Speech       | 1752    1448  | 53.1%    58.7%    48.5%  | 0.86     0.51     0.34   |
#     Running_wa.. | 237     202   | 38.7%    42.1%    35.9%  | 1.14     0.64     0.49   |
#     Dog          | 570     369   | 25.3%    32.2%    20.9%  | 1.23     0.79     0.44   |

