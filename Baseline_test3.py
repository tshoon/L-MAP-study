
# kernel_size = 9

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
        n_in_channel=1,
        activation="glu",
        conv_dropout=0.5,
        kernel_size=[9] * 7,
        padding=[4] * 7,
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

        # x = x.squeeze(-1)              # (B, 128, 168)
        # x = x.permute(0, 2, 1)         # (B, 168, 128) → ✅ GRU에 넣을 수 있는 형태

        return x

    


if __name__ == "__main__":
    model = CNN()  
    x = torch.randn(48, 1, 626, 128)  # (batch_size, channel, height, width)
    print("Input shape:", x.shape)

    x = model(x)  
    print(f"Output shape: {x.shape}")

    # RuntimeError: input.size(-1) must be equal to input_size. Expected 128,
    assert x.size(-1) == 128, f"GRU input mismatch: got {x.size(-1)}, expected 128"

#   Overall metrics (micro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 42.33 %
#     Precision                       : 48.52 %
#     Recall                          : 37.55 %
#   Error rate
#     Error rate (ER)                 : 1.00 
#     Substitution rate               : 0.02 
#     Deletion rate                   : 0.60 
#     Insertion rate                  : 0.38 

#   Class-wise average metrics (macro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 41.45 %
#     Precision                       : 43.56 %
#     Recall                          : 41.07 %
#   Error rate
#     Error rate (ER)                 : 1.13 
#     Deletion rate                   : 0.59 
#     Insertion rate                  : 0.54 
  

#   Class-wise metrics
#   ======================================
#     Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
#     ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
#     Cat          | 341     306   | 41.1%    43.5%    39.0%  | 1.12     0.61     0.51   |
#     Dishes       | 559     330   | 25.2%    33.9%    20.0%  | 1.19     0.80     0.39   |
#     Electric_s.. | 65      71    | 51.5%    49.3%    53.8%  | 1.02     0.46     0.55   |
#     Blender      | 94      121   | 42.8%    38.0%    48.9%  | 1.31     0.51     0.80   |
#     Frying       | 94      145   | 36.0%    29.7%    45.7%  | 1.63     0.54     1.09   |
#     Alarm_bell.. | 420     282   | 41.0%    51.1%    34.3%  | 0.99     0.66     0.33   |
#     Dog          | 570     329   | 20.5%    28.0%    16.1%  | 1.25     0.84     0.42   |
#     Speech       | 1752    1341  | 53.5%    61.7%    47.2%  | 0.82     0.53     0.29   |
#     Vacuum_cle.. | 92      97    | 64.6%    62.9%    66.3%  | 0.73     0.34     0.39   |
#     Running_wa.. | 237     247   | 38.4%    37.7%    39.2%  | 1.26     0.61     0.65   |



# student

## scenrio1
### PDSD: 0.33341 / 0.34193
## scenrio2
### PDSD: 0.58499 / 0.59924