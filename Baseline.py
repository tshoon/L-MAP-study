# CNN 풀어헤치기

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
        kernel_size=[3, 3, 3, 3, 3, 3, 3, 3],
        padding=[1, 1, 1, 1, 1, 1, 1, 1],
        stride=[1, 1, 1, 1, 1, 1, 1, 1],
        nb_filters=[64, 64, 32, 32, 16, 16],
        pooling=[(1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)],
        normalization="batch",
        **transformer_kwargs
    ):

        super(CNN, self).__init__()


        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        self.layer1 = nn.Sequential(
            nn.Conv2d(n_in_channel, nb_filters[0], kernel_size[0], stride[0], padding[0]),
            nn.BatchNorm2d(nb_filters[0]) if normalization == "batch" else nn.GroupNorm(1, nb_filters[0]),
            self.get_activation_func("relu", nb_filters[0]),
            nn.Dropout(conv_dropout),
            nn.AvgPool2d(pooling[0])
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filters[0], nb_filters[1], kernel_size[1], stride[1], padding[1]),
            nn.BatchNorm2d(nb_filters[1]) if normalization == "batch" else nn.GroupNorm(1, nb_filters[1]),
            self.get_activation_func("relu", nb_filters[1]),
            nn.Dropout(conv_dropout),
            nn.AvgPool2d(pooling[1])

        )        

        self.layer3 = nn.Sequential(
            nn.Conv2d(nb_filters[1], nb_filters[2], kernel_size[2], stride[2], padding[2]),
            nn.BatchNorm2d(nb_filters[2]) if normalization == "batch" else nn.GroupNorm(1, nb_filters[2]),
            self.get_activation_func("relu", nb_filters[2]),
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
    
def test_noramlization(x):

    bn = nn.BatchNorm2d(64)

    mean_before = x[0, 0].mean().item()
    std_before = x[0, 0].std().item()

    x_bn = bn(x)

    mean_after = x_bn[0, 0].mean().item()
    std_after = x_bn[0, 0].std().item()

    print(f"[정규화 전] Channel 0 평균: {mean_before:.4f}, 표준편차: {std_before:.4f}")
    print(f"[정규화 후] Channel 0 평균: {mean_after:.4f}, 표준편차: {std_after:.4f}")



if __name__ == "__main__":
    model = CNN(n_in_channel=1)  # 입력 채널 1로 가정
    x = torch.randn(1, 1, 128, 862)  # (batch_size, channel, height, width)

    x = model(x)  # 모델이 알아서 layer1 ~ layer3 다 실행
    print("Output shape:", x.shape)


## sumary

# [전체 평균 (Micro-average)]

# - F1 Score (Micro):     41.78%
# - Precision (Micro):    47.78%
# - Recall (Micro):       37.12%

# [클래스별 평균 (Macro-average)]

# - F1 Score (Macro):     42.42%
# - Precision (Macro):    44.33%
# - Recall (Macro):       42.78%




## train_sed

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
#     F-measure (F1)                  : 41.78 %
#     Precision                       : 47.78 %
#     Recall                          : 37.12 %
#   Error rate
#     Error rate (ER)                 : 1.01 
#     Substitution rate               : 0.02 
#     Deletion rate                   : 0.61 
#     Insertion rate                  : 0.38 

#   Class-wise average metrics (macro-average)
#   ======================================
#   F-measure
#     F-measure (F1)                  : 42.42 %
#     Precision                       : 44.33 %
#     Recall                          : 42.78 %
#   Error rate
#     Error rate (ER)                 : 1.12 
#     Deletion rate                   : 0.57 
#     Insertion rate                  : 0.55 
  

#   Class-wise metrics
#   ======================================
#     Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
#     ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
#     Dishes       | 559     287   | 26.0%    38.3%    19.7%  | 1.12     0.80     0.32   |
#     Speech       | 1752    1371  | 49.5%    56.4%    44.1%  | 0.90     0.56     0.34   |
#     Blender      | 94      126   | 45.5%    39.7%    53.2%  | 1.28     0.47     0.81   |
#     Dog          | 570     370   | 22.3%    28.4%    18.4%  | 1.28     0.82     0.46   |
#     Electric_s.. | 65      82    | 55.8%    50.0%    63.1%  | 1.00     0.37     0.63   |
#     Frying       | 94      156   | 35.2%    28.2%    46.8%  | 1.72     0.53     1.19   |
#     Alarm_bell.. | 420     322   | 46.4%    53.4%    41.0%  | 0.95     0.59     0.36   |
#     Vacuum_cle.. | 92      110   | 64.4%    59.1%    70.7%  | 0.78     0.29     0.49   |
#     Cat          | 341     276   | 42.5%    47.5%    38.4%  | 1.04     0.62     0.43   |
#     Running_wa.. | 237     182   | 36.8%    42.3%    32.5%  | 1.12     0.68     0.44   |
