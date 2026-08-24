import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, nchans=64, samples=128, F1=4, D=2, kernel_length=64, dropout=0.1):
        super(EEGNet, self).__init__()
        F2 = F1 * D
        self.F2 = F2

        # Conv temporal
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise conv espacial
        self.depthwise = nn.Conv2d(
            F1, F2, kernel_size=(nchans, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2)

        # Pool + Dropout
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop1 = nn.Dropout(dropout)

        # Separable Conv (depthwise + pointwise)
        self.sep_conv_dw = nn.Conv2d(F2, F2, kernel_size=(1, 16), padding=(0, 8), groups=F2, bias=False)
        self.sep_conv_pw = nn.Conv2d(F2, F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.avgpool1(x)
        x = self.drop1(x)

        x = self.sep_conv_dw(x)
        x = self.sep_conv_pw(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.avgpool2(x)
        x = self.drop2(x)

        return x  # Shape: (B, F2, 1, T)

class TCNBlock(nn.Module):
    def __init__(self, input_dim, filters, depth=3, kernel_size=10, dropout=0.3, activation='relu'):
        super(TCNBlock, self).__init__()
        self.depth = depth
        self.activation = getattr(F, activation)

        self.blocks = nn.ModuleList()
        self.res_convs = nn.ModuleList()

        for i in range(depth):
            dilation = 2 ** i
            block = nn.Sequential(
                nn.Conv1d(input_dim if i == 0 else filters, filters, kernel_size, padding='causal', dilation=dilation),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(filters, filters, kernel_size, padding='causal', dilation=dilation),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.blocks.append(block)

            # Si no hay correspondencia de dimensiones, agregamos conv 1x1
            if i == 0 and input_dim != filters:
                self.res_convs.append(nn.Conv1d(input_dim, filters, kernel_size=1))
            else:
                self.res_convs.append(None)

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            res = self.res_convs[i](x) if self.res_convs[i] is not None else x
            out = block(x)
            x = self.activation(out + res)
        return x  # (B, filters, T)

class EEGTCNet(nn.Module):
    def __init__(self, nb_classes, nchans=64, samples=128, layers=3, kernel_s=10,
                 filt=10, dropout=0.3, activation='relu', F1=4, D=2, kernel_length=64, dropout_eeg=0.1):
        super(EEGTCNet, self).__init__()

        self.eegnet = EEGNet(nchans=nchans, samples=samples, F1=F1, D=D,
                             kernel_length=kernel_length, dropout=dropout_eeg)

        F2 = F1 * D
        self.tcn = TCNBlock(input_dim=F2, filters=filt, depth=layers,
                            kernel_size=kernel_s, dropout=dropout, activation=activation)

        self.classifier = nn.Linear(filt, nb_classes)

    def forward(self, x):
        # x shape: (B, 1, nchans, samples)
        x = x.permute(0, 1, 3, 2)  # para simular Permute(3,2,1)
        x = self.eegnet(x)  # output: (B, F2, 1, T)
        x = x.squeeze(2)    # eliminar eje de dimensión
