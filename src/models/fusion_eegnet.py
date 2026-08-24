import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The author of this model is Karel Roots and was published along with the paper titled 
"Fusion Convolutional Neural Network for Cross-Subject EEG Motor Imagery Classification"

BUT THE IMPLEMENTATION IN PYTORCH IS FROM Bruno Zorzet 
 """

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, axis):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)


class EEGNetFusion(nn.Module):
    def __init__(self, nb_classes, nchans=64, samples=128,
                 dropoutRate=0.5, norm_rate=0.25, cpu=False):
        super().__init__()

        self.input_shape = (1, nchans, samples)
        conv_filters = (1, 64)
        conv_filters2 = (1, 96)
        conv_filters3 = (1, 128)

        depth_kernel = (nchans, 1)
        pool1 = (1, 4)
        pool2 = (1, 8)
        sep_kernel1 = (1, 8)
        sep_kernel2 = (1, 16)
        sep_kernel3 = (1, 32)

        F1, F1_2, F1_3 = 8, 16, 32
        F2, F2_2, F2_3 = 16, 32, 64
        D = D2 = D3 = 2

        # Primer rama
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=conv_filters, padding='same', bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=depth_kernel,
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(pool1),
            nn.Dropout(dropoutRate)
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(F1 * D, F2, sep_kernel1, padding='same', axis=1),
            nn.ELU(),
            nn.AvgPool2d(pool2),
            nn.Dropout(dropoutRate)
        )

        # Segunda rama
        self.block3 = nn.Sequential(
            nn.Conv2d(1, F1_2, kernel_size=conv_filters2, padding='same', bias=False),
            nn.BatchNorm2d(F1_2),
            nn.Conv2d(F1_2, F1_2 * D2, kernel_size=depth_kernel,
                      groups=F1_2, bias=False),
            nn.BatchNorm2d(F1_2 * D2),
            nn.ELU(),
            nn.AvgPool2d(pool1),
            nn.Dropout(dropoutRate)
        )
        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(F1_2 * D2, F2_2, sep_kernel2, padding='same', axis=1),
            nn.ELU(),
            nn.AvgPool2d(pool2),
            nn.Dropout(dropoutRate)
        )

        # Tercera rama
        self.block5 = nn.Sequential(
            nn.Conv2d(1, F1_3, kernel_size=conv_filters3, padding='same', bias=False),
            nn.BatchNorm2d(F1_3),
            nn.Conv2d(F1_3, F1_3 * D3, kernel_size=depth_kernel,
                      groups=F1_3, bias=False),
            nn.BatchNorm2d(F1_3 * D3),
            nn.ELU(),
            nn.AvgPool2d(pool1),
            nn.Dropout(dropoutRate)
        )
        self.block6 = nn.Sequential(
            DepthwiseSeparableConv(F1_3 * D3, F2_3, sep_kernel3, padding='same', axis=1),
            nn.ELU(),
            nn.AvgPool2d(pool2),
            nn.Dropout(dropoutRate)
        )
        
        # Capa densa final
        self.classifier = nn.Sequential(
            MaxNormLinear(self._get_flattened_size(), nb_classes,max_norm=norm_rate),
            nn.Softmax(dim=1)
        )

    def _get_flattened_size(self):
        # Crea un input dummy para calcular la salida flatten
        dummy = torch.zeros(1, *self.input_shape)
        x1 = self.block2(self.block1(dummy))
        x2 = self.block4(self.block3(dummy))
        x3 = self.block6(self.block5(dummy))
        return x1.numel() + x2.numel() + x3.numel()

    def forward(self, inputs):
        # Check if the input has 4 dimensions
        if len(inputs.shape) == 3:
            # Add one dimension in position 1
            inputs = inputs.unsqueeze(1)

        x1 = self.block2(self.block1(inputs))
        x2 = self.block4(self.block3(inputs))
        x3 = self.block6(self.block5(inputs))
        x = torch.cat([x1.view(x1.size(0), -1),
                       x2.view(x2.size(0), -1),
                       x3.view(x3.size(0), -1)], dim=1)
        return self.classifier(x)


class MaxNormLinear(nn.Linear):
    def __init__(self, in_features, out_features, max_norm=0.25, bias=True):
        super(MaxNormLinear, self).__init__(in_features, out_features, bias)
        self.max_norm = max_norm

    def forward(self, input):
        # Restringe la norma de cada fila de los pesos
        with torch.no_grad():
            norm = self.weight.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=self.max_norm)
            self.weight *= (desired / (1e-8 + norm))
        return super(MaxNormLinear, self).forward(input)