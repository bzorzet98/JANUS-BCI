import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from src.networks.eegnet import EEGNetv4_SM, Expression, squeeze_final_output, _transpose_1_0, _glorot_weight_zero_bias


class EEGNetv4_feature_extractor(nn.Module):
    def __init__(self, in_chans,
                n_classes,
                input_window_samples=None,
                final_conv_length="auto",
                pool_mode="mean",
                F1=8,
                D=2,
                F2=16,  # usually set to F1*D (?)
                kernel_length=64,
                third_kernel_size=(8, 4),
                drop_prob=0.25,):
        super().__init__()
        self.feature_extractor = EEGNetv4_SM(in_chans=in_chans,
                                             n_classes=n_classes,
                                            input_window_samples=input_window_samples,
                                            final_conv_length=final_conv_length,
                                            pool_mode=pool_mode,
                                            F1=F1,
                                            D=D,
                                            F2=F2,
                                            kernel_length=kernel_length,
                                            third_kernel_size=third_kernel_size,
                                            drop_prob=drop_prob)
        
        # ELiminate the last classifier layer
        layer_to_eliminate = ["conv_classifier", "softmax", "permute_back", "squeeze"]
        for layer in layer_to_eliminate:
            delattr(self.feature_extractor, layer)
            
    def forward(self, x):
        return self.feature_extractor(x)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

# Modelo principal
class Adversarial_EEGNet(nn.Module):
    def __init__(self, in_chans,
                n_classes,
                input_window_samples=None,
                final_conv_length="auto",
                pool_mode="mean",
                F1=8,
                D=2,
                F2=16,  # usually set to F1*D (?)
                kernel_length=64,
                third_kernel_size=(8, 4),
                drop_prob=0.25,
                alpha = 1.0):
        
        super(Adversarial_EEGNet, self).__init__()
        

        # Feature extractor
        self.feature_extractor = EEGNetv4_feature_extractor(in_chans=in_chans,
                                                            n_classes=n_classes,
                                                            input_window_samples=input_window_samples,
                                                            final_conv_length=final_conv_length,
                                                            pool_mode=pool_mode,
                                                            F1=F1,
                                                            D=D,
                                                            F2=F2,
                                                            kernel_length=kernel_length,
                                                            third_kernel_size=third_kernel_size,
                                                            drop_prob=drop_prob)
        out = self.feature_extractor((torch.ones((1, in_chans, input_window_samples, 1),
                                            dtype=torch.float32)))
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]
        if final_conv_length == "auto":
            n_out_time = out.cpu().data.numpy().shape[3]
            final_conv_length = n_out_time

        # MI classification head
        self.MI_classifier = nn.Sequential(
            nn.Conv2d(F2,
                      n_classes,
                      (n_out_virtual_chans, final_conv_length),
                      bias=True),
            nn.Softmax(dim=1),
            Expression(_transpose_1_0),
            Expression(squeeze_final_output))
        
        # Adversarial branch for sex classification
        self.grl = GradientReversalLayer(alpha=alpha)
        self.adv_classifier = nn.Sequential(
            nn.Conv2d(F2,
                      n_classes,
                      (n_out_virtual_chans, final_conv_length),
                      bias=True),
            nn.Softmax(dim=1),
            Expression(_transpose_1_0),
            Expression(squeeze_final_output)) # Binary classification (sex)
          
        # Initiating wieghts as EEGNetv4
        _glorot_weight_zero_bias(self)

    def forward(self, x,):
        features = self.feature_extractor(x)
        # MI prediction
        mi_output = self.MI_classifier(features)
        
        # Adversarial prediction (sex)
        adv_features = self.grl(features)
        sex_output = self.adv_classifier(adv_features)
        return mi_output, sex_output
