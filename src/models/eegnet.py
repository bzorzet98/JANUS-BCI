# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn
from torch.nn.functional import elu
import warnings
from einops.layers.torch import Rearrange

# Requiered functions, classes and modules. That in the original code are imported but here are defined.
def deprecated_args(obj, *old_new_args):
    out_args = []
    for old_name, new_name, old_val, new_val in old_new_args:
        if old_val is None:
            out_args.append(new_val)
        else:
            warnings.warn(
                f"{obj.__class__.__name__}: {old_name!r} is depreciated. Use {new_name!r} instead."
            )
            if new_val is not None:
                raise ValueError(
                    f"{obj.__class__.__name__}: Both {old_name!r} and {new_name!r} were specified."
                )
            out_args.append(old_val)
    return out_args

def squeeze_final_output(x):
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x

def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)


def _glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot
     uniform/xavier initialization, and setting biases to zero. Weights from
     batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x


class Expression(nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__ +
            "(expression=%s) " % expression_str
        )


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

def size_after_conv(input_size, kernel_size, stride, padding, dilation):
    return ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1).to(torch.int)

def size_after_avgpool(input_size, kernel_size, stride, padding):
    return ((input_size + 2 * padding - kernel_size) // stride + 1).to(torch.int)

def size_after_maxpool(input_size, kernel_size, stride, padding, dilation):
    return ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1).to(torch.int)






class EEGNetv1(nn.Sequential):
    """EEGNet model from Lawhern et al. 2016.

    See details in [EEGNet]_.

    Parameters
    ----------
    in_chans :
        Alias for n_chans.
    n_classes:
        Alias for n_outputs.
    input_window_samples :
        Alias for n_times.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description.

    References
    ----------
    .. [EEGNet] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
       S. M., Hung, C. P., & Lance, B. J. (2016).
       EEGNet: A Compact Convolutional Network for EEG-based
       Brain-Computer Interfaces.
       arXiv preprint arXiv:1611.08024.
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        final_conv_length="auto",
        pool_mode="max",
        second_kernel_size=(2, 32),
        third_kernel_size=(8, 4),
        drop_prob=0.25,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        in_chans=None,
        n_classes=None,
        input_window_samples=None,
        add_log_softmax=False,
    ):
        n_chans, n_outputs, n_times = deprecated_args(
            self,
            ("in_chans", "n_chans", in_chans, n_chans),
            ("n_classes", "n_outputs", n_classes, n_outputs),
            ("input_window_samples", "n_times", input_window_samples, n_times),
        )
        # super().__init__(
        #     n_outputs=n_outputs,
        #     n_chans=n_chans,
        #     chs_info=chs_info,
        #     n_times=n_times,
        #     input_window_seconds=input_window_seconds,
        #     sfreq=sfreq,
        #     add_log_softmax=add_log_softmax,
        # )

        # How we comment the original code we need these following lines
        super().__init__()
        self.n_outputs = n_outputs
        self.n_chans = n_chans
        self.chs_info = chs_info
        self.n_times = n_times
        self.input_window_seconds = input_window_seconds
        self.sfreq = sfreq
        self.add_log_softmax = add_log_softmax

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del in_chans, n_classes, input_window_samples
        if final_conv_length == "auto":
            assert self.n_times is not None
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.second_kernel_size = second_kernel_size
        self.third_kernel_size = third_kernel_size
        self.drop_prob = drop_prob
        # For the load_state_dict
        # When padronize all layers,
        # add the old's parameters here
        self.mapping = {
            "conv_classifier.weight": "final_layer.conv_classifier.weight",
            "conv_classifier.bias": "final_layer.conv_classifier.bias",
        }

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.add_module("ensuredims", Ensure4d())
        n_filters_1 = 16
        self.add_module(
            "conv_1",
            nn.Conv2d(self.n_chans, n_filters_1, (1, 1), stride=1, bias=True),
        )
        self.add_module(
            "bnorm_1",
            nn.BatchNorm2d(n_filters_1, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module("elu_1", Expression(elu))
        # transpose to examples x 1 x (virtual, not EEG) channels x time
        self.add_module("permute_1", Expression(lambda x: x.permute(0, 3, 1, 2)))

        self.add_module("drop_1", nn.Dropout(p=self.drop_prob))

        n_filters_2 = 4
        # keras pads unequal padding more in front, so padding
        # too large should be ok.
        # Not padding in time so that cropped training makes sense
        # https://stackoverflow.com/questions/43994604/padding-with-even-kernel-size-in-a-convolutional-layer-in-keras-theano

        self.add_module(
            "conv_2",
            nn.Conv2d(
                1,
                n_filters_2,
                self.second_kernel_size,
                stride=1,
                padding=(self.second_kernel_size[0] // 2, 0),
                bias=True,
            ),
        )
        self.add_module(
            "bnorm_2",
            nn.BatchNorm2d(n_filters_2, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module("elu_2", Expression(elu))
        self.add_module("pool_2", pool_class(kernel_size=(2, 4), stride=(2, 4)))
        self.add_module("drop_2", nn.Dropout(p=self.drop_prob))

        n_filters_3 = 4
        self.add_module(
            "conv_3",
            nn.Conv2d(
                n_filters_2,
                n_filters_3,
                self.third_kernel_size,
                stride=1,
                padding=(self.third_kernel_size[0] // 2, 0),
                bias=True,
            ),
        )
        self.add_module(
            "bnorm_3",
            nn.BatchNorm2d(n_filters_3, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module("elu_3", Expression(elu))
        self.add_module("pool_3", pool_class(kernel_size=(2, 4), stride=(2, 4)))
        self.add_module("drop_3", nn.Dropout(p=self.drop_prob))

        output_shape = self.get_output_shape()
        n_out_virtual_chans = output_shape[2]

        if self.final_conv_length == "auto":
            n_out_time = output_shape[3]
            self.final_conv_length = n_out_time

        # Incorporating classification module and subsequent ones in one final layer
        module = nn.Sequential()

        module.add_module(
            "conv_classifier",
            nn.Conv2d(
                n_filters_3,
                self.n_outputs,
                (n_out_virtual_chans, self.final_conv_length),
                bias=True,
            ),
        )

        if self.add_log_softmax:
            module.add_module("softmax", nn.LogSoftmax(dim=1))
        # Transpose back to the logic of braindecode,

        # so time in third dimension (axis=2)
        module.add_module(
            "permute_2",
            Rearrange("batch x y z -> batch x z y"),
        )

        module.add_module("squeeze", Expression(squeeze_final_output))

        self.add_module("final_layer", module)

        _glorot_weight_zero_bias(self)



class EEGNetv4_SM(nn.Sequential):
    """EEGNet v4 model from Lawhern et al 2018.

    See details in [EEGNet4]_.

    Parameters
    ----------
    in_chans : int
        XXX

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description.

    References
    ----------
    .. [EEGNet4] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
       S. M., Hung, C. P., & Lance, B. J. (2018).
       EEGNet: A Compact Convolutional Network for EEG-based
       Brain-Computer Interfaces.
       arXiv preprint arXiv:1611.08024.
    """
    """Here we do not use the original implementation of EEGNetv4, we use the implementation of EEGNetv4 from braindecode. 
    And we change the final layer LogSoftmax to Softmax like the original paper of EEGNetv4."""

    def __init__(
        self,
        in_chans,
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
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.third_kernel_size = third_kernel_size
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        size_input = torch.Tensor([in_chans, input_window_samples]).to(torch.int)

        self.add_module("ensuredims", Ensure4d())
        # b c 0 1
        # now to b 1 0 c
        self.add_module("dimshuffle", Expression(_transpose_to_b_1_c_0))

        self.add_module(
            "conv_temporal",
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernel_length),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length // 2),
            ),
        )
        kernel_size = torch.Tensor([1, self.kernel_length])
        padding = torch.Tensor([0, self.kernel_length // 2])
        stride = torch.Tensor([1, 1])
        dilation= torch.Tensor([1, 1])
        size_output = size_after_conv(input_size=size_input, 
                                      kernel_size=kernel_size, 
                                      stride=stride, 
                                      padding=padding, 
                                      dilation=dilation)
        self.add_module(
            "bnorm_temporal",
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module(
            "conv_spatial",
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.in_chans, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=self.F1,
                padding=(0, 0),
            ),
        )
        kernel_size = torch.Tensor([self.in_chans, 1])
        padding = torch.Tensor([0, 0])
        stride = torch.Tensor([1, 1])
        dilation= torch.Tensor([1, 1])
        size_output = size_after_conv(input_size=size_output, 
                                      kernel_size=kernel_size, 
                                      stride=stride, 
                                      padding=padding, 
                                      dilation=dilation)
        self.add_module(
            "bnorm_1",
            nn.BatchNorm2d(
                self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3
            ),
        )
        self.add_module("elu_1", Expression(elu))

        self.add_module("pool_1", pool_class(kernel_size=(1, 4), stride=(1, 4)))
        if self.pool_mode == "max":
            kernel_size = torch.Tensor([1, 4])
            padding = torch.Tensor([0, 0])
            stride = torch.Tensor([1, 4])
            dilation= torch.Tensor([1, 1])
            size_output = size_after_maxpool(size_output, kernel_size=kernel_size, 
                                            stride=stride, padding=padding, dilation=dilation)
        else:
            kernel_size = torch.Tensor([1, 4])
            padding = torch.Tensor([0, 0])
            stride = torch.Tensor([1, 4])
            size_output = size_after_avgpool(size_output, kernel_size=kernel_size, 
                                             stride=stride, padding=padding)
        self.add_module("drop_1", nn.Dropout(p=self.drop_prob))

        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.add_module(
            "conv_separable_depth",
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, 16),
                stride=1,
                bias=False,
                groups=self.F1 * self.D,
                padding=(0, 16 // 2),
            ),
        )
        kernel_size = torch.Tensor([1, 16])
        padding = torch.Tensor([0, 16//2])
        stride = torch.Tensor([1, 1])
        dilation= torch.Tensor([1, 1])
        size_output = size_after_conv(size_output, kernel_size=kernel_size, 
                                        stride=stride, padding=padding, dilation=dilation)
        
        self.add_module(
            "conv_separable_point",
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                (1, 1),
                stride=1,
                bias=False,
                padding=(0, 0),
            ),
        )

        kernel_size = torch.Tensor([1, 1])
        padding = torch.Tensor([0,0])
        stride = torch.Tensor([1, 1])
        dilation= torch.Tensor([1, 1])
        size_output = size_after_conv(size_output, kernel_size=kernel_size, 
                                        stride=stride, padding=padding, dilation=dilation)
        
        self.add_module(
            "bnorm_2",
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module("elu_2", Expression(elu))
        self.add_module("pool_2", pool_class(kernel_size=(1, 8), stride=(1, 8)))
        if pool_mode == "max":
            kernel_size = torch.Tensor([1, 8])
            padding = torch.Tensor([0,0])
            stride = torch.Tensor([1, 8])
            dilation= torch.Tensor([1, 1])
            size_output = size_after_maxpool(size_output, 
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation)
        else:
            kernel_size = torch.Tensor([1, 8])
            padding = torch.Tensor([0,0])
            stride = torch.Tensor([1, 8])
            size_output = size_after_avgpool(size_output, 
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding)
            
        self.add_module("drop_2", nn.Dropout(p=self.drop_prob))

        out = self(
            torch.ones(
                (1, self.in_chans, self.input_window_samples, 1),
                dtype=torch.float32
            )
        )
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == "auto":
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        self.add_module(
            "conv_classifier",
            nn.Conv2d(
                self.F2,
                self.n_classes,
                (n_out_virtual_chans, self.final_conv_length),
                bias=True,
            ),
        )
        kernel_size = torch.Tensor([n_out_virtual_chans, self.final_conv_length])
        padding = torch.Tensor([0,0])
        stride = torch.Tensor([1,1])
        dilation= torch.Tensor([1,1])
        size_output = size_after_conv(size_output, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation)
        self.add_module("softmax", nn.Softmax(dim=1)) # BEFORE THIS LINE WAS nn.LogSoftmax(dim=1)
        # Transpose back to the the logic of braindecode,
        # so time in third dimension (axis=2)
        self.add_module("permute_back", Expression(_transpose_1_0))
        self.add_module("squeeze", Expression(squeeze_final_output))
        
        _glorot_weight_zero_bias(self)


class EEGNetv5(nn.Module):
    def __init__(
        self,
        in_chans,
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
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.third_kernel_size = third_kernel_size
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        size_input = torch.Tensor([in_chans, input_window_samples]).to(torch.int)

        # Temporal filters 
        self.temporal_filters = nn.Conv2d(
                1,
                self.F1,
                (1, self.kernel_length),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length // 2),
            )
        self.batch_norm_1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        
        # Spatial proyections
        self.spatial_proyection = Conv2dWithConstraint(self.F1,
                self.F1 * self.D,
                (self.in_chans, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=self.F1,
                padding=(0, 0),
            )
        self.batch_norm_2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)

        #### TEMPORAL CONVOLUTIONS ####
        self.temporal_conv1D_1 = nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                (1, 4),
                groups = self.F1 * self.D,
                stride=2,
                bias=False,
                padding=(0, 1),
            )
        
        self.temporal_conv1D_2 = nn.Conv2d(
                self.F2,
                self.F2,
                (1, 4),
                groups = self.F2,
                stride=2,
                bias=False,
                padding=(0, 1),
            )
        
        self.pool_1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 2),padding=(0, 1))

        self.temporal_conv1D_3 = nn.Conv2d(
                self.F2,
                self.F2,
                (1, 4),
                groups = self.F2,
                stride=2,
                bias=False,
                padding=(0, 1),
            )
        
        self.pool_2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 2),padding=(0, 1))
        self.last_size = 8*self.F2
        # Classifiers
        self.fc1 = nn.Linear(in_features=8*self.F2, out_features=4*self.F2, bias=True)
        self.fc2 = nn.Linear(in_features=4*self.F2,out_features=n_classes,bias=True)

        self.softmax = nn.Softmax(dim = 1)

        _glorot_weight_zero_bias(self)

    def forward(self, x):
        # Add dimension
        x = x.unsqueeze(1)
        # Temporal Filters
        x = self.temporal_filters(x)
        x = self.batch_norm_1(x)
        # Spatial Proyections
        x = self.spatial_proyection(x)
        x = self.batch_norm_2(x)
        x = elu(x)

        # Temporal Convolutions
        x = self.temporal_conv1D_1(x)
        x = elu(x)
        x = self.temporal_conv1D_2(x)
        x = elu(x)
        x = self.pool_1(x)
        x = self.temporal_conv1D_3(x)
        x = elu(x)
        x = self.pool_2(x)

        
        # Fully connected layers · Here we need (batch_size, 16*32 )
        x = x.squeeze(2).view(-1, self.last_size)
        x = self.fc1(x)
        x = elu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
