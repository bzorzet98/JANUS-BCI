# Authors: Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Tonio Ball
#
# License: BSD-3

import numpy as np

import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu
from einops.layers.torch import Rearrange
import warnings
import torch.nn.functional as F

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

def np_to_th(X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs):
    """
    Convenience function to transform numpy array to `torch.Tensor`.

    Converts `X` to ndarray using asarray if necessary.

    Parameters
    ----------
    X: ndarray or list or number
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor

    Returns
    -------
    var: `torch.Tensor`
    """
    if not hasattr(X, "__len__"):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = torch.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return X_tensor


class AvgPool2dWithConv(nn.Module):
    """
    Compute average pooling using a convolution, to have the dilation parameter.

    Parameters
    ----------
    kernel_size: (int,int)
        Size of the pooling region.
    stride: (int,int)
        Stride of the pooling operation.
    dilation: int or (int,int)
        Dilation applied to the pooling filter.
    padding: int or (int,int)
        Padding applied before the pooling operation.
    """

    def __init__(self, kernel_size, stride, dilation=1, padding=0):
        super(AvgPool2dWithConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        # don't name them "weights" to
        # make sure these are not accidentally used by some procedure
        # that initializes parameters or something
        self._pool_weights = None

    def forward(self, x):
        # Create weights for the convolution on demand:
        # size or type of x changed...
        in_channels = x.size()[1]
        weight_shape = (
            in_channels,
            1,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        if self._pool_weights is None or (
            (tuple(self._pool_weights.size()) != tuple(weight_shape))
            or (self._pool_weights.is_cuda != x.is_cuda)
            or (self._pool_weights.data.type() != x.data.type())
        ):
            n_pool = np.prod(self.kernel_size)
            weights = np_to_th(np.ones(weight_shape, dtype=np.float32) / float(n_pool))
            weights = weights.type_as(x)
            if x.is_cuda:
                weights = weights.cuda()
            self._pool_weights = weights

        pooled = F.conv2d(
            x,
            self._pool_weights,
            bias=None,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=in_channels,
        )
        return pooled

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
        return self.__class__.__name__ + "(expression=%s) " % expression_str



class EEGResNet(nn.Sequential):
    """Residual Network for EEG  from Schirrmeister et al 2017.

    Model described in [Schirrmeister2017]_.

    Parameters
    ----------
     in_chans :
        Alias for `n_chans`.
     n_classes :
        Alias for `n_outputs`.
     input_window_samples :
        Alias for `n_times`.

    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        final_pool_length="auto",
        n_first_filters=20,
        n_layers_per_block=2,
        first_filter_length=3,
        nonlinearity=elu,
        split_first_layer=True,
        batch_norm_alpha=0.1,
        batch_norm_epsilon=1e-4,
        conv_weight_init_fn=lambda w: init.kaiming_normal_(w, a=0),
        chs_info=None,
        input_window_seconds=None,
        sfreq=250,
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
        self.n_chans = n_chans
        self.n_times = n_times
        self.add_log_softmax = add_log_softmax
        self.n_outputs = n_outputs
        self.sfreq = sfreq
        self.input_window_seconds = input_window_seconds
        self.chs_info = chs_info

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del in_chans, n_classes, input_window_samples

        if final_pool_length == "auto":
            assert self.n_times is not None
        assert first_filter_length % 2 == 1
        self.final_pool_length = final_pool_length
        self.n_first_filters = n_first_filters
        self.n_layers_per_block = n_layers_per_block
        self.first_filter_length = first_filter_length
        self.nonlinearity = nonlinearity
        self.split_first_layer = split_first_layer
        self.batch_norm_alpha = batch_norm_alpha
        self.batch_norm_epsilon = batch_norm_epsilon
        self.conv_weight_init_fn = conv_weight_init_fn

        self.mapping = {
            "conv_classifier.weight": "final_layer.conv_classifier.weight",
            "conv_classifier.bias": "final_layer.conv_classifier.bias",
        }

        self.add_module("ensuredims", Ensure4d())
        if self.split_first_layer:
            self.add_module("dimshuffle", Rearrange("batch C T 1 -> batch 1 T C"))
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    1,
                    self.n_first_filters,
                    (self.first_filter_length, 1),
                    stride=1,
                    padding=(self.first_filter_length // 2, 0),
                ),
            )
            self.add_module(
                "conv_spat",
                nn.Conv2d(
                    self.n_first_filters,
                    self.n_first_filters,
                    (1, self.n_chans),
                    stride=(1, 1),
                    bias=False,
                ),
            )
        else:
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    self.n_chans,
                    self.n_first_filters,
                    (self.first_filter_length, 1),
                    stride=(1, 1),
                    padding=(self.first_filter_length // 2, 0),
                    bias=False,
                ),
            )
        n_filters_conv = self.n_first_filters
        self.add_module(
            "bnorm",
            nn.BatchNorm2d(
                n_filters_conv, momentum=self.batch_norm_alpha, affine=True, eps=1e-5
            ),
        )
        self.add_module("conv_nonlin", Expression(self.nonlinearity))
        cur_dilation = np.array([1, 1])
        n_cur_filters = n_filters_conv
        i_block = 1
        for i_layer in range(self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )
        i_block += 1
        cur_dilation[0] *= 2
        n_out_filters = int(2 * n_cur_filters)
        self.add_module(
            "res_{:d}_{:d}".format(i_block, 0),
            _ResidualBlock(
                n_cur_filters,
                n_out_filters,
                dilation=cur_dilation,
            ),
        )
        n_cur_filters = n_out_filters
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )

        i_block += 1
        cur_dilation[0] *= 2
        n_out_filters = int(1.5 * n_cur_filters)
        self.add_module(
            "res_{:d}_{:d}".format(i_block, 0),
            _ResidualBlock(
                n_cur_filters,
                n_out_filters,
                dilation=cur_dilation,
            ),
        )
        n_cur_filters = n_out_filters
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )

        i_block += 1
        cur_dilation[0] *= 2
        self.add_module(
            "res_{:d}_{:d}".format(i_block, 0),
            _ResidualBlock(
                n_cur_filters,
                n_cur_filters,
                dilation=cur_dilation,
            ),
        )
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )

        i_block += 1
        cur_dilation[0] *= 2
        self.add_module(
            "res_{:d}_{:d}".format(i_block, 0),
            _ResidualBlock(
                n_cur_filters,
                n_cur_filters,
                dilation=cur_dilation,
            ),
        )
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )

        i_block += 1
        cur_dilation[0] *= 2
        self.add_module(
            "res_{:d}_{:d}".format(i_block, 0),
            _ResidualBlock(
                n_cur_filters,
                n_cur_filters,
                dilation=cur_dilation,
            ),
        )
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )
        i_block += 1
        cur_dilation[0] *= 2
        self.add_module(
            "res_{:d}_{:d}".format(i_block, 0),
            _ResidualBlock(
                n_cur_filters,
                n_cur_filters,
                dilation=cur_dilation,
            ),
        )
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )

        self.eval()
        if self.final_pool_length == "auto":
            self.add_module("mean_pool", nn.AdaptiveAvgPool2d((1, 1)))
        else:
            pool_dilation = int(cur_dilation[0]), int(cur_dilation[1])
            self.add_module(
                "mean_pool",
                AvgPool2dWithConv(
                    (self.final_pool_length, 1), (1, 1), dilation=pool_dilation
                ),
            )

        # Incorporating classification module and subsequent ones in one final layer
        module = nn.Sequential()

        module.add_module(
            "conv_classifier",
            nn.Conv2d(
                n_cur_filters,
                self.n_outputs,
                (1, 1),
                bias=True,
            ),
        )

        if self.add_log_softmax:
            module.add_module("logsoftmax", nn.LogSoftmax(dim=1))

        module.add_module("squeeze", Expression(squeeze_final_output))

        self.add_module("final_layer", module)

        # Initialize all weights
        self.apply(lambda module: _weights_init(module, self.conv_weight_init_fn))

        # Start in eval mode
        self.eval()


def _weights_init(module, conv_weight_init_fn):
    """
    initialize weights
    """
    classname = module.__class__.__name__
    if "Conv" in classname and classname != "AvgPool2dWithConv":
        conv_weight_init_fn(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif "BatchNorm" in classname:
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


class _ResidualBlock(nn.Module):
    """
    create a residual learning building block with two stacked 3x3 convlayers as in paper
    """

    def __init__(
        self,
        in_filters,
        out_num_filters,
        dilation,
        filter_time_length=3,
        nonlinearity=elu,
        batch_norm_alpha=0.1,
        batch_norm_epsilon=1e-4,
    ):
        super(_ResidualBlock, self).__init__()
        time_padding = int((filter_time_length - 1) * dilation[0])
        assert time_padding % 2 == 0
        time_padding = int(time_padding // 2)
        dilation = (int(dilation[0]), int(dilation[1]))
        assert (out_num_filters - in_filters) % 2 == 0, (
            "Need even number of extra channels in order to be able to " "pad correctly"
        )
        self.n_pad_chans = out_num_filters - in_filters

        self.conv_1 = nn.Conv2d(
            in_filters,
            out_num_filters,
            (filter_time_length, 1),
            stride=(1, 1),
            dilation=dilation,
            padding=(time_padding, 0),
        )
        self.bn1 = nn.BatchNorm2d(
            out_num_filters,
            momentum=batch_norm_alpha,
            affine=True,
            eps=batch_norm_epsilon,
        )
        self.conv_2 = nn.Conv2d(
            out_num_filters,
            out_num_filters,
            (filter_time_length, 1),
            stride=(1, 1),
            dilation=dilation,
            padding=(time_padding, 0),
        )
        self.bn2 = nn.BatchNorm2d(
            out_num_filters,
            momentum=batch_norm_alpha,
            affine=True,
            eps=batch_norm_epsilon,
        )
        # also see https://mail.google.com/mail/u/0/#search/ilya+joos/1576137dd34c3127
        # for resnet options as ilya used them
        self.nonlinearity = nonlinearity

    def forward(self, x):
        stack_1 = self.nonlinearity(self.bn1(self.conv_1(x)))
        stack_2 = self.bn2(self.conv_2(stack_1))  # next nonlin after sum
        if self.n_pad_chans != 0:
            zeros_for_padding = torch.autograd.Variable(
                x.new_zeros(
                    (x.size()[0], self.n_pad_chans // 2, x.size()[2], x.size()[3])
                )
            )
            x = torch.cat((zeros_for_padding, x, zeros_for_padding), dim=1)
        out = self.nonlinearity(x + stack_2)
        return out