# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn
from torch.nn import ConstantPad2d
import warnings
import numpy as np

from deep4 import Deep4Net
from shallow_fbcsp import ShallowFBCSPNet


# Requiered functions, classes and modules. That in the original code are imported but here are defined.
def to_dense_prediction_model(model, axis=(2, 3)):
    """
    Transform a sequential model with strides to a model that outputs
    dense predictions by removing the strides and instead inserting dilations.
    Modifies model in-place.

    Parameters
    ----------
    model: torch.nn.Module
        Model which modules will be modified
    axis: int or (int,int)
        Axis to transform (in terms of intermediate output axes)
        can either be 2, 3, or (2,3).

    Notes
    -----
    Does not yet work correctly for average pooling.
    Prior to version 0.1.7, there had been a bug that could move strides
    backwards one layer.

    """
    if not hasattr(axis, "__len__"):
        axis = [axis]
    assert all([ax in [2, 3] for ax in axis]), "Only 2 and 3 allowed for axis"
    axis = np.array(axis) - 2
    stride_so_far = np.array([1, 1])
    for module in model.modules():
        if hasattr(module, "dilation"):
            assert module.dilation == 1 or (module.dilation == (1, 1)), (
                "Dilation should equal 1 before conversion, maybe the model is "
                "already converted?"
            )
            new_dilation = [1, 1]
            for ax in axis:
                new_dilation[ax] = int(stride_so_far[ax])
            module.dilation = tuple(new_dilation)
        if hasattr(module, "stride"):
            if not hasattr(module.stride, "__len__"):
                module.stride = (module.stride, module.stride)
            stride_so_far *= np.array(module.stride)
            new_stride = list(module.stride)
            for ax in axis:
                new_stride[ax] = 1
            module.stride = tuple(new_stride)

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

class HybridNet(nn.Module):
    """Hybrid ConvNet model from Schirrmeister et al 2017.

    See [Schirrmeister2017]_ for details.

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
        in_chans=None,
        n_classes=None,
        input_window_samples=None,
        add_log_softmax=False,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
        final_conv_length= "auto",
        args_deep4net=None,
        args_shallowfbcspnet=None,
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
        #     n_times=n_times,
        #     input_window_seconds=input_window_seconds,
        #     sfreq=sfreq,
        #     chs_info=chs_info,
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
        self.final_conv_length = final_conv_length

        self.mapping = {
            "final_conv.weight": "final_layer.weight",
            "final_conv.bias": "final_layer.bias",
        }

        if args_deep4net is None:
            deep_model = Deep4Net(
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_filters_time=20,
                n_filters_spat=30,
                n_filters_2=40,
                n_filters_3=50,
                n_filters_4=60,
                n_times=n_times,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
                chs_info=chs_info,
                final_conv_length=2,
            )
        else:
            deep_model = Deep4Net(**args_deep4net)
        if args_shallowfbcspnet is None:
            shallow_model = ShallowFBCSPNet(
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_times=n_times,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
                chs_info=chs_info,
                n_filters_time=30,
                n_filters_spat=40,
                filter_time_length=28,
                final_conv_length=29,
            )
        else:
            shallow_model = ShallowFBCSPNet(**args_shallowfbcspnet)

        del n_outputs, n_chans, n_times, input_window_seconds, sfreq, chs_info
        del in_chans, n_classes, input_window_samples

        reduced_deep_model = nn.Sequential()
        for name, module in deep_model.named_children():
            if name == "final_layer":
                new_conv_layer = nn.Conv2d(
                    module.conv_classifier.in_channels,
                    60,
                    kernel_size=module.conv_classifier.kernel_size,
                    stride=module.conv_classifier.stride,
                )
                reduced_deep_model.add_module("deep_final_conv", new_conv_layer)
                break
            reduced_deep_model.add_module(name, module)

        reduced_shallow_model = nn.Sequential()
        for name, module in shallow_model.named_children():
            if name == "final_layer":
                new_conv_layer = nn.Conv2d(
                    module.conv_classifier.in_channels,
                    40,
                    kernel_size=module.conv_classifier.kernel_size,
                    stride=module.conv_classifier.stride,
                )
                reduced_shallow_model.add_module("shallow_final_conv", new_conv_layer)
                break
            reduced_shallow_model.add_module(name, module)

        to_dense_prediction_model(reduced_deep_model)
        to_dense_prediction_model(reduced_shallow_model)
        self.reduced_deep_model = reduced_deep_model
        self.reduced_shallow_model = reduced_shallow_model
        
        if self.final_conv_length == "auto":
            x=torch.ones(1, self.n_chans, self.n_times)
            deep_out = self.reduced_deep_model(x)
            shallow_out = self.reduced_shallow_model(x)
            n_diff_deep_shallow = deep_out.size()[2] - shallow_out.size()[2]
            if n_diff_deep_shallow < 0:
                deep_out = ConstantPad2d((0, 0, -n_diff_deep_shallow, 0), 0)(deep_out)
            elif n_diff_deep_shallow > 0:
                shallow_out = ConstantPad2d((0, 0, n_diff_deep_shallow, 0), 0)(shallow_out)
            merged_out = torch.cat((deep_out, shallow_out), dim=1)
            self.final_conv_length = merged_out.shape[2]
            
        # self.final_layer = nn.Sequential(
        #     nn.Conv2d(100, self.n_outputs, kernel_size=(1, 1), stride=1),
        #     nn.LogSoftmax(dim=1) if self.add_log_softmax else nn.Identity(),
        # )

        self.final_layer = nn.Sequential(
            nn.Conv2d(100, self.n_outputs, kernel_size=(self.final_conv_length, 1), stride=1),
            nn.LogSoftmax(dim=1) if self.add_log_softmax else nn.Softmax(dim=1),
        )


    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        deep_out = self.reduced_deep_model(x)
        shallow_out = self.reduced_shallow_model(x)

        n_diff_deep_shallow = deep_out.size()[2] - shallow_out.size()[2]

        if n_diff_deep_shallow < 0:
            deep_out = ConstantPad2d((0, 0, -n_diff_deep_shallow, 0), 0)(deep_out)
        elif n_diff_deep_shallow > 0:
            shallow_out = ConstantPad2d((0, 0, n_diff_deep_shallow, 0), 0)(shallow_out)

        merged_out = torch.cat((deep_out, shallow_out), dim=1)

        output = self.final_layer(merged_out)

        squeezed = output.squeeze()

        return squeezed