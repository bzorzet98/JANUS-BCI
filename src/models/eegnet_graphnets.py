import torch.nn as nn
from torch.nn.functional import elu
import torch 
from graph_convs import Group_ChebConv, Group_GCN, Simple_ChebConv, Simple_GCN
from torch_geometric.nn import ChebConv, GCNConv
from eegnet import (Ensure4d, Expression, 
                    _transpose_to_b_1_c_0, 
                    _transpose_1_0, 
                    squeeze_final_output,
                    _glorot_weight_zero_bias)


class EEGNetv4_JFGP_ChebConv(nn.Module):
    """
    EEGNet v4 model with GCNConv-based spatial filtering.

    Based on:
    [EEGNet4] Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Network for EEG-based BCIs.
    Adapted from Braindecode implementation.

    This version replaces the spatial convolution block with a graph-based GCNConv block.
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples=None,
        final_conv_length="auto",
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        K=2,
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=0.25,
    ):
        super().__init__()
        assert input_window_samples is not None or final_conv_length != "auto"

        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.in_chans = in_chans
        self.kernel_length = kernel_length
        self.final_conv_length = final_conv_length
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        self.ensuredims = Ensure4d()
        self.dimshuffle = Expression(_transpose_to_b_1_c_0)
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length),
            stride=1, bias=False, padding=(0, kernel_length // 2)
        )
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3)

        # Graph spatial block

        # Graph spatial block
        self.spatial_gnn1 = Simple_ChebConv(in_features=F1, out_features=F1*D, K=K)
        
        self.spatial_gnn2 = Simple_ChebConv(in_features=F1*D, out_features=F1*D, K=K)
        
        
        self.depthwise = nn.Conv2d(
            F1 * D, F1 * D, (in_chans,1),
            groups=F1 * D,
            stride=1, bias=False,
            padding=(0, 0)
        )

        self.pool_1 = pool_class(kernel_size=(1, 4), stride=(1, 4))
        self.drop_1 = nn.Dropout(p=drop_prob)

        self.conv_separable_depth = nn.Conv2d(
            F1 * D, F1 * D, (1, 16),
            stride=1, bias=False,
            groups=F1 * D,
            padding=(0, 8)
        )
        self.conv_separable_point = nn.Conv2d(
            F1 * D, F2, (1, 1),
            stride=1, bias=False,
            padding=(0, 0)
        )
        self.bnorm_2 = nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3)
        self.elu_2 = Expression(elu)
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1, 8))
        self.drop_2 = nn.Dropout(p=drop_prob)

        # Dummy forward to infer shape
        with torch.no_grad():
            dummy_x = torch.ones((1, in_chans, input_window_samples, 1), dtype=torch.float32)
            dummy_adjacency_matrix = torch.ones(1, in_chans, in_chans).float()
            x = self.ensuredims(dummy_x)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            
            B,F1,C,T = x.shape
            x = x.permute(0,3,2,1)
            x = x.reshape(B*T,C,F1)
            
            x = self.spatial_gnn1(x, dummy_adjacency_matrix)
            x = self.spatial_gnn2(x, dummy_adjacency_matrix)
            
            x = x.reshape(B,T,C, x.shape[-1])
            x = x.permute(0,3,2,1) # B, F1*D, C, T

            x = self.depthwise(x)
            x = self.pool_1(x)
            x = self.drop_1(x)
            x = self.conv_separable_depth(x)
            x = self.conv_separable_point(x)
            x = self.bnorm_2(x)
            x = self.elu_2(x)
            x = self.pool_2(x)
            x = self.drop_2(x)
            n_out_virtual_chans = x.shape[2]
            if final_conv_length == "auto":
                self.final_conv_length = x.shape[3]

        self.conv_classifier = nn.Conv2d(
            F2, n_classes,
            (n_out_virtual_chans, self.final_conv_length),
            bias=True
        )
        self.softmax = nn.Softmax(dim=1)
        self.permute_back = Expression(_transpose_1_0)
        self.squeeze = Expression(squeeze_final_output)

        _glorot_weight_zero_bias(self)
        
    def forward(self, X, A):
        x = self.ensuredims(X)
        x = self.dimshuffle(x)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        
        B,F1,C,T = x.shape
        
        x = x.permute(0,3,2,1)
        x = x.reshape(B*T,C,F1)
        # Expand the adjacency matrix for each time step
        A_expanded = A.repeat_interleave(T, dim=0)  # (B*T, C, C)
        
        x = self.spatial_gnn1(x, A_expanded)
        x = elu(x)
        x = self.spatial_gnn2(x, A_expanded)
        x = elu(x)
            
        x = x.reshape(B,T,C, x.shape[-1])
        x = x.permute(0,3,2,1) # B, F1*D, C, T
        
        x = self.depthwise(x)
    
        # Convert to [batch, num_fm* D, 1, out_features]
        x = self.pool_1(x)
        x = self.drop_1(x)

        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bnorm_2(x)
        x = self.elu_2(x)
        x = self.pool_2(x)
        x = self.drop_2(x)

        x = self.conv_classifier(x)
        x = self.softmax(x)
        x = self.permute_back(x)
        x = self.squeeze(x)
        return x
            
class EEGNetv4_JFGP_GCNConv(nn.Module):
    """
    EEGNet v4 model with GCNConv-based spatial filtering.

    Based on:
    [EEGNet4] Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Network for EEG-based BCIs.
    Adapted from Braindecode implementation.

    This version replaces the spatial convolution block with a graph-based GCNConv block.
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples=None,
        final_conv_length="auto",
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=0.25
    ):
        super().__init__()
        assert input_window_samples is not None or final_conv_length != "auto"

        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.in_chans = in_chans
        self.kernel_length = kernel_length
        self.final_conv_length = final_conv_length
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        self.ensuredims = Ensure4d()
        self.dimshuffle = Expression(_transpose_to_b_1_c_0)
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length),
            stride=1, bias=False, padding=(0, kernel_length // 2)
        )
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3)

        # Graph spatial block
        self.spatial_gnn1 = Simple_GCN(in_features=F1, out_features=F1*D)
        
        self.spatial_gnn2 = Simple_GCN(in_features=F1*D, out_features=F1*D)
        
        
        self.depthwise = nn.Conv2d(
            F1 * D, F1 * D, (in_chans,1),
            groups=F1 * D,
            stride=1, bias=False,
            padding=(0, 0)
        )

        self.pool_1 = pool_class(kernel_size=(1, 4), stride=(1, 4))
        self.drop_1 = nn.Dropout(p=drop_prob)

        self.conv_separable_depth = nn.Conv2d(
            F1 * D, F1 * D, (1, 16),
            stride=1, bias=False,
            groups=F1 * D,
            padding=(0, 8)
        )
        self.conv_separable_point = nn.Conv2d(
            F1 * D, F2, (1, 1),
            stride=1, bias=False,
            padding=(0, 0)
        )
        self.bnorm_2 = nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3)
        self.elu_2 = Expression(elu)
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1, 8))
        self.drop_2 = nn.Dropout(p=drop_prob)

        # Dummy forward to infer shape
        with torch.no_grad():
            dummy_x = torch.ones((1, in_chans, input_window_samples, 1), dtype=torch.float32)
            dummy_adjacency_matrix = torch.ones(1, in_chans, in_chans).float()
            x = self.ensuredims(dummy_x)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            
            B,F1,C,T = x.shape
            x = x.permute(0,3,2,1)
            x = x.reshape(B*T,C,F1)
            
            x = self.spatial_gnn1(x, dummy_adjacency_matrix)
            x = self.spatial_gnn2(x, dummy_adjacency_matrix)
            
            x = x.reshape(B,T,C, x.shape[-1])
            x = x.permute(0,3,2,1) # B, F1*D, C, T

            x = self.depthwise(x)
            x = self.pool_1(x)
            x = self.drop_1(x)
            x = self.conv_separable_depth(x)
            x = self.conv_separable_point(x)
            x = self.bnorm_2(x)
            x = self.elu_2(x)
            x = self.pool_2(x)
            x = self.drop_2(x)
            n_out_virtual_chans = x.shape[2]
            if final_conv_length == "auto":
                self.final_conv_length = x.shape[3]

        self.conv_classifier = nn.Conv2d(
            F2, n_classes,
            (n_out_virtual_chans, self.final_conv_length),
            bias=True
        )
        self.softmax = nn.Softmax(dim=1)
        self.permute_back = Expression(_transpose_1_0)
        self.squeeze = Expression(squeeze_final_output)

        _glorot_weight_zero_bias(self)
        
    def forward(self, X, A):
        x = self.ensuredims(X)
        x = self.dimshuffle(x)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        
        B,F1,C,T = x.shape
        
        x = x.permute(0,3,2,1)
        x = x.reshape(B*T,C,F1)
        # Expand the adjacency matrix for each time step
        A_expanded = A.repeat_interleave(T, dim=0)  # (B*T, C, C)
    
        x = self.spatial_gnn1(x, A_expanded)
        x = elu(x)
        x = self.spatial_gnn2(x, A_expanded)
        x = elu(x)
            
        x = x.reshape(B,T,C, x.shape[-1])
        x = x.permute(0,3,2,1) # B, F1*D, C, T
        
        x = self.depthwise(x)
    
        # Convert to [batch, num_fm* D, 1, out_features]
        x = self.pool_1(x)
        x = self.drop_1(x)

        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bnorm_2(x)
        x = self.elu_2(x)
        x = self.pool_2(x)
        x = self.drop_2(x)

        x = self.conv_classifier(x)
        x = self.softmax(x)
        x = self.permute_back(x)
        x = self.squeeze(x)
        return x
    
    
class EEGNetv4_FWGB_ChebConv(nn.Module):
    """
    EEGNet v4 model with ChebConv-based spatial filtering.

    Based on:
    [EEGNet4] Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Network for EEG-based BCIs.
    Adapted from Braindecode implementation.

    This version replaces the spatial convolution block with a graph-based ChebConv block.
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples=None,
        final_conv_length="auto",
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        K=2,
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=0.25,
    ):
        super().__init__()
        assert input_window_samples is not None or final_conv_length != "auto"

        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.in_chans = in_chans
        self.kernel_length = kernel_length
        self.final_conv_length = final_conv_length
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        self.ensuredims = Ensure4d()
        self.dimshuffle = Expression(_transpose_to_b_1_c_0)
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length),
            stride=1, bias=False, padding=(0, kernel_length // 2)
        )
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3)

        # Graph spatial block

        # Graph spatial block
        self.spatial_gnn1 = Group_ChebConv(in_features=1, 
                            in_channels = F1, 
                            out_features=1, 
                            out_channels= F1*D, 
                            groups=F1, bias=True,
                            K=K)
        
        self.spatial_gnn2 = Group_ChebConv(in_features=1,
                            in_channels = F1*D, 
                            out_features=1, 
                            out_channels= F1*D, 
                            groups=F1*D, bias=True,
                            K=K)
        
        self.depthwise = nn.Conv2d(
            F1 * D, F1 * D, (in_chans,1),
            groups=F1 * D,
            stride=1, bias=False,
            padding=(0, 0)
        )

        self.pool_1 = pool_class(kernel_size=(1, 4), stride=(1, 4))
        self.drop_1 = nn.Dropout(p=drop_prob)

        self.conv_separable_depth = nn.Conv2d(
            F1 * D, F1 * D, (1, 16),
            stride=1, bias=False,
            groups=F1 * D,
            padding=(0, 8)
        )
        self.conv_separable_point = nn.Conv2d(
            F1 * D, F2, (1, 1),
            stride=1, bias=False,
            padding=(0, 0)
        )
        self.bnorm_2 = nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3)
        self.elu_2 = Expression(elu)
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1, 8))
        self.drop_2 = nn.Dropout(p=drop_prob)

        # Dummy forward to infer shape
        with torch.no_grad():
            dummy_x = torch.ones((1, in_chans, input_window_samples, 1), dtype=torch.float32)
            dummy_adjacency_matrix = torch.ones(1, in_chans, in_chans).float()
            x = self.ensuredims(dummy_x)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            
            B,F1,C,T = x.shape
            x = x.permute(0,3,1,2)
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2],x.shape[3])
            x = x.unsqueeze(3)
            
            x = self.spatial_gnn1(x, dummy_adjacency_matrix)    
            x = self.spatial_gnn2(x, dummy_adjacency_matrix)
            
            BT, F1D, C, L = x.shape
            x = x.reshape(B,T,F1D,C,L)
            x = x.squeeze(dim=4)
            x = x.permute(0,2,3,1)
            
            x = self.depthwise(x)
            x = self.pool_1(x)
            x = self.drop_1(x)
            x = self.conv_separable_depth(x)
            x = self.conv_separable_point(x)
            x = self.bnorm_2(x)
            x = self.elu_2(x)
            x = self.pool_2(x)
            x = self.drop_2(x)
            n_out_virtual_chans = x.shape[2]
            if final_conv_length == "auto":
                self.final_conv_length = x.shape[3]

        self.conv_classifier = nn.Conv2d(
            F2, n_classes,
            (n_out_virtual_chans, self.final_conv_length),
            bias=True
        )
        self.softmax = nn.Softmax(dim=1)
        self.permute_back = Expression(_transpose_1_0)
        self.squeeze = Expression(squeeze_final_output)

        _glorot_weight_zero_bias(self)
        
    def forward(self, X, A):
        x = self.ensuredims(X)
        x = self.dimshuffle(x)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)

        B,F1,C,T = x.shape
        x = x.permute(0, 3, 1, 2)          # (B, T, F1, C)
        x = x.reshape(B * T, F1, C).unsqueeze(-1)  # (B*T, F1, C, 1)
        
        # Expand the adjacency matrix for each time step
        A_expanded = A.repeat_interleave(T, dim=0)  # (B*T, C, C)
        
        x = self.spatial_gnn1(x, A_expanded)
        x = elu(x)
        x = self.spatial_gnn2(x, A_expanded)
        x = elu(x)
        
        x = x.reshape(B, T, -1, C)         # (B, T, F1', C)
        x = x.permute(0, 2, 3, 1)          # (B, F1', C, T)

        x = self.depthwise(x)
        
        # Convert to [batch, num_fm* D, 1, out_features]
        x = self.pool_1(x)
        x = self.drop_1(x)

        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bnorm_2(x)
        x = self.elu_2(x)
        x = self.pool_2(x)
        x = self.drop_2(x)

        x = self.conv_classifier(x)
        x = self.softmax(x)
        x = self.permute_back(x)
        x = self.squeeze(x)
        return x
    
class EEGNetv4_FWGB_GCNConv(nn.Module):
    """
    EEGNet v4 model with ChebConv-based spatial filtering.

    Based on:
    [EEGNet4] Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Network for EEG-based BCIs.
    Adapted from Braindecode implementation.

    This version replaces the spatial convolution block with a graph-based ChebConv block.
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples=None,
        final_conv_length="auto",
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=0.25,
        activation="relu"
    ):
        super().__init__()
        assert input_window_samples is not None or final_conv_length != "auto"

        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.in_chans = in_chans
        self.kernel_length = kernel_length
        self.final_conv_length = final_conv_length
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        self.ensuredims = Ensure4d()
        self.dimshuffle = Expression(_transpose_to_b_1_c_0)
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length),
            stride=1, bias=False, padding=(0, kernel_length // 2)
        )
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3)

        
        # Graph spatial block
        self.spatial_gnn1 = Group_GCN(in_features=1, 
                 in_channels = F1, 
                 out_features=1, 
                 out_channels= F1*D, 
                 groups=F1, bias=True)
        
        self.spatial_gnn2 = Group_GCN(in_features=1,
                    in_channels = F1*D, 
                    out_features=1, 
                    out_channels= F1*D, 
                    groups=F1*D, bias=True)
        
        self.depthwise = nn.Conv2d(
            F1 * D, F1 * D, (in_chans,1),
            groups=F1 * D,
            stride=1, bias=False,
            padding=(0, 0)
        )

        self.pool_1 = pool_class(kernel_size=(1, 4), stride=(1, 4))
        self.drop_1 = nn.Dropout(p=drop_prob)

        self.conv_separable_depth = nn.Conv2d(
            F1 * D, F1 * D, (1, 16),
            stride=1, bias=False,
            groups=F1 * D,
            padding=(0, 8)
        )
        self.conv_separable_point = nn.Conv2d(
            F1 * D, F2, (1, 1),
            stride=1, bias=False,
            padding=(0, 0)
        )
        self.bnorm_2 = nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3)
        self.elu_2 = Expression(elu)
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1, 8))
        self.drop_2 = nn.Dropout(p=drop_prob)

        # Dummy forward to infer shape
        with torch.no_grad():
            dummy_x = torch.ones((1, in_chans, input_window_samples, 1), dtype=torch.float32)
            dummy_adjacency_matrix = torch.ones(1, in_chans, in_chans).float()
            x = self.ensuredims(dummy_x)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            
            B,F1,C,T = x.shape
            x = x.permute(0,3,1,2)
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2],x.shape[3])
            x = x.unsqueeze(3)
            
            x = self.spatial_gnn1(x, dummy_adjacency_matrix)    
            x = self.spatial_gnn2(x, dummy_adjacency_matrix)
            
            BT, F1D, C, L = x.shape
            x = x.reshape(B,T,F1D,C,L)
            x = x.squeeze(dim=4)
            x = x.permute(0,2,3,1)
            
            x = self.depthwise(x)
            x = self.pool_1(x)
            x = self.drop_1(x)
            x = self.conv_separable_depth(x)
            x = self.conv_separable_point(x)
            x = self.bnorm_2(x)
            x = self.elu_2(x)
            x = self.pool_2(x)
            x = self.drop_2(x)
            n_out_virtual_chans = x.shape[2]
            if final_conv_length == "auto":
                self.final_conv_length = x.shape[3]

        self.conv_classifier = nn.Conv2d(
            F2, n_classes,
            (n_out_virtual_chans, self.final_conv_length),
            bias=True
        )
        self.softmax = nn.Softmax(dim=1)
        self.permute_back = Expression(_transpose_1_0)
        self.squeeze = Expression(squeeze_final_output)

        _glorot_weight_zero_bias(self)
        
    def forward(self, X, A):
        x = self.ensuredims(X)
        x = self.dimshuffle(x)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)

        B,F1,C,T = x.shape
        x = x.permute(0, 3, 1, 2)          # (B, T, F1, C)
        x = x.reshape(B * T, F1, C).unsqueeze(-1)  # (B*T, F1, C, 1)
        
        # Expand the adjacency matrix for each time step
        A_expanded = A.repeat_interleave(T, dim=0)  # (B*T, C, C)
        
        x = self.spatial_gnn1(x, A_expanded)
        x = elu(x)
        x = self.spatial_gnn2(x, A_expanded)
        
        x = x.reshape(B, T, -1, C)         # (B, T, F1', C)
        x = x.permute(0, 2, 3, 1)          # (B, F1', C, T)
        x = elu(x)
        
        x = self.depthwise(x)
        
        # Convert to [batch, num_fm* D, 1, out_features]
        x = self.pool_1(x)
        x = self.drop_1(x)

        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bnorm_2(x)
        x = self.elu_2(x)
        x = self.pool_2(x)
        x = self.drop_2(x)

        x = self.conv_classifier(x)
        x = self.softmax(x)
        x = self.permute_back(x)
        x = self.squeeze(x)
        return x
    


class EEGNetv4_JFGP_SerialChebConv(nn.Module):
    """
    EEGNet v4 model with GCNConv-based spatial filtering.

    Based on:
    [EEGNet4] Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Network for EEG-based BCIs.
    Adapted from Braindecode implementation.

    This version replaces the spatial convolution block with a graph-based GCNConv block.
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples=None,
        final_conv_length="auto",
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        K=2,
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=0.25,
    ):
        super().__init__()
        assert input_window_samples is not None or final_conv_length != "auto"

        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.in_chans = in_chans
        self.kernel_length = kernel_length
        self.final_conv_length = final_conv_length
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        self.ensuredims = Ensure4d()
        self.dimshuffle = Expression(_transpose_to_b_1_c_0)
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length),
            stride=1, bias=False, padding=(0, kernel_length // 2)
        )
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3)

        # Graph spatial block

        # Graph spatial block
        self.spatial_gnn1 = ChebConv(in_channels=F1, out_channels=F1*D, K=K)
        
        self.spatial_gnn2 = ChebConv(in_channels=F1*D, out_channels=F1*D, K=K)
        
        
        self.depthwise = nn.Conv2d(
            F1 * D, F1 * D, (in_chans,1),
            groups=F1 * D,
            stride=1, bias=False,
            padding=(0, 0)
        )

        self.pool_1 = pool_class(kernel_size=(1, 4), stride=(1, 4))
        self.drop_1 = nn.Dropout(p=drop_prob)

        self.conv_separable_depth = nn.Conv2d(
            F1 * D, F1 * D, (1, 16),
            stride=1, bias=False,
            groups=F1 * D,
            padding=(0, 8)
        )
        self.conv_separable_point = nn.Conv2d(
            F1 * D, F2, (1, 1),
            stride=1, bias=False,
            padding=(0, 0)
        )
        self.bnorm_2 = nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3)
        self.elu_2 = Expression(elu)
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1, 8))
        self.drop_2 = nn.Dropout(p=drop_prob)

        # Dummy forward to infer shape
        with torch.no_grad():
            dummy_x = torch.ones((1, in_chans, input_window_samples, 1), dtype=torch.float32)
            dummy_edge_index = torch.ones(2, in_chans).long()
            x = self.ensuredims(dummy_x)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            
            B,F1,C,T = x.shape
            x = x.permute(0,2,3,1)
            x = x.reshape(B*C*T,F1)
            
            x = self.spatial_gnn1(x,dummy_edge_index)
            x = self.spatial_gnn2(x, dummy_edge_index)
            
            x = x.reshape(B,C,T, x.shape[-1]) # B,C,T,F1*D
            x = x.permute(0,3,1,2) # B, F1*D, C, T

            x = self.depthwise(x)
            x = self.pool_1(x)
            x = self.drop_1(x)
            x = self.conv_separable_depth(x)
            x = self.conv_separable_point(x)
            x = self.bnorm_2(x)
            x = self.elu_2(x)
            x = self.pool_2(x)
            x = self.drop_2(x)
            n_out_virtual_chans = x.shape[2]
            if final_conv_length == "auto":
                self.final_conv_length = x.shape[3]

        self.conv_classifier = nn.Conv2d(
            F2, n_classes,
            (n_out_virtual_chans, self.final_conv_length),
            bias=True
        )
        self.softmax = nn.Softmax(dim=1)
        self.permute_back = Expression(_transpose_1_0)
        self.squeeze = Expression(squeeze_final_output)

        _glorot_weight_zero_bias(self)
        
    def forward(self, X, edge_index, edge_weight):
        x = self.ensuredims(X)
        x = self.dimshuffle(x)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        
        B,F1,C,T = x.shape
        
        x = x.permute(0,2,3,1) # B, C, T, F1
        x = x.reshape(B*C*T, F1) # (B*C*T, F1)

        # Expand the adjacency matrix for each time step
        edge_index_batched, edge_weight_batched = expand_batched_edges_over_time(edge_index, edge_weight, T, C, device = X.device)
        batch = build_batch_vector(B, T, C, device=X.device)
        x = self.spatial_gnn1(x, edge_index_batched, edge_weight_batched , batch = batch)
        x = elu(x)
        x = self.spatial_gnn2(x, edge_index_batched, edge_weight_batched , batch = batch)
        x = elu(x)
            
        x = x.reshape(B,C,T, x.shape[-1]) # B,C,T,F1*D
        x = x.permute(0,3,1,2) # B, F1*D, C, T
        
        x = self.depthwise(x)
    
        # Convert to [batch, num_fm* D, 1, out_features]
        x = self.pool_1(x)
        x = self.drop_1(x)

        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bnorm_2(x)
        x = self.elu_2(x)
        x = self.pool_2(x)
        x = self.drop_2(x)

        x = self.conv_classifier(x)
        x = self.softmax(x)
        x = self.permute_back(x)
        x = self.squeeze(x)
        return x

class EEGNetv4_JFGP_SerialGCNConv(nn.Module):
    """
    EEGNet v4 model with GCNConv-based spatial filtering.

    Based on:
    [EEGNet4] Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Network for EEG-based BCIs.
    Adapted from Braindecode implementation.

    This version replaces the spatial convolution block with a graph-based GCNConv block.
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples=None,
        final_conv_length="auto",
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=0.25
    ):
        super().__init__()
        assert input_window_samples is not None or final_conv_length != "auto"

        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.in_chans = in_chans
        self.kernel_length = kernel_length
        self.final_conv_length = final_conv_length
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        self.ensuredims = Ensure4d()
        self.dimshuffle = Expression(_transpose_to_b_1_c_0)
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length),
            stride=1, bias=False, padding=(0, kernel_length // 2)
        )
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3)

        # Graph spatial block
        self.spatial_gnn1 = GCNConv(in_channels=F1, out_channels=F1*D)
        
        self.spatial_gnn2 = GCNConv(in_channels=F1*D, out_channels=F1*D)
        
        
        self.depthwise = nn.Conv2d(
            F1 * D, F1 * D, (in_chans,1),
            groups=F1 * D,
            stride=1, bias=False,
            padding=(0, 0)
        )

        self.pool_1 = pool_class(kernel_size=(1, 4), stride=(1, 4))
        self.drop_1 = nn.Dropout(p=drop_prob)

        self.conv_separable_depth = nn.Conv2d(
            F1 * D, F1 * D, (1, 16),
            stride=1, bias=False,
            groups=F1 * D,
            padding=(0, 8)
        )
        self.conv_separable_point = nn.Conv2d(
            F1 * D, F2, (1, 1),
            stride=1, bias=False,
            padding=(0, 0)
        )
        self.bnorm_2 = nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3)
        self.elu_2 = Expression(elu)
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1, 8))
        self.drop_2 = nn.Dropout(p=drop_prob)

        # Dummy forward to infer shape
        with torch.no_grad():
            dummy_x = torch.ones((1, in_chans, input_window_samples, 1), dtype=torch.float32)
            dummy_edge_index = torch.ones(2, in_chans).long()
            x = self.ensuredims(dummy_x)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            
            B,F1,C,T = x.shape
            x = x.permute(0,2,3,1) # B, C, T, F1
            x = x.reshape(B*C*T,F1)
            
            x = self.spatial_gnn1(x, dummy_edge_index)
            x = self.spatial_gnn2(x, dummy_edge_index)
            
            x = x.reshape(B,C,T, x.shape[-1]) # B,C,T,F1*D
            x = x.permute(0,3,1,2) # B, F1*D, C, T

            x = self.depthwise(x)
            x = self.pool_1(x)
            x = self.drop_1(x)
            x = self.conv_separable_depth(x)
            x = self.conv_separable_point(x)
            x = self.bnorm_2(x)
            x = self.elu_2(x)
            x = self.pool_2(x)
            x = self.drop_2(x)
            n_out_virtual_chans = x.shape[2]
            if final_conv_length == "auto":
                self.final_conv_length = x.shape[3]

        self.conv_classifier = nn.Conv2d(
            F2, n_classes,
            (n_out_virtual_chans, self.final_conv_length),
            bias=True
        )
        self.softmax = nn.Softmax(dim=1)
        self.permute_back = Expression(_transpose_1_0)
        self.squeeze = Expression(squeeze_final_output)

        _glorot_weight_zero_bias(self)
        
    def forward(self, X, edge_index, edge_weight=None):
        x = self.ensuredims(X)
        x = self.dimshuffle(x)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        
        B,F1,C,T = x.shape
        x = x.permute(0,2,3,1) # B, C, T, F1
        x = x.reshape(B*C*T, F1) # (B*C*T, F1)

        # Expand the adjacency matrix for each time step
        edge_index_batched, edge_weight_batched = expand_batched_edges_over_time(edge_index, edge_weight, T, C, device = X.device)

        x = self.spatial_gnn1(x, edge_index_batched, edge_weight_batched)
        x = elu(x)
        x = self.spatial_gnn2(x, edge_index_batched, edge_weight_batched)
        x = elu(x)
            
        x = x.reshape(B,C,T, x.shape[-1]) # B,C,T,F1*D
        x = x.permute(0,3,1,2) # B, F1*D, C, T
        
        
        x = self.depthwise(x)
    
        # Convert to [batch, num_fm* D, 1, out_features]
        x = self.pool_1(x)
        x = self.drop_1(x)

        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bnorm_2(x)
        x = self.elu_2(x)
        x = self.pool_2(x)
        x = self.drop_2(x)

        x = self.conv_classifier(x)
        x = self.softmax(x)
        x = self.permute_back(x)
        x = self.squeeze(x)
        return x

def expand_batched_edges_over_time(edge_index, edge_weight, T, N, device='cpu'):
    """
    edge_index: [B, 2, E]
    edge_weight: [B, E]
    T: pasos de tiempo
    N: nodos por grafo

    Devuelve:
    - edge_index_out: [2, E * B * T]
    - edge_weight_out: [E * B * T]
    """
    B, _, E = edge_index.shape
    total = B * T

    # Expandimos edge_index y edge_weight
    edge_index_rep = edge_index.unsqueeze(1).repeat(1, T, 1, 1)  # [B, T, 2, E]
    edge_index_rep = edge_index_rep.permute(2, 0, 1, 3).reshape(2, -1)  # [2, E * B * T]

    edge_weight_rep = edge_weight.unsqueeze(1).repeat(1, T, 1).reshape(-1)  # [E * B * T]

    # Creamos offsets para ajustar los índices
    offsets = torch.arange(total, device=edge_index.device) * N  # [B*T]
    offsets = offsets.repeat_interleave(E)  # [E * B * T]

    edge_index_out = edge_index_rep.clone()
    edge_index_out[0] += offsets
    edge_index_out[1] += offsets

    return edge_index_out.to(device), edge_weight_rep.to(device)

def build_batch_vector(B, T, N, device=None):
    """
    Crea el vector 'batch' de tamaño [B*T*N] para GNNs como ChebConv,
    donde cada grafo es una combinación (b, t) y contiene N nodos.
    """
    return torch.arange(B * T, device=device).repeat_interleave(N)
