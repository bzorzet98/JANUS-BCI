import torch.nn as nn
from torch.nn.functional import elu, sigmoid, relu
from torch_geometric.nn import ChebConv, GCNConv, SAGPooling
from torch_geometric.nn.pool import global_mean_pool, global_max_pool

from src.networks.eegnet import (Ensure4d, Expression, _transpose_to_b_1_c_0, 
                                 _transpose_1_0, _glorot_weight_zero_bias, 
                                 Conv2dWithConstraint, squeeze_final_output)

import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv, global_mean_pool

class SerialChebConvV2(nn.Module):
    def __init__(self, in_channels, out_channels, K, dropout_p=0.3):
        super().__init__()
        self.conv1 = ChebConv(in_channels, out_channels, K)
        self.conv2 = ChebConv(out_channels, out_channels, K)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.relu(x)
        return x

class SpatialGNNBlockV2(nn.Module):
    def __init__(self, in_features, out_features, K, num_feature_maps, D, 
                 process_time_as_instances=False, dropout_p=0.3, use_batchnorm=True):
        super().__init__()
        self.num_feature_maps = num_feature_maps
        self.process_time_as_instances = process_time_as_instances
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.D = D
        self.dropout_p = dropout_p
        self.use_batchnorm = use_batchnorm

        # GNN branch
        self.gnn_block_feature_maps = nn.ModuleList([
            nn.ModuleList([
                SerialChebConvV2(in_features, out_features, K, dropout_p=dropout_p)
                for _ in range(D)
            ])
            for _ in range(num_feature_maps)
        ])

        # Post reshape BatchNorm and Dropout
        total_channels = num_feature_maps * D
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(total_channels)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None):
        outputs = []
        for i in range(self.num_feature_maps):
            xi = x[:, i]  # [B, C, T]
            B, C, T = xi.shape

            xi_, edge_index_, edge_weight_, batch_ = self.reestructure_input_for_GraphConv(xi, edge_index, edge_weight)
            sub_outputs = []

            for j in range(self.D):
                out = self.gnn_block_feature_maps[i][j](xi_, edge_index=edge_index_, edge_weight=edge_weight_, batch=batch_)
                out = out.reshape(B, 1, C, T)
                out = out.mean(dim=2)  # [B, 1, T]
                sub_outputs.append(out)

            sub_outputs = torch.stack(sub_outputs, dim=1)   # [B, D, 1, T]
            outputs.append(sub_outputs)

        outputs = torch.stack(outputs, dim=1)   # [B, num_feature_maps, D, 1, T]
        outputs = outputs.reshape(B, self.num_feature_maps * self.D, 1, T)

        # BatchNorm and Dropout after reshaping
        x_reshaped = outputs.squeeze(2)  # [B, C, T]
        if self.use_batchnorm:
            x_reshaped = self.bn(x_reshaped)
        x_reshaped = self.dropout(x_reshaped)
        outputs = x_reshaped.unsqueeze(2)  # [B, C, 1, T]

        return outputs

    def reestructure_input_for_GraphConv(self, X, edge_index, edge_weight=None):
        B, C, T = X.shape
        if B != 1:
            X = X.reshape(B * C, T)
            edge_index_list, edge_weight_list = [], []
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.shape[1], dtype=edge_index.dtype, device=edge_index.device)
            for i in range(B):
                offset = i * C
                ei = edge_index[i] + offset
                edge_index_list.append(ei)
                edge_weight_list.append(edge_weight[i])
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_weight = torch.cat(edge_weight_list, dim=0)
            batch = torch.arange(B).repeat_interleave(C)
            return X, edge_index.to(X.device), edge_weight.to(X.device), batch.to(X.device)
        else:
            return X, edge_index, edge_weight, None






class SpatialGNNBlockV3(nn.Module):
    def __init__(self, in_features, out_features, K, num_feature_maps, D, 
                 process_time_as_instances=False, dropout_p=0.3, use_batchnorm=True):
        super().__init__()
        self.num_feature_maps = num_feature_maps
        self.process_time_as_instances = process_time_as_instances
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.D = D
        self.dropout_p = dropout_p
        self.use_batchnorm = use_batchnorm

        # GNN branch
        self.gnn_block_feature_maps = nn.ModuleList([
            nn.ModuleList([
                SerialChebConvV2(in_features, out_features, K, dropout_p=dropout_p)
                for _ in range(D)
            ])
            for _ in range(num_feature_maps)
        ])

    def forward(self, x, edge_index=None, edge_weight=None, batch=None):
        outputs = []
        for i in range(self.num_feature_maps):
            xi = x[:, i]  # [B, C, T]
            B, C, T = xi.shape

            xi_, edge_index_, edge_weight_, batch_ = self.reestructure_input_for_GraphConv(xi, edge_index, edge_weight)
            sub_outputs = []

            for j in range(self.D):
                out = self.gnn_block_feature_maps[i][j](xi_, edge_index=edge_index_, edge_weight=edge_weight_, batch=batch_)
                out = out.reshape(B, 1, C, T)
                sub_outputs.append(out)

            sub_outputs = torch.stack(sub_outputs, dim=1)   # [B, D, C, T]
            outputs.append(sub_outputs)

        outputs = torch.stack(outputs, dim=1)   # [B, num_feature_maps, D, C, T]
        outputs = outputs.reshape(B, self.num_feature_maps * self.D, C, T)

        return outputs

    def reestructure_input_for_GraphConv(self, X, edge_index, edge_weight=None):
        B, C, T = X.shape
        if B != 1:
            X = X.reshape(B * C, T)
            edge_index_list, edge_weight_list = [], []
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.shape[1], dtype=edge_index.dtype, device=edge_index.device)
            for i in range(B):
                offset = i * C
                ei = edge_index[i] + offset
                edge_index_list.append(ei)
                edge_weight_list.append(edge_weight[i])
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_weight = torch.cat(edge_weight_list, dim=0)
            batch = torch.arange(B).repeat_interleave(C)
            return X, edge_index.to(X.device), edge_weight.to(X.device), batch.to(X.device)
        else:
            return X, edge_index, edge_weight, None


class EEGNetv4_ChebConv_v3(nn.Module):

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
        drop_prob_2=0.25,
        use_batchnorm=True,
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

        # Fake forward to get size for GNN
        with torch.no_grad():
            dummy_input = torch.ones((1, in_chans, input_window_samples, 1), dtype=torch.float32)
            x = self.ensuredims(dummy_input)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            ntemporal_samples = x.shape[-1]

        # Graph spatial block
        self.spatial_gnn = SpatialGNNBlockV3(
            in_features=ntemporal_samples,
            out_features=ntemporal_samples,
            K=K,
            num_feature_maps=F1,
            D=D,
            dropout_p=drop_prob_2
        )

        self.depthwise = Conv2dWithConstraint(
                self.F1 * self.D,
                self.F1 * self.D,
                (self.in_chans, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=self.F1* self.D,
                padding=(0, 0),
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
            dummy_edge_index = torch.ones(2, 2).long()
            x = self.ensuredims(dummy_x)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            x = self.spatial_gnn(x, dummy_edge_index)
            x = self.depthwise(x)
            x = x.view(x.shape[0], -1, 1, x.shape[-1])
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

        #x , edge_index, edge_weight, batch = self.reestructure_input_for_GraphConv(x, edge_index, edge_weight)

        x = self.spatial_gnn(x, edge_index=edge_index, edge_weight=edge_weight)

        # Depthwise convolution
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















class EEGNetv4_ChebConv_v2(nn.Module):
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
        drop_prob_2=0.25,
        use_batchnorm=True,
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

        # Fake forward to get size for GNN
        with torch.no_grad():
            dummy_input = torch.ones((1, in_chans, input_window_samples, 1), dtype=torch.float32)
            x = self.ensuredims(dummy_input)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            ntemporal_samples = x.shape[-1]

        # Graph spatial block
        self.spatial_gnn = SpatialGNNBlockV2(
            in_features=ntemporal_samples,
            out_features=ntemporal_samples,
            K=K,
            num_feature_maps=F1,
            D=D,
            dropout_p=drop_prob_2,
            use_batchnorm=use_batchnorm
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
            dummy_edge_index = torch.ones(2, 2).long()
            x = self.ensuredims(dummy_x)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            x = self.spatial_gnn(x, dummy_edge_index)
            x = x.view(x.shape[0], -1, 1, x.shape[-1])
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

        #x , edge_index, edge_weight, batch = self.reestructure_input_for_GraphConv(x, edge_index, edge_weight)

        x = self.spatial_gnn(x, edge_index=edge_index, edge_weight=edge_weight)

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









class SerialChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(SerialChebConv, self).__init__()
        self.chebconv1 = ChebConv(in_channels, out_channels, K)
        self.relu = nn.ReLU()
        self.chebconv2 = ChebConv(out_channels, out_channels, K)
        
    def forward(self, x, edge_index=None, edge_weight=None, batch=None):
        x = self.chebconv1(x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        x = self.relu(x)
        x = self.chebconv2(x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        x = self.relu(x)
        return x
    
class SpatialGNNBlock(nn.Module):
    def __init__(self, in_features, out_features, K, num_feature_maps, D,process_time_as_instances=False ):
        super().__init__()
        self.num_feature_maps = num_feature_maps
        self.process_time_as_instances = process_time_as_instances
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.D = D
        self.num_feature_maps = num_feature_maps
        self.gnn_block_feature_maps = nn.ModuleList([
            nn.ModuleList([
                SerialChebConv(in_features, out_features, K)
                for _ in range(D)
            ])
            for _ in range(num_feature_maps)
        ])
            
    def forward(self, x, edge_index=None, edge_weight=None, batch=None):
        outputs = []
        for i in range(self.num_feature_maps):
            xi = x[:, i]  # [batch, C, T]
            B, C, T = xi.shape
            xi_, edge_index_, edge_weight_, batch_ = self.reestructure_input_for_GraphConv(xi, edge_index, edge_weight)
            sub_outputs = []
            for j in range(self.D):
                out = self.gnn_block_feature_maps[i][j](xi_, edge_index=edge_index_, edge_weight=edge_weight_, batch=batch_)
                
                out = out.reshape(B,1,C,T)
                out = out.mean(dim=2)  # [batch, 1, T]

                sub_outputs.append(out)
            sub_outputs = torch.stack(sub_outputs, dim=1)   # [B, D, 1, T]
            outputs.append(sub_outputs)
        outputs = torch.stack(outputs, dim=1)   # [B, num_feature_mapsm D, 1, T]
        outputs = outputs.reshape(B,self.num_feature_maps*self.D,1,T)
        return outputs
    
    def reestructure_input_for_GraphConv(self, X, edge_index, edge_weight=None):
        # Re-structure the input
        B, C, T = X.shape
        if self.process_time_as_instances:
            # Reshape X: [B, C, T] -> [B*T, C]
            X = X.permute(0, 2, 1).reshape(B * T, C)  # Cada fila es un grafo con C nodos y 1 feature por nodo

            # Expand edge_index and edge_weight for B*T grafos
            edge_index_list = []
            edge_weight_list = []
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.shape[1], dtype=edge_index.dtype, device=edge_index.device)

            for i in range(B * T):
                offset = i * C
                ei = edge_index + offset
                edge_index_list.append(ei)
                edge_weight_list.append(edge_weight)

            edge_index = torch.cat(edge_index_list, dim=1)
            edge_weight = torch.cat(edge_weight_list, dim=0)
            
            # Create batch index: B*T grafos con C nodos cada uno
            batch = torch.arange(B * T, device=X.device).repeat_interleave(C)

            return X, edge_index.to(X.device), edge_weight.to(X.device), batch.to(X.device)
        
        else:
            if B != 1:
                X = X.reshape(B * C, T)
                edge_index_list = []
                edge_weight_list = []
                if edge_weight is  None:
                    edge_weight = torch.ones(edge_index.shape[1], dtype=edge_index.dtype, device=edge_index.device)    
                for i in range(B):
                    offset = i * C
                    ei = edge_index[i] + offset  # [2, E]
                    ew = edge_weight[i]          # [E]
                    edge_index_list.append(ei)
                    edge_weight_list.append(ew)
                edge_index = torch.cat(edge_index_list, dim=1)      # [2, B*E]
                edge_weight = torch.cat(edge_weight_list, dim=0)  
                batch = torch.arange(B).repeat_interleave(C) 
                return X, edge_index.to(X.device), edge_weight.to(X.device), batch.to(X.device)
            else:
                return X, edge_index, edge_weight, None
        
class EEGNetv4_ChebConv_v0(nn.Module):
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

        # Fake forward to get size for GNN
        with torch.no_grad():
            dummy_input = torch.ones((1, in_chans, input_window_samples, 1), dtype=torch.float32)
            x = self.ensuredims(dummy_input)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            ntemporal_samples = x.shape[-1]

        # Graph spatial block
        self.spatial_gnn = SpatialGNNBlock(
            in_features=ntemporal_samples,
            out_features=ntemporal_samples,
            K=K,
            num_feature_maps=F1,
            D=D
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
            dummy_edge_index = torch.ones(2, 2).long()
            x = self.ensuredims(dummy_x)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            x = self.spatial_gnn(x, dummy_edge_index)
            x = x.view(x.shape[0], -1, 1, x.shape[-1])
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

        #x , edge_index, edge_weight, batch = self.reestructure_input_for_GraphConv(x, edge_index, edge_weight)

        x = self.spatial_gnn(x, edge_index=edge_index, edge_weight=edge_weight)

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


class SpatialGNNBlock_TimeInstances(nn.Module):
    def __init__(self, in_features, out_features, K, num_feature_maps, D):
        super().__init__()
        self.num_feature_maps = num_feature_maps
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.D = D
        self.num_feature_maps = num_feature_maps
        self.gnn_block_feature_maps = nn.ModuleList([
            nn.ModuleList([
                SerialChebConv(in_features, out_features, K)
                for _ in range(D)
            ])
            for _ in range(num_feature_maps)
        ])
            
    def forward(self, x, edge_index=None, edge_weight=None, batch=None):
        outputs = []
        for i in range(self.num_feature_maps):
            xi = x[:, i]  # [batch, C, T]
            B, C, T = xi.shape
            xi_, edge_index_, edge_weight_, batch_ = self.reestructure_input_for_GraphConv(xi, edge_index, edge_weight)
            # xi_  shape [B*T , C]
            sub_outputs = []
            for j in range(self.D):
                out = self.gnn_block_feature_maps[i][j](xi_, edge_index=edge_index_, edge_weight=edge_weight_, batch=batch_)
                out = out.reshape(B,T,C).permute(0, 2, 1) 
                out = out.reshape(B,1,C,T)  
                out = out.mean(dim=2)  # [batch, 1, T]
                sub_outputs.append(out)
            sub_outputs = torch.stack(sub_outputs, dim=1)   # [B, D, 1, T]
            outputs.append(sub_outputs)
        outputs = torch.stack(outputs, dim=1)   # [B, num_feature_mapsm D, 1, T]
        outputs = outputs.reshape(B,self.num_feature_maps*self.D,1,T)
        return outputs
    
    def reestructure_input_for_GraphConv(self, X, edge_index, edge_weight=None):
        B, C, T = X.shape  # X: [B, C, T]
        X = X.permute(0, 2, 1).reshape(B * T, C, 1)  # [B*T, C, 1]
        X = X.reshape(-1, 1)  # [B*T*C, 1]

        if B != 1:

            # Paso 2: expandir edge_index y edge_weight
            edge_index_list = []
            edge_weight_list = []

            # edge_weight original: [E] → convertir a float si no está
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float, device=edge_index.device)
            else:
                edge_weight = edge_weight.to(torch.float)

            for b in range(B):
                for t in range(T):
                    i = b * T + t  # índice actual del grafo
                    ei = edge_index[b]  # [2, E]
                    if edge_weight is not None:
                        ew = edge_weight[b]  # [E]
                        ew = ew.to(torch.float)
                    else:
                        ew = torch.ones(ei.shape[1], dtype=torch.float, device=X.device)

                    offset = i * C
                    ei = ei + offset  # aplicar offset
                    edge_index_list.append(ei)
                    edge_weight_list.append(ew)

            edge_index = torch.cat(edge_index_list, dim=1)      # [2, total_edges]
            edge_weight = torch.cat(edge_weight_list, dim=0)    # [total_edges]

            batch = torch.arange(B * T, device=X.device).repeat_interleave(C)  # [B*T*C]

            return X, edge_index.to(X.device), edge_weight.to(X.device), batch.to(X.device)
        else:
            # Paso 2: expandir edge_index y edge_weight
            edge_index_list = []
            edge_weight_list = []

            # edge_weight original: [E] → convertir a float si no está
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float, device=edge_index.device)
            else:
                edge_weight = edge_weight.to(torch.float)

            for i in range(B * T):
                offset = i * C  # cada grafo está desplazado C nodos respecto al anterior
                ei = edge_index + offset  # [2, E]
                edge_index_list.append(ei)
                edge_weight_list.append(edge_weight.clone())  # importante clonar si es constante

            edge_index = torch.cat(edge_index_list, dim=1)  # [2, E_total]
            edge_weight = torch.cat(edge_weight_list, dim=0)  # [E_total]

            # Paso 3: construir vector batch (PyG requiere saber a qué grafo pertenece cada nodo)
            batch = torch.arange(B * T, device=X.device).repeat_interleave(C)  # [B*T*C]

            return X, edge_index.to(X.device), edge_weight.to(X.device), batch.to(X.device)
                  
class EEGNetv4_ChebConv_v1(nn.Module):
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
        self.spatial_gnn = SpatialGNNBlock_TimeInstances(
            in_features=1,
            out_features=1,
            K=K,
            num_feature_maps=F1,
            D=D
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
            dummy_edge_index = torch.ones(2, 2).long()
            x = self.ensuredims(dummy_x)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
            x = self.bnorm_temporal(x)
            x = self.spatial_gnn(x, dummy_edge_index)
            x = x.view(x.shape[0], -1, 1, x.shape[-1])
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

        #x , edge_index, edge_weight, batch = self.reestructure_input_for_GraphConv(x, edge_index, edge_weight)

        x = self.spatial_gnn(x, edge_index=edge_index, edge_weight=edge_weight)

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














####################################################### EEGNET MODIFIED TO INCORPORATE A GRAPH CONVOLUTION LAYER ####################################################################
