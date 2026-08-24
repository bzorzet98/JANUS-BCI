""" This code is implemented 
Liu, X., Shen, Y., Liu, J., Yang, J., Xiong, P., & Lin, F. (2020). 
Parallel spatial–temporal self-attention CNN-based motor imagery classification for BCI. 
Frontiers in neuroscience, 14, 587520."""

import numpy as np
import torch
from torch import nn
from torch.nn import Module,  Conv2d, ReLU, Parameter, Softmax, Dropout


class Time_Module(nn.Module):
    """ time attention module"""
    def __init__(self, in_dim, channel_num):
        super(Time_Module, self).__init__()
        self.channel_in = in_dim

        self.conv1 = Conv2d(in_channels=in_dim, out_channels=in_dim * 8, kernel_size=(1, 1))

        self.conv2 = Conv2d(in_channels=in_dim, out_channels=in_dim * 8, kernel_size=(1, 1))

        self.gamma = Parameter(torch.zeros(1))


        self.softmax = Softmax(dim=2)

    def forward(self, x):
        """
        :param x: input feature maps( B X C X H X W)
        :return: out : attention value + input feature
        """
        mini_batchsize, C, Height, Width = x.size()
        
        x1 = self.conv1(x)
        # Now we need to permute the tensor dimension (BxFxHxW) to (BxWxHxF)
        queryv = x1.permute(0,3,2,1)
        queryv_reshaped = queryv.reshape(mini_batchsize, Width, Height * 8)

        x2 = self.conv2(x)
        # Now we need to permute the tensor dimension (BxFxHxW) to (BxFxHxW)
        key = x2.permute(0,1,2,3)
        key_reshaped = key.reshape(mini_batchsize, Height * 8, Width) 

        # queryv = self.conv1(x).permute(0,3,1,2).reshape(mini_batchsize, Width, 8 * Height)
        # proj_query = self.conv1(x).permute(0,2,1,3).reshape(mini_batchsize, Height, 8 * Width)
        # proj_key = self.conv2(x).permute(0,1,3,2).reshape(mini_batchsize, 8 * Width, Height)
        #energy = torch.matmul(proj_query, proj_key)

        energy = torch.bmm(queryv_reshaped, key_reshaped) 

        max_H, _ = energy.max(dim=2, keepdim=True)
        min_H, _ = energy.min(dim=2, keepdim=True)
        temp_b = (energy - min_H)
        temp_c = (max_H - min_H) + 0.00000001
        energy = temp_b / temp_c
        attention = self.softmax(energy)

        #attention = attention.reshape(mini_batchsize, 1, Height, Height)

        #out = torch.matmul(attention, x)

        out = torch.bmm(x.squeeze(), attention) 

        out = self.gamma*out + x.squeeze()

        return out


class Channel_Module(Module):
    """ Channel attention module"""
    """ the implementation doesn't works, so we re adapt some parts of the code"""
    def __init__(self, in_dim = 1):

        super(Channel_Module, self).__init__()
        #self.chanel_in = in_dim

        self.conv1 = Conv2d(in_channels=in_dim, out_channels=in_dim * 8, kernel_size=(1, 1))

        self.conv2 = Conv2d(in_channels=in_dim, out_channels=in_dim * 8, kernel_size=(1, 1))

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=2)


    def forward(self, x):
        """
        :param x: input feature maps( B X C X H X W)
        :return: out : attention value + input feature
        """
        mini_batchsize, C, Height, Width = x.size()

        x1 = self.conv1(x)
        # Now we need to permute the tensor dimension (BxFxHxW) to (BxHxWxF)
        queryv = x1.permute(0,2,3,1)
        queryv_reshaped = queryv.reshape(mini_batchsize, Height, Width * 8)  # Reshape to [50, 3, 2048]
        # queryv = self.conv1(x).permute(0,3,1,2).reshape(mini_batchsize, Width, 8 * Height)

        x2 = self.conv2(x)
        # Now we need to permute the tensor dimension (BxFxHxW) to (BxWxFxH)
        key = x2.permute(0,3,1,2)
        key_reshaped = key.reshape(mini_batchsize, Width * 8, Height)       # Reshape to [50, 2048, 3]
        # key = self.conv2(x).reshape(mini_batchsize, 8 * Height, Width)
        
        # energy = torch.matmul(queryv, key)
        energy = torch.bmm(queryv_reshaped, key_reshaped) 

        max_H, _ = energy.max(dim=2, keepdim=True)
        min_H, _ = energy.min(dim=2, keepdim=True)
        temp_b = (energy - min_H)
        temp_c = (max_H - min_H)+0.00000001
        energy = temp_b / temp_c
        attention = self.softmax(energy)
        
        #attention = attention.reshape(mini_batchsize, 1, Width, Width)

        out = torch.bmm(attention, x.squeeze()) 
        # out = torch.matmul(x, attention.permute(0,1,3,2))

        out = self.gamma * out + x.squeeze()

        return out

class Attention(Module):
    def __init__(self, in_dim, channel_num):
        super(Attention, self).__init__()
        self.ca = Channel_Module(in_dim)
        self.ta = Time_Module(in_dim, channel_num)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 1, stride=1),
            torch.nn.BatchNorm2d(16, momentum=0.1, affine=True),
        )

    def forward(self, x):
        out1 = self.ca(x)
        out1 = out1.unsqueeze(1)
        out2 = self.ta(x)
        out2 = out2.unsqueeze(1)
        out = torch.cat((x, out1, out2), dim=1)
        return out


""" This part of the code is not implemented  or it is not specify in the github repository.
So we implemented it from the information of the paper."""

class FeatureClassificationModule(nn.Module):
    def __init__(self,
                spatial_kernel_size=(22,1), 
                temporal_kernel_size=(1,25), 
                n_outputs = 4,
                avg_kernel_size=(1, 75),
                avg_stride=(1, 15),
                last_conv_kernel_size=(1, 69),
                add_log_softmax=True):
        """Here we need to take care about the arguments. the las conv kernel size must bethe size of the outputs of previous layers"""
        super(FeatureClassificationModule, self).__init__()
        
        self.n_outputs = n_outputs
        self.nfeaturesmaps = int(n_outputs*10)

        # First Convolution Layer (Conv2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=self.nfeaturesmaps, 
                               kernel_size=temporal_kernel_size, 
                               stride=(1, 1))

        # Second Convolution Layer (Conv3)
        self.conv3 = nn.Conv2d(in_channels=self.nfeaturesmaps, out_channels=self.nfeaturesmaps, 
                               kernel_size=spatial_kernel_size, 
                               stride=(1, 1))
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(self.nfeaturesmaps)
        
        # Avg Pooling Layer
        self.avg_pool = nn.AvgPool2d(kernel_size=avg_kernel_size, stride=avg_stride)
        
        # Fourth Convolution Layer (Conv4)
        self.conv4 = nn.Conv2d(in_channels=self.nfeaturesmaps, 
                               out_channels=self.n_outputs, 
                               kernel_size=last_conv_kernel_size, 
                               stride=(1, 1))

        # Output LogSoftmax 
        if add_log_softmax:
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Apply first convolution layer
        x = self.conv2(x)

        # Apply second convolution layer
        x = self.conv3(x)
        
        # Apply batch normalization
        x = self.batch_norm(x)

        # Square the values
        x = torch.square(x)

        # Apply average pooling
        x = self.avg_pool(x)

        # Apply logarithm
        x = torch.log(x + 1e-6)  # adding small constant to avoid log(0)
        
        # Apply fourth convolution layer
        x = self.conv4(x)

        # Final softmax activation
        x= x.squeeze()
        x = self.softmax(x)
        
        return x


"""Complete the whole model"""

class SpatialTemporalAttention(nn.Module):
    def __init__(self,in_dim, channel_num,
                spatial_kernel_size=(22,1), 
                temporal_kernel_size=(1,25), 
                n_outputs = 4,
                avg_kernel_size=(1, 75),
                avg_stride=(1, 15),
                last_conv_kernel_size=(1, 69),
                add_log_softmax=True):
        super(SpatialTemporalAttention, self).__init__()
        self.in_dim = in_dim
        self.channel_num = channel_num

        self.spatial_kernel_size = spatial_kernel_size
        self.temporal_kernel_size = temporal_kernel_size
        self.n_outputs = n_outputs
        self.avg_kernel_size = avg_kernel_size
        self.avg_stride = avg_stride
        self.last_conv_kernel_size = last_conv_kernel_size

        self.add_log_softmax = add_log_softmax
        # Attention Module
        self.attention = Attention(in_dim=self.in_dim, channel_num=self.channel_num)
        
        # Feature Classification Module
        self.feature_classification = FeatureClassificationModule(  spatial_kernel_size=spatial_kernel_size,
                                                                    temporal_kernel_size=temporal_kernel_size,
                                                                    n_outputs=n_outputs,
                                                                    avg_kernel_size=avg_kernel_size,
                                                                    avg_stride=avg_stride,
                                                                    last_conv_kernel_size=last_conv_kernel_size,
                                                                    add_log_softmax=self.add_log_softmax)
        
    def forward(self, x):
        # Here we have to comprobe if x has 4 dimensions or 3 dimensions if have 3 we have to add 1 
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        # Apply attention module
        x = self.attention(x)
        
        # Apply feature classification module
        x = self.feature_classification(x)
        
        return x
