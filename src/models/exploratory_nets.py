import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SmartMI_BCINet(nn.Module):
    def __init__(self, nchans, tchunks, nfreqs, spectral_conv_filters = 8, freqs_kernel_size=3,
                 dropout_prob=0.3):
        super().__init__()
        
        self.nchans = nchans
        self.tchunks = tchunks
        
        self.nfreqs = nfreqs
        self.freqs_kernel_size = freqs_kernel_size
        self.spectral_conv_filters = spectral_conv_filters
        self.dropout_prob = dropout_prob
        
        # Compute the output size after the spectral convolution
        freq_dim_output_spectral_block = compute_output_size(nfreqs,freqs_kernel_size, padding=freqs_kernel_size // 2)

        # For each chunk we create different branch of processing
        self.dropout = nn.Dropout(self.dropout_prob)
        self.spectral_conv_blocks_1 = []
        self.spectral_conv_blocks_2 = []
        self.spectral_conv_blocks_3 = []
        self.batchnorm_blocks = []
        self.attention_block = SpectralAttention(channels=nchans, 
                                                 freqs=freq_dim_output_spectral_block, 
                                                 fmap=spectral_conv_filters)
        for t in range(self.tchunks):
            self.spectral_conv_blocks_1.append(
                nn.Conv2d(in_channels = 1, out_channels = self.spectral_conv_filters,
                        kernel_size = (1, self.freqs_kernel_size), 
                        padding=(0, self.freqs_kernel_size // 2),
                        bias=True,
                        padding_mode='reflect')
            )
            self.spectral_conv_blocks_2.append(
                nn.Conv2d(in_channels = self.spectral_conv_filters, 
                          out_channels = self.spectral_conv_filters,
                            kernel_size = (1, self.freqs_kernel_size), 
                            padding=(0, self.freqs_kernel_size // 2),
                            bias=True,
                            padding_mode='reflect')
            )
            self.spectral_conv_blocks_3.append(
                nn.Conv2d(in_channels = self.spectral_conv_filters, 
                          out_channels = 1,
                            kernel_size = (1, self.freqs_kernel_size), 
                            stride = (1, 2),
                            padding=(0, 1),
                            bias=True)
            )
            self.batchnorm_blocks.append(nn.BatchNorm2d(self.spectral_conv_filters, 
                                                        eps=1e-5, momentum=0.1))
            
            
        # Simple spectral attention
        self.spectral_attention = SimpleSpectralAttention(nchans, tchunks, freq_dim)

        # LSTM layer
        self.hidden_size = freq_dim
        self.lstm = nn.LSTM(input_size=freq_dim, hidden_size=freq_dim,
                            num_layers=1, bidirectional=True, batch_first=True)
        
        # Average polling layer
        self.learned_avg_pool = nn.Conv2d(in_channels=tchunks, out_channels=tchunks,
                                          kernel_size=(nchans,1),
                                          padding=(0, 0), groups=tchunks, bias=False)
        # MLP for each tchunk
        self.hidden_size_mlp = freq_dim // 2
        self.conv_mlp_tchunk = GroupedLinear(
            in_features=freq_dim*2, out_features= self.hidden_size_mlp,
            bias=True, groups=tchunks
        )
        
        # Simple temporal attention
        self.temporal_attention = SimpleTemporalAttention(tchunks,  self.hidden_size_mlp, 
                                                          bias=True)

        # Classifier
        self.classifier = nn.Linear(self.hidden_size_mlp, 2)  # Assuming binary classification

        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def forward(self, x):
        # x: [B, C, F, T]
        B, C, f, T = x.size()
        # Permute to: [B, T, C, F]
        x = x.permute(0, 3, 1, 2)  # [B, T, C, F]
        
        for t in range(self.tchunks):
            x_t = x[:, t, :, :]  # Extract the t-th chunk: [B, t, C, F]
            x_t = x_t.unsqueeze(1)  # Add channel dimension: [B, 1, C, F]
            
            # Apply spectral convolution block
            x_t = self.spectral_conv_blocks_1[t](x_t)

            # Dropout
            x_t = self.dropout(x_t)
            
            # Apply second spectral convolution block
            x_t = self.spectral_conv_blocks_2[t](x_t)
            
            # Apply batch normalization
            x_t = self.batchnorm_blocks[t](x_t)
            
            # Apply attention
            x_t, attention_coefs = self.attention_block(x_t)
            
            
        # Apply spectral convolution block
        x = self.spectral_conv_block(x) # x: [B*T, spectral_conv_filters, C, F']

        # Apply batch normalization
        x = self.batchnorm(x)
        # x: [B, C, F', T]
        x, attention_coefs = self.spectral_attention(x)
        # Permute to: [B, C, T, F']
        x = x.permute(0, 1, 3, 2)
        # Resize to: [B * C, T, F']
        x = x.reshape(x.size(0)*x.size(1), x.size(2), x.size(3))
        # h0, c0 = (torch.zeros(2, x.size(0), self.hidden_size).to(x.device),
        #             torch.zeros(2, x.size(0), self.hidden_size).to(x.device))
        x, (h_n, c_n) = self.lstm(x)
        x = F.relu(x)
        
        # Reshape back to: [B, C, T, F']
        x = x.reshape(B, C, T, -1)

        # Permute to: [B, T, C, F']
        x = x.permute(0, 2, 1, 3)

        # Apply pooling
        x = self.learned_avg_pool(x) # x: [B, T, 1, F']

        # MLP for each tchunk
        x = x.squeeze(2)  # Remove the singleton dimension: [B, T, F']
        x = self.conv_mlp_tchunk(x)  # Apply MLP: [B, T, F/2]
        x = F.relu(x)  # Apply Softplus activation
        
        # Apply temporal attention
        x, attention_coefs = self.temporal_attention(x)
        # x: [B, F/2]
        # Apply classifier
        x = self.classifier(x)
        return x


class SpectralConvBlock(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 8, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels = self.in_channels, out_channels = self.out_channels,
                              kernel_size = (1, self.kernel_size), 
                                padding=(0, self.kernel_size // 2),
                                bias=True,
                             padding_mode='reflect')

        self.activation_function = nn.ReLU()

    def forward(self, x):
        # x: [B, 1, C, F]
        
        # Check input shape
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"Expected input shape [B, 1, C, F], got {x.shape}")  
        # Convolve
        x = self.conv(x) # x: [B, out_channels, C, F']
        
        # Apply activation function
        x = self.activation_function(x)
        
        return x

class SpectralAttention(nn.Module):
    def __init__(self, channels, freqs, fmap=8):
        super().__init__()
        self.att_weight = nn.Parameter(torch.randn(1, fmap, channels, freqs))  # [1, 8, C, F]

    def forward(self, x):
        # x: [B, fmap, C, F]
        scores = x * self.att_weight  
        att_coefs = torch.softmax(scores, dim=-1)  # [B, 8, C, F]
        out = x * att_coefs
        return out, att_coefs


class SimpleSpectralAttention(nn.Module):
    def __init__(self, nchans, tchunks, nfreqs, bias=False):
        super().__init__()
        self.nchans = nchans
        self.tchunks = tchunks
        self.nfreqs = nfreqs
        self.weight = nn.Parameter(
            torch.Tensor(nchans, nfreqs, tchunks)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(nchans, nfreqs, tchunks))
        else:
            self.register_parameter('bias', None)    
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        
    def forward(self, x):
        # x: [B, C, F, T]
        scores = torch.einsum('bcft,cft->bcft', x, self.weight)
        if self.bias is not None:
            scores += self.bias
        attention_coefs = F.softmax(scores, dim=-2) # Softmax over the frequency dimension
        out = torch.einsum('bcft,bcft->bcft', x, attention_coefs)
        return out, attention_coefs
    
    
    
    
    
    +
    
class SimpleTemporalAttention(nn.Module):
    def __init__(self, tchunks, nfreqs, bias=False):
        super().__init__()
        self.tchunks = tchunks
        self.nfreqs = nfreqs
        self.weight = nn.Parameter(
            torch.Tensor(tchunks, nfreqs)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(tchunks, nfreqs))
        else:
            self.register_parameter('bias', None)
            
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

        
    def forward(self, x):
        # x: [B, T, F]
        scores = torch.einsum('btf,tf->btf', x, self.weight)
        if self.bias is not None:
            scores += self.bias
        # Sum over the frequency dimension
        scores_sum = scores.sum(dim=-1)  # [B, T]

        # Softmax over the temporal dimension
        attn_coefs = torch.softmax(scores_sum, dim=1)  # [B, T]

        # Apply attention: weighted sum over the temporal dimension
        out = torch.einsum('bt,btf->bf', attn_coefs, x)  # [B, F]

        return out, attn_coefs
    
    
    
    

class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, groups=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups

        self.weight = nn.Parameter(torch.Tensor(groups, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(groups, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: [B, G, in_features]
        B, G, I = x.shape
        assert G == self.groups, f"Esperaba {self.groups} grupos, recibí {G}"
        # salida: [B, G, out_features]
        out = torch.einsum('bgi,goi->bgo', x, self.weight)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        return out

    
    
def compute_output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size + 2 * padding - kernel_size) // stride + 1