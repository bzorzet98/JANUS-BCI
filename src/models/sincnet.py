import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SincConvMI(nn.Module):
    """
    Sinc-based convolution adapted for Multi-channel Motor Imagery EEG.
    Uses linear frequency spacing and applies filters independently per electrode.
    """
    def __init__(self, out_channels, kernel_size, sample_rate=250, min_low_hz=1, min_band_hz=2):
        super(SincConvMI, self).__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Linear spacing for EEG (focuses evenly on Mu, Beta, etc.)
        low_hz = min_low_hz
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        hz = np.linspace(low_hz, high_hz, out_channels)

        self.low_hz_ = nn.Parameter(torch.from_numpy(hz).float().view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.from_numpy(np.diff(hz, append=hz[-1])).float().view(-1, 1))

        # Hamming window
        n = torch.linspace(0, self.kernel_size - 1, steps=self.kernel_size)
        window = 0.54 - 0.46 * torch.cos(2 * np.pi * n / self.kernel_size)
        self.register_buffer("window", window)
        
        # Time axis
        t_right = torch.linspace(1, (self.kernel_size - 1) / 2, steps=int((self.kernel_size - 1) / 2)) / sample_rate
        self.register_buffer("t_right", t_right)

    def forward(self, x):
        # Expected input x shape: [Batch, Channels, Time]
        B, C, T = x.size()
        
        # 1. Reshape to [Batch * Channels, 1, Time] 
        # This allows us to apply the 1D temporal filters to each electrode independently.
        x = x.view(B * C, 1, T)

        # 2. Compute the Sinc filters dynamically based on learned parameters
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate/2)
        band = (high - low)

        f_times_t_low = torch.matmul(low, self.t_right.view(1, -1))
        f_times_t_high = torch.matmul(high, self.t_right.view(1, -1))

        n = (torch.sin(2 * np.pi * f_times_t_high) - torch.sin(2 * np.pi * f_times_t_low)) / (2 * np.pi * self.t_right)
        n = torch.cat([torch.flip(n, [1]), 2 * band, n], dim=1)
        n = n * self.window

        # Filter shape: [out_channels, 1, kernel_size]
        filters = n.view(self.out_channels, 1, self.kernel_size)
        
        # 3. Apply convolution
        out = F.conv1d(x, filters, stride=1, padding=self.kernel_size//2)
        
        # 4. Reshape back to [Batch, Channels * out_channels, Time]
        # Now every electrode has 'out_channels' number of filtered representations
        out = out.view(B, C * self.out_channels, T)
        
        return out

class SincNetMI(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, sample_rate=250):
        super().__init__()
        
        # --- 1. Temporal Filtering (SincNet) ---
        filters_per_channel = 16  # How many frequency bands to learn
        self.sinc = SincConvMI(
            out_channels=filters_per_channel, 
            kernel_size=65, # ~260ms window for 250Hz data
            sample_rate=sample_rate
        )
        
        # Total channels out of SincConv = Original Channels * filters_per_channel
        sinc_out_dim = n_channels * filters_per_channel
        self.bn1 = nn.BatchNorm1d(sinc_out_dim)
        
        # --- 2. Spatial / Temporal Feature Extraction ---
        # Standard CNN layers to find patterns across the filtered bands
        self.conv1 = nn.Conv1d(sinc_out_dim, 64, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(64)
        
        # --- 3. Information Compression ---
        self.pool = nn.AdaptiveAvgPool1d(10) # Compress time axis down to 10 points
        
        # --- 4. Classifier ---
        self.fc = nn.Sequential(
            nn.Linear(64 * 10, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # Input x: [Batch, Channels, Time]
        
        # Temporal filtering
        x = self.sinc(x)
        x = F.leaky_relu(self.bn1(x), 0.2)
        
        # CNN blocks
        x = self.conv1(x)
        x = F.leaky_relu(self.bn2(x), 0.2)
        
        x = self.conv2(x)
        x = F.leaky_relu(self.bn3(x), 0.2)
        
        # Pooling & Flattening
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        
        # Classification
        out = self.fc(x)
        return out
    
# class SincNetEncoder(nn.Module):
#     def __init__(self, sincconv_hparams, net_hparams):
#         super().__init__()

#         # 1. Standalone SincConv
#         self.sinc_conv = SincConv(
#             out_channels=sincconv_hparams['out_channels'],
#             kernel_size=sincconv_hparams['kernel_size'],
#             sample_rate=sincconv_hparams.get('sample_rate', 16000)
#         )

#         self.relu = nn.LeakyReLU(net_hparams["LeakyReluval"])
        
#         # 2. Normalization
#         if net_hparams.get("batch_norm", True):
#             self.norm1 = nn.BatchNorm1d(sincconv_hparams['out_channels'])
#             self.norm2 = nn.BatchNorm1d(net_hparams["cnn_channels"])
#         else:
#             # Note: nn.LayerNorm requires shape info, usually [Channels, Time]
#             self.norm1 = nn.GroupNorm(1, sincconv_hparams['out_channels'])
#             self.norm2 = nn.GroupNorm(1, net_hparams["cnn_channels"])

#         # 3. Conv Blocks
#         self.cnn1d = nn.Conv1d(
#             in_channels=sincconv_hparams['out_channels'],
#             out_channels=net_hparams["cnn_channels"],
#             kernel_size=net_hparams["cnn_kernelsize"],
#             padding=net_hparams["cnn_kernelsize"] // 2
#         )

#         layers = []
#         for _ in range(net_hparams["num_cnn_layers"]):
#             layers.append(nn.Conv1d(
#                 in_channels=net_hparams["cnn_channels"],
#                 out_channels=net_hparams["cnn_channels"],
#                 kernel_size=net_hparams["cnn_kernelsize"],
#                 padding=net_hparams["cnn_kernelsize"] // 2
#             ))
#             layers.append(nn.LeakyReLU(net_hparams["LeakyReluval"]))
#         self.cnnseq = nn.Sequential(*layers)

#         # 4. Pooling & Classifier
#         self.pool = nn.AdaptiveAvgPool1d(net_hparams["pool_size"])

#         self.classifier = nn.Sequential(
#             nn.Linear(net_hparams["pool_size"] * net_hparams["cnn_channels"], net_hparams["fc_size"]),
#             nn.ReLU(),
#             nn.Dropout(net_hparams["dropout"]),
#             nn.Linear(net_hparams["fc_size"], net_hparams["n_class"]),
#         )

#     def forward(self, x):
#         # x shape: [Batch, Time] or [Batch, 1, Time]
#         if x.dim() == 2:
#             x = x.unsqueeze(1)
            
#         x = self.sinc_conv(x)
#         x = self.relu(x)
#         x = self.norm1(x)
        
#         x = self.cnn1d(x)
#         x = self.cnnseq(x)
#         x = self.norm2(x)
        
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x