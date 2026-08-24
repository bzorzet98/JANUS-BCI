import torch
import numpy as np
from typing import Union
from scipy.signal import butter, sosfiltfilt
import numpy as np

# Create a simple torch dataset
class MIBCI_SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, classification = True, device = 'cpu'):
        # Check if X is a tensor or numpy array
        if isinstance(X,np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float).to(device)
        else:
            self.X = X
        if isinstance(y,np.ndarray):
            if classification:
                self.y = torch.tensor(y, dtype=torch.long).to(device)
            else:
                self.y = torch.tensor(y, dtype=torch.float).to(device)
        else:
            if classification:
                self.y = y.long().to(device)
            else:
                self.y = y.float().to(device)
        self.n = X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.n
    
#%%###########  DATASETS FOR MISSING DATA ##############
class MIBCI_SimpleDataset_MissingData(torch.utils.data.Dataset):
    def __init__(self, X, y, classification=True,
                 originals_channels=None,
                 channels_to_keep=None,
                 missing_data_strategy="zeros",
                 adjacency_matrix_config=None):
        
        self.originals_channels = originals_channels
        self.channels_to_keep = channels_to_keep
        self.missing_data_strategy = missing_data_strategy
        self.adjacency_matrix_config = adjacency_matrix_config
        # If input X has fewer channels than originals_channels, expand it
        if X.shape[1] < len(self.originals_channels):
            # Create expanded tensor with zeros
            expanded_X = torch.zeros((X.shape[0], len(self.originals_channels),X.shape[2]), dtype=torch.float, device=X.device)

            # Map channels_to_keep to their indices in originals_channels
            keep_indices = [self.originals_channels.index(ch) for ch in channels_to_keep]
            # Insert original data into expanded positions
            if isinstance(X, np.ndarray):
                expanded_X[:, keep_indices, :] = torch.tensor(X, dtype=torch.float)
            else:
                expanded_X[:, keep_indices, :] = X
            self.X = expanded_X
        else:
            # Normal case, just convert to tensor if needed
            if isinstance(X, np.ndarray):
                self.X = torch.tensor(X, dtype=torch.float)
            else:
                self.X = X

        # Process y
        if isinstance(y, np.ndarray):
            if classification:
                self.y = torch.tensor(y, dtype=torch.long)
            else:
                self.y = torch.tensor(y, dtype=torch.float)
        else:
            if classification:
                self.y = y.long()
            else:
                self.y = y.float()

        self.n = self.X.shape[0]
        self.transform_data()

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.n

    def transform_data(self):
        # Find missing channels = originals_channels not in channels_to_keep
        missing_channels = [ch for ch in self.originals_channels if ch not in self.channels_to_keep]
        missing_indices = [self.originals_channels.index(ch) for ch in missing_channels]

        # Mask for missing channels
        mask = torch.ones(self.X.shape[1], dtype=torch.bool)
        mask[missing_indices] = False

        if self.missing_data_strategy == "zeros":
            self.X[:, missing_indices] = 0.
        elif self.missing_data_strategy == "mean":
            mean_values = self.X[:, mask].mean(dim=(1, 2), keepdim=True)
            self.X[:, missing_indices] = mean_values

class FrequencyMaskingDataset(torch.utils.data.Dataset):

    """
    Dataset that masks time series data based on a specified frequency band.
    Can perform either bandpass or band-reject filtering.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, fs: int,
                 freqcut: Union[list[float], float],
                 mask_type: str = 'bandpass', order: int = 8,
                 keep_power: bool = True):
        """
        Args:
            X (np.ndarray): The feature data (trials, channels, time).
            y (np.ndarray): The labels.
            fs (int): The sampling rate of the data.
            lowcut (float): The lower cutoff frequency for the band.
            highcut (float): The upper cutoff frequency for the band.
            mask_type (str): 'bandpass' to keep the band, 'bandstop' to remove it.
            order (int): The order of the Butterworth filter.
        """
        self.X = X
        self.y = y
        self.n_samples = self.X.shape[0]
        self.fs = fs
        self.freqcut = freqcut
        self.mask_type = mask_type
        self.order = order
        self.keep_power = keep_power
        self._transform_data()
        self.y = torch.tensor(self.y , dtype=torch.long) if isinstance(self.y ,np.ndarray) else self.y .long()
        self.X = torch.tensor(self.X, dtype=torch.float) if isinstance(self.X,np.ndarray) else self.X.float()


    def _transform_data(self):
        from scipy.signal import butter, filtfilt
        """
        Applies a bandpass or band-reject filter to the dataset.
        """
        nyq = 0.5 * self.fs

        freqcut = np.array(self.freqcut) / nyq

        b, a = butter(self.order, freqcut, btype=self.mask_type)

        # Apply the filter with zero-phase distortion
        X_ = self.X.cpu().numpy() if isinstance(self.X, torch.Tensor) else self.X
        filtered_data = filtfilt(b, a, X_)
        filtered_data = np.ascontiguousarray(filtered_data)
        if self.keep_power:
            power = (X_**2).sum(axis=-1)
            power_filtered = (filtered_data**2).sum(axis=-1)
            k = np.sqrt(power / power_filtered)
            k = k[:, :, np.newaxis]
            filtered_data = k * filtered_data
        self.X = torch.tensor(filtered_data, dtype=torch.float).to(self.X.device)


    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


class MultiFrequencyBandMaskingDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, fs: int,
                 filters: dict, keep_power: bool = True): # Note: Order 8 suggested
        self.X = X
        self.y = y
        self.n_samples = self.X.shape[0]
        self.fs = fs
        self.filters = filters
        self.keep_power = keep_power
        self._transform_data()
        self.y = torch.tensor(self.y , dtype=torch.long) if isinstance(self.y ,np.ndarray) else self.y .long()
        self.X = torch.tensor(self.X, dtype=torch.float) if isinstance(self.X,np.ndarray) else self.X.float()


    def _transform_data(self):
        # 1. Convert to numpy
        data_to_process = self.X.cpu().numpy() if isinstance(self.X, torch.Tensor) else self.X
        # Make a copy so we don't overwrite the original unexpectedly
        processed_data = data_to_process.copy()

        # 2. Get filter configurations
        filters_name = list(self.filters.keys())

        # 3. SEQUENTIAL FILTERING (The "Chain" approach)
        # Instead of summing, each filter acts on the result of the previous one
        for key in filters_name:
            config = self.filters[key]
            freqcut = config.get('freqcut')
            order = config.get('order')
            mask_type = config.get('mask_type')

            # Use SOS for stability and precision
            sos = butter(order, freqcut, btype=mask_type, fs=self.fs, output='sos')

            # Apply to the SAME data variable (cascading)
            processed_data = sosfiltfilt(sos, processed_data, axis=-1)

        # 4. Power Normalization (Optional but kept as per your logic)
        processed_data = np.ascontiguousarray(processed_data)
        if self.keep_power:
            original_power = (data_to_process**2).sum(axis=-1)
            processed_power = (processed_data**2).sum(axis=-1)

            scaling_factor = np.ones_like(processed_power)
            valid = processed_power > 1e-10
            scaling_factor[valid] = np.sqrt(original_power[valid] / processed_power[valid])
            scaling_factor = scaling_factor[:, :, np.newaxis]
            processed_data = scaling_factor * processed_data

        # 5. Back to Torch
        self.X = torch.tensor(processed_data, dtype=torch.float).to(self.X.device) if isinstance(self.X, torch.Tensor) else torch.tensor(processed_data, dtype=torch.float)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]