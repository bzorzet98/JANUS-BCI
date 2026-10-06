"""
Datasets de torch para MI-BCI. Cambio de esta pasada: NINGUNA de las 4
clases castea `y` ni tiene el parámetro `classification` -- todo el
control de dtype de y_true quedó centralizado en
Trainer_DL.target_dtype (ver sesión de diseño: evitar doble conversión
silenciosa entre dataset y trainer). `y` sale como tensor con el dtype
que traía de origen (o el dtype nativo de np.array si no era tensor
todavía) -- Trainer_DL._maybe_move_target es quien decide el cast final,
de forma explícita, no adivinada.

X sigue casteándose a torch.float acá (float32) -- eso NO es redundante
con Trainer_DL: el modelo consume X en su forward pass, y Trainer_DL solo
garantiza el dtype JUSTO ANTES de ese forward (model_infer), como defensa
final -- pero el dataset sigue siendo responsable de darle a X una forma y
tipo razonables desde el vamos, incluidas las transformaciones numéricas
(filtrado, keep_power) que si trabajaran en float64 podrían perder
precisión al truncarse recién en model_infer. y_true, en cambio, no pasa
por ninguna transformación numérica sensible en estos datasets -- por eso
no hay el mismo argumento para castearlo acá.
"""
import numpy as np
import torch


class MIBCI_SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float)
        else:
            self.X = X

        if isinstance(y, np.ndarray):
            self.y = torch.tensor(y)
        else:
            self.y = y

        self.n = X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.n


class MIBCI_SimpleDataset_MissingData(torch.utils.data.Dataset):
    """Simula canales EEG faltantes -- expande X a originals_channels y
    rellena según missing_data_strategy ("zeros", "mean", o
    "neighbourhood_interpolation", esta última incompleta a propósito,
    ver nota en __init__)."""

    def __init__(
        self, X, y, originals_channels, channels_to_keep,
        missing_data_strategy="zeros", adjacency_matrix_config=None,
    ):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float)

        self.originals_channels = originals_channels
        self.channels_to_keep = channels_to_keep
        self.missing_data_strategy = missing_data_strategy
        self.adjacency_matrix_config = adjacency_matrix_config

        if X.shape[1] < len(self.originals_channels):
            # X viene con menos canales que el set completo -- expandir y
            # rellenar los faltantes. Asume X ya es tensor (device de X.device).
            expanded_X = torch.zeros(
                (X.shape[0], len(self.originals_channels), X.shape[2]),
                dtype=torch.float, device=X.device,
            )
            present_indices = [self.originals_channels.index(ch) for ch in self.channels_to_keep]
            expanded_X[:, present_indices, :] = X
            self.X = expanded_X
        else:
            self.X = X

        if isinstance(y, np.ndarray):
            self.y = torch.tensor(y)
        else:
            self.y = y

        self.n = self.X.shape[0]
        self.transform_data()

    def transform_data(self):
        missing_channels = [ch for ch in self.originals_channels if ch not in self.channels_to_keep]
        missing_indices = [self.originals_channels.index(ch) for ch in missing_channels]
        present_indices = [self.originals_channels.index(ch) for ch in self.channels_to_keep]
        mask = torch.zeros(len(self.originals_channels), dtype=torch.bool)
        mask[present_indices] = True

        if self.missing_data_strategy == "zeros":
            pass  # ya están en cero por la expansión inicial
        elif self.missing_data_strategy == "mean":
            # Constante ÚNICA por trial: media de todos los canales
            # presentes, colapsando canal Y tiempo juntos -- mismo valor
            # rellena TODOS los canales faltantes en TODOS los instantes
            # de ese trial (confirmado con el usuario, no varía en el
            # tiempo).
            present_data = self.X[:, mask, :]  # (trials, n_present, time)
            mean_per_trial = present_data.reshape(present_data.shape[0], -1).mean(dim=1)  # (trials,)
            mean_per_trial = mean_per_trial.view(-1, 1, 1)  # para broadcasting
            self.X[:, missing_indices, :] = mean_per_trial.expand(-1, len(missing_indices), self.X.shape[2])
        elif self.missing_data_strategy == "neighbourhood_interpolation":
            # Incompleta a propósito -- self.A (matriz de adyacencia) no
            # está construida a partir de adjacency_matrix_config. Nunca
            # se usó en corridas reales (confirmado con el usuario) --
            # no completar sin pedido explícito.
            A_norm = self.A / (self.A.sum(dim=1, keepdim=True) + 1e-8)
            raise NotImplementedError("neighbourhood_interpolation está incompleta -- self.A nunca se construye.")

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.n


class FrequencyMaskingDataset(torch.utils.data.Dataset):
    """Filtra a UNA banda (freqcut, mask_type), con keep_power
    normalizando POR TRIAL Y CANAL -- referencia correcta, no tocar (ver
    verify_05_keep_power.py)."""

    def __init__(self, X, y, fs, freqcut, mask_type="bandstop", order=4, keep_power=True):
        from scipy.signal import butter, sosfiltfilt

        if isinstance(X, np.ndarray):
            X_ = X
        else:
            X_ = X.cpu().numpy() if isinstance(X, torch.Tensor) else X

        sos = butter(order, freqcut, btype=mask_type, fs=fs, output="sos")
        filtered_data = sosfiltfilt(sos, X_, axis=-1)
        filtered_data = np.ascontiguousarray(filtered_data)

        if keep_power:
            power = (X_ ** 2).sum(axis=-1)
            power_filtered = (filtered_data ** 2).sum(axis=-1)
            k = np.sqrt(power / power_filtered)
            k = k[:, :, np.newaxis]
            filtered_data = k * filtered_data

        device = X.device if isinstance(X, torch.Tensor) else None
        self.X = torch.tensor(filtered_data, dtype=torch.float, device=device)

        if isinstance(y, np.ndarray):
            self.y = torch.tensor(y)
        else:
            self.y = y

        self.n = self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.n


class MultiFrequencyBandMaskingDataset(torch.utils.data.Dataset):
    """Filtra en cascada por varias bandas (filters: dict de
    nombre->{freqcut, mask_type, order}), con keep_power normalizando POR
    TRIAL Y CANAL (fix aplicado -- antes normalizaba con un escalar
    global, ver verify_05_keep_power.py)."""

    def __init__(self, X, y, fs, filters, keep_power=True):
        from scipy.signal import butter, sosfiltfilt

        if isinstance(X, np.ndarray):
            data_to_process = X
        else:
            data_to_process = X.cpu().numpy() if isinstance(X, torch.Tensor) else X

        processed_data = data_to_process.copy()
        for key, config in filters.items():
            sos = butter(config["order"], config["freqcut"], btype=config["mask_type"], fs=fs, output="sos")
            processed_data = sosfiltfilt(sos, processed_data, axis=-1)
        processed_data = np.ascontiguousarray(processed_data)

        if keep_power:
            # Normalización POR TRIAL Y CANAL -- fix ya aplicado y
            # verificado (verify_05_keep_power.py), no volver a la
            # normalización global (bug original).
            power = (data_to_process ** 2).sum(axis=-1)
            power_filtered = (processed_data ** 2).sum(axis=-1)
            k = np.sqrt(power / power_filtered)
            k = k[:, :, np.newaxis]
            processed_data = k * processed_data

        device = X.device if isinstance(X, torch.Tensor) else None
        self.X = torch.tensor(processed_data, dtype=torch.float, device=device)

        if isinstance(y, np.ndarray):
            self.y = torch.tensor(y)
        else:
            self.y = y

        self.n = self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.n