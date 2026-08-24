import torch
import torch.nn as nn

class ZNet(nn.Module):
    def __init__(self, n_channels, n_classes=2, F=4, kernel_time=31, dropout_rate=0.25):
        """
        Arquitectura Ultra-Ligera (~140 parámetros) para BCI.
        
        Args:
            n_channels (int): Número de canales EEG de entrada (ej. 3).
            n_classes (int): Número de clases a predecir (ej. 2 para Izq/Der).
            F (int): Número de filtros espaciales y temporales (mimética de CSP).
            kernel_time (int): Tamaño de la ventana temporal para filtrar frecuencias.
            dropout_rate (float): Tasa de dropout para regularización.
        """
        super(ZNet, self).__init__()

        # ==========================================
        # 1. ENCODER (Extracción de Representación)
        # ==========================================
        
        # Filtro Espacial: Aprende combinaciones lineales de los canales (como la matriz W en CSP)
        # Tamaño: (Canales de entrada) -> F canales espaciales
        self.spatial_conv = nn.Conv1d(in_channels=n_channels,
                                      out_channels=F,
                                      kernel_size=1,
                                      bias=False) # Sin bias para obligar a combinaciones puras

        # Filtro Temporal: Aprende a aislar Alpha/Beta.
        # groups=F hace que sea 'Depthwise', aplicando 1 filtro temporal por cada canal espacial.
        self.temporal_conv = nn.Conv1d(in_channels=F,
                                       out_channels=F,
                                       kernel_size=kernel_time,
                                       padding=kernel_time//2, # Padding para mantener la longitud de tiempo
                                       groups=F, 
                                       bias=False)

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)

        # ==========================================
        # 2. CLASIFICADOR (Lógica Neurofisiológica CSP)
        # ==========================================
        
        # Tras extraer características, el CSP original calcula la varianza y su logaritmo.
        # Haremos esas operaciones matemáticas en el forward pass, por lo que 
        # el clasificador final solo necesita ser una pequeña capa lineal.
        self.classifier = nn.Linear(F, n_classes)

        # ==========================================
        # 3. DECODER (Reconstrucción del Target)
        # ==========================================
        
        # Deshace el filtro espacial: Proyecta los F mapas latentes de vuelta a n_channels
        self.decoder_spatial = nn.Conv1d(in_channels=F,
                                         out_channels=n_channels,
                                         kernel_size=1,
                                         bias=False)

        # Suavizado temporal final para reconstruir la señal EEG filtrada
        self.decoder_temporal = nn.Conv1d(in_channels=n_channels,
                                          out_channels=n_channels,
                                          kernel_size=kernel_time,
                                          padding=kernel_time//2,
                                          groups=n_channels,
                                          bias=False)

    def forward(self, x):
        # Asumimos que x tiene forma: (Batch, Canales, Tiempo)
        
        # --- PASO 1: ENCODER ---
        z_s = self.spatial_conv(x)
        z_t = self.temporal_conv(z_s)
        
        # Z es nuestro ESPACIO LATENTE. Contiene la dinámica de tiempo filtrada.
        # Forma: (Batch, F, Tiempo)
        Z = self.elu(z_t) 

        # --- PASO 2: CLASIFICACIÓN TIPO CSP ---
        # 1. Elevamos al cuadrado para obtener "potencia" (Power)
        z_power = torch.square(Z)
        
        # 2. Promediamos a lo largo del tiempo (Global Average Pooling) -> Varianza
        z_var = torch.mean(z_power, dim=-1) # Forma: (Batch, F)
        
        # 3. Logaritmo para normalizar la distribución (fundamental en CSP)
        z_log_var = torch.log(torch.clamp(z_var, min=1e-6))
        
        # 4. Dropout y capa densa final
        z_drop = self.dropout(z_log_var)
        class_preds = self.classifier(z_drop) # Forma: (Batch, n_classes)

        # --- PASO 3: DECODER ---
        # A partir del espacio latente Z, intentamos reconstruir la señal original filtrada
        dec_s = self.decoder_spatial(Z)
        x_reconstructed = self.decoder_temporal(dec_s) # Forma: (Batch, Canales, Tiempo)

        # Retornamos las predicciones (para la Loss de clasificación) 
        # y la reconstrucción (para la Loss del Autoencoder)
        return class_preds, x_reconstructed