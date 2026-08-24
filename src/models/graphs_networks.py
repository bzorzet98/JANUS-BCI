import torch.nn as nn
from torch.nn.functional import elu, sigmoid, relu
from torch_geometric.nn import ChebConv, GCNConv, SAGPooling
from torch_geometric.nn.pool import global_mean_pool, global_max_pool

class EEG_GNN_structural_v0(nn.Module):
    def __init__(self, n_eeg_channels, nframes, nclasses, k=3, dropoutConv=False, dropoutMLP=False):

        super(EEG_GNN_structural_v0, self).__init__()

        # Block 1 : Structural Graph Convolution Layer
        self.cheb_conv1 = ChebConv(nframes, nframes//2, K=k)
        self.dropoutConv = nn.Dropout(p=dropoutConv) if dropoutConv else None
        self.cheb_conv2 = ChebConv(nframes//2, nframes//4, K=k)

        # BLOCK2: Multi Layer Perceptron
        in_features = int(n_eeg_channels * nframes // 4)
        self.fc1 = nn.Linear(in_features, in_features // 4)
        self.fc2 = nn.Linear(in_features // 4, nclasses)
        self.dropoutMLP = nn.Dropout(p=dropoutMLP) if dropoutMLP else None

        # Activation Function
        self.relu = nn.ReLU()

        # Softmax
        self.softmax = nn.Softmax(dim=1)



    def forward(self, input, edge_index, edge_weight):

        edge_index = edge_index[0] if len(edge_index.shape) == 3 else edge_index
        edge_weight = edge_weight[0] if len(edge_weight.shape) == 2 else edge_weight

        # Block 1: Graph Convolutional Neural Network
        x = self.cheb_conv1(input, edge_index = edge_index, edge_weight = edge_weight)
        x = self.relu(x)
        x = self.dropoutConv(x) if self.dropoutConv else x

        x = self.cheb_conv2(x, edge_index = edge_index, edge_weight = edge_weight)
        x = self.relu(x)
        x = self.dropoutConv(x) if self.dropoutConv else x
        
        # Resize 
        x = x.view(x.shape[0], -1) 
        
        # Block 2: Multi Layer Perceptron
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropoutMLP(x) if self.dropoutMLP else x
        x = self.fc2(x)

        # Softmax
        x = self.softmax(x)
        return x


class SimpleGraphNetwork(nn.Module):
    def __init__(self, nframes, nclasses, dropoutMLP=False, dropoutConv=False):
        super(SimpleGraphNetwork, self).__init__()
        # BLOCK1: Graph Convolutional Neural Network
        self.gcn = GCNConv(nframes, nframes//2)  
        self.gcn2 = GCNConv(nframes//2, nframes//4) 
        self.dropoutConv = nn.Dropout(p=dropoutConv) if dropoutConv else None
        # BLOCK2: Multi Layer Perceptron
        self.fc1 = nn.Linear(nframes // 4, nframes // 8)
        self.fc2 = nn.Linear(nframes // 8, nclasses)
        self.dropoutMLP = nn.Dropout(p=dropoutMLP) if dropoutMLP else None
        # Activation Function
        self.relu = nn.ReLU()
        # Softmax
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self, input, edge_index, edge_weight):

        edge_index = edge_index[0] if len(edge_index.shape) == 3 else edge_index
        edge_weight = edge_weight[0] if len(edge_weight.shape) == 2 else edge_weight

        # Block 1: Graph Convolutional Neural Network
        x = self.gcn(input, edge_index = edge_index, edge_weight = edge_weight)
        x = self.relu(x)
        x = self.dropoutConv(x) if self.dropoutConv else x
        x = self.gcn2(x, edge_index = edge_index, edge_weight = edge_weight)
        x = self.relu(x)
        x = self.dropoutConv(x) if self.dropoutConv else x
        
        # Global Mean Pooling
        x = x.mean(dim=1)
        
        # Block 2: Multi Layer Perceptron
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropoutMLP(x) if self.dropoutMLP else x
        x = self.fc2(x)

        # Softmax
        x = self.softmax(x)
        return x

class SimpleSpectralGraphNetwork(nn.Module):
    def __init__(self, nframes, K, nclasses, dropoutMLP=False, dropoutConv=False):
        super(SimpleSpectralGraphNetwork, self).__init__()
        
        # Definir las capas ChebConv
        self.chebconv1 = ChebConv(nframes, nframes, K=K)  # K es el orden del polinomio de Chebyshev
        self.chebconv2 = ChebConv(nframes, nframes // 2, K=K)
        self.dropoutConv = nn.Dropout(p=dropoutConv) if dropoutConv else None
        # Definir la red MLP
        self.fc1 = nn.Linear(nframes // 2, nframes // 4)
        self.fc2 = nn.Linear(nframes // 4, nframes // 8)
        self.fc3 = nn.Linear(nframes // 8, nclasses)
        self.dropoutMLP = nn.Dropout(p=dropoutMLP) if dropoutMLP else None
        # Función de activación
        self.sigmoid = nn.Sigmoid()
        #Softmax
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input, edge_index, edge_weight):
        # Check edge_index and edge_weight
        edge_index = edge_index[0] if len(edge_index.shape) == 3 else edge_index
        edge_weight = edge_weight[0] if len(edge_weight.shape) == 2 else edge_weight
        # Aplicar las capas ChebConv
        x = self.chebconv1(input, edge_index = edge_index, edge_weight = edge_weight)
        x = self.sigmoid(x)
        x = self.dropoutConv(x) if self.dropoutConv else x
        x = self.chebconv2(x, edge_index = edge_index, edge_weight = edge_weight)
        x = self.sigmoid(x)
        x = self.dropoutConv(x) if self.dropoutConv else x
        
        # Aplicar el global mean pooling en la dimensión de nodos
        x = x.mean(dim=1)
        
        # Pasar por la red MLP
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.dropoutMLP(x) if self.dropoutMLP else x
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.dropoutMLP(x) if self.dropoutMLP else x
        x = self.fc3(x)

        #Softmax
        x = self.softmax(x)
        
        return x