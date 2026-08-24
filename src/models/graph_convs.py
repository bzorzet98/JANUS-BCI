import torch 
import torch.nn as nn


def normalize_adjacency(A, self_loops=True):
    """
    Symmetric normalization of the adjacency matrix:
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    
    Args:
        A: adjacency tensor of shape (B, N, N)
        self_loops : boolean, if True add self-loops to the adjacency matrix

    Returns:
        A_hat: normalized adjacency matrix
    """
    B, N, _ = A.shape
    device = A.device

    A_hat = A.clone()
    if self_loops:
        # Extract the diagonal elements
        diag = A_hat.diagonal(dim1=1, dim2=2)  # shape: (B, N)
        # Mask only where there are not self-loops
        mask = (diag == 0).float()
        # Replace the diagonal with 1 + eps
        diag_new = diag + mask  # Sum
        # Rewrite the diagonal of A_hat
        A_hat[:, torch.arange(N), torch.arange(N)] = diag_new

    # Calculate degree matrix
    D = A_hat.sum(dim=-1)  # (B, N)
    D_inv_sqrt = torch.pow(D , -0.5)  # Avoid division by zero
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
    # Create the diagonal matrix D^{-1/2}
    D_inv_sqrt_mat = torch.diag_embed(D_inv_sqrt)  # (B, N, N)

    # Normalize: A_hat = D^{-1/2} A_hat D^{-1/2}
    A_hat = torch.bmm(D_inv_sqrt_mat, torch.bmm(A_hat, D_inv_sqrt_mat))

    return A_hat.to(device)

def laplacian_normalization(A, lambda_max=None):
    """
    Compute the normalized Laplacian matrix:
    L = I - D^{-1/2} (A + I) D^{-1/2}
    
    Args:
        A (torch.Tensor): Adjacency matrix of shape (B, N, N).
        eps (float): Small value to avoid division by zero.
    
    Returns:
        torch.Tensor: Normalized Laplacian matrix of shape (B, N, N).
    """
    B, N, _ = A.shape
    # 1. Eliminate the self-loops
    A_hat = A.clone()
    A_hat[:, torch.arange(A.shape[1]), torch.arange(A.shape[2])] = 0.

    # 2. Normalizamos
    A_hat = normalize_adjacency(A_hat, self_loops=False)

    # 3. Compute the normalized Laplacian
    I = torch.eye(N, device=A_hat.device).unsqueeze(0).expand(B, -1, -1)  # Shape: (B, N, N)
    L = I - A_hat  # Normalized Laplacian

    # 4. Compute the eigenvalues of the Laplacianç
    if lambda_max is None:
        lambda_max = 2.0 * L.max()

    # 5. Obtain the laplacian rescaled
    L_scaled =  (2.0 / lambda_max) * L
    
    # 6. Eliminate the self-loops
    L_scaled = L_scaled - I

    return L_scaled.to(A_hat.device)

class Simple_GCN(nn.Module):
    def __init__(self, in_features,  
                 out_features,  bias=True, normalize=True):
        super(Simple_GCN, self).__init__()

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            torch.Tensor(in_features, out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = None
        
        self.normalize = normalize
        self.reset_parameters()
        
    def forward(self, X, A):
        """
        Forward pass of the Simple_GCN model.
        Args:
            X (torch.Tensor): Input feature tensor of shape (B, N, F), where
                B is the batch size, N is the number of nodes, and F is the
                number of input features per node.
            A (torch.Tensor): Adjacency matrix tensor of shape (B, N, N) or (N, N).
                If the shape is (N, N), it will be unsqueezed to (1, N, N).
                If the batch size of A is 1 and X has a batch size greater than 1,
                A will be expanded to match the batch size of X.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: Output tensor of shape (B, N, O), where O is the number
                of output features per node.
        """
        if len(A.shape) == 2:
            A =  A.unsqueeze(0)
        if A.shape[0] == 1 and X.shape[0] > 1:
            A = A.expand(X.shape[0], -1, -1)
        
        XW = torch.einsum('bnf,fF->bnF', X, self.weight)
        # Normalize A and check the dimensions
        if self.normalize:
            A_ = normalize_adjacency(A)
        
        output = torch.einsum('bin,bnF->biF', A_, XW)

        if self.bias is not None:
            output = output + self.bias
        return output

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
class Group_GCN(nn.Module):
    """Group_GCN is a PyTorch module that implements a grouped graph convolutional layer. 
    This layer performs convolution operations on graph-structured data, allowing for 
    grouped processing of input channels and efficient computation using Einstein summation.
    Attributes:
        in_features (int): Number of input features per node.
        in_channels (int): Number of input channels.
        out_features (int): Number of output features per node.
        out_channels (int): Number of output channels.
        groups (int): Number of groups for grouped convolution. Default is 1.
        weight (torch.nn.Parameter): Learnable weight tensor of shape 
            (out_channels, in_channels // groups, in_features, out_features).
        bias (torch.nn.Parameter or None): Learnable bias tensor of shape 
            (out_channels, 1, out_features) if bias is True, otherwise None.
    Methods:
        __init__(in_features, in_channels, out_features, out_channels, groups=1, bias=True):
            Initializes the Group_GCN layer with the specified parameters.
        reset_parameters():
            Resets the layer's parameters using Xavier uniform initialization for weights 
            and zeros for biases.
        forward(X, A, *args, **kwargs):
            Performs the forward pass of the graph convolutional layer.
        forward_einsum(X):"""
        
    def __init__(self, in_features, 
                 in_channels, 
                 out_features, 
                 out_channels, 
                 groups=1, bias=True):
        """
        Initializes the custom layer with specified input and output features, channels, 
        and other optional parameters.
        Args:
            in_features (int): Number of input features per channel.
            in_channels (int): Number of input channels.
            out_features (int): Number of output features per channel.
            out_channels (int): Number of output channels.
            groups (int, optional): Number of groups for grouped convolution. Default is 1.
            bias (bool, optional): If True, includes a bias term. Default is True.
        Attributes:
            in_features (int): Number of input features per channel.
            in_channels (int): Number of input channels.
            out_features (int): Number of output features per channel.
            out_channels (int): Number of output channels.
            groups (int): Number of groups for grouped convolution.
            weight (torch.nn.Parameter): Learnable weights of shape 
                (out_channels, in_channels // groups, in_features, out_features).
            bias (torch.nn.Parameter or None): Learnable bias of shape 
                (out_channels, 1, out_features) if bias is True, otherwise None.
        Raises:
            AssertionError: If `in_channels` is not divisible by `groups`.
            AssertionError: If `out_channels` is not divisible by `groups`.
        """
        super().__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        self.in_features = in_features
        self.in_channels = in_channels        
        self.out_features = out_features
        self.out_channels = out_channels
        self.groups = groups
        
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, in_features, out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels,1, out_features))
        else:
            self.bias = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Resets the parameters of the layer.

        This method initializes the weights of the layer using the Xavier uniform 
        initialization method and sets the bias to zero if it exists.

        Initialization:
            - Weights: Xavier uniform distribution.
            - Bias: Zero (if bias is not None).

        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self,X, A, *args, **kwargs):
        """
        Forward pass of the graph convolutional layer.
        Args:
            X (torch.Tensor): Input feature tensor of shape (B, N, F), where
                B is the batch size, N is the number of nodes, and F is the
                number of input features per node.
            A (torch.Tensor): Adjacency matrix tensor of shape (B, N, N) or (N, N).
                If the shape is (N, N), it will be unsqueezed to (1, N, N).
                If the batch size of A is 1 and X has a batch size greater than 1,
                A will be expanded to match the batch size of X.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: Output tensor of shape (B, N, O), where O is the number
                of output features per node.
        """
        if len(A.shape) == 2:
            A =  A.unsqueeze(0)
        if A.shape[0] == 1 and X.shape[0] > 1:
            A = A.expand(X.shape[0], -1, -1)
        
        XW = self.forward_einsum(X)
        
        if self.bias is not None:
            XW = XW + self.bias.unsqueeze(0)
        # Normalize A and check the dimensions
        A_ = normalize_adjacency(A)
        output = torch.einsum('bij,bojf->boif', A_, XW)
        return output
  
    def forward_einsum(self,X):
        """
        Performs the forward pass of the layer using Einstein summation notation for 
        efficient tensor operations.
        Args:
            X (torch.Tensor): Input tensor of shape 
                (batch_size, in_channels, nodes, in_features), where:
                - batch_size: Number of samples in the batch.
                - in_channels: Number of input channels.
                - nodes: Number of nodes in the graph.
                - in_features: Number of input features per node.
        Returns:
            torch.Tensor: Output tensor of shape 
                (batch_size, out_channels, nodes, out_features), where:
                - out_channels: Number of output channels.
                - out_features: Number of output features per node.
        Notes:
            - The input tensor `X` is reshaped to group the input channels for grouped 
              convolution operations.
            - The weight tensor `self.weight` is reshaped to match the grouped 
              convolution dimensions.
            - The Einstein summation notation ('bgcnf,gocfF->bgonF') is used to 
              efficiently compute the output tensor by combining the input tensor 
              and the weight tensor.
        """
        batch_size, in_channels, nodes, in_features = X.shape
        X_ = X.reshape(batch_size, self.groups, in_channels // self.groups, nodes, in_features)
        W_ = self.weight.reshape(self.groups, self.out_channels // self.groups, 
                                 self.in_channels // self.groups, 
                                 in_features, self.out_features)
        
        XW = torch.einsum('bgcnf,gocfF->bgonF', X_, W_)
        
        return XW.reshape(batch_size, self.out_channels, nodes, self.out_features)
  
class Simple_ChebConv(nn.Module):
    def __init__(self, in_features, out_features, K, bias=True):
        super(Simple_ChebConv, self).__init__()
        self.K = K
        self.weight = nn.Parameter(torch.Tensor(K, in_features, out_features))        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = None
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, A):
        if len(A.shape) == 2:
            A =  A.unsqueeze(0)
        if A.shape[0] == 1 and X.shape[0] > 1:
            A = A.expand(X.shape[0], -1, -1)
        
        A_ = laplacian_normalization(A)

        B, N, Fin = X.shape
        
        Tx = torch.zeros((B, self.K, N, Fin)).to(self.weight.device)
        Tx[:, 0, :, :] = X
        if self.K > 1:
            Tx[:, 1, :, :] = torch.einsum('bij,bjf->bif', A_, X)
            for k in range(2, self.K):
                Tx[:, k, :, :] = 2 * torch.einsum('bij,bjf->bif', A_, Tx[:, k - 1, :, :]) - Tx[:, k - 2, :, :]
        
        y_ = torch.einsum('bkif,kfF->bkiF', Tx, self.weight)
        y = y_.sum(dim=1)
        
        return y

class Group_ChebConv(nn.Module):
    """Group_ChebConv is a PyTorch module that implements a grouped Chebyshev graph convolutional layer. 
    This layer performs graph convolutions using Chebyshev polynomials, allowing for efficient 
    message passing on graph-structured data.
    Attributes:
        groups (int): Number of groups for grouped convolution.
        K (int): Size of the convolutional kernel (order of Chebyshev polynomials).
        weight (torch.nn.Parameter): Learnable weights of the layer.
        bias (torch.nn.Parameter or None): Learnable bias of the layer (if enabled).
    Methods:
        __init__(in_features, in_channels, out_features, out_channels, K=1, groups=1, bias=True):
            Initializes the Group_ChebConv layer with the specified parameters.
        reset_parameters():
            Resets the layer's parameters using Xavier uniform initialization for weights 
            and zero initialization for bias (if bias is enabled).
        forward(X, A, *args, **kwargs):
            Performs the forward pass of the grouped Chebyshev graph convolution."""
    def __init__(self, in_features, 
                 in_channels, 
                 out_features, 
                 out_channels, 
                 K = 1,
                 groups=1, bias=True):
        """
        Initializes the graph convolutional layer.
        Args:
            in_features (int): Number of input features per node.
            in_channels (int): Number of input channels.
            out_features (int): Number of output features per node.
            out_channels (int): Number of output channels.
            K (int, optional): Size of the convolutional kernel. Default is 1.
            groups (int, optional): Number of groups for grouped convolution. Default is 1.
            bias (bool, optional): If True, includes a learnable bias parameter. Default is True.
        Raises:
            AssertionError: If `in_channels` is not divisible by `groups`.
            AssertionError: If `out_channels` is not divisible by `groups`.
        """
        super().__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        self.in_features = in_features
        self.in_channels = in_channels        
        self.out_features = out_features
        self.out_channels = out_channels
        self.groups = groups
        self.K = K
        
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, self.K, in_features, out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels,1, out_features))
        else:
            self.bias = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Resets the parameters of the layer.

        This method initializes the weights of the layer using Xavier uniform 
        initialization and sets the bias to zero if it exists.

        Initialization:
            - Weights: Xavier uniform initialization.
            - Bias: Zero initialization (if bias is not None).

        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self,X, A, *args, **kwargs):
        """
        Forward pass for the graph convolutional layer.
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, in_channels, nodes, in_features).
            A (torch.Tensor): Adjacency matrix or batch of adjacency matrices. 
                              Shape can be (nodes, nodes) or (batch_size, nodes, nodes).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, nodes, out_features).
        Notes:
            - If the adjacency matrix `A` is 2D, it is unsqueezed to add a batch dimension.
            - If `A` has a single batch but `X` has multiple batches, `A` is expanded to match the batch size of `X`.
            - The adjacency matrix is normalized using `laplacian_normalization`.
            - Chebyshev polynomials are computed up to order `K` for the graph convolution.
            - The weights are reshaped and used in a tensor contraction with the Chebyshev polynomials.
            - If a bias is defined, it is added to the output.
        """
        if len(A.shape) == 2:
            A =  A.unsqueeze(0)
        if A.shape[0] == 1 and X.shape[0] > 1:
            A = A.expand(X.shape[0], -1, -1)
        L_hat = laplacian_normalization(A)
        batch_size, in_channels, nodes, in_features = X.shape
        X_ = X.reshape(batch_size, self.groups, in_channels // self.groups, nodes, in_features)
        
        Tx = torch.zeros((batch_size, self.groups, in_channels // self.groups, 
                          self.K, nodes, in_features)).to(self.weight.device)
        Tx[:,:,:,0,:,:] = X_
        if self.K > 1:
            Tx[:,:,:,1,:,:] = torch.einsum('bin,bgcnf->bgcif', L_hat, X_ )
            for k in range(2, self.K):
                Tx[:,:,:,k,:,:] = 2 * torch.einsum('bin,bgcnf->bgcif', L_hat,Tx[:,:,:,k-1,:,:])-Tx[:,:,:,k-2,:,:]

        W_ = self.weight.reshape(self.groups, self.out_channels // self.groups, 
                                    self.in_channels // self.groups, 
                                    self.K, in_features, self.out_features)
    
        y_ = torch.einsum('bgcknf,gockfF->bgoknF', Tx, W_)
        
        y = y_.sum(dim=3) # sum over K
        y = y.reshape(batch_size, self.out_channels, nodes, self.out_features)
        
        if self.bias is not None:
            y = y + self.bias      
        return y
        
        