import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)
    
class YourDefaultModelClass(nn.Module):
    def __init__(self, input_size):
        super(YourDefaultModelClass, self).__init__()
        pass

    def forward(self, x):
        return x
