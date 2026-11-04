import torch
from torch import nn


class TinyNet(nn.Module):
    """
    Tiny CPU model for the dummy-project (PRD R21).

    A bank of spatial filters, the log-power of each filter over time, and a
    linear classifier. Input (batch, n_chans, n_times), output raw logits
    (batch, n_outputs), meant for CrossEntropyLoss. The log-power does not
    depend on n_times, so the model takes any trial length.
    """

    def __init__(self, n_chans: int, n_outputs: int, n_filters: int):
        super().__init__()
        self.spatial = nn.Conv1d(n_chans, n_filters, kernel_size=1, bias=False)
        self.classifier = nn.Linear(n_filters, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_power = torch.log(self.spatial(x).pow(2).mean(dim=-1))
        return self.classifier(log_power)
