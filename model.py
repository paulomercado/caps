import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,
                 dropout, activation_fn, activation_fn1, norm_layer_type):
       super(GRUModel, self).__init__()
       self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
       self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # take last time step
        out = self.fc(out)
        return out