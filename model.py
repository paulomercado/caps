import torch
import torch.nn as nn
        

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3, args=None):
        super().__init__()

        self.output_size = output_size
        self.forecast_horizon = getattr(args, 'forecast_horizon', 1)

        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size * self.forecast_horizon)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.layer_norm(gru_out[:, -1, :])
        gru_pred = self.fc_out(out)

        if self.forecast_horizon > 1 and self.output_size > 1:
            gru_pred = gru_pred.view(-1, self.output_size, self.forecast_horizon)

        return gru_pred

