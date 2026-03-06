import torch
import torch.nn as nn


class SimpleGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3, use_branches=True):
        """
        Simple GRU model for revenue forecasting.
        
        Args:
            input_size: Number of input features
            hidden_size: GRU hidden dimension
            output_size: Number of forecast targets (e.g., 3 for BIR, BOC, Other Offices)
            num_layers: Number of stacked GRU layers (default: 2)
            dropout: Dropout rate (default: 0.3)
            use_branches: If True, separate branch per target. If False, single output layer.
        """
        super(SimpleGRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_branches = use_branches
        
        # Single GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout before output
        self.dropout = nn.Dropout(dropout)
        
        if use_branches:
            # Separate branch for each output target
            self.branches = nn.ModuleList([
                nn.Linear(hidden_size, 1) 
                for _ in range(output_size)
            ])
        else:
            # Single shared output layer
            self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size)
        
        # Take last time step
        last_output = gru_out[:, -1, :]  # (batch, hidden_size)
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        if self.use_branches:
            # Pass through each branch and concatenate
            outputs = [branch(last_output) for branch in self.branches]
            output = torch.cat(outputs, dim=1)  # (batch, output_size)
        else:
            # Single output layer
            output = self.fc_out(last_output)  # (batch, output_size)
        
        return output
        
    
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3,
                 num_attention_heads=2, args=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.forecast_horizon = getattr(args, 'forecast_horizon', 1)  

        self.use_attention = getattr(args, 'use_attention', False)
        self.use_branches = getattr(args, 'use_branches', False)


        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)

        if self.use_attention:
            self.multihead_attention = nn.MultiheadAttention(
                hidden_size, num_attention_heads, dropout=dropout * 0.5, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

        # Output: predict all horizons at once
        if self.use_branches:
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.3),
                    nn.Linear(hidden_size // 2, self.forecast_horizon)  # h outputs per branch
                ) for _ in range(output_size)
            ])
        else:
            self.fc_out = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size * self.forecast_horizon)
            )

    def forward(self, x):
        gru_out, _ = self.gru(x)

        if self.use_attention:
            attn_out, _ = self.multihead_attention(gru_out, gru_out, gru_out)
            attn_out = self.attn_norm(attn_out + gru_out)
            context = attn_out[:, -1, :]
        else:
            context = gru_out[:, -1, :]

        out = self.layer_norm(context)

        if self.use_branches:
            gru_pred = torch.cat([branch(out) for branch in self.branches], dim=1)
        else:
            gru_pred = self.fc_out(out)

        # Reshape only for multi-output + multi-horizon
        if self.forecast_horizon > 1 and self.output_size > 1:
            gru_pred = gru_pred.view(-1, self.output_size, self.forecast_horizon)

        return gru_pred

