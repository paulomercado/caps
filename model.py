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
                 num_attention_heads=2, args = None):
        """
        GRU model with multiple literature-backed improvements:
        - Residual connections (He et al., 2016)
        - Multi-head attention (Vaswani et al., 2017)
        - Layer normalization (Ba et al., 2016)
        - Squeeze-and-Excitation (Hu et al., 2018)
        
        Args:
            input_size: Number of input features
            hidden_size: GRU hidden dimension
            output_size: Number of forecast targets
            num_layers: Number of GRU layers
            dropout: Dropout rate
            use_branches: Use separate branches for each output
            use_attention: Use multi-head attention
            use_residual: Use residual connections
            num_attention_heads: Number of attention heads
            use_se: Use Squeeze-and-Excitation blocks
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size  
        self.num_layers = num_layers    

        self.use_attention = args.use_attention if hasattr(args, 'use_attention') else False
        self.use_branches = args.use_branches if hasattr(args, 'use_branches') else True
        self.use_se = args.use_se if hasattr(args, 'use_se') else False
        
        # GRU backbone
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.use_linear_residual = args.use_linear_residual if hasattr(args, 'use_linear_residual') else False

        if self.use_linear_residual:
            self.linear_residual = nn.Linear(input_size, output_size)
        # Multi-head attention (Vaswani et al., 2017)
        if self.use_attention:
            self.multihead_attention = nn.MultiheadAttention(
                hidden_size, num_attention_heads, dropout=dropout*0.5, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(hidden_size)
        
        # Layer normalization (Ba et al., 2016)
        self.layer_norm = nn.LayerNorm(hidden_size)

        
        # Squeeze-and-Excitation (Hu et al., 2018)
        if self.use_se:
            # Adaptive reduction based on hidden_size
            reduction = max(4, hidden_size // 16)  # FIXED: min reduction of 4
            self.se_block = SEBlock(hidden_size, reduction=reduction)
        
        self.relu = nn.ReLU()
        
        # Output branches
        if self.use_branches:
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),  # Shared bottleneck
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.3),
                    nn.Linear(hidden_size // 2, 1)  # Final output: 1 value
                ) for _ in range(output_size)
            ])
        else:
            self.fc_out = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size) 
            )
    
    def forward(self, x):
        # GRU encoding
        gru_out, _ = self.gru(x)

        # Multi-head attention
        if self.use_attention:
            attn_out, _ = self.multihead_attention(gru_out, gru_out, gru_out)
            attn_out = self.attn_norm(attn_out + gru_out)  # Residual
            context = attn_out[:, -1, :]
        else:
            context = gru_out[:, -1, :]
        
        # Layer norm
        out = self.layer_norm(context)
        
        # Squeeze-and-Excitation
        if self.use_se:
            out = self.se_block(out)
        
        # Output
        if self.use_branches:
            gru_pred = torch.cat([branch(out) for branch in self.branches], dim=1)
        else:
            gru_pred = self.fc_out(out)

        if self.use_linear_residual:
            gru_pred = gru_pred + self.linear_residual(x[:, -1, :])

        return gru_pred

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)  # FIXED: ensure at least 1
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
    
    def forward(self, x):
        squeeze = x
        excitation = torch.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        return x * excitation