import torch
import torch.nn as nn


class OldGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3, 
                 use_branches=True, use_attention=True):
        """
        Flexible GRU model for revenue/expenditure forecasting.
        
        Args:
            input_size: Number of input features
            hidden_size: GRU hidden dimension
            output_size: Number of forecast targets (can be any number: 1, 3, 5, etc.)
            num_layers: Number of stacked GRU layers
            dropout: Dropout rate
            use_branches: If True, creates separate branch for each output.
                         If False, uses single shared decoder.
            use_attention: If True, uses temporal attention mechanism.
                          If False, uses last time step output.
        """
        super(OldGRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_branches = use_branches
        self.use_attention = use_attention
        
        # Shared GRU backbone
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Temporal attention mechanism (optional)
        if use_attention:
            self.attention_fc = nn.Linear(hidden_size, hidden_size)
            self.attention_score = nn.Linear(hidden_size, 1)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Shared feature extraction
        self.fc_shared = nn.Linear(hidden_size, hidden_size)
        self.dropout_shared = nn.Dropout(dropout)
        self.relu = nn.GELU()
        
        if use_branches:
            # Create separate branches dynamically for each output
            self.branches = nn.ModuleList()
            for i in range(output_size):
                branch = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden_size // 2, 1)
                )
                self.branches.append(branch)
        else:
            # Single shared decoder
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.dropout2 = nn.Dropout(dropout * 0.5)
            self.fc_out = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # GRU encoding - get all time steps
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size)
        
        if self.use_attention:
            # Temporal attention - learn which past time steps are important
            attn_input = torch.tanh(self.attention_fc(gru_out))
            attn_scores = self.attention_score(attn_input)
            attn_weights = torch.softmax(attn_scores, dim=1)
            
            # Weighted context vector
            context = torch.sum(attn_weights * gru_out, dim=1)
        else:
            # Just use last time step
            context = gru_out[:, -1, :]
        
        # Layer normalization
        out = self.layer_norm(context)
        
        # Shared feature extraction
        shared_features = self.fc_shared(out)
        shared_features = self.relu(shared_features)
        shared_features = self.dropout_shared(shared_features)
        
        if self.use_branches:
            # Pass through each branch and concatenate
            outputs = []
            for branch in self.branches:
                output = branch(shared_features)
                outputs.append(output)
            output = torch.cat(outputs, dim=1)
        else:
            # Shared decoder
            out = self.fc1(shared_features)
            out = self.relu(out)
            out = self.dropout1(out)
            
            out = self.fc2(out)
            out = self.relu(out)
            out = self.dropout2(out)
            
            output = self.fc_out(out)
        
        return output

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3,
                 use_branches=True, use_attention=True, use_residual=False, 
                 num_attention_heads=4, use_se=True):
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
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.use_branches = use_branches
        self.use_se = use_se
        
        # GRU backbone
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Multi-head attention (Vaswani et al., 2017)
        if use_attention:
            self.multihead_attention = nn.MultiheadAttention(
                hidden_size, num_attention_heads, dropout=dropout*0.5, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(hidden_size)
        
        # Layer normalization (Ba et al., 2016)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Shared layers with residual (He et al., 2016)
        self.fc_shared1 = nn.Linear(hidden_size, hidden_size)
        self.fc_shared2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_shared = nn.Dropout(dropout)
        
        # Squeeze-and-Excitation (Hu et al., 2018)
        if use_se:
            # Adaptive reduction based on hidden_size
            reduction = max(4, hidden_size // 16)  # FIXED: min reduction of 4
            self.se_block = SEBlock(hidden_size, reduction=reduction)
        
        self.relu = nn.LeakyReLU()
        
        # Output branches
        if use_branches:
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout * 0.3),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout * 0.3),
                    nn.Linear(hidden_size // 2, 1)
                ) for _ in range(output_size)
            ])
        else:
            self.fc_out = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LeakyReLU(),
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
        
        # Residual block 1
        identity1 = out
        out = self.fc_shared1(out)
        out = self.relu(out)
        out = self.dropout_shared(out)
        if self.use_residual:
            out = out + identity1
        
        # Residual block 2
        identity2 = out
        out = self.fc_shared2(out)
        out = self.relu(out)
        out = self.dropout_shared(out)
        if self.use_residual:
            out = out + identity2
        
        # Squeeze-and-Excitation
        if self.use_se:
            out = self.se_block(out)
        
        # Output
        if self.use_branches:
            outputs = [branch(out) for branch in self.branches]
            return torch.cat(outputs, dim=1)
        else:
            return self.fc_out(out)


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