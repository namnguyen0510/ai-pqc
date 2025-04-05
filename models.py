import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
# Custom PyTorch Dataset
class ByteDataset(Dataset):
    def __init__(self, df):
        self.inputs = torch.tensor(np.stack(df['estr_encoded'].values), dtype=torch.float32)
        self.targets = torch.tensor(np.stack(df['gstr_encoded'].values), dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
# Multi-layer RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        out, _ = self.rnn(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Multi-layer GRN (Gene Regulatory Network)
class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)  # GRN outputs often constrained between 0 and 1
        return out

# Multi-layer FNN (Feedforward Neural Network)
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Multi-layer LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        out, (hn, cn) = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out
class CustomHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(CustomHuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        diff = torch.abs(input - target)
        condition = diff < self.delta
        loss = torch.where(condition, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta))
        return loss.mean()


class TransformerModel(nn.Module):
    def __init__(self, input_size, embed_size, num_heads, hidden_size, num_layers, output_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Embedding layer for input features
        self.embedding = nn.Linear(input_size, embed_size)  # Project input to embedding size
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_size, 
                                                    nhead=num_heads, 
                                                    dim_feedforward=hidden_size, 
                                                    dropout=dropout,
                                                    batch_first=True)  
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Fully connected output layer
        self.fc = nn.Linear(embed_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)  # Project input to embedding size
        x = self.transformer_encoder(x)  # Pass through Transformer layers
        
        # Ensure x has 3D shape (batch_size, seq_len, embed_size)
        if x.dim() == 2:  
            x = x.unsqueeze(1)  # Add sequence length dimension

        x = self.fc(x[:, -1, :])  # Take last time step output safely
        x = self.relu(x)
        return x



def reset_weights(m):
    """Reset weights for the model layers."""
    if isinstance(m, (nn.Linear, nn.LSTM, nn.RNN)):
        for param in m.parameters():
            if param.dim() > 1:  # Initialize weights for layers (not biases)
                init.xavier_uniform_(param)  # Use Xavier initialization for weights
            else:  # Initialize biases
                init.zeros_(param)