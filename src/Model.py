import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    PositionalEncoding is a class for adding positional information to the input data.

    Attributes:
    pe (torch.Tensor): Positional encoding tensor.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Initializes the PositionalEncoding class with the given model dimension and maximum length.

        Parameters:
        d_model (int): The dimension of the model.
        max_len (int): The maximum length of the sequence.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input data.

        Parameters:
        x (torch.Tensor): The input data.

        Returns:
        torch.Tensor: The input data with positional encoding added.
        """
        return x + self.pe[:x.size(0), :]


class EncoderBlock(nn.Module):
    """
    EncoderBlock is a class for the encoder block of the transformer model.

    Attributes:
    ff1 (torch.nn.Linear): The first feed-forward layer.
    ff2 (torch.nn.Linear): The second feed-forward layer.
    norm1 (torch.nn.LayerNorm): The first layer normalization.
    norm2 (torch.nn.LayerNorm): The second layer normalization.
    dropout1 (torch.nn.Dropout): The first dropout layer.
    dropout2 (torch.nn.Dropout): The second dropout layer.
    relu (torch.nn.ReLU): The ReLU activation function.
    self_attn (torch.nn.MultiheadAttention): The multi-head self-attention mechanism.
    """
    def __init__(self, d_model, d_ff, dropout):
        """
        Initializes the EncoderBlock class with the given model dimension, feed-forward dimension, and dropout rate.

        Parameters:
        d_model (int): The dimension of the model.
        d_ff (int): The dimension of the feed-forward network.
        dropout (float): The dropout rate.
        """
        super(EncoderBlock, self).__init__()
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout)

    def forward(self, x):
        """
        Forward pass of the encoder block.

        Parameters:
        x (torch.Tensor): The input data.

        Returns:
        torch.Tensor: The output of the encoder block.
        """
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.ff2(self.relu(self.ff1(x)))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x


class NBAModel(nn.Module):
    """
    NBAModel is a class for the NBA prediction model.

    Attributes:
    embedding (torch.nn.Linear): The embedding layer.
    pos_encoder (PositionalEncoding): The positional encoding layer.
    encoder_layers (torch.nn.ModuleList): The list of encoder blocks.
    fc1 (torch.nn.Linear): The first fully connected layer.
    fc2 (torch.nn.Linear): The second fully connected layer.
    fc3 (torch.nn.Linear): The third fully connected layer.
    dropout (torch.nn.Dropout): The dropout layer.
    relu (torch.nn.ReLU): The ReLU activation function.
    """
    def __init__(self, input_dim, num_encoder_layers, dim_feedforward, dropout=0.1):
        """
        Initializes the NBAModel class with the given input dimension, number of encoder layers, feed-forward dimension, and dropout rate.

        Parameters:
        input_dim (int): The dimension of the input data.
        num_encoder_layers (int): The number of encoder layers.
        dim_feedforward (int): The dimension of the feed-forward network.
        dropout (float): The dropout rate.
        """
        super(NBAModel, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        self.pos_encoder = PositionalEncoding(dim_feedforward)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model=dim_feedforward, d_ff=dim_feedforward * 4, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.fc1 = nn.Linear(dim_feedforward, dim_feedforward // 2)
        self.fc2 = nn.Linear(dim_feedforward // 2, dim_feedforward // 4)
        self.fc3 = nn.Linear(dim_feedforward // 4, 2)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, src):
        """
        Forward pass of the NBA model.

        Parameters:
        src (torch.Tensor): The input data.

        Returns:
        torch.Tensor: The output of the model.
        """
        src = self.embedding(src)
        src = self.pos_encoder(src)

        for layer in self.encoder_layers:
            src = layer(src)

        output = src.mean(dim=0)  # Average over the temporal dimension

        output = self.dropout(self.relu(self.fc1(output)))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)

        return output