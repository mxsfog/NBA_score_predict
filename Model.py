import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Линейные слои для преобразования входных данных в ключи, запросы и значения
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # Разделяем последнее измерение на num_heads и d_k
        return x.view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        # Применяем линейные преобразования и разделяем на головы
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # Вычисляем scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Объединяем головы и применяем финальное линейное преобразование
        attention_output = attention_output.transpose(1, 2).contiguous().view(attention_output.size(0), -1,
                                                                              self.d_model)
        return self.W_o(attention_output)


class ScorePredictionTransformer(nn.Module):
    def __init__(self, input_size=8, d_model=64, num_heads=4, num_layers=2, dropout=0.1):
        super(ScorePredictionTransformer, self).__init__()
        self.embedding = nn.Linear(input_size * 2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=128, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(d_model, 2)

    def forward(self, team1, team2):
        # Объединяем статистику обеих команд
        x = torch.cat((team1, team2), dim=1)  # (10, 16)
        x = x.unsqueeze(1)  # (10, 1, 16)
        x = self.embedding(x)  # (10, 1, d_model)
        x = self.pos_encoder(x)

        # Применяем Multi-Head Attention
        attn_output = self.multi_head_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Проходим через слои трансформера
        for layer in self.layers:
            x = layer(x)

        # Усредняем выход по всей последовательности
        output = x.mean(dim=0)

        # Предсказываем счет
        scores = self.fc(output)
        return scores.squeeze(0)