import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # 若 d_model 為奇數，cos 會略少一個維度
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_classes,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_length=512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_seq_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 可採用簡單的池化方式，例如取 [CLS] token 的輸出
        self.fc = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def forward(self, src):
        # src shape: (batch_size, seq_len) --> token ids
        x = self.embedding(src) * math.sqrt(
            self.embed_dim
        )  # (batch_size, seq_len, embed_dim)
        x = self.pos_encoder(x)
        # TransformerEncoder 預設輸入 shape: (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)  # (seq_len, batch_size, embed_dim)
        # 此處假設第一個 token 為 [CLS] token，可用於分類
        cls_token = x[0]  # shape: (batch_size, embed_dim)
        logits = self.fc(cls_token)
        return logits


if __name__ == "__main__":
    # 範例：假設詞彙大小 10000, 嵌入維度 128, 分類類別數 5, 輸入長度 50
    batch_size = 16
    seq_len = 50
    vocab_size = 10000
    embed_dim = 128
    num_classes = 5

    model = TransformerClassifier(vocab_size, embed_dim, num_classes)
    sample_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(sample_input)
    print(output.shape)  # 預期：(batch_size, num_classes)
