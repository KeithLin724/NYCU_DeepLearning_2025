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
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
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
        pretrained_embedding=None,
    ):
        super().__init__()
        # 若有預訓練向量則使用之；否則隨機初始化
        if pretrained_embedding is not None:
            # 預訓練的 embedding 應該為 shape: (vocab_size, embed_dim)
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding, freeze=False
            )
        else:
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
        self.fc = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def forward(self, src):
        # src shape: (batch_size, seq_len) -> token id 序列
        x = self.embedding(src) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch_size, embed_dim)
        x = self.transformer_encoder(x)
        # 假設 [CLS] token 為序列第一個 token
        cls_token = x[0]  # (batch_size, embed_dim)
        logits = self.fc(cls_token)
        return logits


if __name__ == "__main__":
    import gensim

    # 載入已經訓練好的 fastText 模型
    fasttext_model = gensim.models.FastText.load("model/skipgram.model")
    # 取得 fastText 的字彙表及向量： index_to_key 須與訓練的文本對應
    vocab = fasttext_model.wv.index_to_key
    vocab_size = len(vocab)
    embed_dim = fasttext_model.wv.vector_size

    # 建立預訓練的 embedding 矩陣 (vocab_size, embed_dim)
    import torch

    pretrained_embedding = torch.FloatTensor(
        [fasttext_model.wv[word] for word in vocab]
    )

    num_classes = 5  # 根據你的分類任務設定類別數量
    model = TransformerClassifier(
        vocab_size, embed_dim, num_classes, pretrained_embedding=pretrained_embedding
    )

    # 測試輸入：sample_input 的 token id 須依據 fastText 字彙索引轉換
    sample_input = torch.randint(0, vocab_size, (16, 50))  # 此處僅為範例
    output = model(sample_input)
    print(output.shape)  # 預期輸出 (16, 5)
