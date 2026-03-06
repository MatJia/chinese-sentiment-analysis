from dataset import load_set
import torch
import torch.nn as nn

vocab, word_frq, train_size, train_padded_list, train_label_list, predict_padded_list, predict_label_list = load_set()

class Sentiment(nn.Module):
    def __init__(self, vocab_len, embed_len, class_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_len, embed_len)
        self.fc = nn.Linear(embed_len, class_len)#fully connect
    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
