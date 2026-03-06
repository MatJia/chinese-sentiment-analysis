from dataset import load_set
import torch
import torch.nn as nn

vocab, word_frq, train_size, train_padded_list, train_label_list, predict_padded_list, predict_label_list = load_set()

class sentiment(nn.Module)
    def __init__(self):
        super().__init__()
    def forward(self):