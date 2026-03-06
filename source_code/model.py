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

if __name__ == "__main__":
    model = Sentiment(len(vocab), 16, 4)
    x = torch.tensor(train_padded_list)
    y = torch.tensor(train_label_list)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(224):
        opt = model(x)
        loss = loss_fn(opt, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")
    x_test = torch.tensor(predict_padded_list)
    y_test = torch.tensor(predict_label_list)
    opt_test = model(x_test)
    pred = torch.argmax(opt_test, dim = 1)
    print(pred)
    print(y_test)