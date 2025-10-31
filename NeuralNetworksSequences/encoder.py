import torch


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = torch.nn.Embedding(input_dim, emb_dim)
        self.lstm = torch.nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers)

    def forward(self, src):
        # src: [seq_len, batch_size]
        embedded = self.embedding(src)
        # embedded: [seq_len, batch_size, emb_dim]
        
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [seq_len, batch_size, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell: [num_layers, batch_size, hidden_dim]

        return outputs, (hidden, cell)
