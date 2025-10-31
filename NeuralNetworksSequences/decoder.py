import torch


class Decoder(torch.nn.Module):
    def __init__(self, vocabulary_size, emb_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, emb_dim)
        self.lstm = torch.nn.LSTM(
            input_size=emb_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers
        )  # +hidden_dim for attention context
        self.out = torch.nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, input, hidden, cell, context):
        # input: [batch_size,] la palabra anterior.
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell: [num_layers, batch_size, hidden_dim]
        # context: [batch_size, hidden_dim]
        input = input.unsqueeze(0)
        # input: [1, batch_size]
        embedded = self.embedding(input)
        # embedded: [1, batch_size, emb_dim]
        context = context.unsqueeze(0)
        # context: [1, batch_size, hidden_dim]
        lstm_input = torch.cat((embedded, context), dim=2)
        # lstm_input: [1, batch_size, emb_dim + hidden_dim]
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: [1, batch_size, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell: [num_layers, batch_size, hidden_dim]
        prediction = self.out(output.squeeze(0))
        # prediction: [batch_size, vocabulary_size]
        return prediction, hidden, cell
