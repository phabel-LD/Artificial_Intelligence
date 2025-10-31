import torch


class Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.attn = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = torch.nn.Parameter(torch.rand(hidden_dim))
        # self.v: [hidden_dim,]

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [num_layers, batch_size, hidden_dim]
        # encoder_outputs: [seq_len, batch_size, hidden_dim]

        batch_size = encoder_outputs.shape[1]
        seq_len = encoder_outputs.shape[0]

        # Repeat decoder hidden state seq_len times
        # decoder_hidden[-1] selects the last layer [batch_size, hidden_dim]
        # unsqueeze adds a dimension [batch_size, 1, hidden_dim]
        decoder_hidden = decoder_hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        # decoder_hidden: [batch_size, seq_len, hidden_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs: [batch_size, seq_len, hidden_dim]

        # Compute energy
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim = 2)))
        # energy: [batch_size, seq_len, hidden_dim]
        energy = energy.permute(0, 2, 1)
        # energy: [batch_size, hidden_dim, seq_len]

        # Use v to compute the attention scores
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # v: [batch_size, 1, hidden_dim]
        attention = torch.bmm(v, energy).squeeze(1)
        # attention: [batch_size, seq_len]

        return torch.F.softmax(attention, dim=1) # [batch_size, seq_len]
