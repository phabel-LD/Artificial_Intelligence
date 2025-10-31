import torch


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, attention, output_vocabulary_size, device):
        super().__init__()
        self._encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self._output_vocabulary_size = output_vocabulary_size
        self._device = device

    def forward(self, source, target):
        # src: [seq_len, batch_size]
        # target: [seq_len, batch_size]
        encoder_outputs, (hidden, cell) = self._encoder(source)
        # encoder_outputs: [seq_len, batch_size, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell: [num_layers, batch_size, hidden_dim]

        # First input to the decoder is always the <sos> token. Contains the first word
        # of every sentence in the batch.
        input = target[0, :]
        # input: [batch_size,]

        # Placeholder for the decoder outputs
        outputs = torch.zeros(target.size(0), target.size(1), self._output_vocabulary_size).to(self._device)
        # outputs: [seq_len, batch_size, output_vocabulary_size]

        # Iterate over the length of the target to produce each word/token
        for t in range(1, target.size(1)):
            attention_weights = self.attention(hidden, encoder_outputs)
            # attention_weights: [batch_size, seq_len] -> alphas from the paper

            attention_weights = attention_weights.unsqueeze(2)
            # attention_weights: [batch_size, seq_len, 1]

            context = torch.bmm(encoder_outputs.permute(1, 2, 0), attention_weights)
            # context: [batch_size, hidden_dim, 1]

            context = context.squeeze(2)
            # context: [batch_size, hidden_dim]
            
            # With the context vector, generate the next word/token in the target sequence
            output, (hidden, cell) = self.decoder(input, (hidden, cell), context)
            # output: [batch_size, output_vocabulary_size]
            # hidden: [num_layers, batch_size, hidden_dim]
            # cell: [num_layers, batch_size, hidden_dim]

            # Save the output
            outputs[t] = output

            # Use the most probable word/token as the next input to the decoder
            input = output.argmax(1)
            #input: [batch_size,]

        return outputs
