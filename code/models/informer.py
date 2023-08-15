import torch
import torch.nn as nn


class INFORMER(nn.Module):
    # def __init__(self, input_size, output_size, hidden_size, num_heads, num_encoder_layers, num_decoder_layers):
    def __init__(self, embedding_matrix, opt):
        super(INFORMER, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(opt.hidden_dim, opt.num_heads),
            opt.num_encoder_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(opt.hidden_dim, opt.num_heads),
            opt.num_decoder_layers
        )

        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        # x: batch_size * sequence_length * hidden_size
        # Assuming x has been properly formatted

        # Transpose x for compatibility with Transformer
        x = x.permute(1, 0, 2)

        # Encoder
        encoder_output = self.encoder(x)

        # Decoder (pass encoder_output as memory)
        decoder_output = self.decoder(x, encoder_output)

        # Transpose back to original shape
        decoder_output = decoder_output.permute(1, 0, 2)

        # Reduce sequence dimension to get [batch_size * output_size]
        output = torch.mean(decoder_output, dim=1)  # You can also use torch.sum() if needed

        return output