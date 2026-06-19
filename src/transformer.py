import tensorflow as tf
from tensorflow.keras.layers import Dense

from decoder import Decoder
from encoder import Encoder


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        input_vocab_size,
        target_vocab_size,
        max_positional_encoding_input,
        max_positional_encoding_target,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    ):
        super().__init__()

        self.encoder = Encoder(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            fully_connected_dim=fully_connected_dim,
            input_vocab_size=input_vocab_size,
            maximum_position_encoding=max_positional_encoding_input,
            dropout_rate=dropout_rate,
            layernorm_eps=layernorm_eps,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            fully_connected_dim=fully_connected_dim,
            target_vocab_size=target_vocab_size,
            maximum_position_encoding=max_positional_encoding_target,
            dropout_rate=dropout_rate,
            layernorm_eps=layernorm_eps,
        )

        self.final_layer = Dense(target_vocab_size, activation="softmax")

    def call(
        self,
        input_sentence,
        output_sentence,
        training,
        enc_padding_mask,
        look_ahead_mask,
        dec_padding_mask,
    ):
        enc_output = self.encoder(
            input_sentence, training=training, mask=enc_padding_mask
        )
        dec_output, attention_weights = self.decoder(
            output_sentence,
            enc_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=dec_padding_mask,
        )
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights
