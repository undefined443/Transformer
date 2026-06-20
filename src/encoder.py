import tensorflow as tf
from tensorflow.keras.layers import (
    Dropout,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
)

from utils import FullyConnected, positional_encoding


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate
        )
        self.ffn = FullyConnected(
            embedding_dim=embedding_dim, fully_connected_dim=fully_connected_dim
        )
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, x, training, mask):
        self_mha_output = self.mha(x, x, x, attention_mask=mask, training=training)
        skip_x_attention = self.layernorm1(self_mha_output + x)
        ffn_output = self.ffn(skip_x_attention)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        encoder_layer_out = self.layernorm2(ffn_output + skip_x_attention)
        return encoder_layer_out


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        input_vocab_size,
        maximum_position_encoding,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    ):
        super().__init__()
        self.supports_masking = True

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, self.embedding_dim
        )
        self.enc_layers = [
            EncoderLayer(
                embedding_dim=self.embedding_dim,
                num_heads=num_heads,
                fully_connected_dim=fully_connected_dim,
                dropout_rate=dropout_rate,
                layernorm_eps=layernorm_eps,
            )
            for _ in range(self.num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        return x
