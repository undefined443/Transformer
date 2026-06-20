import tensorflow as tf
from tensorflow.keras.layers import (
    Dropout,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
)

from utils import FullyConnected, positional_encoding


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    ):
        super().__init__()
        self.mha1 = MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate
        )
        self.mha2 = MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate
        )
        self.ffn = FullyConnected(
            embedding_dim=embedding_dim, fully_connected_dim=fully_connected_dim
        )
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        mult_attn_out1, attn_weights_block1 = self.mha1(
            x,
            x,
            attention_mask=look_ahead_mask,
            training=training,
            return_attention_scores=True,
        )

        Q1 = self.layernorm1(mult_attn_out1 + x)

        mult_attn_out2, attn_weights_block2 = self.mha2(
            Q1,
            enc_output,
            attention_mask=padding_mask,
            training=training,
            return_attention_scores=True,
        )
        mult_attn_out2 = self.layernorm2(mult_attn_out2 + Q1)
        ffn_output = self.ffn(mult_attn_out2)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + mult_attn_out2)
        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        target_vocab_size,
        maximum_position_encoding,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(target_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, self.embedding_dim
        )
        self.dec_layers = [
            DecoderLayer(
                embedding_dim=self.embedding_dim,
                num_heads=num_heads,
                fully_connected_dim=fully_connected_dim,
                dropout_rate=dropout_rate,
                layernorm_eps=layernorm_eps,
            )
            for _ in range(self.num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.sqrt(tf.cast(self.embedding_dim, tf.float32))

        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x,
                enc_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask,
            )
            attention_weights["decoder_layer{}_block1_self_att".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2_decenc_att".format(i + 1)] = (
                block2
            )
        return x, attention_weights
