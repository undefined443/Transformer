import numpy as np
import tensorflow as tf


def get_angles(pos, k, d):
    i = k // 2
    angles = pos / np.power(10000, 2 * i / d)
    return angles


def positional_encoding(positions, d):
    angle_rads = get_angles(np.arange(positions)[..., np.newaxis], np.arange(d)[np.newaxis, ...], d)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(decoder_token_ids):
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
    return seq[:, tf.newaxis, :]


def create_look_ahead_mask(sequence_length):
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k.T)
    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / tf.sqrt(tf.cast(dk, tf.float32))
    if mask is not None:
        scaled_attention_logits += (1 - mask) * -1.0e9

    attention_weights = tf.keras.activations.softmax(scaled_attention_logits)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


def FullyConnected(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(fully_connected_dim, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
        ]
    )
