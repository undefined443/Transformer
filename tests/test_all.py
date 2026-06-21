import keras
import numpy as np
import tensorflow as tf

import decoder
import encoder
import transformer
import utils


def test_get_angles():
    position = 4
    d_model = 16
    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    result = utils.get_angles(pos_m, dims, d_model)

    assert isinstance(result, np.ndarray), "You must return a numpy ndarray"
    assert result.shape == (
        position,
        d_model,
    ), f"Wrong shape. We expected: ({position}, {d_model})"
    assert np.sum(result[0, :]) == 0
    assert np.isclose(np.sum(result[:, 0]), position * (position - 1) / 2)
    even_cols = result[:, 0::2]
    odd_cols = result[:, 1::2]
    assert np.all(even_cols == odd_cols), (
        "Submatrices of odd and even columns must be equal"
    )
    limit = (position - 1) / np.power(10000, 14.0 / 16.0)
    assert np.isclose(result[position - 1, d_model - 1], limit), (
        f"Last value must be {limit}"
    )

    print("\033[92mAll tests passed")


def test_positional_encoding():
    position = 8
    d_model = 16

    pos_encoding = utils.positional_encoding(position, d_model)
    sin_part = pos_encoding[:, :, 0::2]
    cos_part = pos_encoding[:, :, 1::2]

    assert tf.is_tensor(pos_encoding), "Output is not a tensor"
    assert pos_encoding.shape == (
        1,
        position,
        d_model,
    ), f"Wrong shape. We expected: (1, {position}, {d_model})"

    ones = sin_part**2 + cos_part**2
    assert np.allclose(ones, np.ones((1, position, d_model // 2))), (
        "Sum of square pairs must be 1 = sin(a)**2 + cos(a)**2"
    )

    angs = np.arctan(sin_part / cos_part)
    angs[angs < 0] += np.pi
    angs[sin_part.numpy() < 0] += np.pi
    angs = angs % (2 * np.pi)

    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    trueAngs = utils.get_angles(pos_m, dims, d_model)[:, 0::2] % (2 * np.pi)

    assert np.allclose(angs[0], trueAngs), (
        "Did you apply sin and cos to even and odd parts respectively?"
    )

    print("\033[92mAll tests passed")


def test_scaled_dot_product_attention():
    q = np.array([[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]).astype(np.float32)
    k = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 1]]).astype(
        np.float32
    )
    v = np.array([[0, 0], [1, 0], [1, 0], [1, 1]]).astype(np.float32)

    _attention, _weights = utils.scaled_dot_product_attention(q, k, v, None)
    assert tf.is_tensor(_weights), "Weights must be a tensor"
    assert tuple(tf.shape(_weights).numpy()) == (
        q.shape[0],
        k.shape[1],
    ), f"Wrong shape. We expected ({q.shape[0]}, {k.shape[1]})"
    assert np.allclose(
        _weights,
        [
            [0.2589478, 0.42693272, 0.15705977, 0.15705977],
            [0.2772748, 0.2772748, 0.2772748, 0.16817567],
            [0.33620113, 0.33620113, 0.12368149, 0.2039163],
        ],
    ), "Wrong unmasked weights"

    assert tf.is_tensor(_attention), "Output must be a tensor"
    assert tuple(tf.shape(_attention).numpy()) == (
        q.shape[0],
        v.shape[1],
    ), f"Wrong shape. We expected ({q.shape[0]}, {v.shape[1]})"
    assert np.allclose(
        _attention,
        [[0.74105227, 0.15705977], [0.7227253, 0.16817567], [0.6637989, 0.2039163]],
    ), "Wrong unmasked attention"

    mask = np.array([[[1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 1]]])
    _attention, _weights = utils.scaled_dot_product_attention(q, k, v, mask)

    assert np.allclose(
        _weights,
        [
            [0.30719590187072754, 0.5064803957939148, 0.0, 0.18632373213768005],
            [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862],
            [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862],
        ],
    ), "Wrong masked weights"
    assert np.allclose(
        _attention,
        [
            [0.6928040981292725, 0.18632373213768005],
            [0.6163482666015625, 0.2326965481042862],
            [0.6163482666015625, 0.2326965481042862],
        ],
    ), "Wrong masked attention"

    print("\033[92mAll tests passed")


def test_EncoderLayer():
    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)
    keras.utils.set_random_seed(10)
    encoder_layer1 = encoder.EncoderLayer(4, 2, 8)
    encoded = encoder_layer1(q, training=True, mask=np.array([[1, 0, 1]]))

    assert tf.is_tensor(encoded), "Wrong type. Output must be a tensor"
    assert tuple(tf.shape(encoded).numpy()) == (
        1,
        q.shape[1],
        q.shape[2],
    ), f"Wrong shape. We expected ((1, {q.shape[1]}, {q.shape[2]}))"

    assert np.allclose(
        encoded.numpy(),
        [
            [1.71576512, -0.74984801, -0.59995812, -0.36595905],
            [0.32841188, 0.92912263, 0.42910266, -1.68663716],
            [1.66207135, -0.32136738, -0.32720715, -1.01349688],
        ],
    ), "Wrong values when training=True"

    encoded = encoder_layer1(q, training=False, mask=np.array([[1, 1, 0]]))
    assert np.allclose(
        encoded.numpy(),
        [
            [1.64606047, -0.51703101, -0.12530622, -1.00372314],
            [1.12798142, 0.39329147, 0.08074424, -1.60201716],
            [1.70261490, -0.42969552, -0.40591767, -0.86700189],
        ],
    ), "Wrong values when training=False"
    print("\033[92mAll tests passed")


def test_Encoder():
    keras.utils.set_random_seed(10)

    embedding_dim = 4

    encoderq = encoder.Encoder(
        num_layers=2,
        embedding_dim=embedding_dim,
        num_heads=2,
        fully_connected_dim=8,
        input_vocab_size=32,
        maximum_position_encoding=5,
    )

    x = np.array([[2, 1, 3], [1, 2, 0]])

    encoderq_output = encoderq(x, training=True, mask=None)

    assert tf.is_tensor(encoderq_output), "Wrong type. Output must be a tensor"
    assert tuple(tf.shape(encoderq_output).numpy()) == (
        x.shape[0],
        x.shape[1],
        embedding_dim,
    ), f"Wrong shape. We expected ({x.shape[0]}, {x.shape[1]}, {embedding_dim})"
    assert np.allclose(
        encoderq_output.numpy(),
        [
            [
                [0.64132398, 1.18332124, -1.42407382, -0.40057123],
                [0.43446064, -0.56744283, -1.25267708, 1.38565934],
                [0.56714535, -1.27937925, -0.58937079, 1.30160475],
            ],
            [
                [-0.11126658, 1.52792621, -1.27823997, -0.13841979],
                [0.26675871, -0.71727818, -1.06176507, 1.51228452],
                [1.44744325, -1.26434302, -0.47206059, 0.28896037],
            ],
        ],
    ), "Wrong values case 1"

    encoderq_output = encoderq(
        x, training=True, mask=np.array([[[[1.0, 1.0, 1.0]]], [[[1.0, 1.0, 0.0]]]])
    )
    assert np.allclose(
        encoderq_output.numpy(),
        [
            [
                [0.45556697, 1.16695023, -1.55771518, -0.06480181],
                [0.99881208, -0.59526044, -1.33460665, 0.93105501],
                [0.39011490, -1.13546515, -0.69520867, 1.44055903],
            ],
            [
                [0.11205333, 1.04332399, -1.63397026, 0.47859314],
                [0.96517229, -0.81126696, -1.17204273, 1.01813734],
                [0.73446023, -1.15113628, -0.80374932, 1.22042549],
            ],
        ],
    ), "Wrong values case 2"

    encoderq_output = encoderq(
        x, training=False, mask=np.array([[[[1.0, 1.0, 1.0]]], [[[1.0, 1.0, 0.0]]]])
    )
    assert np.allclose(
        encoderq_output.numpy(),
        [
            [
                [0.10186142, 1.26234436, -1.53813243, 0.17392626],
                [0.94324565, -0.55363244, -1.36268473, 0.97307140],
                [0.50403488, -1.10701632, -0.77986681, 1.38284838],
            ],
            [
                [0.28565845, 0.97811633, -1.67186785, 0.40809309],
                [0.71860015, -0.81072897, -1.14180911, 1.23393798],
                [0.73158288, -1.15945244, -0.79326874, 1.22113848],
            ],
        ],
    ), "Wrong values case 3"

    print("\033[92mAll tests passed")


def test_DecoderLayer():
    num_heads = 8
    keras.utils.set_random_seed(10)

    decoderLayerq = decoder.DecoderLayer(
        embedding_dim=4,
        num_heads=num_heads,
        fully_connected_dim=32,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    )

    encoderq_output = tf.constant(
        [
            [
                [-0.40172306, 0.11519244, -1.2322885, 1.5188192],
                [0.4017268, 0.33922842, -1.6836855, 0.9427304],
                [0.4685002, -1.6252842, 0.09368491, 1.063099],
            ]
        ]
    )

    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)

    look_ahead_mask = utils.create_look_ahead_mask(q.shape[1])

    padding_mask = None
    out, attn_w_b1, attn_w_b2 = decoderLayerq(
        q,
        encoderq_output,
        training=True,
        look_ahead_mask=look_ahead_mask,
        padding_mask=padding_mask,
    )

    assert tf.is_tensor(attn_w_b1), "Wrong type for attn_w_b1. Output must be a tensor"
    assert tf.is_tensor(attn_w_b2), "Wrong type for attn_w_b2. Output must be a tensor"
    assert tf.is_tensor(out), "Wrong type for out. Output must be a tensor"

    shape1 = (q.shape[0], num_heads, q.shape[1], q.shape[1])
    assert tuple(tf.shape(attn_w_b1).numpy()) == shape1, (
        f"Wrong shape. We expected {shape1}"
    )
    assert tuple(tf.shape(attn_w_b2).numpy()) == shape1, (
        f"Wrong shape. We expected {shape1}"
    )
    assert tuple(tf.shape(out).numpy()) == q.shape, (
        f"Wrong shape. We expected {q.shape}"
    )

    assert np.allclose(attn_w_b1[0, 0, 1], [0.49882561, 0.50117433, 0.0], atol=1e-2), (
        "Wrong values in attn_w_b1. Check the call to self.mha1"
    )
    assert np.allclose(attn_w_b2[0, 0, 1], [0.39384076, 0.37554538, 0.23061390]), (
        "Wrong values in attn_w_b2. Check the call to self.mha2"
    )
    assert np.allclose(out[0, 0], [1.48213482, -1.06372225, -0.74968433, 0.33127174]), (
        "Wrong values in out"
    )

    # Now let's try a example with padding mask
    padding_mask = np.array([[[1, 1, 0]]])
    out, attn_w_b1, attn_w_b2 = decoderLayerq(
        q,
        encoderq_output,
        training=True,
        look_ahead_mask=look_ahead_mask,
        padding_mask=padding_mask,
    )
    assert np.allclose(out[0, 0], [1.46518159, -0.91289985, -0.93467861, 0.38239682]), (
        "Wrong values in out when we mask the last word. Are you passing the padding_mask to the inner functions?"
    )

    print("\033[92mAll tests passed")


def test_Decoder():
    keras.utils.set_random_seed(10)

    num_layers = 7
    embedding_dim = 4
    num_heads = 3
    fully_connected_dim = 8
    target_vocab_size = 33
    maximum_position_encoding = 6

    x_array = np.array([[3, 2, 1], [2, 1, 0]])

    encoderq_output = tf.constant(
        [
            [
                [-0.40172306, 0.11519244, -1.2322885, 1.5188192],
                [0.4017268, 0.33922842, -1.6836855, 0.9427304],
                [0.4685002, -1.6252842, 0.09368491, 1.063099],
            ],
            [
                [-0.3489219, 0.31335592, -1.3568854, 1.3924513],
                [-0.08761203, -0.1680029, -1.2742313, 1.5298463],
                [0.2627198, -1.6140151, 0.2212624, 1.130033],
            ],
        ]
    )

    look_ahead_mask = utils.create_look_ahead_mask(x_array.shape[1])

    decoderk = decoder.Decoder(
        num_layers,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        target_vocab_size,
        maximum_position_encoding,
    )
    x, attention_weights = decoderk(
        x_array,
        encoderq_output,
        training=False,
        look_ahead_mask=look_ahead_mask,
        padding_mask=None,
    )
    assert tf.is_tensor(x), "Wrong type for x. It must be a dict"
    assert np.allclose(tf.shape(x), tf.shape(encoderq_output)), (
        f"Wrong shape. We expected {tf.shape(encoderq_output)}"
    )
    assert np.allclose(x[1, 1], [1.34593654, 0.27714872, -1.44179332, -0.18129183]), (
        "Wrong values in x"
    )

    keys = list(attention_weights.keys())
    assert isinstance(attention_weights, dict), (
        "Wrong type for attention_weights[0]. Output must be a tensor"
    )
    assert len(keys) == 2 * num_layers, (
        f"Wrong length for attention weights. It must be 2 x num_layers = {2 * num_layers}"
    )
    assert tf.is_tensor(attention_weights[keys[0]]), (
        f"Wrong type for attention_weights[{keys[0]}]. Output must be a tensor"
    )
    shape1 = (x_array.shape[0], num_heads, x_array.shape[1], x_array.shape[1])
    assert tuple(tf.shape(attention_weights[keys[1]]).numpy()) == shape1, (
        f"Wrong shape. We expected {shape1}"
    )
    assert np.allclose(
        attention_weights[keys[0]][0, 0, 1], [0.5049869, 0.4950131, 0.0]
    ), f"Wrong values in attention_weights[{keys[0]}]"

    x, attention_weights = decoderk(
        x_array,
        encoderq_output,
        training=True,
        look_ahead_mask=look_ahead_mask,
        padding_mask=None,
    )
    assert np.allclose(x[1, 1], [0.69916469, -1.08003044, -0.87543976, 1.25630558]), (
        "Wrong values in x when training=True"
    )

    x, attention_weights = decoderk(
        x_array,
        encoderq_output,
        training=True,
        look_ahead_mask=look_ahead_mask,
        padding_mask=utils.create_padding_mask(x_array),
    )
    assert np.allclose(x[1, 1], [-0.21460545, -0.95670772, -0.49893081, 1.67024422]), (
        "Wrong values in x when training=True and use padding mask"
    )

    print("\033[92mAll tests passed")


def test_Transformer():
    keras.utils.set_random_seed(10)

    num_layers = 6
    embedding_dim = 4
    num_heads = 4
    fully_connected_dim = 8
    input_vocab_size = 30
    target_vocab_size = 35
    max_positional_encoding_input = 5
    max_positional_encoding_target = 6

    trans = transformer.Transformer(
        num_layers,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        input_vocab_size,
        target_vocab_size,
        max_positional_encoding_input,
        max_positional_encoding_target,
    )
    # 0 is the padding value
    sentence_lang_a = np.array([[2, 1, 4, 3, 0]])
    sentence_lang_b = np.array([[3, 2, 1, 0, 0]])

    enc_padding_mask = utils.create_padding_mask(sentence_lang_a)
    dec_padding_mask = utils.create_padding_mask(sentence_lang_b)

    look_ahead_mask = utils.create_look_ahead_mask(sentence_lang_a.shape[1])

    translation, weights = trans(
        sentence_lang_a,
        sentence_lang_b,
        training=True,  # Training
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=look_ahead_mask,
        dec_padding_mask=dec_padding_mask,
    )

    assert tf.is_tensor(translation), (
        "Wrong type for translation. Output must be a tensor"
    )
    shape1 = (
        sentence_lang_a.shape[0],
        max_positional_encoding_input,
        target_vocab_size,
    )
    assert tuple(tf.shape(translation).numpy()) == shape1, (
        f"Wrong shape. We expected {shape1}"
    )

    assert np.allclose(
        translation[0, 0, 0:8],
        [
            0.02846772,
            0.03779168,
            0.02509714,
            0.02112385,
            0.03053033,
            0.03151495,
            0.01490482,
            0.03214807,
        ],
    ), "Wrong values in translation"

    keys = list(weights.keys())
    assert isinstance(weights, dict), "Wrong type for weights. It must be a dict"
    assert len(keys) == 2 * num_layers, (
        f"Wrong length for attention weights. It must be 2 x num_layers = {2 * num_layers}"
    )
    assert tf.is_tensor(weights[keys[0]]), (
        f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"
    )

    shape1 = (
        sentence_lang_a.shape[0],
        num_heads,
        sentence_lang_a.shape[1],
        sentence_lang_a.shape[1],
    )
    assert tuple(tf.shape(weights[keys[1]]).numpy()) == shape1, (
        f"Wrong shape. We expected {shape1}"
    )
    assert np.allclose(
        weights[keys[0]][0, 0, 1], [0.4912196, 0.5087804, 0.0, 0.0, 0.0]
    ), f"Wrong values in weights[{keys[0]}]"

    translation, weights = trans(
        sentence_lang_a,
        sentence_lang_b,
        training=False,
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=look_ahead_mask,
        dec_padding_mask=dec_padding_mask,
    )

    assert np.allclose(
        translation[0, 0, 0:8],
        [
            0.02849045,
            0.03841006,
            0.02527412,
            0.02153934,
            0.03023914,
            0.03138555,
            0.01497277,
            0.03288213,
        ],
    ), "Wrong values in outd"

    print("\033[92mAll tests passed")


if __name__ == "__main__":
    # test_scaled_dot_product_attention()
    test_Transformer()
