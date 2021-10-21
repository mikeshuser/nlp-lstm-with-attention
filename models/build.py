#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import tensorflow as tf

class TextRNN(tf.keras.Model):

    """
    Main model, implementing LSTM with attention layer
    """
    
    def __init__(self, config: dict, vocab_size: int, **kwargs):

        super().__init__(**kwargs)
        self.num_classes = len(config['data']['labels'])
        self.max_len = config['data']['max_len']
        self.embedding_dim = config["dimensions"]['embedding_dim']
        self.recurrent_units = config["dimensions"]['recurrent_units']
        self.linear_units = config["dimensions"]['linear_units']
        self.sequence_shape = (self.max_len, self.embedding_dim)

        self.dropout_rate = config['regularizer']['dropout_rate']
        weight_reg = config['regularizer']['weights'] 
        if weight_reg["type"] == "L2":
            self.regularizer = tf.keras.regularizers.L2(weight_reg["rate"])
        elif weight_reg["type"] == "L1":
            self.regularizer = tf.keras.regularizers.L1(weight_reg["rate"])

        # model architecture
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, #vocab from DataManager includes <pad>
            output_dim=self.embedding_dim,
            input_length=self.max_len,
            mask_zero=True,
            name="embedding",
        )

        self.input_dropout = tf.keras.layers.SpatialDropout1D(
            rate=self.dropout_rate,
            name=f"input_dropout",
        )

        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=self.recurrent_units,
            return_sequences=True,
            name="rnn",)
        )

        self.attention = RaffelAttention(
            return_attention=True,
            name='attention',
        )

        self.dense_stack = []
        for i, units in enumerate(self.linear_units):
            block = dict(
                fc=tf.keras.layers.Dense(
                    units=units,
                    kernel_regularizer=self.regularizer,
                    use_bias=False,
                    name=f"dense{i + 1}",
                ),

                bn=tf.keras.layers.BatchNormalization(
                    name=f"batchnorm{i + 1}",
                ),

                act=tf.keras.layers.ReLU(
                    name=f"relu{i + 1}",
                ),
                
                drop=tf.keras.layers.Dropout(
                    rate=self.dropout_rate,
                    name=f"drop{i + 1}",
                )
            )
            self.dense_stack.append(block)

        self.head = tf.keras.layers.Dense(
            units=self.num_classes,
            activation="softmax",
            name="output",
        )

    def call(self, inputs, training=False):

        x = self.embedding(inputs)
        mask = x._keras_mask

        x = self.input_dropout(x, training=training)
        x = self.lstm(x, mask=mask)
        x, att_weights = self.attention(x, mask=mask)

        for block in self.dense_stack:
            x = block['fc'](x)
            x = block['bn'](x, training=training)
            x = block['act'](x)
            x = block['drop'](x, training=training)

        return self.head(x), att_weights

    def build_graph(self):

        """
        Used for producing model summary
        """
        x = tf.keras.layers.Input(shape=(self.max_len), name="sequence_input")
        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()

class RaffelAttention(tf.keras.layers.Layer):

    """
    Custom attention layer. Based on the following work:

    @article{DBLP:journals/corr/RaffelE15,
      author    = {Colin Raffel and
                   Daniel P. W. Ellis},
      title     = {Feed-Forward Networks with Attention Can 
                   Solve Some Long-Term Memory Problems},
      journal   = {CoRR},
      volume    = {abs/1512.08756},
      year      = {2015},
      url       = {http://arxiv.org/abs/1512.08756},
      eprinttype = {arXiv},
      eprint    = {1512.08756},
      timestamp = {Mon, 13 Aug 2018 16:48:37 +0200},
      biburl    = {https://dblp.org/rec/journals/corr/RaffelE15.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    """

    def __init__(self,
        regularizer=None,
        return_attention=True,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = tf.keras.initializers.GlorotUniform()
        self.regularizer = regularizer

    def build(self, input_shape):

        assert len(input_shape) == 3

        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer=self.init,
            name=f"{self.name}_W",
            regularizer=self.regularizer,
        )

        self.b = self.add_weight(
            shape=(input_shape[1],),
            initializer='zero',
            name=f"{self.name}_b"
        )

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):

        eij = tf.squeeze(tf.matmul(x, tf.expand_dims(self.W, axis=-1)), axis=-1)
        eij += self.b
        eij = tf.math.tanh(eij)

        a = tf.exp(eij)
        # apply mask after the exp
        if mask is not None:
            a *= tf.cast(mask, 'float32')

        # add epsilon to prevent NaN in case of sum close to 0
        a /= tf.cast(
            tf.reduce_sum(
                a, axis=1, keepdims=True
            ) + tf.keras.backend.epsilon(), 'float32'
        )

        weighted_input = x * tf.expand_dims(a, axis=-1)

        result = tf.reduce_sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        else:
            return result

    def compute_output_shape(self, input_shape):

        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]