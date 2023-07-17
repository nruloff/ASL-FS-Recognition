# Source: Keras Transformer Model for Automatic Speech Recognition - https://keras.io/examples/audio/transformer_asr/
class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=len(all_keys), maxlen=MAX_FRAME_LENGTH, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        print(maxlen)
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions
