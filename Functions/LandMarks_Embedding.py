# Partial Source: Keras Transofrmer model for Automatic Speech Recognition - https://keras.io/examples/audio/transformer_asr/
# Some slight alterations made for applying to video recognition
class LandMarks_Embedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=MAX_FRAME_LENGTH):
        super().__init__()
        # First Convolution
        self.conv1 = tf.keras.layers.Conv1D(num_hid, 9, padding="same", activation="relu")
        # Second Convolution
        self.conv2 = tf.keras.layers.Conv1D(num_hid, 9, padding="same", activation="relu")
        # Third Convolution
        self.conv3 = tf.keras.layers.Conv1D(num_hid, 9, padding="same", activation="relu")


    def call(self, x):
        maxlen = MAX_FRAME_LENGTH
        # Ensure that input landmarks are cast to float32 type
        x = tf.cast(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
