# Partial Source: Keras Transofrmer model for Automatic Speech Recognition - https://keras.io/examples/audio/transformer_asr/
# Some slight alterations made for applying to video recognition
class LandMarks_Embedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=MAX_FRAME_LENGTH, image_kernel_size=3, time_kernel_size=9, square_image_dim=28, num_body_parts=4):
        super().__init__()
        # First Convolution is across the 2D images of each frame
        self.conv1 = tf.keras.layers.Conv3D(num_hid, kernel_size=(1,image_kernel_size,image_kernel_size),
                                            activation='relu', input_shape=(maxlen, square_image_dim, square_image_dim, num_body_parts), padding='same', groups=num_body_parts)
        # Second Convolution is over time across all of the images
        self.conv2 = tf.keras.layers.Conv3D(num_hid, kernel_size=(time_kernel_size, square_image_dim, square_image_dim), activation='relu', groups=num_body_parts)


    def call(self, x):
        maxlen = x.shape[1]
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.squeeze(x)
        temp_len = x.shape[0]
        x = tf.pad(x, ((0, int(maxlen-temp_len)), (0,0)), mode='CONSTANT')
        return x
