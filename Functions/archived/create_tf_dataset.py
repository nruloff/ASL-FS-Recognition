# Source: Keras Transformer Model for Automatic Speech Recognition - https://keras.io/examples/audio/transformer_asr/
def create_tf_dataset(data=train_df, bs=4):
  '''
  INPUT
  data: pd.DataFrame object;
  bs: interger; batch size

  OUTPUT
  ds: tf.data.Dataset object;

  Generates both the text dataset and landmark dataset in zipped pairs to input into Transformer Model

  '''
  landmark_ds = create_landmark_ds(data)
  text_ds = create_text_ds(data)
  ds = tf.data.Dataset.zip((landmark_ds, text_ds))
  ds = ds.map(lambda x,y: {"source": x, "target": y})
  ds = ds.batch(bs)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds
