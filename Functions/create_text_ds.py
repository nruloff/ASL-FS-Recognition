# Source: Keras Transformer Model for Automatic Speech Recognition - https://keras.io/examples/audio/transformer_asr/
def create_text_ds(data=train_df):
  '''
  INPUT
  data: pd.Series object;

  OUTPUT
  text_ds: tf.data.Dataset object;

  '''
  vectorizer, _ = create_text_vectorizer()
  ex_phrases = [preprocess_phrase(ind_phrase) for ind_phrase in data['phrase']]
  text_ds = [vectorizer(ex_phrase) for ex_phrase in ex_phrases]
  text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
  return text_ds
