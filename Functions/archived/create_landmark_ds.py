# Source: Keras Transformer Model for Automatic Speech Recognition - https://keras.io/examples/audio/transformer_asr/
def create_landmark_ds(data=train_df):
  '''
  INPUT
  data: pd.DataFrame object;

  OUTPUT
  landmark_ds: tf.data.Dataset object;

  '''
  landmark_ds = []
  for file_path in data['path'].unique():
    ex_parquet_df = load_relevant_data_subset('/content/drive/MyDrive/kaggle/input/asl-fingerspelling/{}'.format(file_id))
    for seq_identifier in data[data['path'] == file_path].sequence_id.unique():
      landmark_ds.append(process_individual_sequence(ex_parquet_df, seq_identifier))
  landmark_ds = tf.data.Dataset.from_tensor_slices(landmark_ds)
  return landmark_ds
