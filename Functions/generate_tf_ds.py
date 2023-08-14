# Source: Hugging Face - Using Datasets with TensorFlow (https://huggingface.co/docs/datasets/use_with_tensorflow)
# with some specific adaptations to this project
def generate_tf_ds(input_ex_pq_path, input_data_df, input_batch_size=2):
    '''
    INPUT
    input_ex_pq_path: string; filepath for the parquet file to be loaded.
    input_data_df: pandas DataFrame; dataframe containing the metadata information of the parquet files
    input_batch_size: integer; number of ["source", "target"] pairs to collate together into tf.data.Dataset batch.

    OUTPUT
    tf_ds: tf.data.Dataset; Generates a tf.data.Dataset from the underlying arrow_dataset (https://github.com/huggingface/datasets/blob/2.14.3/src/datasets/arrow_dataset.py#L326)
    '''
    ex_parquet_df = load_relevant_data_subset(input_ex_pq_path)
    NAN_FILL_VALUE = determine_nan_fill_value(input_data_df=input_data_df)
    data = {"target": [], "source":[]}
    for ind_seq_id in ex_parquet_df.index.unique():
        if ind_seq_id in input_data_df['sequence_id'].unique():
            ind_seq_parquet_df = ex_parquet_df.loc[[ind_seq_id]].set_index('frame').fillna(NAN_FILL_VALUE)
            ind_seq_parquet_df = ind_seq_parquet_df.reindex(range(int(MAX_FRAME_LENGTH)), fill_value=NAN_FILL_VALUE)
            data["source"].append(ind_seq_parquet_df.to_numpy())
            preprocessed_text = preprocess_phrase(input_data_df[input_data_df['sequence_id'] == ind_seq_id].phrase.iloc[0])
            data["target"].append(vectorizer(preprocessed_text))
    ds = Dataset.from_dict(data)
    tf_ds = ds.to_tf_dataset(columns=["source", "target"], batch_size=input_batch_size, shuffle=True, collate_fn=None)
    return tf_ds
