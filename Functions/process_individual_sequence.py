def process_individual_sequence(input_parquet_df, sequence_ID, square_image_dim=28):
  '''
  INPUT
  input_parquet_df: pandas DataFrame object;
  sequence_ID: integer;

  OUTPUT
  Five_Dimensional_Output: tf.Tensor object;
  '''
  # Multiple initial processing steps on input parquet df
  # 1. Filter the parquet file by the specific sequence
  # 2. Fill any NaN values - first by forward fill, then by backfill, then finally by simply 0
  # 3. Use tf.convert_to_tensor to convert the dataframe to a Tensor
  ex_seq_LM_tensor = tf.convert_to_tensor(
      input_parquet_df.loc[sequence_ID].set_index('frame').fillna(method="ffill").fillna(method="bfill").fillna(0))

  # 4. Use the previously defined 'pad_video_data()' function to bad the tensor to MAX_FRAME_LENGTH
  ex_seq_pad_LM_tensor = pad_video_data(ex_seq_LM_tensor, total_frames=MAX_FRAME_LENGTH)

  # 5. Stack the coordinates using the 'stack_coordinates()' function
  ex_seq_stack_LM_tensor = stack_coordinates(ex_seq_pad_LM_tensor)

  # 6. Multiply by the Square Image Dimensions Desired for translating the (x,y) coordinates into a pseudo-image
  square_image_tensor = tf.cast(tf.math.scalar_mul(square_image_dim, ex_seq_stack_LM_tensor), dtype=tf.int32)

  # 7. Bring the coordinates within the desired image frame dimensions
  shrink_in_tensor = tf.experimental.numpy.where(square_image_tensor > (square_image_dim -1), (square_image_dim -1), square_image_tensor)
  image_indices_tensor = tf.experimental.numpy.where(shrink_in_tensor < 0, 0, shrink_in_tensor)

  # 8. Take a slice out for each body part
  face_tensor = tf.cast(image_indices_tensor[:, tf.where(tf.strings.regex_full_match(selected_columns[1:], "x_face.*"))[0].numpy()[0]:
                                             tf.where(tf.strings.regex_full_match(selected_columns[1:], "x_face.*"))[-1].numpy()[0]], dtype=tf.int64)
  left_hand_tensor = tf.cast(image_indices_tensor[:, tf.where(tf.strings.regex_full_match(selected_columns[1:], "x_left_hand.*"))[0].numpy()[0]:
                                                  tf.where(tf.strings.regex_full_match(selected_columns[1:], "x_left_hand.*"))[-1].numpy()[0]], dtype=tf.int64)
  pose_tensor = tf.cast(image_indices_tensor[:, tf.where(tf.strings.regex_full_match(selected_columns[1:], "x_pose.*"))[0].numpy()[0]:
                                             tf.where(tf.strings.regex_full_match(selected_columns[1:], "x_pose.*"))[-1].numpy()[0]], dtype=tf.int64)
  right_hand_tensor = tf.cast(image_indices_tensor[:, tf.where(tf.strings.regex_full_match(selected_columns[1:], "x_right_hand.*"))[0].numpy()[0]:
                                                   tf.where(tf.strings.regex_full_match(selected_columns[1:], "x_right_hand.*"))[-1].numpy()[0]], dtype=tf.int64)

  # 9. Convert the image indices to a Dense Image using
  dense_face_tensor = tf.map_fn(fn=convert_indices_to_image, elems=face_tensor)
  dense_left_hand_tensor = tf.map_fn(fn=convert_indices_to_image, elems=left_hand_tensor)
  dense_pose_tensor = tf.map_fn(fn=convert_indices_to_image, elems=pose_tensor)
  dense_right_hand_tensor = tf.map_fn(fn=convert_indices_to_image, elems=right_hand_tensor)

  # 10. "Stack" the four images together for a single output to send to the Embedding Layer
  Five_Dimensional_Output = tf.expand_dims(tf.stack([dense_face_tensor, dense_left_hand_tensor, dense_pose_tensor, dense_right_hand_tensor], axis=3), axis=0)
  Five_Dimensional_Output = tf.cast(Five_Dimensional_Output, tf.float64)

  return Five_Dimensional_Output
