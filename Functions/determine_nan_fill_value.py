def determine_nan_fill_value(input_data_df):
  '''
  INPUT
  input_df: pandas dataframe object; dataframe created by asl-fingerspelling kaggle competition train.csv and/or supplemental_landmarks.csv file

  OUTPUT
  nan_fill_value: integer
  '''
  # Generate body_min_max_columns using previously made function
  meta_data_max_min_cols = create_body_min_max_columns()
  # Obtain only the columns of interest from the input dataframe
  meta_data_max_min_df = input_data_df[meta_data_max_min_cols]
  # Calculate the nan_fill_value based on this dataframe subsection
  nan_fill_value = int(np.min(meta_data_max_min_df.min())*10)
  return nan_fill_value
