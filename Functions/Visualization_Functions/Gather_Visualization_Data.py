def Gather_Visualization_Data(input_df, kaggle_env=False):
  '''
  INPUT
  input_df: pandas dataframe object; dataframe created by asl-fingerspelling kaggle competition train.csv and/or supplemental_landmarks.csv file
  kaggle_env: boolean; value to indicate whether this function is used in a Google Colab Environment
  
  OUTPUT
  train_df: pandas dataframe object; Similar to input_df but with input values of information for generating visualizations
  
  ###############################################################################################################################
  
  NOTE: If using in Google Colab Environment, this function requires uploading kaggle.json API key for downloading files remotely
        as well as Mounting of a Google Drive and Authorization to delete files from Google Drive Trashbin.
  '''
  train_df = input_df.copy()
  input_cols = input_df.columns.to_list()
  if 'right_hand_max_x' not in input_cols:
    cols_to_add = create_body_min_max_columns()
    train_df['Frame_Count'] = np.nan
    train_df[[cols_to_add]] = np.nan
  files_of_interest = input_df.groupby('path').count().sort_values('file_id', ascending=False).reset_index().drop(columns=input_cols[1:])
  for file_id in files_of_interest['path']:
    print('Grathering Data from {}'.format(file_id))
    # Load the parquet file into a pandas dataframe based on whether this function is used in Kaggle Environment
    # Or Google Colab Environment
    if kaggle_env:
      temp_landmarks_df = pd.read_parquet('/kaggle/input/asl-fingerspelling/{}'.format(file_id))
    else:
      colab_file_rename = file_id.split('/')[1]
      dwn_load_cmd = 'kaggle competitions download asl-fingerspelling -f {} -p /content/drive/MyDrive/kaggle/input'.format(file_id)
      unzip_cmd = 'unzip /content/drive/MyDrive/kaggle/input/{}.zip -d /content/drive/MyDrive/kaggle/input'.format(colab_file_rename)
      del_cmd = 'rm /content/drive/MyDrive/kaggle/input/{}.zip'.format(colab_file_rename)
      os.system(dwn_load_cmd)
      os.system(unzip_cmd)
      os.system(del_cmd)
      temp_landmarks_df = pd.read_parquet('/content/drive/MyDrive/kaggle/input/{}'.format(colab_file_rename)).reset_index()
    # Once the File is Loaded In to the parquet_df - gather the information
    original_cols = temp_landmarks_df.columns
    file_specific_cut = train_df[train_df['path'] == file_id]
    sequence_ids = train_df[train_df['path'] == file_id].sequence_id.to_list()
    grouped = temp_landmarks_df.groupby('sequence_id').agg(np.nanmax).reset_index()
    grouped_long = pd.wide_to_long(grouped, ["x", "y", "z"], i="sequence_id", j="body_part_id", sep="_", suffix='.+').dropna().reset_index()
    grouped_long['body_part_id'] = grouped_long['body_part_id'].str.replace('_hand', '-hand')
    grouped_long[['body_part', 'sub_id']] = grouped_long['body_part_id'].str.split('_', expand=True)
    grouped_long = grouped_long.drop(columns=['body_part_id']).groupby(['sequence_id', 'body_part']).agg(np.nanmax).reset_index()
    grouped_long['body_part'] = grouped_long['body_part'].str.replace('-hand', '_hand')
    grouped_long = grouped_long.copy()
    grouped = temp_landmarks_df.groupby('sequence_id').agg(np.nanmin).reset_index()
    grouped_long_b = pd.wide_to_long(grouped, ["x", "y", "z"], i="sequence_id", j="body_part_id", sep="_", suffix='.+').dropna().reset_index()
    grouped_long_b['body_part_id'] = grouped_long_b['body_part_id'].str.replace('_hand', '-hand')
    grouped_long_b[['body_part', 'sub_id']] = grouped_long_b['body_part_id'].str.split('_', expand=True)
    grouped_long_b = grouped_long_b.drop(columns=['body_part_id']).groupby(['sequence_id', 'body_part']).agg(np.nanmin).reset_index()
    grouped_long_b['body_part'] = grouped_long_b['body_part'].str.replace('-hand', '_hand')
    grouped_long_b = grouped_long_b.copy()
    for seq_id in sequence_ids:
      frame_count = len(temp_landmarks_df[temp_landmarks_df['sequence_id'] == seq_id])
      row_index_for_update = file_specific_cut[file_specific_cut['sequence_id'] == seq_id].index[0]
      train_df.loc[row_index_for_update, 'Frame_Count'] = frame_count
      grouped_short = grouped_long[grouped_long['sequence_id'] == seq_id]
      for BP_x in ['face', 'pose', 'left_hand', 'right_hand']:
        if BP_x in grouped_short.body_part.values:
          x_value = float(grouped_short[grouped_short['body_part'] == '{}'.format(BP_x)].x.iloc[0])
          y_value = float(grouped_short[grouped_short['body_part'] == '{}'.format(BP_x)].y.iloc[0])
          z_value = float(grouped_short[grouped_short['body_part'] == '{}'.format(BP_x)].z.iloc[0])
          train_df.loc[row_index_for_update, '{}_max_x'.format(BP_x)] = x_value
          train_df.loc[row_index_for_update, '{}_max_y'.format(BP_x)] = y_value
          train_df.loc[row_index_for_update, '{}_max_z'.format(BP_x)] = z_value
        else:
          t_value = 0
      grouped_short = grouped_long_b[grouped_long_b['sequence_id'] == seq_id]
      for BP_x in ['face', 'pose', 'left_hand', 'right_hand']:
        if BP_x in grouped_short.body_part.values:
          x_value = float(grouped_short[grouped_short['body_part'] == '{}'.format(BP_x)].x.iloc[0])
          y_value = float(grouped_short[grouped_short['body_part'] == '{}'.format(BP_x)].y.iloc[0])
          z_value = float(grouped_short[grouped_short['body_part'] == '{}'.format(BP_x)].z.iloc[0])
          train_df.loc[row_index_for_update, '{}_min_x'.format(BP_x)] = x_value
          train_df.loc[row_index_for_update, '{}_min_y'.format(BP_x)] = y_value
          train_df.loc[row_index_for_update, '{}_min_z'.format(BP_x)] = z_value
        else:
          t_value = 0
    if not kaggle_env:      
      final_del_cmd = 'rm /content/drive/MyDrive/kaggle/input/{}'.format(colab_file_rename)
      os.system(final_del_cmd)
      for a_file in my_drive.ListFile({'q': "trashed = true"}).GetList():
        print('The file {}, is about to get deleted permanently.'.format(a_file['title']))
        a_file.Delete()
  return train_df
