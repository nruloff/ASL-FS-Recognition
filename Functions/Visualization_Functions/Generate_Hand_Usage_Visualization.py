def Generate_Hand_Usage_Visualization(input_df):
  '''
  INPUT
  input_df: pandas dataframe object; dataframe created by asl-fingerspelling kaggle competition train.csv and/or supplemental_landmarks.csv file

  OUTPUT
  fig: plotly.graph_objs._figure object; Bar Graph Visualization of Hand Usage in the parquet files
  '''
  # Check if the input dataframe contains right_hand_max_x information - an indication that each of the sequences
  # has been checked for right hand / left hand usage
  if 'right_hand_max_x' not in input_df.columns.to_list():
    data_df = Gather_Visualization_Data(input_df=input_df)
  else:
    data_df = input_df.copy()

  # Separate the bar graph by each of the file groups
  data_df['Data_Subset'] = data_df.path.str.split('_').str[0]

  # Grab some of the original columns of the input_df
  original_columns = ['path', 'sequence_id', 'phrase', 'Data_Subset']

  # Generate a list of column names specific to the hands and combine with the original columns
  hand_specific_columns = create_body_min_max_columns(body_part_IDs=['left_hand', 'right_hand'], dimension_IDs=['x', 'y'])
  final_hand_columns = original_columns + hand_specific_columns

  # Generate a list of columns specific to the left hand
  left_hand_columns = create_body_min_max_columns(body_part_IDs=['left_hand'], dimension_IDs=['x', 'y'])

  # Make a copy of the data_df specifically focusing  on the columns for hands, group that column by 'Data_Subset'
  hand_df = data_df[final_hand_columns].copy()
  hand_grp = hand_df.groupby('Data_Subset').count()
  total_samples = hand_grp.path
  file_list = [grp_id[0].upper() + grp_id[1:] for grp_id in hand_grp.index]

  # Create dataframes focused on either right hand users or left/ambidextrous users
  left_hand_df = hand_df.dropna(subset=left_hand_columns, how='all')
  right_hand_df = hand_df.drop(left_hand_df.index, axis=0)
  left_hand_grp = left_hand_df.groupby('Data_Subset').count()
  right_hand_grp = right_hand_df.groupby('Data_Subset').count()
  fig = go.Figure()
  fig.add_trace(go.Bar(name="Right Hand Only", x=file_list, y=[(right_hand_grp['path'].loc[grp_id]/total_samples.loc[grp_id]) for grp_id in hand_grp.index]))
  fig.add_trace(go.Bar(name="Left Hand or Ambidextrous", x=file_list, y=[(left_hand_grp['path'].loc[grp_id]/total_samples.loc[grp_id]) for grp_id in hand_grp.index]))
  fig.update_layout(title='Percentage of Hand Usage in Train Landmarks and Supplemental Landmarks Datasets')
  return fig
