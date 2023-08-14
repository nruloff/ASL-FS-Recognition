def Frame_Sequence_Ratio_Visualization(input_df):
  '''
  INPUT
  input_df: pandas dataframe object; dataframe created by asl-fingerspelling kaggle competition train.csv and/or supplemental_landmarks.csv file

  OUTPUT
  fig: plotly.graph_objs._figure object; Frame-Sequence Ratio Scatter Plot Visualization
  '''
  data_df = input_df.copy()
  input_cols = input_df.columns.to_list()

  # Ensure that the columns necessary to make the visualization exist in the dataframe
  if 'sequence_length' not in input_cols:
    data_df['sequence_length'] = data_df['phrase'].str.len()
  if 'Frame_Count' not in input_cols:
    data_df = Gather_Visualization_Data(input_df=data_df)

  # Perform calculations to gather data for visualization
  data_df['Frame-Seq-Ratio'] = data_df['Frame_Count'] / data_df['sequence_length']
  data_df['Ratio_1'] = 'Blue'
  data_df.loc[(data_df['Frame-Seq-Ratio'] < 1), 'Ratio_1'] = 'Red'
  data_df['Data_Subset'] = data_df.path.str.split('_').str[0]

  # Generate Data for Line of 'Average Number of Frames per Character'
  slope = int(np.average(data_df['Frame-Seq-Ratio']))
  x_range = np.unique(data_df['sequence_length'])
  x_range = np.arange(x_range.min(), (x_range.max()+1))
  y_range = [x_val*slope for x_val in x_range]

  # Generate Figure
  fig = px.scatter(data_frame=data_df, x='sequence_length', y='Frame_Count', color='Ratio_1', title='Number of Frames per Total Sequence Characters')
  fig.add_trace(go.Scatter(x=x_range, y=y_range, line_shape='linear', showlegend=False, name="Average Frame-Sequence Ratio"))
  fig.update_layout(showlegend=False)
  return fig
