def generate_hand_signing_visualization(pq_path, seq_id, selected_hand):
  '''
  INPUT
  pq_path: string; filepath string for parquet file that contains desired sequence to be visualized.
  seq_id: int; integer value for the sequence ID of the desired fingerspelling sequence to be visualized
  selected_hand: string; choice from ['right', 'left'] - decides which hand to present in visualization

  OUTPUT
  fig: plotly express figure object
  '''
  full_path = './kaggle/input/asl-fingerspelling/' + pq_path
  ex_parquet_df = pd.read_parquet(full_path)
  ex_parquet_df = ex_parquet_df.loc[seq_id].set_index('frame').fillna(method="ffill").fillna(method="bfill")
  ex_parquet_df = ex_parquet_df.reset_index()
  ex_part_long = pd.wide_to_long(ex_parquet_df, ["x", "y", "z"], i="frame", j="body_part_id", sep="_", suffix='.+').dropna().reset_index()
  ex_part_long['body_part_id'] = ex_part_long['body_part_id'].str.replace('_hand', '-hand')
  ex_part_long[['body_part', 'sub_id']] = ex_part_long['body_part_id'].str.split('_', expand=True)
  if selected_hand not in ex_part_long.body_part.unique():
    print("Selected hand is not present in the selected sequence.")
    return None
  if selected_hand == 'right-hand':
    right_hand = ex_part_long[ex_part_long['body_part'] == 'right-hand']
  else:
    right_hand = ex_part_long[ex_part_long['body_part'] == 'left-hand']

  thumb = [1, 2, 3, 4]
  palm = [0, 1, 5, 9, 13, 17]
  index = [5, 6, 7, 8]
  middle = [9, 10, 11, 12]
  ring = [13, 14, 15, 16]
  pinky = [17, 18, 19, 20]
  fingers = {'thumb': thumb, 'index': index, 'middle': middle, 'ring': ring, 'pinky': pinky}
  palm_df = right_hand.copy()
  palm_id_list = []

  for val in palm:
    palm_id_list.append('{}_{}'.format(selected_hand, val))
  palm_df = palm_df[palm_df['body_part_id'].isin(palm_id_list)]
  quick_duplicate = palm_df.copy()
  quick_duplicate = quick_duplicate[quick_duplicate['body_part_id'] == '{}_0'.format(selected_hand)]
  quick_duplicate = quick_duplicate.replace(to_replace='{}_0'.format(selected_hand), value='palm_6')
  for m in np.arange(len(palm)):
    palm_df = palm_df.replace(to_replace='{}_{}'.format(selected_hand, palm[m]), value='palm_{}'.format(m))
  final_df = pd.concat([palm_df, quick_duplicate], ignore_index=True)
  
  for finger in fingers:
    body_part_id_list = []
    finger_list = fingers[finger]
    for val in finger_list:
      body_part_id_list.append('{}_{}'.format(selected_hand, val))
      temp_finger_df = right_hand[right_hand['body_part_id'].isin(body_part_id_list)].copy()
    for n in np.arange(len(body_part_id_list)):
      temp_finger_df = temp_finger_df.replace(to_replace=body_part_id_list[n], value='{}_{}'.format(finger, n))
    if final_df is not None:
      final_df = pd.concat([final_df, temp_finger_df.copy()], ignore_index=True)
    else:
      final_df = temp_finger_df.copy()

  final_df = final_df.drop(columns=['body_part', 'sub_id'])
  final_df[['body_part', 'sub_id']] = final_df['body_part_id'].str.split('_', expand=True)

  fig = px.scatter(final_df, x='x', y='y', color='body_part', animation_frame='frame', 
                   range_x=[0, 1], range_y=[0,1], hover_name='body_part_id', width=600, height=800)
  fig.update_layout(xaxis_title=None, yaxis_title=None)
  fig.update_xaxes(visible=False)
  fig.update_yaxes(visible=False)
  fig.for_each_trace(lambda t: t.update(mode='lines+markers'))

  for fr in fig.frames:
    for d in fr.data:
      d.update(mode='markers+lines')

  
  return fig
