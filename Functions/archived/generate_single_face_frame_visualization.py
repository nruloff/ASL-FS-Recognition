def generate_single_face_frame_visualization(pq_path, seq_id, all_face_indices=False):
  ex_parq_df = load_relevant_data_subset(pq_path)
  ex_parq_df = ex_parq_df.loc[seq_id]
  ex_parq_df = ex_parq_df.head()
  ex_frame_df = ex_parq_df.set_index('frame')
  face_cols = []
  if all_face_indices:
    face_indices = [val for val in np.arange(0, 478)]
  else:
    face_indices = [0, 4, 13, 14, 17, 33, 37, 39, 46, 52, 55, 61, 64, 81, 82, 93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276, 282, 285, 291, 294, 311, 323, 362, 386, 397]
  for dim_id in ['x', 'y']:
    face_cols += ['{}_face_{}'.format(dim_id, col_count_val) for col_count_val in face_indices]
  ex_frame_df = ex_frame_df[face_cols]
  ex_frame_df = ex_frame_df.reset_index()
  ex_part_long = pd.wide_to_long(ex_frame_df, ["x", "y"], i="frame", j="body_part_id", sep="_", suffix='.+').dropna().reset_index()
  ex_part_long = ex_part_long[ex_part_long['frame'] == 0]
  fig = px.scatter(ex_part_long, x='x', y='y', hover_name='body_part_id', )
  fig.show()
  return None
