def create_body_min_max_columns(body_part_IDs=['face', 'left_hand', 'pose', 'right_hand'], dimension_IDs=['x', 'y', 'z']):
  '''
  INPUT
  body_part_IDs: list of strings; limited to ['face', 'left_hand', 'pose', 'right_hand'] based on column names of MediaPipe Outputs
  dimension_IDs: list of strings; limited to ['x', 'y', 'z']

  OUTPUT
  body_min_max_columns: list of strings
  '''
  # Generate Empty List
  body_min_max_columns = []
  # Append body part max and min columns for each variable in dimension_IDs
  for body_part_ID in body_part_IDs:
    for dim_ID in dimension_IDs:
      for val_description in ['max', 'min']:
        body_min_max_columns.append('{}_{}_{}'.format(body_part_ID, val_description, dim_ID))
  return body_min_max_columns
