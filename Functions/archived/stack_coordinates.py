def stack_coordinates(padded_tensor, selected_columns=selected_columns, dimension_IDs=['x', 'y']):
  '''
  INPUT
  padded_tensor: tf.Tensor object;
  selected_columns: list of strings; Matches global variable "selected_columns"
  dimension_IDs: list of strings; limited to only 'x', 'y' and 'z'.

  OUTPUT
  stacked_tensor: tf.Tensor object;
  '''
  # Determine the number of columns per dimension
  col_per_dim = int((len(selected_columns)-1)/len(dimension_IDs))

  # Based on the number of dimensions - perform a dstack() of the Tensor slices
  if len(dimension_IDs) == 2:
    stacked_tensor = tf.stack([padded_tensor[:, :col_per_dim], padded_tensor[:, col_per_dim:]], axis=2)
  else:
    stacked_tensor = tf.stack([padded_tensor[:, :col_per_dim],
                               padded_tensor[:, col_per_dim:int(2*col_per_dim)],
                               padded_tensor[:, int(2*col_per_dim):]], axis=2)
  return stacked_tensor
