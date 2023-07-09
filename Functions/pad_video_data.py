def pad_video_data(input_tensor, total_frames=1000):
  '''
  INPUT
  input_tensor: tensor object from TensorFlow
  total_frames: int; the number of total frames desired for padding

  OUTPUT
  output_tensor: tensor object from TensorFlow

  Function to pad TensorFlow tensors to a maximum length frames such that all tensors have the same
  length for input into transformer model.
  '''
  
  rows_needed = total_frames - input_tensor.shape[0]
  return tf.pad(input_tensor, tf.constant([[0,rows_needed],[0,0]]))
