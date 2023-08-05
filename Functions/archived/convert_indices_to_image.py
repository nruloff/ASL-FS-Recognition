def convert_indices_to_image(input_tensor_slice, square_image_dim=28):
  '''
  INPUT
  input_tensor_slice: tf.Tensor object;
  square_image_dim: integer; the length of one dimension for desired output image size

  OUTPUT
  tf.cast(dense_image, tf.int64): tf.Tensor object; Image generated for indices of input_tensor_slice, converting
                                  each pixel value in tf.Tensor object to integer dtype
  
  REQUIREMENTS
  Function requires tensorflow.experimental.numpy to be imported as tnp
  '''
  indices_values = tnp.sort(input_tensor_slice.numpy(), axis=0)
  update_values = tnp.ones(indices_values.shape[0])*10
  dense_image = tf.scatter_nd(indices=indices_values, updates=update_values, shape=np.array([square_image_dim, square_image_dim]) )
  return tf.cast(dense_image, tf.int64)
