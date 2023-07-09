def custom_splitting(input_string):
  '''
  INPUT
  input_string - string to split. Expected input is a string that has gone through "preprocess_phrase.py"

  OUTPUT:
  A RaggedTensor of rank N+1, the strings split according to the delimiter.
  '''
  return tf.strings.split(input_string, sep="|")
