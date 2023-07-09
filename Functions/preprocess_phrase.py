def preprocess_phrase(phrase):
  '''
  phrase: string
  encoder: dictionary

  Converts everything to lowercase. Removes any potential special characters not included in encoder.
  Adds "<start>" and "<end>" tokens to the phrase.
  Adds "transition tokens" to the phrase in between each of the character tokens (including <start> and <end>).
  Adds "<pad>" token to the phrase at the end.

  Returns:
  processed_phrase: string with "|" inserts to perform splits on
  '''
  phrase = phrase.lower()
  phrase = re.sub(r"[^a-z~_\[\]@?=;:0-9/\.\-\,\+\*\)\(\'\&\%\$\#\! ]", "", phrase)
  first_transition_insertion = '<start>_to_{}'.format(phrase[0])
  processed_example_phrase = ['<start>', first_transition_insertion]
  for i in np.arange((len(phrase)-1)):
    j = i + 1
    transition_insertion = '{}_to_{}'.format(phrase[i], phrase[j])
    processed_example_phrase.append(phrase[i])
    processed_example_phrase.append(transition_insertion)

  processed_example_phrase.append(phrase[len(phrase)-1])
  processed_example_phrase.append('<{}_to_<end>'.format(phrase[len(phrase)-1]))
  processed_example_phrase.append('<end>')
  processed_example_phrase.append('<end>_to_<pad>')
  processed_example_phrase.append('<pad>')

  processed_phrase = '|'.join(processed_example_phrase)
  return processed_phrase
