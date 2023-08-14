def preprocess_phrase(phrase, transition_keys=False):
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
    # Make all letters in sequence lowercase
    phrase = phrase.lower()
    # Replace anything in the phrase which is not included in the character_to_prediction_index provided by
    # the Kaggle competition
    phrase = re.sub(r"[^a-z~_\[\]@?=;:0-9/\.\-\,\+\*\)\(\'\&\%\$\#\! ]", "", phrase)
    # Create a list and make the first token which is <start>
    processed_example_phrase = ['<start>']
    # If transition_keys is set to 'True' include the first transition token
    if transition_keys:
        first_transition_insertion = '<start>_to_{}'.format(phrase[0])
        processed_example_phrase.append(first_transition_insertion)
    # For each letter in the phrase, append it to the list
    for i in np.arange((len(phrase)-1)):
        processed_example_phrase.append(phrase[i])
        # If transition_keys is True, append transition keys in between individual characters
        if transition_keys:
            j = i + 1
            transition_insertion = '{}_to_{}'.format(phrase[i], phrase[j])
            processed_example_phrase.append(transition_insertion)
    # Append the final letter of the phrase to the list
    processed_example_phrase.append(phrase[len(phrase)-1])
    # If transition_keys is True, append the transition from the final character to the <end> token
    if transition_keys:
        processed_example_phrase.append('<{}_to_<end>'.format(phrase[len(phrase)-1]))
    # Append the final token, which is <end>
    processed_example_phrase.append('<end>')
    # Convert the list into a single string which is separated by | (required for splitting by custom_splitting.py)
    processed_phrase = '|'.join(processed_example_phrase)
    return processed_phrase
