def create_text_vectorizer(include_transition_keys=False, include_capitalization=False, seq_length_input=MAX_SEQUENCE_LENGTH):
    '''
    INPUT
    include_transition_keys: Boolean; determines if model should try to capture the hand signs made
                             in between signs for specific letters
    include_capitalization: Boolean; determines if model should try to capture hand signs made to indicate
                            capitalization of a word
    seq_length_input: Integer; Value of the sequence length selected for the Transformer Model

    OUTPUT
    vectorizer: layers.preprocessing.text_vectorization.TextVectorization; Used for letter tokenization
    all_keys: dict; dictionary for encoding/decoding the tokenization of phrases

    '''
    # Load initial character_to_prediction_index.json file provided by competition
    character_to_prediction_index_file_path = '/content/drive/MyDrive/kaggle/input/character_to_prediction_index.json'
    try:
      char_pred_index = json.load(open(character_to_prediction_index_file_path))
    except:
      dwn_load_cmd = 'kaggle competitions download asl-fingerspelling -f {} -p /content/drive/MyDrive/kaggle/input'.format('character_to_prediction_index.json')
      unzip_cmd = 'unzip /content/drive/MyDrive/kaggle/input/{}.zip -d /content/drive/MyDrive/kaggle/input'.format('character_to_prediction_index.json')
      del_cmd = 'rm /content/drive/MyDrive/kaggle/input/{}.zip'.format('character_to_prediction_index.json')
      os.system(dwn_load_cmd)
      os.system(unzip_cmd)
      os.system(del_cmd)
      char_pred_index = json.load(open(character_to_prediction_index_file_path))

    # Add START and END tokens
    char_pred_index['<start>'] = 59
    char_pred_index['<end>'] = 60

    # Add CAPITALIZATION token if 'include_capitalization' is True
    if include_capitalization:
        char_pred_index['<capitalization>'] = 61

    # Generate a copy of the prediction keys to allow for "include_transition_keys=True" input
    all_keys = char_pred_index.copy()

    # Check whether transition_keys should be included
    if include_transition_keys:
        transition_keys = dict()
        i = np.max([val for val in char_pred_index.values()])
        for char in [char_pred_key for char_pred_key in char_pred_index.keys()]:
            for char_2 in [char_pred_key for char_pred_key in char_pred_index.keys()]:
                key_string = '{}_to_{}'.format(char, char_2)
                transition_keys[key_string] = i
                i += 1
        all_keys.update(transition_keys)

    # Build the TextVectorizer for English Text translation to numbers
    vectorizer=tf.keras.layers.TextVectorization(standardize=None,
                                                 split=custom_splitting,
                                                 output_mode='int',
                                                 max_tokens=None,
                                                 vocabulary=[key_str for key_str in all_keys],
                                                 output_sequence_length=int(MAX_SEQUENCE_LENGTH +1))
    return vectorizer
