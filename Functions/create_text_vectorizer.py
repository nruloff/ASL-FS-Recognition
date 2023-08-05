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
    
    '''
    # Load initial character_to_prediction_index.json file provided by competition
    character_to_prediction_index_file_path = '/kaggle/input/asl-fingerspelling/character_to_prediction_index.json'
    char_pred_index = json.load(open(character_to_prediction_index_file_path))

    # Add START, END, and PAD tokens
    char_pred_index['<start>'] = 59
    char_pred_index['<end>'] = 60
    char_pred_index['<pad>'] = 61
    
    # Add CAPITALIZATION token if 'include_capitalization' is True
    if include_capitalization:
        char_pred_index['<capitalization>'] = 62
    
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
                                                 output_sequence_length=int(seq_length_input + 1))
    return vectorizer
