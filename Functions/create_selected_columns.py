def create_selected_columns(body_part_IDs=['face', 'left_hand', 'pose', 'right_hand'], dimension_IDs=['x', 'y']):
    '''
    Function to create the gobal variable "selected_columns" which selects specific columns to use for analysis/prediction of ASL Fingerspelling videos

    INPUT
    body_parts_IDs: list of strings; body parts to include from parquet files
    dimension_IDs: list of strings - limited to 'x', 'y' and 'z'; x, y, or z dimensions to include of body part Mediapipe ellicited locations

    OUTPUT
    selected_columns: list of strings; columns to select to include from the parquet file read into the model
    '''
    selected_columns = ['frame']
    # Indice selections for 'face' and 'pose' based on the preprint of YouTube-ASL
    face_indices = [0, 4, 13, 14, 17, 33, 37, 39, 46, 52, 55, 61, 64, 81, 82, 93, 133, 151, 152, 159,
                    172, 178, 181, 263, 269, 276, 282, 285, 291, 294, 311, 323, 362, 386, 397]
    pose_indices = [11, 12, 13, 14, 23, 24]
    if 'face' in body_part_IDs:
        for dim_id in dimension_IDs:
            selected_columns += ['{}_face_{}'.format(dim_id, col_count_val) for col_count_val in face_indices]
    if 'left_hand' in body_part_IDs:
        for dim_id in dimension_IDs:
            selected_columns += ['{}_left_hand_{}'.format(dim_id, col_count_val) for col_count_val in np.arange(0, 21)]
    if 'pose' in body_part_IDs:
        for dim_id in dimension_IDs:
            selected_columns += ['{}_pose_{}'.format(dim_id, col_count_val) for col_count_val in pose_indices]
    if 'right_hand' in body_part_IDs:
        for dim_id in dimension_IDs:
            selected_columns += ['{}_right_hand_{}'.format(dim_id, col_count_val) for col_count_val in np.arange(0, 21)]

    return selected_columns
