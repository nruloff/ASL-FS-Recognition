# Function used in MADS-ASL-FR Colab Walkthrough Notebook

# Source of download_file(): https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
def download_file(url):
    local_filename = '/content/drive/MyDrive/kaggle/input/' + url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename

# Leverages the download_file() function to access the updated Metadata files
def Acquire_Metadata():
  if os.path.exists('/content/drive/MyDrive/kaggle/input/asl-fingerspelling/train_landmarks'):
    printable_statement = 'File path representative of Kaggle Competition exists'
  else:
    os.makedirs('/content/drive/MyDrive/kaggle/input/asl-fingerspelling/train_landmarks')

  if os.path.exists('/content/drive/MyDrive/kaggle/input/train_and_supplemental_EDA.pkl'):
    data_df = pd.read_pickle('/content/drive/MyDrive/kaggle/input/train_and_supplemental_EDA.pkl')
  else:
    download_file('https://github.com/nruloff/MADS-ASL-FR/raw/main/Preprocessed_Metadata_Files/train_EDA.pkl')
    download_file('https://github.com/nruloff/MADS-ASL-FR/raw/main/Preprocessed_Metadata_Files/supplemental_EDA.pkl')
    train_df = pd.read_pickle('/content/drive/MyDrive/kaggle/input/train_EDA.pkl')
    supplemental_df = pd.read_pickle('/content/drive/MyDrive/kaggle/input/supplemental_EDA.pkl')
    data_df = pd.concat([train_df, supplemental_df], ignore_index=True)
    data_df.to_pickle('/content/drive/MyDrive/kaggle/input/train_and_supplemental_EDA.pkl')

  return data_df
