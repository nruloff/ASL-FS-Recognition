import pandas as pd

def load_relevant_data_subset(pq_path):
  '''
  Function designed by competition to demonstrate how files will be loaded for testing prediction models.
  
  INPUT
  pq_path: file path to parquet file containing mediapipe holistic model data
  selected_columns: global variable which users can assign in their model to predetermine which columns to use for predictions.
  '''
  return pd.read_parquet(pq_path, columns=selected_columns)
