# Function version of visualization created in Jupyter Notebook
def generate_figure_2(input_df):
  '''
  INPUT
  input_df: pandas dataframe object; dataframe created by asl-fingerspelling kaggle competition train.csv and/or supplemental_landmarks.csv file

  OUTPUT
  fig: plotly.graph_objs._figure object; Frame-Sequence Ratio Scatter Plot Visualization
  '''
  # Acquire Frame Counts and decrease the amount of bars present in the graph
  phrase_counts = input_df['phrase'].value_counts().reset_index()
  phrase_counts.columns = ['Phrase', 'Count']
  phrase_counts = phrase_counts[phrase_counts['Count'] > 113]
  # Create a bar graph using Plotly Express
  fig = px.bar(phrase_counts, x='Phrase', y='Count')
  fig.update_layout(
      title={
          'text': 'Types of Phrase and Its Count',
          'font_size': 20,
          'y':0.95, # y position of the title, higher value moves it up
          'x':0.5, # x position, 0.5 is the center
          'xanchor': 'center', # centers the text at the x coordinate
          'yanchor': 'top' # anchors the text at the top of the title space
          },
      xaxis_title='Phrase',
      xaxis_tickangle=45,
      yaxis_title='Phrase Count')
  return fig
