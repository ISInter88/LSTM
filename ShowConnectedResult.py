import numpy as np
import pandas as pd
import plotly
from plotly import graph_objs as go

# Paths to the CSV files
pred_path = "Abilene_prediction.csv"
gt_path = "Abilene_target.csv"

# Read the data
pred = pd.read_csv(pred_path)
gt = pd.read_csv(gt_path)

# Convert data to numpy arrays
pred_list = np.array(pred)
gt_list = np.array(gt)

# Iterate through each dataset (assuming each column represents a dataset)
for p in range(4):
    # Extract the current dataset's predictions and ground truth values
    pred_data = pred_list[:, p]
    gt_data = gt_list[:, p]

    # Create the scatter plots
    d = go.Scatter(y=pred_data, name=f"Prediction {p+1}", line=dict(color='rgba(255, 0, 0, 0.5)'))
    e = go.Scatter(y=gt_data, name=f"True {p+1}", line=dict(color='rgba(0, 0, 255, 0.5)'))

    # Create the figure and add traces
    fig = go.Figure(d)
    fig.add_trace(e)

    #Update the layout of the plot
    fig.update_layout(
        title=dict(text='<b>Abilene dataset</b>', font=dict(size=50)),
        xaxis=dict(title=dict(text='Step', font=dict(size=50)), tickfont=dict(size=40)),
        yaxis=dict(title=dict(text='100Byte', font=dict(size=50)), tickfont=dict(size=40)),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', orientation='h', font=dict(size=30))
    )

    # fig.update_layout(
    #    title=dict(text='<b>ETT dataset</b>', font=dict(size=50)),
    #    xaxis=dict(title=dict(text='Step', font=dict(size=50)), tickfont=dict(size=40)),
    #    yaxis=dict(title=dict(text='Value', font=dict(size=50)), tickfont=dict(size=40)),
    #    legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', orientation='h', font=dict(size=30))
    # )

    fig.show()