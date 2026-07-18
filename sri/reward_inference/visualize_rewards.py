import plotly.graph_objects as go
import numpy as np

# Sample data
np.random.seed(42)
x = np.random.standard_normal(100)
y = np.random.standard_normal(100)
z = np.random.standard_normal(100)
values = np.random.standard_normal(100)  # This could represent a 4th dimension.

# Create a Plotly figure
fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=values,  # Set color to the fourth dimension
        colorscale='Viridis',  # Choose a colorscale
        opacity=0.8
    )
)])

# Set titles and labels
fig.update_layout(title='Interactive 3D Scatter Plot', scene=dict(
                    xaxis_title='X Axis',
                    yaxis_title='Y Axis',
                    zaxis_title='Z Axis'))

# Show the figure
fig.write_html('plot.html')