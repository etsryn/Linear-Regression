from mpl_toolkits.mplot3d import Axes3D  # 3D plotting tools
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Loading Data Set
train_data = pd.read_csv('train_data.csv')

# Load the model from the .pkl file
with open('linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the actual train data points
ax.scatter(train_data['col1'], train_data['col2'], train_data['col3'], 
           color='blue', label='Train Data Points')

# Generate a grid for col1 and col2 to plot the regression plane
col1_range = np.linspace(train_data['col1'].min(), train_data['col1'].max(), 10)
col2_range = np.linspace(train_data['col2'].min(), train_data['col2'].max(), 10)
col1_grid, col2_grid = np.meshgrid(col1_range, col2_range)

# Flatten the grid and predict col3 values
grid_input = np.c_[col1_grid.ravel(), col2_grid.ravel()]
col3_predicted = model.predict(grid_input).reshape(col1_grid.shape)

# Plot the regression plane
reg_plane = ax.plot_surface(col1_grid, col2_grid, col3_predicted, 
                            color='red', alpha=0.5, rstride=100, cstride=100)

# Set plot labels and title
ax.set_xlabel('col1')
ax.set_ylabel('col2')
ax.set_zlabel('col3')
ax.set_title('3D Regression Plot with Alternating Rotation')

# Function to update the view angle for each frame
def update(frame):
    # Alternate between horizontal and vertical rotation
    if frame < 180:
        ax.view_init(elev=20, azim=frame)  # Horizontal rotation
    else:
        ax.view_init(elev=frame - 180, azim=180)  # Vertical rotation

# Create the animation, rotating 180° horizontally, then 180° vertically
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=100)

# Save the animation as a GIF
ani.save('bi_directional_rotation_3d_regression.gif', writer='pillow')

# Show the plot
plt.show()