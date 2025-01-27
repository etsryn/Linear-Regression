import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

train_data = pd.read_csv('train_data.csv')

with open('linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(train_data['col1'], train_data['col2'], train_data['col3'], 
           color='blue', label='Train Data Points')

col1_range = np.linspace(train_data['col1'].min(), train_data['col1'].max(), 10)
col2_range = np.linspace(train_data['col2'].min(), train_data['col2'].max(), 10)
col1_grid, col2_grid = np.meshgrid(col1_range, col2_range)

grid_input = np.c_[col1_grid.ravel(), col2_grid.ravel()]
col3_predicted = model.predict(grid_input).reshape(col1_grid.shape)

reg_plane = ax.plot_surface(col1_grid, col2_grid, col3_predicted, 
                            color='red', alpha=0.5, rstride=100, cstride=100)

ax.set_xlabel('col1')
ax.set_ylabel('col2')
ax.set_zlabel('col3')
ax.set_title('3D Regression Plot with Full Rotation')

def update(frame):
    azim = frame % 360
    elev = (frame // 2) % 180
    ax.view_init(elev=elev, azim=azim)

frames = 360 * 2
ani = FuncAnimation(fig, update, frames=frames, interval=50)

# To Save the animation as a GIF
# ani.save('full_rotating_3d_regression.gif', writer='pillow')

plt.show()
