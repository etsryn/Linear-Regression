import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Read Training Data from Excel
train_data = pd.read_excel('train_data.xlsx')

# Features and labels for training
X_train = train_data[['col1', 'col2']]
y_train = train_data['col3']

# Step 2: Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 3: Read Testing Data from Excel
test_data = pd.read_excel('test_data.xlsx')

# Predict the values of col3
predictions = model.predict(test_data)

# Displaying Predictions
for i in range(len(test_data)):    
    print(f"Input: {test_data.iloc[i].tolist()}, Predicted col3: {predictions[i]}")

# Step 4: Evaluate the Model on Training Data

# Mean Squared Error (MSE)
mse = mean_squared_error(y_train, model.predict(X_train))
print(f"Mean Squared Error (MSE) on Training Data: {mse}")

# R-squared Score (R²)
r2 = r2_score(y_train, model.predict(X_train))
print(f"R-squared Score (R²) on Training Data: {r2}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Step 1: Read Training Data from Excel
train_data = pd.read_excel('train_data.xlsx')

# Features and labels for training
X_train = train_data[['col1', 'col2']]
y_train = train_data['col3']

# Step 2: Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 3: Extracting col1 and col2 for plotting
col1 = train_data['col1']
col2 = train_data['col2']

# Generating predicted values using the trained model for the line of regression
# Creating a range of values for col1 and col2 for plotting the regression surface
col1_range = np.linspace(col1.min(), col1.max(), 100)
col2_range = np.linspace(col2.min(), col2.max(), 100)

# Creating a meshgrid for plotting the surface (for 3D representation)
col1_mesh, col2_mesh = np.meshgrid(col1_range, col2_range)
X_mesh = np.c_[col1_mesh.ravel(), col2_mesh.ravel()]

# Predicting values using the model
y_mesh = model.predict(X_mesh).reshape(col1_mesh.shape)

# Step 4: Plotting the data points and the regression line in multiple 3D views
fig = plt.figure(figsize=(20, 18))  # Adjusted figure size for better clarity

# Angles for different views
angles = [(20, 30), (20, 120), (20, 210), (30, 300), (60, 30), (90, 0)]

# Loop to create subplots in 3 rows and 2 columns
for i, angle in enumerate(angles):
    ax = fig.add_subplot(3, 2, i + 1, projection='3d')  # Changed to 3 rows and 2 columns

    # Plotting the actual data points
    ax.scatter(col1, col2, y_train, color='blue', label='Data Points')

    # Plotting the regression surface
    ax.plot_surface(col1_mesh, col2_mesh, y_mesh, color='red', alpha=0.3)

    ax.set_xlabel('col1')
    ax.set_ylabel('col2')
    ax.set_zlabel('col3')
    ax.set_title(f'View Angle: Elev={angle[0]}, Azim={angle[1]}')
    
    # Setting the view angle
    ax.view_init(elev=angle[0], azim=angle[1])

plt.tight_layout()

# Step 5: Save the plot to a file
plt.savefig('3d_regression_views.png')  # Save as PNG file
plt.show()
