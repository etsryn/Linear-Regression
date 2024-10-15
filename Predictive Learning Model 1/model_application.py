import pickle
import numpy as np
import pandas as pd  # Import pandas

# Load the model from the .pkl file
with open('linear_regression_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Function to predict col3 based on user input for col1 and col2
def predict_col3():
    try:
        # Get user input for col1 and col2
        col1 = float(input("Enter the value for col1: "))
        col2 = float(input("Enter the value for col2: "))

        # Create a DataFrame for prediction
        input_data = pd.DataFrame([[col1, col2]], columns=['col1', 'col2'])

        # Make the prediction using the loaded model
        predicted_col3 = loaded_model.predict(input_data)[0]

        # Calculate the actual value of col3
        actual_col3 = col1 + col2

        # Calculate accuracy percentage
        if actual_col3 != 0:
            accuracy = (1 - abs(predicted_col3 - actual_col3) / abs(actual_col3)) * 100
        else:
            accuracy = 100.0  # If actual_col3 is 0, we can say the accuracy is perfect

        # Print the predicted value and accuracy
        print(f"Predicted value for col3: {predicted_col3}")  # Rounded to two decimal places
        print(f"Actual value for col3: {actual_col3:.2f}")
        print(f"Accuracy: {accuracy:.2f}% approximately")


    except ValueError:
        print("Invalid input. Please enter numeric values for col1 and col2.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to predict col3
predict_col3()
