import pickle
import numpy as np
import pandas as pd

with open('linear_regression_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

def predict_col3():
    try:
        col1 = float(input("Enter the value for col1: "))
        col2 = float(input("Enter the value for col2: "))

        input_data = pd.DataFrame([[col1, col2]], columns=['col1', 'col2'])

        predicted_col3 = loaded_model.predict(input_data)[0]

        actual_col3 = col1 - col2

        if actual_col3 != 0:
            accuracy = (1 - abs(predicted_col3 - actual_col3) / abs(actual_col3)) * 100
        else:
            accuracy = 100.0

        print(f"Predicted value for col3: {predicted_col3}")
        print(f"Actual value for col3: {actual_col3:.2f}")
        print(f"Accuracy: {accuracy:.2f}% approximately")


    except ValueError:
        print("Invalid input. Please enter numeric values for col1 and col2.")
    except Exception as e:
        print(f"An error occurred: {e}")

predict_col3()
