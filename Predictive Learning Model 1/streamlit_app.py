import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the model from the .pkl file
with open('linear_regression_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Streamlit app
st.set_page_config(page_title="Interactive Linear Regression App", page_icon="📊", layout="wide")

# App title and description
st.title("🎉 Interactive Linear Regression Predictor")
st.markdown("""
Welcome to the **Interactive Linear Regression Prediction App**! 🎯 

📥 **Input your values** for `col1` and `col2` to predict `col3`, visualize the results, and gain detailed insights. 
Enjoy an engaging and easy-to-use interface! 🚀
""")

# Input fields on the main page
st.subheader("🔢 Input Features")
col1 = st.number_input("Enter the value for col1:", value=0.0, format="%.2f", key="col1", help="Input a numeric value for the first feature.", label_visibility="collapsed", step=0.1, min_value=-1000.0, max_value=1000.0)
col2 = st.number_input("Enter the value for col2:", value=0.0, format="%.2f", key="col2", help="Input a numeric value for the second feature.", label_visibility="collapsed", step=0.1, min_value=-1000.0, max_value=1000.0)

# Prediction section
st.subheader("📊 Prediction Results")
if st.button("🔮 Predict col3"):
    try:
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

        # Calculate the error (difference)
        error_difference = abs(predicted_col3 - actual_col3)

        # Display the results
        st.success(f"🎯 **Predicted col3:** {predicted_col3:.2f}")
        st.info(f"✅ **Actual col3:** {actual_col3:.2f}")
        st.write(f"### 📈 **Accuracy:** {accuracy:.2f}% approximately")

        # Add an expander for detailed insights
        with st.expander("🔍 Detailed Insights"):
            st.markdown(f"""
            - **Input Features:**
                - `col1`: {col1:.2f}
                - `col2`: {col2:.2f}
            - **Prediction Details:**
                - `Predicted col3`: {predicted_col3:.2f}
                - `Actual col3`: {actual_col3:.2f}
                - **Error (Difference)**: {error_difference:.2f}
                - **Accuracy**: {accuracy:.2f}%
            """)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

# Add a feature to visualize predictions
st.subheader("📈 Visualize Predictions")
if st.button("📊 Generate Visualization"):
    try:
        # Generate a range of values for col1 and col2
        col1_range = np.linspace(col1 - 5, col1 + 5, 100)
        predicted_values = [loaded_model.predict(pd.DataFrame([[c1, col2]], columns=['col1', 'col2']))[0] for c1 in col1_range]

        # Plot the results
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(col1_range, predicted_values, label="Predicted col3", color="blue", linewidth=2)
        ax.axhline(y=col1 + col2, color="green", linestyle="--", label="Actual col3")
        ax.set_title("Prediction Visualization", fontsize=12)
        ax.set_xlabel("col1 Values", fontsize=10)
        ax.set_ylabel("col3 Prediction", fontsize=10)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ An error occurred while generating the visualization: {e}")

# Footer
st.write("---")
st.markdown("""**Developed with ❤️ by Rayyan Ashraf** Feel free to reach out for feedback or suggestions!""")
