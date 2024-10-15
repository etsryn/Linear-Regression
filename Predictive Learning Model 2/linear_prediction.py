import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
# To Save the trained model
# import pickle

train_data = pd.read_csv('train_data.csv')

X_train = train_data[['col1', 'col2']]
y_train = train_data['col3']

model = LinearRegression()
model.fit(X_train, y_train)

test_data = pd.read_csv('test_data.csv')

predictions = model.predict(test_data)

for i in range(len(test_data)):    
    print(f"Input: {test_data.iloc[i].tolist()}, Predicted col3: {predictions[i]}")

mse = root_mean_squared_error(y_train, model.predict(X_train))
print(f"Mean Squared Error (MSE) on Training Data: {mse}")

r2 = r2_score(y_train, model.predict(X_train))
print(f"R-squared Score (R²) on Training Data: {r2}")

# To Save the trained model
# with open('linear_regression_model.pkl', 'wb') as model_file:
    # pickle.dump(model, model_file)
