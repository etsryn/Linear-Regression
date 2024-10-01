import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

train_data = pd.read_excel('train_data.xlsx')

X_train = train_data[['col1', 'col2']]
y_train = train_data['col3']

model = LinearRegression()
model.fit(X_train, y_train)

test_data = pd.read_excel('test_data.xlsx')

predictions = model.predict(test_data)

for i in range(len(test_data)):    
    print(f"Input: {test_data.iloc[i].tolist()}, Predicted col3: {predictions[i]}")

mse = mean_squared_error(y_train, model.predict(X_train))
print(f"Mean Squared Error (MSE) on Training Data: {mse}")

r2 = r2_score(y_train, model.predict(X_train))
print(f"R-squared Score (R²) on Training Data: {r2}")