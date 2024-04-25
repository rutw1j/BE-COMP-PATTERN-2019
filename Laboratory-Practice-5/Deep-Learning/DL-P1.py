# Improting Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error


# Loading the Dataset
boston_data = pd.read_csv('https://raw.githubusercontent.com/rutw1j/BE-COMP-PATTERN-2019/main/Laboratory-Practice-5/Datasets/boston-housing.csv', sep='\\s+', skiprows=22, header=None)
boston_data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


# Splitting data into features and target
X = boston_data.iloc[:, :-1]
y = boston_data.iloc[:, -1]


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Building Neural Netwrok Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0, callbacks=[early_stopping])


# Evaluate the model
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"{'Test Loss (MSE): '}{test_loss}")


# Make predictions
predictions = model.predict(X_test)


# Calculate various evaluation metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
explained_variance = explained_variance_score(y_test, predictions)
median_absolute = median_absolute_error(y_test, predictions)


# Display Evaluation Results
print('\n\nEVALUATION RESULTS\n')
print(f"{'Mean Absolute Error (MAE):':<34}{mae}")
print(f"{'Mean Squared Error (MSE):':<34}{mse}")
print(f"{'Root Mean Squared Error (RMSE):':<34}{rmse}")
print(f"{'R-squared (R2):':<34}{r2}")
print(f"{'Explained Variance Score:':<34}{explained_variance}")
print(f"{'Median Absolute Error:':<34}{median_absolute}\n\n")


# Visualize training history
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.show()