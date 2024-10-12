import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# Load the data from the Excel file
file_path = 'C:/Users/User/Documents/Python/economic indicators analysis OECD/OECD_Economic_Metrics.xlsx'
data = pd.read_excel(file_path)

# Include the 2023 GDP growth rate as a feature (do not drop it)
features = data.drop(columns=['Country'])  # Only drop non-feature columns like 'Country'

# Specify the target column for prediction (you can set it to the 2024 GDP growth rate once it's available)
# Here we're assuming 'target_column' will be the column you want to predict, for example, the future GDP growth
# If you have historical GDP growth data for multiple years, you may want to create a new column for 2024 predictions
target_column = 'WorldBank GDP Growth Rate 2024'  # Replace with actual target column name if it exists
target = data[target_column]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Define a simple neural network model in TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=10, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Predict GDP growth rates for all countries in the dataset
all_predictions = model.predict(features_scaled)

# Add the predictions as a new column to the original data
data['Predicted GDP Growth Rate 2024'] = all_predictions

# Display the predictions along with country names
predicted_data = data[['Country', 'Predicted GDP Growth Rate 2024']]
print(predicted_data)
