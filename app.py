import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Assuming 'car_details.csv' is your original dataset file
df_original = pd.read_csv('Downloads/CAR DETAILS (4).csv')

# Randomly select 20 data points from the original dataset
df_sample = df_original.sample(n=20, random_state=42)

# Select relevant features and target variable
X_sample = df_sample.drop('selling_price', axis=1)  # Features
y_sample = df_sample['selling_price']  # Target

# Apply the same preprocessing to the sample data
X_sample_preprocessed = pipeline.transform(X_sample)

# Load the saved model
loaded_model = joblib.load('best_model.joblib')

# Make predictions using the loaded model
y_pred_sample = loaded_model.predict(X_sample_preprocessed)

# Compare actual vs predicted values
result_df = pd.DataFrame({'Actual Selling Price': y_sample, 'Predicted Selling Price': y_pred_sample})
print(result_df)
