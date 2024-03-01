import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Downloads/CAR DETAILS (4).csv')
df.head()

# Check for missing values
df.isnull().sum()

# Drop rows with missing values or impute them based on your strategy
df.dropna()

# Remove duplicate rows
df.drop_duplicates()

# Convert 'year' to datetime if it's not already
df['year'] = pd.to_datetime(df['year'], format='%Y')
df.head()

from sklearn.preprocessing import StandardScaler

# Scale numerical features
scaler = StandardScaler()
df[['selling_price', 'km_driven']] = scaler.fit_transform(df[['selling_price', 'km_driven']])
df.head()

# Example: Extracting the age of the car from the 'year' feature
current_year = pd.to_datetime('now').year
df['car_age'] = current_year - df['year'].dt.year
df.head()

df = pd.read_csv('Downloads/CAR DETAILS (4).csv')
df.head()

#Distribution of Car Prices
plt.figure(figsize=(12, 6))
sns.histplot(df['selling_price'], bins=20, kde=True)
plt.title('Distribution of Car Prices')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()

# Count the occurrences of each unique year
year_counts = df['transmission'].value_counts()

# Create a pie chart
plt.figure(figsize=(10, 8))
plt.pie(year_counts, labels=year_counts.index, autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title('Distribution of Cars by Year')
plt.show()

#Boxplot of Car Prices based on Fuel Type
plt.figure(figsize=(12, 6))
sns.boxplot(x='fuel', y='selling_price', data=df)
plt.title('Car Prices based on Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Selling Price')
plt.show()

# Scatter plot of Selling Price vs. Kilometers Driven
plt.figure(figsize=(12, 6))
sns.scatterplot(x='km_driven', y='selling_price', data=df)
plt.title('Selling Price vs. Kilometers Driven')
plt.xlabel('Kilometers Driven')
plt.ylabel('Selling Price')
plt.show()

# Pairplot to visualize relationships between numerical features
sns.pairplot(df[['year', 'selling_price', 'km_driven']], height=2.5)
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv('Downloads/CAR DETAILS (4).csv')

# Select relevant features and target variable
X = df.drop('selling_price', axis=1)  # Features
y = df['selling_price']  # Target

# Define categorical and numerical features
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
numerical_features = ['year', 'km_driven']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create a pipeline for preprocessing
pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

# Preprocess the data
X_preprocessed = pipeline.fit_transform(X)

# Get feature names after one-hot encoding
feature_names = numerical_features + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))

# Display the preprocessed features
print(pd.DataFrame(X_preprocessed, columns=feature_names))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('Downloads/CAR DETAILS (4).csv')

# Select relevant features and target variable
X = df.drop('selling_price', axis=1)  # Features
y = df['selling_price']  # Target

# Define categorical and numerical features
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
numerical_features = ['year', 'km_driven']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create a pipeline for preprocessing
pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

# Preprocess the data
X_preprocessed = pipeline.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"----- {name} -----")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}\n")

import joblib
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('Downloads/CAR DETAILS (4).csv')

# Select relevant features and target variable
X = df.drop('selling_price', axis=1)  # Features
y = df['selling_price']  # Target

# Define categorical and numerical features
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
numerical_features = ['year', 'km_driven']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create a pipeline for preprocessing
pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

# Preprocess the data
X_preprocessed = pipeline.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Instantiate and train the best model (RandomForestRegressor in this case)
best_model = RandomForestRegressor()
best_model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(best_model, 'best_model.joblib')

# Later, when you want to use the model
# Load the model from the file
loaded_model = joblib.load('best_model.joblib')

# Make predictions using the loaded model
y_pred_loaded = loaded_model.predict(X_test)
