import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


#Wine CSV paths in my directory
vehicles = 'datasets/vehicles.csv'

vehicles_data = pd.read_csv(vehicles, delimiter=',')

relevant_columns = ['price', 'year', 'manufacturer', 'condition', 'fuel', 'odometer', 
                    'title_status', 'transmission', 'drive', 'type', 'paint_color']
cleaned_data = vehicles_data[relevant_columns].dropna()
cleaned_data = cleaned_data[cleaned_data['price'].between(500, 100000)]

cleaned_data = vehicles_data[relevant_columns].dropna()
cleaned_data = cleaned_data[cleaned_data['price'].between(500, 100000)]

# Separate features and target
X = cleaned_data.drop(columns=['price'])
y = cleaned_data['price']

# Define categorical and numerical columns
categorical_cols = ['manufacturer', 'condition', 'fuel', 'title_status', 
                    'transmission', 'drive', 'type', 'paint_color']
numerical_cols = ['year', 'odometer']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
    'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor(random_state=42))]),
    'Neural Network': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', MLPRegressor(random_state=42, max_iter=500))])
}

# Train models and predict on the test set
predictions = {}
mse_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    mse_scores[name] = mean_squared_error(y_test, y_pred)

# Display MSE scores
print("MSE Scores:", mse_scores)

# Create a DataFrame for actual and predicted values
comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Linear Regression': predictions['Linear Regression'],
    'Decision Tree': predictions['Decision Tree'],
    'Neural Network': predictions['Neural Network']
})

# Calculate absolute errors
error_data = {
    'Linear Regression': abs(comparison_df['Actual'] - comparison_df['Linear Regression']),
    'Decision Tree': abs(comparison_df['Actual'] - comparison_df['Decision Tree']),
    'Neural Network': abs(comparison_df['Actual'] - comparison_df['Neural Network']),
}

# Print statistics for each model
print("\nKey Statistics for Absolute Errors:")
for model, errors in error_data.items():
    print(f"\n{model}:")
    print(f"Mean: {np.mean(errors):.2f}")
    print(f"Q1: {np.percentile(errors, 25):.2f}")
    print(f"Median (Q2): {np.percentile(errors, 50):.2f}")
    print(f"Q3: {np.percentile(errors, 75):.2f}")

# Melt the DataFrame for boxplot visualization
melted_df = comparison_df.melt(id_vars='Actual', var_name='Model', value_name='Predicted')

# Create a boxplot to visualize the differences in predictions
plt.figure(figsize=(12, 6))
plt.boxplot([
    abs(comparison_df['Actual'] - comparison_df['Linear Regression']),
    abs(comparison_df['Actual'] - comparison_df['Decision Tree']),
    abs(comparison_df['Actual'] - comparison_df['Neural Network'])
], labels=['Linear Regression', 'Decision Tree', 'Neural Network'])
plt.title('Prediction Error Boxplot by Model')
plt.ylabel('Absolute Error')
plt.show()

# Plot MSE scores
plt.figure(figsize=(8, 6))
plt.bar(mse_scores.keys(), mse_scores.values())
plt.title('Mean Squared Error Comparison')
plt.ylabel('MSE')
plt.show()

# Print explanations for MSE scores
print("\nMSE Score Explanations:")
for model, mse in mse_scores.items():
    print(f"{model}: The Mean Squared Error is {mse:.2f}. This measures the average squared difference between actual and predicted values. Lower values indicate better predictions.")