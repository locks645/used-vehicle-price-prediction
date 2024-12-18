import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

vehicles = 'datasets/vehicles-2.csv'

vehicles_data = pd.read_csv(vehicles, delimiter=',')

relevant_columns = ['price', 'year', 'manufacturer', 'condition', 'fuel', 'odometer', 
                    'title_status', 'transmission', 'drive', 'type', 'paint_color']

# Define categorical and numerical columns
categorical_cols = ['manufacturer', 'condition', 'fuel', 'title_status', 
                    'transmission', 'drive', 'type', 'paint_color']
numerical_cols = ['year', 'odometer']

# Function to compute co-occurrence matrix
def compute_categorical_matrix(data, categorical_cols):
    matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
    for col1 in categorical_cols:
        for col2 in categorical_cols:
            if col1 == col2:
                matrix.loc[col1, col2] = len(data[col1].unique())  # Number of unique values in the column
            else:
                matrix.loc[col1, col2] = len(data.groupby([col1, col2]).size())  # Co-occurrence count
    return matrix.astype(float)

def compute_categorical_correlation_matrix(data, categorical_cols):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_data = encoder.fit_transform(data[categorical_cols])
    encoded_columns = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
    return encoded_df.corr()

cleaned_data = vehicles_data[relevant_columns].dropna()
cleaned_data = cleaned_data[cleaned_data['price'].between(1000, 100000)]
cleaned_data = cleaned_data[cleaned_data['odometer'].between(0, 500000)]
cleaned_data = cleaned_data[cleaned_data['year'].between(1980, 2024)]

numerical_cols_display = ['price', 'year','odometer']
for col in numerical_cols_display:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vehicles_data[col].hist(bins=30, ax=axes[0])
    cleaned_data[col].hist(bins=30, ax=axes[1])
    axes[0].set_title(f'{col} Before Cleaning')
    axes[1].set_title(f'{col} After Cleaning')
    plt.tight_layout()
    plt.show()

for col in numerical_cols_display:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=vehicles_data[col], ax=axes[0]).set_title(f'{col} Before Cleaning')
    sns.boxplot(y=cleaned_data[col], ax=axes[1]).set_title(f'{col} After Cleaning')
    plt.tight_layout()
    plt.show()

# Compute correlation matrix BEFORE cleaning
correlation_matrix_before = compute_categorical_correlation_matrix(vehicles_data, categorical_cols)

# Compute correlation matrix AFTER cleaning
correlation_matrix_after = compute_categorical_correlation_matrix(cleaned_data, categorical_cols)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Heatmap for correlation matrix BEFORE cleaning
sns.heatmap(correlation_matrix_before, ax=axes[0], cmap="coolwarm", annot=False)
axes[0].set_title("Correlation Matrix (Before Cleaning)")

# Heatmap for correlation matrix AFTER cleaning
sns.heatmap(correlation_matrix_after, ax=axes[1], cmap="coolwarm", annot=False)
axes[1].set_title("Correlation Matrix (After Cleaning)")

plt.tight_layout()
plt.show()

# Separate features and target
X = cleaned_data.drop(columns=['price'])
y = cleaned_data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Define models
models = {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
    'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor(random_state=42))]),
    'Neural Network': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', MLPRegressor(random_state=42, max_iter=500))])
}

dt_param_grid = {
    'regressor__max_depth': [5, 10, 15, 20, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# MLP Regressor parameters
mlp_param_grid = {
    'regressor__hidden_layer_sizes': [(30,), (50,), (30, 30), (50, 50)],
    'regressor__alpha': [0.0001, 0.00001],
    'regressor__learning_rate_init': [0.001, 0.01],
    'regressor__max_iter': [1000]  # Increased maximum iterations
}

# GridSearchCV for Decision Tree
dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor(random_state=42))])
dt_search = GridSearchCV(dt_pipeline, dt_param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
dt_search.fit(X_train, y_train)
best_dt_model = dt_search.best_estimator_

# GridSearchCV for MLP Regressor
mlp_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', MLPRegressor(random_state=42, max_iter=1000, early_stopping=True)
)])
mlp_search = GridSearchCV(mlp_pipeline, mlp_param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
mlp_search.fit(X_train, y_train)
best_mlp_model = mlp_search.best_estimator_

# Replace original models with optimized ones
models['Decision Tree'] = best_dt_model
models['Neural Network'] = best_mlp_model

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


error_data2 = {
    'Linear Regression': comparison_df['Actual'] - comparison_df['Linear Regression'],
    'Decision Tree': comparison_df['Actual'] - comparison_df['Decision Tree'],
    'Neural Network': comparison_df['Actual'] - comparison_df['Neural Network'],
}

total_data_points = cleaned_data.shape[0]
print(f"Total data points: {total_data_points}")

# Calculate outliers using IQR
print("\nOutlier Counts:")
for model, errors in error_data.items():
    Q1 = np.percentile(errors, 25)
    Q3 = np.percentile(errors, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = errors[(errors < lower_bound) | (errors > upper_bound)]
    print(f"{model}: {len(outliers)} outliers")

# Calculate outliers using IQR
print("\nOutlier Counts:")
for model, errors in error_data2.items():
    Q1 = np.percentile(errors, 25)
    Q3 = np.percentile(errors, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = errors[(errors < lower_bound) | (errors > upper_bound)]
    print(f"{model}: {len(outliers)} outliers")

# Print statistics for each model
print("\nKey Statistics for Absolute Errors:")
for model, errors in error_data.items():
    print(f"\n{model}:")
    print(f"Mean: {np.mean(errors):.2f}")
    print(f"Q1: {np.percentile(errors, 25):.2f}")
    print(f"Median (Q2): {np.percentile(errors, 50):.2f}")
    print(f"Q3: {np.percentile(errors, 75):.2f}")

# Print statistics for each model
print("\nKey Statistics for Not Absolute Errors:")
for model, errors in error_data2.items():
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

# Create histograms for error distributions
plt.figure(figsize=(12, 8))
for model, errors in error_data.items():
    plt.hist(errors, bins=50, alpha=0.6, label=model)
plt.title('Histogram of Prediction Errors by Model')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Create a boxplot to visualize the differences in predictions
plt.figure(figsize=(12, 6))
plt.boxplot([
    comparison_df['Actual'] - comparison_df['Linear Regression'],
    comparison_df['Actual'] - comparison_df['Decision Tree'],
    comparison_df['Actual'] - comparison_df['Neural Network']
], labels=['Linear Regression', 'Decision Tree', 'Neural Network'])
plt.title('Prediction Error Boxplot by Model')
plt.ylabel('Error')
plt.show()


# Create histograms for error distributions
plt.figure(figsize=(12, 8))
for model, errors in error_data2.items():
    plt.hist(errors, bins=50, alpha=0.6, label=model)
plt.title('Histogram of Prediction Errors by Model')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot MSE scores
plt.figure(figsize=(8, 6))
plt.bar(mse_scores.keys(), mse_scores.values())
plt.title('Mean Squared Error Comparison')
plt.ylabel('MSE')
plt.show()
