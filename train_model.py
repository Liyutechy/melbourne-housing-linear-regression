"""
Melbourne Housing Price Prediction - Linear Regression
Trains a simple Linear Regression model on the Melbourne housing dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

# Load data
df = pd.read_csv("melb_data.csv")

# Features used for prediction
FEATURES = [
    "Rooms",
    "Bathroom",
    "Landsize",
    "BuildingArea",
    "YearBuilt",
    "Lattitude",
    "Longtitude",
    "Distance",
    "Car",
]

X = df[FEATURES]
y = df["Price"]

# Handle missing values with median imputation
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions & metrics
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("Melbourne Housing - Linear Regression Results")
print("=" * 50)
print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"MAE              : ${mae:,.0f}")
print(f"RMSE             : ${rmse:,.0f}")
print(f"R² Score         : {r2:.4f}")
print()
print("Feature Coefficients:")
for feature, coef in zip(FEATURES, model.coef_):
    print(f"  {feature:15s}: {coef:>12.2f}")
print(f"  {'Intercept':15s}: {model.intercept_:>12.2f}")
print("=" * 50)

# Save model and imputer
joblib.dump(model, "linear_regression_model.joblib")
joblib.dump(imputer, "imputer.joblib")
print("\nModel saved to: linear_regression_model.joblib")
print("Imputer saved to: imputer.joblib")
