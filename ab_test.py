"""
A/B Test: Melbourne Housing Price Prediction
Compares two Linear Regression feature sets with statistical significance testing.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from scipy import stats

# Load data
df = pd.read_csv("melb_data.csv")

# VERSION A (Control) – Basic features
features_A = ["Rooms", "Bathroom", "Distance", "Car"]

# VERSION B (Treatment) – Expanded features
features_B = [
    "Rooms", "Bathroom", "Landsize", "BuildingArea", "YearBuilt",
    "Lattitude", "Longtitude", "Distance", "Car"
]

y = df["Price"]


def prepare(X_cols):
    X = df[X_cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X_cols)
    return X_imp


X_A = prepare(features_A)
X_B = prepare(features_B)

# Identical train/test split for fair comparison
X_train_A, X_test_A, y_train, y_test = train_test_split(
    X_A, y, test_size=0.25, random_state=42
)
X_train_B, X_test_B, _, _ = train_test_split(
    X_B, y, test_size=0.25, random_state=42
)

# Train both models
model_A = LinearRegression().fit(X_train_A, y_train)
model_B = LinearRegression().fit(X_train_B, y_train)

# Predictions
pred_A = model_A.predict(X_test_A)
pred_B = model_B.predict(X_test_B)

# Absolute errors
abs_err_A = np.abs(y_test - pred_A)
abs_err_B = np.abs(y_test - pred_B)

mae_A = abs_err_A.mean()
mae_B = abs_err_B.mean()
improvement = (mae_A - mae_B) / mae_A * 100

# Statistical significance: paired t-test
t_stat, p_value = stats.ttest_rel(abs_err_A, abs_err_B)

# 95% Confidence Interval of the difference
diff = abs_err_A - abs_err_B
ci_low, ci_high = stats.t.interval(
    0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff)
)

print("=" * 60)
print("A/B TEST RESULTS – Melbourne Housing Price Prediction")
print("=" * 60)
print()
print("VERSION A (Control)  – Basic features")
print(f"  Features : {features_A}")
print(f"  MAE      : ${mae_A:,.0f}")
print()
print("VERSION B (Treatment) – Expanded features")
print(f"  Features : {features_B}")
print(f"  MAE      : ${mae_B:,.0f}")
print()
print("-" * 60)
print(f"Improvement of B over A : {improvement:.1f}%")
print(f"Absolute difference     : ${mae_A - mae_B:,.0f}")
print()
print("STATISTICAL SIGNIFICANCE")
print(f"  Paired t-statistic    : {t_stat:.3f}")
print(f"  p-value               : {p_value:.6f}")
print(f"  95% CI of difference  : [${ci_low:,.0f}, ${ci_high:,.0f}]")
print()

if p_value < 0.05:
    print("CONCLUSION: Version B is statistically significantly better")
    print("            (p < 0.05). We reject the null hypothesis.")
else:
    print("CONCLUSION: No statistically significant difference detected.")

print("=" * 60)
print()
print("Error distribution summary:")
print(f"  A – median error     : ${np.median(abs_err_A):,.0f}")
print(f"  B – median error     : ${np.median(abs_err_B):,.0f}")
print(f"  A – 75th percentile  : ${np.percentile(abs_err_A, 75):,.0f}")
print(f"  B – 75th percentile  : ${np.percentile(abs_err_B, 75):,.0f}")
print(f"  Sample size (test)   : {len(y_test)}")
