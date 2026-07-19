"""
CodeAlpha - Data Science Internship
Task 3: Car Price Prediction with Machine Learning
-----------------------------------------------------
Predict a used car's selling price from features such as year, present
price, kilometers driven, fuel type, seller type, transmission and owner.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------
df = pd.read_csv("car_data.csv")
print("Dataset shape:", df.shape)
print(df.head())
print(df.info())
print(df.isna().sum())

# ---------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------
current_year = 2020  # dataset appears to be up to ~2018-2020
df["Car_Age"] = current_year - df["Year"]
df = df.drop(columns=["Year", "Car_Name"])  # Car_Name has too many unique values

# ---------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(df["Selling_Price"], kde=True, color="steelblue")
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price (Lakhs)")
plt.tight_layout()
plt.savefig("car_price_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("car_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.boxplot(x="Fuel_Type", y="Selling_Price", data=df, ax=axes[0])
sns.boxplot(x="Selling_type", y="Selling_Price", data=df, ax=axes[1])
sns.boxplot(x="Transmission", y="Selling_Price", data=df, ax=axes[2])
axes[0].set_title("Selling Price by Fuel Type")
axes[1].set_title("Selling Price by Seller Type")
axes[2].set_title("Selling Price by Transmission")
plt.tight_layout()
plt.savefig("car_price_by_category.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 4. Encode categorical variables
# ---------------------------------------------------------
df_encoded = pd.get_dummies(df, columns=["Fuel_Type", "Selling_type", "Transmission"], drop_first=True)

X = df_encoded.drop(columns=["Selling_Price"])
y = df_encoded["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 5. Train and compare models
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}
    print(f"\n{name}:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

# ---------------------------------------------------------
# 6. Best model - actual vs predicted plot
# ---------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_name]
best_preds = best_model.predict(X_test)
print(f"\nBest model: {best_name} (R2={results[best_name]['R2']:.4f})")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.6, color="darkgreen")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title(f"Actual vs Predicted Selling Price ({best_name})")
plt.tight_layout()
plt.savefig("car_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 7. Feature importance (Random Forest)
# ---------------------------------------------------------
rf = models["Random Forest Regressor"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(7, 4))
importances.plot(kind="bar", color="darkorange")
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("car_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n=== Model Comparison Summary ===")
for name, m in results.items():
    print(f"{name}: R2={m['R2']:.4f}, MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}")
