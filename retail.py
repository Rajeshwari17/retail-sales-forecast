import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# ── Step 1: Load ──────────────────────────────────────────
df = pd.read_excel("online_retail.xlsx")
print("Shape:", df.shape)
print(df.head())

# ── Step 2: Clean ─────────────────────────────────────────
# Remove cancellations and bad rows
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df = df.dropna(subset=["CustomerID"])

# Create a revenue column
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# ── Step 3: Build monthly sales ───────────────────────────
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Month"] = df["InvoiceDate"].dt.to_period("M")
monthly = df.groupby("Month")["Revenue"].sum().reset_index()
monthly["Month"] = monthly["Month"].dt.to_timestamp()
monthly = monthly.sort_values("Month")
print("\nMonthly Sales:")
print(monthly)

# ── Step 4: Engineer features ─────────────────────────────
monthly["MonthNum"]   = range(len(monthly))
monthly["Lag1"]       = monthly["Revenue"].shift(1)  # last month's sales
monthly["Lag2"]       = monthly["Revenue"].shift(2)  # 2 months ago
monthly["RollingAvg"] = monthly["Revenue"].rolling(3).mean()
monthly = monthly.dropna()

X = monthly[["MonthNum", "Lag1", "Lag2", "RollingAvg"]]
y = monthly["Revenue"]

# ── Step 5: Train/test split (last 3 months = test) ───────
X_train, X_test = X.iloc[:-3], X.iloc[-3:]
y_train, y_test = y.iloc[:-3], y.iloc[-3:]

model = GradientBoostingRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Absolute Error: £{mae:,.2f}")

# ── Step 6: Plot actual vs predicted ──────────────────────
plt.figure(figsize=(10, 5))
plt.plot(monthly["Month"], monthly["Revenue"], label="Actual Sales", marker="o")
plt.plot(monthly["Month"].iloc[-3:], y_pred, label="Predicted Sales",
         marker="o", linestyle="--", color="orange")
plt.title("Retail Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Revenue (£)")
plt.legend()
plt.tight_layout()
plt.savefig("sales_forecast.png")
print("\nChart saved as sales_forecast.png on your Desktop!")
plt.show()