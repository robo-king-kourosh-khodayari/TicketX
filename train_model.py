import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

# خواندن داده‌ها
df = pd.read_csv("featured_data.csv")

# تمیز کردن ستون‌ها (حذف فاصله‌های اضافه از اطراف متن‌ها)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# تبدیل به تاریخ با مدیریت بهتر فرمت‌ها
df["purchase_date"] = pd.to_datetime(df["purchase_date"], format='mixed')
df["flight_date"] = pd.to_datetime(df["flight_date"], format='mixed')

# مهندسی ویژگی‌ها
df["purchase_month"] = df["purchase_date"].dt.month
df["flight_month"] = df["flight_date"].dt.month
df["flight_day"] = df["flight_date"].dt.day
df["departure_hour"] = df["departure_time"].str.split(":").str[0].astype(int)

cat_cols = ["day_of_week", "origin", "destination", "airline", "class"]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

features = [
    "day_of_week",
    "origin",
    "destination",
    "airline",
    "class",
    "stops",
    "days_to_flight",
    "departure_hour",
    "purchase_month",
    "flight_month",
    "flight_day"
]

X = df[features]
y = df["ticket_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=-1,
    verbose=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

# ذخیره مدل و انکودرها
joblib.dump(model, "flight_price_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print(f"Training completed successfully.")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
