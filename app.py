from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import traceback

app = Flask(__name__)
CORS(app)

model = joblib.load("flight_price_model.pkl")
encoders = joblib.load("encoders.pkl")

print("model loaded")
print("encoders loaded")


def preprocess_input(data):
    df = pd.DataFrame([data])

    required_fields = [
        "day_of_week", "origin", "destination", "airline",
        "class_type", "stops", "days_to_flight",
        "departure_time", "purchase_date", "flight_date"
    ]

    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Missing fields: {missing_fields}")

    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df["flight_date"] = pd.to_datetime(df["flight_date"])

    df["purchase_month"] = df["purchase_date"].dt.month
    df["flight_month"] = df["flight_date"].dt.month
    df["flight_day"] = df["flight_date"].dt.day

    if "departure_time" in df.columns:
        df["departure_hour"] = pd.to_datetime(df["departure_time"], format="%H:%M", errors="coerce").dt.hour
    else:
        df["departure_hour"] = 0

    df = df.drop(columns=["purchase_date", "flight_date", "departure_time"])
    df = df.rename(columns={"class_type": "class"})

    categorical_columns = ["day_of_week", "origin", "destination", "airline", "class"]
    for column in categorical_columns:
        if column in df.columns:
            df[column] = encoders[column].transform(df[column].astype(str))

    return df


def predict_price_from_payload(payload):
    x = preprocess_input(payload)
    x = x.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(x)[0]
    return float(prediction)


@app.route("/")
def home():
    return "Flight Price API is running."


@app.route("/buy_signal", methods=["POST"])
def buy_signal():
    try:
        payload = request.get_json(force=True)

        today_price = predict_price_from_payload(payload)

        base_purchase_date = pd.to_datetime(payload["purchase_date"])
        base_flight_date = pd.to_datetime(payload["flight_date"])

        forecast = []

        for i in range(30):
            future_purchase_date = base_purchase_date + pd.Timedelta(days=i)
            future_days_to_flight = (base_flight_date - future_purchase_date).days

            if future_days_to_flight < 0:
                break

            future_payload = payload.copy()
            future_payload["purchase_date"] = str(future_purchase_date.date())
            future_payload["days_to_flight"] = future_days_to_flight
            future_payload["day_of_week"] = future_purchase_date.day_name()

            future_price = predict_price_from_payload(future_payload)
            forecast.append(future_price)

        if not forecast:
            return jsonify({
                "today_price": round(today_price, 2),
                "forecast": [],
                "signal": "HOLD",
                "message": "Forecast could not be generated."
            }), 200

        min_future_price = min(forecast)
        max_future_price = max(forecast)
        avg_future_price = sum(forecast) / len(forecast)

        if min_future_price < today_price * 0.97:
            signal = "WAIT"
            message = "احتمال کاهش قیمت وجود دارد — فعلاً صبر کنید."
        elif max_future_price > today_price * 1.05:
            signal = "BUY"
            message = "قیمت احتمالاً افزایش می‌یابد — الان بخرید."
        else:
            signal = "HOLD"
            message = "قیمت پایدار است — هنوز بهترین زمان خرید نیست."

        return jsonify({
            "today_price": round(today_price, 2),
            "forecast": [round(price, 2) for price in forecast],
            "signal": signal,
            "message": message,
            "avg_future_price": round(avg_future_price, 2),
            "min_future_price": round(min_future_price, 2),
            "max_future_price": round(max_future_price, 2),
        }), 200

    except Exception as e:
        print("ERROR IN /buy_signal")
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
