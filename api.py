from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("flight_price_model.pkl")
encoders = joblib.load("encoders.pkl")


class FlightInput(BaseModel):
    day_of_week: str
    origin: str
    destination: str
    airline: str
    class_type: str
    stops: int
    days_to_flight: int
    departure_time: str
    purchase_date: str
    flight_date: str


def preprocess(data: FlightInput):
    df = pd.DataFrame([data.dict()])

    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df["flight_date"] = pd.to_datetime(df["flight_date"])

    df["purchase_month"] = df["purchase_date"].dt.month
    df["flight_month"] = df["flight_date"].dt.month
    df["flight_day"] = df["flight_date"].dt.day
    df["departure_hour"] = pd.to_datetime(df["departure_time"], format="%H:%M").dt.hour

    df = df.drop(columns=["purchase_date", "flight_date", "departure_time"])

    df = df.rename(columns={"class_type": "class"})

    categorical_columns = ["day_of_week", "origin", "destination", "airline", "class"]
    for column in categorical_columns:
        if column in df.columns:
            df[column] = encoders[column].transform(df[column].astype(str))

    return df


@app.post("/predict")
def predict_price(data: FlightInput):
    try:
        x = preprocess(data)
        x = x.reindex(columns=model.feature_names_in_, fill_value=0)
        predicted_price = float(model.predict(x)[0])
        return {"predicted_price": predicted_price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast")
def forecast_price(data: FlightInput):
    try:
        base_input = data.dict()
        future_prices = []

        for i in range(30):
            future_date = pd.to_datetime(base_input["purchase_date"]) + timedelta(days=i)
            future_days_to_flight = (
                pd.to_datetime(base_input["flight_date"]) - future_date
            ).days

            if future_days_to_flight < 0:
                break

            future_input = FlightInput(
                day_of_week=future_date.day_name(),
                origin=base_input["origin"],
                destination=base_input["destination"],
                airline=base_input["airline"],
                class_type=base_input["class_type"],
                stops=base_input["stops"],
                days_to_flight=future_days_to_flight,
                departure_time=base_input["departure_time"],
                purchase_date=str(future_date.date()),
                flight_date=base_input["flight_date"],
            )

            prediction = predict_price(future_input)
            future_prices.append(float(prediction["predicted_price"]))

        if not future_prices:
            raise HTTPException(status_code=400, detail="No forecast could be generated.")

        best_price = float(np.min(future_prices))
        best_day = int(np.argmin(future_prices))

        return {
            "forecast_30_days": future_prices,
            "best_price": best_price,
            "best_day": best_day,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/buy_signal")
def buy_signal(data: FlightInput):
    try:
        today_result = predict_price(data)
        today_price = float(today_result["predicted_price"])

        base_input = data.dict()
        future_prices = []

        for i in range(30):
            future_date = pd.to_datetime(base_input["purchase_date"]) + timedelta(days=i)
            future_days_to_flight = (
                pd.to_datetime(base_input["flight_date"]) - future_date
            ).days

            if future_days_to_flight < 0:
                break

            future_input = FlightInput(
                day_of_week=future_date.day_name(),
                origin=base_input["origin"],
                destination=base_input["destination"],
                airline=base_input["airline"],
                class_type=base_input["class_type"],
                stops=base_input["stops"],
                days_to_flight=future_days_to_flight,
                departure_time=base_input["departure_time"],
                purchase_date=str(future_date.date()),
                flight_date=base_input["flight_date"],
            )

            prediction = predict_price(future_input)
            future_prices.append(float(prediction["predicted_price"]))

        if not future_prices:
            raise HTTPException(status_code=400, detail="No forecast could be generated.")

        min_future_price = float(np.min(future_prices))
        max_future_price = float(np.max(future_prices))
        avg_future_price = float(np.mean(future_prices))
        best_day = int(np.argmin(future_prices))

        if min_future_price < today_price * 0.97:
            signal = "WAIT"
            message = "احتمال کاهش قیمت وجود دارد — فعلاً صبر کنید."
        elif max_future_price > today_price * 1.05:
            signal = "BUY"
            message = "قیمت احتمالاً افزایش می‌یابد — الان بخرید."
        else:
            signal = "HOLD"
            message = "قیمت پایدار است — هنوز بهترین زمان خرید نیست."

        return {
            "today_price": round(today_price, 2),
            "forecast": [round(p, 2) for p in future_prices],
            "signal": signal,
            "message": message,
            "avg_future_price": round(avg_future_price, 2),
            "min_future_price": round(min_future_price, 2),
            "max_future_price": round(max_future_price, 2),
            "best_day": best_day,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
