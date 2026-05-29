from fastapi import FastAPI
from pydantic import BaseModel, model_validator
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="API: Servicio para predecir la probabilidad de cancelación de una reserva hotelera")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
model = joblib.load(MODEL_PATH)

class RawReservationInput(BaseModel):
  hotel: str
  arrival_date_year: int
  arrival_date_month: str
  arrival_date_week_number: int
  arrival_date_day_of_month: int
  stays_in_weekend_nights: int
  stays_in_week_nights: int
  adults: int
  children: float
  babies: int
  meal: str
  country: str
  market_segment: str
  distribution_channel: str
  is_repeated_guest: int
  previous_cancellations: int
  previous_bookings_not_canceled: int
  booking_changes: int
  deposit_type: str
  days_in_waiting_list: int
  customer_type: str
  adr: float
  required_car_parking_spaces: int
  total_of_special_requests: int
  lead_time: int
  reserved_room_type: str
  assigned_room_type: str
  agent: str | None = None


class PredictionOutput(BaseModel):
  prediction: int
  probability: float
  label: str


def engineer_features(data: RawReservationInput) -> pd.DataFrame:
  row = data.model_dump()

  row["room_mismatch"] = int(row["reserved_room_type"] != row["assigned_room_type"])
  row["lead_time_log"] = float(np.log1p(row["lead_time"]))
  row["has_agent"] = int(row["agent"] is not None)

  for col in ("lead_time", "reserved_room_type", "assigned_room_type", "agent"):
    row.pop(col)

    return pd.DataFrame([row])


@app.post("/predict", response_model=PredictionOutput)
def predict(data: RawReservationInput):
  df = engineer_features(data)
  prediction = int(model.predict(df)[0])
  probability = float(model.predict_proba(df)[0][1])
  label = "Will cancel" if prediction == 1 else "Will NOT cancel"
  return PredictionOutput(prediction=prediction, probability=probability, label=label)
