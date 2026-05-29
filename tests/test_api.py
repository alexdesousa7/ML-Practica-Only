import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

WILL_CANCEL = {
  "hotel": "City Hotel",
  "arrival_date_year": 2017,
  "arrival_date_month": "March",
  "arrival_date_week_number": 13,
  "arrival_date_day_of_month": 30,
  "stays_in_weekend_nights": 0,
  "stays_in_week_nights": 2,
  "adults": 2,
  "children": 0.0,
  "babies": 0,
  "meal": "BB",
  "country": "PRT",
  "market_segment": "Groups",
  "distribution_channel": "TA/TO",
  "is_repeated_guest": 0,
  "previous_cancellations": 0,
  "previous_bookings_not_canceled": 0,
  "booking_changes": 0,
  "deposit_type": "Non Refund",
  "days_in_waiting_list": 0,
  "customer_type": "Transient",
  "adr": 62.0,
  "required_car_parking_spaces": 0,
  "total_of_special_requests": 0,
  "lead_time": 629,
  "reserved_room_type": "A",
  "assigned_room_type": "A",
  "agent": "1",
}

WILL_NOT_CANCEL = {
  "hotel": "Resort Hotel",
  "arrival_date_year": 2017,
  "arrival_date_month": "May",
  "arrival_date_week_number": 21,
  "arrival_date_day_of_month": 24,
  "stays_in_weekend_nights": 0,
  "stays_in_week_nights": 2,
  "adults": 2,
  "children": 0.0,
  "babies": 1,
  "meal": "BB",
  "country": "PRT",
  "market_segment": "Direct",
  "distribution_channel": "Direct",
  "is_repeated_guest": 0,
  "previous_cancellations": 0,
  "previous_bookings_not_canceled": 0,
  "booking_changes": 0,
  "deposit_type": "No Deposit",
  "days_in_waiting_list": 0,
  "customer_type": "Transient",
  "adr": 100.0,
  "required_car_parking_spaces": 0,
  "total_of_special_requests": 2,
  "lead_time": 0,
  "reserved_room_type": "A",
  "assigned_room_type": "B",
  "agent": "250",
}


def test_predict_will_cancel():
  response = client.post("/predict", json=WILL_CANCEL)
  assert response.status_code == 200
  body = response.json()
  assert body["prediction"] == 1
  assert body["label"] == "Will cancel"
  assert body["probability"] > 0.5


def test_predict_will_not_cancel():
  response = client.post("/predict", json=WILL_NOT_CANCEL)
  assert response.status_code == 200
  body = response.json()
  assert body["prediction"] == 0
  assert body["label"] == "Will NOT cancel"
  assert body["probability"] < 0.5


def test_response_schema():
  response = client.post("/predict", json=WILL_NOT_CANCEL)
  body = response.json()
  assert "prediction" in body
  assert "probability" in body
  assert "label" in body
  assert isinstance(body["prediction"], int)
  assert isinstance(body["probability"], float)
  assert isinstance(body["label"], str)


def test_missing_required_fields_returns_422():
  response = client.post("/predict", json={"hotel": "City Hotel"})
  assert response.status_code == 422
