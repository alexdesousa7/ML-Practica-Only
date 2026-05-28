import pandas as pd
from src.config import PROCESSED_DATA_PATH

def load_data():
  return pd.read_csv(PROCESSED_DATA_PATH)
