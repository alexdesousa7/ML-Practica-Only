from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.config import DATA_PATH, PROCESSED_DATA_PATH

from src.cleanup_dataset import CleanUpDataset

import numpy as np

def build_preprocessing_pipeline():
  # 1. Elegir  la columna objetivo si está presente (No haremos nada particularmente con ella en esta face)
  target_col = "is_canceled"
  
  # 2. Estas dos columnas parecen contener información directamente relacionada con el target_col, consideramos debemos excluirlas.
  leakage_cols = ["reservation_status", "reservation_status_date"]

  # 3. La columna Company parece ser nulla en más de un 90% de los casos. No creemos que aporte demasiado valor a nuestra predicción, así que la eliminaremos. Por otro lado, la columna agent
  unnecessary_cols = ["company"]
  
  # 4. Elinaremos todas las columnas previamente mencionadas.
  dataset = CleanUpDataset(raw_path=DATA_PATH, processed_path=PROCESSED_DATA_PATH)
  dataset.load_dataset()
  dataset.drop_columns(leakage_cols + unnecessary_cols)

  # 5. De acuerdo a la imagen presentada por el maestro, donde se explican las columnas de nuestro dataset, hemos determinado que estos valores no pueden ser negativos. (Sabemos que al menos uno de ellos lo es, por esto la corrección)
  list_of_columns_that_should_have_positive_values = [
    'total_of_special_requests',
    'adr',
    'days_in_waiting_list',
    'booking_changes',
    'previous_bookings_not_canceled',
    'previous_cancellations',
    'stays_in_week_nights',
    'stays_in_weekend_nights',
    'lead_time'
  ]

  dataset.fix_negatives(list_of_columns_that_should_have_positive_values)

  ########################### Ajustes de Datos #################################

  # Creemos que es mejor utilizar  variables True o False / 1 o 0. Creemos que en este caso el valor mas importante es ver si los mismatchs entre lo que se reserva y se recibe afecta o no a la cancelación.
  dataset.df["room_mismatch"] = (dataset.df["reserved_room_type"] != dataset.df["assigned_room_type"]).astype(int)

  # Como observamos en el notebook hay una distrbución con sesgo a la derecha, lo cual significa que NO es una distribución normal, simetrica, para resolver un poco mejor este problema utilizaremos logaritmo.
  dataset.df["lead_time_log"] = np.log1p(dataset.df["lead_time"])

  # De acuerdo a los datos mostrados por el profesor en una imagen, la columna agent nos da un valor de ID del agente, consideramos que podria ser mucho más valioso simplemente saber si hay o no agente True o False Value.
  dataset.df["has_agent"] = dataset.df["agent"].notna().astype(int)

  # Eliminamos las otras dos columnas.
  dataset.drop_columns(['lead_time', 'reserved_room_type', 'assigned_room_type', 'agent'])

  # Hacemos escritura de la nueva .csv en data/processed/ directory.
  dataset.save_dataset()

  enhanced_df = dataset.get_df()

  # Identificar columnas numéricas y categóricas en el enhanced_df.
  feature_df = enhanced_df.drop(columns=["is_canceled"])
  numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
  categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()

  # Pipelines
  numeric_pipeline = Pipeline(steps=[
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler())
  ])

  categorical_pipeline = Pipeline(steps=[
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("encoder", OneHotEncoder(handle_unknown="ignore"))
  ])

  # ColumnTransformer
  preprocessor = ColumnTransformer(
      transformers=[
          ("num", numeric_pipeline, numeric_cols),
          ("cat", categorical_pipeline, categorical_cols)
      ]
  )

  return preprocessor

