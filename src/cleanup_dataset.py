import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_PATH, PROCESSED_DATA_PATH # Me voy a importar el path de los datos raw y el path en donde quiero guardar los datos ya procesados.

class CleanUpDataset:
  """
  Esta clase se provee métodos para limpiar el dataset, eliminar columnas no deseadas, 
  corregir valores negativos y guardar el dataset procesado.
  """
  def __init__(self, raw_path, processed_path):
    self.raw_path = raw_path
    self.processed_path = processed_path
    self.df = None

  def load_dataset(self):
    self.df = pd.read_csv(self.raw_path)
    return self

  def drop_columns(self, list_of_columns: list):
    cols_to_drop = [col for col in list_of_columns if col in self.df.columns]
    self.df = self.df.drop(columns=cols_to_drop)
    return self

  def fix_negatives(self, list_of_columns: list):
    """
    Corrige los valores negativos en las columnas especificadas. (Hay unas columnas en nuestro dataset que nunca deberian tener valores negativos, cuando ese sea el caso lo pondremos a cero)
    """
    for col in list_of_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].clip(lower=0)

    return self

  def save_dataset(self):
    """
    Guarda el dataset en la ruta especificada. En este caso, estamos creando la carpeta `processed` para guardar el dataset procesado.
    """
    Path(self.processed_path).parent.mkdir(parents=True, exist_ok=True)
    self.df.to_csv(self.processed_path, index=False)
    print(f"Dataset guardado en: {self.processed_path}")
    return self

  def get_df(self) -> pd.DataFrame:
    return self.df