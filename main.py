import os
import pandas as pd

from src.preprocessing import build_preprocessing_pipeline
from src.model_trainer import train_models
from src.config import MODEL_PATH, PROCESSED_DATA_PATH


def main():
    # Paso 1: Construir el preprocesador y generar processed_dataset.csv.
    # build_preprocessing_pipeline() limpia el raw CSV, aplica ingeniería de features
    # (room_mismatch, lead_time_log, has_agent) y devuelve el ColumnTransformer listo
    # para integrarse en cada Pipeline de modelo.
    print("Paso #1: Preprocesando datos")
    preprocessor = build_preprocessing_pipeline()

    # Paso 2: Leer el dataset ya limpio para pasarlo al entrenador.
    # Se usa el procesado (no el raw) porque el preprocesador ya eliminó columnas con
    # data leakage (reservation_status*) y columnas de baja utilidad (company, agent).
    print("Paso #2: Cargando datos procesados")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Paso 3: Entrenar los 5 modelos, guardar cada uno individualmente y
    # retornar el que tenga mayor ROC-AUC como best_model.pkl.
    print("Paso #3: Entrenando modelos")
    results, best_model = train_models(df, preprocessor)

    # Paso 4: Mostrar métricas de todos los modelos para comparación.
    print("\n Paso #4: Resultados de los modelos")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    best_model_path = os.path.join(MODEL_PATH, "best_model.pkl")
    print(f"\nMejor modelo guardado en: {best_model_path}")
    
    print("\n=== Pipeline completado correctamente ===")

if __name__ == "__main__":
    main()