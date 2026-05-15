import joblib
import pandas as pd

def load_model(model_path="../models/best_model.pkl"):
    """Carga el modelo entrenado desde disco."""
    return joblib.load(model_path)


def predict_reservation(model, input_data):
    """
    Realiza una predicción sobre una nueva reserva.
    
    input_data debe ser un diccionario con los mismos campos que el dataset original.
    """
    # Convertir a DataFrame
    df_input = pd.DataFrame([input_data])

    # Predicción
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
