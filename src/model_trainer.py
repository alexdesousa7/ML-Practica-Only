import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

import joblib

from src.preprocessing import build_preprocessing_pipeline



def train_models(df):


    # Eliminamos columnas con leakage (información del futuro de lo contrario tendriamos acceso a la informacion del futuro y el modelo no aprenderia a generalizar.)
    
    leakage_cols = ["reservation_status", "reservation_status_date"]
    df = df.drop(columns=[col for col in leakage_cols if col in df.columns])

    # Separar variables

    X = df.drop("is_canceled", axis=1)
    y = df["is_canceled"]

    # Train-test split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocesamiento

    preprocessor = build_preprocessing_pipeline(X)

    # Modelos a entrenar: Regresión logística, Árbol de decisión, Random Forest, Gradient y Boosting, Red neuronal (MLP)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Neural Network": MLPClassifier(max_iter=300)
    }

    results = {}
    best_model = None
    best_score = -1

    for name, model in models.items():
        print(f"Entrenando modelo: {name}")

        # Pipeline completo
        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Entrenar
        clf.fit(X_train, y_train)

        # Predicciones
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Métricas
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        results[name] = metrics

        # Seleccionar mejor modelo por ROC-AUC, que es lo correcto en este caso para un problema de clasificación con clases desbalanceadas.
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = clf

    # Guardar mejor modelo 
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(project_root, "models", "best_model.pkl")

    joblib.dump(best_model, model_path)


    return results, best_model
