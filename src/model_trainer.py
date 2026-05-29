import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib

def train_models(df, preprocessor):
    """
      De acuerdo a los requerimientos de la tarea, se entrena con los siguientes modelos:
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - Gradient Boosting
        - Neural Network

      Nos quedamos con el modelo que obtiene mejores resultados.
    """
    # Separamos variables
    X = df.drop("is_canceled", axis=1)
    y = df["is_canceled"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modelos a entrenar: Regresión logística, Árbol de decisión, Random Forest, Gradient y Boosting, Red neuronal (MLP)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Neural Network": MLPClassifier(max_iter=300)
    }

    # Crear carpeta models si no existe
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

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

        # Guardar modelos de forma individual
        model_filename = name.lower().replace(" ", "_") + ".pkl"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(clf, model_path)
        print(f"Modelo guardado: {model_path}")

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

    # Guarda el mejor modelo
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"\nMejor modelo guardado en: {best_model_path}")

    return results, best_model
