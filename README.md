# 🏨 Predicción de Cancelación de Reservas de Hotel

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas" />
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Project-ML-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" />
</p>


Modulo 05 – Machine Learning – Ejercicio Final - Pontia Tech

Este proyecto implementa un sistema completo de Machine Learning para predecir si una reserva hotelera será cancelada o no por el cliente.

Incluye:

- Exploración de datos (EDA)  
- Preprocesamiento modular  
- Entrenamiento de múltiples modelos  
- Guardado de todos los modelos entrenados  
- Selección automática del mejor modelo  
- Evaluación con métricas avanzadas  
- Pipeline reproducible desde `main.py`  
- Notebook de predicción para casos reales  

---

## 📂 Estructura del Proyecto

```
ML Practica Only/
│
├── main.py                     # Pipeline completo (entrenamiento + guardado del modelo)
├── requirements.txt
│
├── src/
│   ├── config.py               # Configuración global (rutas, parámetros, constantes)
│   ├── data_loader.py          # Carga del dataset y utilidades de lectura
│   ├── preprocessing.py        # Pipeline de preprocesamiento
│   ├── model_trainer.py        # Entrenamiento, evaluación y guardado de modelos
│   └── predictor.py            # Funciones de predicción con el modelo final
│
├── data/
│   └── raw/
│       └── dataset_practica_final.csv
│
├── models/                     # Modelos entrenados (IGNORADO en GitHub)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── neural_network.pkl
│   └── best_model.pkl
│
├── outputs/                    # Gráficos generados durante el análisis y evaluación
│   ├── cancelaciones_por_segmento.png
│   ├── cancelaciones_por_tipo_cliente.png
│   ├── cancelaciones_por_tipo_hotel.png
│   ├── confusion_matrix.png
│   ├── distribucion_objetivo.png
│   ├── distribuciones_numericas.png
│   ├── feature_importances.png
│   ├── matriz_confusion.png
│   ├── matriz_correlacion.png
│   ├── roc_comparativa.png
│   └── roc_curve.png
│
└── notebooks/                  # Notebooks organizados por fase del proyecto
    ├── exploracion/
    │   └── eda_inicial.ipynb          # Análisis exploratorio del dataset
    │
    ├── modelado/
    │   └── entrenamiento.ipynb        # Pruebas de modelos y tuning inicial
    │
    ├── evaluacion/
    │   └── evaluacion.ipynb           # Métricas, curvas ROC, matriz de confusión
    │
    └── predicion/
        └── prediccion.ipynb           # Predicción con el modelo final

```

---

## 🧠 Descripción del Pipeline

El pipeline completo se ejecuta desde `main.py` y realiza:

### **1. Carga del dataset**
Desde `data/raw/dataset_practica_final.csv`.

### **2. Preprocesamiento**
Implementado en `src/preprocessing.py`:

- imputación  
- escalado  
- codificación OneHot  
- ensamblado en un `ColumnTransformer`  
- integración en un `Pipeline`  

### **3. Entrenamiento de modelos**
En `src/model_trainer.py` se entrenan:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- Neural Network (MLPClassifier)  

### **4. Guardado de modelos**
El sistema guarda:

- **todos los modelos individuales**  
- **el mejor modelo como `best_model.pkl`**

### **5. Evaluación**
Se calculan:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

### **6. Selección del mejor modelo**
El modelo con mejor **ROC-AUC** se guarda como:

```
models/best_model.pkl
```

---

## 📊 Resultados de los Modelos (Ejecución Final)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.8184 | 0.8110 | 0.6646 | 0.7305 | 0.8962 |
| Decision Tree | 0.8577 | 0.8052 | 0.8122 | 0.8087 | 0.8507 |
| Neural Network | 0.8680 | 0.8371 | 0.7992 | 0.8177 | 0.9419 |
| Gradient Boosting | 0.8509 | 0.8510 | 0.7244 | 0.7826 | 0.9268 |
| **Random Forest** | **0.8940** | **0.8920** | **0.8123** | **0.8503** | **0.9581** |

### ⭐ Mejor modelo: **Random Forest Classifier**  
Guardado como:

```
models/best_model.pkl
```

---

## 🔍 Importancia de Variables (Random Forest)

| Feature | Importancia |
|--------|-------------|
| lead_time | 0.106 |
| adr | 0.066 |
| deposit_type_No Deposit | 0.059 |
| country_PRT | 0.059 |
| deposit_type_Non Refund | 0.057 |
| total_of_special_requests | 0.054 |

---

## ▶️ Instalación

### **1. Crear entorno virtual**

**Windows**
```
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac**
```
python3 -m venv venv
source venv/bin/activate
```

---

### **2. Instalar dependencias**

```
pip install -r requirements.txt
```

---

## ▶️ Ejecución del Pipeline Completo

Desde la raíz del proyecto:

```
python main.py
```

Esto:

- entrena todos los modelos  
- guarda cada modelo individual  
- selecciona el mejor  
- guarda `best_model.pkl`  
- imprime métricas  

Nota:
- los gráficos se generan desde los notebooks en la carpeta notebooks y los guarda en en `outputs/`  

---

## 🤖 Ejemplo de Predicción

En el notebook `notebooks/prediccion/prediccion.ipynb`:

```python
import pandas as pd
import joblib

# Cargar modelo final
model = joblib.load("../../models/best_model.pkl")

# Cargar dataset
df = pd.read_csv("../../data/raw/dataset_practica_final.csv")

# Tomar una fila aleatoria
sample = df.sample(1, random_state=42)

# Separar X
X_sample = sample.drop("is_canceled", axis=1)

# Predicción
pred = model.predict(X_sample)[0]
prob = model.predict_proba(X_sample)[0][1]

print("Predicción:", pred)
print("Probabilidad de cancelación:", prob)
```

---

## 🛠️ Reproducibilidad

Para regenerar todos los modelos:

```
python main.py
```

Esto creará nuevamente:

```
models/
    logistic_regression.pkl
    decision_tree.pkl
    random_forest.pkl
    gradient_boosting.pkl
    neural_network.pkl
    best_model.pkl
```

---

## 🏁 Conclusión

Este proyecto implementa un sistema completo y modular de Machine Learning, con:

- arquitectura profesional  
- pipeline reproducible  
- guardado de todos los modelos  
- selección automática del mejor  
- notebooks limpios  
- documentación clara  

---
