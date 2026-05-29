# Distribución de Tareas

### Integrantes del Equipo
 - Enmanuel Alejandro De Oleo Ubiera
 - José Alexander De Sousa

### Tareas
 - José: Creación de repositorio y estructura de carpetas en Github.
    - gitignore
    - README.md
 - José: Análisis estadístico
    - Carga de datos
    - Limpieza de datos
    - Revisión de la descripción de la variables
    - Revisión de los tipos de variables
    - Graficas estadisticas
    - Modelado y evaluación de modelos

 - Enmanuel:
    - Mejoras en la documentación.
    - Creacion de Clase reutilizable para la limpieza de datos y la creación de una DB limpia.
    - Agregar algunas graficas y puntos de interés en el análysis estadístico.
    - Mejoras generales al Codigo
    - Agregar API con fastapi.
    
## 2. Objetivo Principal de análisis y problemas a resolver:
  - 1. Poder predecir si una reserva será cancelada o no. (Utilizando el `is_cancel` con valores True/False o 1 o 0) como nuestra variable dependiente.
  - 2. Utilizando los siguientes modelos elegir el que obtenga mejores resultados de acuerdo a las métricas que se comentarán mas adelante:
    - Modelos:
      - Logistic Regression
      - Desicion Tree
      - Random Forest
      - XGBoost
      - Deep Neural Network

# Análysis exploratorio de datos
### Hallazgos interesantes
**1. Canceladas vs no canceladas**
El 37% de las reservas fueron canceladas y el 63% no. No es un desbalance extremo, 
pero es suficiente para que métricas como accuracy sean engañosas. 
Usaremos ROC-AUC, F1-score y recall como métricas principales.

**2. Lead time parece ser predictor significativo**
Las reservas canceladas se hicieron en promedio con más días de anticipación que las 
no canceladas. Cuanto más tiempo pasa entre la reserva y la llegada, más probable es 
que los planes cambien.

**3. Deposit type es raro (contraintuitivo)**
Las reservas con depósito "Non Refund" tienen una tasa de cancelación cercana al 99%. 
Aunque el cliente ya pagó y no puede recuperar el dinero, cancela igualmente.

**4. Room mismatch reduce las cancelaciones**
Cuando el hotel asigna una habitación diferente a la reservada, la tasa de cancelación 
baja drásticamente. La hipótesis más probable es que estos cambios sean upgrades, 
lo que genera satisfacción en el cliente.

**5. Portugal concentra el mayor volumen y tasa de cancelaciones**
PRT tiene una tasa del 56.6% de cancelación, muy por encima del resto de países. 

## Estructura del Proyecto

```
ML Practica Only/
│
├── main.py                     # Pipeline completo (entrenamiento + guardado del modelo)
├── requirements.txt
│
├── src/
│   ├── cleanup_dataset.py      # Clase que contiene metodos reutilizables para la limpieza de los datos y creación de .csv con los datos processados.
│   ├── config.py               # Configuración global (rutas, parámetros, constantes)
│   ├── data_loader.py          # Carga del dataset y utilidades de lectura
│   ├── preprocessing.py        # Pipeline de preprocesamiento
│   ├── model_trainer.py        # Entrenamiento, evaluación y guardado de modelos
│   └── predictor.py            # Funciones de predicción con el modelo final
│
├── data/
│   ├── raw/
│   │   └── dataset_practica_final.csv
│   └── processed/
│       └── processed_dataset.csv  # Este se genera al momento de correr los main.py.
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

# Conclusiones De evaluación de modelos:

| Modelo               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Random Forest        | 0.8900   | 0.8883    | 0.8043 | 0.8442   | 0.9565  |
| Neural Network       | 0.8622   | 0.7949    | 0.8464 | 0.8198   | 0.9419  |
| Gradient Boosting    | 0.8482   | 0.8320    | 0.7395 | 0.7830   | 0.9240  |
| Logistic Regression  | 0.8284   | 0.8149    | 0.6945 | 0.7499   | 0.9065  |
| Decision Tree        | 0.8572   | 0.8056    | 0.8099 | 0.8078   | 0.8495  |


## Comparación de modelos (Curva ROC)
Los 5 modelos fueron entrenados y evaluados sobre el mismo conjunto de test (20% del dataset).
El modelo con mejor resultado (de ROC-AUC) es **Random Forest** 0.95. Seguido por la  **Red Neuronal** 0.94

## Variables más influyentes (best_model.pkl)
De acuerdo al análisis de importancia del mejor modelo guardado:

| Variable | Importancia | Interpretación |
|---|---|---|
| `lead_time_log` | 11.0% | Más anticipación → más probabilidad de cancelar |
| `adr` | 7.5% | El precio influye en la decisión de cancelar |
| `deposit_type_Non Refund` | 6.8% | Contraintuitivo — pagan sin reembolso pero igual cancelan |
| `deposit_type_No Deposit` | 6.7% | Sin penalización económica, cancelar no cuesta nada |
| `country_PRT` | 6.1% | Portugal es el mercado con mayor tasa de cancelación (56.6%) |
| `total_of_special_requests` | 5.6% | Más peticiones = más comprometido = menos cancelaciones |
| `room_mismatch` | 3.0% | Habitación diferente = posiblemente upgrade = menos cancelaciones |


## Validación con el EDA
Los hallazgos del modelo confirman lo observado en el EDA inicial:
- `lead_time` fue el primer predictor identificado visualmente
- `deposit_type` mostró el comportamiento raro de Non Refund
- `country_PRT` coincide con la tasa del 56.6% observada en el EDA


# Reflexiones, Limitaciones y Críticas.
- Enmanuel Alejandro De Oleo:
  - Reflexiones:
     1. Me pareció mucho más interesante la parte de la exploración de datos. En la cual creo que tuvimos hallazgos inusuales pero que hicieron sentido al profundizar en el analisis.
     2. Me hubiese gustado poder utilizar la optimización de hiperparámetros con GridSearchCV o RandomizedSearchCV, pero por falta de organización personal y tiempo no pude hacerlo. Y cuando lo intenté mi computadora por algun motivo se quedaba congelada. 

- José Alexander De Sousa:
  - Reflexiones:
    1. El proceso de carga y limpieza me obligo a tomar decisiones que pueden condicionar la forma de tratar los datos, ya que no estoy claro si fueron las mas adecuadas o si existían alternativas mejores.
    2. Las graficas estadísticas me ayudaron a entender la distribución de las variables y posibles correlaciones entre los datos.
    3. EL modelado no es elegir el mejor resultado, sino que hay que elegir entre el rendimiento, interpretabilidad de los datos y coste computacional, siendo este ultimo el que mas peso pueda tener a la hora de elegir un modelo.