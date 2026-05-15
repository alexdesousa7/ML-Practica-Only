from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_preprocessing_pipeline(df):

    # Excluir la columna objetivo si está presente
    target_col = "is_canceled"
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Eliminar columnas con leakage (información del futuro)
    leakage_cols = ["reservation_status", "reservation_status_date"]
    df = df.drop(columns=[col for col in leakage_cols if col in df.columns])

    # Identificar columnas numéricas y categóricas en el dataset.
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Eliminar columnas irrelevantes o con muchos valores nulos que no aportan informacion alguna.

    drop_cols = ["company"]
    numeric_cols = [col for col in numeric_cols if col not in drop_cols]
    categorical_cols = [col for col in categorical_cols if col not in drop_cols]

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

