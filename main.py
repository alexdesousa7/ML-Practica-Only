from src.data_loader import load_data
from src.model_trainer import train_models
from src.config import MODEL_PATH

def main():
    print("=== Cargando datos ===")
    df = load_data()

    print("=== Entrenando modelos ===")
    results, best_model = train_models(df)

    print("\n=== Resultados de los modelos ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print(f"\nModelo guardado en: {MODEL_PATH}best_model.pkl")
    print("\n=== Pipeline completado correctamente ===")


if __name__ == "__main__":
    main()
