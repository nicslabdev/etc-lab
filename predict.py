import argparse
import torch
import numpy as np
import json
from joblib import load
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from models import MLP, CNN1D, SAEClassifier  # Asegúrate de que estos estén disponibles
from operator import itemgetter

def load_model(model_name, input_dim, num_classes, path):
    model_name = model_name.lower()
    if model_name == "mlp":
        model = MLP(input_dim, num_classes)
    elif model_name == "cnn1d":
        model = CNN1D(input_dim, num_classes)
    elif model_name == "sae":
        model = SAEClassifier([input_dim, 400, 300, 200, 100, 50], num_classes)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
    
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(
        description="Clasifica paquetes desde un archivo .npz de características utilizando un modelo exportado.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--features", type=str, required=True, help="Archivo .npz con clave 'X'")
    parser.add_argument("--config", type=str, required=True, help="Archivo .json de configuración del modelo")
    parser.add_argument("--weights_dir", type=str, default="models", help="Carpeta donde están .pt y .joblib")

    args = parser.parse_args()

    # 1. Cargar configuración
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
    except Exception as e:
        raise ValueError(f"Error al leer el archivo JSON de configuración: {e}")

    required_keys = ["model_name", "input_dim", "num_classes", "class_labels", "model_file", "label_encoder_file"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Falta la clave '{key}' en el archivo de configuración.")

    model_name = config["model_name"]
    input_dim = config["input_dim"]
    num_classes = config["num_classes"]
    class_labels = config["class_labels"]
    model_file = config["model_file"]
    le_file = config["label_encoder_file"]

    # 2. Cargar modelo y encoder
    model_path = f"{args.weights_dir}/{model_file}"
    le_path = f"{args.weights_dir}/{le_file}"

    model = load_model(model_name, input_dim, num_classes, model_path)
    le = load(le_path)

    # 3. Cargar features
    try:
        data = np.load(args.features)
    except Exception as e:
        raise ValueError(f"Error al cargar el archivo .npz: {e}")

    if "X" not in data:
        raise ValueError(f"El archivo {args.features} no contiene la clave 'X'")

    X = data["X"]

    if X.shape[1] != input_dim:
        raise ValueError(f"Dimensión de entrada no coincide con el modelo: {X.shape[1]} ≠ {input_dim}")

    # 4. Normalizar como en entrenamiento
    X = MinMaxScaler().fit_transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # 5. Predecir
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = F.softmax(outputs, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        labels = le.inverse_transform(preds)

    # 6. Mostrar resultados
    print()
    if len(X) == 1:
        print("🔍 Predicción para paquete único:")
        print(f"Clase probable: {labels[0]}")
        print("Distribución:")
        for i, p in enumerate(probs[0]):
            print(f"- {class_labels[i]}: {p:.2f}")
    else:
        print("🔍 Predicciones por paquete:")
        for i, label in enumerate(labels):
            print(f"Paquete {i+1} → {label}")

        counts = Counter(labels)
        total = sum(counts.values())
        sorted_counts = sorted(counts.items(), key=itemgetter(1), reverse=True)

        print("\n📊 Distribución de clases predichas (ordenada):")
        for cls, count in sorted_counts:
            percentage = (count / total) * 100
            print(f"- {cls}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    main()
