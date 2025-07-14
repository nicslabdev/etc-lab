import argparse
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tabulate import tabulate  # pip install tabulate if not already installed
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder  # 💡 CAMBIO: Añadir LabelEncoder
try:
    from thundersvm import SVC as thunderSVC
    THUNDERSVM_OK = True
except ImportError:
    from sklearn.svm import SVC
    thunderSVC = SVC  # fallback
    THUNDERSVM_OK = False


# -------------------------------
# Define classifier factory
# -------------------------------
def select_classifier(model_name):
    model_name = model_name.lower()
    models = {
        'knn': lambda: KNeighborsClassifier(n_neighbors=10),
        'gaussiannb': lambda: GaussianNB(),
        #'svm': lambda: thunderSVC(kernel='rbf'),
        #'linearsvm': lambda: thunderSVC(kernel='linear'),
        'svm': lambda: SVC(kernel='rbf'),
        'linearsvm': lambda: LinearSVC(max_iter=10000),
        'randomforest': lambda: RandomForestClassifier(n_estimators=100),
        'softmax': lambda: SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, max_iter=1000, tol=1e-3),
        'xgboost': lambda: XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
        'lgbm': lambda: LGBMClassifier()
    }
    if model_name in models:
        return models[model_name]()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# -------------------------------
# Train and evaluate a model
# -------------------------------
def test_classifier(model_name, X_train, X_test, y_train, y_test, label_encoder):  # 💡 CAMBIO: añadir label_encoder
    clf = select_classifier(model_name)

    start_time = time.time()
    print(f"Training {model_name.upper()}...")
    clf.fit(X_train, y_train)
    print(f"Training done in {time.time() - start_time:.2f} seconds.")
    y_pred = clf.predict(X_test)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Training + prediction done in {elapsed:.2f} seconds.")

    report = classification_report(
        y_test, y_pred, output_dict=True,
        target_names=label_encoder.classes_,  # 💡 CAMBIO
        zero_division=0
    )
    
    accuracy = report["accuracy"]
    macro = report["macro avg"]
    weighted = report["weighted avg"]

    print(f"\nResults for {model_name.upper()}:")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {macro['precision']:.4f}")
    print(f"Macro Recall:    {macro['recall']:.4f}")
    print(f"Macro F1-score:  {macro['f1-score']:.4f}")
    print(f"Weighted Precision: {weighted['precision']:.4f}")
    print(f"Weighted Recall:    {weighted['recall']:.4f}")
    print(f"Weighted F1-score:  {weighted['f1-score']:.4f}")
    print(classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))  # 💡 CAMBIO

    return {
        "Model": model_name.lower(),
        "Time (s)": round(elapsed, 2),
        "Accuracy": round(accuracy, 4),
        "Macro Precision": round(macro["precision"], 4),
        "Macro Recall": round(macro["recall"], 4),
        "Macro F1": round(macro["f1-score"], 4),
        "Weighted Precision": round(weighted["precision"], 4),
        "Weighted Recall": round(weighted["recall"], 4),
        "Weighted F1": round(weighted["f1-score"], 4),
    }

# -------------------------------
# Main entry point
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate classifiers using features extracted from a .npz file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, required=True, help="Path to the .npz file with features and labels")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="List of models to train (e.g., knn svm). Use 'all' to run all supported models.")
    parser.add_argument("--train_fraction", type=float, default=1.0,
                    help="Fraction of the dataset to use for training (0.1 to 1.0)")

    args = parser.parse_args()

    if 'svm' in [m.lower() for m in args.models] and not THUNDERSVM_OK:
        print("⚠️ ThunderSVM no está instalado. Usando SVC de scikit-learn (CPU).")


    input_file = args.input
    selected_models = args.models

    print(f"Loading data from: {input_file}")
    data = np.load(input_file)
    X = data["X"].astype(np.float32)
    y = data["y"]
    if args.train_fraction < 1.0:
        print(f"📉 Reducing dataset to {int(args.train_fraction * 100)}% using stratified sampling...")
        sss = StratifiedShuffleSplit(n_splits=1, train_size=args.train_fraction, random_state=42)
        idx, _ = next(sss.split(X, y))
        X = X[idx]
        y = y[idx]

    print("Encoding string labels...")  # 💡 CAMBIO
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # 💡 CAMBIO

    print("Normalizing features...")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.33, stratify=y_encoded, random_state=42
    )

    available_models = ['knn', 'gaussiannb', 'svm', 'randomforest', 'softmax', 'xgboost', 'lgbm']
    models_to_run = available_models if 'all' in [m.lower() for m in selected_models] else selected_models

    summary = []
    for model in models_to_run:
        result = test_classifier(model, X_train, X_test, y_train, y_test, label_encoder)  # 💡 CAMBIO
        summary.append(result)

    # Show summary table
    print("\nSummary:")
    print(tabulate(summary, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    main()
