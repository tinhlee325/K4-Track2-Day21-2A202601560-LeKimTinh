import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

F1_THRESHOLD = 0.65
BASELINE_POS_RATIO = 0.248  # Ti le tham chieu lop duong 24.8%


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh, kiem tra lech lac du lieu (Bonus 5),
    quet nguong quyet dinh toi uu (Bonus 2), tao bao cao chi tiet (Bonus 3),
    va ghi nhan ket qua vao MLflow.
    """
    # 1. Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # BONUS 5: Kiem tra lech lac phan phoi du lieu (Data Drift)
    pos_ratio = float(y_train.mean())
    drift_detected = abs(pos_ratio - BASELINE_POS_RATIO) > 0.05
    if drift_detected:
        print(f"[CANH BAO DRIFT] Ti le lop duong: {pos_ratio:.1%} lech >5% so voi tham chieu {BASELINE_POS_RATIO:.1%}!")
    else:
        print(f"[DATA CHECK OK] Ti le lop duong: {pos_ratio:.1%} (tham chieu: {BASELINE_POS_RATIO:.1%})")

    # Dam bao MLflow tracking va DagsHub credentials duoc thiet lap
    if os.environ.get("DAGSHUB_USERNAME") and not os.environ.get("MLFLOW_TRACKING_USERNAME"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
    if os.environ.get("DAGSHUB_TOKEN") and not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]

    # Thu ket noi MLflow (ho tro fallback an toan ve SQLite neu remote loi xac thuc)
    try:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        active_run = mlflow.start_run()
    except Exception as e:
        print(f"[MLFLOW NOTE] Ket noi remote MLflow that bai ({e}). Chuyen ve SQLite cuc bo.")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        active_run = mlflow.start_run()

    with active_run:
        # Log params va Data Drift metric
        mlflow.log_params(params)
        mlflow.log_metric("pos_ratio", pos_ratio)

        # 3. Huan luyen mo hinh
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 4. Du doan xac suat tren tap holdout
        probs = model.predict_proba(X_eval)[:, 1]

        # BONUS 2: Quet nguong tu 0.1 den 0.9 (buoc 0.05) tim nguong toi uu F1
        thresholds = np.arange(0.1, 0.91, 0.05)
        best_threshold = 0.5
        best_f1 = 0.0
        for thresh in thresholds:
            t_preds = (probs >= thresh).astype(int)
            t_f1 = float(f1_score(y_eval, t_preds))
            if t_f1 > best_f1:
                best_f1 = t_f1
                best_threshold = float(thresh)

        # Danh gia tai nguong mac dinh 0.5 va nguong toi uu
        default_preds = (probs >= 0.5).astype(int)
        default_f1 = float(f1_score(y_eval, default_preds))
        default_acc = float(accuracy_score(y_eval, default_preds))

        opt_preds = (probs >= best_threshold).astype(int)
        opt_acc = float(accuracy_score(y_eval, opt_preds))

        print(f"F1 (nguong 0.5): {default_f1:.4f} | Accuracy: {default_acc:.4f}")
        print(f"[BONUS 2] Nguong toi uu: {best_threshold:.2f} -> Best F1: {best_f1:.4f} | Accuracy: {opt_acc:.4f}")

        # Ghi metric vao MLflow
        mlflow.log_metric("f1_score", default_f1)
        mlflow.log_metric("accuracy", default_acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(model, "model")

        # BONUS 3: Tao bao cao chi tiet Precision / Recall va Confusion Matrix
        cm = confusion_matrix(y_eval, default_preds)
        cls_report = classification_report(y_eval, default_preds, target_names=["<=50K (0)", ">50K (1)"])

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write("=== BAO CAO CHI TIET (BONUS 3) ===\n\n")
            f.write("--- CONFUSION MATRIX ---\n")
            f.write(f"TN: {cm[0,0]} | FP: {cm[0,1]}\n")
            f.write(f"FN: {cm[1,0]} | TP: {cm[1,1]}\n\n")
            f.write("--- CLASSIFICATION REPORT ---\n")
            f.write(cls_report)
            f.write("\n--- THRESHOLD TUNING (BONUS 2) ---\n")
            f.write(f"Default (0.5): F1={default_f1:.4f}, Acc={default_acc:.4f}\n")
            f.write(f"Optimal ({best_threshold:.2f}): F1={best_f1:.4f}, Acc={opt_acc:.4f}\n")

        # Luu report.json cho CI/CD
        with open("outputs/report.json", "w") as f:
            json.dump({
                "f1_score": default_f1,
                "accuracy": default_acc,
                "best_threshold": best_threshold,
                "best_f1_score": best_f1,
                "pos_ratio": pos_ratio,
                "drift_detected": drift_detected,
            }, f, indent=2)

        # Luu model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return default_f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
