
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, TARGET_COLUMN, RANDOM_STATE


def load_data():
    df = pd.read_csv(DATA_PROCESSED / "processed.csv")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def evaluate_model(name, model, X_train, X_test, y_train, y_test, cv_folds=10, display_labels=None):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    labels = display_labels if display_labels is not None else getattr(model, "classes_", None)
    cm = confusion_matrix(y_test, preds, labels=range(len(labels)) if display_labels is not None else labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"confusion_matrix_{name}.png")
    plt.close()

    report_y_test, report_preds = y_test, preds
    if display_labels is not None:
        labels_arr = np.array(display_labels)
        report_y_test = labels_arr[np.asarray(y_test)]
        report_preds = labels_arr[np.asarray(preds)]

    print(f"\n=== {name} ===")
    print(classification_report(report_y_test, report_preds, zero_division=0))

    return {
        "model": name,
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": prec,
        "recall_macro": rec,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
    }


def minimum_feature_set(model, X_train, y_train, X_test, y_test, target_acc=0.85):
    """
    RQ2 asks for 'the minimum feature set required to maintain prediction
    accuracy above 85%'. This does a simple backward elimination driven by
    Random-Forest feature importance: repeatedly drop the least important
    remaining feature and re-check accuracy, stopping just before accuracy
    would fall below target_acc.
    """
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    ranked = importances.sort_values(ascending=False)
    kept = list(ranked.index)

    result = {"target_accuracy": target_acc, "steps": []}
    best_kept = kept.copy()

    for n in range(len(kept), 0, -1):
        subset = kept[:n]
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        clf.fit(X_train[subset], y_train)
        acc = accuracy_score(y_test, clf.predict(X_test[subset]))
        result["steps"].append({"n_features": n, "accuracy": acc})
        if acc >= target_acc:
            best_kept = subset
        else:
            break

    result["minimum_feature_set"] = best_kept
    result["achieved_accuracy"] = next(
        (s["accuracy"] for s in result["steps"] if s["n_features"] == len(best_kept)), None
    )
    return result


def run():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # SVM benefits from scaled features; tree models don't need it but scaling
    # doesn't hurt them, so we scale once and reuse for all three.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, use_label_encoder=False, eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        )
    else:
        print("xgboost not installed — skipping XGBoost (pip install xgboost).")

    results = []
    fitted = {}
    for name, model in models.items():
        if name == "XGBoost":
            # XGBoost needs integer-encoded labels; pass the real class names
            # in for the confusion-matrix display only (can't overwrite
            # XGBClassifier.classes_, it's a read-only property)
            r = evaluate_model(
                name, model, X_train_scaled, X_test_scaled, y_train_enc, y_test_enc,
                display_labels=list(le.classes_),
            )
        else:
            r = evaluate_model(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
        results.append(r)
        fitted[name] = model

    comparison = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\n=== Model comparison (sorted by macro F1) ===")
    print(comparison.to_string(index=False))

    best_name = comparison.iloc[0]["model"]
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # Feature importance (Random Forest — used for RQ2 discussion regardless
    # of which model 'wins' on accuracy, since RF/XGBoost both expose it and
    # it's the most interpretable of the three)
    rf_for_importance = fitted.get("RandomForest") or RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE
    ).fit(X_train_scaled, y_train)
    importances = pd.Series(rf_for_importance.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False).head(15)
    importances.iloc[::-1].plot(kind="barh", figsize=(8, 6))
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png")
    plt.close()

    # Minimum feature set analysis (RQ2)
    mfs = minimum_feature_set(rf_for_importance, X_train_scaled, y_train, X_test_scaled, y_test)
    with open(REPORTS_DIR / "minimum_feature_set.json", "w") as f:
        json.dump(mfs, f, indent=2)
    print(f"\nMinimum feature set ({len(mfs['minimum_feature_set'])} features) "
          f"achieves {mfs['achieved_accuracy']:.3f} accuracy (target {mfs['target_accuracy']}).")

    # Save best model + scaler + feature column order for inference.
    # XGBoost predicts integer-encoded labels (0/1/2), not the original
    # "Low"/"Moderate"/"High" strings, so we save the encoding mapping too;
    # predict.py uses it to decode predictions back to class names.
    label_encoding = None
    if best_name == "XGBoost":
        label_encoding = {int(i): cls for i, cls in enumerate(le.classes_)}

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump({
            "best_model_name": best_name,
            "feature_columns": list(X.columns),
            "risk_classes": sorted(y.unique().tolist()),
            "label_encoding": label_encoding,
        }, f, indent=2)

    print(f"\nSaved best model ({best_name}) -> {MODELS_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    run()
