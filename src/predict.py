"""
Turns one new student's raw survey answers into:
  - a workload risk classification (Low / Moderate / High)
  - a confidence score
  - the student's own top influential features (for that specific profile,
    using the fitted model's global feature importances as a proxy — see
    note in `top_influential_features`)

This is what api.py calls, and what feeds the RAG retrieval query in
rag_pipeline.py (per Chapter 3.4 "Output Integration": the ML risk
classification + influential features become part of the retrieval query).
"""
import json
import joblib
import pandas as pd

from config import DATA_PROCESSED, MODELS_DIR
from preprocessing import ORDINAL_MAPS, NOMINAL_COLUMNS, OUTCOME_COLUMNS, NON_FEATURE_COLUMNS

_model = None
_scaler = None
_model_meta = None
_prep_meta = None


def _load():
    global _model, _scaler, _model_meta, _prep_meta
    if _model is None:
        _model = joblib.load(MODELS_DIR / "best_model.joblib")
        _scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        with open(MODELS_DIR / "model_meta.json") as f:
            _model_meta = json.load(f)
        # JSON object keys are always strings; convert the label_encoding
        # keys back to int so `label_encoding[int(pred)]` lookups work.
        if _model_meta.get("label_encoding"):
            _model_meta["label_encoding"] = {
                int(k): v for k, v in _model_meta["label_encoding"].items()
            }
        with open(DATA_PROCESSED / "preprocessing_meta.json") as f:
            _prep_meta = json.load(f)
    return _model, _scaler, _model_meta, _prep_meta


def encode_profile(raw_profile: dict) -> pd.DataFrame:
    """
    raw_profile: dict of {question_column: raw_answer_string_or_int}, using
    the SAME question text / answer text as the survey (see config.RAW_COLUMNS
    and preprocessing.ORDINAL_MAPS for the exact expected strings).
    Outcome-question keys (Q21/Q22/Q23/Q24) are optional and ignored here,
    since we're predicting risk from workload features, not outcomes.
    """
    _, _, _, prep_meta = _load()
    row = {}
    for col, mapping in ORDINAL_MAPS.items():
        if col in OUTCOME_COLUMNS + NON_FEATURE_COLUMNS:
            continue
        if col in raw_profile:
            val = raw_profile[col]
            row[col] = mapping[val] if isinstance(val, str) else val

    # Q12/Q20/Q23 Likert items pass through unchanged (already 1-5 ints)
    for col in ["Q12 Assignment Difficulty", "Q12 Exam Difficulty", "Q12 Viva Difficulty",
                "Q20 Energy", "Q20 Motivation", "Q20 Financial Pressure", "Q20 Family Pressure"]:
        if col in raw_profile:
            row[col] = raw_profile[col]

    df = pd.DataFrame([row])

    # one-hot encode nominal columns to match training columns
    for col in NOMINAL_COLUMNS:
        if col in raw_profile:
            df[col] = raw_profile[col]
    df = pd.get_dummies(df, columns=[c for c in NOMINAL_COLUMNS if c in df.columns],
                         prefix=NOMINAL_COLUMNS)

    # align to the exact training feature column order, filling missing
    # one-hot columns (e.g. a Degree option not present in this profile) with 0
    feature_cols = prep_meta["feature_columns"]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_cols]
    return df


def top_influential_features(model, feature_row: pd.DataFrame, n=5):
    """
    Approximates 'which features most influenced THIS prediction' by ranking
    the model's global feature_importances_ restricted to features where this
    student's (scaled) value is above the training-set median — i.e. features
    that are both generally important AND elevated for this student.
    For a more rigorous per-prediction explanation, swap this for SHAP
    (pip install shap) — left as a documented extension point.
    """
    if not hasattr(model, "feature_importances_"):
        return []
    importances = pd.Series(model.feature_importances_, index=feature_row.columns)
    elevated = feature_row.iloc[0] > feature_row.iloc[0].median()
    candidates = importances[elevated].sort_values(ascending=False)
    return list(candidates.head(n).index)


def predict_risk(raw_profile: dict) -> dict:
    model, scaler, model_meta, prep_meta = _load()
    feature_row = encode_profile(raw_profile)
    scaled = pd.DataFrame(scaler.transform(feature_row), columns=feature_row.columns)

    # XGBoost was trained on integer-encoded labels (see train_models.py) —
    # decode back to "Low"/"Moderate"/"High" using the saved mapping.
    label_encoding = model_meta.get("label_encoding")

    pred = model.predict(scaled)[0]
    if label_encoding is not None:
        pred = label_encoding[int(pred)]

    proba = None
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled)[0]
        if label_encoding is not None:
            classes = [label_encoding[i] for i in range(len(probs))]
        else:
            classes = getattr(model, "classes_", model_meta["risk_classes"])
        proba = {str(c): float(p) for c, p in zip(classes, probs)}
        confidence = max(proba.values())

    influential = top_influential_features(model, feature_row)

    return {
        "risk_class": str(pred),
        "confidence": confidence,
        "class_probabilities": proba,
        "top_influential_features": influential,
    }


if __name__ == "__main__":
    # quick manual smoke test
    example = {
        "Q1 Year": "Year 3",
        "Q2 Degree": "Computing / IT",
        "Q3 University": "NIBM (National Institute of Business Management)",
        "Q4 Residence": "University hostel / boarding",
        "Q5 Part-time": "Yes - 10 to 20 hours per week",
        "Q6 Assignments": "3 or more", "Q6 Exams": "2", "Q6 Vivas": "1", "Q6 Presentations": "1",
        "Q7 Subjects": "5 – 6 subjects", "Q8 Overlap": "4 or more overlapping",
        "Q9 Notice": "1 – 3 days (very short notice)",
        "Q10 Study Time": "Required much more than available",
        "Q11 Group Work": "Yes – 2 or more group tasks",
        "Q12 Assignment Difficulty": 5, "Q12 Exam Difficulty": 4, "Q12 Viva Difficulty": 4,
        "Q13 New Subjects": "3 or more new or unfamiliar subjects",
        "Q14 Untaught Content": "Most tasks required untaught content",
        "Q15 Self-study": "A lot – more than 5 extra hours",
        "Q16 Self-study Hours": "5 or more hours",
        "Q17 Lost Time": "More than 4 hours",
        "Q18 Study Space": "Rarely or never available",
        "Q19 Sleep": "Less than 4 hours",
        "Q20 Energy": 1, "Q20 Motivation": 2, "Q20 Financial Pressure": 4, "Q20 Family Pressure": 3,
    }
    print(json.dumps(predict_risk(example), indent=2))
