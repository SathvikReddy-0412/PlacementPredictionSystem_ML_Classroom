import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================================
# DATASET PATH
# =========================================================

DATA_PATH = r"placement_predict_50k Dataset (3) 1(in).csv"


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

def run_logistic_regression():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Features
    features = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    # Target
    target = "PlacementStatus"

    # Remove rows with missing values
    data = df[features + [target]].dropna()

    X = data[features]
    y = data[target]

    # Encode target if it is text
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create model
    model = LogisticRegression(max_iter=1000)

    # Train
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        output_dict=True
    )

    return {
        "features": features,
        "target": target,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": round(accuracy * 100, 2),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classes": encoder.classes_.tolist()
    }