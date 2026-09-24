import pandas as pd
from sklearn.ensemble import IsolationForest

data = pd.read_csv(r"preprocessed_placement.csv")

features = {
    col for col in data.select_dtypes(include=["number"]).columns
    if col.lower() != "PlacementStatus"
}

X= data[features]

print("\n features used for isolation forest:")

model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

model.fit(X)

data["Anomaly"] = model.predict(X)

data["Anomaly_Score"] = model.decision_function(X)

print("\n Anomaly Detection Results:")
print(data[features + ["Anomaly","Anomaly_Score"]].head(10))

anomalies = data[data["Anomaly"] == -1

print("\n Detected Anomalies:")
print(anomalies[features + ["Anomaly_Score"]])