import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import os

# Load Data
df = pl.read_csv("data/creditcard.csv", schema_overrides={"Time": pl.Float64})
X = df.drop("Class").to_numpy()
y = df.select("Class").to_numpy().flatten()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Optional: set custom experiment
mlflow.set_experiment("CreditCardFraudDetection")

# Start run
with mlflow.start_run(run_name="RandomForestPolars"):

    # Define and train model
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", acc)

    # === MLflow Logging ===
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("class_weight", "balanced")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save model to file
    os.makedirs("model", exist_ok=True)
    model_path = "model/model.pkl"
    joblib.dump(clf, model_path)

    # Log model manually (can also use mlflow.sklearn.log_model)
    mlflow.log_artifact(model_path, artifact_path="model")

    # Optional: log confusion matrix or classification report
    report_path = "model/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact(report_path, artifact_path="metrics")

    # Add tags for easy filtering
    mlflow.set_tags({
        "model_type": "RandomForest",
        "framework": "scikit-learn",
        "data_lib": "polars",
        "purpose": "fraud_detection"
    })