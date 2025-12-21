import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature


def main():
    df = pd.read_csv("telco_preprocessing/telco_preprocessing.csv")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 3000)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    signature = infer_signature(X_train, pipeline.predict(X_train))

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        input_example=X_train.iloc[:5],
        signature=signature
    )

    print("Training selesai")


if __name__ == "__main__":
    main()