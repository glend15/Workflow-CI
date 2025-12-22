import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    mlflow.set_experiment("telco-churn")
    mlflow.sklearn.autolog()

    df = pd.read_csv("telco_preprocessing/telco_preprocessing.csv")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        params,
        cv=3
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    print("Training CI selesai")

if __name__ == "__main__":
    main()