# Model Training (production - runs inside GitHub Actions)
# 1. Loads the train/test splits downloaded from the previous job's artifact
# 2. Builds a preprocessing + XGBoost pipeline
# 3. Tunes hyperparameters with GridSearchCV, logging every run to MLflow
# 4. Evaluates the best model and saves it so the pipeline can commit it
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")

# These CSVs come from the Data Preparation job's uploaded artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# Handle class imbalance (~19% buyers) by weighting the positive class
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing: scale numeric, one-hot encode categorical
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Base XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight, random_state=42, eval_metric="logloss"
)

# Small grid so the pipeline runs fast on GitHub Actions (widen it for a deeper search)
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every hyperparameter combination as a nested MLflow run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log the best parameters in the parent run
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)
    print(classification_report(ytest, y_pred_test))

    # Log the best model's metrics
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
    })

    # Save next to app.py so Streamlit can load it, and log it as an MLflow artifact
    model_path = "mlops/deployment/best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print("Model saved to", model_path)
