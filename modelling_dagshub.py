import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV


dagshub.init(repo_owner='MedG12', repo_name='Eksperimen-MLflow-Tracking-UI', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/MedG12/Eksperimen-MLflow-Tracking-UI.mlflow")
mlflow.set_experiment("Online_RF_Tuning_DagsHub")

df = pd.read_csv('./StudentPerformance_preprocessing.csv')
X = df.drop('Performance Index', axis=1)
y = df['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2')

print("Memulai Training Online di DagsHub...")
with mlflow.start_run(run_name="RF_Manual_Logging_Artifacts"):
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=y_test, lowess=True, line_kws={'color': 'red'})
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Analysis Plot")
    plt.tight_layout()
    plt.savefig("residuals_plot.png")
    mlflow.log_artifact("residuals_plot.png")
    plt.close()

    mlflow.sklearn.log_model(
        sk_model=best_model, 
        artifact_path="model",
        input_example=X_train[:5],
        registered_model_name="RF_Online_Tuned_Model"
    )

    print(f"Berhasil! Model dan artefak telah dikirim ke DagsHub.")
    print(f"R2 Score: {r2}")